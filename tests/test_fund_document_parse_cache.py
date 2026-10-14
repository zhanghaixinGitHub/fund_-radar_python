"""解析缓存必须真的减少调用，同时保留首次失败、原文版本和历史时间边界。"""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from app.services.fund_document_parse_cache import DocumentParseCache, digest
from scripts.local_fund_incremental_pipeline import execute, import_archive

CUTOFF = "2026-10-13T11:35:00+08:00"
BODY = "公司尚未购买理财产品，相关审批仍在进行。完成审批后方可实施，后续存在不确定性。"


def document(name="公告", kind="ANNOUNCEMENT"):
    return {
        "id": name,
        "kind": kind,
        "title": name,
        "source_hash": digest(name),
        "body": BODY,
        "body_scope": "FULL_TEXT",
        "body_truncated": False,
        "date": "2026-10-08",
        "available_at": "2026-10-08T10:00:00+08:00",
        "codes": ["000988.SZ"],
        "url": "https://example.test",
    }


def result(doc):
    return {
        "id": doc["id"],
        "events": [
            {
                "title": "审批及实施条件",
                "assessment": "UNKNOWN",
                "stage": "进行中",
                "quotes": [doc["body"]],
                "meaning": "公司尚未购买理财产品，仍在审批。",
                "mechanism": "后续影响需结合审批和实施情况观察。",
                "caveat": "完成审批后方可实施，存在不确定性。",
            }
        ],
    }


def snapshot(documents):
    return {
        "fund_code": "002112",
        "as_of": CUTOFF,
        "documents": documents,
        "window": {"target_nav_date": "2026-10-14", "base_nav_date": "2026-10-13"},
        "time_context": {"target_date": "2026-10-14", "baseline_date": "2026-10-13"},
        "companies": [{"code": "000988.SZ", "name": "华工科技", "weight_pct": 10.5}],
        "facts": {"nav": {"category": "净值", "text": "当前程序净值事实"}},
    }


def successful_parser(calls):
    def parse(doc):
        calls.append(doc["id"])
        return {"result": result(doc), "raw": {"first_response": doc["id"]}, "model_called": True}

    return parse


@pytest.mark.parametrize("kind", ["ANNOUNCEMENT", "NEWS", "POLICY"])
def test_persistent_reuse_across_next_prediction_day_without_model_calls(tmp_path, kind):
    path = tmp_path / "parses.sqlite3"
    doc, calls = document(kind, kind), []
    first = execute(snapshot([doc]), DocumentParseCache(path), tmp_path / "first", successful_parser(calls))
    later = snapshot([doc])
    later["as_of"] = "2026-10-14T11:35:00+08:00"
    later["window"]["target_nav_date"] = "2026-10-15"
    later["time_context"]["target_date"] = "2026-10-15"
    later["facts"]["nav"]["text"] = "第二天更新的净值事实"
    second = execute(later, DocumentParseCache(path), tmp_path / "second", successful_parser(calls))
    assert first["model_calls"] == 1
    assert second["model_calls"] == 0 and second["cache_hits_before"] == 1
    assert calls == [kind]
    value = json.loads((tmp_path / "second/synthesis-input.json").read_text(encoding="utf-8"))
    assert value["window"]["target_nav_date"] == "2026-10-15"
    assert any(f["text"] == "第二天更新的净值事实" for f in value["facts"].values())
    assert value["old_daily_prediction_used"] is False


def test_new_document_alone_calls_model_and_existing_document_is_reused(tmp_path):
    cache, calls = DocumentParseCache(tmp_path / "db.sqlite3"), []
    old, new = document("旧公告"), document("新公告")
    execute(snapshot([old]), cache, tmp_path / "first", successful_parser(calls))
    next_run = execute(snapshot([old, new]), cache, tmp_path / "second", successful_parser(calls))
    assert next_run["model_calls"] == 1 and next_run["cache_hits_before"] == 1
    assert calls == [old["id"], new["id"]]


def test_content_restoration_and_contract_change_do_not_hit_old_parse(tmp_path):
    path = tmp_path / "db.sqlite3"
    cache, doc = DocumentParseCache(path), document()
    assert cache.claim(doc, CUTOFF, {})
    cache.finish(doc, result(doc), {})
    changed = {**doc, "body": doc["body"] + "新恢复的完整条款。"}
    assert cache.lookup(changed, CUTOFF)["state"] == "PENDING"
    assert cache.lookup(changed, CUTOFF)["reason"] == "SOURCE_OR_CONTRACT_CHANGED"
    assert DocumentParseCache(path, "NEW_CONTRACT").lookup(doc, CUTOFF)["state"] == "PENDING"
    assert cache.lookup(doc, CUTOFF)["state"] == "REUSE"


def test_known_semantic_error_and_wrong_quote_are_stored_but_not_reused(tmp_path):
    cache, doc = DocumentParseCache(tmp_path / "db.sqlite3"), document()
    cache.claim(doc, CUTOFF, {})
    wrong = result(doc)
    wrong["events"][0]["quotes"] = ["属于另一家公司而非本份原文的内容"]
    stored = cache.finish(doc, wrong, {"first_response": "保留"}, ["KNOWN_MANUAL_SEMANTIC_ERROR"])
    assert stored["state"] == "REJECTED"
    assert any("QUOTE_NOT_IN_THIS_SOURCE" in item for item in stored["issues"])
    calls = []
    summary = execute(snapshot([doc]), cache, tmp_path / "second", successful_parser(calls))
    assert summary["after"] == {"BLOCKED": 1} and summary["model_calls"] == 0
    assert calls == [] and cache.results_count() == 1


def test_unknown_first_transport_is_never_resent_or_overwritten(tmp_path):
    cache, doc, calls = DocumentParseCache(tmp_path / "db.sqlite3"), document(), []

    def timeout(value):
        calls.append(value["id"])
        return {"result": None, "raw": {"error": "TimeoutError"}, "model_called": True, "failure": "UNKNOWN"}

    execute(snapshot([doc]), cache, tmp_path / "first", timeout)
    second = execute(snapshot([doc]), cache, tmp_path / "second", timeout)
    assert calls == [doc["id"]] and second["model_calls"] == 0
    assert second["unknown_outcome_blocks_more_calls"] is True
    with pytest.raises(sqlite3.IntegrityError):
        cache.finish(doc, result(doc), {})
    assert cache.lookup(doc, CUTOFF)["reason"] == "UNKNOWN"


def test_historical_lookup_excludes_future_input_context_and_future_source(tmp_path):
    cache, doc = DocumentParseCache(tmp_path / "db.sqlite3"), document()
    cache.claim(doc, CUTOFF, {})
    cache.finish(doc, result(doc), {})
    historical = "2026-10-09T17:21:00+08:00"
    decision = cache.lookup(doc, historical)
    assert decision["state"] == "EXCLUDED" and "result" not in decision
    assert decision["reason"] == "PARSE_CONTEXT_AFTER_CUTOFF"
    future_doc = {**doc, "available_at": "2026-10-14T09:00:00+08:00"}
    assert cache.lookup(future_doc, CUTOFF)["reason"] == "SOURCE_AFTER_CUTOFF"


def test_two_workers_cannot_claim_same_material_twice(tmp_path):
    path, doc = tmp_path / "db.sqlite3", document()
    first, second = DocumentParseCache(path), DocumentParseCache(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(cache.claim, doc, CUTOFF, {}) for cache in (first, second)]
        assert sum(future.result() for future in futures) == 1


def test_partial_source_does_not_become_complete_parse_cache_hit(tmp_path):
    cache, doc = DocumentParseCache(tmp_path / "db.sqlite3"), document()
    doc["body_truncated"] = True
    cache.claim(doc, CUTOFF, {})
    cache.finish(doc, result(doc), {})
    assert cache.lookup(doc, CUTOFF)["reason"] == "REJECTED"
    full = {**doc, "body_truncated": False}
    assert cache.lookup(full, CUTOFF)["state"] == "PENDING"


def test_corrupted_cached_result_is_rejected_on_read(tmp_path):
    cache, doc = DocumentParseCache(tmp_path / "db.sqlite3"), document()
    cache.claim(doc, CUTOFF, {})
    cache.finish(doc, result(doc), {})
    altered = deepcopy(result(doc))
    altered["events"][0]["meaning"] = "被改写的结果"
    with cache.connection() as connection:
        connection.execute("UPDATE parse_result SET result_json=?", (json.dumps(altered, ensure_ascii=False),))
    assert "CACHE_RESULT_HASH_MISMATCH" in cache.lookup(doc, CUTOFF)["issues"]


def test_import_keeps_old_findings_and_is_idempotent(tmp_path):
    archive = tmp_path / "archive"
    (archive / "events").mkdir(parents=True)
    doc = document()
    files = {
        "paired-input.json": snapshot([doc]),
        "protocol.json": {"model": "old-provider"},
        "event-jobs.json": [{"job_id": "old-job", "document_id": doc["id"], "start": 0, "end": len(BODY)}],
        "events/old-job.json": {"events": result(doc)["events"], "issues": []},
        "manual-findings-01.json": [{"job": "old-job", "severity": "material", "id": "BAD-1", "kind": "已知错误"}],
    }
    for filename, value in files.items():
        (archive / filename).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    cache = DocumentParseCache(tmp_path / "db.sqlite3")
    assert import_archive(cache, archive)["counts"] == {"REJECTED": 1}
    assert import_archive(cache, archive)["counts"] == {"already_stored": 1}
    assert cache.lookup(doc, CUTOFF)["state"] == "BLOCKED"
    assert cache.results_count() == 1


def test_bounded_run_resumes_only_remaining_material(tmp_path):
    cache, calls = DocumentParseCache(tmp_path / "db.sqlite3"), []
    docs = [document("第一份"), document("第二份")]
    first = execute(snapshot(docs), cache, tmp_path / "first", successful_parser(calls), max_calls=1)
    second = execute(snapshot(docs), cache, tmp_path / "second", successful_parser(calls))
    assert first["after"] == {"REUSE": 1, "PENDING": 1}
    assert second["model_calls"] == 1 and second["cache_hits_before"] == 1
    assert calls == ["第一份", "第二份"]


def test_later_review_withdraws_reuse_without_overwriting_first_result(tmp_path):
    cache, doc = DocumentParseCache(tmp_path / "db.sqlite3"), document()
    cache.claim(doc, CUTOFF, {})
    cache.finish(doc, result(doc), {"first": "原样保留"})
    with cache.connection() as connection:
        before = tuple(connection.execute("SELECT * FROM parse_result").fetchone())
    cache.reject_after_review(doc, ["NEW_MANUAL_SEMANTIC_FINDING"], CUTOFF, {"reviewer": "独立原文复核"})
    assert cache.lookup(doc, CUTOFF)["reason"] == "INDEPENDENT_REVIEW_WITHDREW_REUSE"
    with cache.connection() as connection:
        after = tuple(connection.execute("SELECT * FROM parse_result").fetchone())
    assert before == after
    calls = []
    assert execute(snapshot([doc]), cache, tmp_path / "next", successful_parser(calls))["model_calls"] == 0


def test_future_review_context_is_not_applied_to_earlier_replay(tmp_path):
    cache, doc = DocumentParseCache(tmp_path / "db.sqlite3"), document()
    historical = "2026-10-09T17:21:00+08:00"
    cache.claim(doc, historical, {})
    cache.finish(doc, result(doc), {})
    cache.reject_after_review(doc, ["REQUIRES_LATER_INPUT"], CUTOFF, {"reviewer": "later input review"})
    assert cache.lookup(doc, historical)["state"] == "REUSE"
    assert cache.lookup(doc, CUTOFF)["state"] == "BLOCKED"


def test_publication_day_uses_client_timezone_when_cutoff_is_utc(tmp_path):
    cache, doc = DocumentParseCache(tmp_path / "db.sqlite3"), document()
    doc.update(date="2026-10-13", available_at="2026-10-13T00:00:00+08:00")
    cutoff_utc = "2026-10-12T17:00:00+00:00"
    assert cache.claim(doc, cutoff_utc, {})
    cache.finish(doc, result(doc), {})
    assert cache.lookup(doc, cutoff_utc)["state"] == "REUSE"
