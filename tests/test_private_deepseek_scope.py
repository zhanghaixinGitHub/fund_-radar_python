"""私人凭据范围、增量解析和费用边界的离线回归；所有模型请求均为替身。"""

from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import pytest
from app.core.config import Settings, get_settings
from app.core.deepseek_policy import PredictionScope, PrivateDeepSeekSettings, prediction_scope, require_scope
from app.integrations.fund_information_analysis import request
from app.services.deepseek_usage import UsageLedger
from app.services.fund_document_parse_cache import DocumentParseCache, digest
from app.services.fund_material_parsing import parse_documents


def document(identity="article-a", body="公司尚未完成回购，相关计划仍需要股东大会批准。"):
    return {
        "id": identity,
        "kind": "ANNOUNCEMENT",
        "title": "回购计划公告",
        "source_hash": digest(body),
        "body": body,
        "body_scope": "FULL_TEXT",
        "body_truncated": False,
        "date": "2026-10-12",
        "available_at": "2026-10-12T10:00:00+08:00",
        "codes": ["300001"],
    }


def parsed(doc):
    return {
        "id": doc["id"],
        "events": [
            {
                "title": "回购计划",
                "assessment": "NEUTRAL",
                "stage": "拟议",
                "quotes": [doc["body"]],
                "meaning": "计划尚未执行",
                "mechanism": "效果取决于后续执行",
                "caveat": "仍需批准，不能视为已经回购",
            }
        ],
    }


def data(documents):
    return {
        "fund_code": "002112",
        "documents": documents,
        "as_of": "2026-10-13T12:00:00+08:00",
        "window": {"target_nav_date": "2026-10-13"},
    }


@pytest.fixture
def archive(tmp_path, monkeypatch):
    path = tmp_path / "parses.sqlite3"
    monkeypatch.setattr(
        "app.services.fund_material_parsing.get_settings", lambda: SimpleNamespace(fund_parse_cache_path=str(path))
    )
    return DocumentParseCache(path)


@pytest.mark.parametrize("fund,horizon", [("000001", 1), ("002112", 5), ("002112", 20)])
def test_other_funds_and_horizons_cannot_open_private_scope(fund, horizon):
    with pytest.raises(ValueError, match="SCOPE_DENIED"), prediction_scope(fund, horizon, "2026-10-13"):
        pytest.fail("不应获得私人凭据范围")


def test_direct_request_without_live_scope_is_denied_before_budget_or_http():
    with pytest.raises(ValueError, match="SCOPE_DENIED"):
        request("any prompt", {"fund_code": "002112"}, "SYNTHESIS", None)


@pytest.mark.parametrize(
    "stage,value",
    [
        ("NARRATIVE", {}),
        ("EVENT_REPAIR", {}),
        ("EVENT", {"fund_code": "000001"}),
        ("SYNTHESIS", {"target_nav_date": "2026-10-14"}),
    ],
)
def test_scope_checks_each_stage_and_input_identity(stage, value):
    with prediction_scope("002112", 1, "2026-10-13"), pytest.raises(ValueError, match="SCOPE_DENIED"):
        require_scope(stage, value)


def test_scope_is_removed_after_exception():
    with pytest.raises(RuntimeError), prediction_scope("002112", 1, "2026-10-13"):
        raise RuntimeError("caller stopped")
    with pytest.raises(ValueError, match="SCOPE_DENIED"):
        require_scope("EVENT")


def test_legacy_settings_cannot_expose_any_paid_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "old-research-key")
    get_settings.cache_clear()
    try:
        assert get_settings().deepseek_api_key.get_secret_value() == ""
    finally:
        get_settings.cache_clear()


def test_working_directory_does_not_create_a_second_history_or_usage_database(tmp_path, monkeypatch):
    before = Settings(_env_file=None, fund_parse_cache_path="data/fund-materials/parsed-documents.sqlite3")
    monkeypatch.chdir(tmp_path)
    after = Settings(_env_file=None, fund_parse_cache_path="data/fund-materials/parsed-documents.sqlite3")
    assert before.fund_parse_cache_path == after.fund_parse_cache_path
    from pathlib import Path

    assert Path(after.fund_parse_cache_path).is_absolute()


def test_old_material_is_reused_across_new_prediction_day(archive):
    doc = document()
    archive.claim(doc, data([doc])["as_of"], {"model": "historical-model"})
    archive.finish(doc, parsed(doc), {"first": "preserved"})
    later = data([doc])
    later["as_of"] = "2026-10-14T12:00:00+08:00"
    later["window"]["target_nav_date"] = "2026-10-14"
    result = parse_documents(later, "new prompt", None, lambda *_: pytest.fail("旧材料不得付费重读"))
    assert len(result) == 1
    assert archive.results_count() == 1


@pytest.mark.parametrize("kind", ["ANNOUNCEMENT", "NEWS", "POLICY"])
def test_only_new_or_changed_material_calls_parser_once(archive, kind):
    old, new = document(), document("article-b")
    old["kind"] = new["kind"] = kind
    archive.claim(old, data([old])["as_of"], {"model": "old-model"})
    archive.finish(old, parsed(old), {})
    calls = []

    def provider(prompt, value, stage, budget):
        calls.append((stage, [doc["id"] for doc in value["documents"]]))
        return {"items": [parsed(value["documents"][0])]}

    parse_documents(data([old, new]), "prompt", None, provider)
    parse_documents(data([old, new]), "prompt changed", None, provider)
    assert calls == [("EVENT", ["article-b"])]
    changed = deepcopy(new)
    changed["body"] += "后续进展需要另行公告。"
    changed["source_hash"] = digest(changed["body"])
    parse_documents(data([old, changed]), "prompt", None, provider)
    assert calls == [("EVENT", ["article-b"]), ("EVENT", ["article-b"])]


def test_known_error_is_skipped_while_new_material_is_parsed(archive):
    old, new = document(), document("new")
    archive.claim(old, data([old])["as_of"], {})
    archive.finish(old, parsed(old), {})
    archive.reject_after_review(old, ["KNOWN_SEMANTIC_ERROR"], data([old])["as_of"], {"review": "manual"})
    calls, skipped = [], []

    def provider(prompt, value, stage, budget):
        calls.append(value["documents"][0]["id"])
        return {"items": [parsed(value["documents"][0])]}

    events = parse_documents(data([old, new]), "prompt", None, provider, skipped_documents=skipped)
    assert calls == [new["id"]]
    assert {event["document"]["id"] for event in events} == {new["id"]}
    assert skipped[0]["documentId"] == old["id"]
    assert archive.lookup(old, data([old])["as_of"])["reason"] == "INDEPENDENT_REVIEW_WITHDREW_REUSE"


def test_failed_first_response_is_not_automatically_repeated(archive):
    doc, calls = document(), []

    def provider(*_):
        calls.append(1)
        raise TimeoutError("unknown result")

    first_skipped, second_skipped = [], []
    assert parse_documents(data([doc]), "prompt", None, provider, skipped_documents=first_skipped) == []
    assert parse_documents(data([doc]), "prompt", None, provider, skipped_documents=second_skipped) == []
    assert first_skipped[0]["reasonCode"] == "DOCUMENT_PARSE_FAILED"
    assert second_skipped[0]["reasonCode"] == "UNKNOWN"
    assert calls == [1]


def test_daily_quota_is_persistent_and_unknown_calls_keep_their_reservation(tmp_path):
    path = tmp_path / "usage.sqlite3"
    settings = PrivateDeepSeekSettings(_env_file=None, api_key="test-only", max_calls_per_day=2)
    scope = PredictionScope("002112", 1, str(date.today()))
    ledger = UsageLedger(path)
    ledger.reserve("first", scope, settings, "EVENT", 1000, 1000)
    ledger.finish("first", None, None, "UNKNOWN")
    UsageLedger(path).reserve("second", scope, settings, "SYNTHESIS", 1000, 1000)
    with pytest.raises(ValueError, match="DAILY_BUDGET"):
        UsageLedger(path).reserve("third", scope, settings, "AUDIT", 1000, 1000)
    with pytest.raises(ValueError, match="STAGE_FAILED"):
        UsageLedger(path).reserve("first", scope, settings, "EVENT", 1000, 1000)


def test_usage_receipt_keeps_real_tokens_and_cache_discount(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.sqlite3")
    settings = PrivateDeepSeekSettings(_env_file=None, api_key="test-only")
    scope = PredictionScope("002112", 1, str(date.today()))
    ledger.reserve("key", scope, settings, "SYNTHESIS", 30000, 3000)
    receipt = ledger.finish(
        "key",
        {"prompt_tokens": 10000, "completion_tokens": 1000, "prompt_cache_hit_tokens": 8000},
        "test-response",
        "SUCCEEDED",
    )
    assert receipt["estimated_peak_cny"] == pytest.approx(0.01232)
    assert receipt["prompt_tokens"] == 10000


def test_quota_refusal_does_not_turn_an_unparsed_new_file_into_a_permanent_error(archive):
    doc = document()
    with pytest.raises(ValueError, match="DAILY_BUDGET"):
        parse_documents(
            data([doc]), "prompt", None, lambda *_: (_ for _ in ()).throw(ValueError("DEEPSEEK_DAILY_BUDGET"))
        )
    assert archive.lookup(doc, data([doc])["as_of"])["state"] == "PENDING"
    assert archive.results_count() == 0


def test_partial_source_is_skipped_without_any_paid_request(archive):
    doc = document()
    doc["body_truncated"] = True
    skipped = []
    assert (
        parse_documents(
            data([doc]), "prompt", None, lambda *_: pytest.fail("不能收费解析被截断资料"), skipped_documents=skipped
        )
        == []
    )
    assert skipped[0]["reasonCode"] == "PARTIAL_SOURCE_NOT_FULL_PARSE"


def test_invalid_new_response_does_not_stop_other_material_or_repeat_first_attempt(archive):
    """第一份返回错误也继续第二份；再次检查仍保留错误首次响应而不重发。"""
    bad, good = document("bad"), document("good")
    calls, skipped = [], []

    def provider(prompt, value, stage, budget):
        identity = value["documents"][0]["id"]
        calls.append(identity)
        return {"items": []} if identity == "bad" else {"items": [parsed(good)]}

    events = parse_documents(data([bad, good]), "prompt", None, provider, skipped_documents=skipped)
    assert calls == ["bad", "good"]
    assert {event["document"]["id"] for event in events} == {"good"}
    assert skipped[0]["documentId"] == "bad"
    with archive.connection() as connection:
        before = tuple(
            connection.execute("SELECT * FROM parse_result WHERE cache_key=?", (archive.key(bad),)).fetchone()
        )
    again = parse_documents(data([bad, good]), "prompt", None, provider)
    assert events == again and calls == ["bad", "good"]
    with archive.connection() as connection:
        after = tuple(
            connection.execute("SELECT * FROM parse_result WHERE cache_key=?", (archive.key(bad),)).fetchone()
        )
    assert before == after


def test_material_call_limit_leaves_synthesis_capacity_and_keeps_reading_cached_material(archive):
    """新材料达到上限时仍读取排在后面的旧资料，未开始的新材料下次可以继续。"""
    first, pending, old = document("first"), document("pending"), document("old")
    archive.claim(old, data([old])["as_of"], {})
    archive.finish(old, parsed(old), {})
    budget = SimpleNamespace(calls=0, max_calls=3)
    calls, skipped = [], []

    def provider(prompt, value, stage, remaining):
        calls.append(value["documents"][0]["id"])
        remaining.calls += 1
        return {"items": [parsed(value["documents"][0])]}

    events = parse_documents(data([first, pending, old]), "prompt", budget, provider, skipped_documents=skipped)
    assert calls == ["first"] and budget.calls == 1
    assert {event["document"]["id"] for event in events} == {"first", "old"}
    assert skipped[0]["reasonCode"] == "ANALYSIS_CALL_LIMIT"
    assert archive.lookup(pending, data([pending])["as_of"])["state"] == "PENDING"
    later = parse_documents(data([first, pending, old]), "prompt", SimpleNamespace(calls=0, max_calls=3), provider)
    assert calls == ["first", "pending"]
    assert {event["document"]["id"] for event in later} == {"first", "pending", "old"}


def test_incomplete_legacy_lookup_only_skips_unverified_material(archive, monkeypatch):
    """历史核对范围不足不能猜测首次收费，但不阻止已可复用的材料继续使用。"""
    old, unknown = document("old"), document("unknown")
    archive.claim(old, data([old])["as_of"], {})
    archive.finish(old, parsed(old), {})
    monkeypatch.setattr(
        "app.services.fund_material_parsing.get_settings",
        lambda: SimpleNamespace(fund_parse_cache_path=str(archive.path), fund_parse_import_legacy=True),
    )
    monkeypatch.setattr(
        "app.services.fund_material_parse_history.import_legacy",
        lambda *args: (_ for _ in ()).throw(ValueError("ANALYSIS_DOCUMENT_CACHE_BLOCKED")),
    )
    skipped = []
    events = parse_documents(
        data([old, unknown]),
        "prompt",
        None,
        lambda *_: pytest.fail("旧身份未确认不能收费"),
        skipped_documents=skipped,
    )
    assert {event["document"]["id"] for event in events} == {"old"}
    assert skipped[0]["documentId"] == "unknown"
    assert skipped[0]["reasonCode"] == "LEGACY_HISTORY_UNVERIFIED"


def test_invalid_source_metadata_only_skips_that_document(archive):
    invalid, good = document("invalid"), document("good")
    invalid["available_at"] = "invalid-date"
    skipped, calls = [], []

    def provider(prompt, value, stage, budget):
        calls.append(value["documents"][0]["id"])
        return {"items": [parsed(good)]}

    events = parse_documents(data([invalid, good]), "prompt", None, provider, skipped_documents=skipped)
    assert calls == ["good"] and len(events) == 1
    assert skipped[0]["reasonCode"] == "DOCUMENT_SOURCE_METADATA_INVALID"


def test_legacy_audit_metadata_is_rebuilt_without_mutating_or_charging_the_saved_result(archive):
    """旧本地解析附带的审计字段不应让可用事实失效，也不能作为本次来源的凭据。"""
    doc = document("legacy-audit")
    saved = parsed(doc)
    saved["events"][0].update(
        id="old-internal-id",
        document={"id": "different-old-context"},
        quote_spans=[{"start": 999, "end": 1000}],
    )
    archive.claim(doc, data([doc])["as_of"], {})
    archive.finish(doc, saved, {})
    with archive.connection() as connection:
        before = tuple(
            connection.execute("SELECT * FROM parse_result WHERE cache_key=?", (archive.key(doc),)).fetchone()
        )
    events = parse_documents(data([doc]), "prompt", None, lambda *_: pytest.fail("旧材料不能重新收费"))
    assert events[0]["document"]["id"] == doc["id"]
    assert events[0]["id"] == doc["id"] + ":0"
    assert events[0]["quote_spans"] == [{"start": 0, "end": len(doc["body"])}]
    with archive.connection() as connection:
        after = tuple(
            connection.execute("SELECT * FROM parse_result WHERE cache_key=?", (archive.key(doc),)).fetchone()
        )
    assert before == after


@pytest.mark.parametrize("bound", [True, False])
@pytest.mark.parametrize("original_day", ["2026-10-12", "2025-10-12"])
def test_legacy_paid_parse_is_reused_only_with_verified_original_request(archive, bound, original_day):
    from app.schemas.fund_information_analysis import quote_catalog
    from app.services.fund_material_parse_history import import_rows

    doc = document()
    doc["date"] = original_day
    doc["available_at"] = original_day + "T10:00:00+08:00"
    prompt = "original prompt"
    old = {"fund_code": "002112", "documents": [doc], "as_of": original_day + "T12:00:00+08:00"}
    value = {"documents": [doc], "quote_catalogs": {doc["id"]: quote_catalog(doc["body"])}}
    key = digest({"prompt": prompt, "input": value, "model": "deepseek-v4-pro", "endpoint": "https://api.deepseek.com"})
    payload = {"items": [parsed(doc)]}
    rows = [
        {
            "cache_key": key if bound else "unverified",
            "state": "READY",
            "payload": payload,
            "content_hash": digest(payload),
        }
    ]
    snapshots = [{"payload": old, "content_hash": digest(old)}]
    originals = deepcopy((rows, snapshots))
    receipt = import_rows([doc], data([doc])["as_of"], prompt, archive, snapshots, rows)
    assert receipt == {"imported": 1 if bound else 0, "blocked": 0 if bound else 1}
    assert archive.lookup(doc, data([doc])["as_of"])["state"] == ("REUSE" if bound else "BLOCKED")
    assert (rows, snapshots) == originals


def test_transport_cannot_reparse_an_already_saved_material(archive):
    from app.services.fund_material_parsing import require_new_material

    doc = document()
    archive.claim(doc, data([doc])["as_of"], {"origin": "LIVE_002112_T1", "target_nav_date": "2026-10-13"})
    value = {"documents": [doc], "target_nav_date": "2026-10-13"}
    require_new_material(value)
    archive.finish(doc, parsed(doc), {})
    with pytest.raises(ValueError, match="MATERIAL_SCOPE_DENIED"):
        require_new_material(value)


@pytest.mark.parametrize("legacy_version", ["three", "ten"])
def test_quote_limit_prompt_change_does_not_lose_legacy_paid_parse_reuse(archive, legacy_version):
    from app.schemas.fund_information_analysis import quote_catalog
    from app.services.fund_information_analysis import EVENT_PROMPT
    from app.services.fund_material_parse_history import (
        LEGACY_EVENT_PROMPT_TEN_QUOTES,
        LEGACY_EVENT_PROMPT_THREE_QUOTES,
        import_rows,
    )

    doc = document()
    old = {"fund_code": "002112", "documents": [doc], "as_of": "2026-10-12T12:00:00+08:00"}
    value = {"documents": [doc], "quote_catalogs": {doc["id"]: quote_catalog(doc["body"])}}
    historical_prompt = (
        LEGACY_EVENT_PROMPT_THREE_QUOTES if legacy_version == "three" else LEGACY_EVENT_PROMPT_TEN_QUOTES
    )
    key = digest(
        {
            "prompt": historical_prompt,
            "input": value,
            "model": "deepseek-flash",
            "endpoint": "https://api.deepseek.com",
        }
    )
    payload = {"items": [parsed(doc)]}
    rows = [{"cache_key": key, "state": "READY", "payload": payload, "content_hash": digest(payload)}]
    snapshots = [{"payload": old, "content_hash": digest(old)}]
    original = deepcopy((rows, snapshots))
    assert EVENT_PROMPT != historical_prompt
    result = import_rows([doc], data([doc])["as_of"], EVENT_PROMPT, archive, snapshots, rows)
    assert result == {"imported": 1, "blocked": 0}
    assert archive.lookup(doc, data([doc])["as_of"])["state"] == "REUSE"
    assert (rows, snapshots) == original
