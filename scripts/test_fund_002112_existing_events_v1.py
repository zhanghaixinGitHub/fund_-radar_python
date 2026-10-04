"""T01—T19：真实输入只做只读审查；唯一监督测试使用完全人造数据并独立记账。"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import joblib
import numpy as np
import pytest

from scripts import fund_002112_existing_events_features_v1 as f
from scripts import fund_002112_existing_events_fit_v1 as fit
from scripts import fund_002112_existing_events_sources_v1 as s
from scripts import fund_002112_existing_events_v1 as run

DAYS = ["2023-01-06", "2023-01-09", "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2023-01-16"]
TEST_ROOT = Path(os.environ.get("FUND_EVENTS_TEST_ROOT", str(s.ROOT)))


@pytest.mark.parametrize("clock", ["07:00:00", "11:30:00", "14:59:59"])
def test_t01_t02_before_1500(clock):
    assert f.effective_session("2023-01-09", f"2023-01-09T{clock}+08:00", DAYS) == "2023-01-09"


@pytest.mark.parametrize("clock", ["15:00:00", "18:00:00"])
def test_t03_strict_cutoff(clock):
    assert f.effective_session("2023-01-09", f"2023-01-09T{clock}+08:00", DAYS) == "2023-01-10"


def test_t04_date_precision():
    raw = {"published_date": "2023-01-09", "title": "医药行业消息", "available_at": "2023-01-10T08:00:00+08:00"}
    e = f.event_record(raw, {}, "NEWS")
    assert e["published_at"] is None
    assert e["date_only_assumed_same_day"]
    assert f.effective_session(e["published_date"], None, DAYS, "main") == "2023-01-09"
    assert f.effective_session(e["published_date"], None, DAYS, "aux") == "2023-01-10"


@pytest.mark.parametrize("timing", ["main", "aux"])
def test_t05_weekend(timing):
    assert f.effective_session("2023-01-07", None, DAYS, timing) == "2023-01-09"
    assert f.effective_session("2023-01-06", "2023-01-06T15:00:00+08:00", DAYS, timing) == "2023-01-09"


@pytest.mark.parametrize("age,expected", [(0, [1, 1, 1]), (4, [0, 1, 1]), (5, [0, 0, 1]), (19, [0, 0, 1])])
def test_t06_window(age, expected):
    e = f.event_record({"published_date": "2023-01-09", "title": "行业消息"}, {}, "NEWS")
    counts = f.counts_and_members([e], [[0, age]], set())
    assert np.allclose(counts[:3], np.log1p(expected))


def test_t06_age20_membership_not_included():
    event = f.event_record({"title": "行业消息", "published_date": "2023-01-09"}, {}, "NEWS")
    assert f.counts_and_members([event], [[0, 20]], set()) == [0.0] * 36
    assert f.counts_and_members([], [], set()) == [0.0] * 36
    assert len(set(f.STATS)) == 36


def test_t07_original_year():
    sessions = ["2017-01-09", "2017-01-10", "2024-01-02"]
    assert f.effective_session("2017-01-09", None, sessions) == "2017-01-09"
    assert f.effective_session("2017-01-09", None, sessions, "aux") == "2017-01-10"


def test_t08_temporal_dedup_and_progress():
    original = f.event_record(
        {
            "title": "公司回购公告",
            "published_date": "2023-01-09",
            "url": "https://static.cninfo.com.cn/finalpage/2023-01-09/123456.PDF",
        },
        {},
        "ANNOUNCEMENT",
    )
    later = f.event_record(
        {
            "title": "公司回购公告",
            "published_date": "2023-01-10",
            "url": "https://static.cninfo.com.cn/finalpage/2023-01-09/123456.PDF",
        },
        {},
        "ANNOUNCEMENT",
        "后来转载补充正文",
    )
    progress = f.event_record(
        {"title": "公司回购进展公告", "published_date": "2023-01-11", "url": "https://x/2"}, {}, "ANNOUNCEMENT"
    )
    events, decisions = f.deduplicate([later, progress, original])
    assert len(events) == 2 and len(decisions) == 1
    assert events[0]["existing_text"] == "" and events[0]["published_date"] == "2023-01-09"
    assert events[1]["event_stage"] == "UPDATE"


def test_t09_title_only():
    e = f.event_record(
        {"title": "业绩预告", "published_date": "2023-01-09", "training_eligible": False}, {}, "ANNOUNCEMENT"
    )
    assert e["usable_for_text"] and e["text"] == "业绩预告"
    assert e["old_training_eligible"] is False and e["text_level"] == "TITLE_ONLY"


def test_t10_failure_is_not_zero(tmp_path):
    with pytest.raises(FileNotFoundError):
        s.Snapshot(tmp_path).get(tmp_path / "missing.json")
    source = tmp_path / "source.json"
    source.write_text("{}", "utf-8")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        s.Snapshot(tmp_path).get(source, "0" * 64)
    assert f.counts_and_members([], [], set()) == [0.0] * 36


def test_t11_actual_three_types_in_inputs():
    if not (TEST_ROOT / "input-manifest.json").exists():
        pytest.skip("真实输入尚未生成，freeze阶段必须再次执行")
    manifest = s.read(TEST_ROOT / "input-manifest.json")
    for timing in f.TIMINGS:
        for kind in f.TYPES:
            evidence = manifest["evidence"][timing][kind]
            assert evidence["varying_statistic"] and evidence["text_characters"] > 0
            assert evidence["training_document_candidates"] > 0 and evidence["dates_with_information"] > 0


def artificial_data():
    events = [
        f.event_record({"title": title, "published_date": "2023-01-09"}, {}, kind)
        for kind, title in [
            ("NEWS", "医药消息"),
            ("POLICY", "医药采购通知"),
            ("ANNOUNCEMENT", "业绩预告"),
            ("NEWS", "未来专属词彗星闪耀"),
        ]
    ]
    rows = [
        {
            "target": str(i),
            "n8": [float((i + j) % 13) for j in range(8)],
            "main": [float((i + j) % 5) for j in range(36)],
        }
        for i in range(60)
    ]
    members = [{"main": [[0, i % 20], [1, i % 20], [2, i % 20]]} for i in range(60)]
    return rows, members, events


def test_t12_fold_vocabulary_only():
    rows, members, events = artificial_data()
    prep = fit.fit_preprocessor(rows, members, events, "B2", "main")
    assert "彗星" not in prep["vectorizers"]["NEWS"].vocabulary_
    assert prep["training_document_indices"]["NEWS"] == [0]
    assert np.allclose(prep["scaler"].mean_, fit.numeric(rows, "B2", "main").mean(axis=0))
    old = dict(prep["vectorizers"]["NEWS"].vocabulary_)
    fit.transform(prep, rows[:1], [{"main": [[3, 0]]}], events)
    assert old == prep["vectorizers"]["NEWS"].vocabulary_


def test_t13_answer_and_revision_leakage():
    e = f.event_record({"title": "德邦鑫星价值基金净值公告", "published_date": "2023-01-09"}, {}, "ANNOUNCEMENT")
    assert e["force_next"]
    assert f.effective_session(e["published_date"], None, DAYS, force_next=True) == "2023-01-10"
    revised = f.event_record(
        {"title": "医药政策通知", "published_date": "2023-01-09"},
        {},
        "POLICY",
        "后续修订正文",
        {"body_version_updated_at": "2023-01-11T16:00:00+08:00"},
    )
    assert revised["existing_text"] == "" and revised["text_level"] == "TITLE_ONLY"
    assert f.effective_session("2023-01-09", None, DAYS, revision_at="2023-01-11T16:00:00+08:00") == "2023-01-12"
    with pytest.raises(ValueError, match="HISTORY_TOO_SHORT"):
        f.nav_features([1.0] * 60)


def test_t14_historical_holdings():
    reports = [
        {
            "available_at": "2023-01-09T08:00:00+08:00",
            "report_end": "2022-12-31",
            "holdings": [{"stock_code": "600001.SH"}],
        },
        {
            "available_at": "2023-01-11T08:00:00+08:00",
            "report_end": "2023-03-31",
            "holdings": [{"stock_code": "600002.SH"}],
        },
    ]
    assert f.select_holdings(reports, "2023-01-10")[0] == {"600001"}
    e = f.event_record(
        {"title": "公司回购", "published_date": "2023-01-09", "latestHeld": True, "stockCode": "600002.SH"},
        {},
        "ANNOUNCEMENT",
    )
    assert e["relation_scope"] == "BACKGROUND"


def test_t15_policy_not_effective():
    e = f.event_record({"title": "药品采购管理办法征求意见通知", "published_date": "2023-01-09"}, {}, "POLICY")
    assert e["primary_type"] == "POLICY" and e["usable_for_count"]
    assert "effective_date" not in e and "benefit" not in e
    assert f.classify("药品采购办法政策解读", "POLICY") == "NEWS"


def test_t16_network_prohibited():
    with s.offline_guard() as attempts:
        with pytest.raises(RuntimeError, match="BUDGET_ZERO"):
            socket.create_connection(("example.invalid", 443))
        with pytest.raises(RuntimeError, match="BUDGET_ZERO"):
            socket.getaddrinfo("example.invalid", 443)
    assert len(attempts) == 2
    # 测试主动拦截探针无实际网络流量，实验入口不导入HTTP/浏览器/大模型客户端。
    for path in run.CODE[:4]:
        text = path.read_text("utf-8")
        assert "requests.get(" not in text and "httpx.get(" not in text and "openai." not in text


def test_t17_budget_failed_and_resume(tmp_path):
    with s.writer_lock(tmp_path):
        assert run.reserve(tmp_path, "test", "main", fit.RECIPE) == 1
        s.append(tmp_path / "fit-ledger.jsonl", {"action": "FAILED", "attempt": 1, "run_id": "test"})
        assert run.budget(tmp_path)["consumed"] == 1
        assert run.reserve(tmp_path, "test", "retry", fit.RECIPE) == 2
        for _i in range(2):
            run.reserve(tmp_path, "other", "retry", fit.RECIPE)
        with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
            run.reserve(tmp_path, "other", "retry", fit.RECIPE)
    directory = run.run_path(tmp_path, "completed")
    directory.mkdir(parents=True)
    s.save(directory / "dummy.json", {})
    s.save(directory / "complete.json", {"files": {"dummy.json": s.sha(directory / "dummy.json")}})
    before = run.budget(tmp_path)["consumed"]
    run.validate_completed(tmp_path, "completed")
    assert run.budget(tmp_path)["consumed"] == before


def test_t18_serialization_artificial_only(tmp_path):
    rows, members, events = artificial_data()
    prep = fit.fit_preprocessor(rows, members, events, "B2", "main")
    matrix = fit.transform(prep, rows, members, events)
    calls = []

    def reserve(params):
        calls.append(params)
        audit = os.environ.get("FUND_EVENTS_SYNTHETIC_AUDIT")
        if audit:
            s.append(
                Path(audit),
                {
                    "at": s.now(),
                    "synthetic_supervised_fit": True,
                    "source": "artificial_data function; no real rows",
                    "pid": os.getpid(),
                },
            )
        return "ARTIFICIAL_ONLY"

    model, _ = fit.train(prep, matrix, ["UP" if i % 2 else "DOWN" for i in range(60)], reserve)
    assert len(calls) == 1
    joblib.dump(model, tmp_path / "model.joblib")
    first = fit.predict(model, matrix, rows)
    second = fit.predict(joblib.load(tmp_path / "model.joblib"), matrix, rows)
    assert run.probability_error(first, second) <= 1e-10
    assert [r["predicted"] for r in first] == [r["predicted"] for r in second]
    assert all(r["probabilities"]["FLAT"] is None for r in first)


def test_t19_old_protection_readback():
    if not (s.ROOT / "protection-before.json").exists():
        pytest.skip("E00尚未完成")
    protected = s.read(s.ROOT / "protection-before.json")["protected"]
    assert protected and all(s.sha(path) == hashed for path, hashed in protected.items())


def test_label_boundary_requires_frozen_protocol(tmp_path):
    with pytest.raises(FileNotFoundError):
        run.labels(tmp_path, "C2026", "train", [{"target": "2026-09-29"}])


def test_block_interval_respects_gaps():
    result = fit.block_interval([1, 0, -1, 1, 0, -1], [1, 2, 3, 8, 9, 10])
    assert result["blocks"] == 2 and result["replicates"] == 2000
    assert result["missing_sessions_preserved"]


def test_lock_excludes_second_writer(tmp_path):
    with s.writer_lock(tmp_path), pytest.raises(OSError), s.writer_lock(tmp_path):
        pass


def test_review001_official_semantics_and_commentary():
    assert f.classify("医保配套措施的意见", "NEWS", "OFFICIAL_POLICY_OR_NOTICE") == "POLICY"
    assert f.classify("办法征求公众意见", "NEWS", "DRAFT_OR_PUBLIC_CONSULTATION") == "POLICY"
    assert f.classify("医保配套措施意见解读", "NEWS", "INTERPRETATION_NOT_ORIGINAL_POLICY") == "NEWS"
    assert f.classify("专家对政策提出意见", "NEWS") == "NEWS"
    assert f.classify("关于做好新闻宣传工作的通知", "NEWS", "OFFICIAL_POLICY_OR_NOTICE") == "POLICY"
    assert (
        f.classify(
            "国家医疗保障局等九部门印发实施意见",
            "NEWS",
            "OFFICIAL_POLICY_OR_NOTICE",
            url="https://www.nhsa.gov.cn/art/2019/9/30/art_14_1815.html",
            body="国家医疗保障局等9部门日前印发《实施意见》，意味着将全国推开。",
        )
        == "NEWS"
    )
    assert (
        f.classify(
            "国务院关于整合城乡居民基本医疗保险制度的意见",
            "NEWS",
            url="https://www.nhsa.gov.cn/art/2016/1/3/art_104_6426.html",
        )
        == "POLICY"
    )


def test_review002_specific_article_reprint_dedup():
    url = "https://www.nhsa.gov.cn/art/2025/2/17/art_14_15702.html"
    early = f.event_record(
        {"title": "从后付制到即时结——医保结算提速", "published_date": "2025-02-17", "url": url}, {}, "NEWS"
    )
    later = f.event_record(
        {"title": "【新华社】从后付制到即时结——医保结算提速", "published_date": "2025-03-24", "url": url},
        {},
        "NEWS",
        "未来转载补充内容",
    )
    events, excluded = f.deduplicate([later, early])
    assert len(events) == len(excluded) == 1
    assert events[0]["published_date"] == "2025-02-17" and events[0]["existing_text"] == ""
    home1 = f.event_record(
        {"title": "医药行业第一条消息", "published_date": "2021-06-04", "url": "https://www.hubpd.com/"}, {}, "NEWS"
    )
    home2 = f.event_record(
        {"title": "医药行业第二条消息", "published_date": "2021-06-10", "url": "https://www.hubpd.com/"}, {}, "NEWS"
    )
    assert len(f.deduplicate([home1, home2])[0]) == 2


def test_review002_specific_video_id_and_revision():
    url = "https://app.cntv.cn/special/cportal/columnv722/index.html?id=fe2f637e&fromapp=cctvnews"
    a = f.event_record(
        {"title": "【央视新闻】国家医保局 深化医保服务改革", "published_date": "2021-07-27", "url": url}, {}, "NEWS"
    )
    b = f.event_record(
        {"title": "【央视影音】国家医保局 深化医保服务改革", "published_date": "2021-07-27", "url": url}, {}, "NEWS"
    )
    assert len(f.deduplicate([a, b])[0]) == 1
    c = f.event_record(
        {"title": "国家医保局 深化医保服务改革修订公告", "published_date": "2021-07-28", "url": url}, {}, "NEWS"
    )
    assert len(f.deduplicate([a, c])[0]) == 2


def test_revision_shared_budget_and_hard_hold(tmp_path):
    revised = tmp_path / "revisions/r2"
    s.save(revised / "revision.json", {"experiment_root": str(tmp_path)})
    s.append(tmp_path / "fit-ledger.jsonl", {"action": "RESERVED", "bucket": "main", "attempt": 1})
    assert run.budget(revised)["consumed"] == 1
    assert run.budget(revised)["limit"] == 28
    with pytest.raises(ValueError, match="PENDING_EXPLICIT_BUDGET_AUTHORIZATION"):
        run.require_training_authorization(revised)
    with s.writer_lock(tmp_path), pytest.raises(OSError), s.writer_lock(revised):
        pass


def test_review002_full_text_alias_chain_keeps_first_date():
    body = "这一段为相同的现有正文证据。" * 20
    original = f.event_record(
        {
            "title": "原始政策通知",
            "published_date": "2020-04-15",
            "url": "https://www.nhsa.gov.cn/art/2020/4/15/art_104_6481.html",
        },
        {},
        "POLICY",
        body,
    )
    reprint = f.event_record(
        {
            "title": "转载政策通知",
            "published_date": "2020-04-17",
            "url": "https://www.nhsa.gov.cn/art/2020/4/17/art_14_3039.html",
        },
        {},
        "NEWS",
        body,
    )
    catalog = f.event_record(
        {
            "title": "转载政策通知",
            "published_date": "2020-04-17",
            "url": "https://www.nhsa.gov.cn/art/2020/4/17/art_14_3039.html",
        },
        {},
        "NEWS",
    )
    events, _ = f.deduplicate([original, reprint, catalog])
    assert len(events) == 1 and events[0]["published_date"] == "2020-04-15"


@pytest.mark.parametrize(
    "title",
    [
        "国家发展改革委办公厅等关于进一步提高高校学生医疗保障质量的通知",
        "河北省医疗保障局等10部门关于印发《河北省建立长期护理保险制度实施方案》的通知",
        "关于印发《北京市支持商业健康保险高质量发展的若干措施》的通知",
        "新疆生产建设兵团办公厅关于印发《兵团建立长期护理保险制度实施方案》的通知",
    ],
)
def test_review001_formal_government_titles_without_body(title):
    assert f.classify(title, "NEWS", url="https://www.nhsa.gov.cn/art/2026/7/17/art_14_1.html") == "POLICY"


def test_review001_catalog_cannot_override_same_day_body_semantics():
    raw = {
        "title": "国家医疗保障局等九部门印发关于药品采购的实施意见",
        "published_date": "2019-09-30",
        "url": "https://www.nhsa.gov.cn/art/2019/9/30/art_14_1815.html",
    }
    known = f.event_record(
        raw,
        {},
        "NEWS",
        "国家医疗保障局等9部门日前印发《实施意见》，意味着将全国推开。",
        {"kind": "OFFICIAL_POLICY_OR_NOTICE"},
    )
    catalog = f.event_record(raw, {}, "POLICY")
    events, _ = f.deduplicate([catalog, known])
    assert len(events) == 1 and events[0]["primary_type"] == "NEWS"
    assert events[0]["text_level"] == "TITLE_AND_EXISTING_TEXT"


def test_review001_later_timestamp_semantics_cannot_change_earlier_document():
    base = {
        "title": "医药行业文件",
        "published_date": "2023-01-09",
        "url": "https://www.nhsa.gov.cn/art/2023/1/9/art_14_1.html",
    }
    early = f.event_record({**base, "source_published_at": "2023-01-09T14:00:00+08:00"}, {}, "NEWS")
    later = f.event_record(
        {**base, "source_published_at": "2023-01-09T17:00:00+08:00"},
        {},
        "NEWS",
        "后来文字",
        {"kind": "OFFICIAL_POLICY_OR_NOTICE"},
    )
    events, _ = f.deduplicate([later, early])
    assert len(events) == 1 and events[0]["primary_type"] == "NEWS"
    assert events[0]["existing_text"] == "" and events[0]["published_at"] == "2023-01-09T14:00:00+08:00"


@pytest.mark.parametrize("row", [420, 1006, 1239])
def test_review001_frozen_original_number_overrides_old_news_semantics(row):
    frozen = s.Frozen(s.ROOT)
    raw = frozen.entry("public")["rows"][row]
    semantic = frozen.get(raw["semantic_source"]["path"])
    body = frozen.get(raw["source"]["path"]).get("text", "")
    assert semantic["kind"] == "PUBLIC_NEWS_OR_SERVICE_INFORMATION"
    event = f.event_record(raw, {}, "NEWS", body, semantic)
    assert event["primary_type"] == "POLICY"
    assert event["classification_evidence"]["standalone_document_numbers"]
    assert event["classification_evidence"]["rule"] == "ORIGINAL_DOCUMENT_NUMBER_OR_OPERATIVE_STRUCTURE"


@pytest.mark.parametrize("row", [41, 190, 283, 1030, 1069, 1228, 1353, 1449, 1453])
def test_review001_frozen_reporting_and_inline_numbers_remain_news(row):
    frozen = s.Frozen(s.ROOT)
    raw = frozen.entry("public")["rows"][row]
    semantic = frozen.get(raw["semantic_source"]["path"])
    body = frozen.get(raw["source"]["path"]).get("text", "")
    event = f.event_record(raw, {}, "NEWS", body, semantic)
    assert event["primary_type"] == "NEWS"
    assert not event["classification_evidence"]["standalone_document_numbers"]


@pytest.mark.parametrize(
    "title",
    [
        "国务院办公厅印发《关于加快建设分级诊疗体系的若干措施》的通知",
        "国家金融监督管理总局上海监管局、上海市医疗保障局等7部门印发《关于促进商业健康保险高质量发展助力生物医药产业创新的若干措施》的通知",
        "国家医疗保障局关于发布2024年版医保目录新增药品挂网情况和协议期内谈判药品配备情况的公告（截至2024年12月25日）",
    ],
)
def test_review001_issuance_notice_and_date_suffix_are_title_only(title):
    event = f.event_record(
        {"title": title, "published_date": "2025-01-01", "url": "https://www.nhsa.gov.cn/art/2025/1/1/art_14_1.html"},
        {},
        "NEWS",
    )
    assert event["primary_type"] == "POLICY"
    assert event["text_level"] == "TITLE_ONLY"
    assert event["classification_evidence"]["rule"] == "EXPLICIT_GOVERNMENT_DOCUMENT_TITLE_ONLY"


def test_review002_matched_keys_are_prior_evidence_not_new_aliases():
    body = "相同的已有完整原文内容" * 20
    first = f.event_record(
        {
            "title": "较早文件",
            "published_date": "2020-01-02",
            "url": "https://www.nhsa.gov.cn/art/2020/1/2/art_14_1.html",
        },
        {},
        "NEWS",
        body,
    )
    second = f.event_record(
        {
            "title": "新转载别名",
            "published_date": "2020-01-03",
            "url": "https://www.nhsa.gov.cn/art/2020/1/3/art_14_2.html",
        },
        {},
        "NEWS",
        body,
    )
    _, decisions = f.deduplicate([first, second])
    assert len(decisions) == 1
    assert [key[0] for key in decisions[0]["matched_keys"]] == ["full_text"]


def test_review001_later_body_cannot_supply_original_classification():
    event = f.event_record(
        {
            "title": "医药行业最新情况",
            "published_date": "2023-01-09",
            "url": "https://www.nhsa.gov.cn/art/2023/1/9/art_14_1.html",
        },
        {},
        "NEWS",
        "医保办函〔2023〕1号\n各省医疗保障局：\n现通知如下",
        {"kind": "OFFICIAL_POLICY_OR_NOTICE", "body_version_updated_at": "2023-01-11T16:00:00+08:00"},
    )
    assert event["primary_type"] == "NEWS" and event["text_level"] == "TITLE_ONLY"
    assert event["classification_evidence"]["later_version_evidence_ignored"]
    assert event["classification_evidence"]["standalone_document_numbers"] == []


@pytest.mark.parametrize("row", [168, 825])
def test_review001_original_notice_may_cite_another_document(row):
    frozen = s.Frozen(s.ROOT)
    raw = frozen.entry("public")["rows"][row]
    semantic = frozen.get(raw["semantic_source"]["path"])
    body = frozen.get(raw["source"]["path"]).get("text", "")
    event = f.event_record(raw, {}, "NEWS", body, semantic)
    assert event["primary_type"] == "POLICY"
    assert event["classification_evidence"]["issuing_authority_and_document_date"]


@pytest.mark.parametrize("row", [55, 1297, 1314, 1436])
def test_review001_media_reporting_consultation_is_news(row):
    frozen = s.Frozen(s.ROOT)
    raw = frozen.entry("public")["rows"][row]
    semantic = frozen.get(raw["semantic_source"]["path"])
    body = frozen.get(raw["source"]["path"]).get("text", "")
    event = f.event_record(raw, {}, "NEWS", body, semantic)
    assert event["primary_type"] == "NEWS"


def test_review003_restarted_inputs_require_same_generator(tmp_path):
    s.save(tmp_path / "input-manifest.json", {"generator_code_sha256": "old-source"})
    with pytest.raises(ValueError, match="INPUT_GENERATOR_CODE_CHANGED_NEW_IMMUTABLE_REVISION_REQUIRED"):
        f.build_inputs(tmp_path)


def test_review003_completed_inputs_verify_artifacts(tmp_path):
    artifact = tmp_path / "events.jsonl"
    artifact.write_text("new bytes", "utf-8")
    s.save(
        tmp_path / "input-manifest.json",
        {"generator_code_sha256": s.digest(f.GENERATOR_CODE), "artifacts": {"events.jsonl": "old-hash"}},
    )
    with pytest.raises(ValueError, match="INPUT_ARTIFACT_CHANGED"):
        f.build_inputs(tmp_path)


@pytest.mark.parametrize("row,hour,minute", [(38, 17, 4), (62, 15, 28), (71, 7, 16), (164, 12, 55), (1247, 19, 1)])
def test_review004_frozen_publisher_timestamp(row, hour, minute):
    frozen = s.Frozen(s.ROOT)
    raw = frozen.entry("public")["rows"][row]
    semantic = frozen.get(raw["semantic_source"]["path"])
    document = frozen.get(raw["source"]["path"])
    point, evidence = f.publisher_timestamp(raw, document, semantic)
    parsed = f.datetime.fromisoformat(point)
    assert (parsed.hour, parsed.minute) == (hour, minute)
    assert str(parsed.date()) == raw["published_date"]
    assert evidence["date_matches_verified_publication_day"]


def test_review004_timestamp_catalog_cannot_bypass_1500():
    raw = {
        "title": "医保行业消息",
        "published_date": "2023-01-09",
        "url": "https://tv.cctv.com/2023/01/09/VIDE123.shtml",
    }
    dated = f.event_record(raw, {}, "NEWS")
    timed = f.event_record({**raw, "source_published_at": "2023-01-09T15:28:00+08:00"}, {}, "NEWS")
    events, _ = f.deduplicate([dated, timed])
    assert len(events) == 1
    assert events[0]["published_at"] == "2023-01-09T15:28:00+08:00"
    for mode in f.TIMINGS:
        assert f.effective_session(events[0]["published_date"], events[0]["published_at"], DAYS, mode) == "2023-01-10"


def test_review004_body_event_time_and_later_revision_are_not_publication():
    raw = {"published_date": "2023-01-09", "url": "https://tv.cctv.com/2023/01/09/VIDE123.shtml"}
    doc = {"url": raw["url"], "title_identity_verified": True, "text": "事件于2023-01-09 14:00发生"}
    assert f.publisher_timestamp(raw, doc, {}) == (None, None)
    doc["text"] = "来源：央视网 2023-01-09 15:28"
    assert f.publisher_timestamp(raw, doc, {"body_version_updated_at": "2023-01-11 17:00"}) == (None, None)
    with pytest.raises(ValueError, match="TIME_DATE_CONFLICT"):
        f.publisher_timestamp({**raw, "published_date": "2023-01-08"}, doc, {})


def test_review005_unverified_page_body_and_derived_semantics_are_not_features():
    raw = {
        "title": "医保行业消息",
        "published_date": "2023-01-09",
        "url": "https://tv.cctv.com/2023/01/09/VIDE123.shtml",
        "existing_text_exclusion": "SOURCE_BODY_OR_IDENTITY_UNVERIFIED",
    }
    event = f.event_record(
        raw, {}, "NEWS", "登录 导航 2026推荐内容 医保办函〔2023〕1号", {"kind": "OFFICIAL_POLICY_OR_NOTICE"}
    )
    assert event["primary_type"] == "NEWS"
    assert event["text_level"] == "TITLE_ONLY" and event["existing_text"] == ""
    assert event["full_existing_text_sha256"] is None
    assert event["classification_evidence"]["unverified_body_evidence_ignored"]


def test_review005_verified_publisher_clock_survives_title_only_body():
    frozen = s.Frozen(s.ROOT)
    raw = frozen.entry("public")["rows"][38]
    semantic = frozen.get(raw["semantic_source"]["path"])
    doc = frozen.get(raw["source"]["path"])
    assert doc["body_verified"] is False and doc["title_identity_verified"] is True
    point, evidence = f.publisher_timestamp(raw, doc, semantic)
    event = f.event_record(
        {
            **raw,
            "source_published_at": point,
            "publication_time_evidence": evidence,
            "existing_text_exclusion": "SOURCE_BODY_OR_IDENTITY_UNVERIFIED",
        },
        {},
        "NEWS",
        doc["text"],
        semantic,
    )
    assert event["published_at"] == "2019-12-23T17:04:00+08:00"
    assert event["text_level"] == "TITLE_ONLY"
