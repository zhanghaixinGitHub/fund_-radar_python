"""综合规则与产品入口回归；隔离来源，不访问外部供应商或修改历史预测。"""

from copy import deepcopy
from datetime import datetime

import pytest
from app.services import comprehensive_prediction_policy as policy
from app.services.direction_1d_protocol import digest, window
from app.services.fund_exposure_common import read, save
from app.services.prediction_contract import PredictionFailure
from tests.test_prediction_database import database  # noqa: F401


def sample():
    facts = {"F1": {"category": "持仓行情", "fact_id": "holdings", "text": "已发生行情偏弱。"}}
    data = {
        "analysis_policy_version": policy.policy()["version"],
        "report": {"sourceUrl": "https://example.com/report"},
        "documents": [{"id": "notice", "kind": "DISCLOSURE_SEARCH"}],
        "comprehensive_coverage": {
            "EXPOSURE": {"status": "AVAILABLE", "source_refs": ["https://example.com/report"]},
            "MARKET": {"status": "AVAILABLE", "source_refs": ["holdings"]},
            "DISCLOSURES": {"status": "VERIFIED_EMPTY", "source_refs": ["notice"]},
        },
    }
    analysis = {
        "direction": "DOWN",
        "confidence": "LOW",
        "summary": "持仓偏弱，倾向下跌。",
        "reasons": [
            {
                "title": "持仓行情",
                "refs": ["F1"],
                "role": "支持下跌",
                "meaning": "已发生行情偏弱。",
                "implication": "若延续，可能拖累。",
            }
        ],
        "counterpoints": [],
        "synthesis": "已有行情偏弱，延续仍是假设。",
        "change_conditions": ["持仓转强可能改变判断。"],
        "limitations": ["尚未验证效果。"],
    }
    review = {
        "analysis_policy_version": policy.policy()["version"],
        "counter_evidence_checked": True,
        "analysis": analysis,
    }
    return data, review, facts


def test_policy_rejects_nav_only_even_when_all_document_checks_pass():
    data, review, facts = sample()
    policy.validate_review(data, review, facts)
    facts["F1"]["category"] = "基金净值"
    with pytest.raises(ValueError, match="NAV_ONLY_PREDICTION_DISABLED"):
        policy.validate_review(data, review, facts)


@pytest.mark.parametrize(
    "change,error",
    [
        ("coverage", "COMPREHENSIVE_EVIDENCE_REQUIRED"),
        ("counter", "COMPREHENSIVE_COUNTER_REVIEW_REQUIRED"),
        ("conditions", "COMPREHENSIVE_CHANGE_CONDITIONS_REQUIRED"),
        ("identity", "COMPREHENSIVE_POLICY_REQUIRED"),
    ],
)
def test_missing_dimensions_and_review_steps_fail_closed(change, error):
    data, review, facts = sample()
    if change == "coverage":
        data["comprehensive_coverage"]["DISCLOSURES"] = {"status": "MISSING", "source_refs": []}
    elif change == "counter":
        review["counter_evidence_checked"] = False
    elif change == "conditions":
        review["analysis"]["change_conditions"] = []
    else:
        review["analysis_policy_version"] = "unknown"
    with pytest.raises(ValueError, match=error):
        policy.validate_review(data, review, facts)


@pytest.mark.parametrize("code", ["002112", "016008", "006730", "000001"])
def test_all_funds_submit_the_analysis_protocol(code, monkeypatch):
    from app.services import direction_1d_jobs as jobs
    from app.services.fund_information_snapshot import enabled

    assert enabled(code)
    monkeypatch.setattr(jobs.repo, "clock", lambda: datetime.fromisoformat("2026-10-14T10:00:00+08:00"))
    monkeypatch.setattr(jobs, "submit", lambda key, kind, payload: payload)
    assert jobs.submit_forecast(code)["protocol"] == "DIRECTION_1D_ANALYSIS_V1"


def test_automatic_legacy_jobs_stop_before_database_or_training(monkeypatch):
    from app.services import auto_model_selection as auto
    from app.services import direction_experiment as old

    def forbidden():
        pytest.fail("旧净值入口不能接触训练或数据库")

    monkeypatch.setattr(auto, "get_engine", forbidden)
    monkeypatch.setattr(old, "load_experiment_models", forbidden)
    assert auto.check_auto(["016008"])["status"] == "DISABLED_COMPREHENSIVE_POLICY"
    assert auto.recover_auto() == 0
    assert auto.claim_replay() is None
    assert auto.run_auto("old-task") is None
    assert old.get_direction_experiment("008888").reason_codes == ("NAV_ONLY_PREDICTION_DISABLED",)


@pytest.mark.parametrize("horizon", ["T5_V1", "T20_V1", "M6_V1"])
def test_live_multi_cannot_reach_legacy_feature_or_inference(horizon):
    from app.services.prediction_generation import prediction_payload

    with pytest.raises(PredictionFailure) as error:
        prediction_payload(None, None, {"horizon_id": horizon}, None, None)
    assert error.value.payload["code"] == "NAV_ONLY_PREDICTION_DISABLED"


@pytest.mark.parametrize(
    "problem", [None, "future_publication", "late_observation", "same_day_unknown_time", "changed_quote"]
)
@pytest.mark.parametrize("code", ["002112", "016008", "006730", "000001"])
def test_general_material_source_cutoff_and_literal_quotes(problem, code, monkeypatch, tmp_path):
    """资料必须在截止前实际取得；同日缺公开时刻或改写引文都不能进入判断。"""
    from contextlib import nullcontext
    from types import SimpleNamespace

    from app.services import fund_comprehensive_materials as material_service
    from app.services.fund_information_snapshot import prepare

    now = datetime.fromisoformat("2026-10-14T09:35:00+08:00")
    document = {
        "id": "doc",
        "kind": "ANNOUNCEMENT",
        "title": "测试公开资料",
        "url": "https://example.com/doc",
        "published_date": "2026-10-14",
        "published_at": "2026-10-14T09:00:00+08:00",
        "available_at": "2026-10-14T09:01:00+08:00",
        "body": "已核实的原文。",
        "body_scope": "FULL_TEXT",
    }
    document["body_hash"] = digest(document["body"])
    if problem == "future_publication":
        document["published_at"] = "2026-10-14T10:00:00+08:00"
    elif problem == "late_observation":
        document["available_at"] = "2026-10-14T10:00:00+08:00"
    elif problem == "same_day_unknown_time":
        document.pop("published_at")
    material = {
        "fund_code": code,
        "report": {"endDate": "2026-06-30", "publishedDate": "2026-08-31"},
        "documents": [document],
        "facts": {
            "market": {
                "category": "MARKET",
                "source_refs": ["doc"],
                "quotes": ["改写的原文。" if problem == "changed_quote" else document["body"]],
            }
        },
        "exposure_chain": [code, "159779", "931494"],
        "companies": [],
        "market": {},
        "source_manifest": {},
        "inventory": [{} for _ in range(16)],
        "comprehensive_coverage": {},
    }
    path = tmp_path / "materials.json"
    save(path, material)
    monkeypatch.setattr(material_service, "get_engine", lambda: SimpleNamespace(begin=lambda: nullcontext(None)))
    monkeypatch.setattr(material_service.repo, "source", lambda c: {"source_id": "test", "retention_days": 365})
    monkeypatch.setattr(
        material_service.repo,
        "profiles",
        lambda c, codes: [
            {"status": "ACTIVE", "fund_type": "STOCK", "fund_name": "测试ETF联接基金", "fund_master_id": "test"}
        ],
    )
    monkeypatch.setattr(material_service.repo, "navs", lambda *args: [])
    if problem:
        with pytest.raises(ValueError, match="COMPREHENSIVE_(SOURCE_INVALID|FACT_QUOTE_INVALID)"):
            prepare(now, code, path)
    else:
        result = prepare(now, code, path)
        assert result["fund_code"] == code
        assert result["facts"]["market"]["text"] == document["body"]
        assert result["facts"]["market"]["category"] == "市场行情"
        assert result["documents"][0]["date"] == "2026-10-14"
        assert result["latest_nav_date"] is None


@pytest.mark.parametrize("code", ["016008", "006730", "000001"])
def test_missing_own_materials_never_borrows_pilot_sources(code, monkeypatch, tmp_path):
    """未准备资料是资料缺口，不是基金白名单拒绝，也不能借用试点材料凑结果。"""
    from types import SimpleNamespace

    from app.services import fund_information_snapshot as snapshots

    monkeypatch.setattr(snapshots, "get_settings", lambda: SimpleNamespace(fund_material_directory=tmp_path))
    monkeypatch.setattr(snapshots, "_prepare_002112", lambda _: pytest.fail("其他基金不能读取试点来源"))
    with pytest.raises(ValueError, match="COMPREHENSIVE_MATERIALS_REQUIRED"):
        snapshots.prepare(datetime.fromisoformat("2026-10-14T09:35:00+08:00"), code)


@pytest.mark.parametrize("code", ["002112", "016008"])
@pytest.mark.parametrize("horizon", ["T1", "T5_V1", "T20_V1", "M6_V1"])
def test_every_horizon_preserves_requested_fund_and_materials_path(code, horizon, monkeypatch, tmp_path):
    """原试点与其他基金均须使用调用方指定的自身资料，不丢失资料路径或换基金。"""
    from app.services import comprehensive_multi_prediction as multi
    from app.services import fund_codex_analysis as daily
    from app.services import prediction_features as features

    now = datetime.fromisoformat("2026-10-14T09:35:00+08:00")
    path = tmp_path / (code + ".json")
    own_data = {"fund_code": code, "window": window(now), "documents": [], "materials_path": str(path)}

    def prepare(cutoff, actual_code, materials_path):
        assert cutoff == now and actual_code == code and materials_path == path
        return deepcopy(own_data)

    monkeypatch.setattr(daily.repo, "clock", lambda: now)
    monkeypatch.setattr(daily, "prepare", prepare)
    monkeypatch.setattr(multi, "prepare", prepare)
    calendar = features.public_calendar({"fund_code": code, "fund_name": "测试混合基金", "fund_type": "MIXED"})
    monkeypatch.setattr(features, "read_fund_data", lambda *args: {"calendar": calendar})
    directory = tmp_path / "frozen"
    if horizon == "T1":
        daily.freeze_packet(directory, code, path)
    else:
        multi.freeze_packet(directory, code, horizon, path)
    packet = read(directory / "input.json")
    stored = packet["input"] if horizon == "T1" else packet["input"]["data"]
    assert stored == own_data and packet["input_hash"] == digest(packet["input"])


@pytest.mark.parametrize("horizon", ["T5_V1", "T20_V1", "M6_V1"])
def test_multi_import_immutable_readback_and_product_task(database, monkeypatch, tmp_path, horizon):  # noqa: F811
    """真实隔离库验证三周期独立导入、原时间复用、任务消费及缺口拒绝。"""
    from app.services import comprehensive_multi_prediction as multi
    from app.services import prediction_features as features
    from app.services import prediction_generation as generation
    from app.services.fund_codex_analysis import SOURCE, catalog

    monkeypatch.setattr(multi, "get_engine", lambda: database)
    monkeypatch.setattr(generation, "get_engine", lambda: database)
    # 本用例手动执行持久化任务，避免后台线程与断言竞争同一会话锁。
    monkeypatch.setattr(generation, "_dispatch_task", lambda task_id: True)
    with database.connect() as c:
        from sqlalchemy import text

        now = c.execute(text("SELECT clock_timestamp()")).scalar_one()
    monkeypatch.setattr(multi.repo, "clock", lambda: now)
    calendar = features.public_calendar({"fund_code": "016008", "fund_name": "测试混合基金", "fund_type": "MIXED"})
    monkeypatch.setattr(features, "read_fund_data", lambda *args: {"calendar": calendar})
    data, review, _ = sample()
    data.update(
        fund_code="016008",
        fund_name="测试混合基金",
        as_of=now.isoformat(),
        window=window(now),
        latest_nav_date=None,
        nav=[],
        companies=[],
        market={},
        facts={"holdings": {"category": "持仓行情", "text": "已发生行情偏弱。", "source": None}},
        inventory=[
            {"id": f"D{i:02d}", "name": "资料", "status": "AVAILABLE", "detail": "测试来源"} for i in range(1, 17)
        ],
    )
    data["documents"] = [
        {"id": "notice", "kind": "DISCLOSURE_SEARCH", "body": "测试检索回执，没有新增公告。", "body_scope": "FULL_TEXT"}
    ]
    monkeypatch.setattr(multi, "prepare", lambda *args: deepcopy(data))
    directory = tmp_path / horizon
    multi.freeze_packet(directory, "016008", horizon)
    packet_path = directory / "input.json"
    packet = read(packet_path)
    review.update(
        input_hash=packet["input_hash"],
        generation_source=SOURCE,
        horizon_id=horizon,
        documents=[
            {
                "id": "notice",
                "scope": "FULL_TEXT",
                "ranges": [{"start": 0, "end": len(data["documents"][0]["body"])}],
                "events": [],
            }
        ],
        audit={"valid": True, "issues": []},
    )
    ref = next(iter(catalog(data, review)[0]))
    review["analysis"]["reasons"][0]["refs"] = [ref]
    review_path = directory / "review.json"
    save(review_path, review)
    first = multi.import_review(packet_path, review_path)
    original = multi.ready_result("016008", horizon, now)
    second = multi.import_review(packet_path, review_path)
    assert first["prediction_id"] == second["prediction_id"] and second["reused"]
    assert original == multi.ready_result("016008", horizon, now)
    assert original["generationMethod"] == "COMPREHENSIVE" and original["modelManifest"]["features"] == []
    assert original["analysis"]["direction"] == "DOWN" and original["upProbability"] is None
    task = generation.create_task(["016008", "006730"], horizons=[horizon])
    generation.run_task(task["taskId"])
    finished = generation.task_status(task["taskId"])
    assert finished["status"] == "PARTIAL_SUCCESS"
    assert finished["reusedItems"] == 1 and finished["failedItems"] == 1
    assert generation.current_predictions("016008")["predictions"][0]["predictionId"] == first["prediction_id"]
    assert digest(original) == digest(multi.ready_result("016008", horizon, now))
