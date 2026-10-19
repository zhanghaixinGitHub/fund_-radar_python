"""转换边界和资料集隔离的技术算例；全部测试数据不得作为真实校准产品。"""

import calendar
import hashlib
import json
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from app.services.fund_quality_v2_contract import contract_hash, metric_contract
from app.services.fund_quality_v2_materials import (
    METRIC_ROLES,
    benchmark_expression,
    build_cohort,
    clear_verified_role_gaps,
    historical_supplements,
    inspect_sources,
    manager_supplements,
    metric_gaps,
    relative_history,
    verify_packet,
)
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_quality_v2 import assess, candidate_policy, metric_names


def history():
    """连续自然日曲线只用于独立公式算例；不是任何真实基金的估值日历。"""
    ends = [
        date(y, m, calendar.monthrange(y, m)[1])
        for y in range(2023, 2027)
        for m in range(1, 13)
        if date(2023, 9, 1) <= date(y, m, 1) <= date(2026, 9, 1)
    ]
    days = [ends[0] + timedelta(days=i) for i in range((ends[-1] - ends[0]).days + 1)]
    return {
        "dates": list(map(str, days)),
        "month_ends": list(map(str, ends)),
        "adjusted_nav": [str(Decimal(1) + Decimal(i) / 10000) for i in range(len(days))],
        "benchmark_index": [str(Decimal(1) + Decimal(i) / 10000) for i in range(len(days))],
    }


def test_role_requirements_are_metric_specific_and_do_not_fabricate_missing_capacity_for_holder():
    assert set(METRIC_ROLES) == set(metric_contract()["metrics"])
    source = {"label": "half", "sha256": "TEST_ONLY", "state": "ORIGINAL_AVAILABLE_PARSE_REQUIRED"}
    items = metric_gaps("ACTIVE_EQUITY", [source])
    assert items["holder_pressure_band"]["required_roles"] == ["report"]
    assert items["holder_pressure_band"]["missing_roles"] == []
    assert items["capacity_pressure"]["missing_roles"] == ["capacity_review"]
    packet = {"metrics": {"holder_pressure_band": {"value": "0"}}, "inventory": items}
    clear_verified_role_gaps(packet)
    assert "missing_roles" not in packet["inventory"]["holder_pressure_band"]
    assert packet["inventory"]["capacity_pressure"]["missing_roles"] == ["capacity_review"]


@pytest.mark.parametrize("label", ["half", "summary", "prospectus", "nav-database-snapshot", "manager-api"])
def test_historical_attachment_cannot_replace_current_document_roles(label):
    with pytest.raises(ValueError, match="SUPPLEMENT_ROLE"):
        historical_supplements([{"label": label, "published_on": "2026-08-31"}], date(2026, 9, 30))


def test_historical_attachment_dates_and_defensive_copy():
    ref = {"label": "historical-annual", "published_on": "2026-03-31", "report_end": "2025-12-31"}
    result = historical_supplements([ref], date(2026, 9, 30))
    result[0]["published_on"] = "CHANGED_TEST_ONLY"
    assert ref["published_on"] == "2026-03-31"
    ref["published_on"] = "2026-10-01"
    with pytest.raises(ValueError, match="SUPPLEMENT_PUBLICATION"):
        historical_supplements([ref], date(2026, 9, 30))


def manager_attachment(tmp_path):
    """仅供边界反例的虚构响应，不进入任何真实产品、来源或校准目录。"""
    body = {"code": 0, "data": {"fields": ["ts_code", "ann_date", "name", "begin_date", "end_date"],
                                "items": [["123456.OF", "20200101", "测试经理", "20200101", None]]}}
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "sources" / (sha + ".json")
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(raw)
    ref = {"file": "sources/" + path.name, "sha256": sha, "label": "manager-api", "ts_code": "123456.OF",
           "api_name": "fund_manager", "source_code": "TUSHARE_PRO_FUND", "url": "https://api.tushare.pro",
           "acquired_at": "2026-10-16T12:00:00+00:00"}
    return ref, body, path


def test_manager_attachment_is_separate_from_historical_and_preserves_original_reference(tmp_path):
    ref, _, _ = manager_attachment(tmp_path)
    refs, checked = manager_supplements([ref], "123456", tmp_path, datetime.fromisoformat("2026-10-17T00:00:00+00:00"))
    assert checked[0]["state"] == "ORIGINAL_AVAILABLE_PARSE_REQUIRED"
    refs[0]["label"] = "CHANGED_TEST_ONLY"
    assert ref["label"] == "manager-api"
    with pytest.raises(ValueError, match="HISTORICAL_SUPPLEMENT_ROLE"):
        historical_supplements([ref], date(2026, 9, 30))


@pytest.mark.parametrize("refs", [{"label": "manager-api"}, ["not-a-reference"]])
def test_manager_attachment_schema_cannot_be_replaced_by_flags_or_strings(tmp_path, refs):
    with pytest.raises(ValueError, match="MANAGER_SUPPLEMENT_SCHEMA"):
        manager_supplements(refs, "123456", tmp_path, datetime.fromisoformat("2026-10-17T00:00:00+00:00"))


def test_product_manager_attachment_reads_its_own_original_identity_but_does_not_assert_product_relationship(tmp_path):
    ref, body, path = manager_attachment(tmp_path)
    body["data"]["items"][0][0] = "654321.OF"
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    path.write_bytes(raw)
    ref.update(
        label="manager-master-api", ts_code="654321.OF", master_code="654321", sha256=hashlib.sha256(raw).hexdigest()
    )
    refs, _ = manager_supplements([ref], "123456", tmp_path, datetime.fromisoformat("2026-10-17T00:00:00+00:00"))
    assert refs[0]["master_code"] == "654321"
    # 产品归属仍由披露转换器重读正式报告主代码；此入口不产生relationship_verified。
    assert "relationship_verified" not in refs[0]


@pytest.mark.parametrize(
    "problem",
    [
        "role", "host", "port", "api", "source", "request_identity",
        "response_identity", "business_error", "hash", "future",
    ],
)
def test_manager_attachment_rechecks_original_bytes_identity_and_time_despite_success_flag(tmp_path, problem):
    ref, body, path = manager_attachment(tmp_path)
    ref["responsibility_verified"] = True
    if problem == "role":
        ref["label"] = "historical-annual"
    elif problem == "host":
        ref["url"] = "https://example.org/manager"
    elif problem == "port":
        ref["url"] = "https://api.tushare.pro:4444"
    elif problem == "api":
        ref["api_name"] = "fund_nav"
    elif problem == "source":
        ref["source_code"] = "OTHER_SOURCE"
    elif problem == "request_identity":
        ref["ts_code"] = "654321.OF"
    elif problem in {"response_identity", "business_error"}:
        if problem == "response_identity":
            body["data"]["items"][0][0] = "654321.OF"
        else:
            body["code"] = -1
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        path.write_bytes(raw)
        ref["sha256"] = hashlib.sha256(raw).hexdigest()
    elif problem == "hash":
        path.write_bytes(b"{}")
    else:
        ref["acquired_at"] = "2026-10-18T00:00:00+00:00"
    with pytest.raises(ValueError):
        manager_supplements([ref], "123456", tmp_path, datetime.fromisoformat("2026-10-17T00:00:00+00:00"))


def test_identical_curves_have_no_extra_risk_and_ties_are_half_success():
    values = relative_history(history())
    for key in ("relative_return_1y", "relative_return_3y", "extra_drawdown", "extra_downside", "tracking_error"):
        assert values[key] == 0
    assert values["rolling_target_success"] == values["monthly_target_success"] == Decimal("0.5")


def test_benchmark_field_does_not_read_tracking_narrative_as_contract():
    text = (
        "与业绩比较基准之间的日均跟踪偏离度绝对值控制在0.35%以内。"
        "业绩比较基准中证银行指数收益率×95%+银行人民币活期存款利率(税后)×5%风险收益特征股票型基金"
    )
    assert benchmark_expression(text) == "中证银行指数收益率×95%+银行人民币活期存款利率(税后)×5%"
    assert benchmark_expression("业绩比较基准之间的年跟踪误差控制在4%以内。") is None


def test_benchmark_rejects_performance_table_and_cuts_off_following_risk_text():
    table = "业绩比较基准收益阶段率率标准差基准收益①－③②－④率标准差①②率③④2025/10/10-2025/12/31-1.56%0.74%-2.49%"
    assert benchmark_expression(table) is None
    formula = "沪深300指数收益率×95%+同期银行活期存款利率（税后）×5%"
    assert benchmark_expression("业绩比较基准为" + formula + "本基金属于股票型基金，理论风险较高") == formula
    reverse = "95%×中证银行指数收益率＋5%×银行人民币活期存款利率（税后）"
    assert benchmark_expression("业绩比较基准：" + reverse + "。风险收益特征") == reverse
    assert (
        benchmark_expression("业绩比较基准：60%WTI原油价格收益率+40%BRENT原油价格收益率。")
        == "60%WTI原油价格收益率+40%BRENT原油价格收益率"
    )
    assert benchmark_expression("业绩比较基准：" + formula + "。业绩比较基准：中证500指数收益率×100%。") is None


@pytest.mark.parametrize("change", ["duplicate", "zero", "missing_month", "curve_length", "false_month_end"])
def test_history_errors_do_not_generate_partial_score(change):
    value = history()
    if change == "duplicate":
        value["dates"][1] = value["dates"][0]
    elif change == "zero":
        value["adjusted_nav"][5] = "0"
    elif change == "missing_month":
        value["month_ends"].pop(5)
    elif change == "false_month_end":
        value["month_ends"][5] = "2024-02-28"
    else:
        value["benchmark_index"].pop()
    with pytest.raises(ValueError):
        relative_history(value)


def test_losing_fund_is_not_rewarded_by_absolute_market_increase():
    h = history()
    h["adjusted_nav"] = ["1"] * len(h["dates"])
    result = relative_history(h)
    assert result["relative_return_1y"] < 0 and result["extra_drawdown"] > 0
    assert result["extra_downside"] > 0 and result["monthly_target_success"] == 0


def test_sources_are_really_read_and_traversal_and_future_receipts_rejected(tmp_path):
    path = tmp_path / "sources" / "original.json"
    path.parent.mkdir()
    path.write_bytes(b'{"raw":1}')
    ref = {
        "file": "sources/original.json",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "acquired_at": "2026-10-01T00:00:00+08:00",
    }
    frozen = datetime.fromisoformat("2026-10-16T00:00:00+08:00")
    assert inspect_sources([ref], tmp_path, frozen)[0]["state"] == "ORIGINAL_AVAILABLE_PARSE_REQUIRED"
    path.write_bytes(b'{"raw":2}')
    assert inspect_sources([ref], tmp_path, frozen)[0]["reason"] == "SOURCE_HASH"
    assert inspect_sources([{**ref, "file": "../other.json"}], tmp_path, frozen)[0]["reason"] == "SOURCE_PATH"
    path.write_bytes(b'{"raw":1}')
    assert (
        inspect_sources([{**ref, "acquired_at": "2026-10-17T00:00:00+08:00"}], tmp_path, frozen)[0]["reason"]
        == "SOURCE_TIME"
    )


def test_boolean_verification_and_self_computed_hash_are_not_originals(tmp_path):
    packet = {
        "kind": "ACTIVE_EQUITY",
        "as_of_date": "2026-09-30",
        "frozen_at": "2026-10-16T00:00:00+08:00",
        "history_months": 36,
        "benchmark_verified": True,
        "classification_verified": True,
        "sources": {},
        "metrics": {},
    }
    packet["hash"] = digest(packet)
    result = assess(packet, None, root=tmp_path)
    assert result["score"] is result["grade"] is None
    assert "ORIGINALS_REBUILD_REQUIRED" in result["reasons"]
    with pytest.raises(ValueError, match="HASH_OR_COMPILER"):
        verify_packet(packet, tmp_path)


def test_contract_covers_unique_metric_ownership_and_freezes_policy():
    definitions = metric_contract()["metrics"]
    for kind in ("ACTIVE_EQUITY", "PASSIVE_INDEX", "INDEX_ENHANCED", "GLOBAL_ACTIVE_EQUITY"):
        config = candidate_policy(kind)
        assert metric_names(config) <= definitions.keys()
        assert config["raw_contract_hash"] == contract_hash()
        for dimension, terms in config["terms"].items():
            assert all(definitions[name]["dimension"] == dimension for name, _ in terms)


def test_product_partition_controls_all_share_classes_and_months(monkeypatch, tmp_path):
    # 此处仅隔离测试分组算法；原件门禁另有拒绝伪包测试，不把此替身当原件证据。
    monkeypatch.setattr("app.services.fund_quality_v2_materials.verify_packet", lambda *args: None)
    base = {"product_id": "TEST_PRODUCT", "kind": "ACTIVE_EQUITY", "raw_metrics_complete": True, "hash": "TEST_ONLY"}
    packets = [
        {**base, "fund_code": code, "as_of_date": month}
        for code in ("000001", "000002")
        for month in ("2026-08-31", "2026-09-30")
    ]
    result = build_cohort(packets, {"TEST_PRODUCT": "calibration"}, tmp_path)
    assert len(result["members"]) == 2
    assert {r["fund_code"] for r in result["members"]} == {"000001"}
    assert {r["partition"] for r in result["members"]} == {"calibration"}
    with pytest.raises(ValueError, match="UNKNOWN_PRODUCT"):
        build_cohort(packets, {"FAKE": "validation"}, tmp_path)


def test_incomplete_eight_dimensions_cannot_enter_a_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.fund_quality_v2_materials.verify_packet", lambda *args: None)
    packet = {
        "product_id": "TEST_ONLY",
        "fund_code": "000001",
        "as_of_date": "2026-09-30",
        "raw_metrics_complete": False,
    }
    result = build_cohort([packet], {"TEST_ONLY": "validation"}, tmp_path)
    assert result["members"] == [] and result["gaps"][0]["reason"] == "RAW_METRICS_INCOMPLETE"
