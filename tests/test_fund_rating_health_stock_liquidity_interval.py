"""010710区间接线隔离验证；此处测试数据不冒充真实基金财务原件。"""

import json
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from app.commands import fund_ratings
from app.services import fund_rating_health_stock_liquidity_interval as interval
from app.services import fund_rating_pipeline as pipeline
from app.services import fund_rating_reviewed_snapshot as snapshots

LOWER = "43.2375"
UPPER = "83.2375"


def raw_inputs():
    """只验证边界和接线，不用这些空来源生成真实评分。"""
    return {
        "version": interval.VERSION,
        "fund_code": interval.FUND,
        "share_class": "C",
        "currency": "CNY",
        "as_of_date": interval.ASOF,
        "holding_date": interval.HOLDING,
        "legal_name": interval.LEGAL_NAME,
        "manager_name": interval.MANAGER_NAME,
        "primary_ref": interval.PRIMARY_HASH,
        "sources": {},
    }


def bundle(*, all_other_points=False):
    dimensions = {key: None for key in snapshots.KEYS}
    dimensions.update(management="55", fee="66")
    if all_other_points:
        dimensions = {key: "50" for key in snapshots.KEYS}
    dimensions["liquidity"] = {"lower": LOWER, "upper": UPPER, "candidate": None, "reason": "隔离测试区间"}
    raw = raw_inputs()
    return {
        "basis": snapshots.BASIS,
        "fund_code": interval.FUND,
        "family": "ACTIVE_EQUITY",
        "share_class": "C",
        "as_of_date": interval.ASOF,
        "holding_date": interval.HOLDING,
        "verified_at": "2026-10-29T14:00:00+00:00",
        "valid_until": "2026-11-20T00:00:00+00:00",
        "identity": {"legal_name": interval.LEGAL_NAME, "manager_name": interval.MANAGER_NAME},
        "identity_archive": {"file": "identity.json", "sha256": "identity"},
        "original_report_archive": {"file": "report.json", "sha256": interval.PRIMARY_HASH},
        "research_archive": {"file": "research.json", "sha256": "research"},
        "research_format": "single",
        "reviewed_document": {
            "code": interval.FUND,
            "as_of": interval.ASOF,
            "outer_weights_percent": {key: "12.5" for key in snapshots.KEYS},
            "dimensions": dimensions,
        },
        "health_stock_liquidity_interval_calculation": {
            "inputs": raw,
            "input_archive": {"file": "inputs.json", "sha256": "inputs"},
        },
    }


def stub_original_replay(monkeypatch, tmp_path):
    calls = []

    def replay(raw, *, root):
        assert root == tmp_path
        assert raw["primary_ref"] == interval.PRIMARY_HASH
        calls.append(deepcopy(raw))
        return {"lower": LOWER, "upper": UPPER, "candidate": None}

    monkeypatch.setattr(interval, "calculate", replay)
    monkeypatch.setattr(snapshots, "archive_root", lambda: tmp_path)
    return calls


@pytest.mark.parametrize("all_other_points", [False, True])
def test_snapshot_replays_bounds_without_promoting_interval_to_point(monkeypatch, tmp_path, all_other_points):
    calls = stub_original_replay(monkeypatch, tmp_path)
    value = bundle(all_other_points=all_other_points)
    result = snapshots.calculate(value)
    assert calls and result["dimensions"]["liquidity"]["candidate"] is None
    assert result["point_dimensions"] == (7 if all_other_points else 2)
    assert result["interval_dimensions"] == 1
    assert result["score"] is result["grade"] is None and result["status"] == "PARTIAL"


@pytest.mark.parametrize("damage", ["wrong_share", "foreign_primary", "wrong_manager", "conditional_point", "endpoint"])
def test_snapshot_rejects_foreign_binding_conditional_point_or_forged_endpoint(monkeypatch, tmp_path, damage):
    stub_original_replay(monkeypatch, tmp_path)
    value = bundle()
    raw = value["health_stock_liquidity_interval_calculation"]["inputs"]
    if damage == "wrong_share":
        value["share_class"] = "A"
    elif damage == "foreign_primary":
        raw["primary_ref"] = "other-report"
    elif damage == "wrong_manager":
        raw["manager_name"] = "其他管理人"
    elif damage == "conditional_point":
        value["reviewed_document"]["dimensions"]["liquidity"] = {
            "lower": "66.8525",
            "upper": "66.8525",
            "candidate": "66.8525",
        }
    else:
        value["reviewed_document"]["dimensions"]["liquidity"]["upper"] = "82"
    with pytest.raises(ValueError, match="HEALTH_INTERVAL_"):
        snapshots.calculate(value, root=tmp_path)


@pytest.mark.parametrize("damage", [None, "archived_inputs", "binding", "endpoint", "point"])
def test_source_validation_checks_input_archive_binding_and_replays_interval(monkeypatch, tmp_path, damage):
    value = bundle()
    calls = stub_original_replay(monkeypatch, tmp_path)
    raw = value["health_stock_liquidity_interval_calculation"]["inputs"]
    archived = deepcopy(raw)
    if damage == "archived_inputs":
        archived["currency"] = "USD"
    elif damage == "binding":
        value["share_class"] = "A"
    elif damage == "endpoint":
        value["reviewed_document"]["dimensions"]["liquidity"]["upper"] = "82"
    elif damage == "point":
        value["reviewed_document"]["dimensions"]["liquidity"]["candidate"] = "66.8525"

    def read(ref, root):
        assert root == tmp_path
        return {
            "research.json": value["reviewed_document"],
            "identity.json": {
                "identity_verified": True,
                "code": interval.FUND,
                "published_on": "2026-08-28",
                "source": {"sha256": interval.PRIMARY_HASH},
                "title": interval.LEGAL_NAME,
                "pages": [interval.FUND + interval.LEGAL_NAME + interval.MANAGER_NAME],
            },
            "report.json": None,
            "inputs.json": archived,
        }[ref["file"]]

    monkeypatch.setattr(snapshots, "_read", read)
    frozen = datetime(2026, 10, 30, tzinfo=UTC)
    if damage:
        with pytest.raises(ValueError, match="INPUT_ARCHIVE_MISMATCH|HEALTH_INTERVAL_"):
            snapshots.validate_sources(value, frozen, root=tmp_path)
    else:
        snapshots.validate_sources(value, frozen, root=tmp_path)
        assert len(calls) == 1


def test_original_weights_preserve_unknown_capacity_domain_without_midpoint(monkeypatch, tmp_path):
    """算术使用真实R5子项规则；原件编译用隔离观测替身以避免读取实际归档。"""
    from app.services.fund_rating_cmf_liquidity import COMPATIBLE_TYPES, FOUNDATION_GATES, PROTOCOL

    observations = {
        "restricted": {"kind": "restricted", "restricted_amount": "1", "net_assets": "100", "scope_closed": True},
        "known_payment": {
            "kind": "purpose_covered_payment",
            "scope_closed": True,
            "restriction_amount_pending": False,
            "currency": "CNY",
            "as_of": interval.HOLDING,
            "bank_principal": "3",
            "broker_settlement_principal": "30",
            "settlement_liabilities": "10",
            "all_book_liabilities": "20",
            "disclosed_restrictions": [],
        },
        "holders": {
            "kind": "holder_execution_r5",
            "structure_type": None,
            "structure_condition_protocol": PROTOCOL,
            "compatible_structure_types": COMPATIBLE_TYPES,
            "structure_basis": "OPEN_END_CASH_LEGAL_PRODUCT_AND_COMPLETE_REPORTED_CATEGORY_TABLE",
            "foundation_gate_checks": {name: True for name in FOUNDATION_GATES},
            "structure_closed": True,
            "share_flow_closed": True,
            "actual_execution_verified": True,
            "redemption_mechanism_verified": True,
            "confirmed_pressure_effects": [],
        },
    }
    monkeypatch.setattr(interval, "compile_observations", lambda raw, root: {"observations": observations, "audit": {}})
    result = interval.calculate({}, root=tmp_path)
    assert result["lower"] == LOWER and result["upper"] == UPPER
    assert Decimal(result["upper"]) - Decimal(result["lower"]) == 40
    assert result["candidate"] is None and result["capacity_score_domain"]["candidate"] is None
    assert result["weights"] == {"market_capacity": 40, "restricted": 20, "known_payment": 25, "holders": 15}


def test_prepare_changes_only_liquidity_and_keeps_validity(monkeypatch, tmp_path):
    from app.services import fund_rating_pipeline_prepare as preparing

    value = bundle()
    value.pop("health_stock_liquidity_interval_calculation")
    value["reviewed_document"]["dimensions"]["liquidity"] = None
    original = deepcopy(value)
    stub_original_replay(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline,
        "current_inputs",
        lambda session, codes: {
            interval.FUND: {
                "as_of_date": interval.ASOF,
                "input_snapshot": {"members": [{"fund_code": interval.FUND, "bundle": value}]},
            }
        },
    )
    monkeypatch.setattr(preparing, "canonical_source", lambda root, ref: ref)
    monkeypatch.setattr(snapshots, "validate_sources", lambda *args, **kwargs: None)
    request = {
        "fund_code": interval.FUND,
        "as_of_date": interval.ASOF,
        "inputs": raw_inputs(),
        "explanation": "隔离测试保留三项原件事实及未知容量区间，不生成条件单点，不修改其他七项维度和原有效期。",
    }
    dossier = interval.prepare(object(), request, root=tmp_path)
    entry = dossier["entries"][0]
    document = snapshots._read(entry["research_archive"], tmp_path)
    assert all(
        document["dimensions"][k] == v
        for k, v in original["reviewed_document"]["dimensions"].items()
        if k != "liquidity"
    )
    assert document["dimensions"]["liquidity"]["candidate"] is None
    assert entry["verified_at"] == original["verified_at"] and entry["valid_until"] == original["valid_until"]


def test_pipeline_and_cli_prepare_bridge_never_publish(monkeypatch, tmp_path):
    calls = []
    expected = {"basis": snapshots.BASIS, "entries": []}

    def prepare(session, request, *, root):
        assert root == tmp_path
        calls.append(request)
        return expected

    monkeypatch.setattr(interval, "prepare", prepare)
    assert pipeline.prepare_health_stock_liquidity_interval(object(), {"fixture": True}, root=tmp_path) == expected

    class LocalSession:
        def __init__(self, engine):
            assert engine is None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def begin(self):
            return nullcontext()

    monkeypatch.setattr(fund_ratings, "get_engine", lambda: None)
    monkeypatch.setattr(fund_ratings, "Session", LocalSession)
    monkeypatch.setattr(snapshots, "archive_root", lambda: tmp_path)
    monkeypatch.setattr(snapshots, "publish_reviewed", lambda *args, **kwargs: pytest.fail("准备命令不能发布"))
    request_path, output_path = tmp_path / "request.json", tmp_path / "prepared.json"
    request_path.write_text(json.dumps({"fixture": "CLI"}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "fund_ratings",
            "prepare-health-stock-liquidity-interval",
            "--file",
            str(request_path),
            "--output",
            str(output_path),
        ],
    )
    fund_ratings.main()
    assert json.loads(output_path.read_text(encoding="utf-8")) == expected
    assert calls == [{"fixture": True}, {"fixture": "CLI"}]


def test_existing_bd500_snapshot_dispatch_is_preserved(monkeypatch, tmp_path):
    """007832已有分支仍调用它自己的重放器，不被010710区间接线替代。"""
    from app.services import fund_rating_bd500_performance as bd500

    value = bundle()
    value.pop("health_stock_liquidity_interval_calculation")
    value["fund_code"] = value["reviewed_document"]["code"] = "007832"
    value["reviewed_document"]["dimensions"]["liquidity"] = None
    value["bd500_performance_calculation"] = {"inputs": {"fund_code": "007832"}}
    calls = []

    def replay(current, dimensions, *, root):
        assert current["fund_code"] == "007832" and root == tmp_path
        calls.append(dimensions)

    monkeypatch.setattr(bd500, "check_snapshot", replay)
    checked = snapshots.calculate(value, root=tmp_path)
    assert len(calls) == 1 and checked["score"] is None
