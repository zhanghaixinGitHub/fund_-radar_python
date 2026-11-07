"""007950两维接线的隔离反例；不读取研究目录，不连接真实数据库或重放全池。"""

import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction as F
from types import SimpleNamespace

import pytest
from app.commands import fund_ratings
from app.services import fund_rating_active_broad_admission as admission
from app.services import fund_rating_active_broad_components as component
from app.services import fund_rating_pipeline as pipeline
from app.services import fund_rating_pipeline_prepare as preparing
from app.services import fund_rating_reviewed_snapshot as snapshot


def inputs():
    """仅供身份/路由验证；空sources不会被宣称为真实评分资料。"""
    primary = "a" * 64
    return {
        "version": component.VERSION,
        "fund_code": "007950",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "research_on": "2026-10-29",
        "legal_name": "招商量化精选股票型发起式证券投资基金",
        "manager_name": "招商基金管理有限公司",
        "primary_ref": primary,
        "selected_dimensions": ["return", "stability"],
        "sources": {},
        "source_admission": {
            "primary_report_ref": primary,
            "target_quarters": [{"period": "2026-06-30", "source_hash": primary}],
        },
    }


def bundle():
    raw = inputs()
    dimensions = {key: "75" for key in snapshot.KEYS}
    dimensions["management"] = "55"
    return {
        "basis": snapshot.BASIS,
        "fund_code": "007950",
        "share_class": "C",
        "family": "ACTIVE_EQUITY",
        "as_of_date": raw["as_of_date"],
        "holding_date": raw["holding_date"],
        "identity": {"legal_name": raw["legal_name"], "manager_name": raw["manager_name"]},
        "original_report_archive": {"sha256": raw["primary_ref"]},
        "verified_at": "2026-10-29T20:00:00+08:00",
        "valid_until": "2026-11-29T20:00:00+08:00",
        "reviewed_document": {
            "code": "007950",
            "as_of": raw["as_of_date"],
            "dimensions": dimensions,
            "outer_weights_percent": dict(zip(snapshot.KEYS, (15, 20, 10, 10, 15, 10, 10, 10), strict=True)),
        },
    }


@pytest.mark.parametrize("change", ["share", "manager", "family", "primary", "financial_primary"])
def test_share_current_report_and_financial_anchor_binding(change):
    value, raw = bundle(), inputs()
    if change == "share":
        value["share_class"] = "A"
    elif change == "manager":
        value["identity"]["manager_name"] = "其他管理人"
    elif change == "family":
        value["family"] = "INDEX"
    elif change == "primary":
        value["original_report_archive"]["sha256"] = "another-report"
    else:
        raw["source_admission"]["primary_report_ref"] = "another-report"
    with pytest.raises(ValueError):
        component.check_binding(value, raw)


@pytest.mark.parametrize("bound", ["lower", "upper", "candidate"])
def test_snapshot_recalculates_all_bounds_and_cannot_turn_eight_numbers_into_full_score(monkeypatch, tmp_path, bound):
    value = bundle()
    value["active_broad_components_calculation"] = {"inputs": inputs()}
    monkeypatch.setattr(
        component,
        "calculate",
        lambda *args, **kwargs: {
            "dimensions": {"return": "75", "stability": "75"},
            "complete_score": None,
            "grade": None,
        },
    )
    result = snapshot.calculate(value, root=tmp_path)
    assert result["status"] == "PARTIAL" and result["score"] is None and result["grade"] is None
    value["reviewed_document"]["dimensions"]["return"] = {"lower": "75", "upper": "75", "candidate": "75"}
    value["reviewed_document"]["dimensions"]["return"][bound] = "76"
    with pytest.raises(ValueError, match="INTERVAL_INVALID|DIMENSION_ORIGINAL_RECALCULATION"):
        snapshot.calculate(value, root=tmp_path)


def test_frozen_previous_nav_less_cash_and_no_missing_previous_day_imputation():
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    points = dict(zip(days, (F(100), F(85), F(90)), strict=True))
    assert component._growth(points, {days[1]: F(10)}, days, days[0], days[2]) == 0
    assert points[days[2]] / points[days[0]] * (1 + F(10) / points[days[1]]) - 1 != 0
    with pytest.raises(ValueError, match="DIVIDEND_PREVIOUS_NAV_MISSING"):
        component._growth({days[0]: F(100), days[2]: F(90)}, {days[2]: F(10)}, days, days[0], days[2])


def test_region_exclusion_uses_proven_other_period_and_keeps_conflict():
    qs = [
        {"date": "2025-03-31", "domestic_ratio_upper_pct": "55.45"},
        {"date": "2026-03-31", "domestic_ratio_upper_pct": "50"},
    ]
    gaps = [{"day": "2026-03-31", "reason": "ASSET_INDUSTRY_CONFLICT_WITHOUT_PRIMARY_REPAIR"}]
    assert admission.proven_region_exclusions(qs, gaps, 80) == ["2025-03-31"]
    assert gaps[0]["reason"] == "ASSET_INDUSTRY_CONFLICT_WITHOUT_PRIMARY_REPAIR"


def test_empty_dividend_receipt_cannot_borrow_another_share():
    receipt = {
        "code": "015398",
        "rows": [],
        "sources": [{"sha256": "response", "status": 200, "received_at": "2026-10-19T16:39:28+08:00"}],
    }
    source = SimpleNamespace(raw={"receipt": json.dumps(receipt).encode()})
    member = {
        "code": "000126",
        "provider_share_id": "000126.OF",
        "dividend_receipt_ref": "receipt",
        "dividend_refs": ["response"],
    }
    with pytest.raises(ValueError, match="DIVIDEND_RECEIPT_IDENTITY"):
        component._event_receipt({"research_on": "2026-10-29"}, source, member)


def test_prepare_keeps_other_dimensions_and_original_validity(monkeypatch, tmp_path):
    before = bundle()
    before["active_execution_calculation"] = {"inputs": {"preserved": True}}
    original = deepcopy(before)
    monkeypatch.setattr(
        pipeline,
        "current_inputs",
        lambda session, codes: {
            "007950": {
                "as_of_date": "2026-09-30",
                "input_snapshot": {"members": [{"fund_code": "007950", "bundle": before}]},
            }
        },
    )
    monkeypatch.setattr(
        component, "calculate", lambda *args, **kwargs: {"dimensions": {"return": "81", "stability": "82"}}
    )
    monkeypatch.setattr(snapshot, "validate_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        snapshot, "calculate", lambda *args, **kwargs: {"status": "PARTIAL", "score": None, "grade": None}
    )
    request = {
        "fund_code": "007950",
        "as_of_date": "2026-09-30",
        "inputs": inputs(),
        "explanations": {
            key: "按本份额真实原件逐期核验，完整来源池不因缺失成员缩小，其他维度及明确业务限制继续保留。"
            for key in component.SELECTED
        },
    }
    entry = component.prepare(None, request, root=tmp_path)["entries"][0]
    document = snapshot._read(entry["research_archive"], tmp_path)
    assert document["dimensions"]["management"] == "55"
    assert all(
        document["dimensions"][key] == original["reviewed_document"]["dimensions"][key]
        for key in snapshot.KEYS
        if key not in component.SELECTED
    )
    assert entry["verified_at"] == original["verified_at"] and entry["valid_until"] == original["valid_until"]
    assert entry["active_execution_calculation"] == original["active_execution_calculation"]
    assert before == original


def test_prepare_wrapper_canonicalizes_once_without_changing_request(monkeypatch, tmp_path):
    calls = []
    request = {"inputs": {"sources": {"h": {"sha256": "h", "path": "original"}}}}
    original = deepcopy(request)
    monkeypatch.setattr(
        preparing, "canonical_source", lambda root, ref: calls.append(ref["sha256"]) or {**ref, "file": "canonical"}
    )
    monkeypatch.setattr(component, "prepare", lambda session, canonical, **kwargs: canonical)
    result = preparing.prepare_active_broad_components(None, request, root=tmp_path)
    assert calls == ["h"] and result["inputs"]["sources"]["h"]["file"] == "canonical"
    assert request == original


def test_prepare_cli_routes_without_publish_or_real_database(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(fund_ratings, "get_engine", lambda: None)

    class LocalSession:
        def __init__(self, engine):
            assert engine is None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def begin(self):
            return nullcontext()

    monkeypatch.setattr(fund_ratings, "Session", LocalSession)
    monkeypatch.setattr(snapshot, "archive_root", lambda: tmp_path)
    monkeypatch.setattr(
        preparing,
        "prepare_active_broad_components",
        lambda session, request, **kwargs: calls.append(request) or {"basis": snapshot.BASIS, "entries": []},
    )
    request, output = tmp_path / "request.json", tmp_path / "output.json"
    request.write_text('{"fixture":true}', "utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["fund-ratings", "prepare-active-broad-components", "--file", str(request), "--output", str(output)],
    )
    fund_ratings.main()
    assert calls == [{"fixture": True}] and json.loads(output.read_text("utf-8"))["entries"] == []


@pytest.mark.parametrize(
    "field",
    ["mixed_performance_calculation", "ca005187_execution_calculation", "gf_battery_liquidity_interval_calculation"],
)
def test_other_new_typed_snapshot_branches_are_still_called(monkeypatch, tmp_path, field):
    value = bundle()
    value[field] = {"inputs": {"fixture": True}}
    calls = []
    if field == "mixed_performance_calculation":
        from app.services import fund_rating_mixed_performance as module

        monkeypatch.setattr(module, "check_binding", lambda *args: calls.append("binding"))
        monkeypatch.setattr(
            module,
            "calculate",
            lambda *args, **kwargs: (
                calls.append("mixed") or {"admission_ready": True, "dimensions": {"return": "75", "risk": "75"}}
            ),
        )
    elif field == "ca005187_execution_calculation":
        from app.services import fund_rating_ca005187_execution as module

        monkeypatch.setattr(module, "validate_snapshot_component", lambda *args, **kwargs: calls.append("005187"))
    else:
        from app.services import fund_rating_gf_battery_liquidity_interval as module

        monkeypatch.setattr(module, "replay_snapshot_interval", lambda *args, **kwargs: calls.append("013180"))
    result = snapshot.calculate(value, root=tmp_path)
    assert calls and result["status"] == "PARTIAL" and result["score"] is None
