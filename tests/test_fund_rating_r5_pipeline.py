"""新评分链路边界：真实原始观察才可完整，缺项不补分，队列可重放且单只失败隔离。"""

import hashlib
from datetime import date, timedelta
from decimal import Decimal

import pytest
from app.services.fund_rating_r5 import calculate, fee_score, tracking_score
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, policy
from app.services.fund_rating_reviewed_snapshot import KEYS
from app.services.fund_rating_reviewed_snapshot import calculate as calculate_snapshot


@pytest.fixture
def original_packet(tmp_path):
    source = tmp_path / "sources" / "original.json"
    source.parent.mkdir()
    source.write_text('{"isolated_fixture":true}', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    ref = {"path": "sources/original.json", "sha256": digest, "published_on": "2026-08-31"}
    terms = policy(ACTIVE_BASIS)["terms"]
    nodes = {}
    for name in {name for row in terms.values() for name, _ in row}:
        nodes[name] = {
            "kind": "rank",
            "target": ".1",
            "direction": "higher" if name.startswith("return_") else "lower",
            "peers": [{"product_id": "other", "value": ".2"}],
            "expected_products": ["other"],
            "source_refs": [digest],
        }
    for name in ["experience", "governance", "allocation", "mandate"]:
        nodes[name] = {
            "kind": "state",
            "state": "NORMAL_REVIEWED",
            "reason": "隔离测试已核的明确事实",
            "source_refs": [digest],
        }
    for name, count in [("monthly_success", 36), ("annual_success", 25)]:
        nodes[name] = {
            "kind": "windows",
            "expected_windows": count,
            "rows": [
                {"end": str(date(2026, 9, 30) - timedelta(days=i * 30)), "target": ".10", "reference": ".09"}
                for i in range(count)
            ],
            "source_refs": [digest],
        }
    nodes["asset_quality"] = {
        "kind": "asset_quality",
        "stock_total": "100",
        "positions": [{"security": "fixture", "amount": "100", "state": "NORMAL_REVIEWED", "reason": "隔离测试事实"}],
        "source_refs": [digest],
    }
    nodes["restricted"] = {
        "kind": "restricted",
        "restricted_amount": "0",
        "net_assets": "100",
        "scope_closed": True,
        "source_refs": [digest],
    }
    nodes["known_payment"] = {
        "kind": "purpose_covered_payment",
        "scope_closed": True,
        "currency": "CNY",
        "as_of": "2026-06-30",
        "restriction_amount_pending": False,
        "bank_principal": "80",
        "broker_settlement_principal": "20",
        "settlement_liabilities": "20",
        "all_book_liabilities": "100",
        "source_refs": [digest],
    }
    nodes["continuity"] = {
        "kind": "responsibility_timeline",
        "timeline_closed": True,
        "current_role_verified": True,
        "arrangements_review_closed": True,
        "period_start": "2023-09-28",
        "period_end": "2026-09-30",
        "intervals": [{"name": "隔离负责人", "start": "2023-09-01", "end": None, "source_refs": [digest]}],
        "confirmed_functional_defects": [],
        "source_refs": [digest],
    }
    nodes["holders"] = {
        "kind": "holder_execution_r5",
        "structure_closed": True,
        "share_flow_closed": True,
        "actual_execution_verified": True,
        "redemption_mechanism_verified": True,
        "structure_type": "DISPERSED",
        "confirmed_pressure_effects": [],
        "source_refs": [digest],
    }
    return {
        "basis": ACTIVE_BASIS,
        "fund_code": "999999",
        "as_of_date": "2026-09-30",
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "target_product_aliases": ["target"],
            "observations": nodes,
            "material_event_pending": False,
        },
        "r4_evidence_registry": {digest: ref},
        "raw_archive": [{**ref, "kind": "ORIGINAL"}],
        "quality_public": {"dataDates": {"holdings": "2026-06-30"}},
    }, tmp_path


def test_complete_is_numeric_and_does_not_invent_new_grade(original_packet):
    packet, root = original_packet
    result = calculate(packet, {}, root=root)
    assert result["status"] == "SCORED" and result["grade"] is None
    assert set(result["dimensions"]) == set(KEYS.values())
    expected = sum(Decimal(result["dimensions"][k]) * result["weights"][k] / 100 for k in result["weights"])
    assert abs(expected - Decimal(result["score"])) < Decimal("1e-11")


@pytest.mark.parametrize("damage", ["amount", "peer", "month", "unit", "event", "hash", "outside"])
def test_incomplete_or_corrupt_observations_cannot_publish_complete(original_packet, damage):
    packet, root = original_packet
    nodes = packet["quality_metrics"]["observations"]
    if damage == "amount":
        nodes["known_payment"]["bank_principal"] = None
    elif damage == "peer":
        nodes["return_1y"]["peers"] = []
    elif damage == "month":
        nodes["monthly_success"]["rows"].pop()
    elif damage == "unit":
        packet.pop("unit_contract")
    elif damage == "event":
        packet["quality_metrics"]["material_event_pending"] = True
    elif damage == "hash":
        (root / "sources/original.json").write_text("changed", encoding="utf-8")
    else:
        next(iter(packet["r4_evidence_registry"].values()))["path"] = "../outside.json"
    with pytest.raises(ValueError):
        calculate(packet, {}, root=root)


def test_snapshot_needs_original_r5_recalculation_not_eight_imported_points(original_packet):
    packet, root = original_packet
    raw = calculate(packet, {}, root=root)
    doc = {
        "code": packet["fund_code"],
        "as_of": packet["as_of_date"],
        "outer_weights_percent": {k: raw["weights"][v] for k, v in KEYS.items()},
        "dimensions": {k: raw["dimensions"][v] for k, v in KEYS.items()},
    }
    bundle = {"fund_code": packet["fund_code"], "as_of_date": packet["as_of_date"], "reviewed_document": doc}
    assert calculate_snapshot(bundle, root=root)["status"] == "PARTIAL"
    bundle["r5_calculation"] = {"packet": packet, "anchors": {}}
    assert calculate_snapshot(bundle, root=root)["status"] == "SCORED"
    doc["dimensions"]["risk"] = "100"
    with pytest.raises(ValueError, match="RECALCULATION"):
        calculate_snapshot(bundle, root=root)


def test_frozen_small_cost_and_common_tracking_anchor():
    assert fee_score("61") - fee_score("59") == Decimal("-.5")
    assert tracking_score(".01", ".01") == 75
    assert tracking_score(".02", ".01") < tracking_score(".01", ".01")


def test_absolute_cost_accepts_raw_fees_without_unrelated_peer_gate(original_packet):
    packet, root = original_packet
    node = packet["quality_metrics"]["observations"]["cost_365"]
    packet["quality_metrics"]["observations"]["cost_365"] = {
        "kind": "standard_cost", "source_refs": node["source_refs"],
        "fee_inputs": {"subscription": "0", "redemption": "0", "operating_cost": ".0134",
                       "subscription_mode": "EXTERNAL"},
    }
    assert Decimal(calculate(packet, {}, root=root)["dimensions"]["cost"]) == Decimal("66.5")
    packet["quality_metrics"]["observations"]["cost_365"]["target"] = "0"
    with pytest.raises(ValueError, match="RAW_FEES_REQUIRED"):
        calculate(packet, {}, root=root)


def test_direct_compiler_replay_rejects_changed_raw_observation(original_packet, monkeypatch):
    from copy import deepcopy

    from app.services import fund_rating_direct_index

    packet, root = original_packet
    packet["verified_at"] = "2026-10-28T04:00:00+00:00"
    packet["direct_index_compilation"] = {k: packet[k] for k in ["fund_code", "as_of_date", "verified_at"]}
    expected = deepcopy(packet["quality_metrics"]["observations"])
    monkeypatch.setattr(fund_rating_direct_index, "compile_observations",
                        lambda inputs, root: {"observations": expected})
    packet["quality_metrics"]["observations"]["return_1y"]["target"] = ".11"
    with pytest.raises(ValueError, match="ORIGINAL_OBSERVATIONS_CONFLICT"):
        calculate(packet, {}, root=root)


def test_seven_real_dimensions_never_fill_missing_liquidity_or_form_complete_total(original_packet):
    from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen

    packet, root = original_packet
    full = calculate(packet, {}, root=root)
    selected = set(policy(packet['basis'])['terms']) - {'liquidity'}
    needed = {name for dimension, terms in policy(packet['basis'])['terms'].items()
              if dimension in selected for name, _weight in terms}
    packet['quality_metrics']['observations'] = {
        k: v for k, v in packet['quality_metrics']['observations'].items() if k in needed}
    for ref in packet['r4_evidence_registry'].values():
        ref['path'] = str(root / ref['path'])
    partial = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions=selected)
    assert partial['score'] is None and partial['raw_score'] is None and partial['grade'] is None
    assert set(partial['dimensions']) == selected
    assert all(partial['dimensions'][key] == full['dimensions'][key] for key in selected)
    with pytest.raises(ValueError, match='REQUIRED_OBSERVATIONS'):
        calculate(packet, {}, root=root)


def test_component_entry_replays_originals_and_rejects_observation_or_scope_tampering(original_packet, monkeypatch):
    from copy import deepcopy

    from app.services import fund_rating_direct_index
    from app.services.fund_rating_r5 import calculate_passive_components

    packet, root = original_packet
    packet['verified_at'] = '2026-10-28T04:00:00+00:00'
    selected = set(policy(packet['basis'])['terms']) - {'liquidity'}
    needed = {name for dim, terms in policy(packet['basis'])['terms'].items() if dim in selected for name, _ in terms}
    packet['quality_metrics']['observations'] = {
        k: v for k, v in packet['quality_metrics']['observations'].items() if k in needed}
    packet['passive_index_compilation'] = {k: packet[k] for k in ['fund_code', 'as_of_date', 'verified_at']}
    packet['passive_index_compilation']['component_scope'] = 'NON_LIQUIDITY_SEVEN_DIMENSIONS'
    expected = deepcopy(packet['quality_metrics']['observations'])
    anchor = deepcopy(next(iter(packet['r4_evidence_registry'].values())))
    monkeypatch.setattr(fund_rating_direct_index, 'compile_observations',
                        lambda inputs, root: {'observations': expected, 'anchor': anchor})
    result = calculate_passive_components(packet, {packet['fund_code']: anchor}, root=root)
    assert result['score'] is None and len(result['dimensions']) == 7 and 'liquidity' not in result['dimensions']
    packet['quality_metrics']['observations']['return_1y']['target'] = '.99'
    with pytest.raises(ValueError, match='ORIGINAL_OBSERVATIONS_CONFLICT'):
        calculate_passive_components(packet, {packet['fund_code']: anchor}, root=root)
    packet['passive_index_compilation']['component_scope'] = 'CLAIM_COMPLETE'
    with pytest.raises(ValueError, match='PACKET_SCOPE'):
        calculate_passive_components(packet, {packet['fund_code']: anchor}, root=root)


def test_one_broken_fund_does_not_block_following_fund(monkeypatch, tmp_path):
    from app.services import fund_rating_pipeline as module

    monkeypatch.setattr(
        module, "current_inputs", lambda session, codes: {code: {"as_of_date": date(2026, 9, 30)} for code in codes}
    )

    def evaluate(row, code, root, as_of):
        if code == "000001":
            raise ValueError("isolated_corrupt_source")
        return {"status": "PARTIAL", "blockers": []}

    monkeypatch.setattr(module, "_evaluate", evaluate)
    monkeypatch.setattr("app.services.fund_market_review.rating_review_required", lambda code: False)
    from sqlalchemy import create_engine

    result = module.update_new_rules(
        ["000001", "000002"], engine=create_engine("sqlite://"), root=tmp_path, as_of=date(2026, 9, 30)
    )
    assert result["failures"] == ["000001"] and result["items"][1]["status"] == "PARTIAL"
    assert result["published"] == 0
