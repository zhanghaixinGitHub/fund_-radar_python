"""013180分阶段接入验证；测试替身不冒充真实原件或已发布评分。"""

import json
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal as D
from decimal import localcontext

import pytest
from app.commands import fund_ratings
from app.services import fund_rating_gf_battery_liquidity_interval as interval
from app.services import fund_rating_pipeline as pipeline
from app.services import fund_rating_reviewed_snapshot as snapshots
from app.services.fund_rating_inputs import digest

LOWER = "33.5328919788483596577963901763586953715915705653145224379245"
UPPER = "72.2770063696461027736741031740883233574037234597030042275864"


def fixture_input():
    """只供绑定/接线测试；空原件图不会用于真实评分计算。"""
    base = {
        "component_scope": "NON_LIQUIDITY_SEVEN_DIMENSIONS",
        "fund_code": interval.FUND,
        "share_class": "C",
        "as_of_date": interval.ASOF,
        "holding_date": interval.HOLDING,
        "target_refs": {"half": interval.PRIMARY},
        "legal_name": interval.LEGAL,
        "cohort": {"peers": ["015874"]},
        "sources": {},
    }
    return {
        "version": interval.VERSION,
        "fund_code": interval.FUND,
        "share_class": "C",
        "as_of_date": interval.ASOF,
        "holding_date": interval.HOLDING,
        "research_on": "2026-09-30",
        "legal_name": interval.LEGAL,
        "manager_name": interval.MANAGER,
        "primary_ref": interval.PRIMARY,
        "base_compilation": base,
        "base_compilation_sha256": digest(base),
        "sources": {
            sha: {"sha256": sha, "file": "sources/" + sha + ".pdf",
                  "published_on": "2026-08-31", "acquired_at": "2026-09-01T00:00:00+00:00"}
            for sha in [interval.OWN_PDF, interval.ETF_PDF, interval.LEGAL_PDF]
        },
    }


def fixture_bundle():
    raw = fixture_input()
    return {
        "basis": snapshots.BASIS,
        "family": "INDEX",
        "fund_code": interval.FUND,
        "share_class": "C",
        "as_of_date": interval.ASOF,
        "holding_date": interval.HOLDING,
        "identity": {"legal_name": interval.LEGAL, "manager_name": interval.MANAGER},
        "original_report_archive": {"sha256": interval.PRIMARY},
        "verified_at": "2026-10-28T01:35:55.686642+00:00",
        "valid_until": "2026-11-28T01:35:55.686642+00:00",
        "passive_components_calculation": {"packet": {"passive_index_compilation": raw["base_compilation"]}},
        "reviewed_document": {
            "code": interval.FUND,
            "as_of": interval.ASOF,
            "dimensions": {key: (None if key == "liquidity" else str(n + 50))
                           for n, key in enumerate(snapshots.KEYS)},
        },
        interval.PACKAGE: {"inputs": raw},
    }


def point_free_answer():
    return {"lower": LOWER, "upper": UPPER, "candidate": None, "complete_score": None, "grade": None}


def test_current_binding_preserves_same_c_report_and_original_seven_graph():
    bundle = fixture_bundle()
    interval.check_binding(bundle, bundle[interval.PACKAGE]["inputs"])
    changed = deepcopy(bundle[interval.PACKAGE]["inputs"])
    changed["base_compilation"]["cohort"]["peers"] = ["999999"]
    changed["base_compilation_sha256"] = digest(changed["base_compilation"])
    with pytest.raises(ValueError, match="CURRENT_FROZEN_COMPILATION"):
        interval.check_binding(bundle, changed)


@pytest.mark.parametrize("field,value", [("share_class", "A"), ("holding_date", "2025-06-30"),
                                         ("family", "ACTIVE_EQUITY")])
def test_foreign_share_period_or_family_binding_rejected(field, value):
    bundle = fixture_bundle()
    bundle[field] = value
    with pytest.raises(ValueError, match="CURRENT_BINDING"):
        interval.check_binding(bundle, bundle[interval.PACKAGE]["inputs"])


def test_each_scope_call_still_constructs_byte_validator(monkeypatch, tmp_path):
    calls = []

    def originals(raw, root):
        calls.append((raw, root))
        return object()

    monkeypatch.setattr(interval, "Originals", originals)
    raw = fixture_input()
    interval.original_scope(raw, tmp_path)
    interval.original_scope(raw, tmp_path)
    assert len(calls) == 2 and all(root == tmp_path for _, root in calls)


def test_source_graph_or_post_cutoff_report_cannot_be_invented(tmp_path):
    raw = fixture_input()
    raw["sources"][interval.OWN_PDF]["published_on"] = "2026-10-01"
    with pytest.raises(ValueError, match="OFFICIAL_PUBLICATION_DATE"):
        interval.original_scope(raw, tmp_path)
    raw = fixture_input()
    raw["sources"].pop(interval.ETF_PDF)
    with pytest.raises(ValueError, match="SOURCE_GRAPH"):
        interval.original_scope(raw, tmp_path)


def test_original_formula_keeps_unknown_payment_and_restriction_as_ranges(monkeypatch, tmp_path):
    """隔离数学事实有明确97上界；不把区间中点或银行账面值当实际可用值。"""
    compiled = {
        "observations": {
            "market_capacity": {"target": "0.1", "peers": [{"value": "0.1"}]},
            "holders": {"kind": "holder_execution_r5", "structure_type": "DISPERSED",
                        "structure_closed": True, "share_flow_closed": True,
                        "actual_execution_verified": True, "redemption_mechanism_verified": True,
                        "confirmed_pressure_effects": []},
        },
        "audit": {"restriction_risk_budget_lower": "3", "own_balance": {"net_assets": "100"},
                  "available_bank_principal": None, "all_asset_restricted_amount": None},
    }
    monkeypatch.setattr(interval, "compile_observations", lambda *args, **kwargs: deepcopy(compiled))
    result = interval.calculate({}, root=tmp_path)
    assert result["candidate"] is result["complete_score"] is result["grade"] is None
    assert result["subscore_domains"]["known_payment"] == {"lower": "0", "upper": "75", "candidate": None}
    assert D(result["subscore_domains"]["restricted"]["upper"]) == 97
    with localcontext() as context:
        context.prec = 60
        assert D(result["upper"]) - D(result["lower"]) == D("38.15")
    assert result["weights"] == {"market_capacity": 40, "restricted": 20, "known_payment": 25, "holders": 15}
    assert result["audit"]["available_bank_principal"] is None


@pytest.mark.parametrize("field,value", [("candidate", "52.90"), ("lower", "34"), ("upper", "72")])
def test_interval_endpoints_and_none_candidate_are_checked(monkeypatch, tmp_path, field, value):
    bundle = fixture_bundle()
    monkeypatch.setattr(interval, "calculate", lambda *args, **kwargs: point_free_answer())
    bounds = {k: point_free_answer()[k] for k in ["lower", "upper", "candidate"]}
    interval.replay_snapshot_interval(bundle, {"liquidity": bounds}, root=tmp_path)
    bounds[field] = value
    with pytest.raises(ValueError, match="SNAPSHOT_INTERVAL_MISMATCH"):
        interval.replay_snapshot_interval(bundle, {"liquidity": bounds}, root=tmp_path)


def test_prepare_preserves_seven_values_package_and_original_validity(monkeypatch, tmp_path):
    """只验证新增prepare的数据变更范围；三处真实重放已由独占review验证。"""
    from app.services import fund_rating_pipeline_prepare as preparing

    before = fixture_bundle()
    raw = before.pop(interval.PACKAGE)["inputs"]
    original = deepcopy(before)
    monkeypatch.setattr(preparing, "canonical_source", lambda root, ref: ref)
    monkeypatch.setattr(interval, "calculate", lambda *args, **kwargs: point_free_answer())
    calls = []

    def validate(bundle, frozen, *, root):
        calls.append("validate")
        assert root == tmp_path
        assert bundle["verified_at"] == original["verified_at"]

    def calculate(bundle, *, root):
        calls.append("calculate")
        return {"dimensions": bundle["reviewed_document"]["dimensions"],
                "score": None, "grade": None, "status": "PARTIAL"}

    request = {"fund_code": interval.FUND, "as_of_date": interval.ASOF, "inputs": raw,
               "explanation": "保留原七维和原验证时间，仅新增基于本基金真实原件约束的流动性范围；"
               "未知银行可用额与全部受限金额仍不补零。"}
    previous = {interval.FUND: {"as_of_date": interval.ASOF,
                                "input_snapshot": {"members": [{"fund_code": interval.FUND, "bundle": before}]}}}
    dossier = interval.prepare(
        None, request, root=tmp_path,
        current_reader=lambda session, codes: previous,
        source_validator=validate, snapshot_calculator=calculate,
    )
    entry = dossier["entries"][0]
    document = snapshots._read(entry["research_archive"], tmp_path)
    assert calls == ["validate", "calculate"]
    assert all(document["dimensions"][key] == value
               for key, value in original["reviewed_document"]["dimensions"].items() if key != "liquidity")
    assert document["dimensions"]["liquidity"]["candidate"] is None
    assert document["dimensions"]["liquidity"]["lower"] == LOWER
    assert document["dimensions"]["liquidity"]["upper"] == UPPER
    assert entry["passive_components_calculation"] == original["passive_components_calculation"]
    assert entry["verified_at"] == original["verified_at"] and entry["valid_until"] == original["valid_until"]


def test_pipeline_cli_are_prepare_only_and_keep_health_interval_bridge(monkeypatch, tmp_path):
    calls = []
    expected = {"basis": snapshots.BASIS, "entries": []}

    def prepare(session, request, *, root):
        assert root == tmp_path
        calls.append(request)
        return expected

    monkeypatch.setattr(interval, "prepare", prepare)
    assert pipeline.prepare_gf_battery_liquidity_interval(None, {"fixture": "pipeline"}, root=tmp_path) == expected

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
    monkeypatch.setattr(fund_ratings, "get_engine", lambda: None)
    monkeypatch.setattr(snapshots, "archive_root", lambda: tmp_path)
    monkeypatch.setattr(snapshots, "publish_reviewed", lambda *a, **k: pytest.fail("准备命令不应发布"))
    request_path, output_path = tmp_path / "request.json", tmp_path / "prepared.json"
    request_path.write_text(json.dumps({"fixture": "CLI"}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["fund_ratings", "prepare-gf-battery-liquidity-interval",
                                    "--file", str(request_path), "--output", str(output_path)])
    fund_ratings.main()
    assert json.loads(output_path.read_text(encoding="utf-8")) == expected
    assert calls == [{"fixture": "pipeline"}, {"fixture": "CLI"}]
    assert callable(pipeline.prepare_health_stock_liquidity_interval)


def snapshot_fixture(monkeypatch, tmp_path):
    """父层合并snapshot后的控制验证；原七维编译仍走它原有入口的隔离替身。"""
    from app.services import fund_rating_r5

    value = fixture_bundle()
    value["research_archive"] = {"file": "research.json", "sha256": "research"}
    value["research_format"] = "single"
    value["identity_archive"] = {"file": "identity.json", "sha256": "identity"}
    value["original_report_archive"]["file"] = "report.json"
    value["reviewed_document"]["outer_weights_percent"] = {k: "12.5" for k in snapshots.KEYS}
    value["reviewed_document"]["dimensions"]["liquidity"] = {
        "lower": LOWER, "upper": UPPER, "candidate": None, "reason": "隔离测试的原件约束区间"}
    value[interval.PACKAGE]["input_archive"] = {"file": "interval.json", "sha256": "interval"}
    package = value["passive_components_calculation"]
    package["packet"].update(fund_code=interval.FUND, as_of_date=interval.ASOF)
    package.update(anchors={}, input_archive={"file": "passive.json", "sha256": "passive"},
                   anchors_archive={"file": "anchors.json", "sha256": "anchors"})
    points = {snapshots.KEYS[key]: str(point) for key, point in value["reviewed_document"]["dimensions"].items()
              if key != "liquidity"}
    calls = []

    def calculate(raw, *, root):
        assert root == tmp_path
        calls.append("interval")
        return point_free_answer()

    def passive(packet, anchors, *, root):
        assert root == tmp_path and packet == package["packet"] and anchors == {}
        calls.append("passive_seven")
        return {"dimensions": points}

    monkeypatch.setattr(interval, "calculate", calculate)
    monkeypatch.setattr(fund_rating_r5, "calculate_passive_components", passive)
    return value, calls


def test_parent_merged_snapshot_replays_interval_and_existing_seven(monkeypatch, tmp_path):
    value, calls = snapshot_fixture(monkeypatch, tmp_path)
    result = snapshots.calculate(value, root=tmp_path)
    assert calls == ["interval", "passive_seven"]
    assert result["point_dimensions"] == 7 and result["interval_dimensions"] == 1
    assert result["score"] is result["grade"] is None and result["status"] == "PARTIAL"
    assert result["dimensions"]["liquidity"]["candidate"] is None
    assert result["dimensions"]["liquidity"]["lower"] == LOWER
    assert result["dimensions"]["liquidity"]["upper"] == UPPER


@pytest.mark.parametrize("damage", [None, "archive", "share", "endpoint", "point"])
def test_parent_merged_source_gate_checks_archive_binding_and_null_point(monkeypatch, tmp_path, damage):
    from app.services import fund_rating_reviewed_quality

    value, calls = snapshot_fixture(monkeypatch, tmp_path)
    archived_inputs = deepcopy(value[interval.PACKAGE]["inputs"])
    if damage == "archive":
        archived_inputs["fixture_mutation"] = True
    elif damage == "share":
        value["share_class"] = "A"
    elif damage == "endpoint":
        value["reviewed_document"]["dimensions"]["liquidity"]["upper"] = "73"
    elif damage == "point":
        value["reviewed_document"]["dimensions"]["liquidity"]["candidate"] = "52.90"

    def read(ref, root):
        assert root == tmp_path
        return {
            "research.json": value["reviewed_document"],
            "identity.json": {"identity_verified": True, "code": interval.FUND, "published_on": "2026-08-31",
                              "source": {"sha256": interval.PRIMARY}, "title": interval.LEGAL,
                              "pages": [interval.FUND + interval.LEGAL + interval.MANAGER]},
            "report.json": None, "interval.json": archived_inputs,
            "passive.json": value["passive_components_calculation"]["packet"], "anchors.json": {},
        }[ref["file"]]

    monkeypatch.setattr(snapshots, "_read", read)
    monkeypatch.setattr(fund_rating_reviewed_quality, "validate_sources", lambda *args, **kwargs: None)
    if damage:
        with pytest.raises(ValueError, match="INPUT_ARCHIVE_MISMATCH|R5_GF_INTERVAL_"):
            snapshots.validate_sources(value, datetime(2026, 10, 30, tzinfo=UTC), root=tmp_path)
    else:
        snapshots.validate_sources(value, datetime(2026, 10, 30, tzinfo=UTC), root=tmp_path)
        assert calls == ["interval"]
