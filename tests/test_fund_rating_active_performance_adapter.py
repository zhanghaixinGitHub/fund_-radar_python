"""主动组件适配的轻量集成测试；数学引擎注入，真实日期/来源绑定另有原件证明。"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parent


@pytest.fixture
def adapter():
    from app.services import fund_rating_active_performance_adapter

    return fund_rating_active_performance_adapter


@pytest.fixture
def setup(tmp_path, adapter):
    from app.services.fund_rating_reviewed_snapshot import BASIS, KEYS, _read
    from app.services.fund_rating_reviewed_snapshot import calculate as base_snapshot

    root = tmp_path
    (root / "sources").mkdir()

    def archive(root, value):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        h = hashlib.sha256(raw).hexdigest()
        f = root / "sources" / (h + ".json")
        f.write_bytes(raw)
        return {"file": "sources/" + f.name, "sha256": h, "kind": "json"}

    primary_bytes = b"isolated immutable primary original fixture"
    h = hashlib.sha256(primary_bytes).hexdigest()
    (root / "sources" / (h + ".pdf")).write_bytes(primary_bytes)
    primary = {"file": "sources/" + h + ".pdf", "sha256": h, "kind": "pdf"}
    legal = "测试主动灵活配置混合型证券投资基金"
    manager = "测试基金管理有限公司"
    text = (
        f"004604 004605 {legal} 基金管理人：{manager} 本报告期自2026年1月1日起至2026年6月30日止。 当期履职原文已核对。"
    )
    identity = {
        "code": "004605",
        "identity_verified": True,
        "published_on": "2026-08-31",
        "title": legal + "2026年中期报告",
        "source": {"sha256": h},
        "pages": [text],
    }
    report = {
        "success": 1,
        "data": {"security": [{"stock": "004605"}], "notice_date": "2026-08-31", "notice_content": text},
    }
    report_ref = archive(root, report)
    review = {
        "review_closed": True,
        "pending": False,
        "anchors": ["E3"],
        "evidence": [{"id": "E3", "source_ref": report_ref["sha256"], "quote": "当期履职原文已核对。"}],
    }
    review_ref = archive(root, review)
    dims = {key: None for key in KEYS}
    dims.update(management="55", fee="70")
    weights = {
        "return": 20,
        "risk": 20,
        "execution": 15,
        "stability": 10,
        "management": 10,
        "holdings": 10,
        "liquidity": 10,
        "fee": 5,
    }
    document = {"code": "004605", "as_of": "2026-09-30", "dimensions": dims, "outer_weights_percent": weights}
    bundle = {
        "basis": BASIS,
        "family": "ACTIVE_EQUITY",
        "fund_code": "004605",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "identity": {"legal_name": legal, "manager_name": manager},
        "original_report_archive": primary,
        "identity_archive": archive(root, identity),
        "report_published_on": "2026-08-31",
        "verified_at": "2026-10-28T01:35:45+00:00",
        "expires_at": "2026-11-04T01:35:45+00:00",
        "research_format": "single",
        "reviewed_document": document,
        "research_archive": archive(root, document),
        "extra_archives": [],
        "public_metrics": {"return": [{"old": True}]},
    }
    raw = {
        "fund_code": "004605",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "selected_dimensions": ["return"],
        "target_aliases": ["004604", "004605"],
        "target": {
            "fund_code": "004605",
            "share_class": "C",
            "legal_name": legal,
            "manager_name": manager,
            "holding_date": "2026-06-30",
            "primary_report_ref": h,
            "material_primary_report_ref": report_ref["sha256"],
            "material_primary_anchor_ids": ["E3"],
        },
        "material_event_review_ref": review_ref["sha256"],
        "sources": {h: primary, report_ref["sha256"]: report_ref, review_ref["sha256"]: review_ref},
    }

    def material(inputs, root):
        value = adapter._json(inputs["sources"][inputs["material_event_review_ref"]], root)
        return {"pending": value["pending"], "review_closed": value["review_closed"]}

    def binding(bundle, raw, *, root):
        return adapter.check_binding(bundle, raw, root=root, material_reviewer=material)

    def compute(raw, *, root):
        return {
            "dimensions": {
                key: {"return": "80", "risk": "79", "stability": "81"}[key] for key in raw["selected_dimensions"]
            },
            "complete_score": None,
            "grade": None,
        }

    def validator(bundle, frozen, *, root):
        adapter.validate_input_archives(bundle, root=root)

    def snapshot(bundle, *, root):
        dimensions = {
            key: (
                {"lower": "0", "upper": "100", "candidate": None}
                if value is None
                else deepcopy(value)
                if isinstance(value, dict)
                else {"lower": value, "upper": value, "candidate": value}
            )
            for key, value in bundle["reviewed_document"]["dimensions"].items()
        }
        adapter.replay_components(bundle, dimensions, root=root, compute=compute, binding_checker=binding)
        base = deepcopy(bundle)
        base.pop("active_performance_calculations", None)
        return base_snapshot(base, root=root)

    state = {"bundle": bundle, "reads": 0}

    def current(session, codes):
        state["reads"] += 1
        return {
            "004605": {
                "batch_id": "fresh-ref-" + str(state["reads"]),
                "as_of_date": "2026-09-30",
                "input_snapshot": {"members": [{"fund_code": "004605", "bundle": state["bundle"]}]},
            }
        }

    def prepare(raw):
        request = {
            "fund_code": "004605",
            "as_of_date": "2026-09-30",
            "inputs": raw,
            "explanation": (
                "按本份额真实原件重算指定表现维度，保留其他已核分项、原资料取得日期和现有治理事项，"
                "不把组件补件当成完整评分或未来保证。"
            ),
        }
        result = adapter.prepare_active_performance(
            None,
            request,
            root=root,
            current_reader=current,
            compute=compute,
            archiver=archive,
            source_validator=validator,
            snapshot_calculator=snapshot,
            binding_checker=binding,
        )
        entry = deepcopy(result["entries"][0])
        entry["basis"] = BASIS
        entry["reviewed_document"] = _read(entry["research_archive"], root)
        state["bundle"] = entry
        return entry

    return dict(
        root=root,
        raw=raw,
        state=state,
        prepare=prepare,
        binding=binding,
        compute=compute,
        snapshot=snapshot,
        archive=archive,
        report=report,
    )


@pytest.mark.parametrize("mutation", ["share", "fund", "manager", "legal", "primary", "holding", "family"])
def test_wrong_identity_or_period_rejected_before_compute(setup, mutation):
    raw = deepcopy(setup["raw"])
    bundle = setup["state"]["bundle"]
    if mutation == "share":
        raw["share_class"] = "A"
    elif mutation == "fund":
        raw["target"]["fund_code"] = "002112"
    elif mutation == "manager":
        raw["target"]["manager_name"] = "其他管理人"
    elif mutation == "legal":
        raw["target"]["legal_name"] = "其他法律产品"
    elif mutation == "primary":
        raw["target"]["primary_report_ref"] = "0" * 64
    elif mutation == "holding":
        raw["target"]["holding_date"] = "2025-12-31"
    elif mutation == "family":
        bundle["family"] = "INDEX"
    with pytest.raises(ValueError, match="R5_ACTIVE_PERFORMANCE_INPUT_SCOPE"):
        setup["prepare"](raw)


def test_different_report_json_cannot_borrow_current_pdf(setup):
    raw = deepcopy(setup["raw"])
    report = deepcopy(setup["report"])
    report["data"]["notice_content"] = report["data"]["notice_content"].replace(
        "2026年1月1日起至2026年6月30日止", "2025年1月1日起至2025年12月31日止"
    )
    ref = setup["archive"](setup["root"], report)
    raw["sources"][ref["sha256"]] = ref
    raw["target"]["material_primary_report_ref"] = ref["sha256"]
    review = {
        "review_closed": True,
        "pending": False,
        "anchors": ["E3"],
        "evidence": [{"id": "E3", "source_ref": ref["sha256"], "quote": "当期履职原文已核对。"}],
    }
    r = setup["archive"](setup["root"], review)
    raw["sources"][r["sha256"]] = r
    raw["material_event_review_ref"] = r["sha256"]
    with pytest.raises(ValueError, match="R5_ACTIVE_PERFORMANCE_MATERIAL_PRIMARY_REPORT_CONFLICT"):
        setup["prepare"](raw)


def test_two_packages_coexist_and_same_scope_replaced(setup):
    original = deepcopy(setup["state"]["bundle"])
    setup["prepare"](deepcopy(setup["raw"]))
    stable = deepcopy(setup["raw"])
    stable["selected_dimensions"] = ["stability"]
    setup["prepare"](stable)
    replacement = deepcopy(setup["raw"])
    replacement["research_revision"] = "second-review"
    third = setup["prepare"](replacement)
    assert setup["state"]["reads"] == 3
    assert len(third["active_performance_calculations"]) == 2
    by_scope = {tuple(p["inputs"]["selected_dimensions"]): p for p in third["active_performance_calculations"]}
    assert by_scope[("return",)]["inputs"]["research_revision"] == "second-review"
    assert "research_revision" not in by_scope[("stability",)]["inputs"]
    assert third["verified_at"] == original["verified_at"] and third["expires_at"] == original["expires_at"]
    assert third["reviewed_document"]["dimensions"]["management"] == "55"
    assert original["reviewed_document"]["dimensions"]["return"] is None


def test_archive_inline_input_mismatch_is_rejected(setup, adapter):
    bundle = setup["prepare"](deepcopy(setup["raw"]))
    bundle["active_performance_calculations"][0]["inputs"]["target"]["share_class"] = "A"
    with pytest.raises(ValueError, match="R5_ACTIVE_PERFORMANCE_INPUT_ARCHIVE_MISMATCH"):
        adapter.validate_input_archives(bundle, root=setup["root"])


def test_forged_dimension_score_is_rejected(setup):
    bundle = setup["prepare"](deepcopy(setup["raw"]))
    bundle["reviewed_document"]["dimensions"]["return"].update(lower="99", upper="99", candidate="99")
    with pytest.raises(ValueError, match="R5_ACTIVE_PERFORMANCE_DIMENSION_RECALCULATION:return"):
        setup["snapshot"](bundle, root=setup["root"])


def test_repeated_snapshot_scope_is_rejected(setup):
    bundle = setup["prepare"](deepcopy(setup["raw"]))
    bundle["active_performance_calculations"].append(deepcopy(bundle["active_performance_calculations"][0]))
    with pytest.raises(ValueError, match="R5_ACTIVE_PERFORMANCE_DUPLICATE_SCOPE"):
        setup["snapshot"](bundle, root=setup["root"])


def test_eight_numeric_candidates_do_not_make_full_score(setup):
    bundle = setup["prepare"](deepcopy(setup["raw"]))
    stable = deepcopy(setup["raw"])
    stable["selected_dimensions"] = ["stability"]
    bundle = setup["prepare"](stable)
    for key, value in bundle["reviewed_document"]["dimensions"].items():
        if value is None:
            bundle["reviewed_document"]["dimensions"][key] = "70"
    result = setup["snapshot"](bundle, root=setup["root"])
    assert result["point_dimensions"] == 8 and result["status"] == "PARTIAL"
    assert result["score"] is None and result["grade"] is None and not bundle.get("r5_calculation")


@pytest.mark.parametrize("bound", ["lower", "upper", "candidate"])
def test_every_stored_bound_must_match_raw_recalculation(setup, bound):
    bundle = setup["prepare"](deepcopy(setup["raw"]))
    bundle["reviewed_document"]["dimensions"]["return"][bound] = "79"
    with pytest.raises(ValueError, match="R5_ACTIVE_PERFORMANCE_DIMENSION_RECALCULATION:return"):
        setup["snapshot"](bundle, root=setup["root"])


def test_partial_overlap_does_not_silently_remove_other_dimension(setup):
    raw = deepcopy(setup["raw"])
    raw["selected_dimensions"] = ["return", "risk"]
    setup["prepare"](raw)
    original = deepcopy(setup["state"]["bundle"])
    with pytest.raises(ValueError, match="R5_ACTIVE_PERFORMANCE_OVERLAPPING_SCOPE_REQUIRES_REVIEW"):
        setup["prepare"](deepcopy(setup["raw"]))
    assert setup["state"]["bundle"] == original


def test_unlisted_material_anchor_id_cannot_bind_primary_report(setup):
    raw = deepcopy(setup["raw"])
    raw["target"]["material_primary_anchor_ids"] = ["NOT-AN-ANCHOR"]
    with pytest.raises(ValueError, match="R5_ACTIVE_PERFORMANCE_PRIMARY_ANCHOR_IDS_REQUIRED"):
        setup["prepare"](raw)
