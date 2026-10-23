"""补充资料须绑定本基金、本轮来源和逐项证据，不能清空原缺项来放行。"""

from datetime import datetime
from pathlib import Path

import pytest
from app.services.fund_exposure_common import read, save
from app.services.fund_market_preparation import MarketFundPreparation
from app.services.fund_workflow_materials import inspect_materials, registered_materials

NOW = datetime.fromisoformat("2026-10-17T18:00:00+08:00")
CODE = "002112"
GAP = "本次公告目录或原文尚未核验完整"
BODY = "本基金公告检索已经覆盖指定期间且没有遗漏分页，此处是用于隔离测试的完整检索回执原文。"


@pytest.fixture
def material(tmp_path, monkeypatch):
    from app.services import fund_information_snapshot as snapshot
    from app.services import fund_market_review as review

    path = tmp_path / "supplement.json"
    save(path, {"fund_code": CODE, "documents": [BODY]})
    collected = {"fund_code": CODE, "source_hash": "a" * 64, "gaps": [GAP], "candidate_events": []}
    save(tmp_path / "materials" / (CODE + ".json"), collected)

    def prepare(at, code, materials_path=None):
        assert code == CODE and Path(materials_path).is_file()
        return {
            "fund_code": code,
            "materials_path": str(materials_path),
            "report": {},
            "documents": [{"id": "search", "body": BODY, "kind": "DISCLOSURE_SEARCH"}],
            "facts": {},
            "comprehensive_coverage": {
                "EXPOSURE": {"status": "AVAILABLE", "source_refs": ["search"]},
                "MARKET": {"status": "AVAILABLE", "source_refs": ["search"]},
                "DISCLOSURES": {"status": "VERIFIED_EMPTY", "source_refs": ["search"]},
            },
            "time_context": {"baseline_complete": True},
        }

    monkeypatch.setattr(snapshot, "prepare", prepare)
    monkeypatch.setattr(review, "now", lambda: NOW)
    return tmp_path, path, collected, prepare


def resolution():
    return {
        "gap": GAP,
        "decision": "RESOLVED",
        "explanation": "说明本基金检索的实际覆盖范围、分页边界和补充来源的关系，并且记录原文已经核对。" * 4,
        "evidence": [{"document_id": "search", "quote": BODY}],
    }


def test_registered_path_keeps_original_gap_and_cannot_approve_without_evidence(material):
    from app.services.fund_market_review import approve_comprehensive

    root, path, collected, _ = material
    result = inspect_materials(CODE, path, root=root, clock=lambda: NOW)
    assert GAP in result["gaps"] and not result["reviewed"]
    assert registered_materials(CODE, root=root, at=NOW)["materials_path"] == str(path)
    with pytest.raises(ValueError, match="GAPS_REMAIN"):
        approve_comprehensive(CODE, "完成资料审阅并记录具体依据。" * 12, root=root)
    assert read(root / "materials" / (CODE + ".json")) == collected


def test_explicit_path_is_rechecked_and_approved_without_rewriting_collector_gaps(material):
    from app.services.fund_market_review import approve_comprehensive

    root, path, collected, _ = material
    result = approve_comprehensive(
        CODE,
        "完成本基金资料与补充来源的逐项核验，保留评级原始缺口。" * 6,
        root=root,
        materials_path=path,
        resolutions=[resolution()],
    )
    assert result["reviewed"]
    receipt = registered_materials(CODE, root=root, at=NOW)
    assert receipt["reviewed"] and not receipt["gaps"]
    assert read(Path(receipt["materials_path"]))["workflow_source_hash"] == collected["source_hash"]
    assert "workflow_source_hash" not in read(path)
    assert read(root / "materials" / (CODE + ".json")) == collected
    checked = MarketFundPreparation(root=root, clock=lambda: NOW).prepare_inputs([{"fund_code": CODE}])
    assert checked["items"][0]["prediction_ready"]
    assert checked["items"][0]["materials_path"] == receipt["materials_path"]


@pytest.mark.parametrize("changed", ["source", "payload", "fund", "time"])
def test_changed_source_or_other_fund_cannot_reuse_handoff(material, changed):
    root, path, collected, _ = material
    inspect_materials(CODE, path, root=root, clock=lambda: NOW)
    if changed == "source":
        save(root / "materials" / (CODE + ".json"), {**collected, "source_hash": "b" * 64}, replace=True)
    elif changed == "time":
        with pytest.raises(ValueError, match="HANDOFF_CHANGED"):
            registered_materials(CODE, root=root, at=NOW.replace(hour=17))
        return
    else:
        save(path, {"fund_code": "000001" if changed == "fund" else CODE, "different": True}, replace=True)
    with pytest.raises(ValueError, match="HANDOFF_CHANGED"):
        registered_materials(CODE, root=root, at=NOW)


def test_fake_quote_and_duplicate_resolution_are_rejected(material):
    root, path, _, _ = material
    bad = resolution()
    bad["evidence"][0]["quote"] = "与本基金及原始检索无关的虚构引用，不能解除真实的资料缺口。"
    with pytest.raises(ValueError, match="RESOLUTION_SOURCE_INVALID"):
        inspect_materials(CODE, path, root=root, clock=lambda: NOW, resolutions=[bad])
    with pytest.raises(ValueError, match="RESOLUTION_INVALID"):
        inspect_materials(CODE, path, root=root, clock=lambda: NOW, resolutions=[resolution(), resolution()])


def test_missing_baseline_or_required_dimension_still_blocks_with_all_gap_resolutions(material, monkeypatch):
    from app.services import fund_information_snapshot as snapshot

    root, path, _, prepare = material

    def incomplete(*args):
        packet = prepare(*args)
        packet["time_context"]["baseline_complete"] = False
        packet["comprehensive_coverage"]["MARKET"]["status"] = "MISSING"
        return packet

    monkeypatch.setattr(snapshot, "prepare", incomplete)
    result = inspect_materials(CODE, path, root=root, clock=lambda: NOW, resolutions=[resolution()])
    assert len(result["gaps"]) == 2 and not result["reviewed"]


def test_pending_event_cannot_be_cleared_by_supplemental_materials(material, monkeypatch):
    from app.services import fund_market_review as review

    root, path, _, _ = material
    monkeypatch.setattr(review, "pending_events", lambda events: ["pending"])
    result = inspect_materials(CODE, path, root=root, clock=lambda: NOW, resolutions=[resolution()])
    assert "本期新增事项尚需结合原文复核" in result["gaps"]


def test_identical_review_reuses_original_approved_package_and_timestamp(material):
    from app.services.fund_market_review import approve_comprehensive

    root, path, _, _ = material
    note = "已逐条核对本基金补充来源和原缺口，不能把批准等同于生成判断。" * 5
    first = approve_comprehensive(CODE, note, root=root, materials_path=path, resolutions=[resolution()])
    prior = registered_materials(CODE, root=root, at=NOW)
    second = approve_comprehensive(CODE, note, root=root, materials_path=path, resolutions=[resolution()])
    assert first["materials_path"] == second["materials_path"] and second["reused"]
    assert registered_materials(CODE, root=root, at=NOW) == prior


def test_calendar_rejection_keeps_real_candidate_path_and_failure_without_approval(material, monkeypatch):
    from app.services import fund_information_snapshot as snapshot

    root, path, _, _ = material

    def incompatible(*args):
        raise ValueError("COMPREHENSIVE_CALENDAR_POLICY_REQUIRED")

    monkeypatch.setattr(snapshot, "prepare", incompatible)
    result = inspect_materials(CODE, path, root=root, clock=lambda: NOW, resolutions=[resolution()])
    assert result["prepare_failure"] == "COMPREHENSIVE_CALENDAR_POLICY_REQUIRED"
    assert GAP in result["gaps"] and not result["reviewed"]
    assert registered_materials(CODE, root=root, at=NOW)["materials_path"] == str(path)
    checked = MarketFundPreparation(root=root, clock=lambda: NOW).prepare_inputs([{"fund_code": CODE}])
    assert not checked["items"][0]["prediction_ready"]
    assert checked["items"][0]["prepare_failure"] == result["prepare_failure"]


@pytest.mark.parametrize("defect", [None, "future", "hash", "other_fund", "wrong_kind"])
def test_calendar_failure_can_only_handoff_verified_own_disclosures(material, monkeypatch, defect):
    """复用真实来源校验；公告可登记不等于日历、行情和批准条件被绕过。"""
    from app.services import fund_information_snapshot as snapshot
    from app.services.direction_1d_protocol import digest
    from app.services.fund_market_review import approve_comprehensive

    root, path, collected, _ = material
    document = {
        "id": "search", "kind": "DISCLOSURE_SEARCH", "codes": [CODE],
        "url": "https://example.test/disclosures", "body": BODY, "body_hash": digest(BODY),
        "published_date": "2026-10-16", "available_at": NOW.isoformat(), "body_scope": "FULL_TEXT",
    }
    if defect == "future":
        document["available_at"] = "2026-10-18T18:00:00+08:00"
    elif defect == "hash":
        document["body_hash"] = "0" * 64
    elif defect == "other_fund":
        document["codes"] = ["000001"]
    elif defect == "wrong_kind":
        document["kind"] = "MARKET_DATA"
    save(path, {"fund_code": CODE, "documents": [document]}, replace=True)
    collected["gaps"].append("本基金实际基准日净值或所列行情仍未齐备")
    save(root / "materials" / (CODE + ".json"), collected, replace=True)

    def incompatible(*args):
        raise ValueError("COMPREHENSIVE_CALENDAR_POLICY_REQUIRED")

    monkeypatch.setattr(snapshot, "prepare", incompatible)
    result = inspect_materials(CODE, path, root=root, clock=lambda: NOW, resolutions=[resolution()])
    assert (GAP in result["gaps"]) == (defect is not None)
    assert result["prepare_failure"] == "COMPREHENSIVE_CALENDAR_POLICY_REQUIRED"
    assert "本基金实际基准日净值或所列行情仍未齐备" in result["gaps"]
    assert "综合资料校验未通过：COMPREHENSIVE_CALENDAR_POLICY_REQUIRED" in result["gaps"]
    assert not result["reviewed"]
    assert read(root / "materials" / (CODE + ".json")) == collected
    with pytest.raises(ValueError, match="GAPS_REMAIN"):
        approve_comprehensive(CODE, "公告已补齐但日历与行情未通过，不能批准。" * 10, root=root)
