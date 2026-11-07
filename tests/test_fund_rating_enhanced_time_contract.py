"""研究日与估值日分别核验；只验证新增边界，不重跑历史PDF群。"""

import hashlib

import pytest
from app.services.fund_rating_enhanced_performance import TIME_CONTRACT, _nav_time_audit, _originals


def inputs(tmp_path, *, publication="2026-10-08", fundamental=False):
    folder = tmp_path / "sources"
    folder.mkdir()
    raw = b'{"TEST_ONLY":"not formal evidence"}'
    digest = hashlib.sha256(raw).hexdigest()
    path = folder / (digest + ".json")
    path.write_bytes(raw)
    ref = {"sha256": digest, "file": path.relative_to(tmp_path).as_posix(), "published_on": publication}
    price = {"sha256": "other-price"} if fundamental else ref
    return {
        "as_of_date": "2026-09-30",
        "research_on": "2026-10-29",
        "time_contract": TIME_CONTRACT,
        "calendar": {"dates": ["2026-09-30"]},
        "target": {"issuer_curve_source": price, "licensed_history_source": price},
        "index_source": price,
        "peers": [],
        "sources": {digest: ref},
    }


@pytest.mark.parametrize("ann", ["20261008", "2026-10-08"])
def test_nav_late_announcement_preserves_research_knowledge_boundary(tmp_path, ann):
    raw = inputs(tmp_path, publication=ann)
    checked = _originals(raw, tmp_path)
    assert checked.raw and raw["sources"][next(iter(checked.sources))]["published_on"] == ann
    audit = _nav_time_audit([{"nav_date": "20260930", "ann_date": ann}], raw, "014156")
    assert audit == [
        {
            "fund_code": "014156",
            "nav_date": "2026-09-30",
            "original_ann_date": "2026-10-08",
            "research_on": "2026-10-29",
            "was_known_at_nominal_cutoff": False,
        }
    ]


@pytest.mark.parametrize(
    "row,error",
    [
        ({"nav_date": "20261001", "ann_date": "20261008"}, "ENHANCED_NAV_FUTURE_VALUATION"),
        ({"nav_date": "20260930", "ann_date": "20261030"}, "ENHANCED_NAV_ANNOUNCED_AFTER_RESEARCH"),
        ({"nav_date": "20260930"}, "ENHANCED_DATE_SCHEMA"),
    ],
)
def test_nav_missing_date_future_economic_or_research_publication_rejected(tmp_path, row, error):
    with pytest.raises(ValueError, match=error):
        _nav_time_audit([row], inputs(tmp_path), "014156")


def test_fundamentals_disclosed_after_cutoff_are_still_rejected(tmp_path):
    with pytest.raises(ValueError, match="ENHANCED_DOCUMENT_OUTSIDE_KNOWLEDGE_BOUNDARY"):
        _originals(inputs(tmp_path, publication="2026-10-01", fundamental=True), tmp_path)


def test_explicit_research_date_and_contract_are_required(tmp_path):
    raw = inputs(tmp_path)
    raw.pop("research_on")
    with pytest.raises(ValueError, match="ENHANCED_DATE_SCHEMA"):
        _originals(raw, tmp_path)
    raw["research_on"] = "2026-10-29"
    raw["time_contract"] = {"kind": "BACKTEST"}
    with pytest.raises(ValueError, match="ENHANCED_EXPLICIT_TIME_CONTRACT_REQUIRED"):
        _originals(raw, tmp_path)
