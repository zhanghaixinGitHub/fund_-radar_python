from pathlib import Path

import pytest
from app.services.fund_rating_direct_index import th_publisher_adjusted_nav
from app.services.fund_rating_evidence_preview import management


def test_publisher_adjusted_nav_requires_identical_unit_nav():
    observations = {"day": {"unit_nav": 1, "publisher_nav": 2}}
    values, _ = th_publisher_adjusted_nav(observations, {"day": 1}, {"day": 1}, ["day"])
    assert values == {"day": 2}
    with pytest.raises(ValueError, match="OWN_NAV_CONFLICT"):
        th_publisher_adjusted_nav(observations, {"day": 1}, {"day": 3}, ["day"])


def test_reviewed_management_score_must_be_numeric():
    inputs = {"reviewed_management": {"score": None, "source_hash": "report"}}

    with pytest.raises(ValueError, match="MISSING_NUMERIC_OBSERVATION"):
        management(inputs, Path("unused"), "report")


def test_reviewed_management_requires_same_report():
    inputs = {"reviewed_management": {"score": "55", "source_hash": "report"}}

    assert management(inputs, Path("unused"), "report") == 55
    with pytest.raises(ValueError, match="MANAGEMENT_INPUTS_REQUIRED"):
        management(inputs, Path("unused"), "other")
    with pytest.raises(ValueError, match="MANAGEMENT_INPUTS_REQUIRED"):
        management({}, Path("unused"), "report")
