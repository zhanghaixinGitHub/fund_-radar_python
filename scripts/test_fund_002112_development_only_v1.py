"""续研关键边界：未来标签拒绝、冲突时间不能前移、窗口及衰减只看训练日期。"""

import json

import numpy as np
import pytest

from scripts import fund_002112_development_only_search_v1 as search
from scripts import fund_002112_development_timing_audit_v1 as timing


def test_future_label_rejected_before_nav_lookup():
    with pytest.raises(ValueError, match="POST_2025"):
        timing.labels([{"target": "2026-01-05", "base": "2025-12-31"}], {})


def test_mixed_input_filters_before_parsing_future_payload(tmp_path):
    path = tmp_path / "inputs.jsonl"
    path.write_text(
        json.dumps({"target": "2025-12-31", "value": 7}) + "\n"
        '{"target":"2026-01-05","invalid payload intentionally not parsed"\n',
        encoding="utf-8",
    )
    assert timing.dev_lines(path) == [{"target": "2025-12-31", "value": 7}]


def test_cms_late_time_cannot_be_dismissed_as_migration():
    report = {"available_at": "2023-08-31T08:00:00+08:00", "source": {"catalog_publish_ms": "1779330609000"}}
    assert timing.cms_bound(report) == "2026-05-21T10:30:09+08:00"


def test_restricted_nav_removes_future_dates():
    assert timing.restricted_nav({"2025-12-31": 1, "2026-01-05": 2}) == {"2025-12-31": 1}


def test_window_respects_maturity_before_slicing():
    rows = [{"target": str(i), "label_mature_at": "2024-01-02"} for i in range(5)]
    split = {"training": list(range(5)), "cutoff_exclusive": "2024-01-03"}
    assert search.training_rows(rows, split, 2) == rows[-2:]
    rows[0]["label_mature_at"] = "2024-01-03"
    with pytest.raises(ValueError, match="IMMATURE"):
        search.training_rows(rows, split, 2)


def test_half_life_and_mean_normalization():
    values = search.weights([{"session_index": 0}, {"session_index": 126}], 126)
    assert np.isclose(values[1] / values[0], 2)
    assert np.isclose(values.mean(), 1)
    assert search.weights([{"session_index": 0}], None) is None


def test_report_derived_groups_excluded_from_registered_groups():
    assert all("H" not in fields and "F" not in fields for fields in search.GROUPS.values())
