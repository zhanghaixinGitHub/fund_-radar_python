"""更新频率研究的日历和标签成熟边界；不运行任何监督拟合。"""

from datetime import date, timedelta

import pytest

from scripts.fund_002112_update_frequency_v1 import calendar


def sample_rows():
    days = [date(2024, 12, 20) + timedelta(days=i) for i in range(4)]
    days += [date(2025, 1, 2) + timedelta(days=i) for i in range(243)]
    return [
        {
            "target": str(d),
            "as_of": str(d - timedelta(days=1)) + "T08:00:00+08:00",
            "label_mature_at": str(d + timedelta(days=2)) + "T08:00:00+08:00",
            "session_index": i,
        }
        for i, d in enumerate(days)
    ]


def test_updates_share_origin_and_only_use_previously_mature_labels():
    rows = sample_rows()
    c = calendar(rows)
    assert len(c["updates"]) == 13
    for update in c["updates"].values():
        assert all(rows[i]["label_mature_at"] < update["cutoff_exclusive"] for i in update["training"])
        assert set(update["training"]).isdisjoint(update["evaluation"])
    for cadence in (20, 60):
        mapping = c["prediction_maps"][f"RF_D4__EVERY{cadence}"]
        assert len(mapping) == 243
        assert mapping[0]["model_id"].endswith("U000")
        assert mapping[cadence - 1]["model_id"].endswith("U000")
        assert mapping[cadence]["model_id"].endswith(f"U{cadence:03d}")


def test_label_maturing_exactly_at_cutoff_is_excluded():
    rows = sample_rows()
    rows[0]["label_mature_at"] = rows[4]["as_of"]
    assert 0 not in calendar(rows)["updates"]["U000"]["training"]


def test_calendar_never_uses_return_or_prediction_to_schedule():
    rows = sample_rows()
    before = calendar(rows)
    for r in rows:
        r.update(return_value=999, previous_prediction="WRONG")
    assert calendar(rows) == before


def test_missing_session_cannot_silently_shorten_update_cadence():
    rows = sample_rows()
    rows[50]["session_index"] += 1
    with pytest.raises(ValueError, match="CONTIGUOUS"):
        calendar(rows)
