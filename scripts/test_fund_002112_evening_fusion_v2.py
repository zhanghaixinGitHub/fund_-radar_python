"""晚间模型的时间边界、未知值和分支权重测试，不检查历史成绩好坏。"""

from copy import deepcopy

import numpy as np
import pytest

from scripts import fund_002112_evening_fusion_v2 as m


@pytest.mark.parametrize(
    "stamp,expected",
    [
        ("2026-09-24T22:59:00+08:00", 0),
        ("2026-09-24T23:00:00+08:00", 0),
        ("2026-09-24T23:01:00+08:00", 0),
        ("2026-09-27T12:00:00+08:00", 0),
        ("2026-09-27T23:00:00+08:00", 0),
        ("2026-09-27T23:01:00+08:00", 1),
        ("2026-09-28T16:00:00+08:00", 1),
    ],
)
def test_2300_cutoff_and_holiday(stamp, expected):
    assert m.effective_index(["2026-09-24", "2026-09-28", "2026-09-29"], stamp) == expected


def test_date_only_publication_cannot_enter_same_night():
    row = {"nav_date": "2025-01-02", "ann_date": "2025-01-02"}
    assert m.available_nav(row) == "2025-01-03T00:00:00+08:00"


def test_recent_actual_receipt_constrains_input():
    row = {
        "nav_date": "2026-09-29",
        "ann_date": "2026-09-29",
        "created_at": "2026-09-30T01:24:00+00:00",
        "updated_at": "2026-09-30T01:24:00+00:00",
    }
    assert m.available_nav(row) == "2026-09-30T09:24:00+08:00"


def test_later_quote_revision_is_not_backfilled():
    assert not m.quote_ok({"revised_at": "2026-09-30T00:00:00+08:00"}, "2026-09-28", "2026-09-28T23:00:00+08:00")


def test_report_selected_by_public_time_instead_of_period_end():
    reports = [
        {"fund_code": "002112", "report_end": "2026-06-30", "available_at": "2026-08-30T00:00:00+08:00"},
        {"fund_code": "002112", "report_end": "2026-09-30", "available_at": "2026-10-25T00:00:00+08:00"},
    ]
    assert m.choose_report(reports, "2026-09-28T23:00:00+08:00")["report_end"] == "2026-06-30"


def test_missing_never_becomes_claimed_zero_and_future_median_not_learned():
    train = np.array([[1, np.nan], [3, np.nan], [np.nan, np.nan]])
    encoded, state = m.preprocess(train)
    assert state["keep"].tolist() == [True, False]
    assert encoded.tolist() == [[1, 0], [3, 0], [2, 1]]
    future, _ = m.preprocess(np.array([[999, 55], [np.nan, 123]]), state)
    assert future.tolist() == [[999, 0], [2, 1]]
    assert np.isnan(train[2]).all()


def test_no_observed_feature_rejected():
    with pytest.raises(ValueError, match="NO_OBSERVED_FEATURE"):
        m.preprocess(np.array([[np.nan], [np.nan]]))


def test_actual_weights_applied_to_aligned_class_probabilities():
    values = {"information": np.array([[1, 0, 0]]), "market": np.array([[0, 1, 0]]), "history": np.array([[0, 0, 1]])}
    assert np.allclose(m.blend(values), [[0.55, 0.30, 0.15]])
    with pytest.raises(ValueError, match="MISSING_EXPERT"):
        m.blend({"history": values["history"]})


def test_future_public_event_cannot_change_features():
    sessions = ["2026-09-24", "2026-09-28", "2026-09-29"]
    source = {"sessions": sessions, "reports": [], "company_events": [], "public_events": []}
    before, _ = m.information_features(sessions[1], sessions[1] + "T23:00:00+08:00", source)
    changed = deepcopy(source)
    changed["public_events"] = [
        {
            "id": "future",
            "source_kind": "policy",
            "status": "SOURCE_GROUNDED",
            "repeated_facts": False,
            "available_at": "2026-09-29T00:00:00+08:00",
        }
    ]
    after, proof = m.information_features(sessions[1], sessions[1] + "T23:00:00+08:00", changed)
    assert before == after
    assert proof["events"] == []
    assert after[20] == 0 and after[22] is None


def test_public_background_not_forced_to_holding_benefit():
    source = {
        "sessions": ["2026-09-28", "2026-09-29"],
        "reports": [],
        "company_events": [],
        "public_events": [
            {
                "id": "a",
                "source_kind": "policy",
                "status": "SOURCE_GROUNDED",
                "repeated_facts": False,
                "available_at": "2026-09-28T09:00:00+08:00",
                "links": [],
                "direction": "UNKNOWN",
                "stage": "PROPOSAL",
                "kind": "POLICY",
                "channel": "NONE",
            }
        ],
    }
    vector, proof = m.information_features("2026-09-28", "2026-09-28T23:00:00+08:00", source)
    fields = dict(zip(m.INFO_NAMES, vector, strict=True))
    assert fields["policy_1_direction_UNKNOWN"] == 1
    assert fields["policy_1_direction_BENEFIT"] == 0
    assert proof["events"][0]["weight"] == 0


def test_holiday_calendar_does_not_use_nav_presence_as_session():
    sessions, _ = m.load_calendar()
    assert "2026-09-25" not in sessions
    assert sessions[sessions.index("2026-09-24") + 1] == "2026-09-28"
    assert sessions[sessions.index("2026-09-30") + 1] == "2026-10-08"


def test_weekend_policy_included_in_monday_forecast():
    source = {
        "sessions": ["2026-09-24", "2026-09-28", "2026-09-29"],
        "reports": [],
        "company_events": [],
        "public_events": [
            {
                "id": "weekend",
                "source_kind": "policy",
                "status": "SOURCE_GROUNDED",
                "repeated_facts": False,
                "available_at": "2026-09-26T00:00:00+08:00",
                "links": [],
                "direction": "UNKNOWN",
                "stage": "IMPLEMENTATION",
                "kind": "POLICY",
                "channel": "REGULATORY",
            }
        ],
    }
    _, proof = m.information_features("2026-09-24", "2026-09-27T23:00:00+08:00", source)
    assert proof["events"][0]["id"] == "weekend"
    assert proof["events"][0]["age"] == 0


def test_changing_target_nav_does_not_change_history_features():
    all_sessions, _ = m.load_calendar()
    sessions = [d for d in all_sessions if d.startswith("2025")][:70]
    nav = {d: {"unit_nav": str(1 + i / 100), "available_at": m.next_midnight(d)} for i, d in enumerate(sessions)}
    source = {"sessions": sessions, "nav": nav, "dividends": []}
    cutoff = m.prediction_evenings(sessions)[-1]
    before, _ = m.history_features(sessions[-2], cutoff, source)
    assert before is not None
    source["nav"][sessions[-1]]["unit_nav"] = "1000"
    after, _ = m.history_features(sessions[-2], cutoff, source)
    assert before == after
