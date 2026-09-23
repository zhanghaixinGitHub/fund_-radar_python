"""未来公告、已知晚修订、未成熟标签与重复评价区间是硬边界，历史重建假设不能掩盖它们。"""

from datetime import date, datetime
from decimal import Decimal

import pytest
from app.services.prediction_contract import PredictionFailure
from app.services.prediction_features import (
    build_features,
    known_dividend_version,
    known_nav_version,
    select_fact_version,
)
from app.services.prediction_research import (
    label_values,
    mature_training_rows,
    validate_historical_model,
    validate_spec,
)
from tests.test_prediction_contract import calendar


def test_current_model_does_not_gain_earlier_historical_training_eligibility():
    with pytest.raises(PredictionFailure, match="历史拟合截止"):
        validate_historical_model(
            {"adapter": "LOGISTIC_STANDARDIZED_V1", "labelEndMax": "2025-01-01T00:00:00+08:00"},
            datetime.fromisoformat("2023-12-31T23:59:00+08:00"),
        )


def test_label_itself_rejects_late_publication_and_revision():
    cutoff = datetime.fromisoformat("2025-01-01T23:59:00+08:00")
    day = date(2024, 12, 31)
    row = {
        "nav_date": day,
        "unit_nav": Decimal("1.1"),
        "updated_at": cutoff,
        "source_published_at": datetime.fromisoformat("2025-01-02T10:00:00+08:00"),
    }
    with pytest.raises(PredictionFailure, match="截止后才公开"):
        label_values({"navs": [row]}, {day}, cutoff)
    future = datetime.fromisoformat("2025-01-03T10:00:00+08:00")
    row["source_published_at"] = None
    versions = [
        {"payload": row, "first_observed_at": future, "published_at": None},
        {"payload": row | {"unit_nav": Decimal("9")}, "first_observed_at": future, "published_at": None},
    ]
    with pytest.raises(PredictionFailure, match="晚修订"):
        label_values({"navs": [row], "navVersions": {str(day): versions}}, {day}, cutoff)


def test_dividend_amount_revision_uses_known_version_and_rejects_late_only_history():
    cutoff = datetime.fromisoformat("2026-09-22T10:00:00+08:00")
    event = {
        "source_event_key": "event",
        "cash_dividend": "0.2",
        "ex_date": "2026-09-21",
        "updated_at": "2026-09-23T10:00:00+08:00",
    }
    versions = [
        {"payload": event | {"cash_dividend": "0.1"}, "published_at": cutoff, "first_observed_at": cutoff},
        {"payload": event, "published_at": cutoff, "first_observed_at": datetime.fromisoformat(event["updated_at"])},
    ]
    assert known_dividend_version(event, versions, cutoff, True)["cash_dividend"] == Decimal("0.1")
    with pytest.raises(PredictionFailure, match="晚修订"):
        known_dividend_version(event, versions, datetime.fromisoformat("2026-09-20T10:00:00+08:00"), True)


def test_published_later_and_late_revision_are_not_past_inputs():
    cutoff = datetime.fromisoformat("2026-09-22T10:00:00+08:00")
    future = datetime.fromisoformat("2026-09-23T10:00:00+08:00")
    base = {"nav_date": "2026-09-21", "unit_nav": "1.1", "updated_at": cutoff.isoformat(), "source_published_at": None}
    old = {"first_observed_at": cutoff, "published_at": None, "payload_hash": "a", "payload": base}
    revision = {
        "first_observed_at": future,
        "published_at": future,
        "payload_hash": "b",
        "payload": base | {"unit_nav": "9"},
    }
    assert select_fact_version([old, revision], cutoff) == old
    assert known_nav_version(base, [old, revision], cutoff, True)["unit_nav"] == Decimal("1.1")
    with pytest.raises(PredictionFailure) as error:
        known_nav_version(base, [old | {"first_observed_at": future}, revision], cutoff, True)
    assert error.value.payload["code"] == "LATE_REVISION_UNAVAILABLE"
    assert select_fact_version([old | {"published_at": future}], cutoff) is None


def test_short_history_fails_with_counts_and_gap_is_not_filled():
    cal = calendar()
    days = [d for d in cal.sessions if d < date(2026, 9, 22)][-30:]
    now = datetime.fromisoformat("2026-09-22T10:00:00+08:00")
    data = {
        "calendar": cal,
        "navs": [
            {"nav_date": d, "unit_nav": Decimal(1), "updated_at": now, "source_published_at": None} for d in days[-12:]
        ],
    }
    with pytest.raises(PredictionFailure) as error:
        build_features(data, now, 20, persist=False)
    assert error.value.payload["details"] == {"requiredPoints": 21, "actualPoints": 12}
    data["navs"] = [
        {"nav_date": d, "unit_nav": Decimal(1), "updated_at": now, "source_published_at": None}
        for d in days
        if d != days[-3]
    ]
    data["dividends"] = []
    with pytest.raises(PredictionFailure) as error:
        build_features(data, now, 20, persist=False)
    assert error.value.payload["code"] == "NAV_GAP"


def test_immature_labels_and_relabelled_blind_test_are_rejected():
    fit = datetime.fromisoformat("2024-01-01T00:00:00+08:00")
    with pytest.raises(PredictionFailure) as error:
        mature_training_rows(
            [
                {
                    "key": "fund:T20",
                    "labelAvailableAt": "2024-01-02T00:00:00+08:00",
                    "knowledgeCutoff": "2023-12-01T00:00:00+08:00",
                }
            ],
            fit,
        )
    assert error.value.payload["code"] == "TRAINING_LABEL_NOT_MATURE"
    with pytest.raises(PredictionFailure) as error:
        validate_spec(
            {
                "fundCodes": ["006730"],
                "horizonIds": ["T5_V1"],
                "trainStart": "2022-01-04",
                "trainEnd": "2023-12-29",
                "validationEnd": "2024-06-28",
                "selectionEnd": "2025-12-31",
                "evidenceLevel": "BLIND_TEST",
            }
        )
    assert error.value.payload["code"] == "EVALUATION_ALREADY_USED"
