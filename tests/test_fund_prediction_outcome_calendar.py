from datetime import datetime

import pytest
from app.services import fund_prediction_outcome_calendar as service
from app.services.prediction_contract import fingerprint


def fixture(monkeypatch):
    from app.services import comprehensive_multi_prediction as multi

    review = {"horizon_id": "T5_V1"}
    source = {"data": {"fund_code": "019449", "as_of": "2026-10-23T01:00:00+08:00"}, "review": review}
    row = {"fund_code": "019449", "knowledge_cutoff": datetime.fromisoformat(source["data"]["as_of"]),
           "payload": source, "content_hash": fingerprint(source)}
    prediction = {"fundCode": "019449", "featureHash": row["content_hash"], "reviewHash": fingerprint(review),
                  "generatedAt": "2026-10-23T01:01:00+08:00", "horizonId": "T5_V1", "startDate": "2026-10-23",
                  "baseNavDate": "2026-10-22", "endDate": "2026-10-29", "calendarId": "OWN_JP",
                  "calendarHash": "calendar", "targetDefinitionId": "test",
                  "predictionPolicySnapshot": {"target_definition_id": "test"}}
    calendar = object()
    monkeypatch.setattr(multi, "review_calendar", lambda *a: calendar)
    monkeypatch.setattr(service, "target_dates", lambda *a, **kw: {k: prediction[k] for k in (
        "baseNavDate", "startDate", "endDate", "horizonId", "calendarId", "calendarHash", "targetDefinitionId")})
    return prediction, row, calendar


def test_uses_same_original_calendar_for_outcome_only(monkeypatch):
    prediction, row, calendar = fixture(monkeypatch)
    assert service.verified_calendar(prediction, row) is calendar
    assert service.restored_calendar({"generationMethod": "OLD_METHOD"}) is None


@pytest.mark.parametrize("field,value", [("fund_code", "000001"), ("content_hash", "wrong")])
def test_wrong_fund_or_hash_cannot_supply_calendar(monkeypatch, field, value):
    prediction, row, _ = fixture(monkeypatch)
    row[field] = value
    with pytest.raises(ValueError):
        service.verified_calendar(prediction, row)


def test_calendar_revision_cannot_change_original_target(monkeypatch):
    prediction, row, _ = fixture(monkeypatch)
    monkeypatch.setattr(service, "target_dates", lambda *a, **kw: {**prediction, "endDate": "2026-10-30"})
    with pytest.raises(ValueError, match="CALENDAR_CHANGED"):
        service.verified_calendar(prediction, row)
