"""多周期核对的真实隔离库分页、原文保护与北京时间边界；不访问外部来源。"""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

from app.repositories import prediction_store as store
from app.services import prediction_generation as generation
from app.services import prediction_models as models
from app.services.prediction_contract import POLICY_FILE, fingerprint
from app.services.prediction_direction import direction_fields
from sqlalchemy import text
from tests.test_prediction_contract import calendar
from tests.test_prediction_database import database as database


def seed(database, monkeypatch, count, *, end="2026-09-16"):
    monkeypatch.setattr(store, "get_engine", lambda: database)
    monkeypatch.setattr(generation, "get_engine", lambda: database)
    policy = json.loads(POLICY_FILE.with_name("prediction_policy_v2.json").read_text(encoding="utf-8"))
    original = {
        "fundCode": "006730", "horizonId": "T5_V1", "mode": "LIVE", "role": "PRIMARY",
        "modelId": models.freeze_routes()[models.route_key("T5_V1")]["model_id"],
        "activationRevision": 1, "generatedAt": "2026-09-15T01:00:00+00:00",
        "startDate": "2026-09-15", "endDate": end, "direction": "UP",
        "targetDefinitionId": "NEXT_EXECUTABLE_CASH_REINVESTED_THREE_STATE_V2",
        "predictionPolicySnapshot": policy, **direction_fields("T5_V1", policy),
    }
    with database.begin() as c:
        for index in range(count):
            store.save_prediction(c, {**original, "predictionId": str(uuid4())}, f"outcome-{index}")
    now = datetime.now(UTC)
    values = {date(2026, 9, 15): Decimal("1"), date.fromisoformat(end): Decimal("1.1")}
    monkeypatch.setattr(generation, "read_fund_data", lambda *a, **kw: {
        "calendar": calendar(), "dividends": [], "dividendWatermark": now,
        "navs": [{"nav_date": d, "unit_nav": v, "updated_at": now} for d, v in values.items()],
    })
    return values


def test_sync_checks_more_than_one_page_and_repeat_does_not_duplicate(database, monkeypatch):
    seed(database, monkeypatch, 501)
    progress = []
    result = generation.verify_all_outcomes(progress_reporter=lambda *args: progress.append(args))
    assert result["batches"] == 2
    assert result["total"] == result["checked"] == result["completed"] == result["created"] == 501
    assert result["remaining"] == result["pending"] == result["failed"] == 0
    assert [p[0] for p in progress] == [0, 500, 501]
    repeated = generation.verify_all_outcomes(progress_reporter=lambda *args: None)
    assert repeated["checked"] == repeated["completed"] == 501
    assert repeated["created"] == repeated["remaining"] == 0
    with database.connect() as c:
        assert c.execute(text("SELECT count(*) FROM prediction_outcome")).scalar() == 501
        assert c.execute(text("SELECT count(*) FROM prediction_check_state WHERE status='SUCCEEDED'")).scalar() == 501
        for row in c.execute(text("SELECT payload,content_hash FROM fund_prediction_record")).mappings():
            assert fingerprint(row["payload"]) == row["content_hash"]
            assert row["payload"]["direction"] == "UP"


def test_sync_exposes_page_budget_and_does_not_retry_missing_nav_in_same_run(database, monkeypatch):
    values = seed(database, monkeypatch, 501)
    values.clear()
    limited = generation.verify_all_outcomes(progress_reporter=lambda *args: None, max_batches=1)
    assert limited["checked"] == limited["pending"] == 500
    assert limited["remaining"] == 1
    result = generation.verify_all_outcomes(progress_reporter=lambda *args: None)
    assert result["checked"] == result["pending"] == 501
    assert result["completed"] == result["created"] == result["failed"] == result["remaining"] == 0


def test_0030_uses_beijing_day_for_maturity_and_dividend_watermark(database, monkeypatch):
    """北京时间27日00:30时，26日已到期；UTC26日不能让核对和分红水位晚一天。"""
    seed(database, monkeypatch, 1, end="2026-10-26")
    instant = datetime(2026, 10, 26, 16, 30, tzinfo=UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz)

    monkeypatch.setattr(generation, "datetime", Clock)
    # 只提供核对区间的估值日净值；不把真实日历缺日补成零收益。
    dates = [d for d in calendar().sessions if date(2026, 9, 15) <= d <= date(2026, 10, 26)]
    monkeypatch.setattr(generation, "read_fund_data", lambda *a, **kw: {
        "calendar": calendar(), "dividends": [], "dividendWatermark": datetime(2026, 10, 25, 16, 40, tzinfo=UTC),
        "navs": [{"nav_date": d, "unit_nav": Decimal("1.1"), "updated_at": instant} for d in dates],
    })
    result = generation.verify_outcomes()
    assert result["checked"] == result["completed"] == 1
    assert result["pending"] == result["failed"] == 0
