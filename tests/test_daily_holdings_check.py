"""隔离验收每日检查：不能把漏跑、部分失败或持仓变化报告为成功。"""

from app.commands.fund_daily_check import assess
from app.services.direction_1d_protocol import digest


def completed():
    codes = ["123456", "654321"]
    state = {
        "owner_user_id": "owner",
        "scope": codes,
        "scope_hash": digest(codes),
        "checked_at": "2026-11-11T00:40:00+08:00",
        "window": {"target_nav_date": "2026-11-11"},
        "materials_terminal": {"terminal": True, "children": [{"job_type": "FUND_NEWS", "status": "SUCCEEDED"}]},
        "prediction_outcomes": {
            "summary": {
                "total": 4,
                "checked": 4,
                "completed": 4,
                "created": 0,
                "pending": 0,
                "failed": 0,
                "remaining": 0,
            }
        },
        "ratings": {"status": "SUCCEEDED", "scope": codes},
    }
    guard = {"owner_user_id": "owner", "window": state["window"], "status": "CLOSED", "all_predictions_generated": True}
    return state, guard, codes


def test_complete_receipts_and_fresh_owner_readback():
    state, guard, codes = completed()
    ratings = [{"fundCode": c, "status": "SCORED", "score": "70"} for c in codes]
    assert assess(state, guard, codes, ratings, "2026-11-11T09:00:00+08:00")["status"] == "SUCCEEDED"


def test_changed_holdings_and_yesterdays_run_are_not_success():
    state, guard, codes = completed()
    result = assess(state, guard, codes[:1], [{}], "2026-11-12T09:00:00+08:00")
    assert {"HOLDINGS_CHANGED", "RUN_NOT_CHECKED_TODAY", "RATINGS_INCOMPLETE"} <= set(result["issues"])


def test_missing_counts_partial_rating_and_external_blocks_are_reported():
    state, guard, codes = completed()
    state["prediction_outcomes"]["summary"].pop("remaining")
    state["ratings"]["status"] = "PARTIAL_SUCCESS"
    guard["all_predictions_generated"] = False
    result = assess(state, guard, codes, [], "2026-11-11T09:00:00+08:00")
    assert {
        "OUTCOME_RECEIPT_MISSING",
        "RATINGS_INCOMPLETE",
        "PREDICTIONS_INCOMPLETE",
        "RATING_READBACK_INCOMPLETE",
    } <= set(result["issues"])


def test_live_window_and_partial_rating_cannot_pass_on_old_success():
    state, guard, codes = completed()
    ratings = [{"fundCode": c, "status": "PARTIAL", "score": None} for c in codes]
    result = assess(state, guard, codes, ratings, "2026-11-11T09:00:00+08:00", {"target_nav_date": "2026-11-12"})
    assert {"LIVE_PREDICTION_WINDOW_CHANGED", "RATING_RESULTS_NOT_COMPLETE"} <= set(result["issues"])


def test_monthly_rating_publication_is_restricted_to_workflow_holdings(monkeypatch):
    import sqlalchemy.orm
    from app.db import session as database
    from app.services import fund_rating_inputs
    from app.services.fund_market_workflow import MarketSyncWorkflow

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(sqlalchemy.orm, "Session", lambda *args: Session())
    monkeypatch.setattr(database, "get_engine", lambda: object())
    monkeypatch.setattr(
        fund_rating_inputs, "prepare", lambda *args: {"test": [{"fund_code": "123456", "admitted": True}]}
    )
    workflow = MarketSyncWorkflow(scope_provider=lambda: [{"fund_code": "123456"}])
    workflow.news_results = {"123456": {"status": "SUCCEEDED", "candidate_events": []}}
    workflow.material_results = {"123456": {"status": "SUCCEEDED", "gaps": []}}
    received = []

    def updater(*args, **kwargs):
        received.append(kwargs)
        return {"rated": 1, "notRated": 0, "failures": [], "alreadyCurrent": []}

    assert workflow.ratings(lambda *args: None, updater=updater)["status"] == "SUCCEEDED"
    assert received[0]["scope_codes"] == received[0]["ready_codes"] == {"123456"}


def test_daily_continues_rating_after_material_barrier_and_cannot_close_failed_rating(tmp_path):
    from app.services.fund_daily_prediction import DailyPredictionRun
    from tests.test_fund_daily_prediction import Backend, reviews

    backend = Backend(["123456"])
    called = []
    backend.ratings = lambda terminal: called.append(terminal["terminal"]) or {"status": "FAILED"}
    backend.terminal = False
    run = DailyPredictionRun(tmp_path, backend)
    assert run.advance()["status"] == "WAITING_MATERIALS" and called == []
    backend.terminal = True
    reviews(tmp_path, backend.codes)
    result = run.advance()
    assert called == [True] and result["status"] == "RATINGS_PENDING"


def test_scope_uses_held_endpoint_and_validates_pagination():
    from types import SimpleNamespace
    from unittest.mock import patch

    import httpx
    from app.commands.fund_prediction_followthrough import owner_scope

    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=["123456"] if len(seen) == 1 else [])

    original = httpx.Client
    with patch(
        "app.commands.fund_prediction_followthrough.httpx.Client",
        side_effect=lambda **kw: original(transport=httpx.MockTransport(handler), **kw),
    ):
        settings = SimpleNamespace(
            core_service_base_url="http://local", ai_service_token=SimpleNamespace(get_secret_value=lambda: "test")
        )
        assert owner_scope(settings, "owner") == ["123456"]
    assert all("/users/owner/held-fund-codes" in url for url in seen)
