import json
from copy import deepcopy
from datetime import UTC, date, datetime
from types import SimpleNamespace

import httpx
import pytest
from app.services.fund_exposure_common import read
from app.services.fund_rating_monthly_inputs import _acquire, apply_management_review


def test_same_management_review_is_idempotent_but_new_evidence_is_retained(tmp_path, monkeypatch):
    from app.services import fund_rating_monthly_inputs as monthly

    body = "008164 本基金 现任经理 继续履职"
    monkeypatch.setattr(monthly, "Originals", lambda *a, **kw: SimpleNamespace(pages=lambda _: [body]))
    monkeypatch.setattr("app.services.fund_rating_management_anchor.appointment_verified", lambda *a: True)
    inputs = {
        "fund_code": "008164",
        "legal_name": "本基金",
        "as_of_date": "2026-10-30",
        "sources": {},
        "management_inputs": {
            "sources": {},
            "claims": [],
            "experience": {"source_refs": []},
            "continuity": {
                "source_refs": [],
                "intervals": [{"name": "经理", "start": "2020-01-01", "end": None, "source_refs": []}],
            },
        },
    }
    bundle = {"evidence_v1_calculation": {"inputs": inputs}}
    review = {
        "fund_code": "008164",
        "as_of_date": "2026-10-30",
        "reason": "实际独立核对职责" * 20,
        "role_quote": "现任经理",
        "source": {"sha256": "first", "published_on": "2026-10-27"},
    }
    history = {"dates": ["2023-10-31"]}
    apply_management_review(bundle, review, root=tmp_path, history=history)
    frozen = deepcopy(inputs)
    apply_management_review(bundle, review, root=tmp_path, history=history)
    assert inputs == frozen
    changed = deepcopy(review)
    changed.update(role_quote="继续履职", source={"sha256": "second", "published_on": "2026-10-28"})
    apply_management_review(bundle, changed, root=tmp_path, history=history)
    raw = inputs["management_inputs"]
    assert raw["experience"]["source_refs"] == ["first", "second"]
    assert raw["continuity"]["source_refs"] == ["first", "second"]
    assert raw["continuity"]["intervals"][0]["source_refs"] == ["first", "second"]
    assert raw["claims"] == [
        {"source_hash": "first", "quote": "现任经理"},
        {"source_hash": "second", "quote": "继续履职"},
    ]


def test_business_rejection_keeps_real_response_and_request(tmp_path, monkeypatch):
    observed = []

    def respond(request):
        observed.append(request)
        return httpx.Response(200, json={"code": 111, "msg": "signature required", "data": None})

    reference = {"url": "https://www.cmfchina.com/ws-business-server/fund/getIncomeList"}
    params = {"productCode": "016008", "startDate": "2023-10-31", "endDate": "2026-10-30"}
    monkeypatch.setattr(
        "app.integrations.cmf_public.income_list",
        lambda client, params, **kwargs: client.get(reference["url"], params=params),
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError, match="BUSINESS_RESPONSE:111"):
            _acquire(reference, params, root=tmp_path, output=tmp_path, client=client)
    assert observed[0].url.params["endDate"] == "2026-10-30"
    receipt = read(next((tmp_path / "attempts").glob("*.json")))
    assert receipt["params"] == params
    assert json.loads(next((tmp_path / "originals").glob("*.json")).read_bytes())["code"] == 111


def test_unapproved_endpoint_never_requested(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("unexpected request"))) as client:
        with pytest.raises(ValueError, match="ADAPTER_REQUIRED"):
            _acquire({"url": "https://example.com/data"}, {}, root=tmp_path, output=tmp_path, client=client)


def test_management_requires_actual_new_period_review(tmp_path):
    bundle = {"evidence_v1_calculation": {"inputs": {"fund_code": "013472", "as_of_date": "2026-10-30"}}}
    with pytest.raises(ValueError, match="ACTUAL_MANAGEMENT_REVIEW_REQUIRED"):
        apply_management_review(bundle, {"fund_code": "013472", "as_of_date": "2026-09-30"}, root=tmp_path, history={})


def test_future_management_disclosure_cannot_extend_history(tmp_path):
    bundle = {"evidence_v1_calculation": {"inputs": {"fund_code": "013472", "as_of_date": "2026-10-30"}}}
    review = {
        "fund_code": "013472",
        "as_of_date": "2026-10-30",
        "reason": "实际审阅" * 30,
        "role_quote": "现任经理",
        "source": {"published_on": "2026-10-31"},
    }
    with pytest.raises(ValueError, match="FUTURE_DISCLOSURE"):
        apply_management_review(bundle, review, root=tmp_path, history={})


def test_normalized_index_shape_replays_production_table(monkeypatch, tmp_path):
    from app.services import fund_rating_monthly_inputs as monthly
    from app.services.fund_rating_direct_index import _table

    captured = []
    monkeypatch.setattr(monthly, "get_settings", lambda: SimpleNamespace(tushare_api_url="https://api.tushare.pro"))
    monkeypatch.setattr(
        monthly, "_derived", lambda root, value, **kwargs: captured.append(value) or {"sha256": "normalized"}
    )
    client = SimpleNamespace(
        list_index_daily=lambda *args, **kwargs: [
            SimpleNamespace(index_code="931494.CSI", trade_date=date(2026, 10, 30), close_price="123.45")
        ]
    )
    monthly._index_daily("931494.CSI", "2026-10-30", "2026-10-30", root=tmp_path, index_client=client)
    assert _table(captured[0]) == [{"ts_code": "931494.CSI", "trade_date": "2026-10-30", "close": "123.45"}]


def test_provider_retains_real_nontrading_publications_for_issuer_crosscheck(monkeypatch):
    from app.services import fund_rating_monthly_inputs as monthly

    sessions = [
        date(year, month, 28)
        for year in range(2023, 2027)
        for month in range(1, 13)
        if (2023, 10) <= (year, month) <= (2026, 10)
    ]
    sessions[0], sessions[-1] = date(2023, 10, 31), date(2026, 10, 30)
    extra = date(2023, 12, 31)
    rows = [
        {
            "nav_date": day,
            "ann_date": day,
            "unit_nav": "1",
            "accumulated_nav": "1",
            "adjusted_nav": "1",
            "content_hash": "actual-row",
        }
        for day in sorted([*sessions, extra])
    ]
    monkeypatch.setattr(monthly, "calendar", lambda: (sessions, "test-calendar"))

    class Result:
        def __init__(self, value):
            self.value = value

        def mappings(self):
            return self

        def one(self):
            return self.value

        def all(self):
            return self.value

    class Session:
        def execute(self, sql, params=None):
            if "FROM source_registry" in str(sql):
                return Result(
                    {
                        "source_id": "source",
                        "enabled": True,
                        "authorization_verified_at": datetime(2026, 11, 17, tzinfo=UTC),
                    }
                )
            if "FROM nav_daily" in str(sql):
                return Result([r for r in rows if params["start"] <= r["nav_date"] <= params["end"]])
            return Result([])

    result = monthly.provider_snapshot("016008", date(2026, 10, 30), session=Session())
    assert str(extra) not in result["history"]["dates"]
    assert str(extra) in {row["nav_date"] for row in result["history"]["rows"]}
    assert result["history"]["supplementary_non_trading_dates"] == [str(extra)]
