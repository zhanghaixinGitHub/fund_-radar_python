"""评级筛选在分页前生效，使用隔离内存目录及评级替身，不采集或修改真实业务数据。"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.api.dependencies import require_service_token
from app.api.routes import funds
from app.models.fund import FundShareClass
from app.services import fund_catalog_read as service
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool


@pytest.fixture
def catalog(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    FundShareClass.__table__.create(engine)
    with Session(engine) as session:
        session.add_all([
            FundShareClass(fund_code=f"{i:06d}", fund_master_id=uuid4(), share_class="A",
                           fund_name=f"测试基金{i}", fund_type="BOND" if i <= 3 else "MIXED",
                           status="ACTIVE", source_code="TEST")
            for i in range(1, 126)
        ])
        session.commit()
    monkeypatch.setattr(service, "get_engine", lambda: engine)
    reads = []

    def read(codes):
        reads.append(list(codes))
        return {"items": [
            {"fundCode": code, "status": "RATED", "grade": "GOOD", "validUntil": "2099-01-01T00:00:00Z"}
            for code in codes
        ]}

    def summaries(session, codes):
        return [SimpleNamespace(fund=session.get(FundShareClass, code), nav_date=None,
                                performance=SimpleNamespace(day_change_rate=None, week_change_rate=None,
                                                            month_change_rate=None)) for code in reversed(codes)]

    monkeypatch.setattr(service, "read_ratings", read)
    monkeypatch.setattr(service, "list_fund_summaries_by_codes", summaries)
    yield reads
    engine.dispose()


def test_rating_pagination_covers_all_batches_and_retains_catalog_order(catalog):
    result = service.list_funds(None, None, 10, None, 11, "GOOD")
    assert result.total_count == 125 and result.total_pages == 13
    assert [item.fund_code for item in result.items] == [f"{i:06d}" for i in range(101, 111)]
    assert [len(batch) for batch in catalog] == [100, 25]
    assert list(result.ratings) == [item.fund_code for item in result.items]


def test_type_keyword_and_rating_filters_combine_before_count_and_page(catalog):
    result = service.list_funds("基金12", "MIXED", 3, None, 2, "GOOD")
    assert result.total_count == 7 and result.total_pages == 3
    assert [item.fund_code for item in result.items] == ["000122", "000123", "000124"]
    assert len(catalog[0]) == 7
    result = service.list_funds(None, "BOND", 10, None, 1, "EXCELLENT")
    assert result.total_count == 0 and result.items == () and result.ratings == {}


@pytest.mark.parametrize(("rating", "expected"), [
    ("GOOD", ["000001"]), ("EXCELLENT", ["000002"]), ("AVERAGE", ["000003"]), ("WEAK", ["000004"]),
    ("UNRATED", ["000005", "000006"]), ("PENDING", ["000007", "000008"]),
])
def test_unavailable_and_expired_results_never_masquerade_as_valid_grade(catalog, monkeypatch, rating, expected):
    statuses = ["RATED"] * 4 + ["NOT_RATED", "UNSUPPORTED", "STALE", "WITHDRAWN"]
    grades = ["GOOD", "EXCELLENT", "AVERAGE", "WEAK"] + [None] * 4
    def read(codes):
        return {"items": [{"fundCode": code, "status": statuses[int(code) - 1], "grade": grades[int(code) - 1]}
                          for code in codes]}
    monkeypatch.setattr(service, "read_ratings", read)
    monkeypatch.setattr(service, "iter_fund_code_batches", lambda *a, **kw: [[f"{i:06d}" for i in range(1, 9)]])
    result = service.list_funds(None, None, 10, None, 1, rating)
    assert [item.fund_code for item in result.items] == expected


def test_cursor_count_stays_global_but_next_cursor_tracks_filtered_page(catalog):
    result = service.list_funds(None, None, 10, "000110", None, "GOOD")
    assert result.total_count == 125
    assert result.next_cursor == "000120"
    assert result.items[0].fund_code == "000111"


def test_incomplete_rating_read_fails_instead_of_omitting_funds(catalog, monkeypatch):
    monkeypatch.setattr(service, "read_ratings", lambda codes: {"items": []})
    with pytest.raises(ValueError, match="RATING_FILTER_INCOMPLETE"):
        service.list_funds(None, None, 10, None, 1, "GOOD")


def test_http_filter_validation_authorization_and_failure(catalog, monkeypatch):
    app = FastAPI()
    app.include_router(funds.router, prefix="/funds")
    client = TestClient(app)
    assert client.get("/funds?rating=GOOD").status_code == 403
    app.dependency_overrides[require_service_token] = lambda: None
    assert client.get("/funds?rating=INVALID").status_code == 422
    assert client.get("/funds?rating=GOOD&page=1&cursor=000001").status_code == 422
    response = client.get("/funds?rating=GOOD&page=2&pageSize=10&fundType=MIXED")
    assert response.status_code == 200
    assert response.json()["total_count"] == 122
    assert len(response.json()["ratings"]) == 10
    monkeypatch.setattr(service, "read_ratings", lambda codes: {"items": []})
    response = client.get("/funds?rating=GOOD")
    assert response.status_code == 503 and "INCOMPLETE" not in response.text
