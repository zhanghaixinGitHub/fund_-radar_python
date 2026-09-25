"""公共资料发布、金融口径和模型引文校验的回归测试；不调用外部 API。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from app.services import fund_materials
from app.services.announcement_extraction import prepare_request, validate_extraction
from app.services.fund_exposure_common import save
from app.services.fund_materials_build import safe_url, select_statement
from fastapi.testclient import TestClient


def test_statement_excludes_future_parent_single_quarter_and_handles_conflicts():
    base = {"end_date": "20260630", "ann_date": "20260822", "report_type": "1", "revenue": 100}
    rows = [base, {**base, "report_type": "2", "revenue": 200}, {**base, "ann_date": "20261001", "revenue": 300}]
    assert select_statement(rows, "20260630", "20260924")["revenue"] == 100
    assert select_statement(rows, "20260331", "20260924") == {}
    adjusted = [{**base, "report_type": "4", "revenue": 110}, {**base, "report_type": "5", "revenue": 90}]
    assert select_statement([base, *adjusted], "20260630", "20260924")["revenue"] == 110
    assert select_statement([base, {**base, "revenue": 101}], "20260630", "20260924")["revenue"] is None


def test_safe_source_link():
    assert safe_url("javascript:alert(1)") is None
    assert safe_url("file:///C:/private.txt") is None
    assert safe_url("https://secret:password@example.com") is None
    assert safe_url("https://www.cninfo.com.cn/file.pdf")


def test_company_disclosure_preserves_forecast_units_and_excludes_future(monkeypatch, tmp_path):
    from app.services import fund_materials_build as builder

    sources = {
        "forecast": [
            {
                "ann_date": "20260901",
                "end_date": "20260930",
                "type": "预增",
                "net_profit_min": 100,
                "net_profit_max": 200,
            },
            {"ann_date": "20261001", "type": "不应展示"},
        ],
        "disclosure_date": [
            {"ann_date": "20260901", "pre_date": "20261020", "actual_date": "20261021"},
        ],
    }
    monkeypatch.setattr(builder, "financial_rows", lambda root, code, api: sources.get(api, []))
    result = builder.company_financials(tmp_path, "300308.SZ", "20260924")
    assert len(result["disclosures"]) == 2
    assert "100.00 至 200.00 万元" in result["disclosures"][0]["summary"]
    assert "来源实际披露日：暂缺" in result["disclosures"][1]["summary"]
    assert result["quote"]["close"] is None


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(fund_materials, "get_settings", lambda: SimpleNamespace(fund_material_directory=str(tmp_path)))
    fund_materials._snapshot.cache_clear()
    value = {
        "asOfDate": "2026-09-24",
        "notice": "test",
        "reports": [
            {
                "id": "r",
                "title": "报告",
                "endDate": "2026-06-30",
                "publishedDate": "2026-08-31",
                "fullDisclosure": True,
                "sourceUrl": None,
            }
        ],
        "companies": [],
        "documents": [
            {"id": str(i), "kind": "company", "stockCode": "300308.SZ", "title": f"回购{i}", "latestHeld": i % 2 == 0}
            for i in range(45)
        ],
    }
    save(tmp_path / "002112.json", value)
    return tmp_path


def test_missing_fund_never_reuses_002112(data):
    assert fund_materials.overview("008888")["available"] is False
    assert fund_materials.documents("008888")["total"] == 0
    with pytest.raises(ValueError):
        fund_materials.snapshot("../002112")


def test_pagination_and_latest_holdings_filter(data):
    assert len(fund_materials.documents("002112", page=3)["items"]) == 5
    filtered = fund_materials.documents("002112", latest_only=True, keyword="回购")
    assert filtered["total"] == 23
    assert all(r["latestHeld"] for r in filtered["items"])
    assert fund_materials.documents("002112", stock_code="000001.SZ")["total"] == 0
    with pytest.raises(ValueError):
        fund_materials.overview("002112", report_id="wrong")


def test_integrity_failure_does_not_publish(data):
    path = data / "002112.json"
    path.write_text(path.read_text(encoding="utf-8").replace("回购0", "篡改0"), encoding="utf-8")
    with pytest.raises(ValueError):
        fund_materials.overview("002112")


def extraction():
    return {
        "rule_version": "ANNOUNCEMENT_FACTS_V1",
        "document_id": "a",
        "source_sha256": "a" * 64,
        "status": "extracted",
        "facts": [
            {
                "subject": "公司",
                "category": "财务报告",
                "stage": "不适用",
                "summary": "公司披露营业收入。",
                "evidence": [{"page": 2, "quote": "营业收入 100 元"}],
                "metrics": [
                    {
                        "name": "营业收入",
                        "raw_value": "100",
                        "value": "100",
                        "unit": "元",
                        "evidence": [{"page": 2, "quote": "营业收入 100 元"}],
                    }
                ],
            }
        ],
    }


def test_extraction_evidence_and_value_are_checked():
    pages = {2: "公司营业收入 100 元"}
    assert validate_extraction(extraction(), "a", "a" * 64, pages).facts
    for change in ("page", "quote", "value", "id"):
        payload = deepcopy(extraction())
        if change == "page":
            payload["facts"][0]["evidence"][0]["page"] = 3
        if change == "quote":
            payload["facts"][0]["evidence"][0]["quote"] = "公司明天必涨百分之十"
        if change == "value":
            payload["facts"][0]["metrics"][0]["value"] = "10000"
        if change == "id":
            payload["document_id"] = "other"
        with pytest.raises(ValueError):
            validate_extraction(payload, "a", "a" * 64, pages)
    with pytest.raises(ValueError):
        prepare_request("a", "a" * 64, {1: "x" * 60001})


def test_internal_api_requires_token_and_bounds(monkeypatch):
    from app.api.dependencies import require_service_token
    from app.api.routes import fund_materials as routes
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    assert client.get("/002112/materials").status_code == 403
    app.dependency_overrides[require_service_token] = lambda: None
    monkeypatch.setattr(routes, "require_fund", lambda code: None)
    for query in ("pageSize=1000", "page=0", "stockCode=../private", "kind=bad"):
        assert client.get("/002112/materials/documents?" + query).status_code == 422
