"""持仓构成公共资料回归；使用合成快照，不读取个人账户或外部服务。"""

import pytest
from app.services import fund_composition


def report(end: str, **values) -> dict:
    return {
        "id": end,
        "title": f"测试基金{end}报告",
        "endDate": end,
        "publishedDate": end,
        "sourceUrl": "https://example.com/report.pdf",
        "fullDisclosure": False,
        "holdings": [],
        "assets": [],
        "industries": [],
        "investmentGroups": [],
        **values,
    }


def test_assets_and_industries_use_latest_verified_report_without_mixing_denominators(monkeypatch):
    data = {
        "fundCode": "000001",
        "reports": [
            report(
                "2026-09-30",
                assets=[
                    {"name": "权益投资", "weightPct": 60},
                    {"name": "固定收益投资", "weightPct": 20},
                    {"name": "基金", "weightPct": 10},
                    {"name": "银行存款和结算备付金合计", "weightPct": 5},
                    {"name": "其他资产", "weightPct": 5},
                ],
                industries=[{"name": "制造业", "weightPct": 55}],
                stockWeightPct=60,
            ),
            report("2026-06-30", assets=[{"name": "股票", "weightPct": 100}]),
        ],
    }
    monkeypatch.setattr(fund_composition, "snapshot", lambda code: data)
    monkeypatch.setattr(fund_composition, "_chain", lambda code: ([{"fundCode": code, "fundName": "测试"}], None))
    result = fund_composition.batch(["000001"])["funds"][0]
    assert result["assets"]["endDate"] == "2026-09-30"
    assert [(row["key"], row["weightPct"]) for row in result["assets"]["items"]] == [
        ("STOCK", 60),
        ("BOND", 20),
        ("FUND", 10),
        ("CASH_OTHER", 10),
    ]
    assert result["industries"]["stockWeightPct"] == 60
    assert result["industries"]["items"][0]["weightPct"] == 55


def test_invalid_asset_total_is_unknown_instead_of_normalized(monkeypatch):
    data = {"fundCode": "000001", "reports": [report("2026-09-30", assets=[{"name": "股票", "weightPct": 80}])]}
    monkeypatch.setattr(fund_composition, "snapshot", lambda code: data)
    monkeypatch.setattr(fund_composition, "_chain", lambda code: ([{"fundCode": code}], None))
    assert fund_composition.batch(["000001"])["funds"][0]["assets"] is None


def test_theme_lookthrough_requires_identity_and_multiplies_effective_weights(monkeypatch):
    target_name = "华夏国证半导体芯片交易型开放式指数证券投资基金"
    parent = {
        "fundCode": "000001",
        "reports": [
            report(
                "2026-09-30",
                holdings=[{"stockCode": "300394.SZ", "stockName": "天孚通信", "weightPct": 10}],
                investmentGroups=[
                    {"name": "基金持仓", "holdings": [{"stockCode": "", "stockName": target_name, "weightPct": 80}]}
                ],
            )
        ],
    }
    target = {
        "fundCode": "159995",
        "reports": [
            report(
                "2026-09-30",
                holdings=[{"stockCode": "688256.SH", "stockName": "寒武纪", "weightPct": 50}],
            )
        ],
    }
    snapshots = {"000001": parent, "159995": target}
    monkeypatch.setattr(fund_composition, "snapshot", lambda code: snapshots[code])
    monkeypatch.setattr(
        fund_composition,
        "_chain",
        lambda code: ([{"fundCode": code, "fundName": "联接"}, {"fundCode": "159995", "fundName": target_name}], None),
    )
    result = fund_composition.batch(["000001"])["funds"][0]["themes"]
    assert result["lookthrough"] is True
    assert result["fullDisclosure"] is False
    assert result["disclosedWeightPct"] == pytest.approx(50)
    assert [(row["stockCode"], row["weightPct"], row["themes"][0]["id"]) for row in result["holdings"]] == [
        ("300394.SZ", 10, "COMMUNICATION"),
        ("688256.SH", 40, "CHIP"),
    ]
    assert result["sources"][1]["fundCode"] == "159995"


def test_theme_lookthrough_does_not_use_same_name_or_wrong_market_security(monkeypatch):
    parent = {
        "fundCode": "000001",
        "reports": [
            report(
                "2026-09-30",
                holdings=[{"stockCode": "300394.SZ", "stockName": "名称被替换", "weightPct": 10}],
            )
        ],
    }
    monkeypatch.setattr(fund_composition, "snapshot", lambda code: parent)
    monkeypatch.setattr(fund_composition, "_chain", lambda code: ([{"fundCode": code}], None))
    result = fund_composition.batch(["000001"])["funds"][0]["themes"]
    assert result["holdings"][0]["themes"] == []


def test_batch_rejects_duplicate_or_invalid_codes():
    for codes in ([], ["000001", "000001"], ["../000001"]):
        with pytest.raises(ValueError):
            fund_composition.batch(codes)


def test_internal_composition_api_requires_token_and_validates_codes(monkeypatch):
    from app.api.dependencies import require_service_token
    from app.api.routes import fund_materials
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(fund_materials.router)
    client = TestClient(app)
    assert client.get("/composition?fundCodes=000001").status_code == 403
    app.dependency_overrides[require_service_token] = lambda: None
    monkeypatch.setattr(fund_composition, "batch", lambda codes: {"version": 1, "funds": codes, "themeDefinitions": []})
    assert client.get("/composition?fundCodes=000001,000002").status_code == 200
    assert client.get("/composition?fundCodes=../000001").status_code == 422
