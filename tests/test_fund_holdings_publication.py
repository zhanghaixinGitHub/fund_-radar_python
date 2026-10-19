"""披露持仓补录的身份、精度、历史保护与逐基金失败隔离。"""

import hashlib
import json
from decimal import Decimal

import pytest
from app.services import fund_holdings_publication as service
from app.services.fund_exposure_common import read, save


def test_catalog_abbreviations_do_not_break_product_identity():
    assert service._fund_name_matches(
        "广发国证新能源车电池交易型开放式指数证券投资基金发起式联接基金", "广发国证新能源车电池ETF联接-C"
    )
    assert service._fund_name_matches("亚债中国债券指数基金 基金管理人：华夏基金管理有限公司", "华夏亚债中国债券指数-A")
    assert not service._fund_name_matches("其他产品 基金管理人：华夏基金管理有限公司", "华夏亚债中国债券指数-A")


def test_multiline_bond_name_and_nav_weight_are_checked():
    rows = service._security_rows("1 260010 26附息国债\n10 3,500,000 350,593,858.70 2.84", Decimal("12345000000"))
    assert rows[0]["stockName"] == "26附息国债 10"
    assert rows[0]["marketValue"] == 350593858.70
    with pytest.raises(ValueError, match="WEIGHT"):
        service._security_rows("1 260010 国债 100 350,593,858.70 50.00", Decimal("12345000000"))


def table(tmp_path, *, period="2026-06-30", amount="10.00"):
    html = (
        "<div class='boxitem'><h4><a href='https://fund.eastmoney.com/008888.html'>基金</a></h4>"
        f"<p class='t'><font>{period}</font></p><table><thead><tr><th>代码</th><th>万元人民币</th>"
        "</tr></thead><tbody><tr><td>1</td><td>000001</td><td>股票甲</td>"
        f"<td>10.00%</td><td>{amount}</td></tr></tbody></table></div>"
    )
    raw = ("var apidata={content:" + json.dumps(html) + "};").encode()
    path = tmp_path / "table.txt"
    path.write_bytes(raw)
    return {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "fund_code": "008888", "kind": "stocks"}


def test_public_table_requires_correct_period_identity_and_full_amount(tmp_path):
    ref = table(tmp_path)
    assert service._public_table(tmp_path, ref, "008888", "2026-06-30", Decimal(100000), Decimal(1000000))
    with pytest.raises(ValueError, match="PERIOD_OR_FUND"):
        service._public_table(tmp_path, ref, "008888", "2026-03-31", Decimal(100000), Decimal(1000000))
    with pytest.raises(ValueError, match="IDENTITY"):
        service._public_table(tmp_path, ref, "001021", "2026-06-30", Decimal(100000), Decimal(1000000))
    with pytest.raises(ValueError, match="INCOMPLETE"):
        service._public_table(tmp_path, ref, "008888", "2026-06-30", Decimal(200000), Decimal(1000000))


def test_source_rounding_allows_only_the_reported_precision(tmp_path):
    ref = table(tmp_path)
    service._public_table(tmp_path, ref, "008888", "2026-06-30", Decimal("100049"), Decimal(1000000))
    with pytest.raises(ValueError, match="INCOMPLETE"):
        service._public_table(tmp_path, ref, "008888", "2026-06-30", Decimal("100051"), Decimal(1000000))


def test_publication_keeps_original_reports_and_isolates_failures(tmp_path, monkeypatch):
    report = {
        "id": "old",
        "endDate": "2026-06-30",
        "publishedDate": "2026-08-31",
        "fullDisclosure": True,
        "holdings": [{"stockName": "原件完整表"}],
        "investmentGroups": [],
    }
    previous = {
        "fundCode": "008888",
        "asOfDate": "2026-09-30",
        "reports": [report],
        "companies": [{"stockCode": "000001.SZ"}],
        "documents": [{"id": "historical"}],
    }
    save(tmp_path / "008888.json", previous)

    def source(code, *args):
        if code == "001021":
            raise ValueError("HOLDINGS_ORIGINAL_HASH_OR_URL")
        return {**report, "holdings": [], "holdingsPublicationVersion": 1}, []

    monkeypatch.setattr(service, "report_from_original", source)
    manifest = {
        "funds": [
            {"fund_code": c, "fund_name": "测试", "source_root": str(tmp_path), "source": {}}
            for c in ("001021", "008888")
        ]
    }
    result = service.publish(manifest, directory=tmp_path)
    assert [f["status"] for f in result["funds"]] == ["FAILED", "SUCCEEDED"]
    current = read(tmp_path / "008888.json")
    assert current["reports"] == previous["reports"]
    assert current["companies"] == previous["companies"] and current["documents"] == previous["documents"]
    assert current["asOfDate"] == "2026-09-30"
    assert not (tmp_path / "001021.json").exists()


def test_dry_run_never_writes_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        service,
        "report_from_original",
        lambda *args: (
            {
                "id": "r",
                "endDate": "2026-06-30",
                "publishedDate": "2026-08-31",
                "fullDisclosure": False,
                "holdings": [],
                "investmentGroups": [],
            },
            [{"part": "report"}],
        ),
    )
    result = service.publish(
        {"funds": [{"fund_code": "008888", "fund_name": "测试", "source_root": str(tmp_path), "source": {}}]},
        directory=tmp_path,
        dry_run=True,
    )
    assert result["funds"][0]["status"] == "PARTIAL_SUCCESS"
    assert not list(tmp_path.iterdir())


def test_new_period_same_period_report_and_revision_keep_all_history(tmp_path, monkeypatch):
    old = {
        "id": "annual-original",
        "endDate": "2025-12-31",
        "publishedDate": "2026-03-31",
        "fullDisclosure": True,
        "holdings": [{"stockName": "旧持仓"}],
        "investmentGroups": [],
    }
    original = {"fundCode": "008888", "asOfDate": "2026-03-31", "reports": [old], "companies": [], "documents": []}
    save(tmp_path / "008888.json", original)
    incoming = [
        {**old, "id": "half-original", "endDate": "2026-06-30", "publishedDate": "2026-08-31"},
        {**old, "id": "q2-original", "endDate": "2026-06-30", "publishedDate": "2026-07-20", "fullDisclosure": False},
        {**old, "id": "annual-correction", "publishedDate": "2026-04-01"},
    ]
    manifest = {"funds": [{"fund_code": "008888", "fund_name": "测试", "source_root": str(tmp_path), "source": {}}]}
    for report in incoming:
        monkeypatch.setattr(service, "report_from_original", lambda *args, report=report: (report, []))
        assert service.publish(manifest, directory=tmp_path)["funds"][0]["status"] == "SUCCEEDED"
    current = read(tmp_path / "008888.json")
    assert {r["id"] for r in current["reports"]} == {
        "annual-original",
        "annual-correction",
        "half-original",
        "q2-original",
    }
    assert current["reports"][0]["id"] == "half-original"
    assert current["asOfDate"] == "2026-08-31"
    assert current["reports"][-1] == old
    archived = [read(p)["value"] for p in (tmp_path / "versions").glob("*.json")]
    assert original in archived and current in archived
