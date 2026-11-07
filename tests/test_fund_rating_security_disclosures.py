"""退市确权、同一期细行业及数字简称边界；使用隔离合成原件。"""

from copy import deepcopy

import pytest
from app.services.fund_rating_public_stock_tables import cells, parse
from app.services.fund_rating_security_disclosures import verified_securities


@pytest.fixture
def case():
    class Source:
        sources = {key: {"published_on": "2026-08-31", "archive_kind": "ORIGINAL"} for key in ["report", "transfer"]}
        raw = {
            "transfer": ("<table><tr><td>隔离地产股份有限公司</td><td>000999</td><td>400999</td></tr></table>").encode()
        }
        report = (
            "报告期指2026年1月1日-2026年6月30日"
            "公司中文全称隔离地产股份有限公司英文名称TEST"
            "行业（挂牌公司管理型行业分类）房地产业（K）-房地产业（K70）-房地产开发（K701）"
            "主要产品与服务项目房地产股票交易场所两网公司及退市公司板块"
            "证券简称隔离3证券代码400999"
        )

        def pages(self, _key):
            return [self.report]

        def json(self, _key):
            return {"stockList": [{"code": "000999", "orgId": "original-registry-id"}]}

    inputs = {
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "issuer_registry": "registry",
        "security_disclosures": {"400999": {"report": "report", "transfer": "transfer", "original_code": "000999"}},
    }
    return Source(), inputs


def test_same_period_company_and_one_transfer_row_recover_actual_registry_identity(case):
    source, inputs = case
    result = verified_securities(source, inputs)["400999"]
    assert result["issuer"] == "original-registry-id"
    assert result["industry_code"] == "K70" and result["market"] == ".NQ" and result["name"] == "隔离3"


@pytest.mark.parametrize(
    "change,error",
    [
        ("period", "SAME_PERIOD"),
        ("code", "REPORT_IDENTITY"),
        ("industry", "FINE_INDUSTRY"),
        ("future", "CUTOFF"),
        ("derived", "ORIGINAL"),
        ("split_row", "TRANSFER_IDENTITY"),
        ("wrong_legal", "TRANSFER_IDENTITY"),
    ],
)
def test_missing_or_conflicting_security_evidence_never_gets_a_default(case, change, error):
    source, inputs = deepcopy(case)
    if change == "period":
        source.report = source.report.replace("2026年1月1日", "2025年1月1日")
    elif change == "code":
        source.report = source.report.replace("证券代码400999", "证券代码400998")
    elif change == "industry":
        source.report = source.report.replace("房地产业（K70）-", "")
    elif change == "future":
        source.sources["report"]["published_on"] = "2026-10-01"
    elif change == "derived":
        source.sources["report"]["archive_kind"] = "DERIVED"
    elif change == "split_row":
        source.raw["transfer"] = source.raw["transfer"].replace(b"<td>400999", b"</tr><tr><td>400999")
    elif change == "wrong_legal":
        source.raw["transfer"] = source.raw["transfer"].replace("隔离地产".encode(), "另一地产".encode())
    with pytest.raises(ValueError, match=error):
        verified_securities(source, inputs)


def test_verified_numeric_name_does_not_inflate_share_quantity_or_assign_beijing_market():
    name, values = cells(["隔离 3 95,400 13,356.00 0.00"], verified_name="隔离3")
    assert name == "隔离3" and values == ["95,400", "13,356.00", "0.00"]
    with pytest.raises(ValueError, match="NUMERIC_NAME_PROOF_REQUIRED"):
        cells(["隔离 3 95,400 13,356.00 0.00"])
    text = (
        "7.2 期末按行业分类的股票投资组合\nK 房地产业 13,356.00 0.00\n"
        "7.3 期末所有股票投资明细\n1 400999 隔离 3 95,400 13,356.00 0.00\n7.4 其他"
    )
    result = parse([text], "13356", "1000000000", verified_securities={"400999": {"name": "隔离3", "market": ".NQ"}})
    row = result["holdings"][0]
    assert row["quantity_shares"] == "95400" and row["stock_code"] == "400999.NQ"
