"""小范围公开净值核对必须拒绝错基金、越界、缺页和无效数值。"""

import pytest
from scripts.fund_002112_nav_source_crosscheck import parse_rows


def official(rows, count=None):
    return {"dataList": rows, "totalCount": len(rows) if count is None else count, "totalPage": 1}


def row(day="2023-09-20", fund="002112", nav="1.0269"):
    return {"date": day, "fundcode": fund, "netvalue": nav}


def test_official_exact_decimal_text():
    assert parse_rows("dbfund", official([row()]), "2023-09-18", "2023-09-28") == [
        {"date": "2023-09-20", "nav": "1.0269"}
    ]


@pytest.mark.parametrize(
    "data",
    [
        official([row(fund="160323")]),
        official([row(day="2025-01-02")]),
        official([row(day="2023-09-01")]),
        official([row()], count=2),
        official([row(), row()]),
        official([]),
        official([row(nav="NaN")]),
        official([row(nav="0")]),
    ],
)
def test_reject_invalid_official(data):
    with pytest.raises(ValueError):
        parse_rows("dbfund", data, "2023-09-18", "2023-09-28")


def test_secondary_page_completeness():
    data = {"Data": {"LSJZList": [{"FSRQ": "2023-09-20", "DWJZ": "1.0269"}]}, "TotalCount": 1, "PageIndex": 1}
    assert parse_rows("eastmoney", data, "2023-09-18", "2023-09-28")[0]["nav"] == "1.0269"
    data["TotalCount"] = 2
    with pytest.raises(ValueError, match="INCOMPLETE_SECONDARY_PAGE"):
        parse_rows("eastmoney", data, "2023-09-18", "2023-09-28")
