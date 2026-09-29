"""保护事实含义、历史时间、分页和不可退款的公开读取预算。"""

from datetime import datetime
from decimal import Decimal

import httpx
import pytest
from app.services.fund_information_history_v1 import (
    QUERY,
    ZONE,
    BoundedPublicReader,
    anchor,
    buyback_candidates,
    catalog_rows,
    classify,
    covered,
    explicit_money,
    pdf_revision,
    read,
    save,
)


def item(index=1, **changes):
    return {
        "announcementId": str(index),
        "secCode": "000651",
        "announcementTitle": "2020年度业绩快报",
        "announcementTime": int(datetime(2021, 1, 6, tzinfo=ZONE).timestamp() * 1000),
        **changes,
    }


@pytest.mark.parametrize(
    ("title", "stage"),
    [
        ("2020年度业绩预告", "FORECAST"),
        ("2020年度业绩快报", "PRELIMINARY_RESULT"),
        ("2020年年度报告", "REPORTED_RESULT"),
        ("关于签订重大合同的更正公告", "CORRECTION"),
        ("关于签订重大合同的核查意见", "SUPPORTING_DOCUMENT"),
        ("股份回购方案", "PLAN_OR_APPROVAL"),
        ("关于首次回购股份的公告", "PROGRESS"),
    ],
)
def test_distinct_stages(title, stage):
    assert classify(title)["stage"] == stage


@pytest.mark.parametrize("title", ["回购限制性股票", "回购事项前十名股东持股情况", "债券回购公告"])
def test_buyback_non_execution_excluded(title):
    assert classify(title)["category"] is None


def test_catalog_uses_total_not_broken_totalpages():
    pages = [
        (1, {"totalAnnouncement": 31, "totalpages": 1, "announcements": [item(i) for i in range(30)]}),
        (2, {"totalAnnouncement": 31, "totalpages": 1, "announcements": [item(30)]}),
    ]
    assert len(catalog_rows(pages, "000651", "2021-01-01", "2021-01-31")) == 31
    with pytest.raises(ValueError, match="MISSING_OR_DUPLICATE_PAGE"):
        catalog_rows(pages[:1], "000651", "2021-01-01", "2021-01-31")


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ([item(), item()], "DUPLICATE_ANNOUNCEMENT"),
        ([item(secCode="600519")], "WRONG_COMPANY"),
        ([item(announcementTime=0)], "DATE_OUTSIDE_WINDOW"),
    ],
)
def test_bad_catalog_blocks(rows, error):
    with pytest.raises(ValueError, match=error):
        catalog_rows(
            [(1, {"totalAnnouncement": len(rows), "announcements": rows})], "000651", "2021-01-01", "2021-01-31"
        )


def test_verified_empty_and_gap_not_equivalent():
    assert (
        catalog_rows([(1, {"totalAnnouncement": 0, "announcements": None})], "000651", "2021-01-01", "2021-01-31") == []
    )
    assert not covered([["2021-01-01", "2021-01-15"], ["2021-01-17", "2021-01-31"]], "2021-01-01", "2021-01-31")


def test_cumulative_payment_separate_from_plan_and_price():
    page = (
        "计划资金不低于人民币30亿元且不超过人民币60亿元。"
        "截至2020年12月31日，公司累计回购公司股份94,184,662股，最高成交价为57.00元/股，"
        "支付的总金额为5,181,586,503.65元（不含交易费用）。"
    )
    candidates = buyback_candidates([page])
    assert len(candidates) == 1
    assert candidates[0]["amount"]["cny"] == "5181586503.65"
    assert candidates[0]["as_of"] == "2020-12-31"
    assert candidates[0]["basis"] == "CUMULATIVE_EXECUTED_NOT_ADDITIVE"
    assert candidates[0]["fee_basis"] == "EXCLUDES_FEES"


@pytest.mark.parametrize(
    "page",
    [
        "截至2020年12月31日，拟回购累计金额不超过人民币60亿元。",
        "截至2020年12月31日，公司回购支付金额为10万元。",  # 未明确累计
        "截至2020年12月31日，公司累计回购支付金额为10。",  # 单位缺失
        "截至2020年12月31日，公司累计回购支付金额为10元/股。",  # 价格
        "截至2020年12月31日，公司累计回购0股。",  # 不补金额零值
    ],
)
def test_ambiguous_quantity_not_filled(page):
    assert buyback_candidates([page]) == []


def test_units_and_anchor_are_exact():
    assert Decimal(explicit_money("6.77", "亿元")["cny"]) == Decimal("677000000")
    with pytest.raises(ValueError):
        explicit_money("10", "USD")
    assert anchor(["合同总额为 6.77 亿元。"], "总额为6.77亿元")["page"] == 1
    with pytest.raises(ValueError):
        anchor(["总额6.77亿元", "总额6.77亿元"], "总额6.77亿元")


def test_later_pdf_metadata_is_unresolved_not_backdated():
    assert pdf_revision({"ModDate": "D:20240331090000+08'00'"}, "2024-03-30")
    assert pdf_revision({}, "2024-03-30") == []  # 无元数据不额外要求历史下载证明


def test_immutable_evidence(tmp_path):
    path = tmp_path / "decision.json"
    save(path, {"n": 1})
    save(path, {"n": 1})
    with pytest.raises(ValueError, match="IMMUTABLE"):
        save(path, {"n": 2})


def test_failed_request_consumes_budget_and_cannot_retry(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(503)

    reader = BoundedPublicReader(tmp_path, {"company": 1, "public": 1})
    reader.client.close()
    reader.client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="PUBLIC_READ_STOP"):
        reader.fetch(QUERY, params={"pageNum": 1})
    with pytest.raises(ValueError, match="PREVIOUS_PUBLIC_REQUEST_FAILED"):
        reader.fetch(QUERY, params={"pageNum": 1})
    with pytest.raises(ValueError, match="PUBLIC_REQUEST_LIMIT"):
        reader.fetch(QUERY, params={"pageNum": 2})
    reader.close()
    assert len(calls) == 1
    assert len(list((tmp_path / "requests").glob("*.json"))) == 1


def test_successful_receipt_reused_with_hash_check(tmp_path):
    reader = BoundedPublicReader(tmp_path, {"company": 1, "public": 1})
    reader.client.close()
    reader.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"ok")))
    raw, receipt = reader.fetch(QUERY, params={"pageNum": 1})
    assert raw == b"ok"
    assert reader.fetch(QUERY, params={"pageNum": 1})[0] == b"ok"
    from pathlib import Path

    Path(receipt["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="PUBLIC_RAW_CHANGED"):
        reader.fetch(QUERY, params={"pageNum": 1})
    reader.close()
    assert read(next((tmp_path / "requests").glob("*.json")))["group"] == "company"


def test_official_pagination_lookahead_must_match_exactly():
    from scripts.fund_002112_public_history_v1 import append_group

    def rows(start, stop):
        return [{"url": str(i), "display_date": "2023-01-01", "title": str(i)} for i in range(start, stop)]

    result = []
    append_group(result, rows(1, 46), 0, 100)
    append_group(result, rows(46, 92), 1, 100)
    append_group(result, rows(91, 101), 2, 100)
    assert len(result) == 100
    result = rows(1, 92)
    wrong = rows(91, 101)
    wrong[0]["display_date"] = "2023-01-02"
    with pytest.raises(ValueError, match="BOUNDARY_NOT_IDENTICAL"):
        append_group(result, wrong, 2, 100)


def test_unexplained_duplicate_public_urls_stop_not_silently_merge():
    from scripts.fund_002112_public_history_v1 import append_group

    rows = [{"url": str(i), "title": str(i)} for i in range(45)]
    rows[-1]["url"] = rows[0]["url"]
    with pytest.raises(ValueError, match="UNIQUENESS"):
        append_group([], rows, 0, 45)


def test_public_display_date_not_inferred_from_url():
    from scripts.fund_002112_public_history_v1 import records

    page = (
        '<record><![CDATA[<li><a href="/art/2021/1/1/abc.html" title="公告">公告</a>'
        "<span>2023-01-02</span></li>]]></record>"
    )
    assert records(page)[0]["display_date"] == "2023-01-02"
    with pytest.raises(ValueError, match="DISPLAY_DATE"):
        records(page.replace("<span>2023-01-02</span>", ""))


def test_review_does_not_promote_unclear_currency_or_missing_anchor():
    from scripts.fund_002112_fact_review_v1 import replay_field

    field = {
        "name": "contract_total",
        "value": "20",
        "unit": "USD",
        "basis": "ESTIMATE",
        "quote": "合同金额20美元",
        "page": 1,
    }
    with pytest.raises(ValueError, match="UNKNOWN_MONEY"):
        replay_field(field, ["合同金额20美元"])
    field.update(unit="亿元", quote="合同金额20亿元")
    with pytest.raises(ValueError, match="ANCHOR"):
        replay_field(field, ["合同金额未披露"])


def test_million_unit_not_ten_thousand_unit():
    assert Decimal(explicit_money("28928", "百万元")["cny"]) == Decimal("28928000000")
