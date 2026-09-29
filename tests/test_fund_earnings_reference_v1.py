"""确保原始预告的追索不会退化为修正公告追述值或同名异期报告。"""

import pytest
from app.services.fund_earnings_reference_v1 import reference_identity, select_reference

SPEC = {"stock": "300853", "published_date": "2023-01-20", "title": "2022年年度业绩预告", "notice_number": "2023-008"}
ROW = {"announcementId": "1", "secCode": "300853", "published_date": "2023-01-20", "title_plain": "2022年年度业绩预告"}


def test_exact_reference_body_passes():
    assert reference_identity(ROW, ["证券代码：300853 公告编号：2023-008\n2022年年度业绩预告"], SPEC)["passed"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("secCode", "300854"),
        ("published_date", "2023-04-17"),
        ("title_plain", "2022年年度业绩预告修正公告"),
        ("title_plain", "2023年年度业绩预告"),
    ],
)
def test_other_source_is_not_substitute(field, value):
    with pytest.raises(ValueError, match="NOT_UNIQUE"):
        select_reference([{**ROW, field: value}], SPEC)


def test_duplicate_catalog_matches_not_selected_by_id():
    with pytest.raises(ValueError, match="NOT_UNIQUE"):
        select_reference([ROW, {**ROW, "announcementId": "2"}], SPEC)


@pytest.mark.parametrize(
    "body",
    [
        "证券代码：300853 公告编号：2023-035\n2022年年度业绩预告",
        "证券代码：300854 公告编号：2023-008\n2022年年度业绩预告",
    ],
)
def test_body_number_and_issuer_both_required(body):
    with pytest.raises(ValueError, match="IDENTITY_UNRESOLVED"):
        reference_identity(ROW, [body], SPEC)
