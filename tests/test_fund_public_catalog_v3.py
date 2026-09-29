"""用官网实际出现的同日同链接不同媒体标题情形验证目录与新闻去重边界。"""

import pytest
from app.services.fund_public_catalog_v3 import append_catalog


def row(number):
    return {"url": f"https://example.org/{number}", "display_date": "2021-07-27", "title": str(number)}


def test_different_media_titles_same_url_are_two_catalog_records():
    first = {**row(1), "title": "【央视影音】国家医保局 深化医保服务最多跑一次改革"}
    second = {**first, "title": "【央视新闻】国家医保局 深化医保服务最多跑一次改革"}
    values = []
    append_catalog(values, [first, second], 0, 2)
    assert len(values) == 2
    assert len({r["url"] for r in values}) == 1


def test_identical_source_record_is_not_an_alias():
    with pytest.raises(ValueError, match="DUPLICATE_RECORD"):
        append_catalog([], [row(1), row(1)], 0, 2)


def test_only_exact_observed_lookahead_is_accepted():
    values = []
    append_catalog(values, [row(i) for i in range(45)], 0, 135)
    append_catalog(values, [row(i) for i in range(45, 91)], 1, 135)
    append_catalog(values, [row(i) for i in range(90, 135)], 2, 135)
    assert len(values) == 135


def test_changed_lookahead_title_blocks_snapshot():
    values = [row(i) for i in range(91)]
    rows = [row(i) for i in range(90, 135)]
    rows[0] = {**rows[0], "title": "changed"}
    with pytest.raises(ValueError, match="BOUNDARY_CHANGED"):
        append_catalog(values, rows, 2, 135)
