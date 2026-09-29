"""跨页不能跳过正文、隔页借单位、吞掉不适用列或把百分比当金额。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_page_span_v1 import review_page_span_money_row
from tests.test_fund_earnings_scoped_table_v1 import fixture as same_page_fixture


def fixture():
    spec, _ = same_page_fixture()
    first = spec["unit_quote"] + "\n" + spec["header_quote"]
    second = spec["row_quote"]
    spec.pop("scope_quote")
    spec["page_segments"] = [{"page": 1, "quote": first}, {"page": 2, "quote": second}]
    return spec, [first, second]


def test_adjacent_table_keeps_real_pages_and_unknown_cell():
    spec, pages = fixture()
    result = review_page_span_money_row(spec, pages)
    assert result["anchor"]["page_span"] == [1, 2]
    assert result["anchor"]["segments"][1]["page"] == result["page"] == 2
    assert result["money"][0]["cny"] == "77190888.11"
    assert result["column_proof"]["all_cells"][2] is None
    assert result["available_at"] == "2023-04-29T08:00:00+08:00"
    assert not result["training_ready"]


@pytest.mark.parametrize("part,number", [(0, True), (0, 0), (1, 3), (1, 1), (1, "2")])
def test_invalid_and_nonadjacent_pages_rejected(part, number):
    spec, pages = fixture()
    spec["page_segments"][part]["page"] = number
    with pytest.raises(ValueError, match="PAGE_SPAN"):
        review_page_span_money_row(spec, pages)


@pytest.mark.parametrize("part,prefix", [(0, False), (1, True)])
def test_cannot_skip_page_edge_text(part, prefix):
    spec, pages = fixture()
    pages[part] = "被遗漏的新表说明\n" + pages[part] if prefix else pages[part] + "\n被遗漏的新表说明"
    with pytest.raises(ValueError, match="BOUNDARY_GAP"):
        review_page_span_money_row(spec, pages)


@pytest.mark.parametrize("unit", ["单位：万元", "万元"])
def test_second_unit_blocks_join(unit):
    spec, pages = fixture()
    pages[1] = unit + "\n" + pages[1]
    spec["page_segments"][1]["quote"] = pages[1]
    with pytest.raises(ValueError, match="UNIT_BOUNDARY"):
        review_page_span_money_row(spec, pages)


@pytest.mark.parametrize("change", ["drop_cell", "percent", "negative", "period", "unit", "duplicate"])
def test_money_scope_corruption_rejected(change):
    spec, pages = fixture()
    if change == "drop_cell":
        spec["cells"].pop()
    elif change == "percent":
        spec["selected_column"] = 2
    elif change == "negative":
        spec["cells"][1] = "232376308.45"
    elif change == "period":
        spec["selected_column"] = 1
    elif change == "unit":
        spec["unit"] = "万元"
    elif change == "duplicate":
        pages[1] += "\n" + pages[1]
    with pytest.raises(ValueError):
        review_page_span_money_row(spec, pages)


def test_standalone_unit_only_as_complete_line_in_same_page_header():
    spec, pages = fixture()
    spec["unit_quote"] = "元"
    spec["unit_form"] = "STANDALONE_HEADER_LINE"
    page = "元\n" + spec["header_quote"] + "\n" + pages[1]
    spec["page_segments"] = [{"page": 1, "quote": page}]
    assert review_page_span_money_row(spec, [page])["money"][0]["cny"] == "77190888.11"
    changed = deepcopy(spec)
    changed["page_segments"][0]["quote"] = page.replace("元\n", "元说明\n", 1)
    with pytest.raises(ValueError, match="STANDALONE_UNIT"):
        review_page_span_money_row(changed, [changed["page_segments"][0]["quote"]])


def test_standalone_unit_cannot_cross_page():
    spec, pages = fixture()
    spec["unit_quote"] = "元"
    spec["unit_form"] = "STANDALONE_HEADER_LINE"
    pages[0] = "元\n" + spec["header_quote"]
    spec["page_segments"][0]["quote"] = pages[0]
    with pytest.raises(ValueError, match="STANDALONE_UNIT"):
        review_page_span_money_row(spec, pages)
