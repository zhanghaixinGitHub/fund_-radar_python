"""合计行真实合并跨度必须阻止把收益金额搬到投入列。"""

from scripts.local_fund_table_guard import table_claim_issues


def source():
    header = {
        "id": "H",
        "kind": "table_row",
        "cells": [
            {"column_start": c, "column_end": c, "text": label + "（万元）"}
            for c, label in ((3, "实际投入金额"), (4, "实际收回本金"), (5, "实际收益"), (6, "尚未收回本金金额"))
        ],
    }
    row = {
        "id": "R",
        "kind": "table_row",
        "header_rows": ["H"],
        "cells": [
            {"column_start": 1, "column_end": 4, "text": "合计"},
            {"column_start": 5, "column_end": 5, "text": "85.73"},
            {"column_start": 6, "column_end": 6, "text": "8,324.58"},
        ],
    }
    return row, [header, row]


def test_amount_in_another_column_does_not_support_an_empty_total():
    row, units = source()
    issues = table_claim_issues("实际投入金额85.73万元，实际收益85.73万元。", [row], units)
    assert len(issues) == 1
    assert issues[0]["code"] == "TABLE_CLAIM_HAS_NO_NUMERIC_CELL"
    assert issues[0]["column"] == 3
    assert issues[0]["source_cell"] is None


def test_correct_column_amounts_and_declared_absence_are_kept():
    row, units = source()
    assert (
        table_claim_issues("实际收益85.73万元，尚未收回本金8,324.58万元。实际收回本金未单独列出。", [row], units) == []
    )


def test_an_amount_with_wrong_unit_is_not_equal_even_if_digits_match():
    row, units = source()
    assert table_claim_issues("实际收益85.73元。", [row], units)[0]["code"] == "TABLE_UNIT_MISSING_OR_MISMATCH"


def test_multiple_period_rows_do_not_pick_a_convenient_match():
    row, units = source()
    another = {**row, "id": "R2"}
    issues = table_claim_issues("实际收益85.73万元。", [row, another], [*units, another])
    assert issues[0]["code"] == "TABLE_ROW_BINDING_NOT_UNIQUE"
