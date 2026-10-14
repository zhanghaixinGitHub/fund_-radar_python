"""单位检查独立于模型回答，验证序号、百分比、混合单位都不发生暗中换算。"""

from scripts.local_fund_pdf_units import checked_grid, rebind_book


def source_packet(declaration):
    """单位声明和元数据同表、带百分号与明确股数表头，模拟真实公告版面。"""

    def cell(column, text, y):
        return {
            "column_start": column,
            "column_end": column,
            "row_start": 1 if y == 10 else 2,
            "row_end": 1 if y == 10 else 2,
            "bbox": [column * 10, y, (column + 1) * 10, y + 10],
            "text": text,
        }

    return {
        "version": "local-document-v3",
        "source_sha256": "fixture",
        "scope": "FULL_TEXT",
        "issues": [],
        "total_pages": 1,
        "pages": [
            {
                "page": 1,
                "character_coverage_ok": True,
                "blocks": [{"id": "B1", "text": declaration, "bbox": [0, 0, 100, 5]}],
                "tables": [
                    {
                        "bbox": [0, 10, 100, 40],
                        "rows": [
                            {
                                "id": "T1R1",
                                "kind": "table_row",
                                "page": 1,
                                "cells": [
                                    cell(i, t, 10)
                                    for i, t in enumerate(["序号", "认购金额", "获授数量（万股）", "实际占比"], 1)
                                ],
                            },
                            {
                                "id": "T1R2",
                                "kind": "table_row",
                                "page": 1,
                                "cells": [
                                    cell(i, t, 30) for i, t in enumerate(["1", "12,345.67", "222.0000", "42.21%"], 1)
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_table_currency_unit_cannot_turn_sequence_into_amount():
    facts = {f["column"]: f for f in checked_grid(source_packet("单位：万元"))["facts"] if f["row_id"] == "T1R2"}
    assert facts[1]["unit"] is None and facts[1]["status"] == "SOURCE_METADATA_NOT_AMOUNT"
    assert facts[2]["unit"] == "万元" and facts[2]["value"] == "12345.67"
    assert facts[3]["unit"] == "万股"
    assert facts[4]["unit"] == "%" and facts[4]["value"] == "42.21"


def test_mixed_unit_declaration_does_not_choose_first_unit():
    facts = {f["column"]: f for f in checked_grid(source_packet("单位：万元，万股"))["facts"] if f["row_id"] == "T1R2"}
    assert facts[2]["unit"] is None and facts[2]["status"] == "MIXED_UNITS_NOT_BOUND"
    assert facts[3]["unit"] == "万股"


def test_rebind_preserves_ids_quotes_and_first_book():
    original = {
        "facts": [
            {
                "id": "F0001",
                "kind": "pdf_table_cell",
                "data": {"row_id": "P1T1R2", "column": 1, "value": "1", "unit": "万元", "status": "BOUND_CELL"},
                "quote": "1",
            }
        ]
    }
    grid = {
        "version": "LOCAL_PDF_GRID_UNITS_2",
        "facts": [
            {"row_id": "P1T1R2", "column": 1, "value": "1", "unit": None, "status": "SOURCE_METADATA_NOT_AMOUNT"}
        ],
    }
    revised = rebind_book(original, grid)
    assert original["facts"][0]["data"]["unit"] == "万元"
    assert revised["facts"][0]["id"] == "F0001"
    assert revised["facts"][0]["quote"] == "1"
    assert revised["facts"][0]["data"]["unit"] is None


def test_percent_rebinding_keeps_original_quote():
    original = {
        "facts": [
            {
                "id": "F0002",
                "kind": "pdf_table_cell",
                "data": {"row_id": "P1T1R2", "column": 2, "value": None, "unit": "万元", "status": "NO_NUMERIC_SOURCE"},
                "quote": "42.21%",
            }
        ]
    }
    grid = {
        "version": "LOCAL_PDF_GRID_UNITS_2",
        "facts": [{"row_id": "P1T1R2", "column": 2, "value": "42.21", "unit": "%", "status": "BOUND_CELL"}],
    }
    assert rebind_book(original, grid)["facts"][0]["data"]["unit"] == "%"
