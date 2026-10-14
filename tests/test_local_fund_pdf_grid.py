"""PDF网格按几何保留空列和合并格；不能从相邻金额补齐空白。"""

from scripts.local_fund_pdf_grid import column_grid, enrich_packet, preceding_unit


def cell(start, end, text, y=0):
    return {
        "column_start": start,
        "column_end": end,
        "row_start": y + 1,
        "row_end": y + 1,
        "text": text,
        "bbox": [start * 10, y * 10, (end + 1) * 10, (y + 1) * 10],
    }


def packet():
    headers = [
        "序号",
        "类型",
        "实际投入金额（万元）",
        "实际收回本金（万元）",
        "实际收益（万元）",
        "尚未收回本金金额（万元）",
    ]
    return {
        "version": "local-document-v3",
        "source_sha256": "test",
        "scope": "FULL_TEXT",
        "issues": [],
        "total_pages": 1,
        "pages": [
            {
                "page": 1,
                "character_coverage_ok": True,
                "blocks": [],
                "tables": [
                    {
                        "bbox": [0, 0, 70, 30],
                        "rows": [
                            {
                                "id": "T1R1",
                                "kind": "table_row",
                                "page": 1,
                                "cells": [cell(i, i, h) for i, h in enumerate(headers, 1)],
                            },
                            {
                                "id": "T1R2",
                                "kind": "table_row",
                                "page": 1,
                                "cells": [cell(1, 4, "合计", 1), cell(5, 5, "85.73", 1), cell(6, 6, "8,324.58", 1)],
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_real_merged_total_does_not_shift_amounts_to_empty_columns():
    grid = enrich_packet(packet())
    totals = {f["column"]: f for f in grid["facts"] if f["is_total"]}
    assert totals[3]["value"] is None and totals[4]["value"] is None
    assert totals[5]["value"] == "85.73" and totals[5]["label"] == "实际收益"
    assert totals[6]["value"] == "8324.58" and totals[6]["label"] == "尚未收回本金金额"
    assert "原合并格C1-C4" in grid["rendered"]


def test_ordinary_empty_cell_never_inherits_neighbour():
    row = {"id": "row", "cells": [cell(1, 1, ""), cell(2, 2, "100")], "context_cells": []}
    slots = column_grid(row, 3)
    assert slots[0]["text"] == ""
    assert slots[1]["text"] == "100"
    assert slots[2]["status"] == "EMPTY"


def test_overlapping_geometry_is_ambiguous_instead_of_chosen():
    row = {"id": "row", "cells": [cell(1, 2, "合并"), cell(2, 2, "100")], "context_cells": []}
    assert column_grid(row, 2)[1]["status"] == "AMBIGUOUS_GEOMETRY"


def test_unit_only_inherits_explicit_nearby_declaration_not_another_table():
    table = {"bbox": [0, 50, 100, 100]}
    page = {"blocks": [{"id": "B1", "text": "单位：万元", "bbox": [0, 10, 100, 20]}], "tables": [table]}
    assert preceding_unit(page, table)["unit"] == "万元"
    page["tables"].append({"bbox": [0, 25, 100, 40]})
    assert preceding_unit(page, table) is None
