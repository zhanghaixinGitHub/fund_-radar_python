"""按PDF真实表头和单列单元格独立核对金额；不从模型解释或合并格猜值。

只核对能定位到唯一已引用数据行的表述。多表、多行、计算值和无法唯一确认的
列关系交由复核。规则不修改原回答、不填补空值，更不能代替完整语义验收。
"""

import re
from decimal import Decimal

from scripts.local_fund_semantics import compact


def table_claim_issues(answer: str, selected: list[dict], units: list[dict]) -> list[dict]:
    """将明确的“列名+金额+单位”与该列原格核对，表头来源与行编号必须真实。

    合计横跨多列但没有数值时，不把另一列金额移过来，也不汇总分项补填。
    同字段出现在多个已引用行时拒绝自动判定，以免挑中碰巧匹配的一行。
    """
    catalog = {unit["id"]: unit for unit in units}
    bindings = {}
    for row in selected:
        if row.get("kind") != "table_row" or row["id"] in row.get("header_rows", []):
            continue
        headers = [catalog[ref] for ref in row.get("header_rows", []) if ref in catalog]
        for header in headers:
            for cell in header["cells"]:
                if cell["column_start"] != cell["column_end"]:
                    continue
                name = compact(cell["text"])
                unit_match = re.search(r"[（(](亿元|万元|元|亿股|万股|股)[）)]", name)
                if not unit_match:
                    continue
                label = name[: unit_match.start()]
                column = cell["column_start"]
                data_cells = [c for c in row["cells"] if c["column_start"] == c["column_end"] == column]
                # 来源格不是唯一单列格，或只是合并格覆盖这一列，都没有可采用的数值。
                raw = data_cells[0]["text"] if len(data_cells) == 1 else None
                numeric = compact(raw).replace(",", "") if raw is not None else ""
                expected = Decimal(numeric) if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", numeric) else None
                for alias in {label, label.removesuffix("金额")}:
                    if alias:
                        bindings.setdefault(alias, []).append(
                            {
                                "row_id": row["id"],
                                "header_id": header["id"],
                                "column": column,
                                "raw": raw,
                                "expected": expected,
                                "unit": unit_match[1],
                            }
                        )
    text, issues = compact(answer), []
    used = set()
    for label in sorted(bindings, key=len, reverse=True):
        pattern = re.escape(label) + r"(?:为|是|：|:|合计|约|共)?([\d,]+(?:\.\d+)?)(亿元|万元|元|亿股|万股|股)?"
        for match in re.finditer(pattern, text):
            span = (match.start(1), match.end(1))
            if span in used:
                continue
            used.add(span)
            rows = bindings[label]
            unique = {(r["row_id"], r["column"]): r for r in rows}
            if len(unique) != 1:
                issues.append({"code": "TABLE_ROW_BINDING_NOT_UNIQUE", "label": label, "raw_claim": match[0]})
                continue
            source = next(iter(unique.values()))
            base = {
                "label": label,
                "raw_claim": match[0],
                "row_id": source["row_id"],
                "header_id": source["header_id"],
                "column": source["column"],
                "source_cell": source["raw"],
                "source_unit": source["unit"],
            }
            if source["expected"] is None:
                issues.append({"code": "TABLE_CLAIM_HAS_NO_NUMERIC_CELL", **base})
            elif Decimal(match[1].replace(",", "")) != source["expected"]:
                issues.append({"code": "TABLE_AMOUNT_COLUMN_MISMATCH", **base})
            if match[2] != source["unit"]:
                issues.append({"code": "TABLE_UNIT_MISSING_OR_MISMATCH", **base})
    return issues
