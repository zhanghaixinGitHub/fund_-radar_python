"""从公开转载保留的表头及逐行数字片段恢复受限股票表，不猜缺失数字。

仅支持表头同时明确代码、名称、认购、价格、数量和成本/估值的排版。每行
金融片段必须按原顺序使用一次，且五列金额/数量只有唯一合法组合并同时满足
认购单价×股数=成本、估值单价×股数=估值，才接纳。无解或多解均拒收。
"""

import itertools
import re
from datetime import date
from decimal import Decimal

from app.services.fund_rating_public_market import compact
from app.services.fund_rating_rules import decimal

NUMBER_PREFIX = re.compile(r"(?:\d+|\d{1,3}(?:,\d{3})*(?:,\d{0,2})?)(?:\.\d{0,2})?\Z")
MONEY = re.compile(r"(?:\d+|\d{1,3}(?:,\d{3})+)\.\d{2}\Z")
QUANTITY = re.compile(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.00)?\Z")
FRAGMENT = re.compile(r"(?:[0-9][0-9,.]*|\.\d+)\Z")


def _unique_financial_cells(lines: list[list[str]], *, quantity_limit: Decimal, fair_limit: Decimal) -> tuple:
    """五列有序片段的有界穷举；唯一性来自原文金额等式，不是统计拟合。"""
    states = {("", "", "", "", "")}
    for fragments in lines:
        if len(fragments) > 5:
            raise ValueError("QUALITY_RESTRICTED_LAYOUT_TOO_MANY_FINANCIAL_COLUMNS")
        following = set()
        for columns in itertools.combinations(range(5), len(fragments)):
            for state in states:
                value = list(state)
                for column, fragment in zip(columns, fragments, strict=True):
                    value[column] += fragment
                if all(not cell or len(cell) <= 36 and NUMBER_PREFIX.fullmatch(cell) for cell in value):
                    following.add(tuple(value))
        if len(following) > 50_000:
            raise ValueError("QUALITY_RESTRICTED_LAYOUT_SEARCH_BOUND")
        states = following
    valid = {}
    for cells in states:
        if not all((QUANTITY if i == 2 else MONEY).fullmatch(value) for i, value in enumerate(cells)):
            continue
        issue, price, quantity, cost, fair = (decimal(value.replace(",", "")) for value in cells)
        if min(issue, price, quantity) <= 0 or min(cost, fair) < 0 or quantity > quantity_limit or fair > fair_limit:
            continue
        if abs(issue * quantity - cost) <= Decimal("0.01") and abs(price * quantity - fair) <= Decimal("0.01"):
            valid[cells] = (issue, price, quantity, cost, fair)
    if len(valid) != 1:
        raise ValueError("QUALITY_RESTRICTED_LAYOUT_FINANCIAL_ALIGNMENT_" + ("AMBIGUOUS" if valid else "UNRESOLVED"))
    return next(iter(valid.values()))


def fixed_width_stock_restrictions(text: str, positions: list[dict]) -> dict:
    """按实际表头定位日期/代码/金融区域，重建整表，不与失败的半表拼接。

    只恢复境内六位股票代码的股票表；不为港股、基金、债券或缺表头资料猜
    分类。空备注不要求补“－”，任何未消费数字片段、坏日期或超持仓均拒收。
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if "单位：人民币元" not in compact(text):
        raise ValueError("QUALITY_RESTRICTED_LAYOUT_UNIT_UNVERIFIED")
    headers = [
        (i, line) for i, line in enumerate(lines) if all(word in line for word in ("代码", "名称", "认购", "价格"))
    ]
    if not headers:
        raise ValueError("QUALITY_RESTRICTED_LAYOUT_HEADER_UNSUPPORTED")
    number, header = headers[0]
    nearby = compact("\n".join(lines[max(0, number - 3) : number + 4]))
    if not all(word in nearby for word in ("数量", "股", "成本", "估值", "备注")):
        raise ValueError("QUALITY_RESTRICTED_LAYOUT_COLUMNS_UNVERIFIED")
    code_end, date_start, finance_start = header.index("名称"), header.index("认购"), header.index("价格")
    # 转载中日期尾数会向右偏移一至两个字符；日期带止于受限期表头末尾，
    # 不把受限期的“6个月”中的数字接到认购日后面。
    date_end = header.index("受限期") + 3 if "受限期" in header else header.find("期", date_start + 2) + 1
    if not date_start < date_end < finance_start:
        raise ValueError("QUALITY_RESTRICTED_LAYOUT_DATE_COLUMN_UNVERIFIED")
    groups = []
    for line in lines[number + 1 :]:
        if re.match(r"\s*(?:注[：:]|[56]\.4\.12\.1\.[2-9])", line):
            break
        words = list(re.finditer(r"\S+", line))
        if any(date_start <= word.start() < date_end and re.match(r"20\d{2}", word[0]) for word in words):
            groups.append([])
        if groups:
            groups[-1].append(line)
    if not 0 < len(groups) <= 5000:
        raise ValueError("QUALITY_RESTRICTED_LAYOUT_ROW_COUNT")
    by_code = {row["stock_code"].split(".")[0]: row for row in positions}
    rows = []
    for group in groups:
        code_parts, date_parts, financial_lines = [], [], []
        for line in group:
            fragments = []
            for word in re.finditer(r"\S+", line):
                token = word[0]
                if word.start() < code_end and re.fullmatch(r"\d{1,6}", token):
                    code_parts.append(token)
                if date_start <= word.start() < date_end and re.fullmatch(r"(?:20\d{2}[\d-]*|[\d-]+)", token):
                    date_parts.append(token)
                if word.start() >= finance_start and FRAGMENT.fullmatch(token):
                    fragments.append(token)
            if fragments:
                financial_lines.append(fragments)
        code = "".join(code_parts)
        acquired = "".join(date_parts)
        if not re.fullmatch(r"\d{6}", code) or code not in by_code:
            raise ValueError("QUALITY_RESTRICTED_LAYOUT_SECURITY_UNRESOLVED:" + code)
        if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", acquired):
            raise ValueError("QUALITY_RESTRICTED_LAYOUT_ACQUISITION_DATE_UNRESOLVED:" + code + ":" + acquired)
        date.fromisoformat(acquired)
        holding = by_code[code]
        issue, price, quantity, cost, fair = _unique_financial_cells(
            financial_lines,
            quantity_limit=decimal(holding["quantity_shares"]),
            fair_limit=decimal(holding["market_value_cny"]),
        )
        rows.append(
            {
                "code": code,
                "quantity": str(quantity),
                "cost": str(cost),
                "amount": str(fair),
                "layout_evidence": {
                    "source_lines": group,
                    "financial_fragments": financial_lines,
                    "acquired_on": acquired,
                    "issue_price_cny": str(issue),
                    "valuation_price_cny": str(price),
                    "header": header,
                    "rule": "UNIQUE_ORDERED_DIGITS_AND_BOTH_PRICE_QUANTITY_IDENTITIES",
                },
            }
        )
    # 同一新股可分别有新股未上市和半年限售批次，必须合并后仍不超过完整持仓。
    for code, holding in by_code.items():
        matching = [row for row in rows if row["code"] == code]
        if sum(decimal(row["quantity"]) for row in matching) > decimal(holding["quantity_shares"]) or sum(
            decimal(row["amount"]) for row in matching
        ) > decimal(holding["market_value_cny"]):
            raise ValueError("QUALITY_RESTRICTED_LAYOUT_PRODUCT_HOLDING_SUM_CONFLICT:" + code)
    return {
        "rows": rows,
        "row_count": len(rows),
        "header": header,
        "scope": "FULL_STOCK_RESTRICTION_TABLE_FROM_UNIQUE_VISIBLE_CELL_FRAGMENTS",
    }
