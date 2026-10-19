"""公开报告完整股票表的保守适配；金额守恒替代猜测、补零和截取前十大。

只处理已绑定产品的中期报告。保留分组原行，并用资产负债表与行业表独立对账。
旧研究解析器及其首次结果保持原样。
"""

import re
from decimal import Decimal
from itertools import combinations

from app.services.fund_rating_disclosures import compact
from app.services.fund_report_sections import numbered_pages

_ROW = re.compile(r"^\s*(\d{1,4})\s+(\d{5,6})(?=\s|[\u4e00-\u9fff])\s*(.*)$")
_INTEGER = r"(?:\d+|\d{1,3}(?:,\d{3})+)"


def cells(lines: list[str]) -> tuple[str, list[str]]:
    """枚举折行单元格的唯一合法拼接；不补任何数字，金额必须明确到分。"""
    # PDF 的金额末位有时换到下一行，但同一行的最右侧占比已经先输出。
    # 只处理“金额恰缺一位小数 + 下一行只有一位数字”的确定布局，不把页码、
    # 名称或其他缺失金额猜进来；后续仍必须通过全部证券与独立金额对账。
    if len(lines) == 2 and re.fullmatch(r"\d", lines[1].strip()):
        wrapped = re.fullmatch(
            r"([\u4e00-\u9fffA-Za-zＡ-Ｚ*－—-]+)\s+(" + _INTEGER + r")\s+("
            + _INTEGER + r"\.\d)\s+(\d+(?:\.\d+)?%?)\s*",
            lines[0].strip(),
        )
        if wrapped:
            return wrapped[1], [wrapped[2], wrapped[3] + lines[1].strip(), wrapped[4]]
    tokens = " ".join(lines).split()
    words, numbers = [], []
    for token in tokens:
        if re.fullmatch(r"[\d,.]+%?|[-－—]", token):
            numbers.append(token)
        elif re.fullmatch(r"[\u4e00-\u9fffA-Za-zＡ-Ｚ*－—-]+", token):
            words.append(token)
        else:
            raise ValueError("RATING_STOCK_ROW_AMBIGUOUS")
    choices = []
    for a, b in combinations(range(1, len(numbers)), 2):
        values = ["".join(numbers[:a]), "".join(numbers[a:b]), "".join(numbers[b:])]
        if (
            re.fullmatch(_INTEGER + r"(?:\.\d+)?", values[0])
            and re.fullmatch(_INTEGER + r"\.\d{2}", values[1])
            and re.fullmatch(r"(?:\d+(?:\.\d+)?%?|[-－—])", values[2])
        ):
            choices.append(values)
    if len(choices) != 1 or not words:
        raise ValueError("RATING_STOCK_CELLS_INCOMPLETE_OR_AMBIGUOUS")
    return "".join(words), choices[0]


def parse(pages: list[str], stock_amount: str, net_assets: str, *, require_industry: bool = True) -> dict:
    """解析全部股票并合并同一代码的指数/非指数分表，拒绝未统一的跨市场行业分类。"""
    from app.services.fund_rating_public_market import section, stock_pages

    text = "\n".join(stock_pages(numbered_pages(pages)))
    body = section(text, r"(?m)^7\.3\s+[^\n]*股票\s*投\s*资\s*明\s*细", r"(?m)^7\.4\s")
    industry = section(text, r"(?m)^7\.2\s+[^\n]*按行业分类", r"(?m)^7\.3\s")
    # 港股表与境内行业表不能直接拼成不同桶，形成虚假的行业分散。
    if require_industry and "港股通" in industry:
        hk = industry.split("港股通", 1)[-1]
        if not re.search(r"(?:未持有|未投资|无[。.]?)", compact(hk)):
            raise ValueError("RATING_MARKET_MIXED_INDUSTRY_TAXONOMY")
    lines = body.splitlines()
    starts = [i for i, line in enumerate(lines) if _ROW.match(line)]
    if not starts:
        raise ValueError("RATING_STOCK_TABLE_EMPTY")
    positions, edits = {}, []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(lines)
        match = _ROW.match(lines[start])
        rank, code, first = match.groups()
        rest = []
        for offset, line in enumerate(lines[start + 1 : stop], start + 1):
            line = line.strip()
            if re.match(r"(?:注[：:]?|7\.3\.\d+|金额单位|序\s*号|股票代码|占基金|产净值|例[（(])", line):
                break
            if not line or re.fullmatch(r"第?\s*\d+\s*页(?:\s*共\s*\d+\s*页)?", line):
                continue
            if re.search(r"20\d{2}\s*年.*报告$", line):
                continue
            if re.fullmatch(r"\d{1,3}", line) and re.search(r"20\d{2}\s*年.*报告$", lines[offset - 1].strip()):
                continue
            rest.append(line)
        name, values = cells([first, *rest])
        quantity, amount, weight = values
        if require_industry and len(code) == 5:
            raise ValueError("RATING_MARKET_MIXED_INDUSTRY_TAXONOMY")
        exchange = (
            ".HK"
            if len(code) == 5
            else ".SH"
            if code.startswith("6")
            else ".SZ"
            if code.startswith(("0", "3"))
            else ".BJ"
            if code.startswith(("4", "8", "92"))
            else None
        )
        if exchange is None:
            raise ValueError("RATING_STOCK_EXCHANGE_UNKNOWN")
        security = code + exchange
        row = positions.setdefault(
            security,
            {
                "stock_code": security,
                "stock_name": name,
                "quantity_shares": Decimal(0),
                "market_value_cny": Decimal(0),
                "reported_rows": [],
            },
        )
        if row["stock_name"] != name:
            raise ValueError("RATING_STOCK_NAME_CONFLICT")
        qty, value = Decimal(quantity.replace(",", "")), Decimal(amount.replace(",", ""))
        if qty <= 0 or value < 0 or weight not in "-－—" and not 0 <= Decimal(weight.rstrip("%")) <= 100:
            raise ValueError("RATING_STOCK_VALUE_RANGE")
        row["quantity_shares"] += qty
        row["market_value_cny"] += value
        row["reported_rows"].append({"rank": int(rank), "quantity": quantity, "amount": amount, "weight": weight})
        edits.append({"original": lines[start:stop], "cells": values})
    industries = []
    for m in re.finditer(
        r"(?m)^\s*([A-S])\s+([\u4e00-\u9fff、，\s]+?)\s+([\d,]+\.\d{2}|[-－—])\s+([\d.]+|[-－—])\s*%?\s*$", industry
    ):
        code, name, value, weight = m.groups()
        if value in "-－—":
            continue
        industries.append(
            {
                "industry_code": code,
                "name": compact(name),
                "value_cny": value.replace(",", ""),
                "nav_weight_pct": None if weight in "-－—" else weight,
            }
        )
    total = Decimal(stock_amount)
    if abs(sum(row["market_value_cny"] for row in positions.values()) - total) > Decimal(".02"):
        raise ValueError("RATING_STOCK_FULL_AMOUNT_MISMATCH")
    if require_industry and abs(sum(Decimal(row["value_cny"]) for row in industries) - total) > Decimal(".02"):
        raise ValueError("RATING_STOCK_INDUSTRY_AMOUNT_MISMATCH")
    for row in positions.values():
        row["quantity_shares"] = str(row["quantity_shares"])
        row["market_value_cny"] = str(row["market_value_cny"])
    return {
        "holdings": list(positions.values()),
        "reported_industries": industries,
        "stock_value_cny": stock_amount,
        "stock_nav_pct": str(total / Decimal(net_assets) * 100),
        "normalization": edits,
    }


def parse_disclosed(pages: list[str], stock_amount: str, net_assets: str) -> dict:
    """持仓用途核对全部股票与独立行业表，单列有原文证明的替代款增值。

    股票资产账面金额有时含可退替代款估值增值，证券明细却不含该非证券项目。
    只在原文明确说明账面含该项目、资产组合股票项等于资产负债表、全部行业金额
    等于全部股票明细三项同时满足时接受。差额保留为独立金额，不能分摊到
    股票、伪造证券或通过放宽金额误差消除，评级仍由其原有入口单独核验。
    """
    from app.services.fund_rating_public_market import section, stock_pages

    try:
        result = parse(pages, stock_amount, net_assets, require_industry=False)
        result["substitution_increment_cny"] = "0"
        return result
    except ValueError as error:
        if str(error) != "RATING_STOCK_FULL_AMOUNT_MISMATCH":
            raise
    text = "\n".join(stock_pages(numbered_pages(pages)))
    asset = section(text, r"(?m)^\s*7\.1\s+[^\n]*资产组合情况", r"(?m)^\s*7\.2\s")
    industry = section(text, r"(?m)^\s*7\.2\s+[^\n]*按行业分类", r"(?m)^\s*7\.3\s")
    note = compact(asset)
    if not re.search(r"股票[^。]{0,50}含可退替代款[^。]{0,10}估值增值", note):
        raise ValueError("HOLDINGS_SUBSTITUTION_NOTE_UNVERIFIED")
    equity = re.findall(r"(?m)^\s*其中[：:]\s*股票\s+([\d,]+\.\d{2})\s+[\d.]+\s*$", asset)
    totals = re.findall(r"(?m)^\s*合计\s+([\d,]+\.\d{2})\s+[\d.]+\s*$", industry)
    if len(equity) != 1 or not totals:
        raise ValueError("HOLDINGS_SUBSTITUTION_TOTAL_UNVERIFIED")
    original_total = Decimal(stock_amount)
    if abs(Decimal(equity[0].replace(",", "")) - original_total) > Decimal(".02"):
        raise ValueError("HOLDINGS_SUBSTITUTION_BALANCE_CONFLICT")
    disclosed = sum(Decimal(value.replace(",", "")) for value in totals)
    difference = original_total - disclosed
    # 估值增值项目也可能为负，不能将负差额截成零或丢弃；仅允许明确非零差额。
    if difference == 0:
        raise ValueError("HOLDINGS_SUBSTITUTION_INCREMENT_RANGE")
    # 使用真实行业合计作为证券金额，再逐证券相加对账；任何遗漏仍会失败。
    result = parse(pages, str(disclosed), net_assets, require_industry=False)
    if abs(sum(Decimal(row["value_cny"]) for row in result["reported_industries"]) - disclosed) > Decimal(".02"):
        raise ValueError("HOLDINGS_SUBSTITUTION_INDUSTRY_CONFLICT")
    result.update(
        balance_stock_value_cny=stock_amount,
        substitution_increment_cny=str(difference),
        substitution_basis="BALANCE_LESS_INDEPENDENTLY_RECONCILED_SECURITIES",
        reconciliation_quotes=[asset, industry],
    )
    return result
