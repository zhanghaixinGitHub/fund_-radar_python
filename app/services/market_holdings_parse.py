"""新报告持仓适配；季报只展示本期披露部分，不用旧中报补成当期完整组合。"""

import re
import unicodedata
from decimal import Decimal
from pathlib import Path

from app.integrations.fund_report_layout_v3 import parse_text
from app.integrations.market_fund_reports import read_source
from app.services.fund_holdings_publication import (
    _domestic_fund_rows,
    _fund_name_matches,
    _public_table,
    _report_identity_matches,
    report_from_original,
)
from app.services.fund_materials_build import safe_url
from app.services.fund_rating_public_market import public_document, publication


def _unfold_partial_stock_rows(body: str) -> str:
    """连接原表同一行的跨行名称、数量及金额，不按权重反推任何数字。

    数量只接受完整千分组，金额必须完整到两位小数；连续碎片按原列顺序
    唯一闭合后才交给原排名及范围校验。页眉数字不作金额，残缺或歧义拒绝。
    """
    quantity_pattern = r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?"
    amount_pattern = r"(?:\d+|\d{1,3}(?:,\d{3})+)\.\d{2}"
    number = r"[\d,.]+"
    anchor = re.compile(r"^\s*(\d{1,2})\s+(\d{5,6})\s+(.*?)\s*$")
    current, pending_name, pending_value, fragments, rows = None, "", "", [], []

    def complete():
        nonlocal current, fragments
        if current is None:
            return
        choices = []
        for split in range(len(fragments) + 1):
            q = current["quantity"] + "".join(fragments[:split])
            value = current["value"] + "".join(fragments[split:])
            if re.fullmatch(quantity_pattern, q) and re.fullmatch(amount_pattern, value):
                choices.append((q, value))
        if len(set(choices)) > 1:
            raise ValueError("MARKET_PARTIAL_WRAPPED_AMBIGUOUS")
        if choices:
            q, value = choices[0]
            if not current["name"]:
                raise ValueError("MARKET_PARTIAL_WRAPPED_NAME")
            rows.append(f"{current['rank']} {current['code']} {current['name']} {q} {value} {current['weight']}")
            current, fragments = None, []

    for line in body.splitlines():
        if not line.strip():
            continue
        if any(k in line for k in ("股票代码", "股票名称", "公允价值", "值比例", "占基金资产", "数量", "季度报告")):
            continue
        if re.fullmatch(r"\s*[（(](?:元|%|％)[)）]\s*", line):
            continue
        match = anchor.match(line)
        if match:
            if current is not None:
                raise ValueError("MARKET_PARTIAL_WRAPPED_INCOMPLETE")
            tail = match[3]
            tokens = list(re.finditer(number, tail))
            # 名称不能含数字，数量、可选金额和权重必须是末尾连续的数字列。
            if len(tokens) not in {2, 3} or not re.fullmatch(r"\d+(?:\.\d+)?", tokens[-1][0]):
                raise ValueError("MARKET_PARTIAL_WRAPPED_COLUMNS")
            name = tail[: tokens[0].start()].strip()
            if any(tail[tokens[i].end() : tokens[i + 1].start()].strip() for i in range(len(tokens) - 1)):
                raise ValueError("MARKET_PARTIAL_WRAPPED_COLUMNS")
            if tail[tokens[-1].end() :].strip():
                raise ValueError("MARKET_PARTIAL_WRAPPED_COLUMNS")
            value = tokens[1][0] if len(tokens) == 3 else pending_value
            if len(tokens) == 3 and pending_value:
                raise ValueError("MARKET_PARTIAL_WRAPPED_AMBIGUOUS")
            current = {
                "rank": match[1],
                "code": match[2],
                "name": pending_name + re.sub(r"\s+", "", name),
                "quantity": tokens[0][0],
                "value": value,
                "weight": tokens[-1][0],
            }
            pending_name, pending_value, fragments = "", "", []
            complete()
            continue
        pieces = line.split()
        numeric = [p for p in pieces if re.fullmatch(number, p)]
        words = [p for p in pieces if not re.fullmatch(number, p)]
        if words and (len(words) > 2 or any(not re.fullmatch(r"[\u4e00-\u9fffA-Za-z*\-]{1,12}", p) for p in words)):
            raise ValueError("MARKET_PARTIAL_WRAPPED_UNKNOWN_LINE")
        if current is None:
            # 子表标题“资明细”等折行仅可出现在首个持仓行之前。
            if not rows and not numeric and not pending_value:
                continue
            if not pending_value and numeric and not any(c in numeric[0] for c in ",."):
                raise ValueError("MARKET_PARTIAL_WRAPPED_UNBOUND_NUMBER")
            pending_name += "".join(words)
            pending_value += "".join(numeric)
        else:
            current["name"] += "".join(words)
            fragments.extend(numeric)
            complete()
    if current is not None or pending_name or pending_value or not rows:
        raise ValueError("MARKET_PARTIAL_WRAPPED_INCOMPLETE")
    return "\n".join(rows)


def _reported_no_stocks(text: str) -> bool:
    """仅接受原文明确未持股，或股票资产横杠与本期股票明细“无”的双重证据。"""
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    if re.search(r"本基金(?:在)?本?报告期末未持有股票|报告期末本基金未持有股票", compact):
        return True
    heading = re.search(r"5\.[34](?:\.1)?[^\n]*前十名股票[^\n]*明细", text)
    if not heading:
        return False
    following = text[heading.end() :]
    stop = re.search(r"(?m)^\s*5\.\d+\s+", following)
    if stop is None or re.sub(r"\s+", "", following[: stop.start()]) not in {"无。", "无", "注:无。", "注：无。"}:
        return False
    assets = re.search(r"(?m)^\s*5\.1\s+[^\n]*资产组合情况", text)
    if not assets or assets.start() >= heading.start():
        return False
    section = text[assets.end() : heading.start()]
    return bool(re.search(r"(?m)^\s*其中[：:]?\s*股票\s+-\s+-\s*$", section))


def _money_quarter_assets(text: str) -> list[dict]:
    """货币基金5.1原表逐行金额对账；不从份额数或比例倒推资产。"""
    heading = re.search(r"(?m)^\s*5\.1\s+[^\n]*资产组合情况", text)
    if not heading:
        raise ValueError("MARKET_MONEY_ASSET_SECTION")
    tail = text[heading.end() :]
    end = re.search(r"(?m)^\s*5\.2\s+", tail)
    body = tail[: end.start()] if end else ""
    compact = re.sub(r"\s+", "", body)
    if "金额（元）" not in compact or "人民币元" not in re.sub(r"\s+", "", text):
        raise ValueError("MARKET_MONEY_ASSET_UNIT")
    rows = []
    for match in re.finditer(r"(?m)^\s*(\d{1,2})\s+(.+?)\s+([\d,]+\.\d{2})\s+(\d+\.\d+)\s*$", body):
        if int(match[1]) != len(rows) + 1:
            raise ValueError("MARKET_MONEY_ASSET_RANK")
        rows.append(
            {
                "name": re.sub(r"\s+", "", match[2]),
                "value": Decimal(match[3].replace(",", "")),
                "weightPct": Decimal(match[4]),
            }
        )
    if (
        len(rows) < 2
        or rows[-1]["name"] != "合计"
        or rows[-1]["weightPct"] != 100
        or abs(sum(row["value"] for row in rows[:-1]) - rows[-1]["value"]) > Decimal(".02")
    ):
        raise ValueError("MARKET_MONEY_ASSET_RECONCILIATION")
    return [
        {"name": row["name"], "value": float(row["value"]), "weightPct": float(row["weightPct"])} for row in rows[:-1]
    ]


def _money_quarter_bonds(text: str) -> list[dict]:
    """读取货币季报前十债券；保留存单名称的CD尾码和原列账面价值。"""
    head = re.search(r"(?m)^\s*5\.6\s+[^\n]*前十名债券投资明细", text)
    if not head:
        raise ValueError("MARKET_MONEY_BOND_SECTION")
    tail = text[head.end() :]
    stop = re.search(r"(?m)^\s*5\.7", tail)
    if not stop:
        raise ValueError("MARKET_MONEY_BOND_TRUNCATED")
    body = re.split(r"注[：:]", tail[: stop.start()])[0]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    anchors = [(i, m) for i, line in enumerate(lines) if (m := re.match(r"^(\d{1,2})\s+(\d{9})\s+(.+)$", line))]
    rows = []
    for n, (i, match) in enumerate(anchors):
        cells = re.fullmatch(r"(.+?)\s+([\d,]+)\s+([\d,]+\.\d{2})\s+(\d+\.\d+)", match[3])
        if int(match[1]) != n + 1 or not cells:
            raise ValueError("MARKET_MONEY_BOND_ROW")
        end = anchors[n + 1][0] if n + 1 < len(anchors) else len(lines)
        rest = lines[i + 1 : end]
        if len(rest) > 1 or any(not re.fullmatch(r"CD\d{3}", line) for line in rest):
            raise ValueError("MARKET_MONEY_BOND_NAME")
        quantity, amount, weight = (Decimal(value.replace(",", "")) for value in cells.groups()[1:])
        if quantity <= 0 or amount <= 0 or not 0 <= weight <= 100:
            raise ValueError("MARKET_MONEY_BOND_RANGE")
        rows.append(
            {
                "stockCode": match[2],
                "stockName": re.sub(r"\s+", "", cells[1]) + "".join(rest),
                "marketValue": float(amount),
                "weightPct": float(weight),
            }
        )
    if not rows or len(rows) > 10 or len({r["stockCode"] for r in rows}) != len(rows):
        raise ValueError("MARKET_MONEY_BOND_INCOMPLETE")
    return rows


def _wrapped_us_stock_rows(body: str) -> list[dict]:
    """读取证券列与金额均折行的美元股票表，只连接同一原行的金额碎片。

    排名行必须同时有美国、数量和比例；证券单元格必须明示US和纳斯达克。
    金额只从数量列右侧读取，完整千分组、两位小数及连续排名不满足即拒绝。
    不以公开转载表的万元精度覆盖原报告人民币元的精度。
    """
    lines = [line for line in body.splitlines() if line.strip()]
    anchors = []
    for i, line in enumerate(lines):
        match = re.match(r"^\s*(\d{1,2})\s+.*?美国\s+([\d,]+)\s+([\d,.]+)\s+(\d+\.\d+)\s*$", line)
        if not match:
            continue
        candidates = []
        for j in range(max(0, i - 2), i + 1):
            for token in re.finditer(r"\b[A-Z]{2,6}\b(?=\s+纳斯达)", lines[j]):
                if 20 <= token.start() <= 34:
                    candidates.append((j, token[0]))
        if len(candidates) != 1 or int(match[1]) != len(anchors) + 1:
            raise ValueError("MARKET_US_WRAPPED_IDENTITY")
        start, ticker = candidates[0]
        # 数量与金额之间的空白不固定，按实际数量末端约束其他行的金额列。
        anchors.append((start, i, ticker, match))
    row_count = sum(bool(re.match(r"^\s*\d{1,2}\s+", line)) for line in lines)
    if not anchors or len(anchors) > 10 or len(anchors) != row_count:
        raise ValueError("MARKET_US_WRAPPED_ROWS")
    result = []
    for n, (start, index, ticker, match) in enumerate(anchors):
        end = anchors[n + 1][0] if n + 1 < len(anchors) else len(lines)
        block = "\n".join(lines[start:end])
        explicit_market = all(x in re.sub(r"\s+", "", block) for x in ("纳斯达", "克证券", "交易所"))
        if not re.search(r"\bUS\b", block) or not explicit_market:
            raise ValueError("MARKET_US_WRAPPED_MARKET")
        pieces = []
        for j in range(start, end):
            if j == index:
                pieces.append(match[3])
                continue
            tokens = [m[0] for m in re.finditer(r"[\d,.]+", lines[j]) if m.start() >= match.end(2) - 3]
            if len(tokens) > 1:
                raise ValueError("MARKET_US_WRAPPED_AMOUNT")
            pieces.extend(tokens)
        value = "".join(pieces)
        if not re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d{2}", value):
            raise ValueError(f"MARKET_US_WRAPPED_AMOUNT:{ticker}:{value}")
        quantity, amount, weight = (Decimal(v.replace(",", "")) for v in (match[2], value, match[4]))
        if quantity <= 0 or amount <= 0 or not 0 <= weight <= 100:
            raise ValueError("MARKET_US_WRAPPED_RANGE")
        result.append(
            {
                "stockCode": ticker + ".US",
                "stockName": ticker,
                "marketValue": float(amount),
                "weightPct": float(weight),
                "nameBasis": "REPORTED_MARKET_CODE",
            }
        )
    if len({h["stockCode"] for h in result}) != len(result) or sum(h["weightPct"] for h in result) > 100:
        raise ValueError("MARKET_US_WRAPPED_RANGE")
    return result


def _local_us_code_rows(body: str) -> list[dict]:
    """核对当地代码美国股票表；同一公司两类股票可共用一项排名。

    数值列、证券代码列与美国交易所须在同一原行邻域；排名仅绑定最近的唯一
    序号行。所有美国数值行都必须被保留，不能漏掉折行在公司序号之前的A类。
    """
    lines = [line for line in body.splitlines() if line.strip()]
    column = next((line.index("证券代码") for line in lines if "证券代码" in line), None)
    anchors = [(i, int(m[1])) for i, line in enumerate(lines) if (m := re.match(r"^\s*(\d{1,2})\s+", line))]
    if column is None or not anchors or [n for _, n in anchors] != list(range(1, len(anchors) + 1)):
        raise ValueError("MARKET_LOCAL_US_RANK")
    rows, ranks = [], []
    for i, line in enumerate(lines):
        values = re.search(r"美国\s+([\d,]+(?:\.\d+)?)\s+([\d,]+\.\d{2})\s+(\d+\.\d+)\s*$", line)
        if not values:
            continue
        # 中文单元格的显示宽度不等于Python字符数；不能按英文表头下标选到公司名中的INC。
        # 本版式证券代码紧接交易所第二段，二者和美国、数量必须位于同一数据行。
        cells = list(re.finditer(r"\b([A-Z]{2,6})\s+(克证券|约证券)\s+美国\s+", line))
        codes = [m[1] for m in cells]
        distance = min(abs(i - j) for j, _ in anchors)
        rank = [n for j, n in anchors if abs(i - j) == distance]
        prefixes = [m[1] for other in lines[max(0, i - 2) : i] if (m := re.search(r"(纳斯达|美国纽)\s*$", other))]
        tails = any("交易所" in other for other in lines[i + 1 : i + 3])
        exchange = prefixes[-1] + cells[0][2] + "交易所" if prefixes and len(cells) == 1 and tails else ""
        if (
            len(codes) != 1
            or len(rank) != 1
            or distance > 2
            or exchange not in {"纳斯达克证券交易所", "美国纽约证券交易所"}
        ):
            raise ValueError("MARKET_LOCAL_US_IDENTITY")
        quantity, amount, weight = (Decimal(v.replace(",", "")) for v in values.groups())
        if quantity <= 0 or amount <= 0 or not 0 <= weight <= 100:
            raise ValueError("MARKET_LOCAL_US_RANGE")
        ranks.append(rank[0])
        rows.append(
            {
                "stockCode": codes[0] + ".US",
                "stockName": codes[0],
                "marketValue": float(amount),
                "weightPct": float(weight),
                "nameBasis": "REPORTED_MARKET_CODE",
            }
        )
    if (
        not rows
        or len(rows) > 20
        or len(anchors) > 10
        or ranks != sorted(ranks)
        or set(ranks) != {n for _, n in anchors}
        or len(rows) != sum(bool(re.search(r"美国\s+[\d,]+", line)) for line in lines)
        or len({r["stockCode"] for r in rows}) != len(rows)
        or sum(r["weightPct"] for r in rows) > 100
    ):
        raise ValueError("MARKET_LOCAL_US_INCOMPLETE")
    return rows


def _two_country_wrapped_us_rows(body: str) -> list[dict]:
    """同表分别写交易市场/所属国家为美国的列式披露，证券列显式带US。

    数量及金额有时在排名前一行开始，只有原行的两个数字都带逗号且金额以
    小数点结束时才接纳该前置行；其他行的尾数绝不借给下一证券。
    """
    header = re.sub(r"\s+", "", body[:650])
    if not all(word in header for word in ("所在证", "所属", "国家", "数量", "值（人", "民币元）")):
        raise ValueError("MARKET_US_TWO_COUNTRY_HEADER")
    body = re.split(r"(?m)^\s*5\.4\.2\s+", body)[0]
    lines = [line for line in body.splitlines() if line.strip()]
    pattern = r"^\s*(\d{1,2})\s+.*?美国\s+美国\s+([\d,]+)\s+([\d,.]+)\s+(\d+\.\d+)\s*$"
    anchors = [(i, re.match(pattern, line)) for i, line in enumerate(lines)]
    anchors = [(i, m) for i, m in anchors if m]
    row_count = sum(bool(re.match(r"^\s*\d{1,2}\s+", line)) for line in lines)
    if (
        not anchors
        or len(anchors) > 10
        or len(anchors) != row_count
        or [int(m[1]) for _, m in anchors] != list(range(1, len(anchors) + 1))
    ):
        raise ValueError("MARKET_US_TWO_COUNTRY_RANK")
    starts = []
    for i, _ in anchors:
        previous = re.search(r"([\d,]+)\s+(\d{1,3}(?:,\d{3})+\.)\s*$", lines[i - 1]) if i else None
        starts.append(i - 1 if previous and "," in previous[1] and not re.match(r"^\s*\d+\s+", lines[i - 1]) else i)

    def display_position(line, index):
        return sum(2 if unicodedata.east_asian_width(c) in {"W", "F"} else 1 for c in line[:index])

    rows = []
    for n, (i, match) in enumerate(anchors):
        block = lines[starts[n] : starts[n + 1] if n + 1 < len(starts) else len(lines)]
        tickers, market, quantities, amounts = set(), False, [], []
        for j, line in enumerate(block, starts[n]):
            for token in re.finditer(r"\b[A-Z]{1,6}\b", line):
                if 24 <= display_position(line, token.start()) <= 34:
                    if token[0] == "US":
                        market = True
                    else:
                        tickers.add(token[0])
            if j == i:
                quantities.append(match[2])
                amounts.append(match[3])
            else:
                tokens = [
                    m[0]
                    for m in re.finditer(r"(?<![\w.])[\d,]+(?:\.[\d]*)?(?![\w.])", line)
                    if display_position(line, m.start()) >= 45
                ]
                if tokens:
                    if len(tokens) != 2:
                        raise ValueError("MARKET_US_TWO_COUNTRY_FRAGMENTS")
                    quantities.append(tokens[0])
                    amounts.append(tokens[1])
        quantity, amount = "".join(quantities), "".join(amounts)
        if (
            len(tickers) != 1
            or not market
            or not re.fullmatch(r"\d{1,3}(?:,\d{3})*", quantity)
            or not re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d{2}", amount)
        ):
            raise ValueError("MARKET_US_TWO_COUNTRY_IDENTITY_OR_AMOUNT")
        q, value, weight = Decimal(quantity.replace(",", "")), Decimal(amount.replace(",", "")), Decimal(match[4])
        if q <= 0 or value <= 0 or not 0 <= weight <= 100:
            raise ValueError("MARKET_US_TWO_COUNTRY_RANGE")
        ticker = next(iter(tickers))
        rows.append(
            {
                "stockCode": ticker + ".US",
                "stockName": ticker,
                "marketValue": float(value),
                "weightPct": float(weight),
                "nameBasis": "REPORTED_MARKET_CODE",
            }
        )
    if len({h["stockCode"] for h in rows}) != len(rows) or sum(h["weightPct"] for h in rows) > 100:
        raise ValueError("MARKET_US_TWO_COUNTRY_RANGE")
    return rows


def partial_overseas_stock_rows(text: str) -> list[dict]:
    """有限读取美元/日本市场代码的季度前十表，保留人民币金额和部分披露。

    英文公司名跨行时仅显示原文明示市场代码，沿用既有日本原件适配的做法；
    公司法定名称仍由资料阶段另行关联，不拼接相邻公司的英文名。
    """
    heads = list(re.finditer(r"(?m)^\s*5\.4(?:\.1)?\s+[^\n]*前十名股票及存托凭证", text))
    if not heads:
        raise ValueError("MARKET_FOREIGN_QUARTER_SECTION")
    following = text[heads[-1].end() :]
    stop = re.search(r"(?m)^\s*5\.5\s+", following)
    if not stop:
        raise ValueError("MARKET_FOREIGN_QUARTER_TRUNCATED")
    body = following[: stop.start()]
    compact = re.sub(r"\s+", "", body)
    if "证券代码" in compact and "公司名称" in compact and "人民币元" in compact:
        try:
            return _local_us_code_rows(body)
        except ValueError:
            pass
    wrapped_us = all(x in compact[:500] for x in ("证券代", "码", "数量", "值（人", "民币", "元）", "值比例"))
    if wrapped_us and not any(x in compact[:500] for x in ("美元", "港元", "日元")):
        try:
            return _wrapped_us_stock_rows(body)
        except ValueError:
            return _two_country_wrapped_us_rows(body)
    cny_header = "人民币" in compact[:500] or ("人民" in compact[:500] and "币元" in compact[:500])
    if (
        not all(x in compact for x in ("证券", "代码", "数量", "值比例"))
        or not cny_header
        or any(x in compact[:500] for x in ("美元", "港元", "日元"))
    ):
        raise ValueError("MARKET_FOREIGN_QUARTER_HEADER")
    lines = [line for line in body.splitlines() if line.strip()]
    code_column = next((line.index("代码") for line in lines if "代码" in line), None)
    if code_column is None:
        raise ValueError("MARKET_FOREIGN_QUARTER_HEADER")
    anchors = [(i, re.match(r"^\s*(\d{1,2})\s+", line)) for i, line in enumerate(lines)]
    anchors = [(i, m) for i, m in anchors if m]
    result = []
    for i, anchor in anchors:
        line = lines[i]
        numeric = re.search(r"(日本|美国)\s+([\d,]+(?:\.\d+)?)\s+([\d,]+\.\d{2})\s+(\d+(?:\.\d+)?)\s*$", line)
        if int(anchor[1]) != len(result) + 1 or numeric is None:
            raise ValueError("MARKET_FOREIGN_QUARTER_ROW")
        country, quantity, amount, ratio = numeric.groups()
        if country == "日本":
            match = re.search(r"\b([0-9A-Z]{4})\s+(?:交易|交|易所)\s+日本", line)
            if match is None:
                raise ValueError("MARKET_FOREIGN_QUARTER_CODE")
            ticker, suffix = match[1], ".JP"
        else:
            # 只接明确US市场后缀；先读本行证券列，缺前缀时只读紧邻前行同列。
            candidates = []
            for index in [i, i - 1, i - 2]:
                if index < 0:
                    continue
                positioned = [
                    m
                    for m in re.finditer(r"\b[A-Z]{2,6}\b", lines[index])
                    if m[0] != "US" and code_column - 7 <= m.start() <= code_column + 7
                ]
                nearest = min((abs(m.start() - code_column) for m in positioned), default=None)
                cells = [m[0] for m in positioned if abs(m.start() - code_column) == nearest]
                if cells:
                    candidates = cells
                    break
            neighborhood = "\n".join(lines[max(0, i - 2) : i + 4])
            exchange = re.sub(r"\s+", "", neighborhood)
            # 同一行已明示美国时，完整交易所名称也可证明实际市场；不要求原表另印US。
            us_exchange = country == "美国" and any(
                label in exchange for label in ("纳斯达克证券交易所", "纳斯达克交易所", "纽约证券交易所")
            )
            if len(candidates) != 1 or not (re.search(r"\bUS\b", neighborhood) or us_exchange):
                raise ValueError("MARKET_FOREIGN_QUARTER_CODE")
            ticker, suffix = candidates[0], ".US"
        q, value, weight = (Decimal(v.replace(",", "")) for v in (quantity, amount, ratio))
        if q <= 0 or value <= 0 or not 0 <= weight <= 100:
            raise ValueError("MARKET_FOREIGN_QUARTER_RANGE")
        result.append(
            {
                "stockCode": ticker + suffix,
                "stockName": ticker,
                "marketValue": float(value),
                "weightPct": float(weight),
                "nameBasis": "REPORTED_MARKET_CODE",
            }
        )
    if (
        not result
        or len(result) > 10
        or len({r["stockCode"] for r in result}) != len(result)
        or sum(r["weightPct"] for r in result) > 100
    ):
        raise ValueError("MARKET_FOREIGN_QUARTER_RANGE")
    return result


def disclosed_stocks(rows: list[dict], text: str) -> list[dict]:
    """同为6月/12月的公开表可能已经更新为中报/年报，只取原季报明确列出的证券。"""
    sections = []
    heading = r"前十名(?:股票(?:及存托凭证)?|权益)\s*投资\s*明细"
    for match in re.finditer(heading, text):
        following = text[match.end() :]
        stop = re.search(r"(?m)^\s*[5-8]\.\d+(?:\.\d+)?\s+[\u4e00-\u9fff]", following)
        sections.append(following[: stop.start()] if stop else following[:8000])
    selected = []
    for row in rows:
        code = row["stockCode"].removesuffix(".SH").removesuffix(".SZ").removesuffix(".HK")
        if any(re.search(r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])", s) for s in sections):
            selected.append(row)
    if not selected or len(selected) > 20:
        raise ValueError("MARKET_QUARTER_DISCLOSED_STOCK_IDENTITY")
    return selected


def partial_stock_sections(text: str) -> tuple[list[dict], list[dict]]:
    """只解析季报明确的前十/积极投资前五表，分表序号重置不代表重复持仓。

    不扩大成全仓。每个表独立检查表头、完整数字列、连续序号、证券身份和范围，
    有残缺行、跨表同代码或未识别版式时拒绝，继续原公开表核验路径。
    """
    starts = list(re.finditer(r"(?m)^\s*5\.[34]\s*(?:期末|报告期末)[^\n]*股票投资明细", text))
    unnumbered = False
    if not starts:
        # 个别已绑定身份的LOF转载去掉整份报告章节编号，但保留唯一完整标题和下节边界。
        starts = list(re.finditer(r"(?m)^\s*期末按公允价值占基金资产净值比例大小排序的股票投资明细\s*$", text))
        if len(starts) != 1:
            raise ValueError("MARKET_PARTIAL_STOCK_SECTION_MISSING")
        unnumbered = True
    following = text[starts[-1].end() :]
    section_number = int(re.search(r"5\.([34])", starts[-1][0])[1]) if not unnumbered else 3
    stop = re.search(r"(?m)^\s*5\." + str(section_number + 1) + r"\s*(?:报告期末|期末)", following)
    if stop is None and unnumbered:
        stop = re.search(r"(?m)^\s*报告期末按债券品种分类的债券投资组合\s*$", following)
    if not stop:
        raise ValueError("MARKET_PARTIAL_STOCK_SECTION_TRUNCATED")
    section = following[: stop.start()]
    heads = list(re.finditer(r"(?m)^\s*5\." + str(section_number) + r"\.\d+\s+[^\n]*", section))
    blocks = (
        [
            (m[0], section[m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(section)])
            for i, m in enumerate(heads)
        ]
        if heads
        else [("前十名股票投资明细", section)]
    )
    holdings, groups, seen = [], [], set()
    pattern = (
        r"(?m)^[ \t]*(\d{1,2})[ \t]+(\d{5,6})[ \t]+(.+?)[ \t]+"
        r"([\d,]+(?:\.\d+)?)[ \t]+([\d,]+\.\d{2})[ \t]+(\d+(?:\.\d+)?)[ \t]*%?[ \t]*$"
    )
    for heading, body in blocks:
        if not re.search(r"前[十五]", re.sub(r"\s+", "", heading)):
            if re.search(r"(?m)^\s*\d+\s+\d{5,6}\s+", body):
                raise ValueError("MARKET_PARTIAL_UNKNOWN_STOCK_TABLE")
            continue
        compact = re.sub(r"\s+", "", body)
        # PDF把“序号”第二行插在另一列的“净/值比例”之间，只在明确表头位置消除该交错字。
        # 原持仓行完全不改，金额、数量和比例仍逐列核验。
        header = re.split(r"(?m)^\s*\d+\s+\d{5,6}\s+", body, maxsplit=1)[0]
        if re.search(r"(?m)^\s*号\s+值比例", header):
            compact = compact.replace("净号值比例", "净值比例", 1)
        ratio_header = "净值比例" in compact or re.search(
            r"占基金(?:资产)?净.{0,150}值比例", re.sub(r"\s+", "", header)
        )
        if not all(h in compact for h in ("股票代码", "股票名称", "数量", "公允价值")) or not ratio_header:
            raise ValueError("MARKET_PARTIAL_STOCK_HEADER")
        matches = list(re.finditer(pattern, body))
        candidates = list(re.finditer(r"(?m)^\s*\d+\s+\d{5,6}\s+", body))
        if not matches or len(matches) != len(candidates):
            try:
                unfolded = _unfold_partial_stock_rows(body)
            except ValueError as error:
                raise ValueError("MARKET_PARTIAL_STOCK_INCOMPLETE:" + str(error)) from error
            matches = list(re.finditer(pattern, unfolded))
            if not matches or len(matches) != len(candidates):
                raise ValueError("MARKET_PARTIAL_STOCK_INCOMPLETE")
        limit = 5 if "前五" in re.sub(r"\s+", "", heading) else 10
        ranks = []
        previous = None
        for match in matches:
            rank = int(match[1])
            if ranks and rank == ranks[-1]:
                # 同一发行人的A、H两行共用排名；必须名称相同、代码长度分别为六位和五位。
                # 不接受同名同市场或任意“并列排名”，更不会合并两市场金额及权重。
                if (
                    previous is None
                    or re.sub(r"\s+", "", previous[3]) != re.sub(r"\s+", "", match[3])
                    or {len(previous[2]), len(match[2])} != {5, 6}
                ):
                    raise ValueError("MARKET_PARTIAL_STOCK_RANK")
                previous = None
            else:
                if rank != len(ranks) + 1:
                    raise ValueError("MARKET_PARTIAL_STOCK_RANK")
                ranks.append(rank)
                previous = match
        if len(ranks) > limit:
            raise ValueError("MARKET_PARTIAL_STOCK_RANK")
        rows = []
        for m in matches:
            _, code, name, quantity, value, weight = m.groups()
            suffix = (
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
            if (
                not suffix
                or code + suffix in seen
                or Decimal(quantity.replace(",", "")) <= 0
                or Decimal(value.replace(",", "")) <= 0
                or not 0 <= Decimal(weight) <= 100
            ):
                raise ValueError("MARKET_PARTIAL_STOCK_IDENTITY_OR_VALUE")
            seen.add(code + suffix)
            rows.append(
                {
                    "stockCode": code + suffix,
                    "stockName": re.sub(r"\s+", "", name),
                    "marketValue": float(Decimal(value.replace(",", ""))),
                    "weightPct": float(Decimal(weight)),
                }
            )
        holdings.extend(rows)
        groups.append(
            {
                "name": re.sub(r"\s+", "", heading),
                "notice": "本期独立披露部分，不代表完整或实时持仓。",
                "holdings": rows,
            }
        )
    if not holdings or sum(h["weightPct"] for h in holdings) > 100:
        raise ValueError("MARKET_PARTIAL_STOCK_RANGE")
    return holdings, groups


def disclosed_single_fund(text: str, title: str) -> list[dict]:
    """解析仅一只基金的直接投资表；跨列金额只拼接明确数值列，不推算、不借用旧期。"""
    found = list(re.finditer(r"(?m)^\s*5\.\d+(?:\.\d+)?\s+[^\n]*(?:基金投资明细|投资目标基金明细)", text))
    if not found:
        raise ValueError("MARKET_QUARTER_FUND_SECTION")
    following = text[found[-1].end() :]
    stop = re.search(r"(?m)^\s*5\.\d+\s+[\u4e00-\u9fff]", following)
    body = following[: stop.start()] if stop else following[:8000]
    lines = body.splitlines()
    ranked = [(i, line) for i, line in enumerate(lines) if re.match(r"^\s*\d{1,2}\s+", line)]
    if len(ranked) != 1 or not re.match(r"^\s*1\s+", ranked[0][1]):
        raise ValueError("MARKET_QUARTER_FUND_MULTIPLE_OR_EMPTY")
    index, line = ranked[0]
    numbers = list(re.finditer(r"(?<![A-Za-z0-9])\d[\d,.]*(?![A-Za-z0-9])", line))
    if not numbers or not re.fullmatch(r"\d+\.\d{2}", numbers[-1][0]):
        raise ValueError("MARKET_QUARTER_FUND_WEIGHT")
    weight = numbers[-1]
    lower = max(15, weight.start() - 35)
    pieces = [
        m[0]
        for i, current in enumerate(lines)
        for m in re.finditer(r"(?<![A-Za-z0-9])(?:\d[\d,.]*|,[\d,.]+)(?![A-Za-z0-9])", current)
        if m.start() >= lower
        and not (i == index and m.start() == weight.start())
        and not current.lstrip().startswith("注")
    ]
    amount = "".join(pieces)
    if not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)\.\d{2}", amount):
        raise ValueError("MARKET_QUARTER_FUND_AMOUNT_AMBIGUOUS")
    value, ratio = Decimal(amount.replace(",", "")), Decimal(weight[0])
    if value <= 0 or not 0 < ratio <= 100:
        raise ValueError("MARKET_QUARTER_FUND_RANGE")
    if "联接" in title:
        name = title.split("联接", 1)[0].removesuffix("发起式")
        core = re.sub(r"交易型开放式.*|ETF.*", "", name)
        # 原表一行被类型、管理人等列打断时，按原文顺序逐字核对目标名称核心。
        # 只允许已绑定报告身份、且表中明确只有一只基金的联接产品使用此路径。
        remaining = iter(re.sub(r"\s+", "", body))
        if len(core) < 6 or not all(any(char == candidate for candidate in remaining) for char in core):
            raise ValueError("MARKET_QUARTER_TARGET_FUND_NAME")
    else:
        kind = re.search(r"\bETF\b", line)
        if not kind:
            raise ValueError("MARKET_QUARTER_FUND_NAME")
        name = " ".join(
            current[6 : kind.start()].strip()
            for current in lines
            if current[6 : kind.start()].strip()
            and not any(
                word in current
                for word in ("基金名称", "序号", "金额单位", "基金类型", "公允价值", "净值比例", "人民币", "（%）")
            )
        )
    return [{"stockCode": "", "stockName": name, "weightPct": float(ratio), "marketValue": float(value)}]


def partial_vietnam_stock_rows(text: str) -> list[dict]:
    """复原同一越南前十表的ISIN和数值列碎片；不使用其他期或其他基金。

    以原文VN00起始单元格划分记录，用该记录中同时出现的数量/金额列定位
    单独折行的碎片。只有唯一完整的千分组、两位金额和12位ISIN才接受。
    """
    found = list(re.finditer(r"(?m)^\s*5\.4(?:\.1)?\s+[^\n]*前十名股票及存托凭证", text))
    if not found:
        raise ValueError("MARKET_VN_QUARTER_SECTION")
    tail = text[found[-1].end() :]
    end = re.search(r"(?m)^\s*5\.5\s+", tail)
    if not end:
        raise ValueError("MARKET_VN_QUARTER_TRUNCATED")
    body = tail[: end.start()]
    compact = re.sub(r"\s+", "", body)
    if not all(s in compact for s in ("证券", "代码", "数量", "人民", "币元", "ISIN")):
        raise ValueError("MARKET_VN_QUARTER_HEADER")
    lines = body.splitlines()
    starts = [i for i, line in enumerate(lines) if re.search(r"\bVN00\b", line)]
    rows = []
    for begin, stop in zip(starts, starts[1:] + [len(lines)], strict=True):
        block = lines[begin:stop]
        ranks = [m for line in block if (m := re.match(r"^\s*(\d{1,2})\s+", line)) and m.start(1) <= 4]
        if len(ranks) != 1 or int(ranks[0][1]) != len(rows) + 1 or "越南" not in "".join(block):
            raise ValueError("MARKET_VN_QUARTER_RANK")
        code = "".join(m[0] for line in block for m in re.finditer(r"\b[A-Z0-9]{1,4}\b", line) if 22 <= m.start() <= 32)
        if not re.fullmatch(r"VN[0-9A-Z]{9}\d", code):
            raise ValueError("MARKET_VN_QUARTER_CODE")
        weight = None
        numeric_lines = []
        for line in block:
            cells = [m for m in re.finditer(r"(?<![A-Za-z0-9])[\d,.]+(?![A-Za-z0-9])", line) if m.start() >= 35]
            if (rank_match := re.match(r"^\s*(\d{1,2})\s+", line)) and rank_match.start(1) <= 4:
                if not cells or not re.fullmatch(r"\d+\.\d{2}", cells[-1][0]):
                    raise ValueError("MARKET_VN_QUARTER_WEIGHT")
                weight = cells.pop()[0]
            if len(cells) > 2:
                raise ValueError("MARKET_VN_QUARTER_COLUMNS")
            if cells:
                numeric_lines.append(cells)
        pairs = [cells for cells in numeric_lines if len(cells) == 2]
        if not pairs or weight is None:
            raise ValueError("MARKET_VN_QUARTER_COLUMNS")
        qcol = sum(c[0].start() for c in pairs) / len(pairs)
        vcol = sum(c[1].start() for c in pairs) / len(pairs)
        fragments = [[], []]
        for cells in numeric_lines:
            if len(cells) == 2:
                fragments[0].append(cells[0][0])
                fragments[1].append(cells[1][0])
            else:
                cell = cells[0]
                distances = [abs(cell.start() - qcol), abs(cell.start() - vcol)]
                if distances[0] == distances[1]:
                    raise ValueError("MARKET_VN_QUARTER_AMBIGUOUS")
                fragments[distances.index(min(distances))].append(cell[0])
        quantity, value = ("".join(f) for f in fragments)
        if not re.fullmatch(r"\d{1,3}(?:,\d{3})+", quantity) or not re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d{2}", value):
            raise ValueError("MARKET_VN_QUARTER_INCOMPLETE")
        q, amount, ratio = (Decimal(v.replace(",", "")) for v in (quantity, value, weight))
        if q <= 0 or amount <= 0 or not 0 <= ratio <= 100:
            raise ValueError("MARKET_VN_QUARTER_RANGE")
        rows.append(
            {
                "stockCode": code,
                "stockName": code,
                "marketValue": float(amount),
                "weightPct": float(ratio),
                "nameBasis": "REPORTED_MARKET_CODE",
            }
        )
    if (
        not rows
        or len(rows) > 10
        or len({r["stockCode"] for r in rows}) != len(rows)
        or sum(r["weightPct"] for r in rows) > 100
    ):
        raise ValueError("MARKET_VN_QUARTER_RANGE")
    return rows


def disclosed_multiple_funds(text: str) -> list[dict]:
    """读取原油等FOF多基金表；基金名称、金额和权重仅连接原表各自列。

    原表没有市场代码时保持代码空白，由后续资料核对绑定产品身份。排名、
    ETF类型、人民币单位及完整金额均必须可证，不反推金额或底层敞口。
    """
    found = list(re.finditer(r"(?m)^\s*5\.9\s+[^\n]*前十名基金投资明细", text))
    if not found:
        raise ValueError("MARKET_MULTI_FUND_SECTION")
    tail = text[found[-1].end() :]
    end = re.search(r"(?m)^\s*5\.10\s+", tail)
    if not end:
        raise ValueError("MARKET_MULTI_FUND_TRUNCATED")
    body = tail[: end.start()]
    compact = re.sub(r"\s+", "", body)
    # 个别原表把净值比例列纵向拆成五行，并与其他列表头交错；只接受已核的完整列片段。
    ratio_header = "净值比" in compact or all(s in compact for s in ("占基", "金资", "产净", "值比", "例(%)"))
    split_type_header = bool(
        re.search(r"(?m)^\s*序\s+基\s*金\s+作\s+公允价值\s*$", body)
        and re.search(r"(?m)^\s*号\s+类型\s+方\s+[（(]人民币元[)）]\s*$", body)
    )
    if (
        not all(s in compact for s in ("基金名称", "人民币"))
        or not ("基金类型" in compact or split_type_header)
        or not ratio_header
    ):
        raise ValueError("MARKET_MULTI_FUND_HEADER")
    lines = body.splitlines()
    anchors = [(i, m) for i, line in enumerate(lines) if (m := re.match(r"^\s*(\d{1,2})\s+", line))]
    if len(anchors) < 2:
        raise ValueError("MARKET_MULTI_FUND_RANK")
    # 金额首段上移一行时，起点随最邻近的基金名称前缀移动；管理人列不参与名称。
    # 名称可在序号上一行从更靠左处开始，不能按序号行名称的首字符截掉前缀。
    # 以序号数字列的右边界隔开名称列，保留原表所有名称碎片。
    name_col = anchors[0][1].end(1) + 1
    type_matches = list(re.finditer(r"\b(?:ETF|ETC)\b", lines[anchors[0][0]]))
    type_col = type_matches[-1].start() if type_matches else -1
    if type_col <= name_col:
        raise ValueError("MARKET_MULTI_FUND_COLUMNS")
    starts = []
    for index, _ in anchors:
        start = index
        while start > 0:
            previous = lines[start - 1]
            if re.match(r"^\s*\d{1,2}\s+", previous) or re.search(r"\b(?:ETF|ETC)\b", previous):
                break
            if previous[name_col:type_col].strip() or not previous.strip():
                start -= 1
            else:
                break
        starts.append(start)
    rows = []
    for n, (index, rank) in enumerate(anchors):
        if int(rank[1]) != n + 1:
            raise ValueError("MARKET_MULTI_FUND_RANK")
        stop = starts[n + 1] if n + 1 < len(starts) else len(lines)
        type_matches = list(re.finditer(r"\b(?:ETF|ETC)\b", lines[index]))
        row_type_col = type_matches[-1].start() if type_matches else -1
        if row_type_col <= name_col:
            raise ValueError("MARKET_MULTI_FUND_COLUMNS")
        number = re.search(r"([\d,.]+)\s+(\d+\.\d{2})\s*$", lines[index])
        if not number:
            raise ValueError("MARKET_MULTI_FUND_WEIGHT")
        amount_col = number.start(1) - 10
        pieces = [
            m[0]
            for i in range(starts[n], stop)
            for m in re.finditer(r"[\d,.]+", lines[i])
            if m.start() >= amount_col
            and not (i == index and m.start() == number.start(2))
            and "季度报告" not in lines[i]
        ]
        value = "".join(pieces)
        name_pieces = [
            line[name_col : (row_type_col if i == index else type_col)].strip()
            for i, line in enumerate(lines)
            if starts[n] <= i < stop and "季度报告" not in line
        ]
        name = " ".join(x for x in name_pieces if x)
        if not name or not re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d{2}", value):
            raise ValueError("MARKET_MULTI_FUND_INCOMPLETE")
        amount, weight = Decimal(value.replace(",", "")), Decimal(number[2])
        if amount <= 0 or not 0 < weight <= 100:
            raise ValueError("MARKET_MULTI_FUND_RANGE")
        rows.append({"stockCode": "", "stockName": name, "marketValue": float(amount), "weightPct": float(weight)})
    if sum(r["weightPct"] for r in rows) > 100:
        raise ValueError("MARKET_MULTI_FUND_RANGE")
    return rows


def gold_quarter_rows(text: str) -> tuple[list[dict], list[dict]]:
    """核对黄金ETF明确无股票、单一黄金及资产分母，不能把解析空值填成零。

    仅接受原表同一Au99.99行的数量/金额/净值权重，金额须与资产表黄金项
    一致，现金、其他资产及黄金合计须与总资产一致。净值比与总资产比不混用。
    """
    section = re.search(r"5\.1\s+报告期末基金资产组合情况(?P<body>.*?)5\.2\s+", text, re.S)
    stocks = re.search(r"5\.3\s+[^\n]*前十名股票投资明细\s+无。", text)
    metal = re.search(r"5\.7\s+[^\n]*贵金属投资明细(?P<body>.*?)5\.8\s+", text, re.S)
    if not section or not stocks or not metal or "金额（元）" not in re.sub(r"\s+", "", section["body"]):
        raise ValueError("MARKET_GOLD_QUARTER_SECTION")
    body = section["body"]
    if not re.search(r"其中[：:]股票\s+-\s+-", re.sub(r"\s+", " ", body)):
        raise ValueError("MARKET_GOLD_QUARTER_NO_STOCKS_UNPROVEN")
    assets = []
    for name in ("贵金属投资", "银行存款和结算备付金合计", "其他资产", "合计"):
        m = re.search(r"(?m)^\s*\d+\s+" + re.escape(name) + r"\s+([\d,]+\.\d{2})\s+(\d+\.\d{2})\s*$", body)
        if not m:
            raise ValueError("MARKET_GOLD_QUARTER_ASSET_ROW")
        assets.append({"name": name, "value": float(Decimal(m[1].replace(",", ""))), "weightPct": float(m[2])})
    if abs(sum(Decimal(str(a["value"])) for a in assets[:-1]) - Decimal(str(assets[-1]["value"]))) > Decimal(".02"):
        raise ValueError("MARKET_GOLD_QUARTER_BALANCE")
    if (
        assets[-1]["weightPct"] != 100
        or any(a["value"] <= 0 or not 0 < a["weightPct"] <= 100 for a in assets)
        or abs(sum(a["weightPct"] for a in assets[:-1]) - 100) > 0.02
    ):
        raise ValueError("MARKET_GOLD_QUARTER_WEIGHT")
    rows = re.findall(
        r"(?m)^\s*(\d+)\s+Au99\.99\s+Au99\.99\s+([\d,]+)\s+([\d,]+\.\d{2})\s+(\d+\.\d{2})\s*$", metal["body"]
    )
    if len(rows) != 1 or rows[0][0] != "1" or "仅持有以上贵金属" not in metal["body"]:
        raise ValueError("MARKET_GOLD_QUARTER_SINGLE_POSITION")
    _, quantity, amount, weight = rows[0]
    value, ratio = Decimal(amount.replace(",", "")), Decimal(weight)
    if Decimal(quantity.replace(",", "")) <= 0 or value != Decimal(str(assets[0]["value"])) or not 0 < ratio <= 100:
        raise ValueError("MARKET_GOLD_QUARTER_POSITION_BALANCE")
    return assets, [
        {"stockCode": "Au99.99", "stockName": "Au99.99", "marketValue": float(value), "weightPct": float(ratio)}
    ]


def parse_report(client, root: Path, fund: dict, entry: dict, source: dict) -> tuple[dict, list[dict]]:
    """正文身份和报告期先核验；完整报告沿用资产负债表对账，季报独立适配。"""
    code = fund["fund_code"]
    if entry["kind"] != "QUARTER":
        report, issues = report_from_original(code, fund["fund_name"], root, source)
        # 仅有缺项的新报告才查同报告期补充表；已完整收录的报告不会走到这里。
        if any(i["part"] in {"stocks", "bond_amount"} for i in issues):
            tables = []
            for issue in issues:
                if issue["part"] in {"stocks", "bond_amount"}:
                    tables.append(client.table(code, entry["end"], issue["part"]))
            source = {**source, "tables": tables}
            report, issues = report_from_original(code, fund["fund_name"], root, source)
        report["sourceArticleId"] = entry["id"]
        return report, issues

    pages, reprint = public_document(read_source(root, source), source, code, "季度报告")
    text = "\n".join(pages)
    if code not in text or not (
        _fund_name_matches(text, fund["fund_name"])
        or _report_identity_matches(text, fund["fund_name"], code, entry["title"])
    ):
        raise ValueError("MARKET_QUARTER_IDENTITY")
    published = publication(pages)
    if published != entry["published"] or (reprint and reprint["notice_date"][:10] != published):
        raise ValueError("MARKET_QUARTER_PUBLICATION")
    title = re.sub(r"\s+", "", unicodedata.normalize("NFKC", entry["title"]))
    if title not in re.sub(r"\s+", "", unicodedata.normalize("NFKC", text[:4000])):
        raise ValueError("MARKET_QUARTER_TITLE")
    report = {
        "id": source["sha256"],
        "sourceArticleId": entry["id"],
        "title": entry["title"],
        "endDate": entry["end"],
        "publishedDate": published,
        "sourceUrl": safe_url((reprint or {}).get("attach_url") or source["url"]),
        "fullDisclosure": False,
        "stockWeightPct": None,
        "disclosedWeightPct": 0,
        "holdings": [],
        "assets": [],
        "industries": [],
        "investmentGroups": [],
        "holdingsPublicationVersion": 1,
    }
    issues = []
    try:
        # 名称取目录已绑定的报告标题前缀，避免把ETF简称误当另一只基金。
        name = re.split(r"[0-9〇零一二三四五六七八九]{4}\s*年", entry["title"])[0]
        parsed = parse_text(pages, entry["title"], fund_code=code, fund_name=name, master_code=code)
        if parsed["report_end"] != entry["end"]:
            raise ValueError("MARKET_QUARTER_PERIOD")
        report["holdings"] = [
            {
                "stockCode": h["stock_code"],
                "stockName": h["stock_name"],
                "weightPct": float(h["nav_weight_pct"]),
                "marketValue": float(h["market_value_cny"]),
            }
            for h in parsed["holdings"]
        ]
        report["stockWeightPct"] = float(parsed["stock_nav_pct"])
        report["assets"] = [
            {"name": k, "weightPct": float(v["total_asset_pct"]), "value": float(v["value_cny"])}
            for k, v in parsed["assets"].items()
        ]
        report["industries"] = [
            {"name": v["name"], "weightPct": float(v["nav_weight_pct"])} for v in parsed["reported_industries"]
        ]
        if re.search(r"(?m)^\s*5\.[34]\s*(?:期末|报告期末)[^\n]*股票投资明细", text):
            # 通用版式可能跨行把数量并进名称、把金额小数尾数当作比例。
            # 季报明确有部分股票表时，必须再用逐行完整数字校验的结果；
            # 保留通用解析已核对的资产/行业，但不能仅因通用解析未抛错就放行股票。
            report["holdings"], groups = partial_stock_sections(text)
            report["investmentGroups"].extend(groups)
    except ValueError:
        # 公开转载表仍要核对基金、报告期、人民币单位和原表精度；不能按最新行情冒充持仓。
        try:
            report["holdings"], groups = partial_stock_sections(text)
            report["investmentGroups"].extend(groups)
        except ValueError:
            try:
                report["holdings"] = partial_overseas_stock_rows(text)
            except ValueError:
                try:
                    report["holdings"] = partial_vietnam_stock_rows(text)
                except ValueError:
                    table = client.table(code, entry["end"], "stocks")
                    try:
                        report["holdings"] = disclosed_stocks(
                            _public_table(root, table, code, entry["end"], None, None), text
                        )
                    except ValueError:
                        # 明确披露未直接持股才可接受空表；抓取失败不能自动解释为零持仓。
                        money_scope = fund.get("fund_type") == "MONEY" and "货币市场基金" in entry["title"]
                        if not _reported_no_stocks(text) and not money_scope:
                            issues.append({"part": "stocks", "reason": "MARKET_QUARTER_STOCK_TABLE_UNAVAILABLE"})
    if "黄金交易型开放式" in entry["title"] and "联接" not in entry["title"] and not report["assets"]:
        assets, rows = gold_quarter_rows(text)
        report["assets"] = assets
        report["stockWeightPct"] = 0.0  # 仅来自原文明确“无”及资产表股票“-”，不是未知填零。
        report["investmentGroups"].append(
            {"name": "贵金属持仓", "notice": "本期原表唯一黄金，净值占比独立于总资产占比。", "holdings": rows}
        )
        issues = [i for i in issues if i["part"] != "stocks"]
    if fund.get("fund_type") == "MONEY" and "货币市场基金" in entry["title"] and not report["assets"]:
        report["assets"] = _money_quarter_assets(text)
        rows = _money_quarter_bonds(text)
        fixed = next(row["value"] for row in report["assets"] if row["name"] == "固定收益投资")
        if sum(row["marketValue"] for row in rows) > fixed + 0.02:
            raise ValueError("MARKET_MONEY_BONDS_EXCEED_BALANCE")
        report["investmentGroups"].append(
            {"name": "债券持仓", "notice": "原报告前十项，按实际利率账面价值列示。", "holdings": rows}
        )
    # 季报的债券明细本来就可能只有前五名，不能用其合计推断完整债券敞口。
    if not any(group["name"] == "债券持仓" for group in report["investmentGroups"]):
        bonds = client.table(code, entry["end"], "bond_amount")
        try:
            rows = _public_table(root, bonds, code, entry["end"], None, None)
            report["investmentGroups"].append(
                {"name": "债券持仓", "notice": "本期报告主要持仓，可能仅含前五名。", "holdings": rows}
            )
        except ValueError:
            # 来源无表不代表没有债券；只有正文明确显示持有而解析缺失时记录待补。
            if re.search(r"债券投资\s+[\d,]+\.\d{2}\s+[\d.]+", text):
                issues.append({"part": "bonds", "reason": "MARKET_QUARTER_BOND_TABLE_UNAVAILABLE"})
    if "联接" in fund["fund_name"] or "FOF" in fund["fund_name"] or "基金投资明细" in text:
        try:
            rows = disclosed_single_fund(text, entry["title"])
            report["investmentGroups"].append(
                {"name": "基金持仓", "notice": "本基金直接持有的基金，未穿透底层证券。", "holdings": rows}
            )
        except ValueError:
            try:
                try:
                    rows = disclosed_multiple_funds(text)
                except ValueError:
                    # 境内FOF本节独立列示；只取本期前十，不借上期全仓。
                    head = re.search(r"(?m)^\s*6\.1\s+[^\n]*前十名基金投资明细", text)
                    if not head:
                        raise
                    tail = text[head.end() :]
                    stop = re.search(r"(?m)^\s*6\.(?:1\.1|2)\s+", tail)
                    if not stop:
                        raise ValueError("MARKET_FOF_QUARTER_TRUNCATED") from None
                    rows = _domestic_fund_rows(tail[: stop.start()])
                report["investmentGroups"].append(
                    {"name": "基金持仓", "notice": "本期直接持有的基金，未穿透底层证券。", "holdings": rows}
                )
            except ValueError:
                if "联接" in fund["fund_name"] or "FOF" in fund["fund_name"]:
                    issues.append({"part": "funds", "reason": "MARKET_QUARTER_FUND_TABLE_NOT_VERIFIED"})
    report["disclosedWeightPct"] = sum(h["weightPct"] for h in report["holdings"])
    report["holdingsNotice"] = "季度报告披露的部分持仓，不代表完整或实时持仓。"
    # 版本用于恢复旧“已发布”但仍含跨行列错位的结果。只标识本次真实解析，
    # 不改变原件编号、公开日、持仓期末或披露完整性。
    report["reportParserVersion"] = 2
    if report["holdings"]:
        try:
            foreign_verified = partial_overseas_stock_rows(text)
        except ValueError:
            pass
        else:
            if foreign_verified == report["holdings"]:
                report["stockTableVerification"] = "EXACT_PARTIAL_MARKET_COLUMNS_V1"
    if report["holdings"] and re.search(r"(?m)^\s*5\.[34]\s*(?:期末|报告期末)[^\n]*股票投资明细", text):
        try:
            verified, _ = partial_stock_sections(text)
        except ValueError:
            # 原有公开同报告期表的后备核验仍有效，但不能取得严格原文列纠错资格。
            pass
        else:
            if verified == report["holdings"]:
                report["stockTableVerification"] = "EXACT_PARTIAL_NUMERIC_COLUMNS_V1"
    if issues:
        report["holdingsNotice"] += "部分投资明细暂未核齐，相关缺口保留。"
    if not report["holdings"] and not report["investmentGroups"] and not report["assets"]:
        raise ValueError("MARKET_QUARTER_NO_VERIFIED_HOLDINGS")
    return report, issues
