"""已核验上市原油产品的资料身份；不把同名、币种推导或原油价格当成基金估值。"""

import csv
import io
import itertools
import json
import re
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

POLICY = "QDII_CNY_MULTI_MARKET_LISTED_COMMODITY_V1"

# 只收录实际读过发行人/交易所原件的产品。这里不是实时持仓、行情或权重配置。
PRODUCTS = {
    "BRNT.L": ("WisdomTree Brent Crude Oil", "USD", "UK", "JE00B78CGV99", "LISTED_COMMODITY_DEBT"),
    "BNO.US": ("United States Brent Oil Fund LP", "USD", "US", None, "LISTED_FUND"),
    "CRUD.L": ("WisdomTree WTI Crude Oil", "USD", "UK", "GB00B15KXV33", "LISTED_COMMODITY_DEBT"),
    "USO.US": ("United States Oil Fund LP", "USD", "US", None, "LISTED_FUND"),
    "OILUSA.SW": ("UBS CMCI Oil SF ETF", "USD", "CH", "CH0109967858", "LISTED_FUND"),
    "1671.JP": ("Simplex WTI ETF", "JPY", "JP", None, "LISTED_FUND"),
    "1699.JP": (
        "NEXT FUNDS NOMURA Crude Oil Long Index Linked Exchange Traded",
        "JPY",
        "JP",
        "JP3047180009",
        "LISTED_FUND",
    ),
}


def _ordered_name(name: str, text: str) -> bool:
    """只在对应数值行邻域核对被管理人列打断的名称，不在整份报告里拼凑名称。"""
    remaining = iter(re.sub(r"\s+", "", text))
    return all(any(char == candidate for candidate in remaining) for char in re.sub(r"\s+", "", name))


def parent_positions(body: str) -> list[dict]:
    """解析原中报的跨列续行并按分对账；不支持的版式、空列或歧义继续拒绝。

    七项金额、权重与同一原件资产负债表互相校验。外汇表只有美元、日元两项
    非空投资余额时，枚举有界金额分组，只有唯一分组才推导各项的报表币种。
    此推导不是母基金直接披露ISIN或实际下单场所，调用方仍须保存这一局限。
    """
    from app.services.fund_holdings_publication import _section, balance

    section = _section([body], "基金投资明细|投资目标基金明细|投资基金情况")
    lines = (section or "").splitlines()
    ranks = [(i, int(m[1])) for i, line in enumerate(lines) if (m := re.match(r"^\s*(\d{1,2})\s+", line))]
    if [rank for _, rank in ranks] != list(range(1, 8)):
        raise ValueError("LISTED_PRODUCT_PARENT_COVERAGE")
    amounts = []
    rows = []
    prior = -1
    for slot, (index, rank) in enumerate(ranks):
        tokens = lines[index].split()
        suffix, weight = tokens[-2:]
        prefix_rows = [
            re.findall(r"(?<![\w])\d{1,3}(?:,\d{1,3})+(?:\.\d*)?(?![\w])", line) for line in lines[prior + 1 : index]
        ]
        prefixes = [value for values in prefix_rows for value in values]
        if len(prefixes) != 1 or not re.fullmatch(r"(?:\d+)?\.\d{2}", suffix):
            raise ValueError("LISTED_PRODUCT_PARENT_AMOUNT_LAYOUT")
        number = prefixes[0] + suffix
        if not re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d{2}", number):
            raise ValueError("LISTED_PRODUCT_PARENT_AMOUNT")
        amount, ratio = Decimal(number.replace(",", "")), Decimal(weight)
        if not amount.is_finite() or amount <= 0 or not 0 < ratio <= 100:
            raise ValueError("LISTED_PRODUCT_PARENT_RANGE")
        end = ranks[slot + 1][0] if slot + 1 < len(ranks) else len(lines)
        name, *_ = list(PRODUCTS.values())[slot]
        if not _ordered_name(name, "\n".join(lines[prior + 1 : end])):
            raise ValueError("LISTED_PRODUCT_PARENT_NAME")
        rows.append({"rank": rank, "name": name, "amount": str(amount), "weight_pct": str(ratio)})
        amounts.append(amount)
        prior = index
    values = balance([body], require_stock=False)
    total, net = Decimal(values["fund_amount"]), Decimal(values["net_assets"])
    if sum(amounts) != total or net <= 0:
        raise ValueError("LISTED_PRODUCT_PARENT_TOTAL")
    if any(abs(Decimal(r["amount"]) / net * 100 - Decimal(r["weight_pct"])) > Decimal(".006") for r in rows):
        raise ValueError("LISTED_PRODUCT_PARENT_WEIGHT")
    start = body.index("6.4.13.4.2.1 外汇风险敞口")
    stop = body.index("上年度末", start)
    fx = body[start:stop]
    if not all(word in fx for word in ("美元折合人民", "港币折合人民", "欧元折合人民", "日元折合人民", "其他币种折合")):
        raise ValueError("LISTED_PRODUCT_PARENT_CURRENCY_COLUMNS")
    fx_lines = [line.split() for line in fx.splitlines() if line.strip()]
    first = next((i for i, row in enumerate(fx_lines) if row[0] == "交易性金"), None)
    if first is None or first + 1 >= len(fx_lines):
        raise ValueError("LISTED_PRODUCT_PARENT_CURRENCY_ROW")
    cells = fx_lines[first][1:]
    tail = fx_lines[first + 1]
    if len(cells) != 6 or len(tail) != 3 or tail[0] != "融资产" or [cells[i] for i in (1, 2, 4)] != ["-"] * 3:
        raise ValueError("LISTED_PRODUCT_PARENT_CURRENCY_LAYOUT")
    usd = Decimal((cells[0] + tail[1]).replace(",", ""))
    yen = Decimal(cells[3].replace(",", ""))
    fx_total = Decimal((cells[5] + tail[2]).replace(",", ""))
    if usd <= 0 or yen <= 0 or usd + yen != total or fx_total != total:
        raise ValueError("LISTED_PRODUCT_PARENT_CURRENCY_TOTAL")
    choices = [
        bits
        for bits in itertools.product((False, True), repeat=7)
        if sum((amount for amount, selected in zip(amounts, bits, strict=True) if selected), Decimal(0)) == usd
    ]
    if len(choices) != 1:
        raise ValueError("LISTED_PRODUCT_PARENT_CURRENCY_AMBIGUOUS")
    for row, selected in zip(rows, choices[0], strict=True):
        row["currency"] = "USD" if selected else "JPY"
    return rows


def verified_products(material: dict, documents: dict, evidence: dict) -> bool:
    """逐项核对母基金原文、发行人底层投向与同一币种上市身份；没有就绪开关。"""
    try:
        report = documents[evidence["report_ref"]]
        positions = parent_positions(report["body"])
        companies = {c["code"]: c for c in material["companies"]}
        entries = evidence["listed_fund_evidence"]
        if (
            len(material["companies"]) != len(PRODUCTS)
            or set(companies) != set(PRODUCTS)
            or set(entries) != set(PRODUCTS)
        ):
            return False
        # 推导币种、ETC法律结构及实际交易场所未直接披露必须随每次分析保留。
        required = material.get("facts", {}).get(evidence.get("listed_identity_limitation_fact"), {})
        if required.get("required_in_analysis") is not True or required.get("direction_eligible") is not False:
            return False
        for slot, (symbol, (name, currency, market, isin, asset_type)) in enumerate(PRODUCTS.items()):
            company, entry, position = companies[symbol], entries[symbol], positions[slot]
            product = documents[entry["product_ref"]]
            body, host = product["body"], urlparse(product["url"]).hostname
            if (
                product.get("kind") != "FUND_PRODUCT_INFO"
                or company.get("asset_type") != asset_type
                or position["currency"] != currency
                or Decimal(str(company["weight_pct"])) != Decimal(position["weight_pct"])
                or not entry.get("parent_quote")
                or entry["parent_quote"] not in report["body"]
                or not entry.get("product_quote")
                or entry["product_quote"] not in body
            ):
                return False
            quote = company.get("quote") or {}
            if quote.get("currency") != currency or quote.get("valuation_market") != market:
                return False
            if market == "US":
                if host != "www.uscfinvestments.com" or "NYSE Arca" not in body or "futures" not in body.lower():
                    return False
            elif market == "UK":
                ticker = symbol.split(".")[0]
                listing = re.search(r"LSE USD " + ticker + r"[^\n]+", body)
                if (
                    host != "dataspanapi.wisdomtree.com"
                    or isin not in product["url"]
                    # 原件以“Investment Objective”起头，产品名在首段；仍须同一ISIN上市行。
                    or name not in body[:300]
                    or not listing
                    or isin not in listing[0]
                    or "Base Currency USD" not in body
                    or "Currency Hedged No" not in body
                    or "Debt security / ETC" not in body
                    or "futures" not in body.lower()
                ):
                    return False
            elif market == "CH":
                if host not in {"swissfunddata.ch", "www.swissfunddata.ch", "sfd.lbswiss.ch"}:
                    return False
                at = body.index(isin)
                start = body.rfind("C. – CMCI Oil SF ETF (USD) A-dis", 0, at)
                if start < 0 or "OILUSA" not in body[start : at + 160]:
                    return False
                listing = documents[entry["listing_ref"]]
                if (
                    listing.get("kind") != "PRODUCT_LISTING_INFO"
                    or urlparse(listing["url"]).hostname != "www.six-group.com"
                ):
                    return False
                matches = [
                    r for r in csv.DictReader(io.StringIO(listing["body"]), delimiter=";") if r.get("ISIN") == isin
                ]
                if len(matches) != 1 or any(
                    matches[0].get(k) != v
                    for k, v in {
                        "ValorSymbol": "OILUSA",
                        "TradingBaseCurrency": "USD",
                        "Exchange": "XSWX",
                    }.items()
                ):
                    return False
            else:
                allowed = (
                    {"www.simplexasset.com", "www.simplexasset.co.jp"} if symbol == "1671.JP" else {"nextfunds.jp"}
                )
                if host not in allowed or symbol.split(".")[0] not in body:
                    return False
                listing = documents[entry["listing_ref"]]
                if (
                    listing.get("kind") != "PRODUCT_LISTING_INFO"
                    or urlparse(listing["url"]).hostname != "www.jpx.co.jp"
                ):
                    return False
                text = re.sub(r"\s+", " ", listing["body"])
                if symbol.split(".")[0] + " " + name not in text or "Futures Type" not in text:
                    return False
        return True
    except (KeyError, TypeError, ValueError, IndexError, InvalidOperation, StopIteration):
        return False


def european_calendar(market: str, source: dict, year: int) -> tuple[set[date], date, date]:
    """读取官方日历已明确覆盖的区间；英国只返回实际渲染表中可证实的日期范围。"""
    host = urlparse(source.get("url", "")).hostname
    if source.get("kind") != "EXCHANGE_CALENDAR":
        raise ValueError("LISTED_PRODUCT_CALENDAR_KIND")
    data = json.loads(source["body"])
    if market == "CH":
        if host != "www.six-group.com" or f"/{year}/calendar.json" not in source["url"]:
            raise ValueError("LISTED_PRODUCT_SWISS_CALENDAR_SOURCE")
        items = [r for r in data["itemList"] if r.get("mic") == "XSWX"]
        if data.get("status") != "Ok" or len(items) != 1 or len(items[0]["holidays"]) < 10:
            raise ValueError("LISTED_PRODUCT_SWISS_CALENDAR_INCOMPLETE")
        closed = {datetime.strptime(str(r["date"]), "%Y%m%d").date() for r in items[0]["holidays"]}
        if any(day.year != year for day in closed):
            raise ValueError("LISTED_PRODUCT_SWISS_CALENDAR_YEAR")
        return closed, date(year, 1, 1), date(year, 12, 31)
    if market != "UK" or host != "www.londonstockexchange.com":
        raise ValueError("LISTED_PRODUCT_UK_CALENDAR_SOURCE")
    if "each weekday" not in data["weekday_clause"] or "England & Wales" not in data["weekday_clause"]:
        raise ValueError("LISTED_PRODUCT_UK_WEEKDAY_CLAUSE")
    start, end = date.fromisoformat(data["coverage_start"]), date.fromisoformat(data["coverage_end"])
    if start.year != year or end.year != year or start >= end:
        raise ValueError("LISTED_PRODUCT_UK_CALENDAR_COVERAGE")
    closed, early = set(), set()
    for row in data["exchange_rows"]:
        parsed = datetime.strptime(row[0], "%A %d %B %Y")
        day = parsed.date()
        if parsed.strftime("%A") != row[0].split()[0] or not start <= day <= end:
            raise ValueError("LISTED_PRODUCT_UK_CALENDAR_DATE")
        if "NON-trading day" in row[2]:
            closed.add(day)
        elif "12:30 London time" in row[2]:
            early.add(day)
        else:
            raise ValueError("LISTED_PRODUCT_UK_CALENDAR_ROW")
    if len(closed) < 3 or len(early) < 2:
        raise ValueError("LISTED_PRODUCT_UK_CALENDAR_INCOMPLETE")
    return closed | early, start, end


def european_market_ready(market: str, quote: dict, documents: dict, evidence: dict, base: str, target: str) -> bool:
    """检查补充的两处市场；时间是保守完成边界，不能冒充最后一笔成交时间。

    伦敦采用官方参数中的场内人工成交报告结束时点，晚于收市竞价、延长及随机
    阶段；瑞士采用ETF竞价完成时点。实际报价/最后成交日期仍须由来源另行保留。
    """
    try:
        zone, hour, minute = ("Europe/London", 17, 15) if market == "UK" else ("Europe/Zurich", 17, 35)
        if market not in {"UK", "CH"} or quote.get("close_timestamp_kind") != "EXCHANGE_COMPLETION_BOUND":
            return False
        bound = datetime.combine(date.fromisoformat(base), time(hour, minute), ZoneInfo(zone))
        if datetime.fromisoformat(quote["closed_at"]) != bound:
            return False
        source = documents[quote["source_ref"]]
        hours = documents[evidence["market_hours_refs"][market]]
        calendar = documents[evidence["calendar_refs"][market]]
        if source.get("kind") != "MARKET_DATA" or datetime.fromisoformat(source["available_at"]) < bound:
            return False
        if hours.get("kind") != "EXCHANGE_TRADING_HOURS" or calendar.get("kind") != "EXCHANGE_CALENDAR":
            return False
        if market == "CH":
            if any(urlparse(d["url"]).hostname != "www.six-group.com" for d in (hours, calendar)):
                return False
            if not all(v in hours["body"] for v in ("17:35", "Run Auction and Close", "Exchange Traded Funds")):
                return False
        else:
            if any(
                urlparse(d["url"]).hostname not in {"www.londonstockexchange.com", "docs.londonstockexchange.com"}
                for d in (hours, calendar)
            ):
                return False
            if not all(
                s in hours["body"]
                for s in ("End of TRADEcho manual reporting - On Exchange", "17:15", "closing auction", "random")
            ):
                return False
        closed, start, end = european_calendar(market, calendar, date.fromisoformat(target).year)
        return all(start <= (day := date.fromisoformat(value)) <= end and day not in closed for value in (base, target))
    except (KeyError, TypeError, ValueError, IndexError):
        return False
