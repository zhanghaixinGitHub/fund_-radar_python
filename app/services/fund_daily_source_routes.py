"""从已登记的逐项来源取同日数据，保留原响应并按用途核验。"""

import re
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation

from app.services.direction_1d_protocol import ZONE

SHIBOR = "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-shibor/ShiborHis"
TREASURY_REAL_YIELD = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView"
CFETS_BOND_QUOTES = "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-currency/RtimeYldCurv"
CFETS_REPO_TRADES = "https://www.chinamoney.com.cn/r/cms/www/chinamoney/data/currency/prr-md.json"
BOND_TYPES = {
    "100101": "国债", "230101": "国开债", "230102": "进出口行债", "230103": "农发债",
}


def shibor_rows(payload, required):
    if not isinstance(payload, dict) or str(payload.get("head", {}).get("rep_code")) != "200":
        raise ValueError("SHIBOR_RESPONSE_NOT_VERIFIED")
    points = {}
    for row in payload.get("records", []):
        day = row.get("showDateCN")
        if day not in required:
            continue
        if day in points:
            raise ValueError("SHIBOR_DUPLICATE_DATE")
        try:
            rates = {key: Decimal(str(row[key])) for key in ("ON", "1W")}
        except InvalidOperation as error:
            raise ValueError("SHIBOR_SOURCE_VALUE") from error
        if any(not n.is_finite() or not 0 < n < 100 for n in rates.values()):
            raise ValueError("SHIBOR_SOURCE_VALUE")
        points[day] = {key: str(n) for key, n in rates.items()}
    return points


def shibor_publication_time(required_day):
    return datetime.combine(date.fromisoformat(required_day), time(11), ZONE)


def treasury_real_yield_rows(html, cutoff):
    from bs4 import BeautifulSoup

    rows = {}
    page = BeautifulSoup(html, "html.parser")
    titles = [tag.get_text(" ", strip=True) for tag in page.select("h2, h3, h4")]
    if "Daily Treasury Par Real Yield Curve Rates" not in titles:
        raise ValueError("TREASURY_REAL_YIELD_TABLE_IDENTITY")
    for row in page.select("tr"):
        date_cell = row.select_one('td[headers~="view-field-tdr-date-table-column"]')
        rate_cell = row.select_one('td[headers~="view-field-tc-10year-table-column"]')
        if not date_cell or not rate_cell:
            continue
        day = datetime.strptime(date_cell.get_text(strip=True), "%m/%d/%Y").date()
        value = rate_cell.get_text(strip=True)
        if day > cutoff or value in {"N/A", ""}:
            continue
        rate = Decimal(value)
        if not rate.is_finite() or not -100 < rate < 100 or day in rows:
            raise ValueError("TREASURY_REAL_YIELD_ROW")
        rows[day] = str(rate)
    return {day.isoformat(): rows[day] for day in sorted(rows)[-5:]}


def cfets_bond_rows(payload, source_date, source_time):
    if not isinstance(payload, dict) or str(payload.get("head", {}).get("rep_code")) != "200":
        raise ValueError("CFETS_BOND_RESPONSE")
    data = payload.get("data", {})
    if data.get("date") != source_date or str(data.get("time", ""))[:5] != source_time:
        raise ValueError("CFETS_BOND_WRONG_DATE_OR_TIME")
    result = {}
    for row in payload.get("records", []):
        code, name = row.get("bondcode"), row.get("name")
        value = row.get("avgYield")
        if not code or not name or value is None:
            continue
        rate = Decimal(str(value))
        if not re.fullmatch(r"\d{6,10}", str(code)) or not rate.is_finite() or code in result:
            raise ValueError("CFETS_BOND_INVALID_ROW")
        result[code] = {"name": name, "yield_pct": str(rate)}
    return result


def cfets_repo_rows(payload, day, at):
    from datetime import timedelta

    if not isinstance(payload, dict) or str(payload.get("head", {}).get("rep_code")) != "200":
        raise ValueError("CFETS_REPO_RESPONSE")
    when = datetime.fromisoformat(payload["data"]["showDateCN"]).replace(tzinfo=ZONE)
    if when.date().isoformat() != day or when > at:
        raise ValueError("CFETS_REPO_WRONG_DATE_OR_TIME")
    first_next_month = (when.date().replace(day=28) + timedelta(days=4)).replace(day=1)
    result = {}
    for row in payload.get("records", []):
        code = row.get("productCode", "")
        if not re.fullmatch(r"(?:DR|R)\d{3}", code):
            continue
        if row.get("date") != when.strftime("%y-%m-%d"):
            raise ValueError("CFETS_REPO_WRONG_ROW_DATE")
        if row.get("weightedRate") in {None, "---", ""} or row.get("avgPrd") in {None, "---", ""}:
            continue
        rate, term = Decimal(str(row["weightedRate"])), Decimal(str(row["avgPrd"]))
        if (code in result or not rate.is_finite() or not term.is_finite()
                or rate <= 0 or not 0 < term <= 365):
            raise ValueError("CFETS_REPO_INVALID_ROW")
        covers = term >= (first_next_month - when.date()).days
        result[code] = {"weighted_rate_pct": str(rate), "average_term_days": str(term), "covers_month_start": covers}
    return when.isoformat(), result


def target_symbol(profile, target):
    if target not in profile.get("target_funds", []) or not re.fullmatch(r"(?:5\d{5}|1[56]\d{4})", target):
        raise ValueError("ETF_NOT_REGISTERED_FOR_FUND")
    symbol = target + (".SH" if target.startswith("5") else ".SZ")
    declared = [value for value in profile.get("market_references", {}) if value.startswith(target + ".")]
    if declared and declared != [symbol]:
        raise ValueError("ETF_EXCHANGE_IDENTITY_MISMATCH")
    return symbol


def source_values(payload, fields, symbol, date_field, value_field, dates):
    body = payload.get("data") or {}
    if payload.get("code") != 0 or body.get("fields") != fields:
        raise ValueError("ETF_RESPONSE_FIELDS")
    rows = {}
    for source_row in body.get("items", []):
        row = dict(zip(fields, source_row, strict=True))
        day = row[date_field]
        if row["ts_code"] != symbol or day not in dates:
            raise ValueError("ETF_RESPONSE_IDENTITY_OR_DATE")
        try:
            value = Decimal(str(row[value_field]))
        except InvalidOperation as error:
            raise ValueError("ETF_RESPONSE_VALUE_OR_DUPLICATE") from error
        if not value.is_finite() or value <= 0 or day in rows:
            raise ValueError("ETF_RESPONSE_VALUE_OR_DUPLICATE")
        rows[day] = str(value)
    return rows
