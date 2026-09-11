"""ETF份额的日期口径及单一变化指标；不把规模、折算或缺失解释为新增资金。"""

from datetime import date, timedelta
from decimal import Decimal

from app.services.direction_training_artifacts import digest, read_json
from app.services.direction_training_dataset import FUNDS
from app.services.trading_calendar import load_calendar

ETF_CODES = dict(zip(FUNDS, ("159736.SZ", "510500.SH", "159995.SZ"), strict=True))
LIST_DATES = dict(zip(FUNDS, ("20210927", "20130315", "20200210"), strict=True))
SPLIT_INTERVAL = ("20220825", "20220830")
YEARS = (2021, 2022, 2023, 2024)


def normalize(rows, fund):
    """原始万份统一为十进制字符串；保留非交易日及上市前行的排除证据。"""
    if fund not in ETF_CODES or len(rows) > 1464:
        raise ValueError("ETF_SHARE_SOURCE_SCOPE")
    calendar = load_calendar()
    valid = {d.strftime("%Y%m%d") for d in calendar.sessions if d.year in YEARS}
    values = {}
    for code, day, amount in rows:
        if code != ETF_CODES[fund] or not isinstance(day, str) or len(day) != 8:
            raise ValueError("ETF_SHARE_SOURCE_IDENTITY")
        parsed = date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}")
        number = Decimal(str(amount))
        if parsed.year not in YEARS or not number.is_finite() or number <= 0:
            raise ValueError("ETF_SHARE_SOURCE_VALUE")
        if day in values and Decimal(values[day]) != number:
            raise ValueError("ETF_SHARE_SOURCE_CONFLICT")
        values[day] = str(number)
    kept = {d: s for d, s in sorted(values.items()) if d in valid and d >= LIST_DATES[fund]}
    expected = {d for d in valid if d >= LIST_DATES[fund]}
    return kept, {
        "etf": ETF_CODES[fund],
        "raw_rows": len(rows),
        "unique_rows": len(values),
        "used_rows": len(kept),
        "non_calendar_dates": sorted(set(values) - valid),
        "pre_listing_dates": sorted(d for d in values if d < LIST_DATES[fund]),
        "missing_dates": sorted(expected - kept.keys()),
        "first_date": min(kept, default=None),
        "last_date": max(kept, default=None),
    }


def load_history(folder):
    history, audit = {}, {}
    for fund, code in ETF_CODES.items():
        rows = []
        for year in YEARS:
            record = read_json(folder / f"etf-share-history-{code}-{year}.json")
            if record["api_name"] != "fund_share" or record["business_code"] != 0 or record["http_status"] != 200:
                raise ValueError("ETF_SHARE_SOURCE_NOT_SUCCESSFUL")
            if record["params"] != {"ts_code": code, "start_date": f"{year}0101", "end_date": f"{year}1231"}:
                raise ValueError("ETF_SHARE_REQUEST_CHANGED")
            data = record["data"]
            if (
                data["fields"] != ["ts_code", "trade_date", "fd_share"]
                or data.get("has_more", False)
                or len(data["items"]) > 366
            ):
                raise ValueError("ETF_SHARE_SOURCE_TRUNCATED")
            if any(
                len(row) != 3 or not isinstance(row[1], str) or not row[1].startswith(str(year))
                for row in data["items"]
            ):
                raise ValueError("ETF_SHARE_SOURCE_YEAR")
            rows.extend(data["items"])
        history[fund], audit[fund] = normalize(rows, fund)
    return history, audit


def feature(history, fund, cutoff):
    """只取截止日前第二个交易日及之前五日；六个日期都必须有正份额。"""
    if fund not in ETF_CODES:
        raise ValueError("ETF_SHARE_FEATURE_FUND")
    calendar = load_calendar()
    day = date.fromisoformat(cutoff)
    index = calendar.at_or_before_index(day)
    if calendar.sessions[index] != day or day.year not in YEARS or index < 7:
        raise ValueError("ETF_SHARE_FEATURE_CUTOFF")
    dates = [d.strftime("%Y%m%d") for d in calendar.sessions[index - 7 : index - 1]]
    if any(d not in history[fund] for d in dates):
        return None, "ETF_SHARE_HISTORY_UNAVAILABLE"
    if fund == "006730" and dates[0] <= SPLIT_INTERVAL[1] and dates[-1] >= SPLIT_INTERVAL[0]:
        return None, "ETF_SHARE_SPLIT_WINDOW"
    values = [Decimal(history[fund][d]) for d in dates]
    if any(not v.is_finite() or v <= 0 for v in values):
        raise ValueError("ETF_SHARE_FEATURE_VALUE")
    value = float(values[-1] / values[0] - 1)
    # 额外留一个交易日缓冲。这里只声明历史研究可用性假设，不声称恢复了首次发布时间。
    evidence = {
        "etf": ETF_CODES[fund],
        "dates": dates,
        "shares_wan": [str(v) for v in values],
        "value": value,
        "assumed_available_at": str(calendar.sessions[index - 1]),
    }
    return evidence, None


def add_feature(item, evidence):
    return {
        **item,
        "x": [*item["x"], evidence["value"]],
        "input_hash": digest({"nav_input_hash": item["input_hash"], "etf_share": evidence}),
    }


def quarter_month(window):
    return str(date.fromisoformat(window["cal_end"]) + timedelta(days=1))[:7]
