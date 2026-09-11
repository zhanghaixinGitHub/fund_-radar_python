"""量价研究的有界采集和特征：成交额缺失不可补值，价格仅作版本核对。"""

import math
from datetime import date
from decimal import Decimal
from time import monotonic, sleep

from app.schemas.direction_training import DirectionInput
from app.services.direction_market_data import YEARS, client, metadata
from app.services.direction_training_artifacts import digest, file_hash, now, read_json, write_json
from app.services.trading_calendar import load_calendar


def validate_rows(rows, expected_prices):
    """返回日期到成交额的映射；零值、缺值保留原因，重复/越界/行情修订拒绝整个来源。"""
    expected = {str(d) for d in load_calendar().sessions if date(2021, 1, 1) <= d <= date(2024, 12, 31)}
    if len(rows) > 1464:
        raise ValueError("VOLUME_RESPONSE_BUDGET")
    amounts, seen, invalid = {}, set(), {}
    for row in rows:
        day = row["date"]
        if set(row) != {"date", "close", "amount"} or day not in expected or day in seen:
            raise ValueError("VOLUME_DATE_OR_FIELDS")
        seen.add(day)
        close = Decimal(str(row["close"]))
        if not close.is_finite() or close <= 0 or close != Decimal(expected_prices[day]):
            raise ValueError("VOLUME_PRICE_SOURCE_REVISION")
        value = Decimal(str(row["amount"])) if row["amount"] is not None else None
        if value is None or not value.is_finite() or value <= 0 or not math.isfinite(float(value)):
            invalid[day] = "MISSING_NONPOSITIVE_OR_NONFINITE_AMOUNT"
        else:
            amounts[day] = str(value)
    return amounts, {
        "planned": len(expected),
        "received": len(seen),
        "usable": len(amounts),
        "missing_dates": sorted(expected - seen),
        "invalid_dates": invalid,
    }


def fetch_response(api, code, start, end):
    """请求范围和响应身份一同封存；失败只保留类型，避免异常携带凭据。"""
    response = {"api": "index_daily", "code": code, "start": str(start), "end": str(end), "retrieved_at": now()}
    try:
        rows = api.list_index_activity(code, start_date=start, end_date=end)
        if any(r.index_code != code or not start <= r.trade_date <= end for r in rows):
            raise ValueError("VOLUME_REQUEST_RANGE")
        response.update(
            status="DOWNLOADED",
            rows=[
                {
                    "date": str(r.trade_date),
                    "close": str(r.close_price),
                    "amount": str(r.amount) if r.amount is not None else None,
                }
                for r in rows
            ],
        )
    except Exception as error:
        response.update(status="FAILED", error_type=type(error).__name__, rows=[])
    return response


def acquire(folder, mapping, old_prices):
    """固定三指数各四个年度请求，零重试；按来源频率限流，重入复用已保存响应。"""
    codes = [v["index_code"] for v in mapping["funds"].values()]
    meta = metadata(codes)
    if any(c not in meta["catalog"] for c in codes):
        raise ValueError("VOLUME_CATALOG_MISSING")
    gap = max(0.35, 60 / (meta["rate_limit_per_minute"] or 200))
    last, files, requests, amounts, coverage = 0.0, {}, [], {}, {}
    with client() as api:
        for code in codes:
            rows, success = [], True
            for year in YEARS:
                start, end = date(year, 1, 1), date(year, 12, 31)
                path = folder / f"volume-response-{code}-{year}.json"
                if not path.exists():
                    sleep(max(0, gap - (monotonic() - last)))
                    last = monotonic()
                    write_json(path, fetch_response(api, code, start, end))
                response = read_json(path)
                if (response["code"], response["start"], response["end"]) != (code, str(start), str(end)):
                    raise ValueError("VOLUME_CACHED_REQUEST_CHANGED")
                files[path.name] = file_hash(path)
                requests.append({k: v for k, v in response.items() if k != "rows"} | {"rows": len(response["rows"])})
                if response["status"] != "DOWNLOADED":
                    success = False
                    break
                rows.extend(response["rows"])
            amounts[code], coverage[code] = validate_rows(rows, old_prices[code])
            coverage[code]["all_requests_succeeded"] = success
    result = {
        "amounts": amounts,
        "coverage": coverage,
        "requests": requests,
        "source_metadata": meta,
        "unit": "THOUSAND_CNY",
        "database_written": False,
        "historical_first_versions_verified": False,
    }
    files["volume-data.json"] = write_json(folder / "volume-data.json", result)
    return result, files


def augment(raw, amounts):
    """B=原七列加成交额5日均值/20日均值；C另加20日基金收益×(该比值-1)。

    两个均值都结束于同一上一交易日，5日包含在20日内。仅使用原始正成交额，
    不拟合阈值、裁剪或填缺。成交额表示活跃度，不表示净流入。
    """
    item = DirectionInput.model_validate(raw)
    cal = load_calendar()
    end = cal.at_or_before_index(item.cutoff) - 1
    if end < 19 or item.anchor != cal.sessions[end]:
        return None, None, "VOLUME_ANCHOR_OR_PREWARM"
    days = [str(d) for d in cal.sessions[end - 19 : end + 1]]
    if any(d not in amounts for d in days):
        return None, None, "VOLUME_MISSING_SESSION"
    values = [float(amounts[d]) for d in days]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        return None, None, "VOLUME_INVALID_AMOUNT"
    activity = (sum(values[-5:]) / 5) / (sum(values) / 20)
    interaction = item.x[1] * (activity - 1)
    if not math.isfinite(activity) or not math.isfinite(interaction):
        return None, None, "VOLUME_INVALID_FEATURE"
    base = item.model_dump(mode="json")
    inputs = []
    for additions in ([activity], [activity, interaction]):
        inputs.append(
            {
                **base,
                "x": [*item.x, *additions],
                "input_hash": digest(
                    {
                        "original": item.input_hash,
                        "amount_dates": days,
                        "amounts": [amounts[d] for d in days],
                        "additions": additions,
                    }
                ),
            }
        )
    return *inputs, None


def augment_inputs(item, mapping, data):
    b, c, reason = augment(item, data["amounts"][mapping["funds"][item["fund"]]["index_code"]])
    return b, c, reason, reason
