"""沪深300股指期权日线聚合：只用IO，认购/认沽按同月同执行价成对核验。"""

import hashlib
import json
import math
import re
from datetime import datetime

FIELDS = ["ts_code", "trade_date", "exchange", "close", "vol", "oi"]


def parse(raw: bytes, expected_days: list[str]) -> dict:
    """每次最多5交易日、10000行和2MiB；返回每日完整IO成交/持仓，不读基金标签。"""
    if (
        not raw
        or len(raw) > 2_097_152
        or not 1 <= len(expected_days) <= 5
        or len(set(expected_days)) != len(expected_days)
    ):
        raise ValueError("IO_RESPONSE_SCOPE_INVALID")
    value = json.loads(raw)
    if type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("IO_PROVIDER_REJECTED")
    data = value.get("data") or {}
    if data.get("fields") != FIELDS or not isinstance(data.get("items"), list) or not 0 < len(data["items"]) <= 10000:
        raise ValueError("IO_SCHEMA_OR_COUNT_INVALID")
    by_day, seen = {day: [] for day in expected_days}, set()
    for item in data["items"]:
        if not isinstance(item, list) or len(item) != len(FIELDS):
            raise ValueError("IO_ROW_SHAPE_INVALID")
        row = dict(zip(FIELDS, item, strict=True))
        d, code = row["trade_date"], row["ts_code"]
        if not isinstance(d, str) or not re.fullmatch(r"\d{8}", d) or not isinstance(code, str):
            raise ValueError("IO_IDENTITY_INVALID")
        day = str(datetime.strptime(d, "%Y%m%d").date())
        if day not in by_day or row["exchange"] != "CFFEX" or (day, code) in seen:
            raise ValueError("IO_DATE_EXCHANGE_OR_DUPLICATE")
        seen.add((day, code))
        if any(
            type(row[k]) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0 for k in ("close", "vol", "oi")
        ):
            raise ValueError("IO_NUMERIC_VALUE_INVALID")
        if not code.startswith("IO"):
            continue
        matched = re.fullmatch(r"IO(\d{4})-([CP])-(\d+(?:\.\d+)?)\.CFX", code)
        if not matched:
            raise ValueError("IO_CONTRACT_CODE_INVALID")
        month, side, strike = matched.groups()
        if not 1 <= int(month[2:]) <= 12 or "20" + month < d[:6] or float(strike) <= 0:
            raise ValueError("IO_CONTRACT_MONTH_OR_STRIKE_INVALID")
        by_day[day].append(row | {"month": month, "side": side, "strike": float(strike)})
    result = {}
    for day, rows in by_day.items():
        calls, puts = ([r for r in rows if r["side"] == side] for side in ("C", "P"))
        ck, pk = ({(r["month"], r["strike"]) for r in side} for side in (calls, puts))
        if not ck or ck != pk or len(ck) != len(calls) or len(pk) != len(puts):
            raise ValueError("IO_DATE_MISSING_OR_UNPAIRED_CONTRACTS")
        cv, pv = sum(r["vol"] for r in calls), sum(r["vol"] for r in puts)
        co, po = sum(r["oi"] for r in calls), sum(r["oi"] for r in puts)
        result[day] = {
            "call_contracts": len(calls),
            "put_contracts": len(puts),
            "call_volume": cv,
            "put_volume": pv,
            "call_open_interest": co,
            "put_open_interest": po,
            # +1是一手合约的固定平滑，只处理真实零成交/零持仓，不填补缺行。
            "log_put_call_volume": math.log((pv + 1) / (cv + 1)),
            "log_put_call_oi": math.log((po + 1) / (co + 1)),
            "contract_set_sha256": hashlib.sha256("\n".join(sorted(r["ts_code"] for r in rows)).encode()).hexdigest(),
            "quantity_unit": "contracts",
            "all_call_put_pairs_present": True,
        }
    return {"rows": result, "response_rows": len(data["items"]), "io_rows": sum(len(v) for v in by_day.values())}
