"""限定已核验证券身份的历史行情补充；独立版本，不修改冻结的沪深市场口径。"""

import hashlib
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from app.integrations.tushare_sprint_stock_breadth_v2 import FIELDS, validate_quote_values
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, now, read, save

STORE = ROOT / "materials-live" / "bj-history-v1"


@lru_cache(maxsize=1)
def identities():
    """只接受逐只查证的13项映射；不能根据代码前缀或名称相近猜测其他证券。"""
    value = json.loads((Path(__file__).parents[1] / "data/bse_history_identity_v1.json").read_text(encoding="utf-8"))
    return {row["historical_code"]: row for row in value["securities"]}


def parse_history(value, code, days):
    """查询别名可为新代码，输出仍保留当日旧证券身份和来源原代码；不补零。"""
    identity = identities().get(code)
    if not identity or not days or any(not identity["valid_from"] <= day < identity["switch_date"] for day in days):
        raise ValueError("REPORT_SECURITY_IDENTITY_DATE_UNVERIFIED")
    data = value.get("data") or {}
    if value.get("code") != 0 or data.get("fields") != FIELDS or not isinstance(data.get("items"), list):
        raise ValueError("REPORT_QUOTE_SCHEMA_INVALID")
    if len(data["items"]) >= 6000:
        raise ValueError("REPORT_QUOTE_TRUNCATED")
    quotes, seen = {}, set()
    for item in data["items"]:
        row = dict(zip(FIELDS, item, strict=True))
        day = datetime.strptime(row["trade_date"], "%Y%m%d").date().isoformat()
        if row["ts_code"] != identity["provider_code"] or day in seen or not min(days) <= day <= max(days):
            raise ValueError("REPORT_QUOTE_IDENTITY_OR_DATE_MISMATCH")
        seen.add(day)
        validate_quote_values(row)
        if day in days:
            quotes[day] = {key: row[key] for key in FIELDS[2:]}
    return quotes


def acquire_history(code, days, provider):
    """每股一次有界查询，响应和身份依据均留档；未返回的日期仍列为缺口。"""
    days = sorted(set(days))
    identity = identities().get(code)
    # 先核日期再请求，防止错误窗口触发外部调用。
    parse_history({"code": 0, "data": {"fields": FIELDS, "items": []}}, code, days)
    value, receipt = provider.query(
        "daily",
        {
            "ts_code": identity["provider_code"],
            "start_date": days[0].replace("-", ""),
            "end_date": days[-1].replace("-", ""),
        },
        FIELDS,
    )
    raw = (Path(receipt["raw_path"]) if receipt.get("raw_path") else ROOT / receipt["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"] or json.loads(raw) != value:
        raise ValueError("REPORT_QUOTE_RAW_CHANGED")
    quotes = parse_history(value, code, days)
    version = {
        "identity": identity,
        "requested_dates": days,
        "rows": quotes,
        "receipt": receipt,
        "training_eligible": False,
        "availability_basis": "CURRENT_SOURCE_HISTORICAL_RECONSTRUCTION",
    }
    if quotes:
        path = STORE / (digest(version) + ".json")
        if not path.exists():
            save(path, version)
    missing = [day for day in days if day not in quotes]
    return {
        "dates": days,
        "unresolved_dates": missing,
        "received_dates": len(quotes),
        "identity": identity,
        "receipt": receipt,
        "status": "UNRESOLVED" if missing else "VERIFIED_HISTORY",
        "training_eligible": False,
    }


def load_history():
    """仅供新资料核对显式调用；旧研究 QuoteDays 默认不读取新版本。"""
    result = {}
    for path in sorted(STORE.glob("*.json")):
        value = read(path)
        identity, receipt = value["identity"], value["receipt"]
        code = identity["historical_code"]
        if identity != identities().get(code):
            raise ValueError("REPORT_SECURITY_IDENTITY_CHANGED")
        if datetime.fromisoformat(receipt["expires_at"]) <= now():
            raise ValueError("EXPOSURE_EVIDENCE_EXPIRED")
        raw = (Path(receipt["raw_path"]) if receipt.get("raw_path") else ROOT / receipt["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
            raise ValueError("REPORT_QUOTE_RAW_CHANGED")
        quotes = parse_history(json.loads(raw), code, value["requested_dates"])
        if quotes != value["rows"]:
            raise ValueError("REPORT_QUOTE_AGGREGATE_CHANGED")
        for day, row in quotes.items():
            existing = result.setdefault(day, {}).get(code)
            if existing and existing["row"] != row:
                raise ValueError("REPORT_QUOTE_VERSION_CONFLICT")
            result[day][code] = {"row": row, "receipt": receipt, "identity": identity}
    return result


def security_absence(basics, day):
    """来源已证实的上市/退市边界，边界外无行情不能视作下载失败或零收益。"""
    key = day.replace("-", "")
    listed = min((str(r["list_date"]) for r in basics if r.get("list_date")), default="")
    delisted = min((str(r["delist_date"]) for r in basics if r.get("delist_date")), default="")
    if listed and key < listed:
        return "BEFORE_LISTING"
    if delisted and key >= delisted:
        return "AFTER_DELISTING"
    return None


def explain_empty_history(code, days, provider):
    """仅在历史行情为空时读证券状态；不按空响应推断停牌或退市。"""
    basics, receipts = [], []
    for status in ("D", "L", "P"):
        value, receipt = provider.query(
            "stock_basic", {"ts_code": code, "list_status": status}, "ts_code,list_date,delist_date"
        )
        data = value["data"]
        values = [dict(zip(data["fields"], r, strict=True)) for r in data["items"]]
        if any(r.get("ts_code") != code for r in values):
            raise ValueError("REPORT_QUOTE_CONTEXT_IDENTITY_MISMATCH")
        receipts.append(receipt)
        basics.extend(values)
        if values:
            break
    explained = {day: reason for day in days if (reason := security_absence(basics, day))}
    return {
        "stock_code": code,
        "start_date": min(days),
        "end_date": max(days),
        "requested_dates": days,
        "reason": "REPORT_SECURITY_OUTSIDE_LISTING" if len(explained) == len(days) else "SOURCE_RETURNED_EMPTY",
        "explained_dates": explained,
        "unresolved_dates": [d for d in days if d not in explained],
        "security_receipts": receipts,
        "retryable": len(explained) != len(days),
    }
