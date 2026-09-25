"""002112 已披露股票的行情补齐；复用有原文哈希的研究数据，新增调用有预算。"""

import hashlib
import json
import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine
from app.integrations.tushare_sprint_stock_breadth_v2 import FIELDS, parse
from app.services.direction_1d_protocol import calendar, digest
from app.services.fund_exposure_common import ROOT, blob, initialize, now, read, save

INDICES = {"000300.SH": "沪深300", "000905.SH": "中证500"}
BASIC_FIELDS = ["ts_code", "trade_date", "turnover_rate", "turnover_rate_f"]


def reports():
    """只使用本轮成功清单中的解析版本，旧失败或旧解析文件不混入。"""
    result = read(ROOT / "report-result.json")
    if result["errors"]:
        raise ValueError("EXPOSURE_REPORTS_INCOMPLETE")
    return [read(ROOT / "reports" / (name + ".json")) for name in result["report_files"]]


def permission():
    """每次采集重新读取既有来源许可；不自动增购、不扩大登记的接口。"""
    with get_engine().connect() as c:
        row = c.execute(text("SELECT * FROM source_registry WHERE source_code='TUSHARE_PRO_FUND'")).mappings().one()
        if not row["enabled"] or row["retention_days"] <= 0:
            raise ValueError("EXPOSURE_SOURCE_UNAVAILABLE")
        return dict(row)


class Provider:
    """按原权限访问公开证券行情；请求证据不保存 token，响应大小限制为 4 MB。"""

    def __init__(self):
        self.source = permission()
        self.count = 0
        self.last = 0.0
        self.limit = initialize()["maximum_provider_requests_per_command"]

    def query(self, api, params, fields):
        if api not in {"daily", "daily_basic", "index_daily"} or api not in self.source["authorized_api_names"]:
            raise ValueError("EXPOSURE_API_NOT_AUTHORIZED")
        request = {"api": api, "params": params, "fields": fields}
        key = digest(request)
        path = ROOT / "quote-receipts" / (key + ".json")
        if path.exists():
            receipt = read(path)
            query_end = params.get("end_date", params.get("trade_date", ""))
            refresh_recent = query_end >= (now().date() - timedelta(days=7)).strftime("%Y%m%d") and (
                now() - datetime.fromisoformat(receipt["received_at"]) >= timedelta(minutes=30)
            )
            if datetime.fromisoformat(receipt["expires_at"]) > now() and not refresh_recent:
                raw = (ROOT / receipt["file"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                    raise ValueError("EXPOSURE_RAW_HASH_MISMATCH")
                return json.loads(raw), receipt
            if datetime.fromisoformat(receipt["expires_at"]) <= now():
                raise ValueError("EXPOSURE_EVIDENCE_EXPIRED")
        if self.count >= self.limit:
            raise ValueError("EXPOSURE_REQUEST_BUDGET_REACHED")
        cfg = get_settings()
        if cfg.tushare_api_url != "https://api.tushare.pro":
            raise ValueError("EXPOSURE_PROVIDER_URL_UNEXPECTED")
        time.sleep(max(0, max(1.0, 60 / self.source["rate_limit_per_minute"]) - (time.monotonic() - self.last)))
        self.count += 1
        self.last = time.monotonic()
        with httpx.Client(timeout=httpx.Timeout(25, connect=5)) as client:
            with client.stream(
                "POST",
                cfg.tushare_api_url,
                json={
                    "api_name": api,
                    "token": cfg.tushare_token.get_secret_value(),
                    "params": params,
                    "fields": ",".join(fields),
                },
            ) as response:
                response.raise_for_status()
                raw = bytearray()
                for part in response.iter_bytes():
                    raw.extend(part)
                    if len(raw) > 4_194_304:
                        raise ValueError("EXPOSURE_RESPONSE_TOO_LARGE")
        value = json.loads(raw)
        if value.get("code") != 0:
            # 不输出供应商 message，避免其包含凭据或完整请求。
            raise ValueError("EXPOSURE_PROVIDER_REJECTED")
        data = value.get("data") or {}
        if data.get("fields") != fields or not isinstance(data.get("items"), list) or len(data["items"]) >= 6000:
            raise ValueError("EXPOSURE_SCHEMA_OR_TRUNCATION")
        sha, filename = blob(bytes(raw), "json")
        received = now()
        receipt = {
            **request,
            "sha256": sha,
            "file": filename,
            "received_at": received.isoformat(),
            "expires_at": (received + timedelta(days=self.source["retention_days"])).isoformat(),
        }
        # 空返回保留原文，但不冻结为成功缓存，下一次可重新获取。
        if data["items"]:
            version_path = ROOT / "quote-receipt-versions" / (digest(receipt) + ".json")
            save(version_path, receipt)
            save(path, receipt, replace=True)
        return value, receipt


def stock_day(raw, receipt, codes, target):
    """用现有严格检查器核验日线；停牌/未上市/来源缺失未证实前统一记为原因未知。"""
    parse(raw, target)
    value = json.loads(raw)
    rows = {}
    for item in value["data"]["items"]:
        if item[0] in codes:
            rows[item[0]] = dict(zip(FIELDS[2:], item[2:], strict=True))
    return {
        "date": target,
        "rows": rows,
        "receipt": receipt,
        "universe_hash": digest(sorted(codes)),
        "missing_reason": "SOURCE_NOT_RETURNED_CAUSE_UNVERIFIED",
        "return_basis": "PROVIDER_EX_RIGHTS_PRE_CLOSE_PCT",
    }


def reuse_stock_history(codes, retention_days):
    """核验旧原文及回执后按日期复用，保留旧采集时刻与期限，不重新续期。"""
    root = ROOT.parent / "direction-1d-sprint-20260914"
    total = 0
    days = {}
    for directory in (
        "china-stock-breadth-feasibility-v1",
        "china-stock-breadth-history-v1",
        "china-stock-breadth-history-v2",
    ):
        for path in sorted((root / directory / "raw").glob("*.json")):
            receipt_path = path.parent.parent / "responses" / path.name
            if not receipt_path.exists():
                continue
            receipt = read(receipt_path)
            expires = datetime.fromisoformat(receipt["received_at"]) + timedelta(days=retention_days)
            if expires <= now():
                raise ValueError("EXPOSURE_EVIDENCE_EXPIRED")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                raise ValueError("EXPOSURE_RAW_HASH_MISMATCH")
            content = json.loads(raw)
            target = datetime.strptime(content["data"]["items"][0][1], "%Y%m%d").date().isoformat()
            target_path = ROOT / "stock-days" / (target + ".json")
            if target_path.exists():
                days[target] = read(target_path)
                continue
            evidence = {**receipt, "raw_path": str(path), "expires_at": expires.isoformat(), "reused": True}
            day = stock_day(raw, evidence, codes, target)
            save(target_path, day)
            days[target] = day
            total += 1
            if total % 200 == 0:
                print(f"STOCK_HISTORY_REUSED {total}", flush=True)
    return days


def acquire_quotes(*, incremental=False):
    """可重复运行；每轮最多 300 个新增请求，已成功项直接复用，失败与缺口分别留证。"""
    source = permission()
    rs = reports()
    codes = sorted({h["stock_code"] for r in rs for h in r["holdings"]})
    if len(codes) > 600:
        raise ValueError("EXPOSURE_STOCK_UNIVERSE_LIMIT")
    universe = {
        "codes": codes,
        "count": len(codes),
        "report_hash": digest(rs),
        "meaning": "HISTORICALLY_DISCLOSED_NOT_CURRENT_HOLDINGS",
    }
    save(ROOT / "universe.json", universe, replace=True)
    days = {} if incremental else reuse_stock_history(codes, source["retention_days"])
    provider = Provider()
    if incremental:
        provider.limit = 30
    end = now().date()
    sessions, _ = calendar()
    errors = []
    start_day = end - timedelta(days=45) if incremental else date(2021, 1, 1)
    for day in (d for d in sessions if start_day <= d <= end):
        key = str(day)
        path = ROOT / "stock-days" / (key + ".json")
        if path.exists():
            existing = read(path)
            if incremental and existing.get("universe_hash") != digest(codes):
                receipt = existing["receipt"]
                raw_path = ROOT / receipt["file"] if "file" in receipt else Path(receipt["raw_path"])
                raw = raw_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                    raise ValueError("EXPOSURE_RAW_HASH_MISMATCH")
                save(path, stock_day(raw, receipt, codes, key), replace=True)
            if not incremental or day < end - timedelta(days=7):
                continue
        try:
            value, receipt = provider.query("daily", {"trade_date": day.strftime("%Y%m%d")}, FIELDS)
            parsed = stock_day(json.dumps(value).encode(), receipt, codes, key)
            save(path, parsed, replace=path.exists())
            days[key] = parsed
        except Exception as exc:
            errors.append({"kind": "daily", "date": key, "reason": safe_error(exc)})
    # 每只股票仅请求一条连续历史，6000 行上限覆盖本轮 6 年；不请求未经登记的事件接口。
    latest_report = max(
        (r for r in rs if datetime.fromisoformat(r["available_at"]) <= now()),
        key=lambda r: (r["report_end"], r["available_at"]),
    )
    basic_codes = [h["stock_code"] for h in latest_report["holdings"]] if incremental else codes
    for code in basic_codes:
        path = ROOT / "stock-basic" / (code + ".json")
        if path.exists() and read(path)["through"] >= str(end) and not incremental:
            continue
        try:
            previous = read(path) if path.exists() else {"rows": {}, "receipts": []}
            start = date.fromisoformat(previous["through"]) - timedelta(days=7) if path.exists() else date(2021, 1, 1)
            value, receipt = provider.query(
                "daily_basic",
                {"ts_code": code, "start_date": start.strftime("%Y%m%d"), "end_date": end.strftime("%Y%m%d")},
                BASIC_FIELDS,
            )
            rows = previous["rows"].copy()
            for item in value["data"]["items"]:
                if len(item) != 4 or item[0] != code:
                    raise ValueError("EXPOSURE_BASIC_IDENTITY")
                day = datetime.strptime(item[1], "%Y%m%d").date()
                if not start <= day <= end or any(
                    v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0)
                    for v in item[2:]
                ):
                    raise ValueError("EXPOSURE_BASIC_VALUE")
                rows[str(day)] = dict(zip(BASIC_FIELDS[2:], item[2:], strict=True))
            save(
                path,
                {
                    "code": code,
                    "through": str(end),
                    "rows": rows,
                    "receipts": previous["receipts"] + [receipt],
                    "unit": "PERCENT",
                    "empty_is_not_zero": True,
                },
                replace=True,
            )
            if provider.count % 25 == 0:
                print(f"QUOTE_REQUESTS {provider.count}/{provider.limit}", flush=True)
        except Exception as exc:
            errors.append({"kind": "daily_basic", "code": code, "reason": safe_error(exc)})
            if str(exc) in {"EXPOSURE_REQUEST_BUDGET_REACHED", "EXPOSURE_PROVIDER_REJECTED"}:
                break
    for code in INDICES:
        path = ROOT / "indices" / (code + ".json")
        if path.exists() and read(path)["through"] >= str(end) and not incremental:
            continue
        try:
            previous = read(path) if path.exists() else {"rows": {}, "receipts": []}
            start = date.fromisoformat(previous["through"]) - timedelta(days=7) if path.exists() else date(2021, 1, 1)
            value, receipt = provider.query(
                "index_daily",
                {"ts_code": code, "start_date": start.strftime("%Y%m%d"), "end_date": end.strftime("%Y%m%d")},
                FIELDS,
            )
            rows = previous["rows"].copy()
            for item in value["data"]["items"]:
                if len(item) != len(FIELDS) or item[0] != code:
                    raise ValueError("EXPOSURE_INDEX_IDENTITY")
                day = datetime.strptime(item[1], "%Y%m%d").date()
                if (
                    not start <= day <= end
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in item[2:])
                    or item[2] <= 0
                    or item[3] <= 0
                ):
                    raise ValueError("EXPOSURE_INDEX_VALUE")
                if abs(100 * (item[2] / item[3] - 1) - item[4]) > 0.0002:
                    raise ValueError("EXPOSURE_INDEX_RETURN_MISMATCH")
                rows[str(day)] = dict(zip(FIELDS[2:], item[2:], strict=True))
            if not rows:
                raise ValueError("EXPOSURE_INDEX_EMPTY")
            save(
                path,
                {
                    "code": code,
                    "name": INDICES[code],
                    "through": str(end),
                    "rows": rows,
                    "receipts": previous["receipts"] + [receipt],
                },
                replace=True,
            )
        except Exception as exc:
            errors.append({"kind": "index_daily", "code": code, "reason": safe_error(exc)})
    result = {
        "at": now().isoformat(),
        "codes": len(codes),
        "stock_days": len(list((ROOT / "stock-days").glob("*.json"))),
        "basic_codes": len(list((ROOT / "stock-basic").glob("*.json"))),
        "new_requests": provider.count,
        "errors": errors,
        "new_purchase_cny": 0,
    }
    save(ROOT / "quote-result.json", result, replace=True)
    return result


def safe_error(exc):
    """只记录本模块定义的错误码；不把 HTTP 请求和配置带入日志。"""
    return (
        str(exc)
        if isinstance(exc, ValueError) and str(exc).startswith(("EXPOSURE_", "STOCK_BREADTH_", "MATERIAL_", "REPORT_"))
        else type(exc).__name__
    )
