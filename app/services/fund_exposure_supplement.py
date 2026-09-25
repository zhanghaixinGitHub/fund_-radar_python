"""002112 的第二批零采购资料：独立留存财务、分红及停牌证据，不改训练协议。"""

import hashlib
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import httpx
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, blob, now, read, save
from app.services.fund_exposure_quotes import permission, safe_error

SUPPLEMENT = ROOT / "supplement"
FINANCIAL_APIS = ("fina_indicator", "income", "balancesheet", "cashflow", "dividend")
OPERATING_APIS = ("forecast", "express", "fina_audit", "fina_mainbz", "disclosure_date")


def plan():
    """补充采集单独冻结范围；旧训练/检验年份、标签与最少样本数完全不变。"""
    path = SUPPLEMENT / "plan.json"
    if not path.exists():
        codes = read(ROOT / "universe.json")["codes"]
        save(
            path,
            {
                "created_at": now().isoformat(),
                "fund_code": "002112",
                "codes": codes,
                "start_date": "20200101",
                "end_date": now().strftime("%Y%m%d"),
                "new_purchase_cny": 0,
                "financial_apis": list(FINANCIAL_APIS),
                "maximum_new_requests": 5000,
                "maximum_response_bytes": 8_000_000,
                "minimum_request_interval_seconds": 0.45,
                "workers": 3,
                "training_eligible": False,
                "historical_first_publication_verified": False,
                "authorization": "USER_REQUEST_ALL_AVAILABLE_DATA_EXISTING_ACCOUNT_ONLY",
                "raw_retention_days": permission()["retention_days"],
            },
        )
    return read(path)


def verified_bytes(receipt):
    """所有复用必须核对内容；不因存在同名文件就视为采集成功。"""
    if receipt.get("expires_at") and datetime.fromisoformat(receipt["expires_at"]) <= now():
        raise ValueError("EXPOSURE_EVIDENCE_EXPIRED")
    raw = (ROOT / receipt["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError("EXPOSURE_RAW_HASH_MISMATCH")
    return raw


class SupplementProvider:
    """复用已登记账号，新增 API 只允许本轮真实成功的权限探针；限速跨线程和进程共享。"""

    def __init__(self, *, cache_period=None):
        self.scope = plan()
        # 持续同步每个检查日重新询问来源；同日重试仍复用已成功请求，历史缓存保持不动。
        self.cache_period = cache_period
        self.source = permission()
        probe = read(ROOT / "supplement-permission-probes.json")
        extra = SUPPLEMENT / "operating-permission-probes.json"
        if extra.exists():
            probe["probes"].extend(read(extra)["probes"])
        self.allowed = set()
        for item in probe["probes"]:
            response = json.loads(verified_bytes(item))
            if item["code"] == 0 and response.get("code") == 0 and item["rows"] > 0:
                self.allowed.add(item["api"])
        # 新 API 不擅自扩展整个产品的后台同步白名单，只在这个有独立证据的试点内使用。
        self.allowed &= set(FINANCIAL_APIS) | set(OPERATING_APIS) | {"stock_basic", "suspend_d"}
        self.lock = threading.Lock()
        self.last = 0.0
        self.count = 0

    def query(self, api, params, fields=""):
        """以请求摘要缓存完整原文；空结果也记录为来源未返回，绝不当作零值。"""
        if api not in self.allowed:
            raise ValueError("EXPOSURE_SUPPLEMENT_API_NOT_PROVEN")
        request = {"api": api, "params": params, "fields": fields}
        if self.cache_period:
            request["checked_period"] = self.cache_period
        path = SUPPLEMENT / "receipts" / (digest(request) + ".json")
        if path.exists():
            receipt = read(path)
            if datetime.fromisoformat(receipt["expires_at"]) <= now():
                raise ValueError("EXPOSURE_EVIDENCE_EXPIRED")
            return json.loads(verified_bytes(receipt)), receipt
        cfg = get_settings()
        if cfg.tushare_api_url != "https://api.tushare.pro":
            raise ValueError("EXPOSURE_PROVIDER_URL_UNEXPECTED")
        last_error = None
        for attempt in range(2):
            try:
                with self.lock:
                    if self.count >= self.scope["maximum_new_requests"]:
                        raise ValueError("EXPOSURE_REQUEST_BUDGET_REACHED")
                    interval = max(
                        self.scope["minimum_request_interval_seconds"], 60 / self.source["rate_limit_per_minute"]
                    )
                    # 补正文与补字段可能分进程运行；共用数据库锁和时钟，两个任务合计仍不突破限速。
                    with get_engine().connect() as connection:
                        connection.execute(text("SELECT pg_advisory_lock(20260925,211201)"))
                        try:
                            clock_path = SUPPLEMENT / "provider-clock.json"
                            previous = read(clock_path) if clock_path.exists() else None
                            delay = (
                                max(0, interval - (now() - datetime.fromisoformat(previous["at"])).total_seconds())
                                if previous
                                else 0
                            )
                            if delay > 60:
                                raise ValueError("EXPOSURE_PROVIDER_CLOCK_AHEAD")
                            time.sleep(delay)
                            save(clock_path, {"at": now().isoformat()}, replace=True)
                        finally:
                            connection.execute(text("SELECT pg_advisory_unlock(20260925,211201)"))
                    self.count += 1
                with httpx.Client(timeout=httpx.Timeout(30, connect=5)) as client:
                    with client.stream(
                        "POST",
                        cfg.tushare_api_url,
                        json={
                            "api_name": api,
                            "params": params,
                            "fields": fields,
                            "token": cfg.tushare_token.get_secret_value(),
                        },
                    ) as response:
                        response.raise_for_status()
                        raw = bytearray()
                        for part in response.iter_bytes():
                            raw.extend(part)
                            if len(raw) > self.scope["maximum_response_bytes"]:
                                raise ValueError("EXPOSURE_RESPONSE_TOO_LARGE")
                value = json.loads(raw)
                if value.get("code") != 0:
                    raise ValueError("EXPOSURE_PROVIDER_REJECTED")
                data = value.get("data") or {}
                if not isinstance(data.get("fields"), list) or not isinstance(data.get("items"), list):
                    raise ValueError("EXPOSURE_PROVIDER_SCHEMA_INVALID")
                if any(len(row) != len(data["fields"]) for row in data["items"]):
                    raise ValueError("EXPOSURE_PROVIDER_SCHEMA_INVALID")
                sha, filename = blob(bytes(raw), "json")
                receipt = {
                    **request,
                    "received_at": now().isoformat(),
                    "sha256": sha,
                    "file": filename,
                    "expires_at": (now() + timedelta(days=self.source["retention_days"])).isoformat(),
                }
                save(path, receipt)
                return value, receipt
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(2)
        raise last_error


def financial_rows(value, code, end, *, row_limit=1000):
    """保留报表类型、更正标记和空值；报告期不是公布日，晚于截止日的记录隔离。"""
    data = value["data"]
    if row_limit is not None and len(data["items"]) >= row_limit:
        raise ValueError("EXPOSURE_FINANCIAL_PAGINATION_REQUIRED")
    accepted, future = [], []
    for item in data["items"]:
        row = dict(zip(data["fields"], item, strict=True))
        if row.get("ts_code") != code:
            raise ValueError("EXPOSURE_FINANCIAL_STOCK_MISMATCH")
        dates = [str(row[k]).replace("-", "") for k in ("ann_date", "f_ann_date") if row.get(k)]
        if any(len(d) != 8 or not d.isdigit() for d in dates):
            raise ValueError("EXPOSURE_FINANCIAL_DATE_INVALID")
        for day in dates:
            try:
                datetime.strptime(day, "%Y%m%d")
            except ValueError as exc:
                raise ValueError("EXPOSURE_FINANCIAL_DATE_INVALID") from exc
        (future if any(d > end for d in dates) else accepted).append(row)
    return accepted, future


def acquire_financials(*, operating=False):
    """覆盖 469 家历史披露公司，按公司断点恢复；失败不会让其他公司停止。"""
    scope = plan()
    provider = SupplementProvider()
    apis = OPERATING_APIS if operating else FINANCIAL_APIS
    result_path = SUPPLEMENT / ("operating-result.json" if operating else "financial-result.json")
    if operating and not (SUPPLEMENT / "operating-plan.json").exists():
        save(
            SUPPLEMENT / "operating-plan.json",
            {
                "at": now().isoformat(),
                "codes": scope["codes"],
                "apis": list(apis),
                "start_date": scope["start_date"],
                "end_date": scope["end_date"],
                "new_purchase_cny": 0,
                "mainbz_row_limit": 100,
                "mainbz_date_meaning": "REPORT_PERIOD_NOT_ANNOUNCEMENT_DATE",
                "mainbz_missing_publication_time": "NOT_ELIGIBLE_FOR_HISTORICAL_FEATURES",
            },
        )
    result = {"started_at": now().isoformat(), "companies": len(scope["codes"]), "errors": [], "counts": {}}

    def acquire_company(code):
        counts, errors = {}, []
        for api in apis:
            output = SUPPLEMENT / "financials" / code / (api + ".json")
            try:
                if output.exists():
                    stored = read(output)
                    # 主营构成由多段查询合并，复用时也要逐段验证，不能只核对第一段。
                    for cached in stored.get("all_receipts", [stored["receipt"]]):
                        verified_bytes(cached)
                    counts[api] = len(stored["rows"])
                    continue
                params = {"ts_code": code}
                if api not in {"dividend", "disclosure_date"}:
                    params.update(start_date=scope["start_date"], end_date=scope["end_date"])
                if api == "fina_mainbz":
                    value, receipts = main_business(provider, code, scope["start_date"], scope["end_date"])
                    receipt = receipts[0]
                else:
                    value, receipt = provider.query(api, params)
                    receipts = [receipt]
                # 主营构成已经逐页核验；合并后的总量可以超过单次接口的防截断门槛。
                rows, future = financial_rows(
                    value, code, scope["end_date"], row_limit=None if api == "fina_mainbz" else 1000
                )
                save(
                    output,
                    {
                        "api": api,
                        "stock_code": code,
                        "rows": rows,
                        "future_rows": future,
                        "receipt": receipt,
                        "all_receipts": receipts,
                        "status": "AVAILABLE" if rows else "SOURCE_RETURNED_EMPTY",
                        "first_seen_at": max(r["received_at"] for r in receipts),
                        "training_eligible": False,
                        "meaning": "CURRENT_PROVIDER_VINTAGE_NOT_PROVEN_HISTORICAL_POINT_IN_TIME",
                        "publication_date_missing": api == "fina_mainbz",
                    },
                )
                counts[api] = len(rows)
            except Exception as exc:
                errors.append({"stock_code": code, "api": api, "reason": safe_error(exc)})
        return counts, errors

    total = Counter()
    with ThreadPoolExecutor(max_workers=scope["workers"]) as pool:
        futures = [pool.submit(acquire_company, code) for code in scope["codes"]]
        for completed, future in enumerate(as_completed(futures), 1):
            counts, errors = future.result()
            total.update(counts)
            result.update(completed_companies=completed, counts=dict(total), new_requests=provider.count)
            result["errors"].extend(errors)
            save(result_path, result, replace=True)
            if completed % 20 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "operating_companies" if operating else "financial_companies": completed,
                            "total": len(futures),
                            "rows": sum(total.values()),
                            "errors": len(result["errors"]),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    result.update(finished_at=now().isoformat(), status="COMPLETE" if not result["errors"] else "PARTIAL")
    save(result_path, result, replace=True)
    return result


def main_business(provider, code, start, end, fields_requested=""):
    """主营构成单次上限 100 行；日期区间触顶时二分，单日仍触顶则按产品/地区/行业拆开。"""
    receipts, fields, rows = [], None, {}

    def part(left, right, kind=None):
        nonlocal fields
        params = {"ts_code": code, "start_date": left, "end_date": right}
        if kind:
            params["type"] = kind
        value, receipt = (
            provider.query("fina_mainbz", params, fields_requested)
            if fields_requested
            else provider.query("fina_mainbz", params)
        )
        receipts.append(receipt)
        data = value["data"]
        if fields is not None and fields != data["fields"]:
            raise ValueError("EXPOSURE_MAINBZ_SCHEMA_CHANGED")
        fields = data["fields"]
        if len(data["items"]) >= 100:
            first, last = datetime.strptime(left, "%Y%m%d"), datetime.strptime(right, "%Y%m%d")
            if first < last:
                middle = first + (last - first) // 2
                part(left, middle.strftime("%Y%m%d"), kind)
                part((middle + timedelta(days=1)).strftime("%Y%m%d"), right, kind)
            elif kind is None:
                for category in ("P", "D", "I"):
                    part(left, right, category)
            else:
                raise ValueError("EXPOSURE_MAINBZ_SINGLE_PERIOD_TRUNCATED")
            return
        for row in data["items"]:
            mapped = dict(zip(fields, row, strict=True))
            if mapped.get("ts_code") != code or not left <= mapped["end_date"] <= right:
                raise ValueError("EXPOSURE_MAINBZ_SCOPE_INVALID")
            rows[digest(mapped)] = row

    part(start, end)
    return {"data": {"fields": fields, "items": list(rows.values())}}, receipts


def acquire_operating_data():
    """补公司预告、快报、审计、主营构成和财报时间；不存在的预告/快报保留为空。"""
    return acquire_financials(operating=True)


def acquire_stock_context():
    """证券当前基础资料及已知缺口的停牌事实；停牌证明不伪造缺失的行情行。"""
    provider = SupplementProvider()
    codes = set(plan()["codes"])
    basics, receipts = {}, []
    for status in ("L", "D", "P"):
        value, receipt = provider.query(
            "stock_basic",
            {"list_status": status},
            "ts_code,symbol,name,area,industry,fullname,market,exchange,list_status,list_date,delist_date,is_hs",
        )
        data = value["data"]
        if len(data["items"]) >= 10000:
            raise ValueError("EXPOSURE_STOCK_BASIC_TRUNCATED")
        for item in data["items"]:
            row = dict(zip(data["fields"], item, strict=True))
            if row["ts_code"] in codes:
                basics[row["ts_code"]] = row
        receipts.append(receipt)
    suspensions = []
    for code, start, end in (("301486.SZ", "20250408", "20250421"), ("688313.SH", "20250630", "20250710")):
        value, receipt = provider.query("suspend_d", {"ts_code": code, "start_date": start, "end_date": end})
        rows = [dict(zip(value["data"]["fields"], row, strict=True)) for row in value["data"]["items"]]
        if any(row["ts_code"] != code or not start <= row["trade_date"] <= end for row in rows):
            raise ValueError("EXPOSURE_SUSPENSION_SCOPE_MISMATCH")
        suspensions.append({"stock_code": code, "rows": rows, "receipt": receipt})
    result = {
        "at": now().isoformat(),
        "stock_basic": basics,
        "missing_codes": sorted(codes - basics.keys()),
        "receipts": receipts,
        "suspensions": suspensions,
        "industry_meaning": "CURRENT_CLASSIFICATION_NOT_HISTORICAL_INDUSTRY",
        "missing_quote_policy": "PRESERVE_MISSING_PRICE_DO_NOT_REPLACE_WITH_ZERO",
    }
    save(SUPPLEMENT / "stock-context.json", result, replace=True)
    return {
        "stock_basic": len(basics),
        "missing_codes": result["missing_codes"],
        "suspension_rows": sum(len(s["rows"]) for s in suspensions),
    }
