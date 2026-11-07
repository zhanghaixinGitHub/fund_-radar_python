"""基金披露公司资料的共享采集缓存；只调用登记已授权的接口，不扩大002112试点许可。"""

# 原始响应统一归档外接盘；历史路径只作为数据库别名，不生成C盘源文件。
import re
import time
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.services import material_content as _sources
from app.services.direction_1d_protocol import calendar, digest
from app.services.fund_exposure_common import now, read
from app.services.fund_exposure_quotes import permission, published_quote_end
from app.services.fund_materials_store import versioned_save
from app.services.material_content import content_digest, location_exists, source_bytes

_FIELDS = {
    "daily": ["ts_code", "trade_date", "close", "pre_close", "pct_chg", "amount"],
    "income": ["ts_code", "ann_date", "end_date", "revenue", "n_income"],
    "balancesheet": ["ts_code", "ann_date", "end_date", "total_assets", "total_liab"],
    "cashflow": ["ts_code", "ann_date", "end_date", "n_cashflow_act"],
}


class MarketCompanyCollector:
    """按证券和业务日期去重，每批最多200个新请求；行情日线整市场一次取得后分配。"""

    def __init__(self, root: Path):
        self.root = root
        self.count, self.last = 0, 0.0
        self.source = None
        self.client = None
        self._daily_cache = None
        self._companies = {}
        self.checked_at = now()
        self._quote_end = None

    def close(self):
        if self.client is not None:
            self.client.close()

    def query(self, api: str, params: dict) -> tuple[list[dict], dict]:
        if api not in _FIELDS:
            raise ValueError("MARKET_COMPANY_API_UNSUPPORTED")
        self.source = self.source or permission()
        if api not in self.source["authorized_api_names"] or not self.source.get("authorization_verified_at"):
            raise ValueError("MARKET_COMPANY_API_NOT_AUTHORIZED")
        key = digest({"api": api, "params": params, "fields": _FIELDS[api]})
        if api == "daily" and self._daily_cache and self._daily_cache[0] == key:
            return self._daily_cache[1:]
        path = self.root / "queries" / (key + ".json")
        current = now()
        if location_exists(path):
            cached = read(path)
            cache_age = timedelta(minutes=30) if api == "daily" else timedelta(hours=24)
            if (
                cached.get("rows")
                and current < datetime.fromisoformat(cached["expires_at"])
                and current - datetime.fromisoformat(cached["received_at"]) < cache_age
            ):
                raw = source_bytes(self.root / cached["file"])
                if content_digest(raw) != cached["sha256"]:
                    raise ValueError("MARKET_COMPANY_SOURCE_HASH")
                if api == "daily":
                    self._daily_cache = (key, cached["rows"], cached)
                return cached["rows"], cached
        if self.count >= 200:
            raise ValueError("MARKET_COMPANY_REQUEST_LIMIT")
        settings = get_settings()
        token = settings.tushare_token.get_secret_value()
        if not token:
            raise ValueError("MARKET_COMPANY_SOURCE_UNAVAILABLE")
        self.client = self.client or httpx.Client(timeout=httpx.Timeout(25, connect=5), follow_redirects=False)
        time.sleep(max(0, 0.45 - (time.monotonic() - self.last)))
        self.last = time.monotonic()
        self.count += 1
        response = self.client.post(
            settings.tushare_api_url,
            json={"api_name": api, "token": token, "params": params, "fields": ",".join(_FIELDS[api])},
        )
        response.raise_for_status()
        raw = response.content
        if len(raw) > 8_000_000:
            raise ValueError("MARKET_COMPANY_SOURCE_SIZE")
        value = response.json()
        data = value.get("data") or {}
        if value.get("code") != 0 or data.get("fields") != _FIELDS[api]:
            raise ValueError("MARKET_COMPANY_SOURCE_SCHEMA_OR_PERMISSION")
        rows = data.get("items")
        if not isinstance(rows, list) or len(rows) >= 10000 or any(len(r) != len(_FIELDS[api]) for r in rows):
            raise ValueError("MARKET_COMPANY_SOURCE_ROWS")
        rows = [dict(zip(_FIELDS[api], row, strict=True)) for row in rows]
        sha = content_digest(raw)
        raw_path = self.root / "sources" / (sha + ".json")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if location_exists(raw_path):
            if content_digest(source_bytes(raw_path)) != sha:
                raise ValueError("MARKET_COMPANY_SOURCE_HASH")
        else:
            try:
                _sources.write_source(raw_path, raw)
            except FileExistsError:
                if content_digest(source_bytes(raw_path)) != sha:
                    raise ValueError("MARKET_COMPANY_SOURCE_HASH") from None
        receipt = {
            "file": raw_path.relative_to(self.root).as_posix(),
            "sha256": sha,
            "api": api,
            "params": params,
            "received_at": current.isoformat(),
            "expires_at": (current + timedelta(days=self.source["retention_days"])).isoformat(),
            "rows": rows,
        }
        # 空响应是真实失败尝试，不冻结为24小时成功缓存；后续资料披露后允许补查。
        versioned_save(self.root / "query-attempts" / (digest(receipt) + ".json"), receipt)
        if rows:
            versioned_save(path, receipt)
        if api == "daily" and rows:
            self._daily_cache = (key, rows, receipt)
        return rows, receipt

    def company(self, code: str) -> dict:
        """行情和财务分别报告覆盖；未登记的财务权限不能被002112历史探针结果替代。"""
        if code in self._companies:
            return deepcopy(self._companies[code])
        value = {
            "stockCode": code,
            "quote": None,
            "financials": {},
            "sources": [],
            "gaps": [],
            "attempts": [],
            "errors": [],
        }
        if not re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", code):
            value["gaps"].append("该市场尚无已登记的公司行情与财务采集适配")
            return value
        if self._quote_end is None:
            sessions, _ = calendar()
            self._quote_end = published_quote_end(sessions, self.checked_at)
        end = self._quote_end
        for api in _FIELDS:
            try:
                params = {"trade_date": end.strftime("%Y%m%d")} if api == "daily" else {"ts_code": code}
                rows, receipt = self.query(api, params)
                value["attempts"].append(
                    {
                        "api": api,
                        "rows": len(rows),
                        "receipt": {
                            key: receipt[key]
                            for key in ("file", "sha256", "api", "params", "received_at", "expires_at")
                        },
                    }
                )
                own = [row for row in rows if row["ts_code"] == code]
                if api == "daily":
                    if len(own) != 1 or own[0]["trade_date"] != end.strftime("%Y%m%d"):
                        raise ValueError("MARKET_COMPANY_QUOTE_MISSING")
                    for field in ("close", "pre_close", "pct_chg", "amount"):
                        number = Decimal(str(own[0][field]))
                        if (
                            not number.is_finite()
                            or (field in {"close", "pre_close"} and number <= 0)
                            or (field == "amount" and number < 0)
                        ):
                            raise ValueError("MARKET_COMPANY_QUOTE_VALUE")
                    value["quote"] = {**own[0], "date": end.isoformat()}
                else:
                    own = [
                        r
                        for r in own
                        if r.get("ann_date")
                        and r.get("end_date")
                        and r["ann_date"] <= now().strftime("%Y%m%d")
                        and r["end_date"] <= r["ann_date"]
                    ]
                    if not own:
                        raise ValueError("MARKET_COMPANY_FINANCIALS_MISSING")
                    value["financials"][api] = own
                value["sources"].append({k: receipt[k] for k in ("file", "sha256", "api", "received_at", "expires_at")})
            except (ValueError, OSError, httpx.HTTPError, KeyError, InvalidOperation) as error:
                value["gaps"].append(f"{api}：{type(error).__name__}")
                value["errors"].append(
                    {
                        "api": api,
                        "type": type(error).__name__,
                        "reason": str(error) if isinstance(error, ValueError) else type(error).__name__,
                    }
                )
        versioned_save(self.root / "companies" / (code + ".json"), value)
        self._companies[code] = deepcopy(value)
        return value
