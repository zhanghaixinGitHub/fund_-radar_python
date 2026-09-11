"""按已冻结映射补齐Tushare历史指数，来源正文只进入本地研究包。"""

import math
from dataclasses import asdict
from datetime import date
from decimal import Decimal
from time import monotonic, sleep

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_nav_preview_engine, get_nav_sample_storage_engine
from app.integrations.tushare_market_reference import TushareIndexBasic, TushareMarketReferenceClient
from app.models.market_reference import MarketIndexCatalog
from app.repositories.market_reference_sync import (
    require_tushare_source_capabilities,
    upsert_market_index_catalog_batch,
)
from app.schemas.direction_followup import StudyInput
from app.schemas.direction_training import DirectionInput
from app.services.direction_followup_data import add_market
from app.services.direction_training_artifacts import file_hash, now, read_json, write_json
from app.services.trading_calendar import load_calendar
from app.services.tushare_market_reference_sync import _to_index_catalog_upsert

YEARS = (2021, 2022, 2023, 2024)


def metadata(codes):
    """仅查询指定指数身份及来源授权，不读取基金保留期价格和个人数据。"""
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        source = require_tushare_source_capabilities(session, ("index_basic", "index_daily"))
        if source.authorization_verified_at is None:
            raise ValueError("MARKET_SOURCE_AUTHORIZATION_MISSING")
        rows = session.scalars(
            select(MarketIndexCatalog).where(
                MarketIndexCatalog.source_id == source.source_id, MarketIndexCatalog.index_code.in_(codes)
            )
        ).all()
        return {
            "source_code": source.source_code,
            "source_id": str(source.source_id),
            "authorized_api_names": source.authorized_api_names,
            "rate_limit_per_minute": source.rate_limit_per_minute,
            "authorization_verified_at": str(source.authorization_verified_at),
            "catalog": {
                r.index_code: {
                    k: str(getattr(r, k)) if isinstance(getattr(r, k), date) else getattr(r, k)
                    for k in (
                        "index_code",
                        "display_name",
                        "market",
                        "publisher",
                        "category",
                        "base_date",
                        "list_date",
                        "expiry_date",
                    )
                }
                for r in rows
            },
        }


def client():
    settings = get_settings()
    return TushareMarketReferenceClient(
        token=settings.tushare_token.get_secret_value(),
        api_url=settings.tushare_api_url,
        connect_timeout_seconds=5,
        read_timeout_seconds=20,
        max_retries=0,
        catalog_max_rows_per_query=8000,
        max_rows_per_query=367,
    )


def validate_prices(rows, index_code):
    """逐日核对正数、日期和唯一性；缺失留给覆盖检查，绝不补值或平移日期。"""
    expected = {str(d) for d in load_calendar().sessions if date(2021, 1, 1) <= d <= date(2024, 12, 31)}
    prices = {}
    if len(rows) > 1464:
        raise ValueError("MARKET_HISTORY_BUDGET")
    for row in rows:
        if set(row) != {"date", "close"} or row["date"] not in expected or row["date"] in prices:
            raise ValueError("MARKET_DATE_DUPLICATE_OR_OUTSIDE_CALENDAR")
        value = Decimal(str(row["close"]))
        if not value.is_finite() or value <= 0 or not math.isfinite(float(value)):
            raise ValueError("MARKET_CLOSE_INVALID")
        prices[row["date"]] = str(value)
    return prices, {
        "index_code": index_code,
        "rows": len(prices),
        "planned": len(expected),
        "missing_dates": sorted(expected - prices.keys()),
        "first_date": min(prices, default=None),
        "last_date": max(prices, default=None),
    }


def ensure_catalog(folder, meta, mapping):
    """只补用户已授权的缺失指数目录；不激活正式模型基准，不更新已有目录行。"""
    missing = [r for r in mapping["funds"].values() if r["index_code"] not in meta["catalog"]]
    additions = []
    for record in missing:
        # 首次元数据探测已封存，复用其确切响应，不为目录重复调用接口。
        cached = read_json(folder / "evidence-food-index-catalog.json")
        if cached["index_code"] != record["index_code"] or cached["display_name"] != record["provider_name"]:
            raise ValueError("MISSING_INDEX_IDENTITY_NOT_VERIFIED")
        data = {
            k: date.fromisoformat(v) if k in ("base_date", "list_date", "expiry_date") and v else v
            for k, v in cached.items()
        }
        item = TushareIndexBasic(**data)
        with Session(get_nav_sample_storage_engine()) as session, session.begin():
            source = require_tushare_source_capabilities(session, ("index_basic", "index_daily"))
            existing = session.get(MarketIndexCatalog, (source.source_id, item.index_code))
            if existing is not None:
                if existing.display_name != item.display_name:
                    raise ValueError("CATALOG_CONCURRENT_IDENTITY_CHANGE")
                additions.append({"index_code": item.index_code, "status": "ALREADY_EXISTS"})
            else:
                stats = upsert_market_index_catalog_batch(
                    session, source_id=source.source_id, records=(_to_index_catalog_upsert(item),)
                )
                additions.append({"index_code": item.index_code, **asdict(stats)})
    refreshed = metadata([mapping["shared_index"], *(v["index_code"] for v in mapping["funds"].values())])
    for r in mapping["funds"].values():
        if refreshed["catalog"].get(r["index_code"], {}).get("display_name") != r["provider_name"]:
            raise ValueError("INDEX_CATALOG_NOT_VERIFIED_AFTER_WRITE")
    return {"additions": additions, "after": refreshed}


def acquire(folder, mapping, baseline_file):
    """复用沪深300，缺少的三指数按年度最多12次请求；阶段重入复用已保存响应。"""
    codes = [mapping["shared_index"], *(v["index_code"] for v in mapping["funds"].values())]
    meta = metadata(codes)
    if meta["rate_limit_per_minute"] is None or meta["rate_limit_per_minute"] <= 0:
        raise ValueError("SOURCE_RATE_LIMIT_MISSING")
    receipt = ensure_catalog(folder, meta, mapping)
    files, snapshots, coverage, requests = {}, {}, {}, []
    baseline = read_json(baseline_file)
    if {r["code"] for r in baseline["requests"]} != {"000300.SH"}:
        raise ValueError("SHARED_INDEX_SOURCE_CHANGED")
    snapshots["000300.SH"], coverage["000300.SH"] = validate_prices(baseline["prices"], "000300.SH")
    receipt["shared_index_source_file_hash"] = file_hash(baseline_file)
    gap = max(0.35, 60.0 / meta["rate_limit_per_minute"])
    with client() as api:
        last_request = 0.0
        for code in codes[1:]:
            rows, successful = [], True
            for year in YEARS:
                filename = f"market-response-{code}-{year}.json"
                path = folder / filename
                if not path.exists():
                    sleep(max(0, gap - (monotonic() - last_request)))
                    last_request = monotonic()
                    try:
                        fetched = api.list_index_daily(code, start_date=date(year, 1, 1), end_date=date(year, 12, 31))
                        if any(r.trade_date.year != year for r in fetched):
                            raise ValueError("MARKET_RESPONSE_YEAR_MISMATCH")
                        response = {
                            "status": "DOWNLOADED",
                            "api": "index_daily",
                            "code": code,
                            "year": year,
                            "retrieved_at": now(),
                            "prices": [{"date": str(r.trade_date), "close": str(r.close_price)} for r in fetched],
                        }
                    except Exception as error:
                        response = {
                            "status": "FAILED",
                            "api": "index_daily",
                            "code": code,
                            "year": year,
                            "retrieved_at": now(),
                            "error_type": type(error).__name__,
                            "prices": [],
                        }
                    write_json(path, response)
                response = read_json(path)
                if response["code"] != code or response["year"] != year:
                    raise ValueError("MARKET_CACHED_REQUEST_CHANGED")
                files[filename] = file_hash(path)
                requests.append(
                    {k: v for k, v in response.items() if k != "prices"} | {"rows": len(response["prices"])}
                )
                if response["status"] != "DOWNLOADED":
                    successful = False
                    break
                rows.extend(response["prices"])
            snapshots[code], coverage[code] = validate_prices(rows, code)
            coverage[code]["all_requests_succeeded"] = successful
    result = {
        "prices": snapshots,
        "coverage": coverage,
        "requests": requests,
        "catalog_receipt": receipt,
        "historical_first_versions_verified": False,
        "database_price_rows_written": 0,
    }
    files["market-data.json"] = write_json(folder / "market-data.json", result)
    return result, files


def augment(raw, prices):
    """复用旧市场公式，保留原七项；追加值只使用同一锚点前21个指数收盘价。"""
    item = DirectionInput.model_validate(raw)
    expanded, reason = add_market(StudyInput(**item.model_dump(), anchor_lag_sessions=1), prices)
    if reason:
        return None, reason
    data = expanded.model_dump(mode="json")
    data.pop("anchor_lag_sessions")
    return data, None
