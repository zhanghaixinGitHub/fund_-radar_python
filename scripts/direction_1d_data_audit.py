"""只读检查盘中研究授权与既有日线档案，绝不调用来源接口或扩展权限。"""

import argparse
import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path

from app.db.session import get_engine
from app.services import direction_1d_training as storage
from app.services.direction_1d_protocol import calendar, canonical
from sqlalchemy import text


def audit(market_archive: Path) -> dict:
    with get_engine().connect() as c, c.begin():
        c.execute(text("SET TRANSACTION READ ONLY"))
        sources = [
            dict(r)
            for r in c.execute(
                text("""SELECT source_code,enabled,authorized_api_names,
          authorization_verified_at,rate_limit_per_minute,retention_days FROM source_registry ORDER BY source_code""")
            ).mappings()
        ]
        result = {
            "checked_at": str(c.execute(text("SELECT clock_timestamp()")).scalar_one()),
            "sources": sources,
            "minute_table_names": list(
                c.execute(
                    text("""SELECT tablename FROM pg_tables WHERE schemaname='public'
              AND (tablename LIKE '%minute%' OR tablename LIKE '%intraday%') ORDER BY tablename""")
                ).scalars()
            ),
            "benchmark_series_count": c.execute(text("SELECT count(*) FROM benchmark_series")).scalar_one(),
            "benchmark_daily_count": c.execute(text("SELECT count(*) FROM benchmark_nav_daily")).scalar_one(),
            "fund_exchange_daily_count": c.execute(text("SELECT count(*) FROM fund_exchange_daily")).scalar_one(),
            "market_event_count": c.execute(text("SELECT count(*) FROM market_event")).scalar_one(),
        }
    licensed = {api for s in sources if s["enabled"] for api in s["authorized_api_names"]}
    required = {"idx_mins", "rt_idx_min"}
    result["intraday"] = {
        "status": "DATA_AUTHORIZATION_PENDING" if required - licensed else "HISTORY_AND_TIMESTAMP_CHECK_REQUIRED",
        "missing_project_api_authorizations": sorted(required - licensed),
        "slots_china_time": ["08:20", "11:30", "14:30"],
        "provider_account_permission_tested": False,
        "provider_api_calls": 0,
        "enabled": False,
        "official_reference": "https://tushare.pro/document/2?doc_id=419",
        "note": "登记授权不足，不代表已证明供应商账户无权限；按用户限制不新增权限。日线不能还原盘中输入。",
    }
    raw = market_archive.read_bytes()
    content = storage.read(market_archive)
    expected = {str(d) for d in calendar()[0] if date(2021, 1, 1) <= d <= date(2024, 12, 31)}
    coverage = {}
    for code, prices in content["prices"].items():
        if any(not Decimal(value).is_finite() or Decimal(value) <= 0 for value in prices.values()):
            raise ValueError("INVALID_ARCHIVED_INDEX_PRICE")
        coverage[code] = {
            "count": len(prices),
            "first": min(prices),
            "last": max(prices),
            "missing_development_dates": sorted(expected - prices.keys()),
        }
    result["existing_daily_archive"] = {
        "path": str(market_archive.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "coverage": coverage,
        "source": "TUSHARE_PRO_FUND",
        "licensed_daily_api_present": "index_daily" in licensed,
        "historical_first_versions_verified": content["historical_first_versions_verified"],
        "usage": "后续市场信息试验可检查复用；本轮未把这些价格加入统一模型FIT。",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.market_archive)
    storage.write_new(args.out, report)
    print(canonical({"intraday": report["intraday"], "daily_coverage": report["existing_daily_archive"]["coverage"]}))
