"""本机维护人员受控执行 Tushare 基金同步的命令入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date

from app.services.tushare_fund_sync import TushareFundSyncService


def main(argv: Sequence[str] | None = None) -> int:
    """执行受控的基金同步，输出不含 Token 和原始数据的 JSON 摘要。"""
    parser = argparse.ArgumentParser(description="Sync authorized Tushare public-fund data into fund_ai.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog", help="sync fund_company and fund_basic")
    nav_parser = subparsers.add_parser("nav", help="sync fund_nav for one explicit NAV date")
    nav_parser.add_argument("--nav-date", required=True, type=date.fromisoformat, help="NAV date in YYYY-MM-DD")
    market_history_parser = subparsers.add_parser(
        "market-history", help="add explicitly specified funds to the market and backfill complete NAV history"
    )
    market_history_parser.add_argument(
        "--ts-code",
        action="append",
        required=True,
        help="complete Tushare code; repeat for every fund to add, e.g. 002112.OF",
    )
    market_history_parser.add_argument("--start-date", type=date.fromisoformat, help="optional NAV history start date")
    market_history_parser.add_argument("--end-date", type=date.fromisoformat, help="optional NAV history end date")
    incremental_parser = subparsers.add_parser(
        "market-incremental", help="sync missing NAV dates for every active fund in the market"
    )
    incremental_parser.add_argument(
        "--as-of-date", type=date.fromisoformat, help="latest NAV date to request, defaults to local current date"
    )
    detail_parser = subparsers.add_parser(
        "market-details", help="manually sync detailed data for every active fund already in the market"
    )
    detail_parser.add_argument(
        "--start-date", type=date.fromisoformat, help="history start date; defaults to 1990-01-01"
    )
    detail_parser.add_argument(
        "--end-date", type=date.fromisoformat, help="history end date; defaults to local current date"
    )
    arguments = parser.parse_args(argv)

    service = TushareFundSyncService()
    try:
        if arguments.command == "catalog":
            payload: object = service.sync_catalog().to_payload()
        elif arguments.command == "nav":
            payload = service.sync_nav_daily(arguments.nav_date).to_payload()
        elif arguments.command == "market-history":
            ts_codes = tuple(arguments.ts_code)
            catalog = service.sync_market_catalog(ts_codes)
            history = service.sync_market_nav_history(
                ts_codes,
                start_date=arguments.start_date,
                end_date=arguments.end_date,
            )
            payload = {"catalog": catalog.to_payload(), "nav_history": history.to_payload()}
        elif arguments.command == "market-details":
            detail_result = service.sync_market_details(
                history_start_date=arguments.start_date or date(1990, 1, 1),
                history_end_date=arguments.end_date,
            )
            payload = {
                "overall": detail_result.overall_outcome.to_payload(),
                "outcomes": [
                    outcome.to_payload()
                    for outcome in detail_result.outcomes
                ]
            }
        else:
            payload = service.sync_market_nav_incremental(as_of_date=arguments.as_of_date).to_payload()
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"tushare_fund_sync failed: {type(error).__name__}: {str(error)[:300]}", file=sys.stderr)
        return 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
