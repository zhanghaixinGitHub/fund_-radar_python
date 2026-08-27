"""本机维护人员受控执行 Tushare 基金同步的命令入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date

from app.core.config import get_settings
from app.services.tushare_fund_sync import TushareFundSyncService


def main(argv: Sequence[str] | None = None) -> int:
    """执行目录或指定净值日同步，输出不含 Token 和原始数据的 JSON 摘要。"""
    parser = argparse.ArgumentParser(description="Sync authorized Tushare public-fund data into fund_ai.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog", help="sync fund_company and fund_basic")
    nav_parser = subparsers.add_parser("nav", help="sync fund_nav for one explicit NAV date")
    nav_parser.add_argument("--nav-date", required=True, type=date.fromisoformat, help="NAV date in YYYY-MM-DD")
    focused_parser = subparsers.add_parser(
        "focused", help="sync only configured focused fund catalog and complete NAV history"
    )
    focused_parser.add_argument(
        "--ts-code",
        action="append",
        help="complete Tushare code; repeat to override the configured focused list, e.g. 002112.OF",
    )
    focused_parser.add_argument("--start-date", type=date.fromisoformat, help="optional NAV history start date")
    focused_parser.add_argument("--end-date", type=date.fromisoformat, help="optional NAV history end date")
    incremental_parser = subparsers.add_parser(
        "focused-incremental", help="sync only NAV dates missing after each focused fund's Tushare watermark"
    )
    incremental_parser.add_argument(
        "--ts-code",
        action="append",
        help="complete Tushare code; repeat to override the configured focused list, e.g. 002112.OF",
    )
    incremental_parser.add_argument(
        "--as-of-date", type=date.fromisoformat, help="latest NAV date to request, defaults to local current date"
    )
    arguments = parser.parse_args(argv)

    service = TushareFundSyncService()
    try:
        if arguments.command == "catalog":
            payload: object = service.sync_catalog().to_payload()
        elif arguments.command == "nav":
            payload = service.sync_nav_daily(arguments.nav_date).to_payload()
        elif arguments.command == "focused":
            ts_codes = tuple(arguments.ts_code) if arguments.ts_code else get_settings().focused_fund_ts_codes
            catalog = service.sync_focused_catalog(ts_codes)
            history = service.sync_focused_nav_history(
                ts_codes,
                start_date=arguments.start_date,
                end_date=arguments.end_date,
            )
            payload = {"catalog": catalog.to_payload(), "nav_history": history.to_payload()}
        else:
            ts_codes = tuple(arguments.ts_code) if arguments.ts_code else get_settings().focused_fund_ts_codes
            payload = service.sync_focused_nav_incremental(
                ts_codes, as_of_date=arguments.as_of_date
            ).to_payload()
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"tushare_fund_sync failed: {type(error).__name__}: {str(error)[:300]}", file=sys.stderr)
        return 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
