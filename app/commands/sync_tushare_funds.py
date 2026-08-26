"""本机维护人员受控执行 Tushare 基金同步的命令入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date

from app.services.tushare_fund_sync import TushareFundSyncService


def main(argv: Sequence[str] | None = None) -> int:
    """执行目录或指定净值日同步，输出不含 Token 和原始数据的 JSON 摘要。"""
    parser = argparse.ArgumentParser(description="Sync authorized Tushare public-fund data into fund_ai.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog", help="sync fund_company and fund_basic")
    nav_parser = subparsers.add_parser("nav", help="sync fund_nav for one explicit NAV date")
    nav_parser.add_argument("--nav-date", required=True, type=date.fromisoformat, help="NAV date in YYYY-MM-DD")
    arguments = parser.parse_args(argv)

    service = TushareFundSyncService()
    try:
        outcome = (
            service.sync_catalog() if arguments.command == "catalog" else service.sync_nav_daily(arguments.nav_date)
        )
        print(json.dumps(outcome.to_payload(), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"tushare_fund_sync failed: {type(error).__name__}: {str(error)[:300]}", file=sys.stderr)
        return 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
