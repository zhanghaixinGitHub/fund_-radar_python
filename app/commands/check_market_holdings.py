"""python -m app.commands.check_market_holdings：独立检查所有市场基金，仅有新报告才下载。"""

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.fund_exposure_common import now
from app.services.market_holdings_maintenance import MarketHoldingsMaintenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="显式执行本轮；默认按披露期每天、平时每周去重")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    try:
        result = MarketHoldingsMaintenance().check(force=args.force)
    except Exception as exc:
        # 数据库等前置条件失败也覆盖本轮回执，避免调度器把旧成功文件当作本次成功。
        get_logger(__name__).exception("check_market_holdings.main >>> 独立持仓检查未完成")
        result = {"status": "FAILED", "finished_at": now().isoformat(), "failure": type(exc).__name__}
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
