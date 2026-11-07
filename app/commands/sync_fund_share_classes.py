"""动态维护全目录份额关系；预览与发布共用来源核验，不操作持仓或净值。"""

import argparse
import json
from pathlib import Path

from app.services.fund_share_classes import synchronize


def main():
    parser = argparse.ArgumentParser(description="核对全部市场基金的其他份额")
    parser.add_argument("--fund-code", action="append")
    parser.add_argument("--force", action="store_true", help="重新检查已在七天内核对的基金")
    parser.add_argument("--dry-run", action="store_true", help="只采集核对，不更新页面关系")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = synchronize(args.fund_code, due_only=not args.force, publish=not args.dry_run,
                         progress=lambda i, total, code, _: print(f"{i}/{total} {code}", flush=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
