"""经理目录CLI：默认仅核对本地范围，显式选择发布或授权来源读取。"""

import argparse
import json
from pathlib import Path
from uuid import UUID

from app.services.fund_manager_directory_sync import synchronize


def main(argv=None):
    parser = argparse.ArgumentParser(description="维护基金经理公共资料，默认dry-run不联网、不写业务库")
    parser.add_argument("--manager-id", action="append", type=UUID, default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--publish-local", action="store_true", help="只从已保存基金任职资料发布本基金锚点")
    mode.add_argument("--fetch", action="store_true", help="读取既有授权来源并发布，复用来源锁和预算")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = synchronize(args.manager_id, dry_run=not (args.fetch or args.publish_local), fetch=args.fetch)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps({key: value for key, value in result.items() if key not in {"targets", "items"}}, ensure_ascii=False)
    )
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
