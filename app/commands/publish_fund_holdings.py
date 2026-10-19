"""python -m app.commands.publish_fund_holdings --file 清单.json [--dry-run]：离线补齐披露持仓。"""

import argparse
import json
from pathlib import Path

from app.services.fund_holdings_publication import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True, help="逐基金绑定原件、哈希和来源的本地清单")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = publish(json.loads(args.file.read_text(encoding="utf-8")), dry_run=args.dry_run)
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
