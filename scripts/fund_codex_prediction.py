"""本机聊天预测入口：export 冻结、catalog 核对、import 校验保存；没有模型调用。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.fund_codex_analysis import catalog, checked_packet, freeze_packet, import_review
from app.services.fund_exposure_common import read, save


def save_once(path, value):
    """相同命令重试只复用一致回执；不同内容不能覆盖首次文件。"""
    if path.exists():
        if read(path) != value:
            raise ValueError("MANUAL_ARTIFACT_ALREADY_EXISTS")
        return
    save(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["export", "catalog", "import"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.operation == "export":
        result = freeze_packet(args.path)
    elif not args.review:
        parser.error("catalog/import 必须指定 --review；不接受缺少原始审阅的判断")
    elif args.operation == "catalog":
        facts, events, coverage = catalog(checked_packet(args.path)["input"], read(args.review))
        output = args.path.parent / "review-facts.json"
        save_once(output, {"facts": facts, "event_count": len(events), "coverage": coverage})
        result = {"facts_file": str(output.resolve()), "event_count": len(events), "limited_documents": len(coverage)}
    else:
        result = import_review(args.path, args.review)
        save_once(args.path.parent / "import-receipt.json", result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
