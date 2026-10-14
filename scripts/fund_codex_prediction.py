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
    parser.add_argument("--fund", default="002112", help="六位基金代码；默认兼容原002112入口")
    parser.add_argument("--materials", type=Path, help="该基金已整理且保留原文的公共综合资料包")
    parser.add_argument("--horizon", default="T1", choices=["T1", "T5_V1", "T20_V1", "M6_V1"])
    args = parser.parse_args()
    if args.horizon != "T1":
        from app.services import comprehensive_multi_prediction as multi

        if args.operation == "export":
            result = multi.freeze_packet(args.path, args.fund, args.horizon, args.materials)
        elif args.operation == "import" and args.review:
            result = multi.import_review(args.path, args.review)
            save_once(args.path.parent / "import-receipt.json", result)
        else:
            parser.error("多周期使用 export/import；import 必须指定 --review，并由审阅明确 horizon_id")
        print(json.dumps(result, ensure_ascii=True))
        return
    if args.operation == "export":
        result = freeze_packet(args.path, args.fund, args.materials)
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
