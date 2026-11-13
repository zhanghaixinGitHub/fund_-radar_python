"""第二版持仓分类试验：prepare冻结现有输入身份，evaluate导入实际审阅。"""

import argparse
import json
from pathlib import Path

from app.services.fund_exposure_common import now
from app.services.prediction_trial_v2 import evaluate_trial, prepare_trial, save_once, storage_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "evaluate"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--reviews-file", type=Path)
    args = parser.parse_args()
    at = now()
    guard = storage_check(at.replace(hour=9, minute=50, second=0, microsecond=0).isoformat())
    if args.operation == "prepare":
        if args.selection is None:
            parser.error("prepare必须提供实际持仓样本选择文件")
        result = prepare_trial(args.run_dir, args.selection)
    else:
        if args.reviews_file is None:
            parser.error("evaluate必须提供实际逐基金逐周期审阅文件")
        result = evaluate_trial(args.run_dir, args.reviews_file)
    save_once(args.run_dir / "storage-checks" / (at.strftime("%Y%m%d-%H%M%S-%f") + ".json"), guard)
    if args.operation == "prepare":
        print(json.dumps({"selected": result["selection"], "production_publication": False}, ensure_ascii=False))
    else:
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "fund_count",
                        "review_count",
                        "counts",
                        "dated_direction_count",
                        "conditional_outlook_count",
                        "conditional_direction_count",
                        "production_publication",
                    )
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
