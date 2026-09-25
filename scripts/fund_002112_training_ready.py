"""只读复核 002112 一日训练包；补齐动作从同步中心进入，不在此绕过任务队列。"""

import argparse
import json

from app.services.fund_exposure_runtime import execution_lock
from app.services.fund_training_package import training_inputs, verify


def main():
    parser = argparse.ArgumentParser(description="复核 002112 一日离线训练资料，不执行训练")
    parser.add_argument("command", choices=("check", "input-check"))
    parser.add_argument(
        "--variant", choices=("NAV7", "NAV7_HOLDINGS", "NAV7_HOLDINGS_MARKET"), default="NAV7_HOLDINGS_MARKET"
    )
    args = parser.parse_args()
    with execution_lock() as locked:
        if not locked:
            raise ValueError("EXPOSURE_BUSY")
        if args.command == "check":
            result = verify(pointer="ready.json", publish=False)
        else:
            data = training_inputs(args.variant)
            result = {
                "fund_code": data["fund_code"],
                "horizon": data["horizon"],
                "variant": data["variant"],
                "features": len(data["features"]),
                "train_rows": len(data["X_train"]),
                "development_rows": len(data["X_development"]),
                "dataset_hash": data["dataset_hash"],
                "training_runs": 0,
            }
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
