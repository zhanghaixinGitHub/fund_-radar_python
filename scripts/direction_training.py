"""本机自训练专项命令；每阶段显式执行，不自动发布模型。"""

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("inventory", "coverage", "freeze", "prepare", "predict", "score", "verify", "replay")
    )
    parser.add_argument("--inventory", type=UUID)
    parser.add_argument("--run", type=UUID)
    args = parser.parse_args()
    from app.services.direction_training_inventory import inventory

    folder = None
    try:
        if args.command == "inventory":
            if args.inventory or args.run:
                parser.error("inventory generates a new UUID")
            folder, report = inventory()
            output = {"status": "INVENTORIED", "folder": str(folder), "cash_windows": list(report["cash_windows"])}
        elif args.command == "coverage":
            if not args.inventory or args.run:
                parser.error("coverage requires --inventory")
            from app.services.direction_training_artifacts import run_folder
            from app.services.direction_training_dataset import export_coverage

            folder, report = export_coverage(run_folder(args.inventory, prefix="direction-inventory"))
            output = {
                "status": "COVERED",
                "folder": str(folder),
                "windows": {k: v["status"] for k, v in report["windows"].items()},
            }
        else:
            if not args.run or args.inventory:
                parser.error("this stage requires --run only")
            from app.services.direction_training_artifacts import run_folder
            from app.services.direction_training_protocol import freeze
            from app.services.direction_training_runner import predict, prepare, replay, score, verify

            folder = run_folder(args.run)
            stage = {
                "freeze": freeze,
                "prepare": prepare,
                "predict": predict,
                "score": score,
                "verify": verify,
                "replay": replay,
            }
            report = stage[args.command](folder)
            output = {"status": report["status"], "folder": str(folder), "manifest_hash": report["manifest_hash"]}
            if report.get("replay_folder"):
                output["replay_folder"] = report["replay_folder"]
    except Exception as error:
        # 不将底层连接异常中的地址/凭据写到终端；完整类型用于定位。
        failure = {"status": "FAILED", "stage": args.command, "error_type": type(error).__name__}
        if folder and folder.is_dir() and not (folder / "complete.json").exists():
            from app.services.direction_training_artifacts import write_json

            write_json(folder / f"failed-{uuid4()}.json", failure)
        print(json.dumps(failure))
        return 1
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
