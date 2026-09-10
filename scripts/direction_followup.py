"""按实施文档执行T05—T09；所有写入限定本地研究包。"""

import argparse
import json
from uuid import UUID, uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "prepare", "predict", "score", "finish", "verify", "replay"))
    parser.add_argument("--run", type=UUID)
    args = parser.parse_args()
    from app.services.direction_followup_runner import finish, freeze, predict, prepare, replay, score, verify
    from app.services.direction_training_artifacts import run_folder, write_json

    folder = None
    try:
        if args.stage == "freeze":
            if args.run:
                parser.error("freeze creates a new run")
            folder, result = freeze()
        else:
            if not args.run:
                parser.error("stage requires --run")
            folder = run_folder(args.run)
            result = {
                "prepare": prepare,
                "predict": predict,
                "score": score,
                "finish": finish,
                "verify": verify,
                "replay": replay,
            }[args.stage](folder)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "folder": str(folder),
                    "hash": result["manifest_hash"],
                    "replay_folder": result.get("replay_folder"),
                }
            )
        )
        return 0
    except Exception as error:
        result = {"status": "FAILED", "stage": args.stage, "error_type": type(error).__name__}
        if folder and folder.is_dir() and not (folder / "study-complete.json").exists():
            write_json(folder / f"study-failed-{uuid4()}.json", result)
        print(json.dumps(result))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
