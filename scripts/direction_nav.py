"""净值可用性与缺失敏感性 CLI；只接受固定协议、固定已封存来源和受控 UUID。"""

import argparse
import json
from uuid import UUID

from app.services import direction_nav_runner as runner
from app.services.direction_training_artifacts import run_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "prepare", "predict", "score", "verify", "replay"))
    parser.add_argument("--run", type=UUID)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.run:
            parser.error("freeze creates a new run")
        result = runner.freeze()
    else:
        if args.run is None:
            parser.error("--run is required")
        result = getattr(runner, args.action)(run_folder(args.run))
    print(json.dumps(result, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
