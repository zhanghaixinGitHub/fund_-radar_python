"""固定滚动更新实验入口：先冻结，再训练、评分、封存及唯一复跑。"""

import argparse
import json
from pathlib import Path
from uuid import UUID

from app.services import direction_rolling_runner as runner
from app.services.direction_training_artifacts import run_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "predict", "score", "finalize", "verify", "replay"))
    parser.add_argument("--run", type=UUID)
    parser.add_argument("--design", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.run or args.design is None:
            parser.error("freeze requires --design and no --run")
        result = runner.freeze(args.design)
    else:
        if args.run is None or args.design:
            parser.error("this action requires --run and no --design")
        result = getattr(runner, args.action)(run_folder(args.run))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
