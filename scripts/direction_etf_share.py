"""ETF份额单指标研究入口；采集好的本地来源通过覆盖检查后才冻结训练。"""

import argparse
import json
from pathlib import Path
from uuid import UUID

from app.services import direction_etf_share_runner as runner
from app.services.direction_training_artifacts import run_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "predict", "score", "finalize", "verify", "replay"))
    parser.add_argument("--run", type=UUID)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.run or args.design is None or args.data is None:
            parser.error("freeze requires --design and --data, without --run")
        result = runner.freeze(args.design, args.data)
    else:
        if args.run is None or args.design or args.data:
            parser.error("this action requires --run only")
        result = getattr(runner, args.action)(run_folder(args.run))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
