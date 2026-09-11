"""市场广度两组实验：探测、冻结、采集、准备、预测、评分、封存、复核及复跑。"""

import argparse
import json
from uuid import UUID

from app.services import direction_breadth_data as data
from app.services import direction_breadth_runner as runner
from app.services.direction_training_artifacts import run_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("probe", "freeze", "acquire", "prepare", "predict", "score", "finalize", "verify", "replay")
    )
    parser.add_argument("--run", type=UUID)
    args = parser.parse_args()
    if args.action in ("probe", "freeze"):
        if args.run:
            parser.error("probe and freeze do not take --run")
        result = data.probe() if args.action == "probe" else runner.freeze()
    else:
        if args.run is None:
            parser.error("--run is required")
        result = getattr(runner, args.action)(run_folder(args.run))
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
