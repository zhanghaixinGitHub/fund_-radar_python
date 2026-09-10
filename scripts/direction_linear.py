"""固定协议的线性模型有限对照；不开放选时间、调阈值或访问保留年参数。"""

import argparse
import json
from uuid import UUID

from app.services import direction_linear_runner as runner
from app.services.direction_linear_protocol import ABLATION_VERSION, RECENCY_VERSION, VERSION
from app.services.direction_training_artifacts import run_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "prepare", "predict", "score", "finalize", "verify", "replay"))
    parser.add_argument("--run", type=UUID)
    parser.add_argument(
        "--study",
        choices=("refinement", "feature-ablation", "mature-recency"),
        help="Only for freeze; every later stage uses the sealed protocol.",
    )
    args = parser.parse_args()
    if args.action == "freeze":
        if args.run:
            parser.error("freeze creates a new run")
        result = runner.freeze(
            {"feature-ablation": ABLATION_VERSION, "mature-recency": RECENCY_VERSION}.get(args.study, VERSION)
        )
    else:
        if args.study is not None:
            parser.error("--study is only allowed with freeze")
        if args.run is None:
            parser.error("--run is required")
        result = getattr(runner, args.action)(run_folder(args.run))
    print(json.dumps(result, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
