"""单独补核本人当前模拟持仓的一日历史答案，不重新采集、评分或生成预测。"""

import argparse
import json
from pathlib import Path

from app.services.fund_daily_runtime import PROJECT, LocalDailyBackend
from app.services.fund_daily_services import ensure_services
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--all-archived", action="store_true", help="一次性补核本人全部原档案；每日任务仍仅当前模拟持仓"
    )
    args = parser.parse_args()
    directory = args.run_dir or PROJECT / ".local-runs" / ("daily-prediction-" + now().strftime("%Y%m%d"))
    ensure_services(directory)
    backend = LocalDailyBackend(directory)
    try:
        backend.context()
        result = backend.one_day_outcomes(all_archived=args.all_archived)
        state_path = directory / "run.json"
        if state_path.exists() and not args.all_archived:
            state = read(state_path)
            if state["owner_user_id"] != result["owner_user_id"]:
                raise ValueError("DAILY_OWNER_BINDING_CHANGED")
            # 只追加本次一日结果回执，原新预测窗口及整轮未完成状态不改。
            state["one_day_outcomes"] = result
            versioned_save(state_path, state)
    finally:
        backend.close()
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
                "output": str(
                    directory / ("one-day-archive-outcomes.json" if args.all_archived else "one-day-outcomes.json")
                ),
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if result["status"] == "SUCCEEDED" else 2)


if __name__ == "__main__":
    main()
