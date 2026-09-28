"""002112 只读旧输入/模型，输出新的不可覆盖机制复盘；没有训练选项。"""

import json

from app.services.fund_002112_mechanism_review import run

if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "actual_new_fits",
                    "cumulative_actual_fits",
                    "direction_changes",
                    "outcomes",
                    "gates",
                    "peers",
                )
            },
            ensure_ascii=False,
        )
    )
