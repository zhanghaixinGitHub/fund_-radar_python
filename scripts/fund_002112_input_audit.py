"""执行原折输入审计；不会训练或调用数据库。"""

import json

from app.services.fund_002112_input_audit import run

if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                "folds": {k: v["overall"] for k, v in result["folds"].items()},
                "verified_fit_manifests": len(result["actual_fit_manifests_verified"]),
                "new_model_fits": 0,
            },
            ensure_ascii=False,
        )
    )
