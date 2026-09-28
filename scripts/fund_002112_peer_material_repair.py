"""运行冻结本地资料修复和覆盖检查，不提供网络或训练开关。"""

import json

from app.services.fund_002112_peer_material_repair import run

if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                **result,
                "funds": {
                    c: {**v, "new_complete_input_dates": len(v["new_complete_input_dates"])}
                    for c, v in result["funds"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
