"""检查本人当前持仓中已审阅但暂不判断的原区间；只生成变化待办。"""

import json

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now
from app.services.prediction_review_watch import watch_report
from app.services.prediction_trial_v2 import PROJECT, current_holdings, storage_check


def main():
    at = now()
    storage_check(at.isoformat())
    owner = json.loads((PROJECT / ".local-runs/automation-owner/account.json").read_text(encoding="utf-8"))[
        "owner_user_id"
    ]
    inventory = current_holdings()
    if inventory["owner_hash"] != digest(owner):
        raise ValueError("WATCH_CURRENT_OWNER_CHANGED")
    result = watch_report(owner, inventory["scope"], at=at)
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
