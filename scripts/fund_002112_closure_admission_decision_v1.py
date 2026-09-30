"""按原完整资料清单形成不准入决定；引用已有证据，不修改采集/拟合预算或停止规则。"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.services.fund_information_history_v1 import ROOT, read, save, sha


def decide():
    root = ROOT / "closure/20260929-v1"
    if (root / ".company-bodies.lock").exists():
        raise ValueError("COLLECTION_STILL_RUNNING")
    previous = sorted((root / "admission-audit-v3").glob("*.json"))[-1]
    audit = read(previous)
    plan_path = root / "company-bodies/plan.json"
    plan = read(plan_path)
    for path, expected in {**plan["source_hashes"], **plan["code_hashes"]}.items():
        if sha(path) != expected:
            raise ValueError("FROZEN_DEPENDENCY_CHANGED")
    # 只读取已消耗账本的数量和摘要，失败/重跑同样计数，绝不另领一份额度。
    ledger_root = ROOT / "information-research/20260928-v1"
    ledger = sorted((ledger_root / "fit-ledger").glob("*.json"))
    budget = read(ledger_root / "budget-authorized.json")
    consumed = budget["previous_actual_fits"] + len(ledger)
    if consumed != 70 or audit["cumulative_fits"] != consumed or len(ledger) != 12:
        raise ValueError("FIT_LEDGER_REQUIRES_RECONCILIATION")
    rows = audit["rows"]
    if {r["document_id"] for r in rows} != {i["row"]["announcementId"] for i in plan["items"]}:
        raise ValueError("ORIGINAL_SCOPE_CHANGED")
    for row in rows:
        if sha(row["source_record"]) != row["source_record_sha256"]:
            raise ValueError("SOURCE_AUDIT_CHANGED")
    missing = [r for r in rows if not r["body_saved"] and not r["prior_original_available"]]
    conflict = [r for r in rows if r["revision_issues_preserved"]]
    if not missing and not conflict:
        raise ValueError("REASSESS_ADMISSION_INSTEAD_OF_REUSING_OLD_STOP")
    result = {
        "at": datetime.now().astimezone().isoformat(),
        "decision": "NOT_ADMITTED_UNDER_CURRENT_FROZEN_SCOPE",
        "scope": plan["scope"],
        "source_audit": {"path": str(previous), "sha256": sha(previous)},
        "plan_sha256": sha(plan_path),
        "audit_code_sha256": sha(Path(__file__)),
        "planned_documents": len(rows),
        "collection_stops_with_old_originals": sum(
            not r["original_collection_body_saved"] and r["prior_original_available"] for r in rows
        ),
        "remaining_missing_originals": missing,
        "missing_by_reason": dict(Counter(r["collection_stop_reason_preserved"] for r in missing)),
        "version_conflicts": conflict,
        "requests_consumed": len(list((root / "company-bodies/requests").glob("*.json"))),
        "request_limit": plan["maximum_new_requests"],
        "cumulative_fits": consumed,
        "remaining_shared_fits": min(12, budget["authorized"] - len(ledger), 82 - consumed),
        "fit_ledger_sha256": {str(p): sha(p) for p in ledger},
        "new_fits": 0,
        "new_requests": 0,
        "fit_execution_allowed": False,
        "candidate_training_result": None,
        "adoption_allowed": False,
        "blockers": [
            "原完整清单中仍有业绩原件无法取得，不能删除相应样本、填零或重试原停止来源。",
            "部分原件生成/修改时间晚于目录披露时间，尚无可据以消除冲突的原始版本证据。",
            "指定字段和身份审核的旧通过继续有效，未扩张为完整语义和全部历史窗口覆盖。",
            "政策/行业资料仍只有部分正文及历史覆盖证据，不能以目录数量替代合格输入。",
        ],
        "not_executed": {
            "M03": "没有满足全输入条件的新假设，不冻结虚假的合格输入包。",
            "M04_M05": "前置资料未准入，没有开始拟合；不存在新模型成绩或采用通过决定。",
            "M06_M07": "没有通过比较的新模型，不登记替换、不把原模型读回当作新模型已使用。",
            "M08": "新模型尚无真实前向预测，不能计算或补写未来效果。",
        },
        "reopen_condition": "需要可核验的独立原始版本/缺失事实及完整时间证据；不能重复请求、复用预算或放宽原门槛。",
    }
    target = root / "admission-decision-v1" / (datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    save(target, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "missing_originals": len(missing),
                "version_conflicts": len(conflict),
                "cumulative_fits": consumed,
                "remaining_shared_fits": result["remaining_shared_fits"],
                "path": str(target),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    decide()
