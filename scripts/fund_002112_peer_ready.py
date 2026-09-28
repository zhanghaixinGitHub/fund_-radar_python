"""准备或核验独立资料包；没有 train 选项，零预算不能启动任何模型。"""

import argparse

from app.services.fund_002112_peer_ready import PACKAGE, prepare, save_prepared
from app.services.fund_002112_zero_fit_review import file_hash, read_json


def preflight():
    """检查文件冻结和原门槛；只报告资料准备完成，拟合仍等待独立预算授权。"""
    ready = read_json(PACKAGE / "ready.json")
    for path, sha in ready["artifact_hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("PREPARED_ARTIFACT_CHANGED")
    sources = read_json(PACKAGE / "sources.json")
    for path, sha in sources["files"].items():
        if file_hash(path) != sha:
            raise ValueError("PREPARED_SOURCE_CHANGED")
    decision = read_json(PACKAGE / "decision.json")
    if not decision["material_ready"] or not all(f["gate"]["passed"] for f in decision["folds"].values()):
        raise ValueError("MATERIAL_GATE_FAILED")
    # 主程序的通过还不是最终交付；独立核验及代码/方案冻结缺一不可。
    frozen_path = PACKAGE / "execution-protocol.json"
    audit_path = PACKAGE / "independent-audit.json"
    finalized = frozen_path.exists() and audit_path.exists()
    if finalized:
        frozen = read_json(frozen_path)
        for path, sha in frozen["files"].items():
            if file_hash(path) != sha:
                raise ValueError("EXECUTION_PROTOCOL_SOURCE_CHANGED")
        if not read_json(audit_path)["passed"] or frozen["current_execute_budget"] != 0:
            raise ValueError("INDEPENDENT_AUDIT_OR_BUDGET_FAILED")
    return {
        "material_ready": finalized,
        "material_checks_passed": True,
        "fit_execution_allowed": False,
        "current_fit_budget": 0,
        "actual_new_fits": 0,
        "train_rows": decision["train_pool_rows"],
        "status": decision["status"] if finalized else "PENDING_INDEPENDENT_AUDIT_AND_FREEZE",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight"))
    args = parser.parse_args()
    if args.command == "prepare":
        outputs = prepare()
        save_prepared(outputs)
    print(preflight())


if __name__ == "__main__":
    main()
