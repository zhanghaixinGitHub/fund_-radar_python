"""独立准入命令：freeze 固定已有证据，review 重放；没有 train/execute 开关。"""

import argparse
from pathlib import Path

from app.services.fund_002112_peer_admission import OUTPUT, PREVIOUS, REPAIR, THIRD, report_inventory, run
from app.services.fund_002112_peer_fold_impact import FOLDS
from app.services.fund_002112_round3_data import COHORT
from app.services.fund_002112_zero_fit_review import ROOT, file_hash, read_json, save_once


def freeze():
    """绑定本轮代码和全部必要来源；已存在协议原样复用，变更资料应创建新版本。"""
    target = OUTPUT / "protocol.json"
    if target.exists():
        return read_json(target)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = read_json(PREVIOUS / "protocol.json")
    named = {k: v["path"] for k, v in previous["sources"].items()}
    for spec in previous["sources"].values():
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError("PREVIOUS_SOURCE_CHANGED")
    named.update(
        {
            "worklist": str(PREVIOUS / "report-admission-worklist.json"),
            "next_plan": str(PREVIOUS / "independent-next-plan.json"),
            "previous_decision": str(PREVIOUS / "decision.json"),
            "previous_validation": str(PREVIOUS / "validation.json"),
            "previous_protection": str(PREVIOUS / "protection-after.json"),
            "fold_impact": str(PREVIOUS / "fold-impact.json"),
            "repair_protocol": str(REPAIR / "protocol.json"),
            "third_protocol": str(THIRD / "protocol.json"),
            "third_sources": str(THIRD / "sources.json"),
            "database": str(OUTPUT / "database-source-review.json"),
            "model_code": str(ROOT.parents[1] / "app/services/fund_002112_round3_model.py"),
            "issuer_page": str(ROOT / "peer-material-gaps/20260927-v1/hx2022half-page-receipt.json"),
            "admission_code": str(ROOT.parents[1] / "app/services/fund_002112_peer_admission.py"),
            "admission_command": str(Path(__file__).resolve()),
            "admission_tests": str(ROOT.parents[1] / "tests/test_fund_002112_peer_admission.py"),
        }
    )
    for fold in FOLDS:
        for model in ("N7", "L20"):
            for mode in ("main", "replay"):
                key = f"{fold}-{model}-{mode}"
                named[key] = str(THIRD / "predictions" / (key + ".json"))
    files = {p: file_hash(p) for p in named.values()}
    worklist, snapshot = read_json(named["worklist"]), read_json(named["snapshot"])
    report_inventory(worklist, snapshot)
    for item in worklist["reports"]:
        if item["parsed_file"]:
            files[item["parsed_file"]] = file_hash(item["parsed_file"])
    page = read_json(named["issuer_page"])
    files[page["path"]] = page["sha256"]
    protocol = {
        "version": "PEER_REPORT_AND_ROW_ADMISSION_ZERO_FIT_V1",
        "current_fit_budget": 0,
        "proposed_fit_cap": 12,
        "cohort": list(COHORT),
        "folds": FOLDS,
        "named": named,
        "files": files,
        "read_scope": "Original <=2024 snapshot; new answer checks only peer 2021-2023; no sealed labels",
        "report_version_rule": (
            "Do not infer exact historical content version from receipt date, print date, "
            "URL date, or PDF metadata alone"
        ),
        "alternative_historical_evidence": (
            "Issuer/provider dated content version, correction history, "
            "or independently preserved historical public copy; "
            "no requirement for our own past downloads"
        ),
        "label_rule": (
            "Adjacent trading days; raw unit NAV; exact equality is FLAT; keep original announcement and maturity"
        ),
        "stop": (
            "Missing version evidence stops package admission. Budget zero stops all fitting, "
            "irrespective of material results."
        ),
    }
    save_once(target, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "review"))
    args = parser.parse_args()
    result = freeze() if args.command == "freeze" else run()
    if args.command == "review":
        print(result)
    print("独立资料准备完成；新增拟合 0，当前拟合预算 0。")


if __name__ == "__main__":
    main()
