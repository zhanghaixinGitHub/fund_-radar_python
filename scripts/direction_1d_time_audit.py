"""只读复核冻结运行的全部输入公告日期；不改原包、不拟合、不生成真实预测。"""

import argparse
from datetime import date, datetime, time
from pathlib import Path

from app.services import direction_1d_training as training
from app.services.direction_1d_protocol import ZONE, canonical, digest, input_days


def audit(root: Path):
    training.verify(root)
    history = training.read(root / "history.json")
    rows = training.read(root / "dataset.json")
    models = training.read(root / "models.json")
    predictions = training.read(root / "predictions.json")
    lookup = {f["fund_code"]: {r["date"]: r for r in f["rows"]} for f in history["funds"]}
    available = {}
    for r in rows:
        if r["kind"] != "HISTORICAL_RECONSTRUCTION":
            raise ValueError("AUDIT_HISTORICAL_ONLY")
        days = [str(d) for d in input_days(date.fromisoformat(r["t"]))]
        latest = max(lookup[r["fund_code"]][d]["ann_date"] or d for d in days)
        available[r["fund_code"], r["t"]] = datetime.combine(date.fromisoformat(latest), time(8), ZONE)
    by_key = {(r["fund_code"], r["t"]): r for r in rows}
    reports = []
    for key, model in models.items():
        cutoff = datetime.fromisoformat(model["train_as_of"])
        fit = training.select_fit([r for r in rows if r["group"] == model["group_id"]], cutoff)
        if digest(fit) != model["fit_hash"]:
            raise ValueError("FIT_SOURCE_MISMATCH")
        bad_fit = [r for r in fit if available[r["fund_code"], r["t"]] > cutoff]
        exam = [p for p in predictions if p["job"] == key]
        valid = [
            p
            for p in exam
            if available[p["fund_code"], p["t"]] <= datetime.combine(date.fromisoformat(p["u"]), time(8), ZONE)
        ]
        reports.append(
            {
                "job": key,
                "fit_count": len(fit),
                "fit_unavailable_count": len(bad_fit),
                "original_exam_count": len(exam),
                "time_qualified_exam_count": len(valid),
                "excluded_exam_count": len(exam) - len(valid),
                "time_qualified_metrics": training.metrics(
                    [by_key[p["fund_code"], p["t"]] for p in valid], [p["score"] for p in valid], model["majority"]
                ),
            }
        )
    return {
        "kind": "HISTORICAL_RECONSTRUCTION_AUDIT",
        "fits": 0,
        "assumption": "ANN_DATE_AT_08_00_NOT_FIRST_RECEIPT_PROOF",
        "reports": reports,
        "fit_time_integrity_valid": all(r["fit_unavailable_count"] == 0 for r in reports),
        "input_matures_after_label_assumption_count": sum(
            available[r["fund_code"], r["t"]] > datetime.fromisoformat(r["mature_at"]) for r in rows
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir)
    training.write_new(args.out, result)
    print(
        canonical(
            {
                "fit_time_integrity_valid": result["fit_time_integrity_valid"],
                "excluded_exam_count": sum(r["excluded_exam_count"] for r in result["reports"]),
                "fits": 0,
            }
        )
    )
