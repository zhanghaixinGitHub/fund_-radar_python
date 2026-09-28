"""独立复核准入结果：从冻结净值重算答案和两端时点，不重复原折权重统计。"""

from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal

from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once

OUTPUT = ROOT / "peer-admission/20260927-v1"


def verify():
    """核对逐条来源和身份，另查季末净值晚公告对标签两端的实际影响。"""
    protocol = read_json(OUTPUT / "protocol.json")
    for path, sha in protocol["files"].items():
        if file_hash(path) != sha:
            raise ValueError("FROZEN_SOURCE_CHANGED")
    named = protocol["named"]
    snapshot, old, oldfolds = [read_json(named[k]) for k in ("snapshot", "third_inputs", "third_folds")]
    rows, inputs, reports, prep = [
        read_json(OUTPUT / k)
        for k in (
            "row-admission.json",
            "candidate-inputs-not-admitted.json",
            "report-admission.json",
            "independent-preparation.json",
        )
    ]
    vectors = {(r["fund_code"], r["target"]): r for r in inputs}
    original = {(r["fund_code"], r["target"]): r for r in old["train"]}
    report_index = {r["raw_sha256"]: r for r in reports}
    nav = {c: {r["date"]: r for r in f["nav"]["rows"]} for c, f in snapshot["funds"].items()}
    days = sorted(snapshot["indices"]["000300.SH"]["rows"])
    assert len(rows) == len(vectors) == len({(r["fund_code"], r["target"]) for r in rows}) == 750
    current = {(r["fund_code"], r["nav_date"]): r for r in read_json(named["database"])["nav_rows"]}
    pair_time, unchanged = [], 0
    for row in rows:
        code, day = row["fund_code"], row["target"]
        assert code in {"017493", "160323"} and "2021-01-04" <= day <= "2023-12-29"
        pos = days.index(day)
        base_day = days[pos - 1]
        before, after = nav[code][base_day], nav[code][day]
        a, b = Decimal(before["nav"]), Decimal(after["nav"])
        label = {1: "UP", -1: "DOWN", 0: "FLAT"}[(b > a) - (b < a)]
        assert label == row["actual_direction"] and row["base"] == base_day
        assert row["base_source_hash"] == before["source_hash"] and row["target_source_hash"] == after["source_hash"]
        mature = (datetime.fromisoformat(max(day, after["ann_date"])) + timedelta(days=1)).strftime("%Y-%m-%d")
        assert row["mature_at"] == mature + "T08:00:00+08:00"
        window = row["nav_window"]
        end = days.index(window["S"])
        assert window["nav_dates"] == days[end - 60 : end + 1] and len(window["nav_dates"]) == 61
        assert window["nav_hashes"] == [nav[code][d]["source_hash"] for d in window["nav_dates"]]
        assert all(nav[code][d]["ann_date"] < day for d in window["nav_dates"])
        assert row["quote_dates"] == days[pos - 21 : pos]
        for d in row["nav_dependency_dates"]:
            frozen, live = nav[code][d], current[code, d]
            assert (Decimal(frozen["nav"]), frozen["ann_date"], frozen["source_hash"]) == (
                Decimal(live["unit_nav"]),
                live["ann_date"],
                live["content_hash"],
            )
        assert digest(vectors[code, day]["x"]) == row["input_vector_sha256"]
        if (code, day) in original:
            unchanged += 1
            assert original[code, day]["x"] == vectors[code, day]["x"]
            assert original[code, day]["actual_direction"] == label
        assert row["report_sha256"] in report_index
        assert report_index[row["report_sha256"]]["conservative_publication_unchanged"] < day
        assert not row["training_eligible"] and not vectors[code, day]["training_eligible"]
        if before["ann_date"] > after["ann_date"]:
            # 原 mature_at 保留不变；额外核查“答案两端均已知”时点，不提前季末日期。
            both_day = (
                datetime.fromisoformat(max(day, before["ann_date"], after["ann_date"])) + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            eligibility = {
                f: both_day + "T08:00:00+08:00" < start + "T00:00:00+08:00"
                for f, start in protocol["folds"].items()
                if row["fold_time"][f] == "TIME_ELIGIBLE"
            }
            assert all(eligibility.values())
            pair_time.append(
                {
                    "fund_code": code,
                    "base": base_day,
                    "target": day,
                    "base_publication": before["ann_date"],
                    "target_publication": after["ann_date"],
                    "original_mature_at_preserved": row["mature_at"],
                    "both_endpoints_mature_at": both_day + "T08:00:00+08:00",
                    "original_time_eligible_folds_still_valid": eligibility,
                }
            )
    assert unchanged == 112 and len(pair_time) == 9
    # 只对依赖身份做核对，不重算、比较或输出家族权重。
    worklist = read_json(named["worklist"])
    for item in worklist["reports"]:
        for fold in protocol["folds"]:
            ids = {tuple(k) for k in next(f for f in oldfolds if f["name"] == fold)["train_ids"]}
            additional = [
                r
                for r in rows
                if r["report_sha256"] == item["raw_sha256"]
                and r["fold_time"][fold] == "TIME_ELIGIBLE"
                and (r["fund_code"], r["target"]) not in ids
            ]
            assert len(additional) == item["additional_dates_vs_original_per_fold"][fold]
    for f in oldfolds:
        expected_nine = [k for k in f["train_ids"] if k[0] not in {"017493", "160323"}]
        assert prep["folds"][f["name"]]["other_nine_train_ids"] == expected_nine
        assert prep["folds"][f["name"]]["exam_dates"] == f["expected_dates"]
    assert prep["current_execute_budget"] == 0 and len(prep["proposed_slots_not_reserved"]) == 12
    ledger_paths = {
        "round1": ROOT / "training-runs/002112-b9fcc6eb85cce6e19466db3a/attempts",
        "round2": ROOT / "optimization-runs/round-20260925-v1/attempts",
        "round3_original": ROOT / "round3-runs/002112-r3-0ec4bfcbef4d93b3c49470f7/attempts",
        "round3_new": ROOT / "round3-repair-runs/002112-r3r-46b69f78f27a82dc30acbe66/attempts",
    }
    counts = {k: len(list(v.glob("*.json"))) for k, v in ledger_paths.items()}
    assert counts == {"round1": 6, "round2": 28, "round3_original": 5, "round3_new": 13}
    result = {
        "passed": True,
        "independent_numeric_label_checks": 750,
        "original_peer_rows_unchanged": 112,
        "two_endpoint_publication_reviews": pair_time,
        "extra_fold_exclusions_from_two_endpoint_check": 0,
        "per_report_new_row_dependencies_match": True,
        "other_nine_funds_and_exam_dates_unchanged": True,
        "class_counts": dict(Counter(r["actual_direction"] for r in rows)),
        "real_fit_counts": counts,
        "new_fits": 0,
        "cumulative_fits": sum(counts.values()),
        "interpretation": "Both endpoints pass the original fold cutoffs; this does not establish historical versions.",
    }
    save_once(OUTPUT / "independent-audit.json", result)
    return result


if __name__ == "__main__":
    result = verify()
    print("独立核验通过：750 条答案、112 条旧记录、24 份依赖、原考试日；新增拟合 0。")
