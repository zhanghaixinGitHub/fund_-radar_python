"""补证结果的独立复核与接手清单；只读冻结值，不调用模型或网络。"""

import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once

OUTPUT = ROOT / "peer-gap-evidence/20260927-v1"
PREVIOUS = ROOT / "peer-admission/20260927-v1"


def verify():
    """重算新增答案及截止时点，核原 112 行、全 26 份来源映射和拟合账本。

    合格输入和合格训练包分开：本次不能取得历史版本时，按每个净值日期、
    每份原件列出尚缺的版本证据。公开站点查询失败不构成版本不存在的证明。
    """
    protocol = read_json(PREVIOUS / "protocol.json")
    for path, sha in protocol["files"].items():
        if file_hash(path) != sha:
            raise ValueError("ORIGINAL_SOURCE_CHANGED")
    snapshot = read_json(protocol["named"]["snapshot"])
    old_rows = read_json(PREVIOUS / "row-admission.json")
    old_inputs = read_json(PREVIOUS / "candidate-inputs-not-admitted.json")
    added = read_json(OUTPUT / "additional-inputs-not-admitted.json")
    labels = read_json(OUTPUT / "additional-labels-not-admitted.json")
    original = read_json(protocol["named"]["third_inputs"])
    folds = read_json(protocol["named"]["third_folds"])
    reports = read_json(OUTPUT / "report-supplements-v2.json")
    assert len(reports) == 18 and sum(r["newly_recovered_report"] for r in reports) == 2
    assert all(not r["table_differences"] for r in reports)
    current = {(r["fund_code"], r["nav_date"]): r for r in read_json(protocol["named"]["database"])["nav_rows"]}
    for r in reports:
        assert file_hash(r["parsed_file"]) == r["sha256"]
        parsed = read_json(r["parsed_file"])
        assert file_hash(parsed["raw"]["path"]) == r["raw_sha256"]
    nav = {c: {r["date"]: r for r in snapshot["funds"][c]["nav"]["rows"]} for c in ("017493", "160323")}
    days = sorted(snapshot["indices"]["000300.SH"]["rows"])
    added_index = {(r["fund_code"], r["target"]): r for r in added}
    old_index = {(r["fund_code"], r["target"]): r for r in old_inputs}
    assert len(added_index) == len(labels) == 80 and not set(added_index).intersection(old_index)
    assert len(old_index) == len(old_rows) == 750
    assert all(r["fund_code"] == "017493" and "2023-09-01" <= r["target"] <= "2023-12-29" for r in labels)
    extra_by_fold = Counter()
    for row in labels:
        code, u = row["fund_code"], row["target"]
        t = days[days.index(u) - 1]
        base, target = nav[code][t], nav[code][u]
        a, b = Decimal(base["nav"]), Decimal(target["nav"])
        actual = {1: "UP", 0: "FLAT", -1: "DOWN"}[(b > a) - (b < a)]
        assert row["actual_direction"] == actual and row["base"] == t
        assert (row["base_source_hash"], row["target_source_hash"]) == (base["source_hash"], target["source_hash"])
        maturity = (datetime.fromisoformat(max(u, target["ann_date"])) + timedelta(days=1)).strftime("%Y-%m-%d")
        assert row["mature_at"] == maturity + "T08:00:00+08:00"
        vector = added_index[code, u]
        assert len(vector["x"]) == 20 and digest(vector["x"]) == vector["input_vector_sha256"]
        assert not vector["training_eligible"] and not row["training_eligible"]
        report = next(r for r in reports if r["raw_sha256"] == vector["report_sha256"])
        assert report["source_publication_date"] < u
        for fold in folds:
            start = fold["start"]
            eligible = u < start and row["mature_at"] < start + "T00:00:00+08:00"
            if eligible:
                assert base["ann_date"] < start and target["ann_date"] < start
                extra_by_fold[fold["name"]] += 1
            if fold["name"] != "FULL":
                assert (row["fold_time"][fold["name"]] == "TIME_ELIGIBLE") == eligible
    assert dict(extra_by_fold) == {"2023Q4": 20, "FULL": 80}
    old_labels = {(r["fund_code"], r["target"]): r for r in old_rows}
    controls = [r for r in original["train"] if r["fund_code"] in nav]
    assert len(controls) == 112
    for row in controls:
        key = row["fund_code"], row["target"]
        assert row["x"] == old_index[key]["x"]
        assert row["actual_direction"] == old_labels[key]["actual_direction"]

    # 每个日期列出需要绑定的当前原始值及依赖目标日，避免只留下笼统“缺版本”。
    uses = defaultdict(list)
    for row in old_rows + added:
        for day in row["nav_dependency_dates"]:
            uses[row["fund_code"], day].append(row["target"])
    nav_gaps = []
    for (code, day), targets in sorted(uses.items()):
        raw, live = nav[code][day], current[code, day]
        assert (Decimal(raw["nav"]), raw["ann_date"], raw["source_hash"]) == (
            Decimal(live["unit_nav"]),
            live["ann_date"],
            live["content_hash"],
        )
        nav_gaps.append(
            {
                "fund_code": code,
                "nav_date": day,
                "unit_nav": raw["nav"],
                "ann_date_unchanged": raw["ann_date"],
                "frozen_source_hash": raw["source_hash"],
                "dependent_target_dates": sorted(set(targets)),
                "current_value_matches": True,
                "needed": "Dated provider version/revision record or independently archived historical NAV value",
                "historical_local_download_required": False,
                "status": "HISTORICAL_VERSION_UNRESOLVED",
            }
        )
    assert len(nav_gaps) == 979
    save_once(OUTPUT / "nav-version-gap-worklist.json", nav_gaps)

    by_key = {(r["fund_code"], r["report_end"], r["report_type"]): r for r in reports}
    report_gaps = []
    for old in read_json(PREVIOUS / "report-admission.json"):
        key = tuple(old[k] for k in ("fund_code", "report_end", "report_type"))
        supplement = by_key.get(key)
        report_gaps.append(
            {
                "fund_code": key[0],
                "report_end": key[1],
                "report_type": key[2],
                "original_raw_sha256_preserved": old["raw_sha256"],
                "original_priority": old["priority_new_row_dependency"],
                "primary_pdf_url": supplement["source_url"] if supplement else old["url"],
                "primary_pdf_sha256": supplement["raw_sha256"] if supplement else old["raw_sha256"],
                "primary_pdf_path": read_json(supplement["parsed_file"])["raw"]["path"]
                if supplement
                else old["raw_path"],
                "public_date_unchanged": old["conservative_publication_unchanged"],
                "primary_table_matches_previous": supplement["table_equal"]
                if supplement
                else old["table_replay_equal"],
                "historical_version_verified": False,
                "training_eligible": False,
                "needed": (
                    "Bind historical public version to this content; "
                    "current primary copy does not establish old revision history"
                ),
            }
        )
    for row in reports:
        if row["newly_recovered_report"]:
            report_gaps.append(
                {
                    **row,
                    "original_priority": False,
                    "new_report_dependency": True,
                    "needed": "Historical content version binding remains unresolved",
                }
            )
    assert len(report_gaps) == 26 and sum(r["original_priority"] for r in report_gaps) == 18
    for row in report_gaps[:24]:
        assert Path(row["primary_pdf_path"]).read_bytes().startswith(b"%PDF")
        assert file_hash(row["primary_pdf_path"]) == row["primary_pdf_sha256"]
        assert row["primary_table_matches_previous"]
    save_once(OUTPUT / "report-version-gap-worklist.json", report_gaps)

    # 延续原计划，只增加补证得到的候选身份，不改变考核、模型、特征和权重规则。
    prep = deepcopy(read_json(PREVIOUS / "independent-preparation.json"))
    for fold in folds:
        key, start = fold["name"], fold["start"]
        additions = [
            [r["fund_code"], r["target"]]
            for r in labels
            if r["target"] < start and r["mature_at"] < start + "T00:00:00+08:00"
        ]
        identity = prep["folds"][key]["candidate_peer_ids_pending_admission"] + additions
        assert len(identity) == len({tuple(r) for r in identity})
        prep["folds"][key]["candidate_peer_ids_pending_admission"] = sorted(identity)
    prep["supplement_sources"] = {
        str(OUTPUT / name): file_hash(OUTPUT / name)
        for name in (
            "additional-inputs-not-admitted.json",
            "additional-labels-not-admitted.json",
            "report-supplements-v2.json",
        )
    }
    prep["status"] = "REPORT_CONTENT_COMPLETED_HISTORICAL_VERSIONS_NOT_ADMITTED"
    assert prep["current_execute_budget"] == 0 and len(prep["proposed_slots_not_reserved"]) == 12
    save_once(OUTPUT / "independent-preparation.json", prep)
    stage_paths = {
        "round1": "training-runs/002112-b9fcc6eb85cce6e19466db3a",
        "round2": "optimization-runs/round-20260925-v1",
        "round3_original": "round3-runs/002112-r3-0ec4bfcbef4d93b3c49470f7",
        "round3_repaired": "round3-repair-runs/002112-r3r-46b69f78f27a82dc30acbe66",
    }
    counts = {k: len(list((ROOT / v / "attempts").glob("*.json"))) for k, v in stage_paths.items()}
    assert list(counts.values()) == [6, 28, 5, 13]
    assert not any(m.startswith("sklearn") or m.endswith("fund_002112_round3_model") for m in sys.modules)
    result = {
        "passed": True,
        "additional_numeric_labels_checked": 80,
        "new_label_classes": dict(Counter(r["actual_direction"] for r in labels)),
        "original_peer_controls_unchanged": 112,
        "prior_complete_rows_preserved": 750,
        "current_nav_dependencies_checked": 979,
        "new_primary_copies_compared_with_old": 16,
        "all_original_24_have_matching_primary_pdf": True,
        "total_reports": 26,
        "additional_time_eligible": {f["name"]: extra_by_fold[f["name"]] for f in folds},
        "original_exam_dates_parameters_classes_weights_unchanged": True,
        "old_fit_counts": counts,
        "actual_new_fits": 0,
        "cumulative_fits": 52,
        "qualification_still_blocked": True,
    }
    save_once(OUTPUT / "independent-audit.json", result)
    return result


if __name__ == "__main__":
    print(verify())
