"""两只参考基金补资料后的原折影响审计；仅统计日期和权重，不生成标签或训练包。

完整输入来自上阶段不可变证据；成熟时点沿用原规则。这里的假设样本量只用于
回答权重如何分摊，不能绕过报告历史版本、标签核查及新实验准入。
"""

import math
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from app.services.fund_002112_peer_coverage_audit import calendar_days
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once

OUTPUT = ROOT / "peer-fold-impact/20260927-v1"
PEERS = ("017493", "160323")
COHORT = ("002112", "002170", "004237", "004605", "005187", "005312", "006038", "007509", "008960", *PEERS)
FOLDS = {"2023Q2": "2023-04-01", "2023Q3": "2023-07-01", "2023Q4": "2023-10-01"}
STAGES = (
    "strict_inputs_existing_reports",
    "after_six_local_repairs",
    "after_five_public_reports",
    "after_nine_public_reports",
)
ZONE = timezone(timedelta(hours=8))


def maturity(target: str, publication: str | None) -> str | None:
    """按原净值公告日计算下一自然日 08:00；缺日期保留未知，不猜首次公开日。

    只读取时间元数据，不比较净值，也不生成 UP/FLAT/DOWN 标签。已晚公布的
    季末净值仍按较晚公告日处理，不能因为当前查到数值就把日期提前。
    """
    if not publication:
        return None
    dates = [date.fromisoformat(d) for d in (target, publication)]
    if any(d.isoformat() != raw for d, raw in zip(dates, (target, publication), strict=True)):
        raise ValueError("NON_CANONICAL_DATE")
    if any(d > date(2024, 12, 31) for d in dates):
        raise ValueError("DATE_OUTSIDE_FROZEN_SCOPE")
    return datetime.combine(max(dates) + timedelta(days=1), time(8), ZONE).isoformat()


def time_status(target: str, publication: str | None, start: str) -> tuple[str, str | None]:
    """互斥地标记日期范围、未知公开时间、未公开、未成熟；边界必须严格早于截止点。"""
    if start not in FOLDS.values():
        raise ValueError("FOLD_CUTOFF_CHANGED")
    parsed = date.fromisoformat(target)
    if parsed.isoformat() != target or not "2021-01-04" <= target <= "2023-12-29":
        raise ValueError("PEER_TARGET_OUTSIDE_ORIGINAL_RANGE")
    if target >= start:
        return "TARGET_NOT_BEFORE_CUTOFF", None
    mature = maturity(target, publication)
    if mature is None:
        return "PUBLICATION_UNKNOWN", None
    if publication >= start:
        return "LABEL_NOT_PUBLIC_BEFORE_CUTOFF", mature
    if datetime.fromisoformat(mature) >= datetime.fromisoformat(start + "T00:00:00+08:00"):
        return "LABEL_NOT_MATURE_BEFORE_CUTOFF", mature
    return "TIME_ELIGIBLE", mature


def weight_table(counts: dict[str, int]) -> dict:
    """原等家族权重的算术推演；相对权重有意义，原始权重还随全池行数改变。

    返回总权重、单条归一化权重和相对 002112 单条的倍数。不把缺失家族或
    零样本默认为零权重，也不将该算术表传入任何模型。
    """
    if set(counts) != set(COHORT) or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("FIXED_COHORT_OR_POSITIVE_COUNT_REQUIRED")
    total = sum(counts.values())
    return {
        "rows": total,
        "by_fund": {
            code: {
                "rows": n,
                "family_weight_share": 1 / 11,
                "raw_per_row_weight": total / (11 * n),
                "normalized_per_row_weight": 1 / (11 * n),
                "per_row_relative_to_002112": counts["002112"] / n,
            }
            for code, n in sorted(counts.items())
        },
    }


def index_daily(rows: list[dict], days: list[str]) -> dict:
    """必须覆盖原两基金各 727 个交易日，重复行、漏日、换日期均直接拒绝。"""
    result = {(r["fund_code"], r["target"]): r for r in rows}
    expected = {(code, day) for code in PEERS for day in days}
    if len(rows) != len(result) or set(result) != expected:
        raise ValueError("DAILY_SCOPE_OR_DUPLICATE_CHANGED")
    return result


def load_source(protocol: dict, name: str):
    """来源必须符合事先摘要；不追随 ready 指针，不接受另一个文件的内容。"""
    spec = protocol["sources"][name]
    if file_hash(spec["path"]) != spec["sha256"]:
        raise ValueError("FROZEN_SOURCE_CHANGED:" + name)
    return read_json(spec["path"])


def run(output: Path = OUTPUT) -> dict:
    """复用四阶段逐日摘要与原三折，保存独立计数、逐日时间证据和权重表。"""
    protocol = read_json(output / "protocol.json")
    if (
        protocol["folds"] != FOLDS
        or protocol["new_fit_budget"] != 0
        or protocol["peer_codes"] != list(PEERS)
        or protocol["stages"] != ["original_used", *STAGES]
    ):
        raise ValueError("ZERO_FIT_PROTOCOL_CHANGED")
    # 包括原成熟公式源码在内全部验摘要，源码只读而不导入其数据库依赖。
    for name, spec in protocol["sources"].items():
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError("FROZEN_SOURCE_CHANGED:" + name)
    load = lambda name: load_source(protocol, name)  # noqa: E731
    for path, expected in load("current_sources").items():
        if file_hash(path) != expected:
            raise ValueError("RECOVERED_REPORT_SOURCE_CHANGED")
    days = calendar_days([load("calendar_older"), load("calendar_recent")])
    targets = [d for d in days if "2021-01-04" <= d <= "2023-12-29"]
    if len(targets) != protocol["days_per_peer"]:
        raise ValueError("ORIGINAL_CALENDAR_CHANGED")
    repair, public, current = [index_daily(load(n), targets) for n in ("repair_daily", "public_daily", "current_daily")]
    if digest(list(current.values())) != load("current_summary")["input_sha256"]:
        raise ValueError("LATEST_COVERAGE_DIGEST_CHANGED")
    snapshot = load("snapshot")
    maps = {}
    for code, fund in snapshot["funds"].items():
        mapping = {r["date"]: r for r in fund["nav"]["rows"]}
        if len(mapping) != len(fund["nav"]["rows"]) or max(mapping) > "2024-12-31":
            raise ValueError("NAV_DATE_OR_IDENTITY_CHANGED")
        maps[code] = mapping
    data, old_folds = load("third_inputs"), load("third_folds")
    if len(data["train"]) != 4419 or len(data["development"]) != 230 or set(maps) != set(COHORT):
        raise ValueError("ORIGINAL_PACKAGE_CHANGED")
    # 仅核对原训练记录。开发标签不用于本次统计，更不补造新增日期答案。
    index = {(r["fund_code"], r["target"]): r for r in data["train"]}
    if len(index) != 4419:
        raise ValueError("DUPLICATE_ORIGINAL_TRAIN_ID")
    original_peer_checks = 0
    for key, row in index.items():
        nav = maps[row["fund_code"]][row["target"]]
        if row["label_publication"] != nav["ann_date"] or row["mature_at"] != maturity(row["target"], nav["ann_date"]):
            raise ValueError("ORIGINAL_MATURITY_FORMULA_MISMATCH")
        if row["family"] != snapshot["funds"][row["fund_code"]]["family"]:
            raise ValueError("FUND_FAMILY_CHANGED")
        if row["fund_code"] in PEERS:
            after = current[key]["after"]
            if after["status"] != "INPUT_COMPLETE" or after["input_vector_sha256"] != digest(row["x"]):
                raise ValueError("ORIGINAL_PEER_INPUT_CHANGED")
            original_peer_checks += 1
    lineage, daily = {}, []
    for key in sorted(current):
        a, b, c = repair[key], public[key], current[key]
        if a["after"] != b["before"] or b["after"] != c["before"]:
            raise ValueError("COVERAGE_STAGE_LINEAGE_CHANGED")
        stages = dict(zip(STAGES, (a["before"], a["after"], b["after"], c["after"]), strict=True))
        for before, after in zip(list(stages.values())[:-1], list(stages.values())[1:], strict=True):
            if before["status"] == "INPUT_COMPLETE" and before != after:
                raise ValueError("PREVIOUS_COMPLETE_INPUT_CHANGED")
        code, target = key
        publication = maps[code].get(target, {}).get("ann_date")
        times = {}
        for fold, start in FOLDS.items():
            status, mature = time_status(target, publication, start)
            times[fold] = {"status": status, "mature_at": mature}
        lineage[key] = stages
        daily.append({"fund_code": code, "target": target, "publication": publication, "stages": stages, "time": times})
    daily_index = {(r["fund_code"], r["target"]): r for r in daily}
    results = {}
    old_audit = load("old_weight_audit")["folds"]
    for name, start in FOLDS.items():
        fold = next(f for f in old_folds if f["name"] == name)
        if fold["start"] != start or fold["exam_count"] != protocol["expected_exam_counts"][list(FOLDS).index(name)]:
            raise ValueError("ORIGINAL_FOLD_CHANGED")
        rows = [index[tuple(k)] for k in fold["train_ids"]]
        eligible = [
            r
            for r in data["train"]
            if r["target"] < start
            and r["label_publication"] < start
            and datetime.fromisoformat(r["mature_at"]) < datetime.fromisoformat(start + "T00:00:00+08:00")
        ]
        if rows != eligible or digest(rows) != fold["train_hash"]:
            raise ValueError("ORIGINAL_FOLD_MEMBERSHIP_CHANGED")
        if len({r["family"] for r in rows}) != 11 or len({(r["family"], r["target"]) for r in rows}) != len(rows):
            raise ValueError("ORIGINAL_FAMILY_DATE_CHANGED")
        counts = dict(Counter(r["fund_code"] for r in rows))
        actual = weight_table(counts)
        for row, weight in zip(rows, fold["weights"], strict=True):
            expected = actual["by_fund"][row["fund_code"]]["raw_per_row_weight"]
            if not math.isclose(weight, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError("ORIGINAL_WEIGHTS_CHANGED")
        if old_audit[name]["weights_sha256"] != digest(fold["weights"]):
            raise ValueError("PREVIOUS_WEIGHT_AUDIT_CHANGED")
        original_ids = {tuple(k) for k in fold["train_ids"]}
        result = {"start": start, "exam_count_unchanged": fold["exam_count"], "original_used": actual, "stages": {}}
        for stage in STAGES:
            replacement = dict(counts)
            peers = {}
            for code in PEERS:
                pool = [key for key in lineage if key[0] == code and key[1] < start]
                complete = [key for key in pool if lineage[key][stage]["status"] == "INPUT_COMPLETE"]
                available = [key for key in complete if daily_index[key]["time"][name]["status"] == "TIME_ELIGIBLE"]
                rejected = Counter(daily_index[key]["time"][name]["status"] for key in complete if key not in available)
                if not {key for key in original_ids if key[0] == code}.issubset(available):
                    raise ValueError("ORIGINAL_PEER_ROWS_LOST")
                replacement[code] = len(available)
                peers[code] = {
                    "targets_before_cutoff": len(pool),
                    "complete_before_time_check": len(complete),
                    "time_excluded": dict(rejected),
                    "input_complete_and_time_eligible": len(available),
                    "additional_vs_original_used": len(available) - counts[code],
                    "eligible_dates": sorted(key[1] for key in available),
                    "input_statuses_before_cutoff": dict(Counter(lineage[key][stage]["status"] for key in pool)),
                }
            result["stages"][stage] = {"peers": peers, "hypothetical_weights": weight_table(replacement)}
        results[name] = result
    result = {
        "protocol_sha256": file_hash(output / "protocol.json"),
        "new_fits": 0,
        "cumulative_fits": 52,
        "original_maturity_rows_verified": len(index),
        "original_peer_inputs_unchanged": original_peer_checks,
        "folds": results,
        "interpretation": "INPUT_AND_TIMING_ONLY_NOT_TRAINING_ADMISSION_OR_ACCURACY",
    }
    save_once(output / "fold-daily.json", daily)
    save_once(output / "fold-impact.json", result)
    return result


if __name__ == "__main__":
    summary = run()
    for fold, item in summary["folds"].items():
        final = item["stages"][STAGES[-1]]
        print(fold, {code: final["peers"][code]["input_complete_and_time_eligible"] for code in PEERS})
    print("新增真实拟合 0 次，累计仍为 52 次；未形成合格训练包。")
