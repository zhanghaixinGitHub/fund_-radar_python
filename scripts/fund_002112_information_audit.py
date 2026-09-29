"""独立核验新增标签、来源留存、旧行保留及折成员；不导入或拟合模型。"""

import json
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal

from app.services.fund_002112_information_admission import BASE, OUT, PREVIOUS, RUN, save_once
from app.services.fund_002112_zero_fit_review import digest, file_hash, read_json


def audit():
    """复用前阶段另一套窗口/十进制公式的验证摘要，再独立重算新增答案与成熟时间。"""
    protected = read_json(PREVIOUS / "all-protected-sources.json")
    # 前阶段保护清单可能带 files 包装；只取明确的文件哈希映射。
    sources = protected.get("files", protected)
    for path, sha in sources.items():
        if file_hash(path) != sha:
            raise ValueError("PROTECTED_SOURCE_CHANGED")
    previous = read_json(PREVIOUS / "independent-audit.json")
    if not previous["passed"] or previous["complete_20_inputs_and_time"] != 439:
        raise ValueError("PRIOR_INDEPENDENT_FORMULA_AUDIT_REQUIRED")
    added = read_json(OUT / "added-rows.json")
    inputs = read_json(OUT / "inputs.json")
    old = read_json(BASE / "inputs.json")
    index = {(r["fund_code"], r["target"]): r for r in inputs["train"]}
    feasible = {
        (r["fund_code"], r["target"]): r
        for r in read_json(PREVIOUS / "input-feasibility.json")
        if r["status"] == "INPUT_AND_TIME_ELIGIBLE"
    }
    db = read_json(OUT / "database-review.json")
    if db["status"] != "VERIFIED_CURRENT_READ_ONLY":
        raise ValueError("CURRENT_DATABASE_EVIDENCE_UNAVAILABLE")
    provider = db["source"]
    if (
        not provider["enabled"]
        or "fund_nav" not in provider["authorized_api_names"]
        or not provider["authorization_verified_at"]
    ):
        raise ValueError("NAV_SOURCE_USE_NOT_AUTHORIZED")
    stamp = datetime.fromisoformat(db["checked_at"])
    database = {(c, r["nav_date"]): r for c, rows in db["rows"].items() for r in rows}
    expiries = [
        datetime.fromisoformat(r["created_at"]) + timedelta(days=provider["retention_days"]) for r in database.values()
    ]
    if any(t <= stamp for t in expiries):
        raise ValueError("SOURCE_RETENTION_EXPIRED")
    if (
        len(index) != len(inputs["train"])
        or len(old["train"]) != 5137
        or any(index[r["fund_code"], r["target"]] != r for r in old["train"])
    ):
        raise ValueError("ORIGINAL_ROWS_CHANGED")
    if inputs["development"] != old["development"]:
        raise ValueError("DEVELOPMENT_INPUT_CHANGED")
    for row in added:
        reference = feasible[row["fund_code"], row["target"]]
        a, b = [database[row["fund_code"], row[k]] for k in ("base", "target")]
        delta = Decimal(b["unit_nav"]) - Decimal(a["unit_nav"])
        direction = {True: "UP", False: "DOWN"}[delta > 0] if delta else "FLAT"
        maturity = datetime.fromisoformat(max(row["target"], a["ann_date"], b["ann_date"])) + timedelta(days=1, hours=8)
        if direction != row["actual_direction"] or maturity.isoformat() + "+08:00" != row["mature_at"]:
            raise ValueError("INDEPENDENT_LABEL_OR_MATURITY_MISMATCH")
        if digest(row["x"]) != reference["input_vector_sha256"]:
            raise ValueError("INDEPENDENT_INPUT_CHANGED")
    folds = read_json(OUT / "folds.json")
    for f, original in zip(folds, read_json(BASE / "folds.json"), strict=True):
        original_ids = {tuple(k) for k in original["train_ids"]}
        expected = original_ids | {
            (r["fund_code"], r["target"]) for r in added if r["mature_at"] < f["start"] + "T00:00:00+08:00"
        }
        if {tuple(k) for k in f["train_ids"]} != expected or f["exam"] != original["exam"]:
            raise ValueError("FOLD_MEMBERSHIP_OR_EXAM_CHANGED")
        train = [index[tuple(k)] for k in f["train_ids"]]
        n = Counter(r["family"] for r in train)
        expected_weights = [len(train) / (11 * n[r["family"]]) for r in train]
        if expected_weights != f["weights"] or digest(train) != f["train_sha256"]:
            raise ValueError("FOLD_WEIGHTS_OR_ROWS_CHANGED")
    files = dict(sources)
    for name in ("protocol.json", "independent-audit.json", "input-feasibility.json"):
        files[str(PREVIOUS / name)] = file_hash(PREVIOUS / name)
    files[str(RUN / "protocol.json")] = file_hash(RUN / "protocol.json")
    files[str(BASE / "inputs.json")] = file_hash(BASE / "inputs.json")
    files[str(BASE / "folds.json")] = file_hash(BASE / "folds.json")
    save_once(OUT / "sources.json", {"files": files})
    result = {
        "passed": True,
        "new_labels_independently_checked": len(added),
        "old_rows_preserved": 5137,
        "total_rows": len(index),
        "protected_sources_unchanged": len(sources),
        "nav_dependencies": len(database),
        "retention_passed": True,
        "earliest_nav_expiry": min(expiries).isoformat(),
        "original_development_and_exams_unchanged": True,
        "new_fits": 0,
        "independence": "Separate decimal labels and membership; prior independent window/formula audit reused",
        "code_sha256": file_hash(__file__),
    }
    # 保留初稿审计，最终代码的复算另存一版，不能覆盖已有研究证据。
    save_once(OUT / "independent-audit-v2.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    audit()
