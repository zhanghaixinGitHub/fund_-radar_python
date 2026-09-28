"""复核原第三轮每折真正传入的样本权重，只描述输入，不拟合或选择新样本。"""

import math
from collections import Counter
from pathlib import Path

from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once

OUTPUT = ROOT / "input-storage-audit/20260926-v1"
COHORT = {"002112", "002170", "004237", "004605", "005187", "005312", "006038", "007509", "008960", "017493", "160323"}


def weighted_median(values, weights):
    """取累计权重首次达到一半的原观测值；不插值，不把重复日期视为独立样本。"""
    if not values or len(values) != len(weights):
        raise ValueError("EMPTY_OR_MISALIGNED_VALUES")
    if any(not math.isfinite(v) for v in values) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("INVALID_WEIGHTED_VALUES")
    pairs = sorted(zip(values, weights, strict=True))
    threshold = math.fsum(weights) / 2
    # 使用 fsum 避免普通累加的舍入误差把恰好一半推到下一观测值。
    for i, (value, _) in enumerate(pairs):
        if math.fsum(w for _, w in pairs[: i + 1]) >= threshold:
            return value
    return pairs[-1][0]


def describe(rows, weights):
    """股票仓位来自当时已公开报告；权重是学习权重，不是资产资金权重。"""
    total = math.fsum(weights)
    values = [r["x"][13] * 100 for r in rows]
    return {
        "rows": len(rows),
        "distinct_dates": len({r["target"] for r in rows}),
        "start": min(r["target"] for r in rows),
        "end": max(r["target"] for r in rows),
        "total_learning_weight": total,
        "stock_pct_min": min(values),
        "stock_pct_max": max(values),
        "stock_pct_weighted_median": weighted_median(values, weights),
        "stock_pct_unweighted_median": weighted_median(values, [1.0] * len(rows)),
        "stock_bins": {
            label: {
                "rows": sum(a <= r["x"][13] < b for r in rows),
                "learning_weight_share": math.fsum(w for r, w in zip(rows, weights, strict=True) if a <= r["x"][13] < b)
                / total,
            }
            for label, a, b in (("<50%", 0, 0.5), ("50%-<80%", 0.5, 0.8), (">=80%", 0.8, math.inf))
        },
    }


def audit_fold(fold, index):
    """按冻结 train_ids 原顺序重建，不导入旧训练模块；任一身份或权重变化即拒绝。"""
    ids = [tuple(k) for k in fold["train_ids"]]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPLICATE_TRAIN_ID")
    rows = [index[k] for k in ids]
    weights = fold["weights"]
    if digest(rows) != fold["train_hash"] or len(rows) != len(weights):
        raise ValueError("FROZEN_TRAIN_OR_WEIGHTS_CHANGED")
    if {r["fund_code"] for r in rows} != COHORT:
        raise ValueError("FIXED_COHORT_CHANGED")
    families = Counter(r["family"] for r in rows)
    if len(families) != 11 or len({(r["family"], r["target"]) for r in rows}) != len(rows):
        raise ValueError("FAMILY_IDENTITY_CHANGED")
    for r, w in zip(rows, weights, strict=True):
        if r["target"] >= fold["start"] or r["target"] > "2023-12-31":
            raise ValueError("TRAIN_DATE_NOT_PRIOR")
        if len(r["x"]) != 20 or not 0 <= r["x"][13] <= 1:
            raise ValueError("INVALID_REPORTED_STOCK_WEIGHT")
        expected = len(rows) / (11 * families[r["family"]])
        if not math.isfinite(w) or w <= 0 or not math.isclose(w, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("NOT_ORIGINAL_FAMILY_EQUAL_WEIGHT")
    if not math.isclose(math.fsum(weights), len(rows), rel_tol=0, abs_tol=1e-8):
        raise ValueError("TOTAL_WEIGHT_CHANGED")
    total = math.fsum(weights)
    result = {"overall": describe(rows, weights), "by_fund": {}, "by_year": {}}
    for field, group in (("fund_code", "by_fund"), ("year", "by_year")):
        keys = sorted({r["target"][:4] if field == "year" else r[field] for r in rows})
        for key in keys:
            pairs = [
                (r, w)
                for r, w in zip(rows, weights, strict=True)
                if (r["target"][:4] if field == "year" else r[field]) == key
            ]
            subset, ws = zip(*pairs, strict=True)
            item = describe(subset, ws)
            item["learning_weight_share"] = math.fsum(ws) / total
            item["families"] = sorted({r["family"] for r in subset})
            result[group][key] = item
    exam = [index[("002112", d)] for d in fold["expected_dates"]]
    result["exam_inputs"] = describe(exam, [1.0] * len(exam))
    result["weights_sha256"] = digest(weights)
    result["training_sha256"] = digest(rows)
    result["actual_fit_status"] = "NOT_FITTED_PLANNED_ONLY" if fold["name"] == "FULL" else "HISTORICAL_FIT_COMPLETE"
    return result


def run():
    protocol = read_json(OUTPUT / "protocol.json")
    protection = read_json(OUTPUT / "protection-before.json")

    def load(name):
        spec = protocol["sources"][name]
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError("SOURCE_CHANGED")
        return read_json(spec["path"])

    data, folds = load("third_inputs"), load("third_folds")
    index = {(r["fund_code"], r["target"]): r for part in ("train", "development") for r in data[part]}
    if len(index) != 4419 + 230:
        raise ValueError("FROZEN_ROW_COUNT_CHANGED")
    results = {f["name"]: audit_fold(f, index) for f in folds}
    # 数字模型的清单可核查实际训练身份；不读取或恢复模型二进制。
    model_dir = Path(protocol["sources"]["third_inputs"]["path"]).parent / "models"
    verified = []
    for name, result in results.items():
        if name == "FULL":
            if list(model_dir.glob("FULL-*")):
                raise ValueError("UNEXPECTED_FULL_MODEL")
            continue
        for model in ("N7", "L20", "T20"):
            for stage in ("main", "replay"):
                path = model_dir / f"{name}-{model}-{stage}.manifest.json"
                if file_hash(path) != protection["artifacts"][str(path)]:
                    raise ValueError("MODEL_MANIFEST_CHANGED")
                manifest = read_json(path)
                if (
                    manifest["training_hash"] != result["training_sha256"]
                    or manifest["weights_hash"] != result["weights_sha256"]
                ):
                    raise ValueError("ACTUAL_TRAIN_IDENTITY_MISMATCH")
                verified.append(str(path))
    output = {
        "protocol_sha256": file_hash(OUTPUT / "protocol.json"),
        "folds": results,
        "actual_fit_manifests_verified": verified,
        "new_model_fits": 0,
        "interpretation": "INPUT_DISTRIBUTION_ONLY_NOT_CAUSE_OR_PREDICTIVE_GAIN",
    }
    save_once(OUTPUT / "input-audit.json", output)
    return output
