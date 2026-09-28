"""002112 补资料实验后的零拟合机制复盘；固定问题，只解释已有模型和来源。

原始答案只用于事后诊断，不能生成选模规则或修改已有预测。新资料、多数类、
股票重叠均不自动构成预测增益。任何模型拟合调用在本入口中直接拒绝。
"""

from collections import Counter
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.services.fund_002112_model_state_v2 import restore_model, verify_all_predictions
from app.services.fund_002112_round3_data import FEATURES, public_day, select_report
from app.services.fund_002112_zero_fit_review import (
    digest,
    file_hash,
    quarter,
    read_json,
    save_once,
    validate_row,
)

ROOT = Path(__file__).resolve().parents[2] / ".local-runs/fund-exposure-002112"
OUTPUT = ROOT / "peer-mechanism-review/20260928-v1"
NEW = ROOT / "peer-recovered-experiment/20260928-v1"
OLD = ROOT / "round3-repair-runs/002112-r3r-46b69f78f27a82dc30acbe66"
SNAPSHOT = ROOT / "training-ready/sources/98fb1c51f2582687a09ac0c404d514c2ef74520e4cb63a9a9f89a300ef150611.json"
CLASSES = ("DOWN", "FLAT", "UP")


@contextmanager
def no_fits():
    """运行时封死三种估计器的 fit；所有诊断只用已保存对象/逐日预测。"""
    classes = (LogisticRegression, StandardScaler, HistGradientBoostingClassifier)
    original = {c: c.fit for c in classes}

    def forbidden(*args, **kwargs):
        raise RuntimeError("ZERO_FIT_REVIEW_FORBIDS_TRAINING")

    try:
        for c in classes:
            c.fit = forbidden
        yield
    finally:
        for c, function in original.items():
            c.fit = function


def stats(values):
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("NONFINITE_DIAGNOSTIC")
    return {"n": len(x), "min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}


def prior_equal_run(row, nav):
    """只读原窗口 S 及之前的公开净值，统计尾部相邻完全相等次数。

    不是把目标日答案塞进输入；S 可能落后基准日 T，日期/公开时间都逐项核验。
    上限受原 61 个净值窗口约束，不能把未知窗口外值补成相等。
    """
    dates = row["nav"]["nav_dates"]
    if not dates or dates != sorted(set(dates)) or dates[-1] != row["nav"]["S"] or dates[-1] >= row["target"]:
        raise ValueError("NAV_WINDOW_IDENTITY_CHANGED")
    values = []
    for day in dates:
        r = nav[day]
        if not r.get("ann_date") or r["ann_date"] >= row["target"]:
            raise ValueError("NAV_NOT_KNOWN_BEFORE_TARGET")
        values.append(Decimal(r["nav"]))
    count = 0
    for i in range(len(values) - 1, 0, -1):
        if values[i] != values[i - 1]:
            break
        count += 1
    return count


def flat_source(row, nav):
    """用冻结原来源核对持平的两端十进制值，不把小涨小跌改成持平。"""
    validate_row(row)
    before, after = nav[row["base"]], nav[row["target"]]
    if Decimal(before["nav"]) != Decimal(row["base_unit_nav"]) or Decimal(after["nav"]) != Decimal(
        row["target_unit_nav"]
    ):
        raise ValueError("LABEL_SOURCE_VALUE_CONFLICT")
    return {
        "target": row["target"],
        "base": row["base"],
        "actual": row["actual_direction"],
        "base_nav": before["nav"],
        "target_nav": after["nav"],
        "base_publication": before["ann_date"],
        "target_publication": after["ann_date"],
        "base_source_hash": before["source_hash"],
        "target_source_hash": after["source_hash"],
        "public_window_end": row["nav"]["S"],
        "prior_equal_run": prior_equal_run(row, nav),
        "reported_stock_nav_weight": row["x"][13],
    }


def flat_groups(rows, nav):
    daily = [flat_source(r, nav) for r in rows]

    def summarize(part):
        groups = {}
        for group in ("0", "1", "2+"):
            selected = [r for r in part if (str(r["prior_equal_run"]) if r["prior_equal_run"] < 2 else "2+") == group]
            groups[group] = {
                "days": len(selected),
                "actual": dict(Counter(r["actual"] for r in selected)),
                "flat_dates": [r["target"] for r in selected if r["actual"] == "FLAT"],
            }
        return groups

    return {
        "groups": summarize(daily),
        "quarters": {
            q: summarize([r for r in daily if quarter(r["target"]) == q])
            for q in sorted({quarter(r["target"]) for r in daily})
        },
        "flat_days": [r for r in daily if r["actual"] == "FLAT"],
        "daily": daily,
    }


def margin_parts(fitted, rows, anchor, positive="UP", negative="DOWN"):
    """在线性模型原始输入单位下准确分解两类 logit 差，使用同一个旧训练均值锚点。

    贡献反映模型的数值计算，不是股票涨跌的经济因果；概率不直接线性相加。
    """
    scaler, classifier = fitted["scaler"], fitted["classifier"]
    a, b = list(classifier.classes_).index(positive), list(classifier.classes_).index(negative)
    beta = (classifier.coef_[a] - classifier.coef_[b]) / scaler.scale_
    offset = float(classifier.intercept_[a] - classifier.intercept_[b] + (anchor - scaler.mean_) @ beta)
    x = np.array([r["x"] for r in rows])
    terms = (x - anchor) * beta
    reconstructed = terms.sum(axis=1) + offset
    direct = classifier.decision_function(scaler.transform(x))
    if not np.allclose(reconstructed, direct[:, a] - direct[:, b], atol=1e-12, rtol=0):
        raise ValueError("LOGIT_DECOMPOSITION_FAILED")
    return offset, terms, reconstructed


def changes(rows, old_model, new_model, old_predictions, new_predictions):
    anchor = old_model["scaler"].mean_
    ob, ot, om = margin_parts(old_model, rows, anchor)
    nb, nt, nm = margin_parts(new_model, rows, anchor)
    delta = nt - ot
    records = []
    for i, (row, old, new) in enumerate(zip(rows, old_predictions, new_predictions, strict=True)):
        expected = (row["target"], row["actual_direction"], digest(row))
        for item in (old, new):
            if (item["target"], item["actual_direction"], item["input_hash"]) != expected:
                raise ValueError("PREDICTION_IDENTITY_CHANGED")
        oc, nc = old["direction"] == row["actual_direction"], new["direction"] == row["actual_direction"]
        outcome = "both_correct" if oc and nc else "gained" if nc else "lost" if oc else "both_wrong"
        records.append(
            {
                "target": row["target"],
                "actual": row["actual_direction"],
                "outcome": outcome,
                "old_direction": old["direction"],
                "new_direction": new["direction"],
                "direction_changed": old["direction"] != new["direction"],
                "old_scores": old["scores"],
                "new_scores": new["scores"],
                "old_up_down_logit": float(om[i]),
                "new_up_down_logit": float(nm[i]),
                "delta_logit": float(nm[i] - om[i]),
                "delta_anchor": nb - ob,
                "delta_features": dict(zip(FEATURES, delta[i].tolist(), strict=True)),
                "delta_blocks": {
                    "nav7": float(delta[i, :7].sum()),
                    "holdings9": float(delta[i, 7:16].sum()),
                    "market4": float(delta[i, 16:].sum()),
                },
                "x": row["x"],
                "report_end": row["report_end"],
                "report_publication": row["report_publication"],
            }
        )
    order = np.argsort(-np.abs(delta).mean(axis=0))
    return {
        "daily": records,
        "direction_changes": sum(r["direction_changed"] for r in records),
        "outcomes": dict(Counter(r["outcome"] for r in records)),
        "all_days_absolute_feature_changes": [
            {"feature": FEATURES[i], "mean_abs_delta": float(np.abs(delta[:, i]).mean())} for i in order
        ],
        "anchor_delta": nb - ob,
        "max_abs_score_change": float(
            np.max(
                np.abs(
                    np.array([p["scores"] for p in new_predictions]) - np.array([p["scores"] for p in old_predictions])
                )
            )
        ),
    }


def flat_learning(rows, predictions, weights):
    if len(rows) != len(predictions) or len(rows) != len(weights):
        raise ValueError("TRAIN_PREDICTIONS_OR_WEIGHTS_MISMATCH")
    if any(digest(r) != p["input_hash"] for r, p in zip(rows, predictions, strict=True)):
        raise ValueError("TRAIN_IDENTITY_CHANGED")
    total = sum(weights)
    result = {}
    for name, ids in (
        ("all", list(range(len(rows)))),
        ("own", [i for i, r in enumerate(rows) if r["fund_code"] == "002112"]),
        ("peers", [i for i, r in enumerate(rows) if r["fund_code"] in ("017493", "160323")]),
    ):
        selected = [rows[i] for i in ids]
        actual_flat = [i for i in ids if rows[i]["actual_direction"] == "FLAT"]
        result[name] = {
            "rows": len(ids),
            "actual_flat_rows": len(actual_flat),
            "actual_flat_dates": len({rows[i]["target"] for i in actual_flat}),
            "flat_share_of_total_saved_weight": sum(weights[i] for i in actual_flat) / total,
            "predicted_flat": sum(predictions[i]["direction"] == "FLAT" for i in ids),
            "correct_flat": sum(predictions[i]["direction"] == "FLAT" for i in actual_flat),
            "flat_score_on_actual_flat": stats([predictions[i]["scores"][1] for i in actual_flat]),
            "flat_score_on_other": stats([predictions[i]["scores"][1] for i in ids if i not in actual_flat]),
            "flat_score_below_winner": stats(
                [
                    max(predictions[i]["scores"][0], predictions[i]["scores"][2]) - predictions[i]["scores"][1]
                    for i in actual_flat
                ]
            ),
            "by_fund_flat_counts": dict(Counter(r["fund_code"] for r in selected if r["actual_direction"] == "FLAT")),
        }
    return result


def stock_overlap(left, right):
    """披露股票的净资产权重交集；部分披露时仅为公开部分下界。"""

    def holdings(report):
        output = {}
        for r in report["holdings"]:
            weight = Decimal(r["nav_weight_pct"])
            if weight < 0 or not weight.is_finite() or r["stock_code"] in output:
                raise ValueError("INVALID_HOLDING_WEIGHT_OR_DUPLICATE")
            output[r["stock_code"]] = weight
        return {k: v for k, v in output.items() if v > 0}

    a, b = holdings(left), holdings(right)
    common = set(a) & set(b)
    return {
        "common_stocks": sorted(common),
        "count": len(common),
        "shared_nav_pct_lower_bound": float(sum(min(a[k], b[k]) for k in common)),
        "target_disclosed_pct": float(sum(a.values())),
        "peer_disclosed_pct": float(sum(b.values())),
        "target_stock_pct": float(left["stock_nav_pct"]),
        "peer_stock_pct": float(right["stock_nav_pct"]),
        "both_full_disclosure": bool(left["full_stock_disclosure"] and right["full_stock_disclosure"]),
    }


def parsed_report_paths(protocol):
    """以原清单的 parsed_file 为准，不能按文件名猜测是否为报告。"""
    work = read_json(ROOT / "peer-fold-impact/20260927-v1/report-admission-worklist.json")
    supplements = read_json(ROOT / "peer-gap-evidence/20260927-v1/report-supplements-v2.json")
    paths = {r["parsed_file"] for r in work["reports"] + supplements if r.get("parsed_file")}
    if not paths.issubset(protocol["files"]):
        raise ValueError("PARSED_REPORT_NOT_FROZEN")
    return sorted(paths)


def peers_review(inputs, snapshot, protocol):
    reports = {r["raw"]["sha256"]: r for f in snapshot["funds"].values() for r in f["reports"]}
    # 仅加载已经写入本次协议的解析文件；不去目录里挑选更新的报告。
    for name in parsed_report_paths(protocol):
        value = read_json(name)
        reports[value["raw"]["sha256"]] = value
    target = snapshot["funds"]["002112"]
    daily = []
    for row in inputs["train"]:
        if row["fund_code"] not in ("017493", "160323"):
            continue
        peer = reports[row["report_sha256"]]
        if peer["fund_code"] != row["fund_code"] or public_day(peer) >= row["target"]:
            raise ValueError("PEER_REPORT_IDENTITY_OR_TIME_CONFLICT")
        item = {"fund_code": row["fund_code"], "target": row["target"], "peer_report_sha256": row["report_sha256"]}
        try:
            own = select_report(target["reports"], target["catalog"], row["target"])
        except ValueError as exc:
            if str(exc) not in ("NO_PRIOR_PUBLIC_REPORT", "LATEST_DISCLOSURE_MISSING_NO_FALLBACK"):
                raise
            daily.append({**item, "status": str(exc)})
            continue
        daily.append(
            {
                **item,
                "status": "KNOWN",
                "target_report_sha256": own["raw"]["sha256"],
                "target_report_end": own["report_end"],
                "peer_report_end": peer["report_end"],
                **stock_overlap(own, peer),
            }
        )
    summary = {}
    for code in ("017493", "160323"):
        all_rows = [r for r in daily if r["fund_code"] == code]
        known = [r for r in all_rows if r["status"] == "KNOWN"]
        summary[code] = {
            "rows": len(all_rows),
            "known": len(known),
            "unknown": len(all_rows) - len(known),
            "zero_disclosed_overlap": sum(r["count"] == 0 for r in known),
            "shared_nav_pct_lower_bound": stats([r["shared_nav_pct_lower_bound"] for r in known]),
            "distinct_report_pairs": len({(r["target_report_sha256"], r["peer_report_sha256"]) for r in known}),
            "both_full_disclosure_days": sum(r["both_full_disclosure"] for r in known),
        }
    return {
        "summary": summary,
        "daily": daily,
        "claim": "Disclosed overlap only; not current holdings or proof of transfer value",
    }


def gate_diagnostic(controls, rows):
    sets = {
        name: {p["target"] for p in preds if p["direction"] == p["actual_direction"]}
        for name, preds in controls.items()
    }
    required = {
        c: max(sum(p["actual_direction"] == c and p["direction"] == c for p in preds) for preds in controls.values())
        for c in CLASSES
    }
    shared = set.intersection(*sets.values())
    union = set.union(*sets.values())
    return {
        "required_class_correct": required,
        "minimum_total_from_class_gates": sum(required.values()),
        "same_day_oracle_union_correct": len(union),
        "both_controls_wrong": len(rows) - len(union),
        "both_controls_correct": len(shared),
        "warning": "Oracle uses answers; it is not executable prediction or an allowed gate change",
    }


def run(directory=OUTPUT):
    protocol = read_json(directory / "protocol.json")
    if protocol["new_fit_budget"] != 0:
        raise ValueError("REVIEW_BUDGET_MUST_BE_ZERO")
    for path, expected in protocol["files"].items():
        if file_hash(path) != expected:
            raise ValueError("REVIEW_SOURCE_CHANGED:" + path)
    save_once(
        directory / ("code-freeze-" + file_hash(__file__)[:12] + ".json"),
        {
            "protocol_sha256": file_hash(directory / "protocol.json"),
            "files": {
                str(p): file_hash(p)
                for p in (
                    Path(__file__),
                    Path(__file__).resolve().parents[2] / "scripts/fund_002112_mechanism_review.py",
                    Path(__file__).resolve().parents[2] / "tests/test_fund_002112_mechanism_review.py",
                )
            },
        },
    )
    inputs, original = read_json(NEW / "frozen/inputs.json"), read_json(NEW / "frozen/original-inputs.json")
    folds, original_folds = read_json(NEW / "frozen/folds.json"), read_json(NEW / "frozen/original-folds.json")
    snapshot = read_json(SNAPSHOT)
    for value in (inputs, original):
        for row in value["train"] + value["development"]:
            validate_row(row)
    nav = {r["date"]: r for r in snapshot["funds"]["002112"]["nav"]["rows"]}
    all_exam, all_daily, fold_results = [], [], {}
    controls = {"N7": [], "L20": []}
    with no_fits():
        for new_fold, old_fold in zip(folds[:3], original_folds[:3], strict=True):
            q = new_fold["name"]
            exam = new_fold["exam"]
            all_exam.extend(exam)
            by = {}
            learning = {}
            for name, source, fold, pool in (("old", OLD, old_fold, original), ("new", NEW, new_fold, inputs)):
                spec = protocol["models"][q + "-" + name]
                fitted = restore_model(spec["directory"], spec["slot"], spec["manifest_sha256"])
                predictions = read_json(source / "predictions" / (spec["slot"] + ".json"))
                index = {(r["fund_code"], r["target"]): r for r in pool["train"]}
                train = [index[tuple(key)] for key in fold["train_ids"]]
                if fitted["training_hash"] != digest(train) or fitted["weights_hash"] != digest(fold["weights"]):
                    raise ValueError("MODEL_TRAIN_IDENTITY_CHANGED")
                verify_all_predictions(fitted, train, exam, predictions)
                learning[name] = flat_learning(train, predictions["train"], fold["weights"])
                by[name] = (fitted, predictions)
            movement = changes(exam, by["old"][0], by["new"][0], by["old"][1]["exam"], by["new"][1]["exam"])
            all_daily.extend(movement["daily"])
            old_n, new_n = sum(old_fold["weights"]), sum(new_fold["weights"])
            fold_results[q] = {
                "movement": {k: v for k, v in movement.items() if k != "daily"},
                "flat_learning": learning,
                "regularization": {
                    "old_saved_weight_sum": old_n,
                    "new_saved_weight_sum": new_n,
                    "old_l2_strength": 1 / old_n,
                    "new_l2_strength": 1 / new_n,
                    "relative_strength_new_to_old": old_n / new_n,
                    "claim": (
                        "Fixed C and formula imply effective normalization change; "
                        "no attribution to errors without controlled experiment"
                    ),
                },
            }
            for name in controls:
                controls[name].extend(read_json(NEW / f"frozen/controls/{q}-{name}-main.json")["exam"])
    if len(all_exam) != 161 or len(inputs["development"]) != 230:
        raise ValueError("FIXED_EXAM_DATES_CHANGED")
    flat = {
        "historical": flat_groups(all_exam, nav),
        "own_train": flat_groups([r for r in inputs["train"] if r["fund_code"] == "002112"], nav),
        "development_input_only": flat_groups(inputs["development"], nav),
    }
    peer = peers_review(inputs, snapshot, protocol)
    gates = gate_diagnostic(controls, all_exam)
    summary = {
        "actual_new_fits": 0,
        "cumulative_actual_fits": 58,
        "folds": fold_results,
        "gates": gates,
        "direction_changes": sum(r["direction_changed"] for r in all_daily),
        "outcomes": dict(Counter(r["outcome"] for r in all_daily)),
        "changed_dates": [r["target"] for r in all_daily if r["direction_changed"]],
        "flat": {k: {a: b for a, b in v.items() if a != "daily"} for k, v in flat.items()},
        "peers": peer["summary"],
        "evidence": "Posthoc descriptive mechanism audit, not a new fitted candidate or unseen validation",
    }
    for name, value in (("daily", all_daily), ("flat-daily", flat), ("peer-overlap", peer), ("summary", summary)):
        save_once(directory / (name + ".json"), value)
    return summary
