"""同题方向/概率指标和同步时间块区间；数据不足不产生胜出结论。"""

import math
from collections import Counter, defaultdict
from datetime import date

from app.services.direction_training_dataset import exam_dates
from app.services.direction_training_protocol import BASELINES, CANDIDATES


def metrics(pairs):
    if not pairs:
        return None
    if any(type(y) is not int or y not in (0, 1) or not math.isfinite(s) or not 0 <= s <= 1 for y, s in pairs):
        raise ValueError("INVALID_METRIC_PAIRS")
    tn, fp, fn, tp = (
        sum(y == actual and int(s > 0.5) == predicted for y, s in pairs)
        for actual, predicted in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    n = len(pairs)
    up_recall = tp / (tp + fn) if tp + fn else None
    non_up_recall = tn / (tn + fp) if tn + fp else None
    bins = []
    for index in range(5):
        rows = [(y, s) for y, s in pairs if min(4, int(s * 5)) == index]
        bins.append(
            {
                "lower": index / 5,
                "upper": (index + 1) / 5,
                "count": len(rows),
                "mean_score": sum(s for y, s in rows) / len(rows) if rows else None,
                "up_rate": sum(y for y, s in rows) / len(rows) if rows else None,
                "low_sample": len(rows) < 30,
            }
        )
    return {
        "sample_count": n,
        "correct_count": tn + tp,
        "accuracy": (tn + tp) / n,
        "balanced_accuracy": (up_recall + non_up_recall) / 2
        if up_recall is not None and non_up_recall is not None
        else None,
        "actual_up_rate": (tp + fn) / n,
        "predicted_up_rate": (tp + fp) / n,
        "up_recall": up_recall,
        "non_up_recall": non_up_recall,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "brier_score": sum((s - y) ** 2 for y, s in pairs) / n,
        "log_loss": -sum(
            y * math.log(min(1 - 1e-8, max(1e-8, s))) + (1 - y) * math.log(1 - min(1 - 1e-8, max(1e-8, s)))
            for y, s in pairs
        )
        / n,
        "log_loss_clip_count": sum(s < 1e-8 or s > 1 - 1e-8 for y, s in pairs),
        "reliability": bins,
    }


def grouped_metrics(records, funds):
    details = {f: metrics([(r["y"], r["score"]) for r in records if r["fund"] == f]) for f in funds}
    macro = {}
    for field in ("accuracy", "balanced_accuracy", "brier_score", "log_loss"):
        values = [m[field] if m else None for m in details.values()]
        macro[field] = sum(values) / len(values) if all(v is not None for v in values) else None
    return {
        "equal_fund_macro": macro,
        "per_fund": details,
        "sample_weighted": metrics([(r["y"], r["score"]) for r in records]),
    }


def complete_blocks(protocol, common_keys, *, planned_by_window=None):
    funds = protocol["funds"]
    size = protocol["bootstrap"]["sessions"]
    blocks, exclusions = [], {}
    for window in protocol["windows"]:
        name = window["name"]
        planned = (
            planned_by_window[name]
            if planned_by_window is not None
            else exam_dates(date.fromisoformat(window["cal_end"]), date.fromisoformat(window["exam_end"]))[1]
        )
        candidate_count, discarded = len(planned) // size, 0
        for start in range(0, candidate_count * size, size):
            dates = [str(d) for d in planned[start : start + size]]
            keys = [(name, f"{f}:{d}") for f in funds for d in dates]
            if all(k in common_keys for k in keys):
                blocks.append({"window": name, "dates": dates, "keys": keys})
            else:
                discarded += 1
        exclusions[name] = {
            "calendar_blocks": candidate_count,
            "incomplete_blocks": discarded,
            "tail_cutoffs_per_fund": len(planned) % size,
        }
    return blocks, exclusions


def paired_interval(protocol, blocks, left, right):
    import numpy as np

    if not blocks:
        return {"block_count": 0, "point_delta": None, "interval": None, "per_fund": {}}
    funds = protocol["funds"]
    data = np.asarray(
        [
            [
                sum((left[k]["correct"] - right[k]["correct"]) for k in b["keys"] if left[k]["fund"] == f)
                / len(b["dates"])
                for f in funds
            ]
            for b in blocks
        ]
    )
    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    strata = [[i for i, b in enumerate(blocks) if b["window"] == w] for w in sorted({b["window"] for b in blocks})]
    indices = np.concatenate(
        [np.asarray(s)[rng.integers(0, len(s), size=(protocol["bootstrap"]["repeats"], len(s)))] for s in strata],
        axis=1,
    )
    boot = data[indices].mean(axis=1)
    brier_data = np.asarray(
        [
            [
                sum(
                    (left[k]["score"] - left[k]["y"]) ** 2 - (right[k]["score"] - right[k]["y"]) ** 2
                    for k in b["keys"]
                    if left[k]["fund"] == f
                )
                / len(b["dates"])
                for f in funds
            ]
            for b in blocks
        ]
    )
    brier_boot = brier_data[indices].mean(axis=1)
    # 同一时间块索引用于两候选和全部基金，保留其相关性。
    interval = np.quantile(boot.mean(axis=1), [0.025, 0.975]).tolist()
    return {
        "block_count": len(blocks),
        "point_delta": float(data.mean()),
        "interval": interval,
        "sampling": "PAIRED_SYNCHRONIZED_STRATIFIED_BY_EXAM_WINDOW",
        "brier_point_delta": float(brier_data.mean()),
        "brier_interval": np.quantile(brier_boot.mean(axis=1), [0.025, 0.975]).tolist(),
        "per_fund": {
            f: {
                "point_delta": float(data[:, i].mean()),
                "interval": np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
                "brier_point_delta": float(brier_data[:, i].mean()),
                "brier_interval": np.quantile(brier_boot[:, i], [0.025, 0.975]).tolist(),
            }
            for i, f in enumerate(funds)
        },
    }


def evaluate(protocol, predictions, answers, execution):
    branches = (*CANDIDATES, *BASELINES)
    funds = protocol["funds"]
    if set(execution) != {w["name"] for w in protocol["windows"]}:
        raise ValueError("EXECUTION_WINDOW_MISMATCH")
    expected = set()
    for window in protocol["windows"]:
        if execution[window["name"]]["A"]["status"] == "INSUFFICIENT_DATA":
            continue
        _, dates = exam_dates(date.fromisoformat(window["cal_end"]), date.fromisoformat(window["exam_end"]))
        expected.update((window["name"], f"{f}:{d}", b) for f in funds for d in dates for b in branches)
    if {(r["window"], r["sample_key"], r["candidate"]) for r in predictions} != expected:
        raise ValueError("PLANNED_CALENDAR_PREDICTION_MISMATCH")
    answer_map = {}
    for row in answers:
        key = (row["window"], row["sample_key"])
        if key in answer_map:
            raise ValueError("DUPLICATE_ANSWER")
        answer_map[key] = row
    available = {b: {} for b in branches}
    planned, failures, identities = defaultdict(set), Counter(), set()
    for row in predictions:
        branch, key = row["candidate"], (row["window"], row["sample_key"])
        identity = (*key, branch)
        if branch not in branches or identity in identities or row["fund"] not in funds:
            raise ValueError("PREDICTION_IDENTITY_INVALID")
        identities.add(identity)
        planned[row["fund"]].add(key)
        answer = answer_map.get(key, {}).get("answer")
        if row["score"] is None:
            failures[(branch, row["status"])] += 1
        elif answer is None:
            failures[(branch, "ANSWER_UNAVAILABLE")] += 1
        else:
            if row["predicted_up"] != int(row["score"] > 0.5):
                raise ValueError("DIRECTION_SCORE_MISMATCH")
            y = answer["y"]
            available[branch][key] = {**row, "y": y, "correct": int(y == row["predicted_up"])}
    if any((w, k, b) not in identities for keys in planned.values() for w, k in keys for b in branches):
        raise ValueError("MISSING_PLANNED_PREDICTION")
    common = set.intersection(*(set(rows) for rows in available.values()))
    common_rows = {b: [r for k, r in available[b].items() if k in common] for b in branches}
    protocol_planned = sum(
        len(exam_dates(date.fromisoformat(w["cal_end"]), date.fromisoformat(w["exam_end"]))[1])
        for w in protocol["windows"]
    )
    coverage = {
        f: {
            "all_protocol_planned": protocol_planned,
            "planned": len(planned[f]),
            "common": sum(k in common for k in planned[f]),
            "ratio": sum(k in common for k in planned[f]) / len(planned[f]) if planned[f] else 0,
            "all_protocol_coverage": sum(k in common for k in planned[f]) / protocol_planned,
        }
        for f in funds
    }
    windows = sorted({w for w, k in common})
    valid_windows = [
        w
        for w in windows
        if all(
            sum(r["window"] == w and r["fund"] == f for r in common_rows["A"]) >= protocol["minimum"]["EXAM"]
            and sum((w, k) in common for w2, k in planned[f] if w2 == w) / sum(w2 == w for w2, k in planned[f])
            >= protocol["minimum_coverage"]
            for f in funds
        )
    ]
    result = {
        "version": "DIRECTION_EVALUATION_V1",
        "common_key_count": len(common),
        "coverage": coverage,
        "common": {b: grouped_metrics(common_rows[b], funds) for b in branches},
        "available_descriptive": {b: grouped_metrics(list(available[b].values()), funds) for b in branches},
        "per_window": {},
        "failures": [{"candidate": b, "reason": r, "count": n} for (b, r), n in sorted(failures.items())],
        "execution": execution,
        "independent_test": False,
        "publication_status": "MODEL_NOT_RELEASED",
    }
    for window in protocol["windows"]:
        name = window["name"]
        result["per_window"][name] = {
            b: grouped_metrics([r for r in common_rows[b] if r["window"] == name], funds) for b in branches
        }
    blocks, exclusions = complete_blocks(protocol, common)
    result["time_blocks"] = {
        "complete_count": len(blocks),
        "blocks": [{"window": b["window"], "dates": b["dates"]} for b in blocks],
        "excluded": exclusions,
        "interval_domain": "COMPLETE_BLOCKS_ONLY",
    }
    for branch in branches:
        values = [result["per_window"][w][branch]["equal_fund_macro"]["accuracy"] for w in windows]
        result["common"][branch]["window_equal_mean_accuracy"] = sum(values) / len(values) if values else None
    enough = (
        len(valid_windows) >= protocol["minimum_windows"]
        and len(blocks) >= protocol["bootstrap"]["minimum_complete_blocks"]
        and all(c["ratio"] >= protocol["minimum_coverage"] for c in coverage.values())
        and not any(r["status"] == "FAILED" for r in predictions)
    )
    result["evidence_gates"] = {
        "scored_window_count": len(windows),
        "valid_window_count": len(valid_windows),
        "valid_windows": valid_windows,
        "minimum_windows": protocol["minimum_windows"],
        "complete_block_count_per_fund": len(blocks),
        "minimum_blocks": protocol["bootstrap"]["minimum_complete_blocks"],
        "sufficient": enough,
    }
    comparisons, statuses = {}, {}
    for branch in CANDIDATES:
        references = (*BASELINES, *(("A",) if branch != "A" else ()))
        comparisons[branch] = {}
        for reference in references:
            left, right = result["common"][branch]["equal_fund_macro"], result["common"][reference]["equal_fund_macro"]
            comparison = {
                "accuracy_delta": left["accuracy"] - right["accuracy"] if common else None,
                "balanced_accuracy_delta": left["balanced_accuracy"] - right["balanced_accuracy"]
                if left["balanced_accuracy"] is not None and right["balanced_accuracy"] is not None
                else None,
                "time_block_delta": paired_interval(protocol, blocks, available[branch], available[reference]),
            }
            comparison["positive_windows"] = sum(
                result["per_window"][w][branch]["equal_fund_macro"]["accuracy"]
                > result["per_window"][w][reference]["equal_fund_macro"]["accuracy"]
                for w in windows
            )
            comparison["window_denominator"] = len(windows)
            comparison["positive_funds"] = (
                sum(
                    result["common"][branch]["per_fund"][f]["accuracy"]
                    > result["common"][reference]["per_fund"][f]["accuracy"]
                    for f in funds
                )
                if common
                else 0
            )
            comparison["fund_denominator"] = len(funds)
            comparisons[branch][reference] = comparison
        primary = comparisons[branch]["TRAIN_UP_FREQUENCY" if branch == "A" else "A"]
        stable = enough and all(
            c["accuracy_delta"] > 0 and c["time_block_delta"]["interval"][0] > 0 for c in comparisons[branch].values()
        )
        stable = stable and (
            primary["balanced_accuracy_delta"] is not None
            and primary["balanced_accuracy_delta"] >= 0
            and primary["positive_windows"] * 3 >= len(windows) * 2
            and primary["positive_funds"] * 3 >= len(funds) * 2
            and all(p["interval"][1] >= 0 for p in primary["time_block_delta"]["per_fund"].values())
        )
        statuses[branch] = "INSUFFICIENT_DATA" if not enough else "RESEARCH_CANDIDATE" if stable else "NO_STABLE_GAIN"
    result["comparisons"], result["candidate_status"] = comparisons, statuses
    result["research_status"] = (
        "INSUFFICIENT_DATA"
        if not enough
        else ("RESEARCH_CANDIDATE" if "RESEARCH_CANDIDATE" in statuses.values() else "NO_STABLE_GAIN")
    )
    result["reference_to_keep"] = "A"
    return result
