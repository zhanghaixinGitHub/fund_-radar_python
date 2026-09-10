"""逐基金/窗口及同题块比较；人为缺失不增加独立样本量，不用于宣称正式模型有效。"""

import math
from datetime import date

from app.schemas.direction_training import DirectionAnswer
from app.services.direction_nav_protocol import PAIRS, SCENARIOS, VERSION
from app.services.direction_training_dataset import exam_dates
from app.services.direction_training_evaluation import complete_blocks, grouped_metrics, paired_interval
from app.services.trading_calendar import load_calendar


def evaluate(protocol, predictions, answers):
    windows = {w["name"]: w for w in protocol["windows"]}
    planned = {
        (w["name"], f"{f}:{d}")
        for w in windows.values()
        for f in protocol["funds"]
        for d in exam_dates(date.fromisoformat(w["cal_end"]), date.fromisoformat(w["exam_end"]))[1]
    }
    answer_map = {(r["window"], r["sample_key"]): r for r in answers}
    if len(answer_map) != len(answers) or set(answer_map) != planned:
        raise ValueError("ANSWER_PLAN_OR_DUPLICATE")
    for (_window, key), row in answer_map.items():
        if key != f"{row['fund']}:{row['cutoff']}":
            raise ValueError("ANSWER_IDENTITY")
        for field in ("answer", "legacy_answer"):
            if row[field]:
                item = DirectionAnswer.model_validate(row[field])
                if item.key != key:
                    raise ValueError("ANSWER_KEY_MISMATCH")
                if item.end != load_calendar().future_sessions(item.cutoff)[-1]:
                    raise ValueError("ANSWER_HORIZON")
    expected = {(w, k, s) for w, k in planned for s in SCENARIOS}
    identities = {(r["window"], r["sample_key"], r["scenario"]) for r in predictions}
    if len(identities) != len(predictions) or identities != expected:
        raise ValueError("PREDICTION_PLAN_OR_DUPLICATE")
    available, failures = {s: {} for s in SCENARIOS}, {s: {} for s in SCENARIOS}
    for row in predictions:
        scenario, key = row["scenario"], (row["window"], row["sample_key"])
        if row["sample_key"] != f"{row['fund']}:{row['cutoff']}":
            raise ValueError("PREDICTION_IDENTITY")
        answer = answer_map[key]["legacy_answer" if scenario == "LEGACY_CLEAN" else "answer"]
        score = row["score"]
        reason = None
        if score is None:
            if row["predicted_up"] is not None or row["status"] == "PREDICTED":
                raise ValueError("INVALID_UNAVAILABLE_PREDICTION")
            reason = row["status"]
        elif not math.isfinite(score) or not 0 <= score <= 1 or row["predicted_up"] != int(score > 0.5):
            raise ValueError("INVALID_PREDICTION_NUMBER")
        elif answer is None:
            reason = "ANSWER_UNAVAILABLE"
        elif answer["available_at"] > windows[row["window"]]["exam_end"]:
            reason = "ANSWER_NOT_MATURE"
        if reason:
            failures[scenario][reason] = failures[scenario].get(reason, 0) + 1
        else:
            available[scenario][key] = {**row, "y": answer["y"], "correct": int(row["predicted_up"] == answer["y"])}
    per_window, coverage = {}, {}
    for name in windows:
        per_window[name], coverage[name] = {}, {}
        for scenario in SCENARIOS:
            rows = [r for (w, k), r in available[scenario].items() if w == name]
            per_window[name][scenario] = grouped_metrics(rows, protocol["funds"])
            coverage[name][scenario] = {
                f: {
                    "planned": sum(w == name and k.startswith(f + ":") for w, k in planned),
                    "usable": sum(r["fund"] == f for r in rows),
                }
                for f in protocol["funds"]
            }
    comparisons = {}
    for left, right in PAIRS:
        keys = set(available[left]) & set(available[right])
        lrows, rrows = ([available[s][k] for k in sorted(keys)] for s in (left, right))
        lm, rm = (grouped_metrics(rows, protocol["funds"]) for rows in (lrows, rrows))
        blocks, exclusions = complete_blocks(protocol, keys)
        interval = paired_interval(protocol, blocks, available[left], available[right])
        deltas = {
            m: lm["equal_fund_macro"][m] - rm["equal_fund_macro"][m]
            if lm["equal_fund_macro"][m] is not None and rm["equal_fund_macro"][m] is not None
            else None
            for m in ("accuracy", "balanced_accuracy", "brier_score", "log_loss")
        }
        valid = [
            w
            for w in windows
            if all(
                sum(kw == w and k.startswith(f + ":") for kw, k in keys) >= protocol["minimum"]["EXAM"]
                and sum(kw == w and k.startswith(f + ":") for kw, k in keys)
                / max(1, sum(pw == w and k.startswith(f + ":") for pw, k in planned))
                >= protocol["minimum_coverage"]
                for f in protocol["funds"]
            )
        ]
        score_changes = [abs(a["score"] - b["score"]) for a, b in zip(lrows, rrows, strict=True)]
        per_fund_delta = {
            f: lm["per_fund"][f]["accuracy"] - rm["per_fund"][f]["accuracy"]
            if lm["per_fund"][f] and rm["per_fund"][f]
            else None
            for f in protocol["funds"]
        }
        enough = (
            len(valid) >= protocol["minimum_windows"]
            and len(blocks) >= protocol["bootstrap"]["minimum_complete_blocks"]
        )
        comparisons[f"{left}_VS_{right}"] = {
            "same_question_count": len(keys),
            "left": lm,
            "right": rm,
            "delta": deltas,
            "valid_windows": valid,
            "sufficient_blocks_and_windows": enough,
            "time_blocks": interval,
            "excluded_blocks": exclusions,
            "per_fund_accuracy_delta": per_fund_delta,
            "mean_absolute_score_change": sum(score_changes) / len(score_changes) if score_changes else None,
            "max_absolute_score_change": max(score_changes, default=None),
            "direction_changes": sum(a["predicted_up"] != b["predicted_up"] for a, b in zip(lrows, rrows, strict=True)),
        }
    checks = {}
    gate = protocol["tolerance_gate"]
    for scenario in ("TOLERANT_CLEAN", "TOLERANT_DROP1", "TOLERANT_DROP2"):
        comparison = comparisons[f"{scenario}_VS_COMPLETE_CLEAN"]
        interval = comparison["time_blocks"]
        checks[scenario] = bool(
            comparison["sufficient_blocks_and_windows"]
            and interval["interval"]
            and interval["interval"][0] >= gate["accuracy_ci_lower_min"]
            and interval["brier_interval"][1] <= gate["brier_ci_upper_max"]
            and comparison["delta"]["balanced_accuracy"] is not None
            and comparison["delta"]["balanced_accuracy"] >= gate["balanced_accuracy_delta_min"]
            and all(
                v is not None and v >= gate["per_fund_accuracy_delta_min"]
                for v in comparison["per_fund_accuracy_delta"].values()
            )
        )
    return {
        "version": VERSION,
        "purpose": "DEVELOPMENT_AND_SYNTHETIC_GAP_STRESS_ONLY",
        "available_descriptive": {
            s: grouped_metrics(list(rows.values()), protocol["funds"]) for s, rows in available.items()
        },
        "per_window": per_window,
        "coverage": coverage,
        "failures": failures,
        "comparisons": comparisons,
        "tolerance_checks": checks,
        "tolerance_status": "SYNTHETIC_CHECK_PASSED_REAL_GAPS_UNVERIFIED"
        if all(checks.values())
        else "NOT_ADOPTED_NONINFERIORITY_NOT_ESTABLISHED",
        "default_gap_policy": "REQUIRE_COMPLETE",
        "default_nav_availability_policy": protocol["availability_policy"],
        "independent_test": False,
        "model_released": False,
        "limitation": "Development periods and synthetic isolated masks; real missingness and release unverified.",
    }
