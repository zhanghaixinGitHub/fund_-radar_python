"""已观察样本的错误分解与事前规则下的同题评价；相关贡献不是因果证明。"""

import math
from collections import Counter
from datetime import date
from fractions import Fraction

from app.schemas.direction_training import DirectionAnswer
from app.services.direction_linear_protocol import BASELINES, VERSION, study_rules
from app.services.direction_training_artifacts import read_json, read_jsonl
from app.services.direction_training_dataset import FUNDS, exam_dates
from app.services.direction_training_evaluation import complete_blocks, grouped_metrics, metrics, paired_interval
from app.services.trading_calendar import load_calendar


def macro_accuracy(grouped):
    values = list(grouped["per_fund"].values())
    if not all(values):
        return None
    return sum(Fraction(v["correct_count"], v["sample_count"]) for v in values) / len(values)


def accuracy_delta(left, right):
    a, b = macro_accuracy(left), macro_accuracy(right)
    return float(a - b) if a is not None and b is not None else None


def diagnose(folder):
    old = read_json(folder / "nav-metrics.json")
    answers = {
        (r["window"], r["sample_key"]): r["answer"] for r in read_jsonl(folder / "nav-answers.jsonl") if r["answer"]
    }
    rows, windows = [], {}
    for name in old["per_window"]:
        bundle = read_json(folder / f"nav-prepared-{name}.json")
        job = bundle["jobs"].get("COMPLETE")
        if job is None:
            continue
        model = read_json(folder / f"nav-models-{name}.json")["COMPLETE"]["artifact"]
        scores = {
            r["sample_key"]: r
            for r in read_jsonl(folder / f"nav-predictions-{name}.jsonl")
            if r["scenario"] == "COMPLETE_CLEAN" and r["score"] is not None
        }
        windows[name] = {}
        for item in job["exam"]["CLEAN"]:
            key = f"{item['fund']}:{item['cutoff']}"
            answer = answers.get((name, key))
            if not answer or answer["available_at"] > job["window"]["exam_end"]:
                continue
            p, y = scores[key], answer["y"]
            terms = {
                feature: (x - mean) / scale * coef
                for feature, x, mean, scale, coef in zip(
                    model["feature_names"], item["x"], model["mean"], model["scale"], model["coefficients"], strict=True
                )
            }
            kind = (
                "CORRECT_UP"
                if y == p["predicted_up"] == 1
                else "CORRECT_NON_UP"
                if y == p["predicted_up"]
                else "FALSE_UP"
                if p["predicted_up"]
                else "MISSED_UP"
            )
            rows.append(
                {
                    "window": name,
                    "fund": item["fund"],
                    "cutoff": item["cutoff"],
                    "sample_key": key,
                    "y": y,
                    "score": p["score"],
                    "predicted_up": p["predicted_up"],
                    "error_kind": kind,
                    "x": item["x"],
                    "logit_terms": terms,
                    "intercept": model["intercept"],
                }
            )
        for fund in FUNDS:
            exam = [r for r in rows if r["window"] == name and r["fund"] == fund]
            fit = [r for r in job["fit"] if r["input"]["fund"] == fund]
            windows[name][fund] = {
                "exam": metrics([(r["y"], r["score"]) for r in exam]),
                "fit_count": len(fit),
                "fit_up_rate": sum(r["answer"]["y"] for r in fit) / len(fit),
                "coefficients": dict(zip(model["feature_names"], model["coefficients"], strict=True)),
                "error_logit_contributions": {
                    kind: {
                        feature: sum(r["logit_terms"][feature] for r in exam if r["error_kind"] == kind)
                        / sum(r["error_kind"] == kind for r in exam)
                        for feature in model["feature_names"]
                    }
                    for kind in ("FALSE_UP", "MISSED_UP")
                    if any(r["error_kind"] == kind for r in exam)
                },
            }
    monthly = {}
    for fund, month in sorted({(r["fund"], r["cutoff"][:7]) for r in rows}):
        selected = [r for r in rows if r["fund"] == fund and r["cutoff"].startswith(month)]
        monthly[f"{fund}:{month}"] = {
            "metrics": metrics([(r["y"], r["score"]) for r in selected]),
            "descriptive_only": True,
            "low_sample": len(selected) < 30,
        }
    return {
        "version": VERSION,
        "source_status": "PREVIOUSLY_OBSERVED_DEVELOPMENT",
        "aggregate": grouped_metrics(rows, FUNDS),
        "per_window": windows,
        "monthly": monthly,
        "rows": rows,
        "hypotheses": {
            "RECENT_18M": "Window error reversals and FIT/EXAM class shifts warrant testing stale history.",
            "DROP_60D_GROUP": "Negative return60/drawdown weights and positive position weight warrant group ablation.",
            "PER_FUND": "Cross-fund error asymmetry warrants testing shared versus separate coefficients.",
        },
        "limitations": "Descriptive correlations and logit contributions do not establish causal error explanations.",
    }


def evaluate(protocol, predictions, answers):
    branches, _ = study_rules(protocol["version"])
    if protocol["branches"] != list(branches) or protocol["baselines"] != list(BASELINES):
        raise ValueError("LINEAR_EVALUATION_BRANCHES")
    scenarios = (*branches, *BASELINES)
    windows = {w["name"]: w for w in protocol["windows"]}
    planned = {
        (w["name"], f"{f}:{d}")
        for w in windows.values()
        for f in FUNDS
        for d in exam_dates(date.fromisoformat(w["cal_end"]), date.fromisoformat(w["exam_end"]))[1]
    }
    answer_map = {(r["window"], r["sample_key"]): r["answer"] for r in answers}
    if len(answer_map) != len(answers) or set(answer_map) != planned:
        raise ValueError("LINEAR_ANSWER_PLAN")
    for (_window, key), value in answer_map.items():
        if value is not None:
            answer = DirectionAnswer.model_validate(value)
            if answer.key != key or answer.end != load_calendar().future_sessions(answer.cutoff)[-1]:
                raise ValueError("LINEAR_ANSWER_IDENTITY_OR_HORIZON")
    identities = {(r["window"], r["sample_key"], r["branch"]) for r in predictions}
    if identities != {(w, k, b) for w, k in planned for b in scenarios} or len(identities) != len(predictions):
        raise ValueError("LINEAR_PREDICTION_PLAN")
    available, failures = {b: {} for b in scenarios}, {b: Counter() for b in scenarios}
    for row in predictions:
        branch, key, score = row["branch"], (row["window"], row["sample_key"]), row["score"]
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("LINEAR_PREDICTION_NUMBER")
        if row["sample_key"] != f"{row['fund']}:{row['cutoff']}":
            raise ValueError("LINEAR_PREDICTION_IDENTITY")
        answer = answer_map[key]
        reason = None
        if score is None:
            if row["status"] == "PREDICTED" or row["predicted_up"] is not None:
                raise ValueError("LINEAR_PREDICTION_STATE")
            reason = row["status"]
        elif row["status"] != "PREDICTED" or row["predicted_up"] != int(score > 0.5):
            raise ValueError("LINEAR_PREDICTION_STATE")
        elif answer is None:
            reason = "ANSWER_UNAVAILABLE"
        elif answer["available_at"] > windows[row["window"]]["exam_end"]:
            reason = "ANSWER_NOT_MATURE"
        if reason:
            failures[branch][reason] += 1
        else:
            available[branch][key] = {**row, "y": answer["y"], "correct": int(row["predicted_up"] == answer["y"])}
    common = set.intersection(*(set(v) for v in available.values()))
    summary = {b: grouped_metrics([v[k] for k in sorted(common)], FUNDS) for b, v in available.items()}
    per_window = {
        w: {
            b: grouped_metrics([r for (wn, k), r in v.items() if wn == w and (wn, k) in common], FUNDS)
            for b, v in available.items()
        }
        for w in windows
    }
    coverage = {
        w: {
            f: {
                "planned": sum(wn == w and k.startswith(f + ":") for wn, k in planned),
                "common": sum(wn == w and k.startswith(f + ":") for wn, k in common),
            }
            for f in FUNDS
        }
        for w in windows
    }
    valid = [
        w
        for w in windows
        if all(
            v["common"] >= protocol["minimum"]["EXAM"]
            and v["common"] / max(1, v["planned"]) >= protocol["minimum_coverage"]
            for v in coverage[w].values()
        )
    ]
    blocks, exclusions = complete_blocks(protocol, common)
    enough = (
        len(valid) >= protocol["minimum_windows"]
        and len(blocks) >= protocol["bootstrap"]["minimum_complete_blocks"]
        and not any(r["status"] == "FAILED" for r in predictions)
    )
    comparisons, statuses = {}, {}
    gate = protocol["selection"]
    for branch in branches[1:]:
        comparisons[branch] = {}
        passed, reasons = enough, []
        if not enough:
            reasons.append("INSUFFICIENT_COMMON_WINDOWS_OR_BLOCKS")
        left = summary[branch]
        for reference in ("REFERENCE", *BASELINES):
            right = summary[reference]
            lm, rm = left["equal_fund_macro"], right["equal_fund_macro"]
            delta = {
                k: lm[k] - rm[k] if lm[k] is not None and rm[k] is not None else None
                for k in ("accuracy", "balanced_accuracy", "brier_score", "log_loss")
            }
            delta["accuracy"] = accuracy_delta(left, right)
            interval = paired_interval(protocol, blocks, available[branch], available[reference])
            comparisons[branch][reference] = {"delta": delta, "time_blocks": interval}
            good = (
                delta["accuracy"] is not None
                and delta["accuracy"] > 0
                and interval["interval"]
                and interval["interval"][0] > 1e-12
                and delta["balanced_accuracy"] is not None
                and delta["balanced_accuracy"] >= -1e-12
            )
            if not good:
                passed = False
                reasons.append(f"NO_STABLE_ADVANTAGE_VS_{reference}")
        per_fund = {
            f: left["per_fund"][f]["accuracy"] - summary["REFERENCE"]["per_fund"][f]["accuracy"]
            if left["per_fund"][f] and summary["REFERENCE"]["per_fund"][f]
            else None
            for f in FUNDS
        }
        positive_funds = sum(v is not None and v > 0 for v in per_fund.values())
        positive_windows = sum(accuracy_delta(per_window[w][branch], per_window[w]["REFERENCE"]) > 0 for w in valid)
        brier = comparisons[branch]["REFERENCE"]["delta"]["brier_score"]
        stable = (
            positive_funds / len(FUNDS) >= gate["minimum_positive_fund_fraction"]
            and positive_windows / max(1, len(valid)) >= gate["minimum_positive_window_fraction"]
            and all(v is not None and v >= gate["per_fund_accuracy_delta_min"] for v in per_fund.values())
            and brier is not None
            and brier <= gate["brier_delta_vs_reference_max"]
        )
        if not stable:
            passed = False
            reasons.append("CROSS_FUND_WINDOW_OR_BRIER_STABILITY_FAILED")
        statuses[branch] = {
            "status": "INSUFFICIENT_DATA"
            if not enough
            else "QUALIFIED_DEVELOPMENT_CANDIDATE"
            if passed
            else "NO_STABLE_GAIN",
            "reasons": reasons,
            "positive_funds": positive_funds,
            "positive_windows": positive_windows,
            "window_denominator": len(valid),
            "per_fund_accuracy_delta": per_fund,
        }
    qualified = [b for b in branches[1:] if statuses[b]["status"] == "QUALIFIED_DEVELOPMENT_CANDIDATE"]
    qualified.sort(
        key=lambda b: (
            -macro_accuracy(summary[b]),
            -summary[b]["equal_fund_macro"]["balanced_accuracy"],
            summary[b]["equal_fund_macro"]["brier_score"],
            b,
        )
    )
    return {
        "version": protocol["version"],
        "same_question_count": len(common),
        "common": summary,
        "available_descriptive": {b: grouped_metrics(list(v.values()), FUNDS) for b, v in available.items()},
        "per_window": per_window,
        "coverage": coverage,
        "valid_windows": valid,
        "time_blocks": {"count": len(blocks), "exclusions": exclusions, "scope": "COMPLETE_BLOCKS_ONLY"},
        "comparisons": comparisons,
        "candidate_status": statuses,
        "selected_candidate": qualified[0] if qualified else None,
        "failures": {b: dict(v) for b, v in failures.items()},
        "independent_test": False,
        "model_released": False,
    }
