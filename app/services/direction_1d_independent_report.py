"""按事前规则评价未来对照；未知答案不猜测，漏题不从应做题分母移除。"""

from datetime import date, datetime, time
from pathlib import Path

import numpy as np

from app.services import direction_1d_independent as p
from app.services.direction_1d_training import weights


def metric(rows: list[dict], branch: str) -> dict:
    if not rows:
        return {
            "count": 0,
            "correct": 0,
            "accuracy": None,
            "weighted_accuracy": None,
            "up_recall": None,
            "non_up_recall": None,
        }
    w = weights(rows)
    correct = np.asarray([r["directions"].get(branch) == r["y"] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    result = {
        "count": len(rows),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "weighted_accuracy": float(np.dot(w, correct) / w.sum()),
    }
    for value, name in ((1, "up_recall"), (0, "non_up_recall")):
        mask = y == value
        result[name] = float(np.dot(w[mask], correct[mask]) / w[mask].sum()) if mask.any() else None
    scores = [r.get("scores", {}).get(branch) for r in rows]
    if all(v is not None for v in scores):
        values = np.asarray(scores)
        clipped = np.clip(values, p.POLICY["log_loss_clip"], 1 - p.POLICY["log_loss_clip"])
        result["brier"] = float(np.average((values - y) ** 2, weights=w))
        result["log_loss"] = float(np.average(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)), weights=w))
        result["score_bins"] = []
        for i in range(10):
            selected = (values >= i / 10) & ((values < (i + 1) / 10) if i < 9 else (values <= 1))
            result["score_bins"].append(
                {
                    "lower": i / 10,
                    "upper": (i + 1) / 10,
                    "count": int(selected.sum()),
                    "mean_score": float(np.average(values[selected], weights=w[selected])) if selected.any() else None,
                    "actual_up_fraction": float(np.average(y[selected], weights=w[selected]))
                    if selected.any()
                    else None,
                }
            )
    return result


def paired_interval(rows: list[dict], targets: list[str]) -> list[float] | None:
    """5个连续计划交易日为块；同日所有基金一起抽，漏采日保留零权重位置。

    重采样每日加权差值的分子/分母，避免不同有效基金数的日期被当成同样证据量。
    10000次固定seed，非循环moving-block、线性百分位；不按观察结果改算法。
    """
    if not rows or len(targets) < 5:
        return None
    positions = {d: i for i, d in enumerate(targets)}
    numerator, denominator = np.zeros(len(targets)), np.zeros(len(targets))
    for row, w in zip(rows, weights(rows), strict=True):
        i = positions[row["u"]]
        numerator[i] += w * (
            int(row["directions"]["ACTIVITY12"] == row["y"]) - int(row["directions"]["FIXED7"] == row["y"])
        )
        denominator[i] += w
    rng = np.random.default_rng(p.POLICY["bootstrap_seed"])
    starts = rng.integers(0, len(targets) - 4, size=(p.POLICY["bootstrap_repetitions"], (len(targets) + 4) // 5))
    selected = (starts[:, :, None] + np.arange(5)).reshape(len(starts), -1)[:, : len(targets)]
    totals = denominator[selected].sum(axis=1)
    if np.any(totals == 0):
        return None
    differences = numerator[selected].sum(axis=1) / totals
    return np.quantile(differences, [0.025, 0.975], method="linear").tolist()


def evaluate(contract: dict, questions: list[dict], at: datetime) -> dict:
    """question一行对应冻结范围内应做题；可缺预测或答案，不允许未知基金/重复题。"""
    targets = contract["schedule"]["targets"]
    members = contract["members"]
    expected = {(code, target) for code in members for target in targets}
    indexed = {}
    for q in questions:
        key = (q["fund_code"], q["u"])
        if key not in expected or key in indexed or q.get("y") not in (None, 0, 1):
            raise ValueError("REPORT_QUESTION_SCOPE_INVALID")
        indexed[key] = q
    due = [d for d in targets if at >= p.bounds(contract["sessions"], d)[2]]
    due_keys = {(code, target) for code in members for target in due}
    rows = []
    for code, target in sorted(due_keys):
        q = indexed.get((code, target), {})
        rows.append(
            {
                "fund_code": code,
                "family": members[code]["family"],
                "t": p.bounds(contract["sessions"], target)[0],
                "u": target,
                "y": q.get("y"),
                "directions": q.get("directions", {}),
                "scores": q.get("scores", {}),
                "reason": q.get("reason", "MISSING_FORECAST_OR_OUTCOME"),
                "events_status": q.get("events_status", "UNKNOWN"),
                "actual_direction": q.get("actual_direction"),
            }
        )
    common = [
        r
        for r in rows
        if r["y"] is not None and all(r["directions"].get(b) in (0, 1) for b in ("FIXED7", "ACTIVITY12"))
    ]
    mature = [r for r in rows if r["y"] is not None]
    branches = ["FIXED7", "ACTIVITY12", *p.POLICY["baselines"]]
    metrics = {b: metric(common, b) for b in branches}
    fixed, candidate = metrics["FIXED7"], metrics["ACTIVITY12"]
    gain = candidate["weighted_accuracy"] - fixed["weighted_accuracy"] if common else None
    interval = paired_interval(common, due)
    by_fund, by_family, segments = {}, {}, []
    for field, identifiers, output in (
        ("fund_code", list(members), by_fund),
        ("family", sorted({m["family"] for m in members.values()}), by_family),
    ):
        for key in identifiers:
            subset = [r for r in common if r[field] == key]
            a, b = metric(subset, "FIXED7"), metric(subset, "ACTIVITY12")
            eligible = len(subset) >= max(
                p.POLICY["minimum_fund_common_answers"], len(due) * p.POLICY["minimum_fund_common_fraction"]
            )
            output[key] = {
                "fixed": a,
                "candidate": b,
                "difference": b["weighted_accuracy"] - a["weighted_accuracy"] if subset else None,
                "status": "AVAILABLE" if eligible else "INSUFFICIENT",
            }
    for i in range(4):
        planned = targets[i * 30 : (i + 1) * 30]
        subset = [r for r in common if r["u"] in planned]
        a, b = metric(subset, "FIXED7"), metric(subset, "ACTIVITY12")
        segments.append(
            {
                "segment": i + 1,
                "planned_targets": planned,
                "count": len(subset),
                "difference": b["weighted_accuracy"] - a["weighted_accuracy"] if subset else None,
            }
        )
    divisor = len(rows)
    coverage = {
        "planned_total": len(expected),
        "due_total": divisor,
        "candidate": sum(r["directions"].get("ACTIVITY12") in (0, 1) for r in rows) / divisor if divisor else None,
        "common": sum(all(r["directions"].get(b) in (0, 1) for b in ("FIXED7", "ACTIVITY12")) for r in rows) / divisor
        if divisor
        else None,
        "mature": len(mature) / divisor if divisor else None,
        "unknown_answers": divisor - len(mature),
    }
    conservative = {b: sum(r["directions"].get(b) == r["y"] for r in mature) for b in ("FIXED7", "ACTIVITY12")}
    conservative["net_extra_correct"] = conservative["ACTIVITY12"] - conservative["FIXED7"]
    direction_dates = {str(y): len({r["u"] for r in common if r["y"] == y}) for y in (0, 1)}
    checks = {
        "gain_at_least_2pp": gain is not None and gain >= 0.02,
        "beats_every_simple_baseline": bool(common)
        and all(candidate["weighted_accuracy"] > metrics[b]["weighted_accuracy"] for b in p.POLICY["baselines"]),
        "interval_lower_positive": interval is not None and interval[0] > 0,
        "three_positive_segments": sum(s["difference"] is not None and s["difference"] > 0 for s in segments) >= 3,
        "two_thirds_funds_improve": sum(v["status"] == "AVAILABLE" and v["difference"] > 0 for v in by_fund.values())
        >= len(members) * (2 / 3),
        "no_fund_loses_over_5pp": all(
            v["status"] == "AVAILABLE" and v["difference"] >= -0.05 for v in by_fund.values()
        ),
        "both_direction_recalls_preserved": bool(common)
        and all(
            candidate[k] is not None and fixed[k] is not None and candidate[k] >= fixed[k]
            for k in ("up_recall", "non_up_recall")
        ),
        "direction_dates_sufficient": min(direction_dates.values()) >= 20,
        "coverage_sufficient": divisor > 0 and all(coverage[k] >= 0.95 for k in ("candidate", "common", "mature")),
        "conservative_net_positive": conservative["net_extra_correct"] > 0,
    }
    final_target = datetime.combine(date.fromisoformat(targets[-1]), time(18), p.ZONE)
    end = datetime.combine(date.fromisoformat(contract["schedule"]["grace_targets"][-1]), time(23, 59, 59), p.ZONE)
    finished = at >= end or (at >= final_target and len(mature) == len(expected))
    if not finished:
        verdict = "PENDING_FINAL_REVIEW"
    elif (
        not checks["coverage_sufficient"]
        or not checks["direction_dates_sufficient"]
        or interval is None
        or interval[0] <= 0 < interval[1]
    ):
        verdict = "INSUFFICIENT_EVIDENCE"
    else:
        verdict = "STABLE_GAIN_EVIDENCE" if all(checks.values()) else "NO_IMPROVEMENT"
    events = {
        status: {
            "count": sum(r["events_status"] == status for r in common),
            "fixed": metric([r for r in common if r["events_status"] == status], "FIXED7"),
            "candidate": metric([r for r in common if r["events_status"] == status], "ACTIVITY12"),
        }
        for status in ("KNOWN_EVENT", "UNKNOWN")
    }
    return {
        "protocol": p.PROTOCOL,
        "model_released": False,
        "new_fits": 0,
        "as_of": at.isoformat(),
        "verdict": verdict,
        "stage_finished": finished,
        "due_target_days": len(due),
        "coverage": coverage,
        "common_count": len(common),
        "metrics": metrics,
        "weighted_accuracy_gain": gain,
        "paired_interval_95": interval,
        "segments": segments,
        "funds": by_fund,
        "families": by_family,
        "direction_target_days": direction_dates,
        "conservative_all_mature": conservative,
        "checks": checks,
        "event_diagnostics": events,
        "flat_common_count": sum(r["actual_direction"] == "FLAT" for r in common),
        "failed_checks": [k for k, v in checks.items() if not v],
    }


def collect_records(root: Path, contract: dict, models: dict) -> list[dict]:
    """只认完整恢复链及首次答案；迟到/损坏保留原因，不并入共同成绩。"""
    rows = []
    for target in contract["schedule"]["targets"]:
        for code in contract["members"]:
            folder = root / "questions" / target / code
            row = {"fund_code": code, "u": target}
            try:
                checked = p.verify_question(folder, contract, models)
                if checked["status"] == "VERIFIED":
                    row.update(directions=checked["prediction"]["directions"], scores=checked["prediction"]["scores"])
                else:
                    row["reason"] = checked["status"]
            except (OSError, ValueError, KeyError) as exc:
                row["reason"] = "MISSING_OR_INVALID_FORECAST:" + type(exc).__name__
            answer_file = folder / "first-outcome.json"
            if answer_file.exists():
                first = p.unseal(answer_file)
                evidence = p.unseal(folder / "outcomes" / (first["version"] + ".json"))
                if p.digest(evidence) != first["evidence_hash"]:
                    raise ValueError("REPORT_OUTCOME_HASH_MISMATCH")
                row["y"] = evidence["answer"]["y"]
                row["actual_direction"] = evidence["answer"]["actual_direction"]
                row["events_status"] = evidence["observation"].get("events_status", "UNKNOWN")
            elif (folder / "missing-forecast-answer.json").exists():
                row["y"] = p.unseal(folder / "missing-forecast-answer.json")["answer"]["y"]
            rows.append(row)
    return rows
