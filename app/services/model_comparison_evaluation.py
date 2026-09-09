"""共享评分及按真实交易日分块的配对差值；不把失败和缺失题压缩掉。"""

import math
from collections import Counter
from decimal import Decimal

from app.services.cash_reinvestment_research import FUNDS
from app.services.historical_nav_calibration import reliability_report
from app.services.historical_nav_evaluation import calculate_baseline_metrics


def metrics(pairs: list[tuple[int, float]]) -> dict | None:
    if not pairs:
        return None
    labels = tuple(int(y) for y, _ in pairs)
    scores = tuple(Decimal(str(p)) for _, p in pairs)
    result = calculate_baseline_metrics(labels, scores).model_dump(mode="json")
    result["log_loss"] = -sum(
        y * math.log(max(1e-8, min(1 - 1e-8, p))) + (1 - y) * math.log(max(1e-8, min(1 - 1e-8, 1 - p)))
        for y, p in pairs
    ) / len(pairs)
    result["reliability"] = reliability_report(labels, scores).model_dump(mode="json")
    return result


def _macro(per_fund):
    if any(per_fund.get(f) is None for f in FUNDS):
        return None
    return sum(float(per_fund[f]["brier_score"]) for f in FUNDS) / len(FUNDS)


def paired_bootstrap(planned_dates: list[str], errors: dict[str, dict[str, float]], config: dict) -> dict:
    import numpy as np

    block_size = config["block_sessions"]
    blocks = [planned_dates[i : i + block_size] for i in range(0, len(planned_dates), block_size)]
    blocks = [b for b in blocks if len(b) == block_size]
    counts = {f: sum(all(day in errors.get(f, {}) for day in block) for block in blocks) for f in FUNDS}
    sums = np.asarray([[sum(errors.get(f, {}).get(day, 0.0) for day in block) for f in FUNDS] for block in blocks])
    numbers = np.asarray([[sum(day in errors.get(f, {}) for day in block) for f in FUNDS] for block in blocks])
    if not blocks or np.any(numbers.sum(axis=0) == 0):
        return {"status": "INSUFFICIENT_PAIRED_DATA", "complete_blocks_per_fund": counts, "macro_ci": None}
    rng = np.random.default_rng(config["seed"])
    estimates = []
    for _ in range(config["draws"]):
        chosen = rng.integers(0, len(blocks), len(blocks))
        n = numbers[chosen].sum(axis=0)
        if np.all(n > 0):
            estimates.append(sums[chosen].sum(axis=0) / n)
    if not estimates:
        return {"status": "INSUFFICIENT_PAIRED_DATA", "complete_blocks_per_fund": counts, "macro_ci": None}
    samples = np.asarray(estimates)
    tail = (1 - config["confidence"]) / 2
    return {
        "status": "EXPLORATORY_INTERVAL",
        "complete_blocks_per_fund": counts,
        "eligible_for_winner": min(counts.values()) >= config["minimum_disjoint_blocks_per_fund"],
        "block_count": len(blocks),
        "tail_dates_not_in_bootstrap": len(planned_dates) % block_size,
        "valid_draws": len(estimates),
        "macro_ci": np.quantile(samples.mean(axis=1), [tail, 1 - tail]).tolist(),
        "per_fund_ci": {f: np.quantile(samples[:, i], [tail, 1 - tail]).tolist() for i, f in enumerate(FUNDS)},
        "block_domain_mean_delta": dict(zip(FUNDS, (sums.sum(axis=0) / numbers.sum(axis=0)).tolist(), strict=True)),
        "delta_convention": "Brier(Chronos calibrated) - Brier(self trained calibrated)",
    }


def evaluate_window(predictions: list[dict], labels: dict[str, int], planned_dates: list[str], config: dict) -> dict:
    expected = {(f, day) for f in FUNDS for day in planned_dates}
    actual = {(p["fund_code"], p["cutoff_date"]) for p in predictions}
    if actual != expected or len(predictions) != len(expected):
        raise ValueError("prediction coverage differs from frozen denominator")
    models = ["A", "B", "ALWAYS_UP", "TRAIN_UP_FREQUENCY", "MOMENTUM_20D", "FIXED_MOMENTUM_SCORE"]
    summary, common_rows = {}, []
    for p in predictions:
        if p["key"] in labels and all(p["models"][m].get("probability") is not None for m in ("A", "B")):
            common_rows.append(p)
    for model in models:
        per_fund, coverage, common = {}, {}, {}
        for fund in FUNDS:
            rows = [p for p in predictions if p["fund_code"] == fund]
            probabilities = [p for p in rows if p["models"][model].get("probability") is not None]
            scored = [p for p in probabilities if p["key"] in labels]
            per_fund[fund] = metrics([(labels[p["key"]], p["models"][model]["probability"]) for p in scored])
            raw = [p for p in rows if p["key"] in labels and p["models"][model].get("raw_direction") is not None]
            coverage[fund] = {
                "planned": len(rows),
                "predicted": len(probabilities),
                "scored": len(scored),
                "scored_coverage": len(scored) / len(rows),
                "predicted_but_label_unavailable": sum(p["key"] not in labels for p in probabilities),
                "prediction_failures": dict(
                    Counter(
                        p["models"][model].get("reason") or "NO_PROBABILITY"
                        for p in rows
                        if p["models"][model].get("probability") is None
                    )
                ),
                "raw_direction_count": len(raw),
                "raw_direction_accuracy": sum(p["models"][model]["raw_direction"] == labels[p["key"]] for p in raw)
                / len(raw)
                if raw
                else None,
            }
            shared = [
                p for p in common_rows if p["fund_code"] == fund and p["models"][model].get("probability") is not None
            ]
            common[fund] = metrics([(labels[p["key"]], p["models"][model]["probability"]) for p in shared])
        summary[model] = {
            "per_fund": per_fund,
            "coverage": coverage,
            "macro_brier": _macro(per_fund),
            "common_per_fund": common,
            "common_macro_brier": _macro(common),
        }
    errors = {f: {} for f in FUNDS}
    for p in common_rows:
        y = labels[p["key"]]
        errors[p["fund_code"]][p["cutoff_date"]] = (p["models"]["B"]["probability"] - y) ** 2 - (
            p["models"]["A"]["probability"] - y
        ) ** 2
    bootstrap = paired_bootstrap(planned_dates, errors, config)
    conclusion = "INSUFFICIENT_EVIDENCE"
    deltas = {f: sum(v.values()) / len(v) if v else None for f, v in errors.items()}
    if common_rows and all(v is not None for v in deltas.values()):
        conclusion = "DIFFERENCE_UNCLEAR"
        equal_coverage = all(
            summary["A"]["coverage"][f]["scored"] == summary["B"]["coverage"][f]["scored"] == len(errors[f])
            for f in FUNDS
        )
        if bootstrap.get("eligible_for_winner") and equal_coverage:
            low, high = bootstrap["macro_ci"]
            cis = bootstrap["per_fund_ci"]
            if high < 0 and sum(v < 0 for v in deltas.values()) >= 2 and not any(ci[0] > 0 for ci in cis.values()):
                conclusion = "CHRONOS_BETTER_HISTORICALLY"
            elif low > 0 and sum(v > 0 for v in deltas.values()) >= 2 and not any(ci[1] < 0 for ci in cis.values()):
                conclusion = "SELF_TRAINED_BETTER_HISTORICALLY"
    return {
        "models": summary,
        "paired_count": len(common_rows),
        "paired_brier_delta_per_fund": deltas,
        "bootstrap": bootstrap,
        "conclusion": conclusion,
    }
