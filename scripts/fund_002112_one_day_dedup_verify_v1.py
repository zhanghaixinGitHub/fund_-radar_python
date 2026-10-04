"""独立核验去重实验及复用参照，不重新训练、不覆盖既有回执。"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import fund_002112_one_day_dedup_v1 as a
from scripts import fund_002112_signal_verify_v1 as numeric

c = a.c
# 独立写出删除业务含义对应的绝对列，避免训练端列号错误被原样复用而漏检。
EXPECTED_DROPS = {
    "DEDUP": {28},
    "DEDUP_NO_COUNT": {27, 28},
    "DEDUP_NO_EARNINGS_WEIGHT": {24, 25, 28},
    "DEDUP_NO_COUNT_EARNINGS_WEIGHT": {24, 25, 27, 28},
    "FULL": set(),
}


def raw(rows, keep):
    return np.asarray([r["groups"]["N"] + r["groups"]["NE"] + r["E"] for r in rows], dtype=float)[:, keep]


def verify_model(folder, attempt, rows, nav, truth):
    """逐模型检查完整训练集合、严格成熟边界、列映射和手算预处理后的概率。"""
    done = c.io.read(folder / "complete.json")
    assert c.io.sha(folder / "model.joblib") == done["model_sha256"]
    assert c.io.sha(folder / "predictions.jsonl") == done["predictions_sha256"]
    saved = joblib.load(folder / "model.joblib")
    keep = [j for j in range(35) if j not in EXPECTED_DROPS[attempt["candidate"]]]
    assert saved["kept_indices"] == keep
    assert saved["seed"] == attempt["seed"] == saved["model"].random_state
    for name, value in a.t.RECIPE.items():
        if name != "random_state":
            assert saved["model"].get_params()[name] == value
    assert saved["model"].max_features == "sqrt"
    update = attempt["update"]
    train, evaluate = ([rows[i] for i in update[k]] for k in ("training", "evaluate"))
    assert update["cutoff"] == evaluate[0]["as_of"]
    expected_training = [
        i
        for i, r in enumerate(rows)
        if r["target"] < evaluate[0]["target"] and c.moment(r["label_mature_at"]) < c.moment(update["cutoff"])
    ]
    assert expected_training == update["training"]
    assert c.io.digest(train) == attempt["training_digest"] == saved["training_digest"]
    assert c.io.digest([truth[r["target"]] for r in train]) == attempt["labels_digest"]
    assert all(
        c.moment(nav[d]["available_at"]) < c.moment(update["cutoff"]) for r in train for d in (r["base"], r["target"])
    )
    x = numeric.prepare_manually(raw(train, keep), raw(evaluate, keep), saved)
    assert list(x.shape)[1] == attempt["fit_shape"][1]
    with threadpool_limits(limits=2):
        p = saved["model"].predict_proba(x)
    aligned = np.zeros((len(evaluate), 3))
    for i, label in enumerate(saved["model"].classes_):
        aligned[:, c.CLASSES.index(label)] = p[:, i]
    stored = c.lines(folder / "predictions.jsonl")
    assert [p["target"] for p in stored] == [r["target"] for r in evaluate]
    assert [p["as_of"] for p in stored] == [r["as_of"] for r in evaluate]
    error = float(np.max(np.abs(aligned - [p["probabilities"] for p in stored])))
    assert error <= 1e-12
    assert all(p["predicted"] == c.CLASSES[int(np.argmax(p["probabilities"]))] for p in stored)
    return error, stored, x.shape[1]


def verify_score(score, ps, reference, truth):
    assert [p["target"] for p in ps] == [p["target"] for p in reference]
    good = [p["predicted"] == truth[p["target"]] for p in ps]
    base_good = [p["predicted"] == truth[p["target"]] for p in reference]
    assert score["correct"] == sum(good)
    assert abs(score["accuracy"] - sum(good) / len(ps)) <= 1e-12
    brier = np.mean(
        [
            sum(
                (v - (label == truth[p["target"]])) ** 2 for label, v in zip(c.CLASSES, p["probabilities"], strict=True)
            )
            for p in ps
        ]
    )
    assert abs(brier - score["brier"]) <= 1e-12
    for label in c.CLASSES:
        subset = [p for p in ps if truth[p["target"]] == label]
        recall = sum(p["predicted"] == label for p in subset) / len(subset) if subset else None
        assert score["recall"][label] == recall
    for quarter, result in score["quarters"].items():
        subset = [p for p in ps if str((int(p["target"][5:7]) - 1) // 3 + 1) == quarter]
        assert result["correct"] == sum(p["predicted"] == truth[p["target"]] for p in subset)
    effect = score["versus_full_events"]
    assert effect["extra_correct"] == sum(good) - sum(base_good)
    assert effect["wrong_to_right"] == sum(x and not y for x, y in zip(good, base_good, strict=True))
    assert effect["right_to_wrong"] == sum(y and not x for x, y in zip(good, base_good, strict=True))


def run(root):
    a.check_root(root)
    c.verify_files(c.io.read(root / "training-freeze.json"))
    c.verify_files(c.io.read(root / "source-manifest.json"))
    assert c.io.sha(root / "inputs.jsonl") == c.io.sha(a.PRIOR / "inputs.jsonl")
    assert c.io.sha(root / "nav.json") == c.io.sha(a.PRIOR / "nav.json")
    rows, nav = c.lines(root / "inputs.jsonl"), c.io.read(root / "nav.json")
    assert all(r["E"][11] == r["E"][13] for r in rows)
    truth = {}
    for r in rows:
        change = Decimal(nav[r["target"]]["unit_nav"]) - Decimal(nav[r["base"]]["unit_nav"])
        truth[r["target"]] = "UP" if change > 0 else "DOWN" if change < 0 else "FLAT"
    ledger = c.lines(root / "fit-ledger.jsonl")
    assert len(ledger) == 76 and len({r["id"] for r in ledger}) == 76
    selection = c.io.read(root / "selection.json")
    chosen = selection["selected_for_verification"]
    expected = Counter({(g, 0, "2025", False): 13 for g in a.CANDIDATES})
    expected.update({(chosen, 1, "2025", False): 13, (chosen, 0, "2026", False): 10, (chosen, 0, "2025", True): 1})
    assert Counter((r["candidate"], r["seed"], r["update"]["id"][:4], r["replay"]) for r in ledger) == expected
    maximum, count, reference_count = 0.0, 0, 0
    outputs, dimensions = {}, {}
    for attempt in ledger:
        assert attempt["freeze_sha256"] == c.io.sha(root / "training-freeze.json")
        error, ps, dimension = verify_model(root / "runs" / attempt["id"], attempt, rows, nav, truth)
        maximum, count = max(maximum, error), count + len(ps)
        outputs[attempt["id"]] = {p["target"]: p for p in ps}
        dimensions.setdefault(attempt["candidate"], set()).add(dimension)
    reused = [p for p in c.lines(a.PRIOR / "fit-ledger.jsonl") if p["candidate"] == "FULL" and p["seed"] == 1]
    assert len(reused) == 13
    reference_outputs = {}
    for attempt in reused:
        error, ps, _ = verify_model(a.PRIOR / "runs" / attempt["id"], attempt, rows, nav, truth)
        maximum, reference_count = max(maximum, error), reference_count + len(ps)
        reference_outputs[attempt["id"]] = {p["target"]: p for p in ps}
    for p in c.lines(root / "references/2025-B1-S1.jsonl"):
        assert p == reference_outputs[p["model_id"]][p["target"]]
    aggregate = 0
    for path in (root / "predictions").glob("*.jsonl"):
        ps = c.lines(path)
        assert len(ps) == (243 if path.name.startswith("2025") else 181)
        assert len({p["target"] for p in ps}) == len(ps)
        assert all(p == outputs[p["model_id"]][p["target"]] for p in ps)
        aggregate += len(ps)
    primary = c.io.read(root / "scores-2025-primary.json")
    assert chosen == min(
        a.CANDIDATES, key=lambda g: (-primary[g]["correct"], primary[g]["brier"], 35 - len(EXPECTED_DROPS[g]), g)
    )
    for entry_name, relevant in (
        ("validation-entry.json", [r for r in ledger if r["seed"] == 1]),
        ("diagnostic-entry.json", [r for r in ledger if r["update"]["id"].startswith("2026")]),
    ):
        entry = c.io.read(root / entry_name)
        assert entry["selection_sha256"] == c.io.sha(root / "selection.json")
        assert c.moment(entry["at"]) > c.moment(selection["at"])
        assert all(c.moment(r["at"]) >= c.moment(entry["at"]) for r in relevant)
    for name, year, seed, groups in (
        ("scores-2025-primary.json", "2025", 0, a.CANDIDATES),
        ("scores-2025-seed1.json", "2025", 1, [chosen]),
        ("scores-2026-diagnostic.json", "2026", 0, [chosen]),
    ):
        scores = c.io.read(root / name)
        preds = {g: c.lines(root / "predictions" / f"{year}-{g}-S{seed}.jsonl") for g in groups}
        ref_name = "2025-B1-S1.jsonl" if seed else f"{year}-B1.jsonl"
        preds["B1"] = c.lines(root / "references" / ref_name)
        preds["B0"] = c.lines(root / "references" / f"{year}-B0.jsonl")
        for group, ps in preds.items():
            verify_score(scores[group], ps, preds["B1"], truth)
    for effect in c.io.read(root / "group-effects-2025.json").values():
        assert (
            effect["extra_correct"] == primary[effect["candidate"]]["correct"] - primary[effect["reference"]]["correct"]
        )
    result = {
        "passed": True,
        "new_models_verified": len(ledger),
        "reused_models_verified": len(reused),
        "new_probability_rows": count,
        "reused_probability_rows": reference_count,
        "aggregate_prediction_rows": aggregate,
        "max_probability_error": maximum,
        "probability_atol": 1e-12,
        "input_and_nav_identical": True,
        "one_day_maturity_boundaries": True,
        "selected_before_seed1_and_2026": True,
        "new_fits_during_verification": 0,
        "dimensions": {k: sorted(v) for k, v in dimensions.items()},
        "freeze_sha256": c.io.sha(root / "training-freeze.json"),
    }
    if (root / "independent-verification.json").exists():
        previous = c.io.read(root / "independent-verification.json")
        assert previous["freeze_sha256"] == result["freeze_sha256"] and previous["passed"]
    else:
        c.io.save(root / "independent-verification.json", {"at": c.io.now(), **result})
    print(c.io.canonical(result))


if __name__ == "__main__":
    with c.io.writer_lock(a.ROOT), c.io.offline_guard():
        run(a.ROOT)
