"""一日删减的独立验收：核对训练行、被删除字段、标签、手算预处理和模型概率。"""

from __future__ import annotations

from decimal import Decimal

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import fund_002112_one_day_ablation_v1 as a
from scripts import fund_002112_signal_verify_v1 as numeric

c = a.c


def run(root):
    c.verify_files(c.io.read(root / "training-freeze.json"))
    c.verify_files(c.io.read(root / "source-manifest.json"))
    rows, nav = c.lines(root / "inputs.jsonl"), c.io.read(root / "nav.json")
    assert c.io.sha(root / "inputs.jsonl") == c.io.sha(a.PRIOR / "features-r4/inputs.jsonl")
    truth = {}
    for r in rows:
        change = Decimal(nav[r["target"]]["unit_nav"]) - Decimal(nav[r["base"]]["unit_nav"])
        truth[r["target"]] = "UP" if change > 0 else "DOWN" if change < 0 else "FLAT"
    ledger = c.lines(root / "fit-ledger.jsonl")
    assert len(ledger) == 76 and len({r["id"] for r in ledger}) == 76
    maximum, count = 0.0, 0
    model_outputs, dimensions = {}, {}
    for attempt in ledger:
        folder = root / "runs" / attempt["id"]
        done = c.io.read(folder / "complete.json")
        assert c.io.sha(folder / "model.joblib") == done["model_sha256"]
        assert c.io.sha(folder / "predictions.jsonl") == done["predictions_sha256"]
        model = joblib.load(folder / "model.joblib")
        candidate = attempt["candidate"]
        expected = [j for j in range(35) if j not in a.DROPS[candidate]]
        assert model["kept_indices"] == expected
        assert model["seed"] == attempt["seed"] == model["model"].random_state
        update = attempt["update"]
        train, evaluate = ([rows[i] for i in update[k]] for k in ("training", "evaluate"))
        assert c.io.digest(train) == attempt["training_digest"]
        assert c.io.digest([truth[r["target"]] for r in train]) == attempt["labels_digest"]
        assert all(c.moment(r["label_mature_at"]) < c.moment(update["cutoff"]) for r in train)
        assert all(
            c.moment(nav[d]["available_at"]) < c.moment(update["cutoff"])
            for r in train
            for d in (r["base"], r["target"])
        )
        # 独立拼接、选择列，避免复用被核验路径的matrix/transform。
        def raw(rs, indices):
            return np.asarray([r["groups"]["N"] + r["groups"]["NE"] + r["E"] for r in rs], dtype=float)[:, indices]

        x = numeric.prepare_manually(raw(train, expected), raw(evaluate, expected), model)
        dimensions.setdefault(candidate, set()).add(x.shape[1])
        with threadpool_limits(limits=2):
            p = model["model"].predict_proba(x)
        aligned = np.zeros((len(evaluate), 3))
        for i, label in enumerate(model["model"].classes_):
            aligned[:, c.CLASSES.index(label)] = p[:, i]
        stored = c.lines(folder / "predictions.jsonl")
        assert [p["target"] for p in stored] == [r["target"] for r in evaluate]
        error = float(np.max(np.abs(aligned - [p["probabilities"] for p in stored])))
        assert error <= 1e-12
        assert all(p["predicted"] == c.CLASSES[int(np.argmax(p["probabilities"]))] for p in stored)
        maximum = max(maximum, error)
        count += len(stored)
        model_outputs[attempt["id"]] = {p["target"]: p for p in stored}
    aggregate_rows = 0
    for path in (root / "predictions").glob("*.jsonl"):
        ps = c.lines(path)
        assert len(ps) == (243 if path.name.startswith("2025") else 181)
        assert len({p["target"] for p in ps}) == len(ps)
        assert all(p == model_outputs[p["model_id"]][p["target"]] for p in ps)
        aggregate_rows += len(ps)
    selection = c.io.read(root / "selection.json")
    primary = c.io.read(root / "scores-2025-primary.json")
    assert selection["selected_for_verification"] == a.choose(primary)
    for entry_name in ("validation-entry.json", "diagnostic-entry.json"):
        entry = c.io.read(root / entry_name)
        assert entry["selection_sha256"] == c.io.sha(root / "selection.json")
        assert c.moment(entry["at"]) > c.moment(selection["at"])
    validation_at = c.moment(c.io.read(root / "validation-entry.json")["at"])
    diagnostic_at = c.moment(c.io.read(root / "diagnostic-entry.json")["at"])
    assert all(c.moment(r["at"]) >= validation_at for r in ledger if r["seed"] == 1)
    assert all(c.moment(r["at"]) >= diagnostic_at for r in ledger if r["update"]["id"].startswith("2026"))
    chosen = selection["selected_for_verification"]
    for name, year, seed, groups in (
        ("scores-2025-primary.json", "2025", 0, list(a.CANDIDATES)),
        ("scores-2025-seed1.json", "2025", 1, [chosen]),
        ("scores-2026-diagnostic.json", "2026", 0, [chosen]),
    ):
        scores = c.io.read(root / name)
        preds = {g: c.lines(root / "predictions" / f"{year}-{g}-S{seed}.jsonl") for g in groups}
        preds["B1"] = (
            c.lines(root / "predictions/2025-FULL-S1.jsonl")
            if seed
            else c.lines(root / "references" / f"{year}-B1.jsonl")
        )
        preds["B0"] = c.lines(root / "references" / f"{year}-B0.jsonl")
        for g, ps in preds.items():
            assert scores[g]["correct"] == sum(p["predicted"] == truth[p["target"]] for p in ps)
            brier = float(
                np.mean(
                    [
                        sum(
                            (v - (label == truth[p["target"]])) ** 2
                            for label, v in zip(c.CLASSES, p["probabilities"], strict=True)
                        )
                        for p in ps
                    ]
                )
            )
            assert abs(brier - scores[g]["brier"]) <= 1e-12
            difference = scores[g]["correct"] - scores["B1"]["correct"]
            assert scores[g]["versus_full_events"]["extra_correct"] == difference
    result = {
        "passed": True,
        "models": len(ledger),
        "probability_rows": count,
        "aggregate_prediction_rows": aggregate_rows,
        "max_probability_error": maximum,
        "probability_atol": 1e-12,
        "input_identical_to_original": True,
        "dimensions": {k: sorted(v) for k, v in dimensions.items()},
        "one_day_maturity_boundaries": True,
        "selected_before_seed1_and_2026": True,
        "new_fits_during_verification": 0,
        "freeze_sha256": c.io.sha(root / "training-freeze.json"),
    }
    receipt = root / "independent-verification.json"
    if receipt.exists():
        previous = c.io.read(receipt)
        assert previous["freeze_sha256"] == result["freeze_sha256"] and previous["passed"]
    else:
        c.io.save(receipt, {"at": c.io.now(), **result})
    print(c.io.canonical(result))


if __name__ == "__main__":
    with c.io.writer_lock(a.ROOT), c.io.offline_guard():
        run(a.ROOT)
