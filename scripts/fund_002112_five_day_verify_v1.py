"""五日实验独立验收：原输入不变、五日期间、成熟与隔离、手算预处理及概率。"""

from __future__ import annotations

from decimal import Decimal

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import fund_002112_five_day_v1 as f
from scripts import fund_002112_signal_verify_v1 as numeric

c = f.c


def run(root):
    c.verify_files(c.io.read(root / "training-freeze.json"))
    c.verify_files(c.io.read(root / "source-manifest.json"))
    rows, nav = c.lines(root / "inputs.jsonl"), c.io.read(root / "nav.json")
    sessions = c.io.read(root / "sessions.json")
    original = {r["target"]: r for r in c.lines(f.PRIOR / "features-r4/inputs.jsonl")}
    truths, rowmap = {}, {r["target"]: r for r in rows}
    for r in rows:
        old = original[r["one_day_target"]]
        assert c.io.digest(old) == r["source_row_sha256"]
        assert all(r[k] == old[k] for k in ("base", "as_of", "groups", "E", "ED", "trigger", "trigger_decay"))
        assert sessions.index(r["target"]) - sessions.index(r["base"]) == 5
        assert c.moment(r["label_mature_at"]) == max(c.moment(nav[d]["available_at"]) for d in (r["base"], r["target"]))
        delta = Decimal(nav[r["target"]]["unit_nav"]) - Decimal(nav[r["base"]]["unit_nav"])
        truths[r["target"]] = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
    ledger = c.lines(root / "fit-ledger.jsonl")
    assert 69 <= len(ledger) <= 80
    by_model, dimensions = {}, {}
    maximum, count = 0.0, 0
    for attempt in ledger:
        folder = root / "runs" / attempt["id"]
        done = c.io.read(folder / "complete.json")
        assert c.io.sha(folder / "model.joblib") == done["model_sha256"]
        assert c.io.sha(folder / "predictions.jsonl") == done["predictions_sha256"]
        model = joblib.load(folder / "model.joblib")
        update = attempt["update"]
        training = [rows[i] for i in update["training"]]
        evaluate = [rows[i] for i in update["evaluate"]]
        assert c.io.digest(training) == attempt["training_digest"]
        assert c.io.digest([truths[r["target"]] for r in training]) == attempt["label_digest"]
        assert all(c.moment(r["label_mature_at"]) < c.moment(update["cutoff"]) for r in training)
        assert all(r["target"] < evaluate[0]["base"] for r in training)
        # 手算中位数、缺失指示、标准化，与训练路径分开复算。
        a = numeric.raw_matrix(training, attempt["group"])
        b = numeric.raw_matrix(evaluate, attempt["group"])
        matrix = numeric.prepare_manually(a, b, model)
        dimensions.setdefault(attempt["group"], set()).add(matrix.shape[1])
        with threadpool_limits(limits=2):
            raw = model["model"].predict_proba(matrix)
        aligned = np.zeros((len(evaluate), 3))
        for j, label in enumerate(model["model"].classes_):
            aligned[:, c.CLASSES.index(label)] = raw[:, j]
        stored = c.lines(folder / "predictions.jsonl")
        error = float(np.max(np.abs(aligned - [r["probabilities"] for r in stored])))
        assert error <= 1e-12
        assert [p["target"] for p in stored] == [r["target"] for r in evaluate]
        maximum = max(maximum, error)
        count += len(stored)
        by_model[attempt["id"]] = {r["target"]: p.tolist() for r, p in zip(evaluate, aligned, strict=True)}
    fallback, blend_count = 0, 0
    for year, expected_count in (("2025", 243), ("2026", 181)):
        predictions = {g: c.lines(root / "predictions" / f"{year}-{g}.jsonl") for g in f.GROUPS}
        scores = c.io.read(root / f"scores-{year}.json")
        expected_dates = [r["target"] for r in rows if r["target"].startswith(year)]
        assert len(expected_dates) == expected_count
        for group, ps in predictions.items():
            assert [p["target"] for p in ps] == expected_dates
            assert scores[group]["correct"] == sum(p["predicted"] == truths[p["target"]] for p in ps)
            brier = np.mean(
                [
                    sum(
                        (v - (label == truths[p["target"]])) ** 2
                        for label, v in zip(c.CLASSES, p["probabilities"], strict=True)
                    )
                    for p in ps
                ]
            )
            assert abs(brier - scores[group]["brier"]) <= 1e-12
            assert all(p["predicted"] == c.CLASSES[int(np.argmax(p["probabilities"]))] for p in ps)
            for phase in range(5):
                subset = [p for p in ps if rowmap[p["target"]]["base_session_index"] % 5 == phase]
                assert all(
                    rowmap[a["target"]]["target"] <= rowmap[b["target"]]["base"]
                    for a, b in zip(subset, subset[1:], strict=False)
                )
                assert scores[group]["all_nonoverlap_phases"][str(phase)]["correct"] == sum(
                    p["predicted"] == truths[p["target"]] for p in subset
                )
        for base, one, two, three in zip(*(predictions[g] for g in f.GROUPS), strict=True):
            for p in (base, one):
                np.testing.assert_allclose(p["probabilities"], by_model[p["model_id"]][p["target"]], atol=1e-12, rtol=0)
            for p, trigger_name in ((two, "trigger"), (three, "trigger_decay")):
                assert p["trigger"] == rowmap[p["target"]][trigger_name]
                if not p["trigger"]:
                    assert p["probabilities"] == base["probabilities"] and p["predicted"] == base["predicted"]
                    fallback += 1
                else:
                    expected = 0.75 * np.asarray(base["probabilities"]) + 0.25 * np.asarray(
                        by_model[p["event_model"]][p["target"]]
                    )
                    np.testing.assert_allclose(p["probabilities"], expected, atol=1e-12, rtol=0)
                    blend_count += 1
    selection = c.io.read(root / "selection.json")
    diagnostic = c.io.read(root / "diagnostic-entry.json")
    assert diagnostic["selection_sha256"] == c.io.sha(root / "selection.json")
    assert c.moment(selection["at"]) < c.moment(diagnostic["at"])
    assert all(c.moment(a["at"]) >= c.moment(diagnostic["at"]) for a in ledger if a["update"]["id"].startswith("2026"))
    result = {
        "at": c.io.now(),
        "passed": True,
        "models_reloaded": len(ledger),
        "probability_rows": count,
        "max_probability_error": maximum,
        "atol": 1e-12,
        "exact_fallback_rows": fallback,
        "blend_rows": blend_count,
        "input_rows_verified": len(rows),
        "dimensions": {k: sorted(v) for k, v in dimensions.items()},
        "five_sessions_verified": True,
        "mature_labels_strictly_before_update": True,
        "training_eval_intervals_disjoint": True,
        "all_five_nonoverlap_phases_verified": True,
        "selection_frozen_before_2026": True,
        "new_fits_in_verification": 0,
    }
    c.io.save(root / "independent-verification.json", result)
    print(c.io.canonical(result))


if __name__ == "__main__":
    with c.io.writer_lock(f.ROOT), c.io.offline_guard():
        run(f.ROOT)
