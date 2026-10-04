"""交付独立复算：手算中位数/缺失指示/标准化，再回读模型核对所有概率。"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import fund_002112_signal_common_v1 as c


def raw_matrix(rows, group):
    values = []
    for r in rows:
        x = list(r["groups"]["N"]) + list(r["groups"]["NE"])
        if group != "B0":
            x.extend(r["E"] if group == "B1" else r["ED"])
        values.append(x)
    return np.array(values, dtype=float)


def prepare_manually(train, evaluate, saved):
    """不调用生产的transform或sklearn的fit；全空列0只发生在预处理层。"""
    medians = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in train.T])
    indicators = np.flatnonzero(np.isnan(train).any(axis=0))
    state = saved["preprocessing"]
    np.testing.assert_array_equal(medians, state["imputer"].statistics_)
    np.testing.assert_array_equal(indicators, state["imputer"].indicator_.features_)
    filled = np.where(np.isnan(train), medians, train)
    filled = np.column_stack([filled, np.isnan(train[:, indicators]).astype(float)])
    np.testing.assert_allclose(filled.mean(axis=0), state["scaler"].mean_, atol=1e-12, rtol=0)
    np.testing.assert_allclose(filled.var(axis=0), state["scaler"].var_, atol=1e-12, rtol=0)
    # 模型预处理的参数已独立核对；使用存储尺度以避免浮点计算顺序造成树阈值边界偏移。
    x = np.where(np.isnan(evaluate), medians, evaluate)
    x = np.column_stack([x, np.isnan(evaluate[:, indicators]).astype(float)])
    return (x - state["scaler"].mean_) / state["scaler"].scale_


def run(root):
    root = Path(root)
    c.verify_files(c.io.read(root / "training-freeze.json"))
    folder = root / c.io.read(root / "active-inputs.json")["directory"]
    rows = c.lines(folder / "inputs.jsonl")
    ledger = c.lines(root / "fit-ledger.jsonl")
    assert len(ledger) <= 80
    probability_rows, maximum = 0, 0.0
    dimensions, raw_dimensions = {}, {}
    by_model = {}
    for r in ledger:
        path = root / "runs" / r["id"]
        done = c.io.read(path / "complete.json")
        assert c.io.sha(path / "model.joblib") == done["model_sha256"]
        assert c.io.sha(path / "predictions.jsonl") == done["predictions_sha256"]
        model = joblib.load(path / "model.joblib")
        training = [rows[i] for i in r["update"]["training"]]
        evaluate = [rows[i] for i in r["update"]["evaluate"]]
        assert c.io.digest(training) == r["training_digest"]
        assert all(c.moment(x["label_mature_at"]) < c.moment(r["update"]["cutoff"]) for x in training)
        a, b = raw_matrix(training, r["group"]), raw_matrix(evaluate, r["group"])
        raw_dimensions.setdefault(r["group"], set()).add(a.shape[1])
        transformed = prepare_manually(a, b, model)
        dimensions.setdefault(r["group"], set()).add(transformed.shape[1])
        with threadpool_limits(limits=2):
            probs = model["model"].predict_proba(transformed)
        aligned = np.zeros((len(evaluate), 3))
        for j, label in enumerate(model["model"].classes_):
            aligned[:, c.CLASSES.index(label)] = probs[:, j]
        stored = c.lines(path / "predictions.jsonl")
        error = float(np.max(np.abs(aligned - [p["probabilities"] for p in stored])))
        assert error <= 1e-12
        maximum = max(maximum, error)
        probability_rows += len(stored)
        by_model[r["id"]] = {p["target"]: aligned[j].tolist() for j, p in enumerate(stored)}
    exact_fallback, mixtures = 0, 0
    for year in ("2025", "2026"):
        predictions = {g: c.lines(root / "predictions" / f"{year}-{g}.jsonl") for g in ("B0", "B1", "B2", "B3")}
        assert len(predictions["B0"]) == (243 if year == "2025" else 181)
        for base, one, two, three in zip(*(predictions[g] for g in ("B0", "B1", "B2", "B3")), strict=True):
            assert base["target"] == one["target"] == two["target"] == three["target"]
            for enhanced in (two, three):
                if not enhanced["trigger"]:
                    assert (
                        enhanced["probabilities"] == base["probabilities"]
                        and enhanced["predicted"] == base["predicted"]
                    )
                    exact_fallback += 1
                else:
                    expected = 0.75 * np.array(base["probabilities"]) + 0.25 * np.array(
                        by_model[enhanced["event_model"]][base["target"]]
                    )
                    np.testing.assert_allclose(expected, enhanced["probabilities"], atol=1e-12, rtol=0)
                    mixtures += 1
    # 2026入口须在2025选型之后，且入口记录的选择摘要仍原样。
    selection = c.io.read(root / "selection.json")
    entry = c.io.read(root / "diagnostic-entry.json")
    assert c.moment(selection["at"]) < c.moment(entry["at"])
    assert entry["selection_sha256"] == c.io.sha(root / "selection.json")
    assert all(c.moment(r["at"]) >= c.moment(entry["at"]) for r in ledger if r["update"]["id"].startswith("2026"))
    result = {
        "at": c.io.now(),
        "passed": True,
        "models": len(ledger),
        "probabilities": probability_rows,
        "max_probability_error": maximum,
        "exact_fallback_rows": exact_fallback,
        "mixture_rows": mixtures,
        "business_dimensions": {k: sorted(v) for k, v in raw_dimensions.items()},
        "preprocessed_dimensions": {k: sorted(v) for k, v in dimensions.items()},
        "selection_before_diagnostic": True,
        "new_fits_in_this_verification": 0,
        "preprocessing_variance_comparison_atol": 1e-12,
        "probability_comparison_atol": 1e-12,
    }
    c.io.save(root / "independent-verification-strict.json", result)
    print(c.io.canonical(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root)
