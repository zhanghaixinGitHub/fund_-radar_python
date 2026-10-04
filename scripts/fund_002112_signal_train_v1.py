"""固定B0/B1/B2/B3实验：2025选择先落盘，2026只诊断，真实拟合硬上限80。"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts import fund_002112_recent_bodies_train_v1 as reference
from scripts import fund_002112_signal_common_v1 as c

RECIPE = {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 10, "random_state": 0, "n_jobs": 2}
GROUPS = ("B0", "B1", "B3_MODEL")


def feature_root(root):
    pointer = c.io.read(root / "active-inputs.json")
    folder = (root / pointer["directory"]).resolve()
    if not folder.is_relative_to(root.resolve()) or c.io.sha(folder / "inputs.jsonl") != pointer["sha256"]:
        raise ValueError("INPUT_POINTER_INVALID")
    return folder


def matrix(rows, group):
    return np.array(
        [
            r["groups"]["N"] + r["groups"]["NE"] + (r["E"] if group == "B1" else r["ED"] if group == "B3_MODEL" else [])
            for r in rows
        ],
        dtype=float,
    )


def transform(rows, group, state=None):
    data = matrix(rows, group)
    if state is None:
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        filled = imputer.fit_transform(data)
        state = {"imputer": imputer, "scaler": StandardScaler().fit(filled)}
    return state["scaler"].transform(state["imputer"].transform(data)), state


def calendar(rows, year):
    indices = [i for i, r in enumerate(rows) if r["target"].startswith(year)]
    updates = []
    for start in range(0, len(indices), 20):
        evaluate = indices[start : start + 20]
        cutoff = rows[evaluate[0]]["as_of"]
        train = [
            i
            for i, r in enumerate(rows)
            if r["target"] < rows[evaluate[0]]["target"] and c.moment(r["label_mature_at"]) < c.moment(cutoff)
        ]
        updates.append({"id": year + f"_U{start:03d}", "training": train, "evaluate": evaluate, "cutoff": cutoff})
    return updates


def labels_nav():
    nav = c.bundle()["nav"]
    for day, r in c.io.read(c.OLD / "reserved-labels.json")["values"].items():
        nav[day] = {"unit_nav": r["unit_nav"], "available_at": c.nav_inputs.available_at(r["ann_date"])}
    return nav


def verify_freeze(root):
    c.verify_files(c.io.read(root / "training-freeze.json"))


def freeze(root, rows):
    path = root / "training-freeze.json"
    if path.exists():
        verify_freeze(root)
        return
    features = feature_root(root)
    review = c.io.read(features / "source-review.json")
    tests = c.io.read(root / "test-results.json")
    if not review["passed"] or review["actual_change_events_reviewed"] < 50 or not tests["passed"]:
        raise ValueError("REVIEW_OR_TEST_GATE_NOT_MET")
    dates = {year: calendar(rows, year) for year in ("2025", "2026")}
    old_dates = c.io.read(c.OLD / "training-calendar.json")
    assert dates == old_dates, "OUTER_OR_TRAINING_CALENDAR_CHANGED"
    c.io.save(root / "training-calendar.json", dates)
    protocol = {
        "at": c.io.now(),
        "recipe": RECIPE,
        "cadence": 20,
        "maximum_fits": 80,
        "planned_fits": 69,
        "replay_fits_max": 2,
        "probability_atol": 1e-12,
        "development": "2025",
        "diagnostic_only": "2026-01-01..2026-09-30; all labels previously exposed",
        "selection": {
            "minimum_extra_correct": 5,
            "quarters_not_worse": 3,
            "down_recall_max_drop": 0.05,
            "brier_not_worse": True,
            "full_243_dates": True,
            "tie_break": ["correct_desc", "brier_asc", "B1", "B2", "B3"],
        },
        "business_dimensions": {"B0": 15, "B1": 35, "B3_MODEL": 35},
        "adoption": False,
        "future_observation_auto_start": False,
    }
    c.io.save(root / "training-protocol.json", protocol)
    files = list(features.glob("*.json")) + list(features.glob("*.jsonl"))
    files += [
        root / n
        for n in (
            "active-inputs.json",
            "nav-summary.json",
            "nav-sources.json",
            "nav-source-manifest.json",
            "nav-focus-71.jsonl",
            "collection-catalog.json",
            "collection-completion.json",
            "supplement-documents.jsonl",
            "training-calendar.json",
            "training-protocol.json",
            "test-results.json",
        )
    ]
    files += list(c.io.PY.glob("scripts/fund_002112_signal_*_v1.py"))
    files += [
        c.io.PY / "scripts/test_fund_002112_signal_v1.py",
        Path(reference.__file__),
        Path(c.nav_inputs.__file__),
        Path(c.io.__file__),
        Path(c.bodies.__file__),
        c.bodies.BUNDLE,
        c.OLD / "reserved-labels.json",
    ]
    for manifest in (root / "nav-source-manifest.json", features / "event-source-manifest.json"):
        saved = c.io.read(manifest)
        c.verify_files(saved)
        files += [Path(p) for p in saved["files"]]
    for p in list(files):
        if p.suffix == ".py":
            dest = root / "code-snapshot" / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                with dest.open("xb") as f:
                    f.write(p.read_bytes())
            assert c.io.sha(dest) == c.io.sha(p)
            files.append(dest)
    c.io.save(
        path,
        {
            "at": c.io.now(),
            "python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "joblib")},
            "files": {str(p.resolve()): c.io.sha(p) for p in set(files)},
        },
    )
    c.stage(root, "冻结", "COMPLETE", ["training-freeze.json", "training-protocol.json"], [], "固定39次开发拟合")


def aligned_probabilities(saved, rows):
    x, _ = transform(rows, saved["group"], saved["preprocessing"])
    with threadpool_limits(limits=2):
        raw = saved["model"].predict_proba(x)
    aligned = np.zeros((len(rows), 3))
    for j, label in enumerate(saved["model"].classes_):
        aligned[:, c.CLASSES.index(label)] = raw[:, j]
    return aligned


def fit_one(root, group, update, rows, nav, replay=False):
    identifier = group + "__" + update["id"] + ("__replay" if replay else "")
    folder = root / "runs" / identifier
    if (folder / "complete.json").exists():
        done = c.io.read(folder / "complete.json")
        assert c.io.sha(folder / "model.joblib") == done["model_sha256"]
        return done
    train = [rows[i] for i in update["training"]]
    evaluate = [rows[i] for i in update["evaluate"]]
    assert all(c.moment(r["label_mature_at"]) < c.moment(update["cutoff"]) for r in train)
    assert all(
        c.moment(nav[day]["available_at"]) < c.moment(update["cutoff"])
        for r in train
        for day in (r["base"], r["target"])
    )
    x, state = transform(train, group)
    y = reference.labels(train, nav)
    record = {
        "group": group,
        "update": update,
        "replay": replay,
        "fit_shape": list(x.shape),
        "training_digest": c.io.digest(train),
        "label_digest": c.io.digest(y),
        "freeze_sha256": c.io.sha(root / "training-freeze.json"),
        "retry_of": None,
    }
    c.reserve(root, "fit", identifier, record)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        model = RandomForestClassifier(**RECIPE)
        with threadpool_limits(limits=2):
            model.fit(x, y)
        saved = {"group": group, "model": model, "preprocessing": state, "training_digest": record["training_digest"]}
        probabilities = aligned_probabilities(saved, evaluate)
        predictions = [
            {
                "target": r["target"],
                "as_of": r["as_of"],
                "generated_at": c.io.now(),
                "model_id": identifier,
                "probabilities": p.tolist(),
                "predicted": c.CLASSES[int(np.argmax(p))],
            }
            for r, p in zip(evaluate, probabilities, strict=True)
        ]
        joblib.dump(saved, folder / "model.joblib")
        c.io.save_lines(folder / "predictions.jsonl", predictions)
        done = {
            "at": c.io.now(),
            "id": identifier,
            "model_sha256": c.io.sha(folder / "model.joblib"),
            "predictions_sha256": c.io.sha(folder / "predictions.jsonl"),
            "state": "COMPLETE",
        }
        c.io.save(folder / "complete.json", done)
        print(c.io.canonical({"fit": identifier, "count": len(c.lines(root / "fit-ledger.jsonl"))}), flush=True)
        return done
    except Exception as exc:
        c.io.save(folder / "failed.json", {"at": c.io.now(), "type": type(exc).__name__, "message": str(exc)})
        raise


def blend(base, enhanced, trigger):
    if not trigger:
        return list(base)
    return (0.75 * np.asarray(base) + 0.25 * np.asarray(enhanced)).tolist()


def predictions(root, rows, year):
    by_target = {r["target"]: r for r in rows}
    updates = calendar(rows, year)
    output = {
        group: [p for u in updates for p in c.lines(root / "runs" / (group + "__" + u["id"]) / "predictions.jsonl")]
        for group in GROUPS
    }
    for group, model_group, trigger_key in (("B2", "B1", "trigger"), ("B3", "B3_MODEL", "trigger_decay")):
        output[group] = []
        for b, enhanced in zip(output["B0"], output[model_group], strict=True):
            assert b["target"] == enhanced["target"]
            trigger = by_target[b["target"]][trigger_key]
            p = blend(b["probabilities"], enhanced["probabilities"], trigger)
            output[group].append(
                {
                    **b,
                    "probabilities": p,
                    "predicted": c.CLASSES[int(np.argmax(p))],
                    "trigger": trigger,
                    "baseline_fallback": not trigger,
                    "event_model": enhanced["model_id"],
                    "baseline_model": b["model_id"],
                }
            )
    return {k: output[k] for k in ("B0", "B1", "B2", "B3")}


def metrics(preds, truth):
    correct = sum(p["predicted"] == truth[p["target"]] for p in preds)
    brier = np.mean(
        [
            sum((v - float(c.CLASSES[i] == truth[p["target"]])) ** 2 for i, v in enumerate(p["probabilities"]))
            for p in preds
        ]
    )
    return {
        "days": len(preds),
        "correct": correct,
        "accuracy": correct / len(preds),
        "brier": float(brier),
        "recall": {
            label: (
                sum(p["predicted"] == label and truth[p["target"]] == label for p in preds)
                / sum(truth[p["target"]] == label for p in preds)
            )
            if any(truth[p["target"]] == label for p in preds)
            else None
            for label in c.CLASSES
        },
        "class_counts": {label: sum(truth[p["target"]] == label for p in preds) for label in c.CLASSES},
    }


def score(preds, rows, nav):
    truth = dict(zip([r["target"] for r in rows], reference.labels(rows, nav), strict=True))
    rowmap = {r["target"]: r for r in rows}
    base = {p["target"]: p for p in preds["B0"]}
    out = {}
    for group, ps in preds.items():

        def quarter(p):
            return (int(p["target"][5:7]) - 1) // 3 + 1

        def good(p):
            return p["predicted"] == truth[p["target"]]

        out[group] = {
            **metrics(ps, truth),
            "quarters": {
                str(q): metrics([p for p in ps if quarter(p) == q], truth)
                for q in range(1, 5)
                if any(quarter(p) == q for p in ps)
            },
            "wrong_to_right": sum(good(p) and not good(base[p["target"]]) for p in ps),
            "right_to_wrong": sum(not good(p) and good(base[p["target"]]) for p in ps),
            "trigger_days": sum(p.get("trigger", False) for p in ps),
            "groups": {},
        }
        predicates = {
            "nav_lag_0_1": lambda r: r["groups"]["N"][7] <= 1,
            "nav_lag_over_1": lambda r: r["groups"]["N"][7] > 1,
            "important_event": lambda r: r["trigger"],
            "no_important_event": lambda r: not r["trigger"],
        }
        for name, predicate in predicates.items():
            subset = [p for p in ps if predicate(rowmap[p["target"]])]
            out[group]["groups"][name] = metrics(subset, truth) if subset else None
    targets = list(base)
    out["always_up"] = {
        "days": len(targets),
        "correct": sum(truth[d] == "UP" for d in targets),
        "accuracy": sum(truth[d] == "UP" for d in targets) / len(targets),
    }
    return out


def select(scores):
    base = scores["B0"]
    results = {}
    for group in ("B1", "B2", "B3"):
        s = scores[group]
        gates = {
            "full_coverage": s["days"] == base["days"] == 243,
            "five_more_correct": s["correct"] - base["correct"] >= 5,
            "three_quarters": sum(s["quarters"][q]["correct"] >= b["correct"] for q, b in base["quarters"].items())
            >= 3,
            "down_recall": s["recall"]["DOWN"] >= base["recall"]["DOWN"] - 0.05,
            "brier": s["brier"] <= base["brier"],
        }
        results[group] = {"gates": gates, "passed": all(gates.values())}
    passed = [g for g, r in results.items() if r["passed"]]
    selected = min(passed, key=lambda g: (-scores[g]["correct"], scores[g]["brier"], g)) if passed else "B0"
    return {
        "at": c.io.now(),
        "selected": selected,
        "candidates": results,
        "future_observation_eligible": bool(passed),
        "selection_data": "2025_ONLY",
        "adopted": False,
    }


def verify_models(root, rows, nav):
    count, max_error = 0, 0.0
    for record in c.lines(root / "fit-ledger.jsonl"):
        folder = root / "runs" / record["id"]
        done = c.io.read(folder / "complete.json")
        assert c.io.sha(folder / "model.joblib") == done["model_sha256"]
        assert c.io.sha(folder / "predictions.jsonl") == done["predictions_sha256"]
        saved = joblib.load(folder / "model.joblib")
        update = record["update"]
        train = [rows[i] for i in update["training"]]
        assert c.io.digest(train) == record["training_digest"]
        assert c.io.digest(reference.labels(train, nav)) == record["label_digest"]
        assert all(c.moment(r["label_mature_at"]) < c.moment(update["cutoff"]) for r in train)
        _, state = transform(train, record["group"])
        for k in ("statistics_",):
            np.testing.assert_array_equal(getattr(state["imputer"], k), getattr(saved["preprocessing"]["imputer"], k))
        np.testing.assert_array_equal(
            state["imputer"].indicator_.features_, saved["preprocessing"]["imputer"].indicator_.features_
        )
        for k in ("mean_", "var_", "scale_"):
            np.testing.assert_array_equal(getattr(state["scaler"], k), getattr(saved["preprocessing"]["scaler"], k))
        probs = aligned_probabilities(saved, [rows[i] for i in update["evaluate"]])
        stored = c.lines(folder / "predictions.jsonl")
        error = float(np.max(np.abs(probs - np.array([p["probabilities"] for p in stored]))))
        max_error = max(error, max_error)
        assert error <= 1e-12
        assert [p["predicted"] for p in stored] == [c.CLASSES[int(np.argmax(p))] for p in probs]
        count += len(stored)
    return {
        "models": len(c.lines(root / "fit-ledger.jsonl")),
        "probabilities_recomputed": count,
        "max_absolute_error": max_error,
        "training_only_preprocessing": True,
        "maturity_boundary": True,
    }


def run(root):
    root = Path(root).resolve()
    rows = c.lines(feature_root(root) / "inputs.jsonl")
    freeze(root, rows)
    nav = labels_nav()
    for year in ("2025", "2026"):
        verify_freeze(root)
        if year == "2026":
            assert (root / "selection.json").exists()
            c.io.save(
                root / "diagnostic-entry.json",
                {
                    "at": c.io.now(),
                    "selection_sha256": c.io.sha(root / "selection.json"),
                    "previously_exposed_dates": 181,
                    "not_an_unseen_test": True,
                },
            )
        for update in calendar(rows, year):
            for group in GROUPS:
                fit_one(root, group, update, rows, nav)
        result = predictions(root, rows, year)
        for group, records in result.items():
            c.io.save_lines(root / "predictions" / (year + "-" + group + ".jsonl"), records)
        scores = score(result, rows, nav)
        c.io.save(root / ("scores-" + year + ".json"), scores)
        if year == "2025":
            c.io.save(root / "selection.json", select(scores))
        c.stage(
            root,
            "开发比较" if year == "2025" else "历史诊断",
            "COMPLETE",
            ["scores-" + year + ".json"],
            [],
            "继续固定验收",
        )
    selected = c.io.read(root / "selection.json")["selected"]
    replay_groups = ["B0"] + (["B1"] if selected in ("B1", "B2") else ["B3_MODEL"] if selected == "B3" else [])
    for group in replay_groups:
        update = calendar(rows, "2025")[0]
        fit_one(root, group, update, rows, nav, True)
        identifier = group + "__" + update["id"]
        old = c.lines(root / "runs" / identifier / "predictions.jsonl")
        replay = c.lines(root / "runs" / (identifier + "__replay") / "predictions.jsonl")
        np.testing.assert_allclose(
            [p["probabilities"] for p in old], [p["probabilities"] for p in replay], atol=1e-12, rtol=0
        )
    verified = verify_models(root, rows, nav)
    old_max = 0.0
    for year in ("2025", "2026"):
        before = [
            p
            for u in calendar(rows, year)
            for p in c.lines(c.OLD / "runs" / ("BASE__" + u["id"]) / "predictions.jsonl")
        ]
        after = c.lines(root / "predictions" / (year + "-B0.jsonl"))
        assert [p["target"] for p in before] == [p["target"] for p in after]
        delta = float(
            np.max(
                np.abs(np.array([p["probabilities"] for p in before]) - np.array([p["probabilities"] for p in after]))
            )
        )
        old_max = max(old_max, delta)
        assert delta <= 1e-12, "EQUIVALENT_BASELINE_NOT_REPRODUCED"
    verify_freeze(root)
    c.io.save(
        root / "model-verification.json",
        {
            **verified,
            "old_baseline_max_difference": old_max,
            "freeze_unchanged": True,
            "actual_fits": verified["models"],
        },
    )
    print(c.io.canonical(verified), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with c.io.writer_lock(args.root):
        run(args.root)
