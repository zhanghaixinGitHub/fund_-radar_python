"""固定四组、同一滚动日历和拟合预算；2025选型，2026诊断，新增日先预测后揭晓。"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts import fund_002112_recent_bodies_train_v1 as old
from scripts import fund_002112_semantic_features_v1 as source

io, ROOT, core = source.io, source.ROOT, source.core
GROUPS = ("BASE", "HOLDINGS", "FACTS", "HOLDINGS_FACTS")


def matrix(rows, group):
    return np.array(
        [
            r["groups"]["N"]
            + r["groups"]["NE"]
            + (r["H"] if group in ("HOLDINGS", "HOLDINGS_FACTS") else [])
            + (r["F"] if group in ("FACTS", "HOLDINGS_FACTS") else [])
            for r in rows
        ],
        dtype=float,
    )


def transform(rows, group, state=None):
    values = matrix(rows, group)
    if state is None:
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        filled = imputer.fit_transform(values)
        state = {"imputer": imputer, "scaler": StandardScaler().fit(filled)}
    return state["scaler"].transform(state["imputer"].transform(values)), state


def calendar(rows, year):
    indices = [i for i, r in enumerate(rows) if r["target"].startswith(year)]
    updates = []
    for start in range(0, len(indices), 20):
        evaluate = indices[start : start + 20]
        cutoff = rows[evaluate[0]]["as_of"]
        training = [
            i for i, r in enumerate(rows) if r["target"] < rows[evaluate[0]]["target"] and r["label_mature_at"] < cutoff
        ]
        updates.append({"id": year + f"_U{start:03d}", "training": training, "evaluate": evaluate, "cutoff": cutoff})
    return updates


def freeze(rows):
    path = ROOT / "training-freeze.json"
    if path.exists():
        verify_freeze()
        return
    assert (source.FEATURE_ROOT / "features-completion.json").exists() and (ROOT / "test-results-r3.json").exists()
    dates = {year: calendar(rows, year) for year in ("2025", "2026")}
    previous = io.read(core.OLD / "calendar.json")
    assert len([r for r in rows if r["target"].startswith("2025")]) == 243
    for current, past in zip(dates["2025"], previous["updates"].values(), strict=True):
        assert current["training"] == past["training"]
        assert current["cutoff"] == past["cutoff_exclusive"]
    io.save(ROOT / "training-calendar.json", dates)
    protocol = {
        "at": io.now(),
        "groups": GROUPS,
        "recipe": core.plan()["recipe"],
        "cadence": 20,
        "dimensions": {g: int(matrix(rows[:1], g).shape[1]) for g in GROUPS},
        "maximum_fits": 100,
        "planned_fits": 4 * sum(map(len, dates.values())) + 1,
        "replay_fits": 1,
        "development_selection": "2025正确数最多，平局少特征优先",
        "audit_selection": False,
        "adoption": False,
    }
    assert protocol["planned_fits"] <= 100
    io.save(ROOT / "training-protocol.json", protocol)
    paths = list(ROOT.glob("*.json")) + list(ROOT.glob("*.jsonl"))
    paths += list(source.FEATURE_ROOT.glob("*.json")) + list(source.FEATURE_ROOT.glob("*.jsonl"))
    paths = [p for p in paths if not p.name.endswith("progress.json")]
    paths += list(io.PY.glob("scripts/fund_002112_semantic_*v1.py"))
    paths += [
        io.PY / "scripts/test_fund_002112_semantic_v1.py",
        Path(old.__file__),
        Path(io.__file__),
        Path(core.previous.__file__),
        Path(core.previous.holdings.__file__),
        core.previous.BUNDLE,
        core.previous.PREVIOUS / "nav-through-2025.json",
        core.previous.PREVIOUS / "development-predictions/RF_D4__EVERY20.jsonl",
    ]
    for p in paths.copy():
        if p.suffix == ".py":
            dest = ROOT / "code-snapshot" / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as stream:
                stream.write(p.read_bytes())
            paths.append(dest)
    io.save(
        path,
        {
            "at": io.now(),
            "python": sys.version,
            "packages": {k: importlib.metadata.version(k) for k in ("numpy", "scikit-learn", "joblib")},
            "files": {str(p.resolve()): io.sha(p) for p in paths},
        },
    )


def verify_freeze():
    for path, digest in io.read(ROOT / "training-freeze.json")["files"].items():
        if io.sha(path) != digest:
            raise ValueError("FROZEN_FILE_CHANGED:" + path)


def fit_one(group, update, rows, nav, replay=False):
    identifier = group + "__" + update["id"] + ("__replay" if replay else "")
    folder = ROOT / "runs" / identifier
    if (folder / "complete.json").exists():
        return io.read(folder / "complete.json")
    ledger = io.lines(ROOT / "fit-ledger.jsonl")
    if any(r["id"] == identifier for r in ledger):
        raise ValueError("FAILED_FIT_CANNOT_SILENTLY_RETRY:" + identifier)
    assert len(ledger) < 100
    if replay:
        assert not any(r["replay"] for r in ledger)
    train = [rows[i] for i in update["training"]]
    evaluate = [rows[i] for i in update["evaluate"]]
    assert all(r["label_mature_at"] < update["cutoff"] for r in train)
    assert all(r["target"] < evaluate[0]["target"] for r in train)
    x, state = transform(train, group)
    xe, _ = transform(evaluate, group, state)
    labels = old.labels(train, nav)
    record = {
        "at": io.now(),
        "id": identifier,
        "group": group,
        **update,
        "update_id": update["id"],
        "replay": replay,
        "fit_shape": list(x.shape),
    }
    record["id"] = identifier
    io.append(ROOT / "fit-ledger.jsonl", record)
    try:
        model = RandomForestClassifier(**core.plan()["recipe"], n_jobs=2)
        with threadpool_limits(limits=2):
            model.fit(x, labels)
            predicted = model.predict_proba(xe)
        aligned = np.zeros((len(evaluate), 3))
        for column, label in enumerate(model.classes_):
            aligned[:, old.CLASSES.index(label)] = predicted[:, column]
        predictions = [
            {
                "target": r["target"],
                "as_of": r["as_of"],
                "model_id": identifier,
                "predicted": old.CLASSES[int(np.argmax(aligned[k]))],
                "probabilities": aligned[k].tolist(),
            }
            for k, r in enumerate(evaluate)
        ]
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"group": group, "model": model, "preprocessing": state}, folder / "model.joblib")
        io.save_lines(folder / "predictions.jsonl", predictions)
        done = {
            "at": io.now(),
            "id": identifier,
            "model_sha256": io.sha(folder / "model.joblib"),
            "predictions_sha256": io.sha(folder / "predictions.jsonl"),
        }
        io.save(folder / "complete.json", done)
        return done
    except Exception as exc:
        io.save(folder / "failed.json", {"at": io.now(), "type": type(exc).__name__, "message": str(exc)})
        raise


def predictions(group, updates):
    return [p for u in updates for p in io.lines(ROOT / "runs" / (group + "__" + u["id"]) / "predictions.jsonl")]


def verify_models(rows):
    count = 0
    ledger = io.lines(ROOT / "fit-ledger.jsonl")
    for entry in ledger:
        folder = ROOT / "runs" / entry["id"]
        done = io.read(folder / "complete.json")
        assert io.sha(folder / "model.joblib") == done["model_sha256"]
        assert io.sha(folder / "predictions.jsonl") == done["predictions_sha256"]
        saved = joblib.load(folder / "model.joblib")
        training = [rows[i] for i in entry["training"]]
        assert all(r["label_mature_at"] < entry["cutoff"] for r in training)
        _, expected = transform(training, entry["group"])
        for name in ("statistics_",):
            np.testing.assert_array_equal(
                getattr(expected["imputer"], name), getattr(saved["preprocessing"]["imputer"], name)
            )
        np.testing.assert_array_equal(expected["scaler"].mean_, saved["preprocessing"]["scaler"].mean_)
        xe, _ = transform([rows[i] for i in entry["evaluate"]], entry["group"], saved["preprocessing"])
        aligned = np.zeros((len(xe), 3))
        with threadpool_limits(limits=2):
            probs = saved["model"].predict_proba(xe)
        for col, label in enumerate(saved["model"].classes_):
            aligned[:, old.CLASSES.index(label)] = probs[:, col]
        stored = io.lines(folder / "predictions.jsonl")
        np.testing.assert_allclose(aligned, [p["probabilities"] for p in stored], atol=1e-12, rtol=0)
        assert [p["predicted"] for p in stored] == [old.CLASSES[int(np.argmax(p))] for p in aligned]
        count += len(xe)
    return {
        "model_count": len(ledger),
        "probability_rows": count,
        "training_only_preprocessing": True,
        "mature_labels_only": True,
    }


def read_reserved_labels():
    """唯一联网边界是本机只读数据库；先核验已经固定预测，再读取新增测试日的答案。"""
    out = ROOT / "reserved-labels.json"
    if out.exists():
        return io.read(out)
    receipt = io.read(ROOT / "predictions-before-reveal.json")
    for path, digest in receipt["prediction_files"].items():
        assert io.sha(path) == digest
    assert io.sha(ROOT / "selection.json") == receipt["selection_sha256"]
    from sqlalchemy import text

    from scripts import fund_002112_recent_completion_v1 as recent

    values = {}
    e = recent.engine()
    with e.connect() as connection:
        for day in io.read(ROOT / "exposure-audit.json")["independent_dates"]:
            results = list(
                connection.execute(
                    text(
                        "select unit_nav::text,nav_date::text,ann_date::text,source_id::text,content_hash,"
                        "created_at::text,updated_at::text from nav_daily where fund_code=:fund and nav_date=:day"
                    ),
                    {"fund": "002112", "day": day},
                ).mappings()
            )
            assert results and len({r["unit_nav"] for r in results}) == 1, "NAV_MISSING_OR_SOURCES_CONFLICT"
            values[day] = {**dict(results[0]), "all_source_rows": [dict(r) for r in results]}
    e.dispose()
    result = {
        "at": io.now(),
        "values": values,
        "predictions_frozen_at": receipt["at"],
        "selection_before_reveal": True,
        "read_only_local_database": True,
    }
    io.save(out, result)
    return result


def run():
    rows = core.previous.read_lines(source.FEATURE_ROOT / "inputs.jsonl")
    freeze(rows)
    if (ROOT / "completion.json").exists():
        return io.read(ROOT / "completion.json")
    dates = io.read(ROOT / "training-calendar.json")
    bundle_nav = io.read(core.previous.BUNDLE)["nav"]
    nav = {**bundle_nav, **io.read(core.previous.PREVIOUS / "nav-through-2025.json")}
    with io.offline_guard():
        for group in GROUPS:
            verify_freeze()
            for update in dates["2025"]:
                fit_one(group, update, rows, nav)
            print("2025比较已完成：" + group, flush=True)
        dev = [r for r in rows if r["target"].startswith("2025")]
        pred = {g: predictions(g, dates["2025"]) for g in GROUPS}
        scores = {g: old.score(dev, nav, pred[g]) for g in GROUPS}
        baseline_old = io.lines(core.previous.PREVIOUS / "development-predictions/RF_D4__EVERY20.jsonl")
        assert [p["predicted"] for p in baseline_old] == [p["predicted"] for p in pred["BASE"]], (
            "BASELINE_NOT_REPRODUCED"
        )
        np.testing.assert_allclose(
            [p["probabilities"] for p in baseline_old], [p["probabilities"] for p in pred["BASE"]], atol=1e-12, rtol=0
        )
        selected = min(GROUPS, key=lambda g: (-scores[g]["correct"], matrix(rows[:1], g).shape[1]))
        if not (ROOT / "selection.json").exists():
            io.save(
                ROOT / "selection.json",
                {
                    "at": io.now(),
                    "selected": selected,
                    "scores": scores,
                    "baseline_reproduced": True,
                    "choice_uses_2025_only": True,
                    "adoption": False,
                },
            )
        else:
            assert io.read(ROOT / "selection.json")["selected"] == selected
        for group in GROUPS:
            for update in dates["2026"]:
                fit_one(group, update, rows, nav)
            print("2026固定方案预测已完成：" + group, flush=True)
        fit_one(selected, dates["2026"][-1], rows, nav, replay=True)
        name = selected + "__" + dates["2026"][-1]["id"]
        a = io.lines(ROOT / "runs" / name / "predictions.jsonl")
        b = io.lines(ROOT / "runs" / (name + "__replay") / "predictions.jsonl")
        np.testing.assert_allclose([p["probabilities"] for p in a], [p["probabilities"] for p in b], atol=1e-12, rtol=0)
        checks = verify_models(rows)
        verify_freeze()
        if not (ROOT / "predictions-before-reveal.json").exists():
            io.save(
                ROOT / "predictions-before-reveal.json",
                {
                    "at": io.now(),
                    "selection_sha256": io.sha(ROOT / "selection.json"),
                    "prediction_files": {str(p): io.sha(p) for p in (ROOT / "runs").glob("*/predictions.jsonl")},
                    "verification": checks,
                },
            )
    revealed = read_reserved_labels()
    nav.update(revealed["values"])
    results = {"development": scores, "audit_exposed": {}, "independent": {}, "comparisons": {}}
    independent_dates = set(io.read(ROOT / "exposure-audit.json")["independent_dates"])
    for stage, desired in (
        ("audit_exposed", [r for r in rows if r["target"].startswith("2026") and r["target"] not in independent_dates]),
        ("independent", [r for r in rows if r["target"] in independent_dates]),
    ):
        if not desired:
            continue
        targets = {r["target"] for r in desired}
        subset = {g: [p for p in predictions(g, dates["2026"]) if p["target"] in targets] for g in GROUPS}
        results[stage] = {g: old.score(desired, nav, subset[g]) for g in GROUPS}
        results["comparisons"][stage] = {
            g: old.paired(desired, nav, subset["BASE"], subset[g]) for g in GROUPS if g != "BASE"
        }
    results["comparisons"]["development"] = {
        g: old.paired(dev, nav, pred["BASE"], pred[g]) for g in GROUPS if g != "BASE"
    }
    protected = io.read(ROOT / "protection-before.json")["files"]
    changed = [path for path, digest in protected.items() if io.sha(path) != digest]
    assert not changed, changed
    result = {
        "at": io.now(),
        "selected": selected,
        "scores": results,
        "actual_fits": len(io.lines(ROOT / "fit-ledger.jsonl")),
        "verification": checks,
        "protected_existing_files": len(protected),
        "changed_existing_files": changed,
        "adoption": False,
        "new_independent_sample_count": len(independent_dates),
    }
    io.save(ROOT / "completion.json", result)
    return result


if __name__ == "__main__":
    with io.writer_lock(ROOT):
        print(io.canonical(run()), flush=True)
