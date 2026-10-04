"""固定N+NE输入的20/60交易会话更新研究：先冻结日历，训练标签必须先成熟。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts import fund_002112_price_search_v1 as prior
from scripts import fund_002112_source_admission_v1 as admission

io, ROOT = admission.io, admission.ROOT
RECIPES = ("LR_C1", "HGB_D2", "RF_D4")
CADENCES = (20, 60)
TOLERANCE = 1e-13


def matrix(rows):
    """只取冻结的8项原净值与7项额外净值特征，不读取大盘或事件组。"""
    return np.asarray([r["groups"]["N"] + r["groups"]["NE"] for r in rows], dtype=float)


def calendar(rows):
    """从首个2025预测会话统一计数，不因候选方法、涨跌或预测误差移动更新日。"""
    dev = [i for i, r in enumerate(rows) if r["target"].startswith("2025")]
    indices = [rows[i]["session_index"] for i in dev]
    if len(dev) != 243 or indices != list(range(indices[0], indices[0] + 243)):
        raise ValueError("COMMON_243_CONTIGUOUS_SESSIONS_REQUIRED")
    updates = {}
    for start in range(0, len(dev), 20):
        cutoff = rows[dev[start]]["as_of"]
        training = [
            i for i, r in enumerate(rows) if r["target"] < rows[dev[start]]["target"] and r["label_mature_at"] < cutoff
        ]
        # 60日模型复用同一更新日已拟合的20日模型；需要覆盖的预测区间取二者并集。
        span = 60 if start % 60 == 0 else 20
        updates[f"U{start:03d}"] = {
            "start": start,
            "cutoff_exclusive": cutoff,
            "training": training,
            "evaluation": dev[start : min(start + span, len(dev))],
            "max_training_label_mature_at": max(rows[i]["label_mature_at"] for i in training),
        }
    mappings = {
        f"{recipe}__EVERY{cadence}": [
            {
                "target": rows[i]["target"],
                "row_index": i,
                "as_of": rows[i]["as_of"],
                "model_id": f"{recipe}__U{position // cadence * cadence:03d}",
            }
            for position, i in enumerate(dev)
        ]
        for recipe in RECIPES
        for cadence in CADENCES
    }
    return {"development_indices": dev, "updates": updates, "prediction_maps": mappings}


def prepare():
    """注册六候选、全部更新日及逐日模型映射，冻结后才允许监督拟合。"""
    if (ROOT / "protocol.json").exists():
        return io.read(ROOT / "protocol.json")
    admission.audit()
    raw = io.lines(prior.ROOT / "inputs.jsonl")
    assert all(r["target"] <= "2025-12-31" for r in raw)
    rows = [{**r, "groups": {k: r["groups"][k] for k in ("N", "NE")}} for r in raw]
    assert matrix(rows).shape == (484, 15) and np.isfinite(matrix(rows)).all()
    io.save_lines(ROOT / "inputs.jsonl", rows)
    shutil.copyfile(prior.ROOT / "nav-through-2025.json", ROOT / "nav-through-2025.json")
    cal = calendar(rows)
    assert cal["updates"]["U000"]["training"] == io.read(prior.ROOT / "splits.json")["V1"]["training"]
    io.save(ROOT / "update-calendar.json", cal)
    reuse = {}
    protected = [
        prior.ROOT / n
        for n in (
            "inputs.jsonl",
            "nav-through-2025.json",
            "splits.json",
            "freeze.json",
            "source-manifest.json",
            "feature-lineage.jsonl",
            "final-verification.json",
        )
    ]
    for recipe in RECIPES:
        record = prior.ROOT / "runs" / f"NNE__{recipe}__V1" / "complete.json"
        value = io.read(record)
        reuse[recipe] = {"record": str(record), **value}
        protected.extend([record, Path(value["model_path"]), Path(value["predictions_path"])])
    baseline_records = [prior.ROOT / "development-predictions/NNE__RF_D4.jsonl"]
    previous_baselines = io.read(prior.ROOT / "baselines.json")
    baseline_records.extend(Path(previous_baselines[f]["N_reference"]["path"]) for f in prior.engine.FOLDS)
    protected.extend(baseline_records)
    io.save(
        ROOT / "baseline-and-reuse.json",
        {
            "initial_models": reuse,
            "fixed_nne": str(baseline_records[0]),
            "N": [str(p) for p in baseline_records[1:]],
            "expected_correct": {"N": 124, "FIXED_NNE": 129, "ALWAYS_UP": 128},
        },
    )
    plan = {
        "at": io.now(),
        "version": "FIXED_20_60_UPDATE_FREQUENCY_V1",
        "horizon": "D0800_TO_NEXT_U",
        "candidates": [{"id": f"{r}__EVERY{n}", "recipe": r, "cadence": n} for r in RECIPES for n in CADENCES],
        "input": "N8_PLUS_NE7_ONLY",
        "all_training_history": True,
        "label_max": "2025-12-31",
        "update_counting": "FIRST_COMMON_DEVELOPMENT_SESSION_ZERO_THEN_20_OR60_NO_ERROR_DEPENDENCE",
        "maturity": "LABEL_MATURE_AT_STRICTLY_BEFORE_UPDATE_AS_OF",
        "preprocessing": "MEDIAN_MISSING_INDICATOR_STANDARDIZATION_FIT_ON_EACH_TRAINING_INTERVAL_ONLY",
        "recipes": {r: prior.engine.kernel.RECIPES[r] for r in RECIPES},
        "max_actual_supervised_fits": 80,
        "internal_supervised_steps_per_model": 1,
        "expected_new_fits": 36,
        "maximum_replay_fits": 1,
        "reuse_initial_models": 3,
        "shared_updates": "60SESSION_UPDATES_REUSE_IDENTICAL_20SESSION_MODELS",
        "selection": "CORRECT_DESC_MIN_OLD_FOLD_ACCURACY_DESC_CADENCE_DESC_ID_ASC",
        "stop": "FINISH6CANDIDATES_AND_AT_MOST1_REPLAY_OR80ATTEMPTS_NO_EXTENSION",
        "replay": "WINNER_LAST_UPDATE_ONCE; ABS_TOL1e-13,REL_TOL0,IDENTICAL_DIRECTION",
        "historical_development_previously_seen": True,
        "new_2026_scores": None,
        "source_first_seen_and_revision_history_proven": False,
        "adoption": False,
        "event_training_authorized": False,
    }
    io.save(ROOT / "protocol.json", plan)
    old_env = io.read(prior.ROOT / "environment.json")
    assert old_env["python"] == sys.version
    assert all(importlib.metadata.version(k) == v for k, v in old_env["packages"].items())
    code = {
        Path(__file__),
        Path(admission.__file__),
        Path("scripts/fund_002112_update_frequency_verify_v1.py"),
        Path("scripts/test_fund_002112_update_frequency_v1.py"),
    }
    code.update(Path(k) for k in io.read(prior.ROOT / "freeze.json")["files"] if k.endswith(".py"))
    copies = {}
    for path in sorted(code):
        dest = ROOT / "code-snapshot" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        copies[str(path)] = {"sha256": io.sha(path), "copy": str(dest)}
    io.save(
        ROOT / "environment.json",
        {"at": io.now(), "python": sys.version, "packages": old_env["packages"], "code": copies},
    )
    protected.extend(sorted(code))
    protected.extend(
        ROOT / n
        for n in (
            "protocol.json",
            "inputs.jsonl",
            "nav-through-2025.json",
            "update-calendar.json",
            "baseline-and-reuse.json",
            "environment.json",
            "event-source-admission.json",
        )
    )
    io.save(ROOT / "freeze.json", {"at": io.now(), "files": {str(p): io.sha(p) for p in protected}})
    return {
        "registered_candidates": 6,
        "unique_update_models": 39,
        "planned_new_fits_including_replay": 37,
        "at": plan["at"],
    }


def verify_freeze():
    for path, value in io.read(ROOT / "freeze.json")["files"].items():
        assert io.sha(Path(path)) == value, path


def fit_update(recipe, update_id, replay=False):
    """真实fit先记账，初始模型只读复用；任何失败都留存且占预算，不暗中重试。"""
    rid = f"{recipe}__{update_id}" + ("__replay" if replay else "")
    folder = ROOT / "runs" / rid
    if (folder / "complete.json").exists():
        return io.read(folder / "complete.json")
    if (folder / "failed.json").exists():
        raise ValueError("FAILED_ATTEMPT_PRESERVED_NO_AUTOMATIC_RETRY")
    verify_freeze()
    rows = io.lines(ROOT / "inputs.jsonl")
    nav = io.read(ROOT / "nav-through-2025.json")
    update = io.read(ROOT / "update-calendar.json")["updates"][update_id]
    tr, er = ([rows[i] for i in update[k]] for k in ("training", "evaluation"))
    x, xe = matrix(tr), matrix(er)
    y = prior.source.timing.labels(tr, nav)
    signature = io.digest({"dates": [r["target"] for r in tr], "x": x.tolist(), "y": y, "recipe": recipe})
    reference = None
    if update_id == "U000" and not replay:
        reference = io.read(ROOT / "baseline-and-reuse.json")["initial_models"][recipe]
        assert io.sha(Path(reference["model_path"])) == reference["model_sha256"]
        model = joblib.load(reference["model_path"])
        old_split = io.read(prior.ROOT / "splits.json")["V1"]["training"]
        old_rows = io.lines(prior.ROOT / "inputs.jsonl")
        assert [rows[i]["target"] for i in update["training"]] == [old_rows[i]["target"] for i in old_split]
        np.testing.assert_array_equal(x, matrix([old_rows[i] for i in old_split]))
    else:
        ledger = io.lines(ROOT / "fit-ledger.jsonl")
        if len(ledger) >= 80:
            raise ValueError("80_ACTUAL_SUPERVISED_FIT_BUDGET_REACHED")
        io.append(
            ROOT / "fit-ledger.jsonl",
            {
                "at": io.now(),
                "id": rid,
                "recipe": recipe,
                "update": update_id,
                "signature": signature,
                "training_dates": [r["target"] for r in tr],
                "cutoff_exclusive": update["cutoff_exclusive"],
                "max_label_mature_at": update["max_training_label_mature_at"],
                "replay": replay,
                "supervised_fits": 1,
            },
        )
        try:
            imputer = SimpleImputer(strategy="median", add_indicator=True)
            transformed = imputer.fit_transform(x)
            scaler = StandardScaler().fit(transformed)
            learner = prior.engine.kernel.estimator(recipe)
            with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                learner.fit(scaler.transform(transformed), y, sample_weight=None)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise ValueError("NOT_CONVERGED")
            model = {"model": learner, "imputer": imputer, "scaler": scaler}
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, folder / "model.joblib")
        except Exception as exc:
            io.save(folder / "failed.json", {"at": io.now(), "id": rid, "type": type(exc).__name__, "reason": str(exc)})
            raise
    with threadpool_limits(limits=2):
        probs = model["model"].predict_proba(model["scaler"].transform(model["imputer"].transform(xe)))
    aligned = np.zeros((len(er), 3))
    for j, label in enumerate(model["model"].classes_):
        aligned[:, prior.engine.kernel.CLASSES.index(label)] = probs[:, j]
    predictions = [
        {
            "target": r["target"],
            "as_of": r["as_of"],
            "model_id": rid,
            "model_cutoff": update["cutoff_exclusive"],
            "max_training_label_mature_at": update["max_training_label_mature_at"],
            "predicted": prior.engine.kernel.CLASSES[int(np.argmax(aligned[i]))],
            "probabilities": aligned[i].tolist(),
        }
        for i, r in enumerate(er)
    ]
    io.save_lines(folder / "predictions.jsonl", predictions)
    model_path = Path(reference["model_path"]) if reference else folder / "model.joblib"
    done = {
        "at": io.now(),
        "id": rid,
        "recipe": recipe,
        "update": update_id,
        "signature": signature,
        "training_count": len(tr),
        "evaluation_count": len(er),
        "reused": reference is not None,
        "reused_from": reference["record"] if reference else None,
        "model_path": str(model_path),
        "model_sha256": io.sha(model_path),
        "predictions_path": str(folder / "predictions.jsonl"),
        "predictions_sha256": io.sha(folder / "predictions.jsonl"),
    }
    io.save(folder / "complete.json", done)
    return done


def run():
    prior.engine.access_guard()
    verify_freeze()
    if (ROOT / "completion.json").exists():
        return io.read(ROOT / "completion.json")
    cal = io.read(ROOT / "update-calendar.json")
    for recipe in RECIPES:
        for update in cal["updates"]:
            fit_update(recipe, update)
        print(f"FINISHED {recipe} 13 update models", flush=True)
    rows = io.lines(ROOT / "inputs.jsonl")
    nav = io.read(ROOT / "nav-through-2025.json")
    dev = [rows[i] for i in cal["development_indices"]]
    old_splits = io.read(prior.ROOT / "splits.json")
    results = []
    for candidate in io.read(ROOT / "protocol.json")["candidates"]:
        predictions = []
        cache = {}
        for mapping in cal["prediction_maps"][candidate["id"]]:
            rid = mapping["model_id"]
            if rid not in cache:
                cache[rid] = {p["target"]: p for p in io.lines(ROOT / "runs" / rid / "predictions.jsonl")}
            predictions.append(cache[rid][mapping["target"]])
        scores = []
        for fold in prior.engine.FOLDS:
            dates = set(old_splits[fold]["evaluation_dates"])
            er = [r for r in dev if r["target"] in dates]
            values = [p for p in predictions if p["target"] in dates]
            scores.append(prior.engine.score(er, nav, values))
        result = {
            "candidate": candidate,
            "score": prior.engine.score(dev, nav, predictions),
            "fold_scores": scores,
            "min_fold_accuracy": min(v["accuracy"] for v in scores),
        }
        results.append(result)
        io.save_lines(ROOT / "development-predictions" / (candidate["id"] + ".jsonl"), predictions)
    ranking = sorted(
        results,
        key=lambda r: (
            -r["score"]["correct"],
            -r["min_fold_accuracy"],
            -r["candidate"]["cadence"],
            r["candidate"]["id"],
        ),
    )
    io.save(ROOT / "candidate-results.json", results)
    io.save(ROOT / "selection.json", {"at": io.now(), "ranking": ranking, "winner": ranking[0]})
    winner = ranking[0]["candidate"]
    original = fit_update(winner["recipe"], "U240")
    replay = fit_update(winner["recipe"], "U240", replay=True)
    a, b = (io.lines(Path(r["predictions_path"])) for r in (original, replay))
    va, vb = (np.asarray([p["probabilities"] for p in values]) for values in (a, b))
    np.testing.assert_allclose(va, vb, atol=TOLERANCE, rtol=0)
    assert [p["predicted"] for p in a] == [p["predicted"] for p in b]
    io.save(
        ROOT / "replay-audit.json",
        {
            "at": io.now(),
            "max_absolute_probability_difference": float(np.max(abs(va - vb))),
            "absolute_tolerance": TOLERANCE,
            "relative_tolerance": 0,
            "directions_equal": True,
            "protocol_registered_before_fit": True,
        },
    )
    done = {
        "at": io.now(),
        "status": "FIXED_FREQUENCY_SEARCH_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "actual_fits": len(io.lines(ROOT / "fit-ledger.jsonl")),
        "reused_initial_models": 3,
        "candidates": 6,
        "registered_remaining": 0,
        "winner": ranking[0],
        "new_2026_scores": None,
        "adopted": False,
    }
    io.save(ROOT / "completion.json", done)
    return {k: v for k, v in done.items() if k != "winner"} | {
        "winner": winner["id"],
        "correct": ranking[0]["score"]["correct"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "run": run}[args.command](), ensure_ascii=False, indent=2))
