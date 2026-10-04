"""固定36候选的开发期续研：窗口、时间衰减和准入后的信息组，不计算2026成绩。"""

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

from scripts import fund_002112_all_information_search_v1 as kernel
from scripts import fund_002112_development_timing_audit_v1 as timing

io, ROOT = timing.io, timing.ROOT
GROUPS = {
    "N": ["N"],
    "NM": ["N", "M"],
    "NMI": ["N", "M", "I"],
    "NMI_COUNTS": ["N", "M", "I", "COUNTS"],
    "NMI_GLOBAL": ["N", "M", "I", "GLOBAL"],
    "NMI_COUNTS_GLOBAL": ["N", "M", "I", "COUNTS", "GLOBAL"],
}
FOLDS = ("V1", "V2", "V3")


def access_guard() -> None:
    """训练与复核进程禁止打开混合年代标签原件、旧2026结果及T模型。"""

    def check(name, args):
        if name != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = str(args[0]).replace("\\", "/").lower()
        forbidden = (
            "audit-result.json",
            "selected-audit-predictions.jsonl",
            "snapshot/nav-facts.json",
            "snapshot/normalized-inputs.json",
            "snapshot/nav-by-date.json",
        )
        if any(s in path for s in forbidden) or ("/models/" in path and "__t/" in path):
            raise PermissionError("POST_2025_OR_MIXED_LABEL_ARTIFACT_FORBIDDEN")

    sys.addaudithook(check)


def matrix(rows: list[dict], group: str) -> np.ndarray:
    return np.asarray([[v for key in GROUPS[group] for v in row["groups"][key]] for row in rows], dtype=float)


def training_rows(rows: list[dict], split: dict, window: int | None) -> list[dict]:
    """先应用标签成熟和外层时间边界，再从合格行末端取固定长度，不按涨跌筛选。"""
    values = [rows[i] for i in split["training"]]
    if any(r["label_mature_at"] >= split["cutoff_exclusive"] for r in values):
        raise ValueError("IMMATURE_TRAINING_LABEL")
    return values[-window:] if window is not None else values


def weights(rows: list[dict], half_life: int | None) -> np.ndarray | None:
    """126个交易会话的半衰期；只按训练日期计算，归一到均值1以固定正则化量级。"""
    if half_life is None:
        return None
    age = np.asarray([rows[-1]["session_index"] - r["session_index"] for r in rows], dtype=float)
    values = np.exp2(-age / half_life)
    return values / values.mean()


def prepare() -> dict:
    """登记所有候选及停止条件；不看新结果、不计算任何2026标签。"""
    if (ROOT / "protocol.json").exists():
        return io.read(ROOT / "protocol.json")
    review = io.read(ROOT / "timing-audit.json")
    if review["repaired_input_gate"] != "PASS_FOR_RECONSTRUCTED_DEVELOPMENT_ONLY_AFTER_QUARANTINE":
        raise ValueError("INPUT_TIME_REVIEW_NOT_PASSED")
    rows = io.lines(ROOT / "inputs.jsonl")
    if any(r["target"] > timing.LAST_LABEL_DATE for r in rows):
        raise ValueError("POST_2025_INPUT")
    splits = {}
    for fold in FOLDS:
        start, end = kernel.old.EVAL_RANGE[fold]
        er = [i for i, r in enumerate(rows) if start <= r["target"] <= end]
        cutoff = rows[er[0]]["as_of"]
        tr = [i for i, r in enumerate(rows) if r["target"] < start and r["label_mature_at"] < cutoff]
        splits[fold] = {
            "training": tr,
            "evaluation": er,
            "cutoff_exclusive": cutoff,
            "train_dates": [rows[i]["target"] for i in tr],
            "evaluation_dates": [rows[i]["target"] for i in er],
        }
    candidates = []
    for group in GROUPS:
        for window in (None, 252, 126):
            for half_life in (None, 126):
                candidates.append(
                    {
                        "id": f"{group}__W{window or 'ALL'}__D{half_life or 'NONE'}",
                        "group": group,
                        "window": window,
                        "half_life": half_life,
                        "recipe": "LR_C1",
                    }
                )
    protocol = {
        "at": io.now(),
        "version": "DEVELOPMENT_ONLY_QUARANTINED_V1",
        "candidates": candidates,
        "input_groups": GROUPS,
        "folds": FOLDS,
        "max_actual_fits": 120,
        "candidate_count": 36,
        "label_end": timing.LAST_LABEL_DATE,
        "2026_evaluation": "FORBIDDEN",
        "causal_basis": {
            "window": "Compare broad history with roughly one year and half year to reduce stale regimes",
            "decay": "Fixed 126-session half life reduces older observations; no future-error based weighting",
            "groups": "Add market, industry background, observed counts and unweighted evidence separately",
        },
        "preprocessing": "Training-only median+missing indicator and unweighted standardization; no outer-fold fitting",
        "sample_weight": "Training-date only exponential decay, mean normalized to1; learner only",
        "parameters": {"recipe": "LR_C1", "C": 1.0, "max_iter": 5000, "tol": 1e-8, "seed": 0},
        "prediction": "Learned-class probability argmax; no tuned threshold or class definition",
        "selection": "POOLED_DEV_CORRECT_DESC_MIN_FOLD_ACCURACY_DESC_FEATURES_ASC_ID_ASC",
        "stop": "All36 candidates evaluated or120 actual attempts reached; no adaptive expansion",
        "baseline": "N_LR_C1,ALWAYS_UP,TRAIN_MAJORITY,OLD_FIXED_RECIPES_AFTER_REPORT_QUARANTINE",
        "old_fixed_recipes": [["NMI", "LR_C1"], ["NMI", "HGB_D3"], ["N", "RF_D4"]],
        "old_recipe_change": "H and F removed by timing audit; not equivalent to old full-information ensemble",
        "old_scores_immutable": True,
        "historical_2025_previously_seen": True,
        "repeat": "Repeat winner V3 once; additional attempts count in120",
        "deployment": False,
    }
    io.save(ROOT / "protocol.json", protocol)
    io.save(ROOT / "splits.json", splits)
    modules = (
        Path(__file__),
        Path(timing.__file__),
        Path(kernel.__file__),
        Path(kernel.impact.__file__),
        Path(kernel.sources.__file__),
        Path(kernel.old.__file__),
        Path(io.__file__),
        Path(timing.previous.__file__),
        Path(kernel.impact.stats.__file__),
        Path(timing.impact.legacy_features.__file__),
    )
    copies = {}
    for source in modules:
        target = ROOT / "code-snapshot" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copies[str(source)] = {"sha256": io.sha(source), "copy": str(target)}
    io.save(
        ROOT / "environment.json",
        {
            "at": io.now(),
            "python": sys.version,
            "source_copies": copies,
            "packages": {
                p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn", "joblib", "threadpoolctl")
            },
        },
    )
    tracked = [
        ROOT / n
        for n in (
            "inputs.jsonl",
            "nav-through-2025.json",
            "protocol.json",
            "splits.json",
            "timing-audit.json",
            "environment.json",
        )
    ] + list(modules)
    io.save(ROOT / "freeze.json", {"at": io.now(), "files": {str(p): io.sha(p) for p in tracked}})
    return protocol


def verify_freeze():
    if any(io.sha(Path(p)) != sha for p, sha in io.read(ROOT / "freeze.json")["files"].items()):
        raise ValueError("DEVELOPMENT_FREEZE_CHANGED")


def fit(candidate: dict, fold: str, replay=False) -> dict:
    """每次真实fit先计次；输入等价只做别名引用，旧批次只读V1/V2/V3模型。"""
    verify_freeze()
    if fold not in FOLDS:
        raise ValueError("NON_DEVELOPMENT_FOLD_FORBIDDEN")
    rows, nav = io.lines(ROOT / "inputs.jsonl"), io.read(ROOT / "nav-through-2025.json")
    split = io.read(ROOT / "splits.json")[fold]
    tr = training_rows(rows, split, candidate["window"])
    er = [rows[i] for i in split["evaluation"]]
    x, xe = matrix(tr, candidate["group"]), matrix(er, candidate["group"])
    y = timing.labels(tr, nav)
    w = weights(tr, candidate["half_life"])
    identity = {
        "train_dates": [r["target"] for r in tr],
        "evaluation_dates": [r["target"] for r in er],
        "x": x.tolist(),
        "xe": xe.tolist(),
        "y": y,
        "weights": None if w is None else w.tolist(),
        "recipe": candidate["recipe"],
    }
    signature = io.digest(identity)
    rid = candidate["id"] + "__" + fold + ("__replay" if replay else "")
    folder = ROOT / "runs" / rid
    if (folder / "complete.json").exists():
        return io.read(folder / "complete.json")
    if not replay:
        for p in (ROOT / "runs").glob("*/complete.json"):
            done = io.read(p)
            if done.get("signature") == signature:
                done = {**done, "id": rid, "reused_from": str(p), "is_alias": True}
                io.save(folder / "complete.json", done)
                return done
        if candidate["window"] is None and candidate["half_life"] is None and candidate["group"] in ("N", "NM"):
            previous = timing.previous.ROOT / "models" / f"{candidate['group']}__{candidate['recipe']}__{fold}"
            if (previous / "complete.json").exists():
                prior_rows = timing.dev_lines(timing.previous.ROOT / "inputs.jsonl")
                prior_map = {r["target"]: r for r in prior_rows}
                source_x = matrix([prior_map[r["target"]] for r in tr], candidate["group"])
                source_xe = matrix([prior_map[r["target"]] for r in er], candidate["group"])
                old_ledger = [
                    d for d in io.lines(timing.previous.ROOT / "fit-ledger.jsonl") if d["id"] == previous.name
                ][0]
                if (
                    np.array_equal(x, source_x)
                    and np.array_equal(xe, source_xe)
                    and io.digest(identity["train_dates"]) == old_ledger["training_dates_sha256"]
                    and io.digest(x.tolist()) == old_ledger["input_sha256"]
                ):
                    old = io.read(previous / "complete.json")
                    done = {
                        "id": rid,
                        "signature": signature,
                        "is_alias": True,
                        "reused_from": str(previous),
                        "model_path": str(previous / "model.joblib"),
                        "model_sha256": old["model_sha256"],
                        "predictions_path": str(previous / "predictions.jsonl"),
                        "predictions_sha256": old["predictions_sha256"],
                        "features": x.shape[1],
                    }
                    io.save(folder / "complete.json", done)
                    return done
    ledger = io.lines(ROOT / "fit-ledger.jsonl")
    if len(ledger) >= 120:
        raise ValueError("ACTUAL_FIT_BUDGET_REACHED")
    io.append(
        ROOT / "fit-ledger.jsonl",
        {
            "at": io.now(),
            "id": rid,
            "signature": signature,
            "candidate": candidate,
            "fold": fold,
            "training_dates": identity["train_dates"],
            "evaluation_dates": identity["evaluation_dates"],
            "replay": replay,
        },
    )
    folder.mkdir(parents=True, exist_ok=True)
    io.save(
        folder / "input-proof.json",
        {
            "signature": signature,
            "x_sha256": io.digest(x.tolist()),
            "y_sha256": io.digest(y),
            "weights": None if w is None else w.tolist(),
        },
    )
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    xx = imputer.fit_transform(x)
    scaler = StandardScaler().fit(xx)
    learner = kernel.estimator(candidate["recipe"])
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        learner.fit(scaler.transform(xx), y, sample_weight=w)
        probs = learner.predict_proba(scaler.transform(imputer.transform(xe)))
    if any(issubclass(v.category, ConvergenceWarning) for v in caught):
        io.save(folder / "failed.json", {"id": rid, "reason": "NOT_CONVERGED"})
        raise ValueError("FIXED_CANDIDATE_NOT_CONVERGED")
    aligned = np.zeros((len(er), 3))
    for j, label in enumerate(learner.classes_):
        aligned[:, kernel.CLASSES.index(label)] = probs[:, j]
    predictions = [
        {
            "target": r["target"],
            "predicted": kernel.CLASSES[int(np.argmax(aligned[i]))],
            "probabilities": aligned[i].tolist(),
        }
        for i, r in enumerate(er)
    ]
    model_path = folder / "model.joblib"
    pred_path = folder / "predictions.jsonl"
    joblib.dump({"model": learner, "imputer": imputer, "scaler": scaler}, model_path)
    io.save_lines(pred_path, predictions)
    done = {
        "id": rid,
        "signature": signature,
        "is_alias": False,
        "features": x.shape[1],
        "model_path": str(model_path),
        "model_sha256": io.sha(model_path),
        "predictions_path": str(pred_path),
        "predictions_sha256": io.sha(pred_path),
    }
    io.save(folder / "complete.json", done)
    return done


def predicted(run: dict) -> list[dict]:
    if io.sha(Path(run["model_path"])) != run["model_sha256"]:
        raise ValueError("MODEL_CHANGED")
    if io.sha(Path(run["predictions_path"])) != run["predictions_sha256"]:
        raise ValueError("PREDICTIONS_CHANGED")
    return io.lines(Path(run["predictions_path"]))


def score(rows, nav, predictions):
    assert [r["target"] for r in rows] == [p["target"] for p in predictions]
    return kernel.impact.stats.score(timing.labels(rows, nav), [p["predicted"] for p in predictions])


def run() -> dict:
    """完成固定开发比较和一次真实复拟合，不建立T或FINAL任务。"""
    access_guard()
    verify_freeze()
    if (ROOT / "completion.json").exists():
        return io.read(ROOT / "completion.json")
    protocol = io.read(ROOT / "protocol.json")
    rows = io.lines(ROOT / "inputs.jsonl")
    nav = io.read(ROOT / "nav-through-2025.json")
    splits = io.read(ROOT / "splits.json")
    dev = [r for r in rows if r["target"].startswith("2025-")]
    results = []
    for candidate in protocol["candidates"]:
        joined = []
        scores = []
        for fold in FOLDS:
            fitted = fit(candidate, fold)
            p = predicted(fitted)
            joined.extend(p)
            scores.append(score([rows[i] for i in splits[fold]["evaluation"]], nav, p))
        result = {
            "candidate": candidate,
            "score": score(dev, nav, joined),
            "fold_scores": scores,
            "min_fold_accuracy": min(s["accuracy"] for s in scores),
            "features": fitted["features"],
        }
        io.save_lines(ROOT / "development-predictions" / (candidate["id"] + ".jsonl"), joined)
        results.append(result)
        print(f"DEV {len(results)}/36 {candidate['id']}: {result['score']['correct']}/243", flush=True)
    ranking = sorted(
        results, key=lambda v: (-v["score"]["correct"], -v["min_fold_accuracy"], v["features"], v["candidate"]["id"])
    )
    selected = ranking[0]
    io.save(ROOT / "candidate-results.json", results)
    io.save(
        ROOT / "selection.json",
        {"at": io.now(), "winner": selected, "ranking": ranking, "development_only": True, "future_gain_proven": False},
    )
    baselines = {}
    for fold, split in splits.items():
        er = [rows[i] for i in split["evaluation"]]
        tr = [rows[i] for i in split["training"]]
        actual = timing.labels(er, nav)
        train_y = timing.labels(tr, nav)
        majority = max(kernel.CLASSES, key=lambda c: (train_y.count(c), c))
        fixed = []
        for group, recipe in protocol["old_fixed_recipes"]:
            c = {"id": f"FIXED_{group}_{recipe}", "group": group, "recipe": recipe, "window": None, "half_life": None}
            fixed.append(predicted(fit(c, fold)))
        p = kernel.ensemble(fixed)
        io.save_lines(ROOT / "baselines" / (fold + "-fixed-recipes.jsonl"), p)
        baselines[fold] = {
            "fixed_recipes_after_quarantine": score(er, nav, p),
            "always_up": kernel.impact.stats.score(actual, ["UP"] * len(actual)),
            "majority": kernel.impact.stats.score(actual, [majority] * len(actual)),
        }
    io.save(ROOT / "baselines.json", baselines)
    repeat = fit(selected["candidate"], "V3", replay=True)
    original = fit(selected["candidate"], "V3")
    assert predicted(repeat) == predicted(original)
    a, b = joblib.load(repeat["model_path"]), joblib.load(original["model_path"])
    assert np.array_equal(a["model"].coef_, b["model"].coef_)
    done = {
        "at": io.now(),
        "status": "DEVELOPMENT_BATCH_COMPLETE",
        "actual_fits": len(io.lines(ROOT / "fit-ledger.jsonl")),
        "candidates": 36,
        "replay_probabilities_and_coefficients_exact": True,
        "winner": selected,
        "post_2025_labels_or_scores_read": False,
        "adopted": False,
    }
    io.save(ROOT / "completion.json", done)
    return {k: v for k, v in done.items() if k != "winner"} | {
        "winner": selected["candidate"]["id"],
        "correct": selected["score"]["correct"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.command == "prepare" else run(), ensure_ascii=False, indent=2))
