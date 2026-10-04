"""002112 一日输入删减对照：先固定三个候选，再复核种子和已暴露历史。

仅研究F15正文处理数量、F20持仓报告年龄的影响。原35项输入与模型不变，
没有删除字段的方案直接复用一期模型；新模型只写本轮独立目录。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts import fund_002112_signal_common_v1 as c
from scripts import fund_002112_signal_features_v1 as features
from scripts import fund_002112_signal_forward_v1 as forward
from scripts import fund_002112_signal_train_v1 as t

ROOT = c.io.RESEARCH / "one-day-feature-ablation/20261002-v1"
PRIOR = c.DEFAULT_ROOT
# 下标是业务矩阵中的绝对位置：先15项净值，再F01..F20事件字段。
DROPS = {"FULL": [], "NO_F15": [29], "NO_F20": [34], "NO_F15_F20": [29, 34]}
CANDIDATES = ("NO_F15", "NO_F20", "NO_F15_F20")


def kept(candidate):
    return [i for i in range(35) if i not in DROPS[candidate]]


def matrix(rows, candidate):
    return np.asarray([r["groups"]["N"] + r["groups"]["NE"] + r["E"] for r in rows], dtype=float)[:, kept(candidate)]


def transform(rows, candidate, state=None):
    values = matrix(rows, candidate)
    if state is None:
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        filled = imputer.fit_transform(values)
        state = {"imputer": imputer, "scaler": StandardScaler().fit(filled)}
    return state["scaler"].transform(state["imputer"].transform(values)), state


def probabilities(saved, rows):
    x, _ = transform(rows, saved["candidate"], saved["preprocessing"])
    with threadpool_limits(limits=2):
        raw = saved["model"].predict_proba(x)
    aligned = np.zeros((len(rows), 3))
    for j, label in enumerate(saved["model"].classes_):
        aligned[:, c.CLASSES.index(label)] = raw[:, j]
    return aligned


def validate_training(rows, update, nav):
    """两端净值版本都必须严格早于更新时刻成熟，保留原一日截点。"""
    if not update["training"] or not update["evaluate"]:
        raise ValueError("EMPTY_TRAIN_OR_EVALUATE")
    cutoff = c.moment(update["cutoff"])
    first = rows[update["evaluate"][0]]
    for i in update["training"]:
        row = rows[i]
        if c.moment(row["label_mature_at"]) >= cutoff:
            raise ValueError("IMMATURE_LABEL")
        if row["target"] >= first["target"]:
            raise ValueError("TRAIN_EVALUATION_ORDER")
        if any(c.moment(nav[d]["available_at"]) >= cutoff for d in (row["base"], row["target"])):
            raise ValueError("IMMATURE_NAV_VERSION")


def screen(root, rows):
    """只用于训练前定位待查项：分裂使用率不表示该特征有益或有害。"""
    ledger = [
        a
        for a in c.lines(PRIOR / "fit-ledger.jsonl")
        if a["group"] == "B1" and a["update"]["id"].startswith("2025") and not a["replay"]
    ]
    imports, files = [], []
    training = set()
    for a in ledger:
        path = PRIOR / "runs" / a["id"]
        complete = c.io.read(path / "complete.json")
        assert c.io.sha(path / "model.joblib") == complete["model_sha256"]
        saved = joblib.load(path / "model.joblib")
        values = saved["model"].feature_importances_
        imp = values[:35].copy()
        for k, j in enumerate(saved["preprocessing"]["imputer"].indicator_.features_):
            imp[j] += values[35 + k]
        imports.append(imp)
        files += [path / "model.joblib", path / "complete.json"]
        training.update(a["update"]["training"])
    imp = np.asarray(imports)
    dev = [r for r in rows if r["target"].startswith("2025")]
    fields = [
        {
            "id": f"F{i + 1:02d}",
            "name": name,
            "known_2025": sum(r["E"][i] is not None for r in dev),
            "unique_training_values_including_missing": len({rows[j]["E"][i] for j in training}),
            "used_by_models": int((imp[:, 15 + i] > 0).sum()),
            "mean_impurity_importance": float(imp[:, 15 + i].mean()),
        }
        for i, name in enumerate(features.FIELDS)
    ]
    duplicates = [
        [f"F{i + 1:02d}", f"F{j + 1:02d}"]
        for i in range(20)
        for j in range(i + 1, 20)
        if len({r["E"][i] for r in rows}) > 1 and all(r["E"][i] == r["E"][j] for r in rows)
    ]
    c.io.save(
        root / "feature-screen.json",
        {
            "at": c.io.now(),
            "fields": fields,
            "models_read": len(ledger),
            "exact_duplicate_columns_all665": duplicates,
            "duplicate_removal_not_in_this_trial": True,
            "importance_does_not_establish_harm": True,
            "new_fits": 0,
        },
    )
    return files


def prepare(root):
    if root.resolve() != ROOT.resolve():
        raise ValueError("FIXED_ROOT_REQUIRED")
    if (root / "prepared.json").exists():
        c.verify_files(c.io.read(root / "source-manifest.json"))
        return
    assert c.io.read(root / "protection-summary.json")["complete"]
    c.verify_files(c.io.read(PRIOR / "training-freeze.json"))
    rows = c.lines(PRIOR / "features-r4/inputs.jsonl")
    nav = t.labels_nav()
    paths = screen(root, rows)
    calendar = {y: t.calendar(rows, y) for y in ("2025", "2026")}
    for updates in calendar.values():
        for update in updates:
            validate_training(rows, update, nav)
    c.io.save_lines(root / "inputs.jsonl", rows)
    c.io.save(root / "nav.json", nav)
    c.io.save(root / "training-calendar.json", calendar)
    for year in ("2025", "2026"):
        for group in ("B0", "B1"):
            p = PRIOR / "predictions" / f"{year}-{group}.jsonl"
            dest = root / "references" / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as stream:
                stream.write(p.read_bytes())
            paths += [p, dest]
    paths += [
        PRIOR / "training-freeze.json",
        PRIOR / "features-r4/inputs.jsonl",
        PRIOR / "features-r4/event-lineage.jsonl",
        PRIOR / "features-r4/feature-protocol.json",
        PRIOR / "fit-ledger.jsonl",
        c.bodies.BUNDLE,
        c.OLD / "reserved-labels.json",
    ]
    c.register_sources(root, paths)
    c.io.save(
        root / "prepared.json",
        {
            "at": c.io.now(),
            "rows": len(rows),
            "by_year": dict(Counter(r["target"][:4] for r in rows)),
            "new_input_values": 0,
            "new_fits": 0,
        },
    )
    c.stage(
        root,
        "保护与特征初筛",
        "COMPLETE",
        ["feature-screen.json", "source-manifest.json"],
        ["F12与F14重复仅留证，本次不扩展删除候选"],
        "固定三组删减、测试并冻结",
    )


def freeze(root):
    if (root / "training-freeze.json").exists():
        c.verify_files(c.io.read(root / "training-freeze.json"))
        return
    assert c.io.read(root / "test-results.json")["passed"]
    protocol = {
        "at": c.io.now(),
        "scope": "ONE_DAY_FEATURE_ABLATION_F15_F20",
        "target": "NAV(U) versus NAV(D)",
        "as_of": "D 08:00 Asia/Shanghai",
        "cadence_sessions": 20,
        "recipe": t.RECIPE,
        "candidates": {g: {"removed_absolute_indices": DROPS[g], "business_columns": len(kept(g))} for g in CANDIDATES},
        "references": "reuse original B0 NAV15 and B1 NAV15+EVENT20 predictions; no refits with seed0",
        "primary": "2025, 3 candidates x13 updates, seed0; 39 fits",
        "selection": "2025 correct descending, Brier ascending, dimensions ascending, candidate ID ascending",
        "seed_validation": "only selected deletion versus unchanged FULL, seed1, 13 updates each; 26 fits",
        "diagnostic": "only selected seed0, 2026 10 updates; known history, never used to reselect",
        "replay": "selected seed0 first2025 update once; 1 fit",
        "planned_fits": 76,
        "maximum_fits": 80,
        "failures_and_retries_count": True,
        "no_budget_reset_by_revision": True,
        "probability_atol": 1e-12,
        "comparison": "same dates versus B1 measures removal effect; B0 remains true NAV baseline",
        "interval": "paired difference, moving blocks20, resamples2000, seed0,95%; exploratory not unseen",
        "support_check": "positive correct gain vsFULL in both2025 seeds and2026; >=3 quarters not worse; "
        "Brier not worse and DOWN recall drop<=5pp in each comparison",
        "historical_gate_vs_B0": "2025 extra>=5, >=3 quarters not worse, Brier not worse, DOWN recall drop<=5pp",
        "no_single_importance_causality": True,
        "auto_promotion": False,
        "public_requests": 0,
        "llm_requests": 0,
        "new_bodies": 0,
        "adopted": False,
    }
    c.io.save(root / "training-protocol.json", protocol)
    files = [
        root / n
        for n in (
            "inputs.jsonl",
            "nav.json",
            "training-calendar.json",
            "training-protocol.json",
            "source-manifest.json",
            "prepared.json",
            "feature-screen.json",
            "test-results.json",
        )
    ]
    files += [Path(p) for p in c.io.read(root / "source-manifest.json")["files"]]
    files += list(c.io.PY.glob("scripts/*fund_002112_one_day_ablation*_v1.py"))
    files += [Path(m.__file__) for m in (c, features, forward, t, t.reference, c.io, c.bodies, c.nav_inputs)]
    files += [c.io.PY / "scripts/fund_002112_signal_verify_v1.py"]
    for path in files.copy():
        if path.suffix == ".py":
            dest = root / "code-snapshot" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                with dest.open("xb") as stream:
                    stream.write(path.read_bytes())
            assert c.io.sha(path) == c.io.sha(dest)
            files.append(dest)
    c.io.save(
        root / "training-freeze.json",
        {
            "at": c.io.now(),
            "python": sys.version,
            "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scikit-learn", "joblib")},
            "files": {str(p.resolve()): c.io.sha(p) for p in set(files)},
        },
    )
    c.stage(root, "冻结", "COMPLETE", ["training-protocol.json", "training-freeze.json"], [], "执行固定76次拟合")


def fit(root, candidate, seed, update, rows, nav, replay=False):
    identifier = candidate + f"__S{seed}__" + update["id"] + ("__REPLAY" if replay else "")
    folder = root / "runs" / identifier
    if (folder / "complete.json").exists():
        complete = c.io.read(folder / "complete.json")
        assert c.io.sha(folder / "model.joblib") == complete["model_sha256"]
        assert c.io.sha(folder / "predictions.jsonl") == complete["predictions_sha256"]
        return c.lines(folder / "predictions.jsonl")
    validate_training(rows, update, nav)
    training, evaluation = ([rows[i] for i in update[key]] for key in ("training", "evaluate"))
    x, state = transform(training, candidate)
    y = t.reference.labels(training, nav)
    record = {
        "candidate": candidate,
        "seed": seed,
        "update": update,
        "replay": replay,
        "training_digest": c.io.digest(training),
        "labels_digest": c.io.digest(y),
        "fit_shape": list(x.shape),
        "freeze_sha256": c.io.sha(root / "training-freeze.json"),
        "retry_of": None,
    }
    c.reserve(root, "fit", identifier, record)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        model = RandomForestClassifier(**{**t.RECIPE, "random_state": seed})
        with threadpool_limits(limits=2):
            model.fit(x, y)
        saved = {
            "candidate": candidate,
            "seed": seed,
            "model": model,
            "preprocessing": state,
            "kept_indices": kept(candidate),
            "training_digest": record["training_digest"],
        }
        probs = probabilities(saved, evaluation)
        predictions = [
            {
                "target": row["target"],
                "as_of": row["as_of"],
                "generated_at": c.io.now(),
                "model_id": identifier,
                "probabilities": p.tolist(),
                "predicted": c.CLASSES[int(np.argmax(p))],
                "historical_reconstruction": True,
            }
            for row, p in zip(evaluation, probs, strict=True)
        ]
        joblib.dump(saved, folder / "model.joblib")
        c.io.save_lines(folder / "predictions.jsonl", predictions)
        c.io.save(
            folder / "complete.json",
            {
                "at": c.io.now(),
                "state": "COMPLETE",
                "model_sha256": c.io.sha(folder / "model.joblib"),
                "predictions_sha256": c.io.sha(folder / "predictions.jsonl"),
            },
        )
        print(c.io.canonical({"fit": identifier, "count": len(c.lines(root / "fit-ledger.jsonl"))}), flush=True)
        return predictions
    except Exception as exc:
        c.io.save(folder / "failed.json", {"at": c.io.now(), "type": type(exc).__name__, "message": str(exc)})
        raise


def comparison(candidate, reference, truth):
    if [p["target"] for p in candidate] != [p["target"] for p in reference]:
        raise ValueError("COMMON_DATES_REQUIRED")
    def good(p):
        return p["predicted"] == truth[p["target"]]

    diffs = [int(good(a)) - int(good(b)) for a, b in zip(candidate, reference, strict=True)]
    return {
        "extra_correct": sum(diffs),
        "accuracy_gain": sum(diffs) / len(diffs),
        "wrong_to_right": sum(x == 1 for x in diffs),
        "right_to_wrong": sum(x == -1 for x in diffs),
        "paired_block_ci95": forward.bootstrap_difference(diffs),
    }


def scores(predictions, rows, nav, reference="B1"):
    truth = dict(zip([r["target"] for r in rows], t.reference.labels(rows, nav), strict=True))
    result = t.score(predictions, rows, nav)
    for group, ps in predictions.items():
        result[group]["versus_full_events"] = comparison(ps, predictions[reference], truth)
        result[group]["versus_nav_baseline"] = comparison(ps, predictions["B0"], truth)
    return result


def choose(result):
    """2025选出的唯一方案只用于后续复核，不能把入选等同晋级或采用。"""
    return min(CANDIDATES, key=lambda g: (-result[g]["correct"], result[g]["brier"], len(kept(g)), g))


def support(candidate, reference):
    down = candidate["recall"]["DOWN"], reference["recall"]["DOWN"]
    gates = {
        "more_correct": candidate["correct"] > reference["correct"],
        "three_quarters_not_worse": sum(
            candidate["quarters"][q]["correct"] >= r["correct"] for q, r in reference["quarters"].items()
        )
        >= 3,
        "brier_not_worse": candidate["brier"] <= reference["brier"],
        "down_recall_protected": None not in down and down[0] >= down[1] - 0.05,
    }
    return {"gates": gates, "passed": all(gates.values())}


def execute(root):
    if root.resolve() != ROOT.resolve():
        raise ValueError("FIXED_ROOT_REQUIRED")
    freeze(root)
    rows, nav = c.lines(root / "inputs.jsonl"), c.io.read(root / "nav.json")
    updates = c.io.read(root / "training-calendar.json")
    predictions = {g: c.lines(root / "references" / f"2025-{g}.jsonl") for g in ("B0", "B1")}
    for candidate in CANDIDATES:
        predictions[candidate] = [p for update in updates["2025"] for p in fit(root, candidate, 0, update, rows, nav)]
        c.io.save_lines(root / "predictions" / f"2025-{candidate}-S0.jsonl", predictions[candidate])
    primary = scores(predictions, rows, nav)
    c.io.save(root / "scores-2025-primary.json", primary)
    if not (root / "selection.json").exists():
        c.io.save(
            root / "selection.json",
            {
                "at": c.io.now(),
                "selected_for_verification": choose(primary),
                "selected_using": "2025_ONLY",
                "adopted": False,
                "future_observation_auto_start": False,
            },
        )
    selected = c.io.read(root / "selection.json")["selected_for_verification"]
    assert selected == choose(primary)
    c.stage(root, "2025固定删减", "COMPLETE", ["scores-2025-primary.json", "selection.json"], [], "固定种子1复核")
    c.verify_files(c.io.read(root / "training-freeze.json"))
    if not (root / "validation-entry.json").exists():
        c.io.save(
            root / "validation-entry.json", {"at": c.io.now(), "selection_sha256": c.io.sha(root / "selection.json")}
        )
    stability = {"B0": predictions["B0"]}
    for candidate in ("FULL", selected):
        key = "B1" if candidate == "FULL" else selected
        stability[key] = [p for update in updates["2025"] for p in fit(root, candidate, 1, update, rows, nav)]
        c.io.save_lines(root / "predictions" / f"2025-{candidate}-S1.jsonl", stability[key])
    secondary = scores(stability, rows, nav)
    c.io.save(root / "scores-2025-seed1.json", secondary)
    c.stage(root, "随机种子稳定性", "COMPLETE", ["scores-2025-seed1.json"], [], "唯一已选方案的2026历史诊断")
    c.verify_files(c.io.read(root / "training-freeze.json"))
    if not (root / "diagnostic-entry.json").exists():
        c.io.save(
            root / "diagnostic-entry.json",
            {
                "at": c.io.now(),
                "selection_sha256": c.io.sha(root / "selection.json"),
                "previously_exposed_history": True,
                "not_an_unseen_test": True,
            },
        )
    diagnostic = {g: c.lines(root / "references" / f"2026-{g}.jsonl") for g in ("B0", "B1")}
    diagnostic[selected] = [p for update in updates["2026"] for p in fit(root, selected, 0, update, rows, nav)]
    c.io.save_lines(root / "predictions" / f"2026-{selected}-S0.jsonl", diagnostic[selected])
    ds = scores(diagnostic, rows, nav)
    c.io.save(root / "scores-2026-diagnostic.json", ds)
    replay = fit(root, selected, 0, updates["2025"][0], rows, nav, replay=True)
    np.testing.assert_allclose(
        [p["probabilities"] for p in replay],
        [p["probabilities"] for p in predictions[selected][: len(replay)]],
        atol=1e-12,
        rtol=0,
    )
    checks = {
        "2025_seed0": support(primary[selected], primary["B1"]),
        "2025_seed1": support(secondary[selected], secondary["B1"]),
        "2026_seed0": support(ds[selected], ds["B1"]),
    }
    baseline_gates = support(primary[selected], primary["B0"])
    baseline_gates["gates"]["at_least_five_more"] = primary[selected]["correct"] >= primary["B0"]["correct"] + 5
    baseline_gates["passed"] = all(baseline_gates["gates"].values())
    decision = {
        "selected": selected,
        "removal_support": checks,
        "consistent_support_across_checks": all(v["passed"] for v in checks.values()),
        "historical_gate_vs_nav": baseline_gates,
        "confirmed_universal_harm": False,
        "future_observation_auto_start": False,
        "adopted": False,
        "limit": "已见历史的条件性删减实验；随机种子复核不是未见测试，其他字段未逐项消融。",
    }
    c.io.save(root / "decision.json", decision)
    c.verify_files(c.io.read(root / "training-freeze.json"))
    c.stage(
        root, "历史诊断与重放", "COMPLETE", ["scores-2026-diagnostic.json", "decision.json"], [], "独立回读与保护验收"
    )
    print(c.io.canonical(decision), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "train"))
    args = parser.parse_args()
    with c.io.writer_lock(ROOT), c.io.offline_guard():
        {"prepare": prepare, "freeze": freeze, "train": execute}[args.action](ROOT)


if __name__ == "__main__":
    main()
