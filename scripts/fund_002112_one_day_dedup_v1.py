"""002112一日模型第二轮删减：重复覆盖、事件数量与业绩事件类型权重。

仅运行预先固定的四组删除对照。复用完整模型参照，不改动旧实验、业务服务或现用模型。
2025用于选择一个复核对象，2026始终是已经看过答案的历史诊断。
"""

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

from scripts import fund_002112_one_day_ablation_v1 as prior

c, t = prior.c, prior.t
ROOT = c.io.RESEARCH / "one-day-dedup-optimization/20261002-v1"
PRIOR = prior.ROOT
# 前15列是净值；F10/F11是预告/实际业绩事件权重，F13是合格事件数，F14与F12重复。
# 四组是固定的2×2删除组合：先去重复列，再分别/同时去掉数量与业绩类型权重。
DROPS = {
    "DEDUP": [28],
    "DEDUP_NO_COUNT": [27, 28],
    "DEDUP_NO_EARNINGS_WEIGHT": [24, 25, 28],
    "DEDUP_NO_COUNT_EARNINGS_WEIGHT": [24, 25, 27, 28],
}
CANDIDATES = tuple(DROPS)
PLANNED_FITS = 76


def check_root(root):
    if Path(root).resolve() != ROOT.resolve():
        raise ValueError("FIXED_ROOT_REQUIRED")


def kept(candidate):
    return [i for i in range(35) if i not in DROPS[candidate]]


def matrix(rows, candidate):
    """只删除冻结的业务列；缺失保持NaN，保留字段原值不变。"""
    return np.asarray([r["groups"]["N"] + r["groups"]["NE"] + r["E"] for r in rows], dtype=float)[:, kept(candidate)]


def transform(rows, candidate, state=None):
    values = matrix(rows, candidate)
    if state is None:
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        filled = imputer.fit_transform(values)
        state = {"imputer": imputer, "scaler": StandardScaler().fit(filled)}
    return state["scaler"].transform(state["imputer"].transform(values)), state


def duplicate_evidence(rows):
    """同时检查数值和缺失；全空/常量两列不能冒充有意义的重复线索。"""
    if not rows or len({r["E"][11] for r in rows}) <= 1:
        raise ValueError("NO_VARIABLE_DUPLICATE_EVIDENCE")
    if any(r["E"][11] != r["E"][13] for r in rows):
        raise ValueError("F12_F14_NOT_IDENTICAL")
    return {"rows": len(rows), "duplicate": ["F12", "F14"], "values_and_missingness_identical": True}


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert c.io.sha(destination) == c.io.sha(source)
    else:
        with destination.open("xb") as stream:
            stream.write(source.read_bytes())


def prepare(root):
    check_root(root)
    if (root / "prepared.json").exists():
        c.verify_files(c.io.read(root / "source-manifest.json"))
        return
    assert c.io.read(root / "protection-summary.json")["complete"]
    c.verify_files(c.io.read(PRIOR / "training-freeze.json"))
    assert c.io.read(PRIOR / "completion.json")["fixed_historical_experiment_complete"]
    paths = [PRIOR / "training-freeze.json", PRIOR / "completion.json", PRIOR / "independent-verification.json"]
    for name in ("inputs.jsonl", "nav.json", "training-calendar.json"):
        copy_file(PRIOR / name, root / name)
        paths += [PRIOR / name, root / name]
    for name in ("2025-B0.jsonl", "2025-B1.jsonl", "2026-B0.jsonl", "2026-B1.jsonl"):
        copy_file(PRIOR / "references" / name, root / "references" / name)
        paths += [PRIOR / "references" / name, root / "references" / name]
    # 上轮已拟合的完整增强种子1可以逐模型核实复用，本轮不为参照重复耗费13次拟合。
    source = PRIOR / "predictions/2025-FULL-S1.jsonl"
    copy_file(source, root / "references/2025-B1-S1.jsonl")
    paths += [source, root / "references/2025-B1-S1.jsonl", PRIOR / "fit-ledger.jsonl"]
    ledger = [p for p in c.lines(PRIOR / "fit-ledger.jsonl") if p["candidate"] == "FULL" and p["seed"] == 1]
    assert len(ledger) == 13
    for attempt in ledger:
        folder = PRIOR / "runs" / attempt["id"]
        complete = c.io.read(folder / "complete.json")
        assert c.io.sha(folder / "model.joblib") == complete["model_sha256"]
        assert c.io.sha(folder / "predictions.jsonl") == complete["predictions_sha256"]
        paths += [folder / name for name in ("model.joblib", "predictions.jsonl", "complete.json")]
    # 明确保留原正文/持仓来源链。本轮不重新提取或修改事实，更不把今天持仓套回过去。
    paths += [prior.PRIOR / "features-r4" / name for name in ("event-lineage.jsonl", "feature-protocol.json")]
    rows, nav = c.lines(root / "inputs.jsonl"), c.io.read(root / "nav.json")
    for updates in c.io.read(root / "training-calendar.json").values():
        for update in updates:
            prior.validate_training(rows, update, nav)
    c.io.save(root / "duplicate-evidence.json", {"at": c.io.now(), **duplicate_evidence(rows)})
    c.register_sources(root, paths)
    c.io.save(root / "prepared.json", {"at": c.io.now(), "rows": len(rows), "new_values": 0, "new_fits": 0})
    c.stage(root, "保护与输入核对", "COMPLETE", ["source-manifest.json", "duplicate-evidence.json"], [], "测试并冻结")


def freeze(root):
    check_root(root)
    if (root / "training-freeze.json").exists():
        c.verify_files(c.io.read(root / "training-freeze.json"))
        return
    assert c.io.read(root / "test-results.json")["passed"]
    protocol = {
        "at": c.io.now(),
        "scope": "ONE_DAY_DEDUP_COUNT_EARNINGS_WEIGHT",
        "authorization": "用户在上轮交付后明确要求开始下一轮优化；本次新假设单独冻结，旧账本不变",
        "target": "NAV(U) versus NAV(D), U is next trading day",
        "as_of": "D 08:00 Asia/Shanghai",
        "recipe": t.RECIPE,
        "cadence_sessions": 20,
        "candidates": {g: {"removed_absolute_indices": DROPS[g], "business_columns": len(kept(g))} for g in CANDIDATES},
        "primary": "2025 seed0, four candidates x13 updates =52 fits",
        "selection": "correct descending, Brier ascending, business columns ascending, candidate ID ascending",
        "validation": "selected only,2025 seed1,13 fits; reuse verified prior FULL seed1 reference",
        "diagnostic": "selected only,2026 seed0,10 fits; previously exposed history, no reselection",
        "replay": "selected first2025 seed0 update once;1 fit",
        "planned_fits": PLANNED_FITS,
        "maximum_fits": 80,
        "failed_or_interrupted_fits_count": True,
        "automatic_retry": False,
        "no_adaptive_candidates_or_parameters": True,
        "references": "B0 NAV15; B1 complete NAV15+EVENT20; all unchanged and reused",
        "seed1_nav_reference": "seed0 B0 reused for context only; not a matched-seed B0 test",
        "group_effects": "2025 DEDUP vsFULL; count/earnings-weight/joint removals vsDEDUP; paired group differences",
        "interval": "paired moving blocks20,resamples2000,seed0,95%; exploratory, not unseen or multiplicity-adjusted",
        "support": "vsFULL:correct gain in both2025 seeds and2026;>=3 quarters no worse;Brier no worse;DOWN drop<=5pp",
        "nav_gate": "2025 seed0 >=B0+5 correct;>=3 quarters not worse;Brier not worse;DOWN drop<=5pp",
        "forest_sampling": "max_features=sqrt unchanged; removing columns also changes forest feature sampling",
        "probability_atol": 1e-12,
        "public_requests": 0,
        "llm_requests": 0,
        "new_bodies": 0,
        "automatic_promotion": False,
        "automatic_future_observation": False,
        "adopted": False,
    }
    c.io.save(root / "training-protocol.json", protocol)
    files = [
        root / n
        for n in (
            "training-protocol.json",
            "source-manifest.json",
            "prepared.json",
            "duplicate-evidence.json",
            "test-results.json",
        )
    ]
    files += [Path(p) for p in c.io.read(root / "source-manifest.json")["files"]]
    files += list(c.io.PY.glob("scripts/*fund_002112_one_day_dedup*_v1.py"))
    files += [
        Path(m.__file__)
        for m in (prior, c, t, prior.features, prior.forward, t.reference, c.io, c.bodies, c.nav_inputs)
    ]
    files += [c.io.PY / "scripts/fund_002112_signal_verify_v1.py"]
    for path in files.copy():
        if path.suffix == ".py":
            dest = root / "code-snapshot" / path.name
            copy_file(path, dest)
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
    c.stage(
        root, "固定方案冻结", "COMPLETE", ["training-protocol.json", "training-freeze.json"], [], "执行76次固定拟合"
    )


def fit(root, candidate, seed, update, rows, nav, replay=False):
    """先登记真实尝试，再拟合；失败留账并停止，不自动重试或覆盖已完成模型。"""
    check_root(root)
    identifier = candidate + f"__S{seed}__" + update["id"] + ("__REPLAY" if replay else "")
    folder = root / "runs" / identifier
    if (folder / "complete.json").exists():
        done = c.io.read(folder / "complete.json")
        assert c.io.sha(folder / "model.joblib") == done["model_sha256"]
        assert c.io.sha(folder / "predictions.jsonl") == done["predictions_sha256"]
        return c.lines(folder / "predictions.jsonl")
    prior.validate_training(rows, update, nav)
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
        values, _ = transform(evaluation, candidate, state)
        with threadpool_limits(limits=2):
            raw = model.predict_proba(values)
        aligned = np.zeros((len(evaluation), 3))
        for j, label in enumerate(model.classes_):
            aligned[:, c.CLASSES.index(label)] = raw[:, j]
        ps = [
            {
                "target": row["target"],
                "as_of": row["as_of"],
                "generated_at": c.io.now(),
                "model_id": identifier,
                "probabilities": p.tolist(),
                "predicted": c.CLASSES[int(np.argmax(p))],
                "historical_reconstruction": True,
            }
            for row, p in zip(evaluation, aligned, strict=True)
        ]
        joblib.dump(saved, folder / "model.joblib")
        c.io.save_lines(folder / "predictions.jsonl", ps)
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
        return ps
    except Exception as exc:
        c.io.save(folder / "failed.json", {"at": c.io.now(), "type": type(exc).__name__, "message": str(exc)})
        raise


def choose(scores):
    return min(CANDIDATES, key=lambda g: (-scores[g]["correct"], scores[g]["brier"], len(kept(g)), g))


def group_effects(predictions, rows, nav):
    """除完整模型参照外，报告去重基础上的组删除，避免把组合效果都归因于某一列。"""
    truth = dict(zip([r["target"] for r in rows], t.reference.labels(rows, nav), strict=True))
    pairs = {
        "dedup_vs_full": ("DEDUP", "B1"),
        "count_after_dedup": ("DEDUP_NO_COUNT", "DEDUP"),
        "earnings_weight_after_dedup": ("DEDUP_NO_EARNINGS_WEIGHT", "DEDUP"),
        "both_after_dedup": ("DEDUP_NO_COUNT_EARNINGS_WEIGHT", "DEDUP"),
        "count_after_earnings_weight_removal": ("DEDUP_NO_COUNT_EARNINGS_WEIGHT", "DEDUP_NO_EARNINGS_WEIGHT"),
        "earnings_weight_after_count_removal": ("DEDUP_NO_COUNT_EARNINGS_WEIGHT", "DEDUP_NO_COUNT"),
    }
    return {
        name: {"candidate": left, "reference": right, **prior.comparison(predictions[left], predictions[right], truth)}
        for name, (left, right) in pairs.items()
    }


def execute(root):
    check_root(root)
    freeze(root)
    rows, nav = c.lines(root / "inputs.jsonl"), c.io.read(root / "nav.json")
    updates = c.io.read(root / "training-calendar.json")
    predictions = {g: c.lines(root / "references" / f"2025-{g}.jsonl") for g in ("B0", "B1")}
    for candidate in CANDIDATES:
        predictions[candidate] = [p for u in updates["2025"] for p in fit(root, candidate, 0, u, rows, nav)]
        c.io.save_lines(root / "predictions" / f"2025-{candidate}-S0.jsonl", predictions[candidate])
    primary = prior.scores(predictions, rows, nav)
    c.io.save(root / "scores-2025-primary.json", primary)
    c.io.save(root / "group-effects-2025.json", group_effects(predictions, rows, nav))
    if not (root / "selection.json").exists():
        c.io.save(
            root / "selection.json",
            {
                "at": c.io.now(),
                "selected_for_verification": choose(primary),
                "selected_using": "2025_ONLY",
                "adopted": False,
            },
        )
    selected = c.io.read(root / "selection.json")["selected_for_verification"]
    assert selected == choose(primary)
    c.stage(root, "2025四组对照", "COMPLETE", ["scores-2025-primary.json", "selection.json"], [], "仅入选对象种子1复核")
    c.verify_files(c.io.read(root / "training-freeze.json"))
    if not (root / "validation-entry.json").exists():
        c.io.save(
            root / "validation-entry.json", {"at": c.io.now(), "selection_sha256": c.io.sha(root / "selection.json")}
        )
    secondary_ps = {"B0": predictions["B0"], "B1": c.lines(root / "references/2025-B1-S1.jsonl")}
    secondary_ps[selected] = [p for u in updates["2025"] for p in fit(root, selected, 1, u, rows, nav)]
    c.io.save_lines(root / "predictions" / f"2025-{selected}-S1.jsonl", secondary_ps[selected])
    secondary = prior.scores(secondary_ps, rows, nav)
    c.io.save(root / "scores-2025-seed1.json", secondary)
    c.stage(root, "种子复核", "COMPLETE", ["scores-2025-seed1.json"], [], "2026固定历史诊断")
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
    diagnostic[selected] = [p for u in updates["2026"] for p in fit(root, selected, 0, u, rows, nav)]
    c.io.save_lines(root / "predictions" / f"2026-{selected}-S0.jsonl", diagnostic[selected])
    ds = prior.scores(diagnostic, rows, nav)
    c.io.save(root / "scores-2026-diagnostic.json", ds)
    replay = fit(root, selected, 0, updates["2025"][0], rows, nav, replay=True)
    np.testing.assert_allclose(
        [p["probabilities"] for p in replay],
        [p["probabilities"] for p in predictions[selected][: len(replay)]],
        rtol=0,
        atol=1e-12,
    )
    checks = {
        "2025_seed0": prior.support(primary[selected], primary["B1"]),
        "2025_seed1": prior.support(secondary[selected], secondary["B1"]),
        "2026_seed0": prior.support(ds[selected], ds["B1"]),
    }
    nav_gate = prior.support(primary[selected], primary["B0"])
    nav_gate["gates"]["at_least_five_more"] = primary[selected]["correct"] >= primary["B0"]["correct"] + 5
    nav_gate["passed"] = all(nav_gate["gates"].values())
    decision = {
        "selected": selected,
        "removal_support": checks,
        "consistent_support_across_checks": all(v["passed"] for v in checks.values()),
        "historical_gate_vs_nav": nav_gate,
        "adopted": False,
        "automatic_future_observation": False,
        "confirmed_universal_harm": False,
        "limit": "已见历史的条件性删除效果；没有新事实或未来样本，不作普遍因果结论",
    }
    c.io.save(root / "decision.json", decision)
    c.verify_files(c.io.read(root / "training-freeze.json"))
    c.stage(root, "历史诊断与重放", "COMPLETE", ["decision.json"], [], "独立验收与结果交付")
    print(c.io.canonical(decision), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "train"))
    args = parser.parse_args()
    with c.io.writer_lock(ROOT), c.io.offline_guard():
        {"prepare": prepare, "freeze": freeze, "train": execute}[args.action](ROOT)


if __name__ == "__main__":
    main()
