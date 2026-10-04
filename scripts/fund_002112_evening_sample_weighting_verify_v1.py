"""独立核对加权训练时序、真实权重、预处理、模型概率及历史比较；不调用fit。"""

from __future__ import annotations

import bisect
import subprocess
from pathlib import Path

import joblib
import numpy as np

from scripts import fund_002112_evening_sample_weighting_v1 as m
from scripts.fund_002112_evening_verify_v2 import manual_probabilities

old = m.old


def report_at(reports: list, cutoff: str):
    eligible = [r for r in reports if r["fund_code"] == "002112" and r["available_at"] <= cutoff]
    return sorted(eligible, key=lambda r: (r["report_end"], r["available_at"]))[-1] if eligible else None


def vector(report):
    if report is None:
        return None
    selected = sorted(report["holdings"], key=lambda x: int(x["reported_rank"]))[:10]
    total = sum(float(h["nav_weight_pct"]) for h in selected)
    return {h["stock_code"]: float(h["nav_weight_pct"]) / total for h in selected} if total else None


def expected_weights(policy: str, rows: list, cutoff: str, source: dict):
    """另一套实现重建权重与披露时点，禁止复用训练器权重计算函数。"""
    current = report_at(source["reports"], cutoff)
    current_vector = vector(current)
    overlaps, reports, ages = [], [], []
    anchor = sum(d <= cutoff[:10] for d in source["sessions"]) - 1
    for row in rows:
        report = report_at(source["reports"], row["as_of"])
        sample_vector = vector(report)
        reports.append(report)
        overlaps.append(
            None
            if current_vector is None or sample_vector is None
            else sum(min(current_vector.get(code, 0), w) for code, w in sample_vector.items())
        )
        ages.append(anchor - bisect.bisect_left(source["sessions"], row["target"]))
    if policy == "RECENT":
        raw = [2 ** (-age / 126) for age in ages]
    else:
        known = [1 + 3 * s for s in overlaps if s is not None]
        neutral = sum(known) / len(known) if known else 1
        raw = [neutral if s is None else 1 + 3 * s for s in overlaps]
    mean = sum(raw) / len(raw)
    return np.array([v / mean for v in raw]), overlaps, reports, ages, current


def preservation(root: Path) -> dict:
    snapshot = old.read(root / "protection-before.json")
    checked = 0
    for repo, state in snapshot.items():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == state["head"]
        for file, digest in state["files"].items():
            assert Path(file).is_file() and old.sha(Path(file)) == digest, file
            checked += 1
    m.verify_freeze(root)
    return {
        "prior_workspace_files": checked,
        "repositories_head_unchanged": len(snapshot),
        "prior_artifacts": len(old.read(root / "prior-artifacts.json")["files"]),
        "unchanged": True,
    }


def verify(root: Path) -> dict:
    m.verify_freeze(root)
    joint, history, baseline, source = m.inputs()
    lookup = {r["target"]: r for r in joint}
    oldpred = {r["target"]: r for r in old.lines(old.ROOT / "historical-predictions.jsonl")}
    preds = old.lines(root / "predictions.jsonl")
    ledger = old.lines(root / "fit-ledger.jsonl")
    assert len(ledger) == len({e["id"] for e in ledger}) == 138
    assert len(preds) == len({(r["policy"], r["target"]) for r in preds}) == 3 * 424
    recomputed, maximum_error, count, bootstrap_trees = {}, 0.0, 0, 0
    for entry in ledger:
        fit = old.read(root / "fits" / (entry["id"] + ".json"))
        path = root / "models" / (entry["id"] + ".joblib")
        assert old.sha(path) == fit["model_sha256"]
        saved = joblib.load(path)
        branch, policy, cutoff = entry["branch"], entry["policy"], entry["cutoff"]
        pool = joint if branch == "information" else history
        train = [
            r for r in pool if branch in r["groups"] and r["target"] < cutoff[:10] and r["label_mature_at"] < cutoff
        ]
        assert [r["target"] for r in train] == entry["training_targets"] == fit["training_targets"]
        assert all(r["as_of"] <= cutoff and r["label_mature_at"] < cutoff for r in train)
        assert saved["cutoff"] == cutoff and saved["row_count"] == len(train)
        prefix = entry["id"].removeprefix(policy + "_").removesuffix(branch)
        reference_path = m.baseline_path(prefix, branch)
        reference = joblib.load(reference_path)
        reference_fit = old.read(reference_path.parent.parent / "fits" / (reference_path.stem + ".json"))
        assert reference_fit["training_targets"] == entry["training_targets"]
        assert reference_fit["evaluation_targets"] == fit["evaluation_targets"]
        assert reference["cutoff"] == cutoff
        assert all(saved["model"].get_params()[key] == value for key, value in old.RECIPE.items())
        for key in ("keep", "median"):
            assert np.array_equal(saved["preprocessing"][key], reference["preprocessing"][key])
        matrix = np.array([r["groups"][branch] for r in train], dtype=float)
        keep = ~np.isnan(matrix).all(axis=0)
        assert np.array_equal(keep, saved["preprocessing"]["keep"])
        assert np.array_equal(np.nanmedian(matrix[:, keep], axis=0), saved["preprocessing"]["median"])
        weight_path = Path(entry["weight_file"])
        assert old.sha(weight_path) == entry["weight_sha256"] == saved["weight_sha256"]
        proof = old.read(weight_path)
        expected, overlaps, reports, ages, current = expected_weights(policy, train, cutoff, source)
        recorded = np.array([r["weight"] for r in proof["rows"]])
        assert np.allclose(expected, recorded, atol=1e-12, rtol=1e-12)
        assert np.allclose(expected, saved["model"]._sample_weight, atol=1e-12, rtol=1e-12)
        assert proof["current_report_hash"] == (current["raw"]["sha256"] if current else None)
        for row, saved_row, similarity, report, age in zip(train, proof["rows"], overlaps, reports, ages, strict=True):
            assert saved_row["target"] == row["target"] and saved_row["age_sessions"] == age
            assert (similarity is None and saved_row["top10_overlap"] is None) or (
                similarity is not None and abs(similarity - saved_row["top10_overlap"]) < 1e-12
            )
            if report:
                assert saved_row["sample_report_hash"] == report["raw"]["sha256"]
                assert report["available_at"] <= row["as_of"]
            elif policy == "SIMILAR":
                assert abs(saved_row["weight"] - 1) < 1e-12
        # 实际安装版本用权重控制bootstrap。逐棵树独立重建抽样并核对根节点类别分布，
        # 证明权重确实进入训练，而不是仅写进回执或只影响最后的展示。
        forest = saved["model"]
        labels = np.array([r["label"] for r in train])
        for tree, drawn in zip(forest.estimators_, forest.estimators_samples_, strict=True):
            independent = np.random.RandomState(tree.random_state).choice(
                len(train), forest._n_samples_bootstrap, replace=True, p=recorded / recorded.sum()
            )
            assert np.array_equal(drawn, independent)
            proportions = np.array([(labels[drawn] == c).mean() for c in forest.classes_])
            assert np.allclose(proportions, tree.tree_.value[0, 0], atol=1e-12, rtol=0)
            bootstrap_trees += 1
        evaluation = [lookup[d] for d in fit["evaluation_targets"]]
        assert min(r["as_of"] for r in evaluation) == cutoff
        assert all(r["label_mature_at"] > r["as_of"] for r in evaluation)
        computed = manual_probabilities(saved, evaluation, branch)
        error = float(np.abs(computed - np.array(fit["probabilities"])).max())
        assert error < 1e-12
        maximum_error = max(maximum_error, error)
        for row, prob in zip(evaluation, computed, strict=True):
            recomputed[(policy, row["target"], branch)] = prob
        count += len(evaluation)
    # 原方案69个模型也实际回读，三组均在相同424天比较。
    for prefix in dict.fromkeys(r["fit_prefix"] for r in baseline):
        rows = [lookup[r["target"]] for r in baseline if r["fit_prefix"] == prefix]
        for branch in m.WEIGHTS:
            path = m.baseline_path(prefix, branch)
            saved = joblib.load(path)
            probs = manual_probabilities(saved, rows, branch)
            for row, prob in zip(rows, probs, strict=True):
                recomputed[("ALL", row["target"], branch)] = prob
            count += len(rows)
    for row in preds:
        assert row["as_of"] == lookup[row["target"]]["as_of"]
        assert row["label"] == lookup[row["target"]]["label"]
        branches = {}
        for branch in m.WEIGHTS:
            p = recomputed[(row["policy"], row["target"], branch)]
            delta = float(np.abs(p - row["branches"][branch]).max())
            assert delta < 1e-12
            maximum_error = max(maximum_error, delta)
            branches[branch] = p
        value = 0.4 * branches["information"] + 0.4 * branches["market"] + 0.2 * branches["history"]
        assert np.allclose(value, row["probabilities"], atol=1e-12, rtol=0)
        assert np.isfinite(value).all() and abs(value.sum() - 1) < 1e-12
        assert oldpred[row["target"]]["label"] == row["label"]
    scores = old.read(root / "scores.json")
    pairs = old.read(root / "paired-comparisons.json")
    for period, groups in scores.items():
        subsets = {
            p: sorted(
                [r for r in preds if r["policy"] == p and m.m.matches(r["target"], period)], key=lambda r: r["target"]
            )
            for p in m.POLICIES
        }
        for policy, rows in subsets.items():
            y = np.array([old.CLASSES.index(r["label"]) for r in rows])
            p = np.array([r["probabilities"] for r in rows])
            correct = p.argmax(axis=1) == y
            assert int(correct.sum()) == groups[policy]["correct"] and len(rows) == groups[policy]["n"]
            assert abs(np.square(p - np.eye(3)[y]).sum(axis=1).mean() - groups[policy]["brier"]) < 1e-12
            if policy != "ALL":
                base = np.array([np.argmax(r["probabilities"]) for r in subsets["ALL"]]) == y
                pair = pairs[period][policy]
                assert pair["corrected"] == int((correct & ~base).sum())
                assert pair["spoiled"] == int((base & ~correct).sum())
                assert pair["net_correct"] == int(correct.sum() - base.sum())
    result = {
        "passed": True,
        "at": old.now(),
        "new_fits": 138,
        "new_models_read_back": 138,
        "reused_models_read_back": 69,
        "branch_probability_rows": count,
        "weighted_bootstrap_trees_checked": bootstrap_trees,
        "maximum_probability_error": maximum_error,
        "same_training_dates_and_preprocessing": True,
        "time_checks_passed": True,
        "preservation": preservation(root),
        "verifier_sha256": old.sha(Path(__file__)),
    }
    old.write(root / "independent-verification.json", result)
    print(old.json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    verify(m.ROOT)
