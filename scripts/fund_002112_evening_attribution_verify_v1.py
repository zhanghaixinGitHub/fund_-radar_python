"""独立回读四组拆分和融合实验：模型、时间、概率、选择顺序及旧产物保护。"""

from __future__ import annotations

import argparse
import bisect
import math
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np

from scripts import fund_002112_evening_attribution_v1 as m
from scripts.fund_002112_evening_verify_v2 import manual_probabilities

old = m.old


def independent_metric(rows, probabilities):
    truth = np.array([old.CLASSES.index(r["label"]) for r in rows])
    values = np.array(probabilities)
    guess = values.argmax(axis=1)
    correct = int((guess == truth).sum())
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "brier": float(np.square(values - np.eye(3)[truth]).sum(axis=1).mean()),
        "down_recall": float((guess[truth == 0] == 0).mean()),
    }


def check_metric(reported, rows, probabilities):
    expected = independent_metric(rows, probabilities)
    for key, value in expected.items():
        assert math.isclose(reported[key], value, rel_tol=0, abs_tol=1e-12), (key, reported[key], value)


def verify(root: Path):
    m.verify_hashes(root)
    source = old.read(old.ROOT / "sources.json")
    datasets = {g: old.lines(root / (g + "-inputs.jsonl")) for g in m.GROUPS}
    lookup = {g: {r["target"]: r for r in rows} for g, rows in datasets.items()}
    previous_inputs = {g: old.lines(m.prior.ROOT / (g + "-inputs.jsonl")) for g in ("B_LINK", "C_EVENT")}
    base, both = previous_inputs["B_LINK"], previous_inputs["C_EVENT"]
    cnames = m.prior.feature_names("C_EVENT")
    retained = [i for i, name in enumerate(cnames) if name not in m.REACTION_FIELDS]
    reaction = [cnames.index(name) for name in m.REACTION_FIELDS]
    for group, rows in datasets.items():
        assert len(rows) == 665
        for i, row in enumerate(rows):
            a, b = base[i], both[i]
            assert {k: v for k, v in row.items() if k != "groups"} == {k: v for k, v in a.items() if k != "groups"}
            assert row["groups"]["market"] == a["groups"]["market"]
            assert row["groups"]["history"] == a["groups"]["history"]
            info_a, info_b = a["groups"]["information"], b["groups"]["information"]
            expected = {
                "CONTENT": info_a,
                "TIMING": [info_b[j] for j in retained],
                "REACTION": info_a + [info_b[j] for j in reaction],
                "BOTH": info_b,
            }[group]
            assert row["groups"]["information"] == expected
            assert (
                row["as_of"] == (date.fromisoformat(row["target"]) - timedelta(days=1)).isoformat() + "T23:00:00+08:00"
            )
            assert row["label_mature_at"] > row["as_of"]
    evidence = {r["target"]: r for r in old.lines(root / "quality-evidence.jsonl")}
    proofs = {p["target"]: p for p in old.lines(m.prior.ROOT / "feature-lineage.jsonl")}
    companies = {e["id"]: e for e in source["company_events"]}
    evenings = old.prediction_evenings(source["sessions"])
    for row in base:
        saved, proof = evidence[row["target"]], proofs[row["target"]]
        assert saved["as_of"] == row["as_of"]
        weights = {}
        for event in proof["company_events"]:
            src = companies[event["id"]]
            assert src["version_available_at"] <= row["as_of"]
            age = source["sessions"].index(row["base"]) - bisect.bisect_left(evenings, src["version_available_at"])
            valid = any(
                (v := src["numeric"].get(field)) and v.get("value") is not None and v.get("quote")
                for field in ("revenue_yoy", "profit_yoy", "order_ratio")
            )
            if src["status"] == "QUALIFIED" and age in (0, 1) and valid:
                weights[src["code"]] = max(weights.get(src["code"], 0), event["weight"])
        coverage = sum(weights.values())
        state = (
            "STRONG" if coverage >= 0.03 else "CONTEXT" if proof["company_events"] or proof["public_events"] else "NONE"
        )
        assert saved["state"] == state and saved["strong_coverage"] == coverage
        assert saved == m.quality(row, proof, companies, source["sessions"])
    selection = old.read(root / "selection.json")
    ledger = old.lines(root / "fit-ledger.jsonl")
    expected_fit_count = 46 if selection["group"] == "BOTH" else 48
    assert len(ledger) == expected_fit_count <= m.MAX_FITS
    assert len({e["id"] for e in ledger}) == len(ledger)
    assert max(e["at"] for e in ledger if "2025_" in e["id"]) <= selection["at"]
    assert min(e["at"] for e in ledger if "2026_" in e["id"]) >= selection["at"]
    probabilities, max_error, new_rows = {}, 0.0, 0
    for entry in ledger:
        group = next(g for g in m.GROUPS if g in entry["id"])
        fit = old.read(root / "fits" / (entry["id"] + ".json"))
        path = root / "models" / (entry["id"] + ".joblib")
        assert old.sha(path) == fit["model_sha256"]
        training = [
            r for r in datasets[group] if r["label_mature_at"] < entry["cutoff"] and r["target"] < entry["cutoff"][:10]
        ]
        assert entry["training_targets"] == fit["training_targets"] == [r["target"] for r in training]
        model = joblib.load(path)
        matrix = np.array([r["groups"]["information"] for r in training], dtype=float)
        keep = ~np.isnan(matrix).all(axis=0)
        assert np.array_equal(keep, model["preprocessing"]["keep"])
        assert np.allclose(np.nanmedian(matrix[:, keep], axis=0), model["preprocessing"]["median"], atol=0, rtol=0)
        assert model["row_count"] == len(training)
        evaluation = [lookup[group][d] for d in fit["evaluation_targets"]]
        if evaluation:
            values = manual_probabilities(model, evaluation, "information")
            delta = float(np.abs(values - fit["probabilities"]).max())
            assert delta <= 1e-12
            max_error = max(max_error, delta)
            new_rows += len(evaluation)
            for row, p in zip(evaluation, values, strict=True):
                probabilities[(str(path), row["target"])] = p
    reused = old.read(root / "reused-models.json")["files"]
    old_inputs = {r["target"]: r for r in old.lines(old.ROOT / "dataset.jsonl")}
    baseline = {r["target"]: r for r in old.lines(old.ROOT / "historical-predictions.jsonl")}
    prior_predictions = {(r["group"], r["target"]): r for r in old.lines(m.prior.ROOT / "predictions.jsonl")}
    reused_rows, reused_historical = 0, 0
    for raw_path, digest in reused.items():
        path = Path(raw_path)
        assert old.sha(path) == digest
        model = joblib.load(path)
        if path.stem.startswith("FULL_"):
            continue
        fit = old.read(path.parent.parent / "fits" / (path.stem + ".json"))
        assert fit["model_sha256"] == digest
        if path.parent.parent == m.prior.ROOT:
            group = "CONTENT" if path.stem.startswith("B_LINK_") else "BOTH"
            evaluation = [lookup[group][d] for d in fit["evaluation_targets"]]
        else:
            evaluation = [old_inputs[d] for d in fit["evaluation_targets"]]
        values = manual_probabilities(model, evaluation, fit["branch"])
        assert np.allclose(values, fit["probabilities"], atol=1e-12, rtol=0)
        for row, p in zip(evaluation, values, strict=True):
            probabilities[(str(path), row["target"])] = p
            if fit["branch"] != "information":
                assert np.allclose(p, baseline[row["target"]]["branches"][fit["branch"]], atol=1e-12, rtol=0)
        reused_rows += len(evaluation)
        reused_historical += 1
    assert reused_historical == 92
    factorial = old.lines(root / "factorial-predictions.jsonl")
    assert len(factorial) == 4 * 424
    indexed = {(r["group"], r["target"]): r for r in factorial}
    for row in factorial:
        ref = baseline[row["target"]]
        assert row["label"] == ref["label"] and row["as_of"] == ref["as_of"]
        info = probabilities[(row["model_path"], row["target"])]
        assert np.allclose(info, row["information"], atol=1e-12, rtol=0)
        expected = (
            0.55 * info + 0.30 * np.array(ref["branches"]["market"]) + 0.15 * np.array(ref["branches"]["history"])
        )
        assert np.allclose(expected, row["probabilities"], atol=1e-12, rtol=0)
        if row["group"] in m.REUSED:
            prior = prior_predictions[(m.REUSED[row["group"]], row["target"])]
            assert np.allclose(prior["probabilities"], row["probabilities"], atol=1e-12, rtol=0)
    weighted = old.lines(root / "weight-predictions.jsonl")
    assert len(weighted) == len(m.POLICIES) * 424
    for row in weighted:
        ref = baseline[row["target"]]
        group_row = indexed[(selection["group"], row["target"])]
        q = evidence[row["target"]]
        alpha = (
            {"STRONG": 0.55, "CONTEXT": 0.25, "NONE": 0}[q["state"]]
            if row["policy"] == "QUALITY_GATE"
            else {"FIXED_25": 0.25, "FIXED_40": 0.40, "FIXED_55": 0.55, "FIXED_70": 0.70}[row["policy"]]
        )
        expected = (
            alpha * np.array(group_row["information"])
            + (1 - alpha) * (2 * np.array(ref["branches"]["market"]) + np.array(ref["branches"]["history"])) / 3
        )
        assert np.allclose(expected, row["probabilities"], atol=1e-12, rtol=0)
        assert row["weights"]["information"] == alpha and row["evidence_state"] == q["state"]
    factorial_scores = old.read(root / "factorial-scores.json")
    weight_scores = old.read(root / "weight-scores.json")
    scores = old.read(root / "scores.json")
    for period in scores:
        for group, reported in factorial_scores[period].items():
            rows = [r for r in factorial if r["group"] == group and m.matches(r["target"], period)]
            check_metric(reported, rows, [r["probabilities"] for r in rows])
        for policy, reported in weight_scores[period].items():
            rows = [r for r in weighted if r["policy"] == policy and m.matches(r["target"], period)]
            check_metric(reported, rows, [r["probabilities"] for r in rows])
        for method, reported in scores[period].items():
            rows = [r for r in baseline.values() if m.matches(r["target"], period)]
            if method == "SELECTED":
                rows = [r for r in weighted if r["policy"] == selection["policy"] and m.matches(r["target"], period)]
                values = [r["probabilities"] for r in rows]
            elif method in ("A_FIX", "PRIOR_C"):
                key = "A_FIX" if method == "A_FIX" else "C_EVENT"
                values = [prior_predictions[(key, r["target"])]["probabilities"] for r in rows]
            else:
                key = {"PRICE": "price_history", "NAV": "history_only", "ALWAYS_UP": "always_up"}[method]
                values = [r[key] for r in rows]
            check_metric(reported, rows, values)
    assert selection["representation_scores_2025"] == factorial_scores["2025"]
    assert selection["weight_scores_2025"] == weight_scores["2025"]
    assert selection["group"] == m.choose(factorial_scores["2025"], {g: len(m.feature_names(g)) for g in m.GROUPS})
    assert selection["policy"] == m.choose(weight_scores["2025"], {p: int(p == "QUALITY_GATE") for p in m.POLICIES})
    bundle = joblib.load(root / "002112-evening-attribution.joblib")
    assert bundle["information_group"] == selection["group"] and bundle["fusion_policy"] == selection["policy"]
    assert bundle["research_only"] and not bundle["production_eligible"] and not bundle["calibrated"]
    sample = datasets[selection["group"]]
    branches = {branch: manual_probabilities(model, sample, branch) for branch, model in bundle["experts"].items()}
    actual = m.predict_bundle(bundle, sample, evidence)
    for i, row in enumerate(sample):
        q = evidence[row["target"]]
        alpha = m.weights_for(selection["policy"], q)["information"]
        expected = (
            alpha * branches["information"][i] + (1 - alpha) * (2 * branches["market"][i] + branches["history"][i]) / 3
        )
        assert np.allclose(expected, actual[i], atol=1e-12, rtol=0)
    counts = {}
    for branch in bundle["experts"]:
        path = (
            Path(bundle["full_information_origin"])
            if branch == "information"
            else old.ROOT / "models" / ("FULL_" + branch + ".joblib")
        )
        assert np.allclose(
            manual_probabilities(joblib.load(path), sample, branch), branches[branch], atol=1e-12, rtol=0
        )
        counts[branch] = bundle["experts"][branch]["row_count"]
    replay_error = None
    if selection["group"] != "BOTH":
        replay = joblib.load(root / "models" / ("REPLAY_" + selection["group"] + ".joblib"))
        replay_error = float(
            np.abs(manual_probabilities(replay, sample[-8:], "information") - branches["information"][-8:]).max()
        )
        assert replay_error <= 1e-12
    decision = old.read(root / "decision.json")
    assert decision["historical_gate"] == m.prior.gate(scores, "SELECTED")
    assert decision["actual_new_fits"] == len(ledger) and not decision["adopted"]
    before = old.read(root / "protection-before.json")
    changed = [
        path
        for repo in before.values()
        for path, digest in repo["files"].items()
        if not Path(path).exists() or old.sha(Path(path)) != digest
    ]
    assert not changed, changed
    after = old.git_snapshot()
    assert all(after[repo]["head"] == item["head"] for repo, item in before.items())
    result = {
        "passed": True,
        "at": old.now(),
        "new_fits_read_back": len(ledger),
        "reused_historical_models_read_back": reused_historical,
        "reused_full_models_read_back": len(reused) - reused_historical,
        "new_probability_rows": new_rows,
        "reused_probability_rows": reused_rows,
        "factorial_rows": len(factorial),
        "weight_rows": len(weighted),
        "dataset_rows_reconstructed": 4 * 665,
        "quality_rows_verified": len(evidence),
        "full_predictions_verified": len(sample),
        "maximum_probability_error": max_error,
        "replay_error": replay_error,
        "final_training_rows": counts,
        "protected_workspace_files": sum(len(r["files"]) for r in before.values()),
        "old_workspace_changes": changed,
        "old_artifact_hashes_unchanged": True,
        "selection_uses_2026": False,
        "formal_admission": False,
    }
    old.write(root / "independent-verification.json", result)
    print(old.json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=m.ROOT)
    verify(parser.parse_args().root)
