"""回读全部新模型和复用模型，核验时点、同日期对照、原文关联及旧产物保护。"""

from __future__ import annotations

import argparse
import bisect
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np

from scripts import fund_002112_evening_refine_v1 as m
from scripts.fund_002112_evening_verify_v2 import manual_probabilities

old = m.old


def period_matches(target, period):
    return period == "ALL" or (
        target.startswith(period[:4]) and (len(period) == 4 or (int(target[5:7]) - 1) // 3 + 1 == int(period[-1]))
    )


def verify(root: Path):
    m.check_freeze(root)
    source = old.read(old.ROOT / "sources.json")
    originals = old.lines(old.ROOT / "dataset.jsonl")
    original = {r["target"]: r for r in originals}
    previous = old.lines(old.ROOT / "historical-predictions.jsonl")
    old_predictions = {r["target"]: r for r in previous}
    datasets = {g: old.lines(root / (g + "-inputs.jsonl")) for g in m.GROUPS}
    lookup = {g: {r["target"]: r for r in rows} for g, rows in datasets.items()}
    proofs = old.lines(root / "feature-lineage.jsonl")
    expected_dates = [r["target"] for r in originals if "information" in r["groups"]]
    names = old.read(root / "feature-names.json")
    for group, rows in datasets.items():
        assert [r["target"] for r in rows] == expected_dates
        for row in rows:
            ref = original[row["target"]]
            assert {k: v for k, v in row.items() if k != "groups"} == {k: v for k, v in ref.items() if k != "groups"}
            assert row["groups"]["market"] == ref["groups"]["market"]
            assert row["groups"]["history"] == ref["groups"]["history"]
            assert len(row["groups"]["information"]) == len(names[group])
            if group == "A_FIX":
                assert all(
                    v == ref["groups"]["information"][i]
                    for i, v in enumerate(row["groups"]["information"])
                    if i not in (3, 6, 16, 18)
                )
            else:
                assert row["groups"]["information"][:20] == lookup["A_FIX"][row["target"]]["groups"]["information"][:20]
            expected = (date.fromisoformat(row["target"]) - timedelta(days=1)).isoformat() + "T23:00:00+08:00"
            assert row["as_of"] == expected and row["label_mature_at"] > row["as_of"]
    material = old.read(root / "verified-business-material.json")
    public = {e["id"]: e for e in material["public"]}
    profiles = defaultdict(list)
    for profile in sorted(material["profiles"], key=lambda p: (p["available_at"], p["document_id"])):
        profiles[profile["code"]].append(profile)
    evenings = old.prediction_evenings(source["sessions"])
    for pool, key in [(source["company_events"], "version_available_at"), (material["public"], "available_at")]:
        for event in pool:
            if event.get(key):
                event["evening_index"] = bisect.bisect_left(evenings, event[key])
    company = {e["id"]: e for e in source["company_events"]}
    reports = {r["raw"]["sha256"]: r for r in source["reports"]}
    # 从正文缓存独立核对业务引文和政策对象引文；不能只相信关联器给出的状态。
    documents = {}
    manifest = old.read(root / "supplement-source-manifest.json")["files"]
    for path in manifest:
        if path.endswith(".jsonl"):
            for doc in old.lines(Path(path)):
                if doc.get("event_id"):
                    documents[doc["event_id"]] = doc
    for profile in material["profiles"]:
        doc = documents[profile["document_id"]]
        assert m.clean(profile["quote"]) in m.clean(doc.get("text") or doc.get("body"))
    for event in material["public"]:
        text = m.clean(documents[event["id"]].get("text") or documents[event["id"]].get("body"))
        assert all(m.clean(q) in text for q in event["facts"])
        assert all(m.clean(t["quote"]) in text for t in event["targets"])
        if event["direction"] != "UNKNOWN":
            assert event["impact_quote"] and m.clean(event["impact_quote"]) in text
    event_links = 0
    for proof in proofs:
        row = original[proof["target"]]
        rebuilt, rebuilt_proof = m.revised_features(row, source, material, profiles)
        assert rebuilt_proof == proof
        for group in m.GROUPS:
            assert rebuilt[group] == lookup[group][row["target"]]["groups"]["information"]
        assert proof["as_of"] == row["as_of"]
        current = old.holding_weights(old.choose_report(source["reports"], row["as_of"]))
        for identifier in proof["profit_events"]:
            e = company[identifier]
            assert e["status"] == "QUALIFIED" and e["version_available_at"] <= row["as_of"]
            assert e["numeric"]["profit_yoy"]["period"] == e["period"]
        for event in proof["public_events"]:
            original_event = public[event["id"]]
            assert event["available_at"] <= row["as_of"] and 0 <= event["age"] < 20
            assert event["age"] == source["sessions"].index(row["base"]) - bisect.bisect_left(
                evenings, event["available_at"]
            )
            for repeated_id in original_event["repeated_source_ids"]:
                assert public[repeated_id]["available_at"] <= event["available_at"]
            for link in event["links"]:
                event_links += 1
                report = reports[link["report_hash"]]
                assert report["available_at"] <= event["available_at"]
                assert link["weight"] == min(old.holding_weights(report)[link["code"]], current[link["code"]])
                if link["basis"] != "ISSUER":
                    assert link["direction"] == "UNKNOWN"
                    assert link["profile"]["available_at"] <= event["available_at"]
                reaction = link["reaction"]
                assert all(event["available_at"][:10] <= day <= row["base"] for day in reaction["price_days"])
                if reaction["value"] is not None:
                    assert reaction["status"] == "OBSERVED_NOT_CAUSAL"
        for event in proof["company_events"]:
            assert event["available_at"] <= row["as_of"]
            assert all(day <= row["base"] for day in event["reaction"]["price_days"])
            if "buyback_delta" in event:
                assert event["buyback_delta"]["previous_available_at"] < event["available_at"]
        assert proof["extras"]["guidance_revision_delta"] is None
    probabilities, maximum_error, new_probability_rows = {}, 0.0, 0
    ledger = old.lines(root / "fit-ledger.jsonl")
    assert len(ledger) == 71 and len({e["id"] for e in ledger}) == 71
    selection = old.read(root / "selection.json")
    selected = selection["selected"]
    assert max(e["at"] for e in ledger if "2025_" in e["id"]) <= selection["at"]
    assert min(e["at"] for e in ledger if "2026_" in e["id"]) >= selection["at"]
    for entry in ledger:
        group = next(g for g in m.GROUPS if g in entry["id"])
        path = root / "models" / (entry["id"] + ".joblib")
        fit = old.read(root / "fits" / (entry["id"] + ".json"))
        assert old.sha(path) == fit["model_sha256"]
        training = [
            r for r in datasets[group] if r["label_mature_at"] < entry["cutoff"] and r["target"] < entry["cutoff"][:10]
        ]
        assert entry["training_targets"] == [r["target"] for r in training]
        assert fit["training_targets"] == entry["training_targets"]
        saved = joblib.load(path)
        assert saved["row_count"] == len(training) and saved["cutoff"] == entry["cutoff"]
        matrix = np.array([r["groups"]["information"] for r in training], dtype=float)
        keep = ~np.isnan(matrix).all(axis=0)
        assert np.array_equal(keep, saved["preprocessing"]["keep"])
        assert np.allclose(np.nanmedian(matrix[:, keep], axis=0), saved["preprocessing"]["median"], rtol=0, atol=0)
        evaluation = [lookup[group][d] for d in fit["evaluation_targets"]]
        if evaluation:
            p = manual_probabilities(saved, evaluation, "information")
            delta = float(np.abs(p - np.array(fit["probabilities"])).max())
            assert delta <= 1e-12
            maximum_error = max(maximum_error, delta)
            new_probability_rows += len(evaluation)
            for row, values in zip(evaluation, p, strict=True):
                probabilities[(entry["id"], row["target"])] = values
    # 价格、净值分支未重新拟合，逐个回读原模型以验证复用概率没有变化。
    reused = old.read(root / "reused-models.json")["files"]
    assert len(reused) == 46
    reused_probability_rows = 0
    for name, expected_hash in reused.items():
        path = Path(name)
        assert old.sha(path) == expected_hash
        fit = old.read(old.ROOT / "fits" / (path.stem + ".json"))
        saved = joblib.load(path)
        evaluation = [original[d] for d in fit["evaluation_targets"]]
        p = manual_probabilities(saved, evaluation, fit["branch"])
        assert np.allclose(p, fit["probabilities"], rtol=0, atol=1e-12)
        for row, values in zip(evaluation, p, strict=True):
            assert np.allclose(values, old_predictions[row["target"]]["branches"][fit["branch"]], atol=1e-12, rtol=0)
        reused_probability_rows += len(evaluation)
    predictions = old.lines(root / "predictions.jsonl")
    assert len(predictions) == 3 * len(previous)
    for row in predictions:
        ref = old_predictions[row["target"]]
        info = probabilities[(row["new_fit"], row["target"])]
        assert row["label"] == ref["label"]
        assert np.allclose(info, row["information"], atol=1e-12, rtol=0)
        combined = (
            0.55 * info + 0.30 * np.array(ref["branches"]["market"]) + 0.15 * np.array(ref["branches"]["history"])
        )
        assert np.allclose(combined, row["probabilities"], atol=1e-12, rtol=0)
    scores = old.read(root / "scores.json")
    for period, methods in scores.items():
        for group, reported in methods.items():
            if group in m.GROUPS:
                rows = [r for r in predictions if r["group"] == group and period_matches(r["target"], period)]
                values = np.array([r["probabilities"] for r in rows])
            else:
                rows = [r for r in previous if period_matches(r["target"], period)]
                field = {
                    "OLD": "weighted_55",
                    "PRICE": "price_history",
                    "NAV": "history_only",
                    "ALWAYS_UP": "always_up",
                }[group]
                values = np.array([r[field] for r in rows])
            actual = np.array([old.CLASSES.index(r["label"]) for r in rows])
            guess = values.argmax(axis=1)
            correct = int((guess == actual).sum())
            brier = float(np.square(values - np.eye(3)[actual]).sum(axis=1).mean())
            assert reported["n"] == len(rows) and reported["correct"] == correct
            assert math.isclose(reported["accuracy"], correct / len(rows), abs_tol=1e-12)
            assert math.isclose(reported["brier"], brier, abs_tol=1e-12)
            assert math.isclose(reported["down_recall"], float((guess[actual == 0] == 0).mean()), abs_tol=1e-12)
    expected_selection = min(
        ("B_LINK", "C_EVENT"),
        key=lambda g: (-scores["2025"][g]["correct"], scores["2025"][g]["brier"], len(names[g]), g),
    )
    assert selected == expected_selection
    package = joblib.load(root / "002112-evening-refined.joblib")
    sample = datasets[selected][-8:]
    final_counts = {}
    for branch in old.WEIGHTS:
        path = (
            root / "models" / ("FULL_" + selected + ".joblib")
            if branch == "information"
            else old.ROOT / "models" / ("FULL_" + branch + ".joblib")
        )
        saved = joblib.load(path)
        assert np.allclose(
            manual_probabilities(package["experts"][branch], sample, branch),
            manual_probabilities(saved, sample, branch),
            atol=1e-12,
            rtol=0,
        )
        final_counts[branch] = saved["row_count"]
    replay = joblib.load(root / "models" / ("REPLAY_" + selected + ".joblib"))
    replay_error = float(
        np.abs(
            manual_probabilities(replay, sample, "information")
            - manual_probabilities(package["experts"]["information"], sample, "information")
        ).max()
    )
    assert replay_error <= 1e-12
    decision = old.read(root / "decision.json")
    assert decision["historical_gate"] == m.gate(scores, selected)
    assert not decision["adopted"] and not decision["production_eligible"]
    assert package["research_only"] and not package["historical_first_seen_proven"]
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
        "candidate_dataset_rows_rebuilt": 3 * len(proofs),
        "new_models_read_back": len(ledger),
        "reused_historical_models_read_back": len(reused),
        "new_probability_rows": new_probability_rows,
        "reused_probability_rows": reused_probability_rows,
        "maximum_probability_error": maximum_error,
        "replay_error": replay_error,
        "public_time_weight_links": event_links,
        "verified_profiles": len(material["profiles"]),
        "final_training_rows": final_counts,
        "old_workspace_changes": changed,
        "protected_workspace_files": sum(len(r["files"]) for r in before.values()),
        "old_artifact_hashes_unchanged": True,
        "historical_first_seen_proven": False,
        "formal_admission": False,
        "new_fits": len(ledger),
    }
    old.write(root / "independent-verification.json", result)
    print(old.json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=m.ROOT)
    verify(parser.parse_args().root)
