"""逐预测回查更新模型、全部合格成熟标签和训练内预处理，独立回算开发成绩。"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import fund_002112_update_frequency_v1 as run

io, ROOT = run.io, run.ROOT


def verify():
    """只读全部模型；不调用fit或借用已保存正确数代替独立算分。"""
    run.prior.engine.access_guard()
    run.verify_freeze()
    rows = io.lines(ROOT / "inputs.jsonl")
    nav = io.read(ROOT / "nav-through-2025.json")
    assert max(nav) <= "2025-12-31" and max(r["target"] for r in rows) <= "2025-12-31"
    cal = io.read(ROOT / "update-calendar.json")
    dev = [r for r in rows if r["target"].startswith("2025")]
    truth = {}
    for r in rows:
        before, after = Decimal(nav[r["base"]]["unit_nav"]), Decimal(nav[r["target"]]["unit_nav"])
        truth[r["target"]] = "UP" if after > before else "DOWN" if after < before else "FLAT"
    predictions_count = 0
    run_records = {}
    for path in sorted((ROOT / "runs").glob("*/complete.json")):
        record = io.read(path)
        update = cal["updates"][record["update"]]
        cutoff = dev[update["start"]]["as_of"]
        eligible = [i for i, r in enumerate(rows) if r["as_of"] < cutoff and r["label_mature_at"] < cutoff]
        assert eligible == update["training"]
        assert cutoff == update["cutoff_exclusive"]
        assert max(rows[i]["label_mature_at"] for i in eligible) == update["max_training_label_mature_at"] < cutoff
        assert io.sha(Path(record["model_path"])) == record["model_sha256"]
        assert io.sha(Path(record["predictions_path"])) == record["predictions_sha256"]
        model = joblib.load(record["model_path"])
        tr = np.asarray([rows[i]["groups"]["N"] + rows[i]["groups"]["NE"] for i in eligible])
        er = [rows[i] for i in update["evaluation"]]
        xe = np.asarray([r["groups"]["N"] + r["groups"]["NE"] for r in er])
        assert tr.shape[1] == xe.shape[1] == 15
        np.testing.assert_allclose(model["imputer"].statistics_, np.median(tr, axis=0), atol=1e-13, rtol=0)
        np.testing.assert_allclose(model["scaler"].mean_, tr.mean(axis=0), atol=1e-13, rtol=0)
        np.testing.assert_allclose(model["scaler"].var_, tr.var(axis=0), atol=1e-13, rtol=0)
        assert model["model"].get_params() == run.prior.engine.kernel.estimator(record["recipe"]).get_params()
        with threadpool_limits(limits=2):
            values = model["model"].predict_proba(model["scaler"].transform(model["imputer"].transform(xe)))
        aligned = np.zeros((len(er), 3))
        for j, label in enumerate(model["model"].classes_):
            aligned[:, run.prior.engine.kernel.CLASSES.index(label)] = values[:, j]
        saved = io.lines(Path(record["predictions_path"]))
        assert len(saved) == len(er)
        for r, p, probability in zip(er, saved, aligned, strict=True):
            assert p["target"] == r["target"] and p["as_of"] == r["as_of"]
            assert p["max_training_label_mature_at"] < p["model_cutoff"] <= p["as_of"]
            assert p["max_training_label_mature_at"] == update["max_training_label_mature_at"]
            np.testing.assert_allclose(p["probabilities"], probability, atol=1e-13, rtol=0)
            assert p["predicted"] == run.prior.engine.kernel.CLASSES[int(np.argmax(probability))]
        predictions_count += len(saved)
        run_records[record["id"]] = {p["target"]: p for p in saved}
    results = io.read(ROOT / "candidate-results.json")
    old_splits = io.read(run.prior.ROOT / "splits.json")
    for result in results:
        candidate = result["candidate"]
        predictions = io.lines(ROOT / "development-predictions" / (candidate["id"] + ".jsonl"))
        mapping = cal["prediction_maps"][candidate["id"]]
        assert len(predictions) == len(mapping) == len(dev) == 243
        for index, (p, m, r) in enumerate(zip(predictions, mapping, dev, strict=True)):
            elapsed = r["session_index"] - dev[0]["session_index"]
            assert elapsed == index
            anchor = elapsed - elapsed % candidate["cadence"]
            expected_id = f"{candidate['recipe']}__U{anchor:03d}"
            assert p["model_id"] == m["model_id"] == expected_id
            assert p == run_records[expected_id][r["target"]]
        correct = sum(p["predicted"] == truth[p["target"]] for p in predictions)
        assert result["score"]["correct"] == correct
        assert result["score"]["accuracy"] == correct / 243
        for label in run.prior.engine.kernel.CLASSES:
            cls = result["score"]["classes"][label]
            assert cls["actual"] == sum(truth[r["target"]] == label for r in dev)
            assert cls["correct"] == sum(p["predicted"] == truth[p["target"]] == label for p in predictions)
        for fold, score in zip(run.prior.engine.FOLDS, result["fold_scores"], strict=True):
            expected = set(old_splits[fold]["evaluation_dates"])
            assert score["correct"] == sum(
                p["predicted"] == truth[p["target"]] for p in predictions if p["target"] in expected
            )
        assert result["min_fold_accuracy"] == min(s["accuracy"] for s in result["fold_scores"])
    ranking = sorted(
        results,
        key=lambda r: (
            -r["score"]["correct"],
            -r["min_fold_accuracy"],
            -r["candidate"]["cadence"],
            r["candidate"]["id"],
        ),
    )
    assert ranking == io.read(ROOT / "selection.json")["ranking"]
    refs = io.read(ROOT / "baseline-and-reuse.json")
    bases = {
        "N": [p for path in refs["N"] for p in io.lines(Path(path))],
        "FIXED_NNE": io.lines(Path(refs["fixed_nne"])),
        "ALWAYS_UP": [{"target": r["target"], "predicted": "UP"} for r in dev],
    }
    winner = io.lines(ROOT / "development-predictions" / (ranking[0]["candidate"]["id"] + ".jsonl"))
    comparisons = {}
    for name, base in bases.items():
        assert [p["target"] for p in base] == [r["target"] for r in dev]
        correct = sum(p["predicted"] == truth[p["target"]] for p in base)
        assert correct == refs["expected_correct"][name]
        differences = [
            int(w["predicted"] == truth[w["target"]]) - int(b["predicted"] == truth[b["target"]])
            for w, b in zip(winner, base, strict=True)
        ]
        comparisons[name] = {
            "correct": correct,
            "net_correct_days": sum(differences),
            "interval": run.prior.engine.kernel.impact.stats.block_interval(
                differences, [r["session_index"] for r in dev]
            ),
        }
    io.save(
        ROOT / "baseline-comparison.json",
        {"at": io.now(), "post_selection_descriptive_only": True, "comparisons": comparisons},
    )
    ledger = io.lines(ROOT / "fit-ledger.jsonl")
    assert len(ledger) == 37 and sum(r["replay"] for r in ledger) == 1
    plan = io.read(ROOT / "protocol.json")
    assert all(r["at"] > plan["at"] and r["supervised_fits"] == 1 for r in ledger)
    assert {r["id"] for r in ledger} == {k for k in run_records if "__U000" not in k}
    sources = io.read(ROOT / "event-audit-source-manifest.json")
    assert all(io.sha(Path(p)) == digest for p, digest in sources["files"].items())
    assert all(io.sha(Path(v["copy"])) == v["sha256"] for v in sources["code"].values())
    env = io.read(ROOT / "environment.json")
    assert all(io.sha(Path(v["copy"])) == v["sha256"] for v in env["code"].values())
    oldproof = io.read(run.prior.ROOT / "final-verification.json")
    assert all(io.sha(run.prior.ROOT / p) == digest for p, digest in oldproof["artifacts"].items())
    outcome = {
        "at": io.now(),
        "models_verified": len(run_records),
        "new_supervised_fits": len(ledger),
        "predictions_recomputed": predictions_count,
        "candidate_prediction_maps_verified": 1458,
        "source_audit_files_verified": len(sources["files"]),
        "old_final_artifacts_preserved": len(oldproof["artifacts"]),
        "all_scores_ranking_and_time_boundaries_match": True,
        "new_2026_scores": None,
        "strict_historical_first_seen_proven": False,
        "script_sha256": io.sha(Path(__file__)),
    }
    io.save(ROOT / "independent-verification.json", outcome)
    return outcome


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
