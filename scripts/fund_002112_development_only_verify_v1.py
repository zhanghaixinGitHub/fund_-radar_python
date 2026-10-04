"""开发期续研的独立读回：逐模型重算概率、训练统计量、权重和选择，不调用fit。"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import fund_002112_development_only_search_v1 as research

io, ROOT = research.io, research.ROOT
CLASSES = ("DOWN", "FLAT", "UP")


def truth(rows, nav):
    """独立比较单位净值，先拒绝越界日期，保留FLAT类别而非强行归涨或跌。"""
    values = []
    for row in rows:
        if row["target"] > "2025-12-31" or row["base"] > "2025-12-31":
            raise AssertionError("FUTURE_LABEL")
        a, b = Decimal(nav[row["base"]]["unit_nav"]), Decimal(nav[row["target"]]["unit_nav"])
        values.append("UP" if b > a else "DOWN" if b < a else "FLAT")
    return values


def verify() -> dict:
    research.access_guard()
    research.verify_freeze()
    protocol = io.read(ROOT / "protocol.json")
    completion = io.read(ROOT / "completion.json")
    rows, nav = io.lines(ROOT / "inputs.jsonl"), io.read(ROOT / "nav-through-2025.json")
    assert max(nav) <= "2025-12-31" and max(r["target"] for r in rows) <= "2025-12-31"
    splits = io.read(ROOT / "splits.json")
    boundaries = {
        "V1": ("2025-01-02", "2025-06-30"),
        "V2": ("2025-07-01", "2025-09-30"),
        "V3": ("2025-10-09", "2025-12-31"),
    }
    for fold, (start, end) in boundaries.items():
        er = [i for i, r in enumerate(rows) if start <= r["target"] <= end]
        cutoff = rows[er[0]]["as_of"]
        tr = [i for i, r in enumerate(rows) if r["target"] < start and r["label_mature_at"] < cutoff]
        assert er == splits[fold]["evaluation"] and tr == splits[fold]["training"]
        assert not set(tr) & set(er)
    ledger = io.lines(ROOT / "fit-ledger.jsonl")
    assert len(ledger) <= 120 and len(ledger) == completion["actual_fits"]
    assert len({r["id"] for r in ledger}) == len(ledger)
    assert all(r["at"] > protocol["at"] for r in ledger)
    selected = io.read(ROOT / "selection.json")
    fixed = [
        {"id": f"FIXED_{g}_{r}", "group": g, "recipe": r, "window": None, "half_life": None}
        for g, r in protocol["old_fixed_recipes"]
    ]
    definitions = [(c, fold, False) for c in protocol["candidates"] + fixed for fold in boundaries]
    definitions.append((selected["winner"]["candidate"], "V3", True))
    models, total_predictions, alias_refs = set(), 0, 0
    candidate_scores, fixed_preds = {}, {}
    for candidate, fold, replay in definitions:
        rid = candidate["id"] + "__" + fold + ("__replay" if replay else "")
        run = io.read(ROOT / "runs" / rid / "complete.json")
        alias_refs += run["is_alias"]
        tr = [rows[i] for i in splits[fold]["training"]]
        if candidate["window"] is not None:
            tr = tr[-candidate["window"] :]
        er = [rows[i] for i in splits[fold]["evaluation"]]
        fields = protocol["input_groups"][candidate["group"]]
        x = np.asarray([[v for group in fields for v in r["groups"][group]] for r in tr], dtype=float)
        xe = np.asarray([[v for group in fields for v in r["groups"][group]] for r in er], dtype=float)
        w = None
        if candidate["half_life"] is not None:
            age = np.asarray([tr[-1]["session_index"] - r["session_index"] for r in tr], dtype=float)
            w = np.exp2(-age / candidate["half_life"])
            w /= w.mean()
        identity = {
            "train_dates": [r["target"] for r in tr],
            "evaluation_dates": [r["target"] for r in er],
            "x": x.tolist(),
            "xe": xe.tolist(),
            "y": truth(tr, nav),
            "weights": None if w is None else w.tolist(),
            "recipe": candidate["recipe"],
        }
        assert io.digest(identity) == run["signature"]
        model_path, prediction_path = Path(run["model_path"]), Path(run["predictions_path"])
        assert io.sha(model_path) == run["model_sha256"]
        assert io.sha(prediction_path) == run["predictions_sha256"]
        model = joblib.load(model_path)
        imputer, scaler, learner = (model[k] for k in ("imputer", "scaler", "model"))
        np.testing.assert_allclose(imputer.statistics_, np.nanmedian(x, axis=0), atol=1e-13, rtol=0)
        imputed = imputer.transform(x)
        np.testing.assert_allclose(scaler.mean_, imputed.mean(axis=0), atol=1e-13, rtol=0)
        np.testing.assert_allclose(scaler.var_, imputed.var(axis=0), atol=1e-13, rtol=0)
        saved = io.lines(prediction_path)
        assert [p["target"] for p in saved] == [r["target"] for r in er]
        with threadpool_limits(limits=2):
            probabilities = learner.predict_proba(scaler.transform(imputer.transform(xe)))
        aligned = np.zeros((len(er), 3))
        for j, label in enumerate(learner.classes_):
            aligned[:, CLASSES.index(label)] = probabilities[:, j]
        for p, probs in zip(saved, aligned, strict=True):
            np.testing.assert_allclose(p["probabilities"], probs, atol=1e-13, rtol=0)
            assert p["predicted"] == CLASSES[int(np.argmax(probs))]
        models.add(str(model_path))
        total_predictions += len(saved)
        actual = truth(er, nav)
        correct = sum(p["predicted"] == y for p, y in zip(saved, actual, strict=True))
        if candidate["id"].startswith("FIXED_"):
            fixed_preds.setdefault(fold, []).append(aligned)
        elif not replay:
            candidate_scores.setdefault(candidate["id"], []).append((correct, len(actual)))
    assert len(candidate_scores) == 36
    results = io.read(ROOT / "candidate-results.json")
    for result in results:
        checks = candidate_scores[result["candidate"]["id"]]
        assert sum(c for c, _ in checks) == result["score"]["correct"]
        assert sum(n for _, n in checks) == result["score"]["dates"] == 243
        assert [c for c, _ in checks] == [v["correct"] for v in result["fold_scores"]]
        assert min(c / n for c, n in checks) == result["min_fold_accuracy"]
    ranked = sorted(
        results, key=lambda r: (-r["score"]["correct"], -r["min_fold_accuracy"], r["features"], r["candidate"]["id"])
    )
    assert ranked == selected["ranking"] and ranked[0] == selected["winner"]
    baselines = io.read(ROOT / "baselines.json")
    for fold in boundaries:
        tr = [rows[i] for i in splits[fold]["training"]]
        er = [rows[i] for i in splits[fold]["evaluation"]]
        actual, training = truth(er, nav), truth(tr, nav)
        majority = max(CLASSES, key=lambda c: (training.count(c), c))
        assert actual.count("UP") == baselines[fold]["always_up"]["correct"]
        assert actual.count(majority) == baselines[fold]["majority"]["correct"]
        probs = np.mean(fixed_preds[fold], axis=0)
        predicted = [CLASSES[int(i)] for i in np.argmax(probs, axis=1)]
        assert (
            sum(a == b for a, b in zip(actual, predicted, strict=True))
            == baselines[fold]["fixed_recipes_after_quarantine"]["correct"]
        )
    expected_runs = {c["id"] + "__" + f + ("__replay" if rep else "") for c, f, rep in definitions}
    assert {p.parent.name for p in (ROOT / "runs").glob("*/complete.json")} == expected_runs
    result = {
        "at": io.now(),
        "actual_fits": len(ledger),
        "run_references_verified": len(definitions),
        "unique_model_files_loaded": len(models),
        "alias_references": alias_refs,
        "probabilities_recomputed_including_aliases": total_predictions,
        "all36_candidate_scores_and_ranking_verified": True,
        "fold_dates_weights_medians_scalers_verified": True,
        "2026_access_guard_enabled": True,
        "historical_first_seen_proven": False,
        "audit_script_sha256": io.sha(Path(__file__)),
    }
    io.save(ROOT / "independent-verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
