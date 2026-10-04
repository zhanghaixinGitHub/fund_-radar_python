"""独立读回新实验：复核模型文件、逐日概率、训练折统计量、来源及代码保护边界。"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from decimal import Decimal

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import fund_002112_all_information_search_v1 as search
from scripts import fund_002112_holding_impact_fit_v1 as impact
from scripts import fund_002112_holding_impact_v1 as event

io = event.io


def truth(rows, nav):
    """不调用训练代码的标签函数，独立按原协议的净值精确小数比较。"""
    result = []
    for row in rows:
        before = Decimal(nav[row["base"]]["unit_nav"])
        after = Decimal(nav[row["target"]]["unit_nav"])
        result.append("UP" if after > before else "DOWN" if after < before else "FLAT")
    return result


def check_probabilities(model, scaler, x, saved, imputer=None):
    if not saved:
        return 0
    with threadpool_limits(limits=2):
        probabilities = model.predict_proba(scaler.transform(imputer.transform(x) if imputer else x))
    for row, probs in zip(saved, probabilities, strict=True):
        if isinstance(row["probabilities"], dict):
            recorded = [row["probabilities"][label] for label in model.classes_]
        else:
            recorded = [row["probabilities"][search.CLASSES.index(label)] for label in model.classes_]
        if not np.allclose(probs, recorded, atol=1e-13, rtol=0):
            raise AssertionError("SAVED_PROBABILITY_DIFFERS_FROM_MODEL")
        if row["predicted"] != model.classes_[int(np.argmax(probs))]:
            raise AssertionError("SAVED_DIRECTION_DIFFERS_FROM_MODEL")
    return len(saved)


def audit_search(partial=False, root=None):
    root = root or search.ROOT
    search.check_freeze(root)
    rows, splits = search.input_rows(root), io.read(root / "splits.json")
    if io.read(root / "plan.json").get("version") == "TRUE_NEXT_TRADING_DAY_V1":
        morning = {r["target"]: r for r in io.lines(search.ROOT / "inputs.jsonl")}
        for row in rows:
            if (
                row["as_of"] != row["base"] + "T08:00:00+08:00"
                or row["base"] >= row["target"]
                or row["groups"] != morning[row["base"]]["groups"]
            ):
                raise AssertionError("NEXT_DAY_USED_TARGET_DAY_INFORMATION")
        for split in splits.values():
            if split["evaluation"] and split["cutoff_exclusive"] != rows[split["evaluation"][0]]["as_of"]:
                raise AssertionError("NEXT_DAY_TRAINING_CUTOFF_WRONG")
    nav = io.read(event.OLD / "snapshot/nav-facts.json")
    records = io.lines(root / "fit-ledger.jsonl")
    if not partial and not (root / "completion.json").exists():
        raise ValueError("SEARCH_NOT_COMPLETE")
    predictions_checked, models_checked, scores = 0, 0, {}
    for record in records:
        folder = root / "models" / record["id"]
        if not (folder / "complete.json").exists():
            if partial:
                continue
            raise AssertionError("FIT_WITHOUT_COMPLETION")
        complete = io.read(folder / "complete.json")
        if io.sha(folder / "model.joblib") != complete["model_sha256"]:
            raise AssertionError("MODEL_BYTES_CHANGED")
        model = joblib.load(folder / "model.joblib")
        split = splits[record["fold"]]
        train_rows = [rows[i] for i in split["training"]]
        test_rows = [rows[i] for i in split["evaluation"]]
        if any(r["label_mature_at"] >= split["cutoff_exclusive"] for r in train_rows):
            raise AssertionError("FUTURE_TRAINING_LABEL")
        if set(split["training"]) & set(split["evaluation"]):
            raise AssertionError("TRAIN_EVAL_OVERLAP")
        # 独立组装训练矩阵，并读回各折中位数，防止把后段样本用于补值/标准化。
        names = search.GROUPS[model["group"]]
        x = np.asarray([[v for name in names for v in row["groups"][name]] for row in train_rows], dtype=float)
        digest = io.digest([[float(v) if np.isfinite(v) else None for v in r] for r in x])
        if digest != record["input_sha256"]:
            raise AssertionError("FIT_INPUT_CHANGED")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(x, axis=0)
        if not np.allclose(median, model["imputer"].statistics_, equal_nan=True):
            raise AssertionError("IMPUTER_NOT_TRAINING_ONLY")
        transformed = model["imputer"].transform(x)
        if not np.allclose(np.mean(transformed, axis=0), model["scaler"].mean_, atol=1e-12):
            raise AssertionError("SCALER_NOT_TRAINING_ONLY")
        predicted = io.lines(folder / "predictions.jsonl")
        if [p["target"] for p in predicted] != split["evaluation_dates"]:
            raise AssertionError("SCORE_DATES_CHANGED")
        if test_rows:
            z = np.asarray([[v for name in names for v in row["groups"][name]] for row in test_rows], dtype=float)
            predictions_checked += check_probabilities(model["model"], model["scaler"], z, predicted, model["imputer"])
            actual = truth(test_rows, nav)
            scores[record["id"]] = {
                "correct": sum(a == p["predicted"] for a, p in zip(actual, predicted, strict=True)),
                "dates": len(actual),
            }
        models_checked += 1
    result = {
        "at": io.now(),
        "partial": partial,
        "models_verified": models_checked,
        "predictions_recomputed": predictions_checked,
        "train_only_preprocessors_verified": True,
        "scores_recomputed": scores,
    }
    if not partial:
        selection = io.read(root / "selection.json")
        for candidate in selection["all_development"]:
            if candidate["failed"]:
                continue
            base = candidate["group"] + "__" + candidate["recipe"] + "__"
            if sum(scores[base + fold]["correct"] for fold in ("V1", "V2", "V3")) != candidate["score"]["correct"]:
                raise AssertionError("DEVELOPMENT_SCORE_MISMATCH")
        audit = io.read(root / "audit-result.json")
        pred = io.lines(root / "selected-audit-predictions.jsonl")
        actual = truth([rows[i] for i in splits["T"]["evaluation"]], nav)
        if sum(a == p["predicted"] for a, p in zip(actual, pred, strict=True)) != audit["selected"]["correct"]:
            raise AssertionError("AUDIT_SCORE_MISMATCH")
        result["selected_audit_correct"] = audit["selected"]["correct"]
        if selection["2026_score_accessed_for_selection"]:
            raise AssertionError("ILLEGAL_SELECTION_SOURCE")
    io.save(root / ("independent-base-audit.json" if partial else "independent-final-audit.json"), result)
    return {k: v for k, v in result.items() if k != "scores_recomputed"}


def audit_event():
    root = event.ROOT
    docs = io.lines(root / "documents.jsonl")
    counts, warnings_found, usage = Counter(), Counter(), Counter()
    validated_hashes = {}
    for doc in docs:
        path = root / "validated-v4" / (doc["event_id"] + ".json")
        result = io.read(path)
        if result["event_id"] != doc["event_id"] or result["text_sha256"] != doc["text_sha256"]:
            raise AssertionError("EXTRACTION_SOURCE_IDENTITY_MISMATCH")
        if "raw" in result:
            again = event.validate(result["raw"], doc)
            if any(result[k] != v for k, v in again.items()):
                raise AssertionError("EXTRACTION_VALIDATION_NOT_REPRODUCIBLE")
        counts[result["status"]] += 1
        warnings_found.update(result.get("field_warnings", []))
        usage.update({k: v for k, v in result.get("usage", {}).items() if isinstance(v, int)})
        validated_hashes[str(path)] = io.sha(path)
    io.save(root / "validated-inventory.json", validated_hashes)
    for path, expected in io.read(root / "plan.json")["sources"].items():
        if io.sha(path) != expected:
            raise AssertionError("OLD_EXPERIMENT_SOURCE_CHANGED")
    rows = io.lines(root / "feature-rows.jsonl")
    splits = io.read(event.OLD / "split-manifest.json")
    models_checked, predictions_checked = 0, 0
    for record in io.lines(root / "fit-ledger.jsonl"):
        folder = root / "models" / record["name"]
        complete = io.read(folder / "complete.json")
        if io.sha(folder / "model.joblib") != complete["model_sha256"]:
            raise AssertionError("EVENT_MODEL_CHANGED")
        model = joblib.load(folder / "model.joblib")
        stage = record["name"].split("_")[0]
        split = splits[stage]
        tr = [rows[i] for i in split["train_indices"]]
        er = [rows[i] for i in split["evaluation_indices"]]
        if stage == "FINAL" and record["name"].endswith("replay"):
            er = tr[-180:]
        x = impact.numeric(tr, model["candidate"], model["timing"])
        if io.digest(x.tolist()) != record["input_sha256"]:
            raise AssertionError("EVENT_TRAINING_INPUT_CHANGED")
        if not np.allclose(np.mean(x, axis=0), model["scaler"].mean_, atol=1e-12):
            raise AssertionError("EVENT_SCALER_NOT_TRAINING_ONLY")
        saved = io.lines(folder / "predictions.jsonl")
        if [p["target"] for p in saved] != [r["target"] for r in er]:
            raise AssertionError("EVENT_EVALUATION_DATE_MISMATCH")
        if er:
            predictions_checked += check_probabilities(
                model["model"], model["scaler"], impact.numeric(er, model["candidate"], model["timing"]), saved
            )
        models_checked += 1
    for stage in ("C2026", "FINAL"):
        original = joblib.load(root / "models" / f"{stage}_aux_M3" / "model.joblib")
        replay = joblib.load(root / "models" / f"{stage}_aux_M3_replay" / "model.joblib")
        if not np.array_equal(original["model"].coef_, replay["model"].coef_):
            raise AssertionError("EVENT_REFIT_COEFFICIENTS_DIFFER")
    reports = {r["raw"]["sha256"]: r for r in io.read(root / "reports.json")}
    evidence_count = 0
    for row in io.lines(root / "holding-evidence.jsonl"):
        for link in row["links"]:
            report = reports[link["report_sha256"]]
            if report["available_at"] > row["target"] + "T15:00:00+08:00":
                raise AssertionError("FUTURE_HOLDING_REPORT")
            h = next(h for h in report["holdings"] if h["stock_code"] == link["stock_code"])
            if float(h["nav_weight_pct"]) != link["nav_weight_pct"]:
                raise AssertionError("HOLDING_WEIGHT_WRONG_DENOMINATOR")
            if link["relation"] != "ISSUER_CODE" and link["company_direction"] != "UNKNOWN":
                raise AssertionError("POLICY_SIGN_FORCED_ON_COMPANY")
            evidence_count += 1
    result = {
        "at": io.now(),
        "documents_verified": len(docs),
        "statuses": dict(counts),
        "field_warnings": dict(warnings_found),
        "usage": dict(usage),
        "models_verified": models_checked,
        "predictions_recomputed": predictions_checked,
        "holding_links_verified": evidence_count,
        "exact_refits": ["C2026_M3", "FINAL_M3"],
        "old_source_hashes_unchanged": True,
    }
    io.save(root / "independent-final-audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("base", "event", "search", "next"))
    args = parser.parse_args()
    if args.command == "next":
        from scripts.fund_002112_next_day_search_v1 import ROOT

        result = audit_search(root=ROOT)
    else:
        result = audit_event() if args.command == "event" else audit_search(partial=args.command == "base")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
