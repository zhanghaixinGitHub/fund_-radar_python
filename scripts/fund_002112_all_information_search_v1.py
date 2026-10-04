"""002112 现有信息的固定范围优化：时间顺序开发选择，后续年份单次历史审计。

只读取已验收的本地冻结资料。2026结果以前已经看过，因此本批次不声称盲测，
也不把历史最优称为未来最优。搜索选择不读取本批次2026评分，不增删困难日期。
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts import fund_002112_existing_data_inputs_v1 as old
from scripts import fund_002112_holding_impact_fit_v1 as impact
from scripts import fund_002112_holding_impact_v1 as sources

io = sources.io
ROOT = io.RESEARCH / "all-information-optimization/20260930-v1"
GROUPS = {
    "N": ["N"],
    "NM": ["N", "M"],
    "NHMI": ["N", "H", "M", "I"],
    "NHMIF": ["N", "H", "M", "I", "F"],
    "NHMIF_COUNTS": ["N", "H", "M", "I", "F", "COUNTS"],
    "ALL_SEMANTICS": ["N", "H", "M", "I", "F", "COUNTS", "SEMANTICS"],
}
RECIPES = {
    "LR_C01": {"family": "LR", "C": 0.1},
    "LR_C1": {"family": "LR", "C": 1.0},
    "LR_C10": {"family": "LR", "C": 10.0},
    "HGB_D2": {"family": "HGB", "max_depth": 2, "l2_regularization": 5.0},
    "HGB_D3": {"family": "HGB", "max_depth": 3, "l2_regularization": 10.0},
    "RF_D4": {"family": "RF", "max_depth": 4, "min_samples_leaf": 10, "n_estimators": 200},
    "ET_D4": {"family": "ET", "max_depth": 4, "min_samples_leaf": 10, "n_estimators": 200},
    "ET_D8": {"family": "ET", "max_depth": 8, "min_samples_leaf": 20, "n_estimators": 300},
}
CLASSES = ("DOWN", "FLAT", "UP")


def prepare(root: Path = ROOT) -> dict:
    """先登记数据清单、全部48个候选和选择规则；不读取方向标签。"""
    if (root / "plan.json").exists():
        return io.read(root / "plan.json")
    pointer = io.read(sources.RECENT / "handoff/data/active-revision.json")
    source = sources.RECENT / pointer["inputs_path"]
    if io.sha(source) != pointer["inputs_sha256"]:
        raise ValueError("ACCEPTED_RECENT_INPUT_HASH_MISMATCH")
    data = io.read(source)
    io.save(root / "accepted-existing-inputs.json", data)
    tracked = [
        source,
        sources.ROOT / "documents.jsonl",
        sources.ROOT / "reports.json",
        sources.OLD / "events.jsonl",
        sources.OLD / "daily-inputs.jsonl",
        sources.OLD / "snapshot/nav-facts.json",
    ]
    plan = {
        "version": "ALL_INFORMATION_SEARCH_V1",
        "registered_at": io.now(),
        "authority": "用户继续完成请求及授权协调会话转达的现有全信息优化边界",
        "prediction_cutoff": "TARGET_TRADING_DATE_08:00_ASIA_SHANGHAI",
        "target": "SAME_TRADING_DATE_UNIT_NAV_VS_PREVIOUS_TRADING_DATE_UP_FLAT_DOWN",
        "sources": {str(p): io.sha(p) for p in tracked},
        "rows": len(data["rows"]),
        "date_start": data["rows"][0]["target_date"],
        "date_end": data["rows"][-1]["target_date"],
        "feature_groups": GROUPS,
        "recipes": RECIPES,
        "candidates": 48,
        "development_folds": ["V1", "V2", "V3"],
        "development_ranges": {k: old.EVAL_RANGE[k] for k in ("V1", "V2", "V3")},
        "audit_range": old.EVAL_RANGE["T"],
        "max_supervised_fits": 153,
        "selection": "POOLED_DEV_CORRECT_DESC_MIN_FOLD_ACCURACY_DESC_FEATURE_COUNT_ASC_ID_ASC",
        "ensemble": "TOP_THREE_DISTINCT_FAMILIES_EQUAL_PROBABILITY_MEAN; choose only if dev correct strictly higher",
        "test_access": "AFTER_SELECTION_JSON_FREEZE_ONLY; all chosen members and N_LR_C1 once",
        "final_refit": "CHOSEN_MEMBERS_ON_ALL_MATURE_EXISTING_ROWS; no deployment",
        "limitations": [
            "2026_ALREADY_SEEN_BEFORE_THIS_BATCH_NOT_A_NEW_BLIND_TEST",
            "FROZEN_EXISTING_DATA_ONLY_NO_NEW_MATERIALS",
            "BODY_COVERAGE_ONLY_TO_2023",
            "FULL_RAW_RECORD_AVAILABILITY_IS_NOT_COMPLETE_USABLE_FEATURE_COVERAGE",
        ],
        "previous_disabled_group_G": data["optional_decisions"],
        "future_label_access_for_selection": False,
        "production_changes": False,
    }
    io.save(root / "plan.json", plan)
    return plan


def build(root: Path = ROOT, base_only: bool = False) -> dict:
    """合并08:00前的现有输入；日期按原665行固定，不为特定模型删日期。"""
    plan = io.read(root / "plan.json")
    for path, expected in plan["sources"].items():
        if io.sha(path) != expected:
            raise ValueError("FROZEN_SOURCE_CHANGED")
    data = io.read(root / "accepted-existing-inputs.json")
    semantic = {r["target"]: r for r in io.lines(sources.ROOT / "feature-rows.jsonl")} if not base_only else {}
    original_rows = {r["target"]: r for r in io.lines(sources.OLD / "daily-inputs.jsonl")}
    rows = []
    for raw in data["rows"]:
        target = raw["target_date"]
        sem = semantic.get(target)
        groups = {k: raw["groups"][k] for k in ("N", "M", "H", "I", "F")}
        # 源数据保留null；训练算法的补值在各训练折拟合，同时保留缺失指示列。
        for group, values in groups.items():
            if values is None:
                groups[group] = [None] * len(data["group_columns"][group])
            elif any(v is not None and not np.isfinite(v) for v in values):
                raise ValueError("INVALID_INFINITE_BASE_FEATURE")
        groups["COUNTS"] = sem["morning"]["legacy_counts"] if sem else None
        groups["SEMANTICS"] = (
            (sem["context"] + sem["morning"]["quality"] + sem["morning"]["global"] + sem["morning"]["weighted"])
            if sem
            else None
        )
        if raw["as_of"] != target + "T08:00:00+08:00" or raw["report"]["available_at"] > raw["as_of"]:
            raise ValueError("MORNING_REPORT_CUTOFF_BROKEN")
        rows.append(
            {
                "target": target,
                "base": raw["base_date"],
                "as_of": raw["as_of"],
                "groups": groups,
                "label_mature_at": raw["label_mature_at"],
                "session_index": original_rows[target]["session_index"],
                "source_digest": raw["source_digest"],
                "body_semantics": sem["morning"]["valid_events"] if sem else None,
                "weighted_semantics": sem["morning"]["weighted_events"] if sem else None,
            }
        )
    splits = {}
    for fold in ("V1", "V2", "V3", "T", "FINAL"):
        cutoff = old.FIT_CUTOFF.get(fold, io.read(sources.ROOT / "plan.json")["created_at"])
        tr = [i for i, r in enumerate(rows) if r["target"] <= old.TRAIN_END[fold] and r["label_mature_at"] < cutoff]
        start, end = old.EVAL_RANGE.get(fold, ("", ""))
        er = [i for i, r in enumerate(rows) if start and start <= r["target"] <= end]
        if er and set(tr) & set(er):
            raise ValueError("TRAIN_TEST_OVERLAP")
        splits[fold] = {
            "training": tr,
            "evaluation": er,
            "cutoff_exclusive": cutoff,
            "train_dates": [rows[i]["target"] for i in tr],
            "evaluation_dates": [rows[i]["target"] for i in er],
        }
    input_file = root / ("base-inputs.jsonl" if base_only else "inputs.jsonl")
    io.save_lines(input_file, rows)
    io.save(root / "splits.json", splits)
    io.save(
        root / ("base-freeze.json" if base_only else "freeze.json"),
        {
            "at": io.now(),
            "files": {
                str(p): io.sha(p)
                for p in (
                    input_file,
                    root / "splits.json",
                    root / "plan.json",
                    Path(__file__),
                )
                + (() if base_only else (sources.ROOT / "feature-freeze.json",))
            },
        },
    )
    return {
        "dates": len(rows),
        "splits": {k: {"train": len(v["training"]), "evaluate": len(v["evaluation"])} for k, v in splits.items()},
    }


def matrix(rows: list[dict], group: str) -> np.ndarray:
    return np.asarray([sum((r["groups"][g] for g in GROUPS[group]), []) for r in rows], dtype=float)


def estimator(name: str):
    p = RECIPES[name]
    family = p["family"]
    if family == "LR":
        return LogisticRegression(C=p["C"], max_iter=5000, tol=1e-8, solver="lbfgs", random_state=0)
    if family == "HGB":
        return HistGradientBoostingClassifier(
            max_depth=p["max_depth"],
            l2_regularization=p["l2_regularization"],
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=0,
        )
    cls = RandomForestClassifier if family == "RF" else ExtraTreesClassifier
    return cls(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"],
        random_state=0,
        n_jobs=2,
    )


def check_freeze(root: Path):
    if not any(p.exists() for p in (root / "base-freeze.json", root / "freeze.json")):
        raise ValueError("SEARCH_INPUTS_NOT_FROZEN")
    for freeze in (root / "base-freeze.json", root / "freeze.json"):
        if freeze.exists():
            for path, expected in io.read(freeze)["files"].items():
                if io.sha(path) != expected:
                    raise ValueError("SEARCH_FREEZE_CHANGED")


def input_rows(root: Path):
    return io.lines(root / ("inputs.jsonl" if (root / "inputs.jsonl").exists() else "base-inputs.jsonl"))


def fit(root: Path, group: str, recipe: str, fold: str, replay=False) -> dict:
    """统一折内标准化；先保存未评分预测，再由允许的阶段计算指标。"""
    check_freeze(root)
    run_id = f"{group}__{recipe}__{fold}" + ("__replay" if replay else "")
    folder = root / "models" / run_id
    if (folder / "complete.json").exists():
        return io.read(folder / "complete.json")
    if fold in ("T", "FINAL") and not (root / "selection.json").exists():
        raise ValueError("SELECTION_MUST_PRECEDE_AUDIT_TRAINING")
    rows = input_rows(root)
    split = io.read(root / "splits.json")[fold]
    tr, er = [rows[i] for i in split["training"]], [rows[i] for i in split["evaluation"]]
    if any(r["label_mature_at"] >= split["cutoff_exclusive"] for r in tr):
        raise ValueError("TRAINING_LABEL_NOT_MATURE")
    nav = io.read(sources.OLD / "snapshot/nav-facts.json")
    y, x = impact.labels(tr, nav), matrix(tr, group)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    imputed = imputer.fit_transform(x)
    scaler = StandardScaler().fit(imputed)
    learner = estimator(recipe)
    count = len(io.lines(root / "fit-ledger.jsonl"))
    if count >= io.read(root / "plan.json")["max_supervised_fits"]:
        raise ValueError("SEARCH_FIT_BUDGET_REACHED")
    io.append(
        root / "fit-ledger.jsonl",
        {
            "id": run_id,
            "at": io.now(),
            "group": group,
            "recipe": recipe,
            "fold": fold,
            "training_dates": len(tr),
            "evaluation_dates": len(er),
            "training_dates_sha256": io.digest(split["train_dates"]),
            "input_sha256": io.digest([[float(v) if np.isfinite(v) else None for v in row] for row in x]),
            "training_missing_cells": int(np.isnan(x).sum()),
            "imputer_fitted_only_on_training": True,
        },
    )
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        learner.fit(scaler.transform(imputed), y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        io.save(folder / "failed.json", {"reason": "NOT_CONVERGED", "id": run_id})
        return {"id": run_id, "failed": True}
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": learner, "scaler": scaler, "imputer": imputer, "group": group, "recipe": recipe},
        folder / "model.joblib",
    )
    predicted = []
    if er:
        with threadpool_limits(limits=2):
            probs = learner.predict_proba(scaler.transform(imputer.transform(matrix(er, group))))
        aligned = np.zeros((len(er), 3))
        for j, label in enumerate(learner.classes_):
            aligned[:, CLASSES.index(label)] = probs[:, j]
        predicted = [
            {
                "target": r["target"],
                "predicted": CLASSES[int(np.argmax(aligned[i]))],
                "probabilities": aligned[i].tolist(),
            }
            for i, r in enumerate(er)
        ]
    io.save_lines(folder / "predictions.jsonl", predicted)
    result = {
        "id": run_id,
        "failed": False,
        "features": x.shape[1],
        "training_dates": len(tr),
        "evaluation_dates": len(er),
        "model_sha256": io.sha(folder / "model.joblib"),
        "predictions_sha256": io.sha(folder / "predictions.jsonl"),
    }
    io.save(folder / "complete.json", result)
    return result


def predictions(root: Path, group: str, recipe: str, fold: str) -> list[dict]:
    return io.lines(root / "models" / f"{group}__{recipe}__{fold}" / "predictions.jsonl")


def score_rows(rows: list[dict], nav: dict, predicted: list[dict]) -> dict:
    if [r["target"] for r in rows] != [p["target"] for p in predicted]:
        raise ValueError("PREDICTION_COVERAGE_CHANGED")
    return impact.stats.score(impact.labels(rows, nav), [p["predicted"] for p in predicted])


def ensemble(prediction_lists: list[list[dict]]) -> list[dict]:
    dates = [[p["target"] for p in values] for values in prediction_lists]
    if any(d != dates[0] for d in dates):
        raise ValueError("ENSEMBLE_DATES_DIFFER")
    probs = np.mean([np.asarray([p["probabilities"] for p in values]) for values in prediction_lists], axis=0)
    return [
        {"target": day, "predicted": CLASSES[int(np.argmax(probs[i]))], "probabilities": probs[i].tolist()}
        for i, day in enumerate(dates[0])
    ]


def search(root: Path = ROOT, base_only: bool = False) -> dict:
    """完成所有预登记开发折，不访问2026评分；已完成的等价run直接复用。"""
    check_freeze(root)
    rows, splits = input_rows(root), io.read(root / "splits.json")
    nav = io.read(sources.OLD / "snapshot/nav-facts.json")
    all_results, dev_predictions = [], {}
    for group in GROUPS:
        if base_only and group in ("NHMIF_COUNTS", "ALL_SEMANTICS"):
            continue
        for recipe in RECIPES:
            combined, scores, failed = [], [], False
            feature_count = 0
            for fold in ("V1", "V2", "V3"):
                run = fit(root, group, recipe, fold)
                if run["failed"]:
                    failed = True
                    break
                feature_count = run["features"]
                saved = predictions(root, group, recipe, fold)
                er = [rows[i] for i in splits[fold]["evaluation"]]
                scores.append(score_rows(er, nav, saved))
                combined.extend(saved)
            if failed:
                all_results.append({"group": group, "recipe": recipe, "failed": True})
                continue
            er = [r for r in rows if "2025-01-01" <= r["target"] <= "2025-12-31"]
            score = score_rows(er, nav, combined)
            result = {
                "group": group,
                "recipe": recipe,
                "failed": False,
                "features": feature_count,
                "score": score,
                "fold_scores": scores,
                "min_fold_accuracy": min(s["accuracy"] for s in scores),
            }
            all_results.append(result)
            dev_predictions[(group, recipe)] = combined
            io.replace(
                root / "development-progress.json",
                {"at": io.now(), "completed": len(all_results), "total": 48, "latest": result},
            )
            print(f"DEV {len(all_results)}/48 {group} {recipe}: {score['correct']}/{score['dates']}", flush=True)
    if base_only:
        io.save(root / "base-development.json", all_results)
        return {
            "completed_base_candidates": len(all_results),
            "selected": False,
            "reason": "WAIT_ALL_REGISTERED_GROUPS",
        }
    ranked = sorted(
        [r for r in all_results if not r["failed"]],
        key=lambda r: (-r["score"]["correct"], -r["min_fold_accuracy"], r["features"], r["group"], r["recipe"]),
    )
    if not ranked:
        raise ValueError("NO_VALID_CANDIDATE")
    top, families = [], set()
    for candidate in ranked:
        family = RECIPES[candidate["recipe"]]["family"]
        if family not in families:
            top.append(candidate)
            families.add(family)
        if len(top) == 3:
            break
    combined = ensemble([dev_predictions[(r["group"], r["recipe"])] for r in top])
    er = [r for r in rows if "2025-01-01" <= r["target"] <= "2025-12-31"]
    ensemble_score = score_rows(er, nav, combined)
    use_ensemble = ensemble_score["correct"] > ranked[0]["score"]["correct"]
    selected = top if use_ensemble else [ranked[0]]
    selection = {
        "at": io.now(),
        "basis": "2025_EXPANDING_FOLD_DEVELOPMENT_ONLY",
        "mode": "ENSEMBLE" if use_ensemble else "SINGLE",
        "members": selected,
        "selected_dev_score": ensemble_score if use_ensemble else ranked[0]["score"],
        "best_single": ranked[0],
        "ensemble_dev_score": ensemble_score,
        "all_development": all_results,
        "2026_score_accessed_for_selection": False,
    }
    io.save_lines(
        root / "selected-development-predictions.jsonl",
        combined if use_ensemble else dev_predictions[(ranked[0]["group"], ranked[0]["recipe"])],
    )
    io.save(root / "selection.json", selection)
    return {k: selection[k] for k in ("mode", "members", "selected_dev_score", "2026_score_accessed_for_selection")}


def audit(root: Path = ROOT) -> dict:
    """选择冻结后一次审计2026既有日期；评分不反馈给搜索循环。"""
    selection = io.read(root / "selection.json")
    chosen = selection["members"]
    for candidate in chosen:
        fit(root, candidate["group"], candidate["recipe"], "T")
    fit(root, "N", "LR_C1", "T")
    predicted = ensemble([predictions(root, c["group"], c["recipe"], "T") for c in chosen])
    reference = predictions(root, "N", "LR_C1", "T")
    io.save_lines(root / "selected-audit-predictions.jsonl", predicted)
    io.append(
        root / "audit-access-ledger.jsonl",
        {
            "at": io.now(),
            "selection_sha256": io.sha(root / "selection.json"),
            "prediction_sha256": io.sha(root / "selected-audit-predictions.jsonl"),
            "purpose": "SINGLE_FIXED_AUDIT",
        },
    )
    rows, split = io.lines(root / "inputs.jsonl"), io.read(root / "splits.json")["T"]
    er = [rows[i] for i in split["evaluation"]]
    nav = io.read(sources.OLD / "snapshot/nav-facts.json")
    actual = impact.labels(er, nav)
    delta = [
        int(p["predicted"] == a) - int(b["predicted"] == a)
        for a, p, b in zip(actual, predicted, reference, strict=True)
    ]
    result = {
        "at": io.now(),
        "selected": score_rows(er, nav, predicted),
        "same_dates_N8_baseline": score_rows(er, nav, reference),
        "extra_correct": sum(delta),
        "paired_interval": impact.stats.block_interval(delta, [r["session_index"] for r in er]),
        "always_up": impact.stats.score(actual, ["UP"] * len(actual)),
        "years_are_previously_seen_history": True,
        "adopted": False,
        "body_semantics_dates": sum(r["body_semantics"] > 0 for r in er),
        "weighted_semantics_dates": sum(r["weighted_semantics"] > 0 for r in er),
    }
    io.save(root / "audit-result.json", result)
    for candidate in chosen:
        fit(root, candidate["group"], candidate["recipe"], "FINAL")
    best = chosen[0]
    replay_result = fit(root, best["group"], best["recipe"], "T", replay=True)
    replay_path = root / "models" / replay_result["id"] / "predictions.jsonl"
    original_path = root / "models" / f"{best['group']}__{best['recipe']}__T" / "predictions.jsonl"
    if io.read(replay_path.with_name("complete.json"))["predictions_sha256"] != io.sha(original_path):
        raise ValueError("REPLAY_PREDICTIONS_NOT_EXACT")
    io.save(
        root / "completion.json",
        {
            "at": io.now(),
            "fits": len(io.lines(root / "fit-ledger.jsonl")),
            "selection_sha256": io.sha(root / "selection.json"),
            "audit_sha256": io.sha(root / "audit-result.json"),
            "replay_exact": True,
            "status": "RESEARCH_BATCH_COMPLETE",
            "production_activation": False,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "base", "build", "search", "audit"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "base":
        build(args.root, base_only=True)
        result = search(args.root, base_only=True)
    else:
        result = {"prepare": prepare, "build": build, "search": search, "audit": audit}[args.command](args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
