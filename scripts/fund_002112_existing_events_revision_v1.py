"""G_REVIEW_001/002 一次性非拟合修订：只用原冻结来源，生成差异、B0等价证明与预算续跑契约。"""

from __future__ import annotations

import shutil
from collections import defaultdict
from decimal import Decimal

import joblib
import numpy as np

from scripts import fund_002112_existing_events_features_v1 as f
from scripts import fund_002112_existing_events_fit_v1 as fit
from scripts import fund_002112_existing_events_sources_v1 as s
from scripts import fund_002112_existing_events_v1 as run

REVISION = "20260930-r7-verified-source"


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    if s.sha(source) != s.sha(target):
        raise ValueError("REVISION_COPY_MISMATCH")


def ref_key(ref):
    return s.canonical({k: ref[k] for k in ("entry", "row", "column") if k in ref})


def impact(original, revised):
    old_rows, old_members, old_events = run.load(original)
    rows, members, events = run.load(revised)
    by_ref = {ref_key(ref): e for e in events for ref in e["source_refs"]}
    mapping, classifications = {}, []
    combined = defaultdict(list)
    for old in old_events:
        matches = {by_ref[ref_key(ref)]["event_id"] for ref in old["source_refs"] if ref_key(ref) in by_ref}
        if len(matches) != 1:
            raise ValueError("OLD_CANONICAL_EVENT_NOT_SINGLE_MAPPED:" + old["event_id"])
        new = by_ref[next(ref_key(ref) for ref in old["source_refs"] if ref_key(ref) in by_ref)]
        mapping[old["event_id"]] = new["event_id"]
        combined[new["event_id"]].append(old)
        if new["primary_type"] != old["primary_type"]:
            classifications.append(
                {
                    "old_event_id": old["event_id"],
                    "new_event_id": new["event_id"],
                    "published_date": old["published_date"],
                    "title": old["title"],
                    "url": old["url"],
                    "old_type": old["primary_type"],
                    "new_type": new["primary_type"],
                    "classification_evidence": new["classification_evidence"],
                    "source_refs": new["source_refs"],
                }
            )
    merges = [
        {
            "canonical_event_id": new,
            "old_events": [
                {k: old[k] for k in ("event_id", "published_date", "title", "url", "text_level")} for old in group
            ],
        }
        for new, group in combined.items()
        if len(group) > 1
    ]
    changes = {}
    splits = s.read(original / "split-manifest.json")
    old_map = {r["target"]: (r, m) for r, m in zip(old_rows, old_members, strict=True)}
    new_map = {r["target"]: (r, m) for r, m in zip(rows, members, strict=True)}

    def text_signature(event_set, member, mode):
        # 不拿重新生成的event_id当文字变化；比较真实类别、文字、窗口年龄和重复次数。
        return sorted((event_set[i]["primary_type"], event_set[i]["text"], age) for i, age in member[mode])

    for stage, split in splits.items():
        for mode in f.TIMINGS:
            changed_training_stats, changed_training_text, changed_eval_stats, changed_eval_text = [], [], [], []
            for use in ("train_dates", "evaluation_dates"):
                for day in split[use]:
                    a, am = old_map[day]
                    b, bm = new_map[day]
                    if a[mode] != b[mode]:
                        (changed_training_stats if use == "train_dates" else changed_eval_stats).append(day)
                    if text_signature(old_events, am, mode) != text_signature(events, bm, mode):
                        (changed_training_text if use == "train_dates" else changed_eval_text).append(day)
            changes[f"{stage}_{mode}"] = {
                "training_statistics_changed_dates": changed_training_stats,
                "training_text_changed_dates": changed_training_text,
                "evaluation_statistics_changed_dates": changed_eval_stats,
                "evaluation_text_changed_dates": changed_eval_text,
            }
    result = {
        "original_events": len(old_events),
        "revised_events": len(events),
        "classification_changes": classifications,
        "deduplication_merges": merges,
        "fold_changes": changes,
        "input_dates_unchanged": [r["target"] for r in old_rows] == [r["target"] for r in rows],
        "n8_unchanged": [r["n8"] for r in old_rows] == [r["n8"] for r in rows],
        "labels_source_unchanged": s.sha(original / "snapshot/nav-facts.json")
        == s.sha(revised / "snapshot/nav-facts.json"),
        "frozen_source_inventory_unchanged": s.sha(original / "source-inventory.json")
        == s.sha(revised / "source-inventory.json"),
    }
    s.save(revised / "repair-impact.json", result)
    return result


def actual_labels(nav, rows):
    return [
        "UP"
        if Decimal(nav[r["target"]]["unit_nav"]) > Decimal(nav[r["base"]]["unit_nav"])
        else "DOWN"
        if Decimal(nav[r["target"]]["unit_nav"]) < Decimal(nav[r["base"]]["unit_nav"])
        else "FLAT"
        for r in rows
    ]


def prove_b0_reuse(original, revised, stage):
    """无监督尺度重算和旧模型预测回读不消费监督fit；原训练标签逐值等价，评估答案不读。"""
    old_rows, old_members, old_events = run.load(original)
    rows, members, events = run.load(revised)
    old_split = s.read(original / "split-manifest.json")[stage]
    split = s.read(revised / "split-manifest.json")[stage]
    if old_split != split:
        raise ValueError("B0_SPLIT_CHANGED")
    ti, ei = split["train_indices"], split["evaluation_indices"]
    old_tr, tr = [old_rows[i] for i in ti], [rows[i] for i in ti]
    old_er, er = [old_rows[i] for i in ei], [rows[i] for i in ei]
    nav_old, nav = s.read(original / "snapshot/nav-facts.json"), s.read(revised / "snapshot/nav-facts.json")
    y_old, y = actual_labels(nav_old, old_tr), actual_labels(nav, tr)
    s.append(
        revised / "label-access-ledger.jsonl",
        {
            "at": s.now(),
            "stage": stage,
            "purpose": "TRAIN_LABEL_EQUIVALENCE_ONLY_NO_EVAL_LABELS",
            "date_count": len(tr),
            "dates_sha256": s.digest(split["train_dates"]),
        },
    )
    run_id = stage + "_main_B0"
    old_dir = run.run_path(original, run_id)
    marker = run.validate_completed(original, run_id)
    model = joblib.load(old_dir / "model.joblib")
    original_preprocessor = model["preprocessor"]
    prep = fit.fit_preprocessor(tr, [members[i] for i in ti], events, "B0", "main")
    scaled_old = fit.transform(original_preprocessor, old_er, [old_members[i] for i in ei], old_events)
    scaled_new = fit.transform(prep, er, [members[i] for i in ei], events)
    predicted = fit.predict(model, scaled_new, er)
    previous = s.lines(old_dir / "predictions.jsonl")
    checks = {
        "dates_and_maturity": old_split == split,
        "training_n8_exact": np.array_equal(fit.numeric(old_tr, "B0", "main"), fit.numeric(tr, "B0", "main")),
        "prediction_n8_exact": np.array_equal(fit.numeric(old_er, "B0", "main"), fit.numeric(er, "B0", "main")),
        "training_labels_exact": y_old == y,
        "parameters_exact": all(model["estimator"].get_params()[k] == v for k, v in fit.RECIPE.items()),
        "numeric_column_order": original_preprocessor["numeric_order"] == prep["numeric_order"],
        "preprocessor_exact": all(
            np.array_equal(getattr(original_preprocessor["scaler"], k), getattr(prep["scaler"], k))
            for k in ("mean_", "var_", "scale_", "n_samples_seen_", "n_features_in_")
        ),
        "prediction_matrix_exact": (scaled_old != scaled_new).nnz == 0,
        "prediction_classes_exact": [p["predicted"] for p in previous] == [p["predicted"] for p in predicted],
    }
    error = run.probability_error(previous, predicted)
    proof = {
        "stage": stage,
        "checks": checks,
        "equivalent": all(checks.values()) and error <= 1e-10,
        "max_probability_error": error,
        "training_label_hash": s.digest(y),
        "old_nav_sha256": s.sha(original / "snapshot/nav-facts.json"),
        "new_nav_sha256": s.sha(revised / "snapshot/nav-facts.json"),
        "old_protocol_sha256": s.sha(original / "protocol.json"),
        "new_protocol_sha256": s.sha(revised / "protocol.json"),
        "old_complete_sha256": s.sha(old_dir / "complete.json"),
        "new_supervised_fits": 0,
        "original_attempt": marker["attempt"],
    }
    if not proof["equivalent"]:
        raise ValueError("B0_NOT_EQUIVALENT")
    new_dir = run.run_path(revised, run_id)
    for name in marker["files"]:
        copy_exact(old_dir / name, new_dir / name)
    s.save(new_dir / "reuse-equivalence.json", proof)
    s.save(
        new_dir / "complete.json",
        {
            **marker,
            "reused_from_previous_revision": True,
            "files": {**marker["files"], "reuse-equivalence.json": s.sha(new_dir / "reuse-equivalence.json")},
        },
    )
    return proof


def main():
    original = s.ROOT
    revised = original / "revisions" / REVISION
    before = run.budget(original)["consumed"]
    if before != 8:
        raise ValueError("REPAIR_EXPECTS_RECORDED_EIGHT_FITS_NO_NEW_FITS")
    with s.writer_lock(original), s.offline_guard():
        if not (revised / "revision.json").exists():
            s.save(
                revised / "revision.json",
                {
                    "revision": REVISION,
                    "experiment_root": str(original),
                    "experiment": s.EXPERIMENT,
                    "created_at": s.now(),
                    "frozen_source_root": str(original),
                    "findings": ["G_REVIEW_001", "G_REVIEW_002"],
                    "training_execution_allowed": False,
                    "budget_ledger": str(original / "fit-ledger.jsonl"),
                },
            )
        for name in ("source-inventory.json", "protection-before.json", "identity.json", "snapshot/database.json"):
            copy_exact(original / name, revised / name)
        f.build_inputs(revised)
        changes = impact(original, revised)
        run.freeze(revised)
        proofs = [prove_b0_reuse(original, revised, stage) for stage in ("V2023", "C2024")]
        invalidated = []
        for item in s.lines(original / "fit-ledger.jsonl"):
            if item["action"] != "RESERVED" or item["run_id"].endswith("_B0"):
                continue
            stage, mode, _ = item["run_id"].split("_")
            changed = changes["fold_changes"][f"{stage}_{mode}"]
            if not changed["training_statistics_changed_dates"]:
                raise ValueError("EXPECTED_AFFECTED_MODEL_NOT_CHANGED")
            invalidated.append(
                {
                    "attempt": item["attempt"],
                    "run_id": item["run_id"],
                    "status": "OLD_REVISION_INVALID_FOR_CORRECTED_CONCLUSION",
                    "original_directory": str(run.run_path(original, item["run_id"])),
                    "statistics_changed_training_dates": len(changed["training_statistics_changed_dates"]),
                    "text_changed_training_dates": len(changed["training_text_changed_dates"]),
                }
            )
        s.save(revised / "invalidated-models.json", invalidated)
        minimum = {
            "current_authorized_limit": 28,
            "global_consumed": before,
            "old_invalidated_fits": len(invalidated),
            "equivalent_b0_reused": len(proofs),
            "remaining_required_fits": 23,
            "minimum_total_fits": before + 23,
            "shortfall_against_28": before + 23 - 28,
            "remaining_breakdown": {"main": 10, "aux": 8, "final": 3, "independent_refit": 2},
            "minimum_total_bucket_limits": {"main": 16, "aux": 10, "final": 3, "replay": 2, "retry": 0},
            "training_execution_allowed": False,
            "extra_failure_capacity_at_minimum_31": 0,
        }
        s.save(revised / "continuation-budget-plan.json", minimum)
        s.save(
            revised / "continuation-authorization.template.json",
            {
                "execution_allowed": False,
                "max_real_fits": 31,
                "bucket_limits": minimum["minimum_total_bucket_limits"],
                "revision_protocol_sha256": s.sha(revised / "protocol.json"),
                "authorization_evidence": None,
                "instruction": "仅在用户明确授权调整预算后，由协调者记录真实授权"
                "并保存到原实验根continuation-authorization.json",
            },
        )
        result = {
            "at": s.now(),
            "repair_engineering_complete": True,
            "additional_real_fits": 0,
            "global_real_fits": run.budget(revised)["consumed"],
            "frozen_sources_unchanged": True,
            "classification_changed_events": len(changes["classification_changes"]),
            "deduplication_merged_groups": len(changes["deduplication_merges"]),
            "duplicate_event_reduction": changes["original_events"] - changes["revised_events"],
            "reusable_b0": [
                {
                    "stage": p["stage"],
                    "equivalent": p["equivalent"],
                    "max_probability_error": p["max_probability_error"],
                }
                for p in proofs
            ],
            "invalidated_old_models": len(invalidated),
            "budget_plan": minimum,
            "revision_protocol_sha256": s.sha(revised / "protocol.json"),
            "full_experiment_complete": False,
            "budget_approval_pending": True,
        }
        if run.budget(revised)["consumed"] != before:
            raise ValueError("REPAIR_UNEXPECTEDLY_CONSUMED_REAL_FIT")
        s.save(revised / "repair-acceptance.json", result)
        s.replace(
            original / "active-revision.json",
            {
                "path": revised.relative_to(original).as_posix(),
                "protocol_sha256": result["revision_protocol_sha256"],
                "execution_allowed": False,
                "reason": "G_REVIEW_001_002_REPAIR_BUDGET_HOLD",
            },
        )
        run.progress(
            revised, "E03_REVISION", "REPAIR_READY_BUDGET_HOLD", "修复验收完成；8次保留，最少总需31次，未获补充授权"
        )
        print(s.canonical(result), flush=True)


if __name__ == "__main__":
    main()
