"""只读复核存量量价实验并保存结项证据；不拟合，不改冻结脚本或旧结果。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from scripts import fund_002112_price_search_v1 as search

io, ROOT = search.io, search.ROOT


def closeout():
    """保留逐字重放失败，另核对树结构和既有数值容差；事后补充不伪装为预登记。"""
    search.configure()
    search.engine.access_guard()
    search.engine.verify_freeze()
    if (ROOT / "completion.json").exists():
        return io.read(ROOT / "completion.json")
    winner = io.read(ROOT / "selection.json")["winner"]
    candidate = winner["candidate"]
    records = [
        io.read(ROOT / "runs" / (candidate["id"] + "__V3" + suffix) / "complete.json") for suffix in ("", "__replay")
    ]
    for record in records:
        assert io.sha(Path(record["model_path"])) == record["model_sha256"]
        assert io.sha(Path(record["predictions_path"])) == record["predictions_sha256"]
    fitted = [joblib.load(record["model_path"]) for record in records]
    saved = [search.engine.predicted(record) for record in records]
    assert [p["target"] for p in saved[0]] == [p["target"] for p in saved[1]]
    probabilities = [np.asarray([p["probabilities"] for p in group]) for group in saved]
    tolerance = 1e-13  # 沿用已经冻结的独立模型读回校验容差，不根据本次差值扩大。
    np.testing.assert_allclose(*probabilities, atol=tolerance, rtol=0)
    direction_differences = sum(a["predicted"] != b["predicted"] for a, b in zip(*saved, strict=True))
    assert direction_differences == 0
    for key, attributes in {
        "imputer": ("statistics_",),
        "scaler": ("mean_", "var_", "scale_", "n_samples_seen_"),
    }.items():
        for attribute in attributes:
            np.testing.assert_array_equal(getattr(fitted[0][key], attribute), getattr(fitted[1][key], attribute))
    for model in fitted:
        assert len(model["imputer"].indicator_.features_) == 0
    a, b = (model["model"] for model in fitted)
    assert a.get_params() == b.get_params()
    np.testing.assert_array_equal(a.classes_, b.classes_)
    assert a.n_features_in_ == b.n_features_in_
    assert len(a.estimators_) == len(b.estimators_) == 200
    # 同时检查每棵树的分裂、阈值、叶节点权重及随机种子，不能只比较最终方向。
    attributes = (
        "children_left",
        "children_right",
        "feature",
        "threshold",
        "value",
        "n_node_samples",
        "weighted_n_node_samples",
        "impurity",
        "missing_go_to_left",
    )
    for ta, tb in zip(a.estimators_, b.estimators_, strict=True):
        assert ta.get_params() == tb.get_params()
        for attribute in attributes:
            np.testing.assert_array_equal(getattr(ta.tree_, attribute), getattr(tb.tree_, attribute))
    rows = io.lines(ROOT / "inputs.jsonl")
    splits = io.read(ROOT / "splits.json")
    evaluation = [rows[i] for i in splits["V3"]["evaluation"]]
    x = search.engine.matrix(evaluation, candidate["group"])
    transformed = [m["scaler"].transform(m["imputer"].transform(x)) for m in fitted]
    np.testing.assert_array_equal(*transformed)
    # 固定树次序串行求均值，只读已保存模型，避开并行累加顺序的末位舍入。
    serial = [
        np.mean(np.stack([tree.predict_proba(values) for tree in model["model"].estimators_]), axis=0)
        for model, values in zip(fitted, transformed, strict=True)
    ]
    np.testing.assert_array_equal(*serial)
    for original, calculated in zip(probabilities, serial, strict=True):
        aligned = np.zeros_like(original)
        for index, label in enumerate(a.classes_):
            aligned[:, search.engine.kernel.CLASSES.index(label)] = calculated[:, index]
        np.testing.assert_allclose(original, aligned, atol=tolerance, rtol=0)
    replay = {
        "at": io.now(),
        "post_fit_validation_addendum": True,
        "original_validation": "FAILED_LITERAL_LIST_EQUALITY_ASSERTION_AFTER_46_SUCCESSFUL_FITS",
        "original_runner_preserved": True,
        "new_fits": 0,
        "literal_probability_equality": saved[0] == saved[1],
        "max_absolute_probability_difference": float(np.max(np.abs(probabilities[0] - probabilities[1]))),
        "absolute_tolerance_from_frozen_verifier": tolerance,
        "relative_tolerance": 0,
        "probabilities_within_existing_tolerance": True,
        "direction_differences": direction_differences,
        "trees_exact": 200,
        "preprocessors_exact": True,
        "serial_tree_mean_probabilities_exact": True,
        "interpretation": (
            "Identical fitted trees and serial means; "
            "parallel accumulation rounding is consistent with observed difference"
        ),
        "references": records,
    }
    io.save(ROOT / "numerical-replay-audit.json", replay)
    io.save(
        ROOT / "validation-failures.json",
        {
            "recorded_at": io.now(),
            "training_fit_failures": 0,
            "post_fit_validation_failures": 1,
            "exception": "AssertionError",
            "condition": "engine.predicted(repeat) == engine.predicted(original)",
            "script": str(Path(search.__file__)),
            "script_sha256": io.sha(Path(search.__file__)),
            "resolution": "numerical-replay-audit.json",
            "exact_equality_claimed": False,
        },
    )
    # 差异区间仅用于描述已选择候选，不能当作事先独立检验或新的选模依据。
    nav = io.read(ROOT / "nav-through-2025.json")
    development = [row for row in rows if row["target"].startswith("2025")]
    actual = search.source.timing.labels(development, nav)
    predicted = io.lines(ROOT / "development-predictions" / (candidate["id"] + ".jsonl"))
    assert [v["target"] for v in predicted] == [r["target"] for r in development]
    baselines = io.read(ROOT / "baselines.json")
    base = [p for fold in search.engine.FOLDS for p in io.lines(Path(baselines[fold]["N_reference"]["path"]))]
    assert [p["target"] for p in base] == [p["target"] for p in predicted]
    intervals = {}
    for name, labels in {"N": [p["predicted"] for p in base], "ALWAYS_UP": ["UP"] * len(actual)}.items():
        difference = [
            int(p["predicted"] == truth) - int(baseline == truth)
            for p, baseline, truth in zip(predicted, labels, actual, strict=True)
        ]
        intervals[name] = {
            "net_correct_days": sum(difference),
            "accuracy_delta": sum(difference) / len(actual),
            "interval": search.engine.kernel.impact.stats.block_interval(
                difference, [r["session_index"] for r in development]
            ),
        }
    io.save(
        ROOT / "development-comparison.json",
        {
            "at": io.now(),
            "dates": 243,
            "post_selection_descriptive_only": True,
            "independent_blind_test": False,
            "comparisons": intervals,
        },
    )
    manifest = io.read(ROOT / "source-manifest.json")
    for value in manifest["sources"].values():
        assert io.sha(Path(value["snapshot"])) == value["sha256"]
    previous = search.source.timing.ROOT
    previous_proof = io.read(previous / "final-verification.json")
    for name, digest in previous_proof["artifacts"].items():
        assert io.sha(previous / name) == digest
    io.save(
        ROOT / "preservation-verification.json",
        {
            "at": io.now(),
            "frozen_sources_verified": len(manifest["sources"]),
            "training_freeze_verified": True,
            "previous_final_artifacts_verified": len(previous_proof["artifacts"]),
            "previous_final_verification_sha256": io.sha(previous / "final-verification.json"),
            "prior_2026_scores_read": False,
        },
    )
    verified = io.read(ROOT / "independent-verification.json")
    ledger = io.lines(ROOT / "fit-ledger.jsonl")
    assert len(ledger) == verified["models_verified"] == 46
    assert len(list((ROOT / "runs").glob("*/complete.json"))) == 46
    assert verified["all_candidate_scores_and_selection_match"]
    done = {
        "at": io.now(),
        "status": "COMPLETE_RESEARCH_WITH_NUMERICAL_REPLAY",
        "actual_fits": 46,
        "candidate_count": 15,
        "registered_remaining": 0,
        "winner": winner,
        "replay_probabilities_exact": False,
        "replay_numerical_tolerance_passed": True,
        "replay_absolute_tolerance": tolerance,
        "replay_trees_exact": True,
        "post_fit_validation_addendum": True,
        "new_2026_scores": None,
        "adopted": False,
        "strict_historical_first_seen_proven": False,
        "global_optimum_proven": False,
        "closeout_script_sha256": io.sha(Path(__file__)),
    }
    io.save(ROOT / "completion.json", done)
    return {k: v for k, v in done.items() if k != "winner"} | {"winner": candidate["id"], "correct": 129}


if __name__ == "__main__":
    print(json.dumps(closeout(), ensure_ascii=False, indent=2))
