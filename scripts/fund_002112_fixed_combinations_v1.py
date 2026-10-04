"""只合并已冻结的EVERY20概率：恰好四个等权组合，监督fit预算为零。"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import numpy as np

from scripts import fund_002112_morning_coverage_revision_v1 as coverage

io, ROOT, old = coverage.io, coverage.ROOT, coverage.previous
CLASSES = ("DOWN", "FLAT", "UP")
MEMBERS = {
    "LR_HGB": ("LR_C1", "HGB_D2"),
    "LR_RF": ("LR_C1", "RF_D4"),
    "HGB_RF": ("HGB_D2", "RF_D4"),
    "LR_HGB_RF": ("LR_C1", "HGB_D2", "RF_D4"),
}


def prepare():
    """先锁定成员、等权规则、同票顺序和全部源预测摘要；此阶段不计算组合得分。"""
    if (ROOT / "combination-protocol.json").exists():
        return io.read(ROOT / "combination-protocol.json")
    proof = io.read(old.ROOT / "final-verification.json")
    assert all(io.sha(old.ROOT / k) == v for k, v in proof["artifacts"].items())
    sources = {r: str(old.ROOT / "development-predictions" / (r + "__EVERY20.jsonl")) for r in old.RECIPES}
    rows = [r for r in io.lines(old.ROOT / "inputs.jsonl") if r["target"].startswith("2025")]
    calendar = io.read(old.ROOT / "update-calendar.json")
    for recipe, path in sources.items():
        preds = io.lines(Path(path))
        assert len(preds) == len(rows) == 243
        for i, (p, r) in enumerate(zip(preds, rows, strict=True)):
            expected = f"{recipe}__U{i // 20 * 20:03d}"
            update = calendar["updates"][f"U{i // 20 * 20:03d}"]
            assert p["target"] == r["target"] <= "2025-12-31" and p["as_of"] == r["as_of"]
            assert p["model_id"] == expected
            assert p["max_training_label_mature_at"] == update["max_training_label_mature_at"]
            assert p["max_training_label_mature_at"] < p["model_cutoff"] <= p["as_of"]
            assert len(p["probabilities"]) == 3 and all(np.isfinite(p["probabilities"]))
            assert all(0 <= v <= 1 for v in p["probabilities"])
            assert abs(sum(p["probabilities"]) - 1) < 1e-12
    refs = io.read(old.ROOT / "baseline-and-reuse.json")
    paths = [Path(p) for p in sources.values()] + [
        old.ROOT / n
        for n in (
            "inputs.jsonl",
            "nav-through-2025.json",
            "baseline-and-reuse.json",
            "update-calendar.json",
            "final-verification.json",
            "independent-verification.json",
            "training-ledger-audit.json",
        )
    ]
    paths.extend(Path(p) for p in [*refs["N"], refs["fixed_nne"]])
    paths.append(old.prior.ROOT / "splits.json")
    code = [
        Path(__file__),
        Path(coverage.__file__),
        Path(io.__file__),
        Path(old.prior.engine.kernel.impact.stats.__file__),
    ]
    for path in code:
        dest = ROOT / "combination-code-snapshot" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        paths.extend([path, dest])
    plan = {
        "at": io.now(),
        "candidates": MEMBERS,
        "candidate_count": 4,
        "supervised_fit_budget": 0,
        "source_predictions": sources,
        "probability_class_order": CLASSES,
        "combination": "ARITHMETIC_MEAN_EQUAL_WEIGHT_PER_MEMBER_PER_CLASS",
        "direction": "ARGMAX_WITH_EXACT_TIES_DOWN_THEN_FLAT_THEN_UP",
        "selection": "CORRECT_DESC_MIN_OLD_FOLD_ACCURACY_DESC_MEMBER_COUNT_ASC_ID_ASC",
        "comparison": "ALL4_VS_N124_FIXED_NNE129_EVERY20_RF130_ALWAYS_UP128",
        "uncertainty": "DESCRIPTIVE_POST_SELECTION_FIXED5SESSION_BLOCK_2000_SEED0_NO_MULTIPLE_SEARCH_ADJUSTMENT",
        "horizon": "D0800_TO_NEXT_U",
        "label_max": "2025-12-31",
        "coverage": "COMMON243_DATES_100PERCENT",
        "weights_scanned": False,
        "threshold_tuned": False,
        "member_choice_by_daily_errors": False,
        "stop": "EXACTLY4_FIXED_COMBINATIONS_NO_EXPANSION",
        "new_2026_scores": None,
        "adoption": False,
    }
    io.save(ROOT / "combination-protocol.json", plan)
    paths.append(ROOT / "combination-protocol.json")
    io.save(ROOT / "combination-freeze.json", {"at": io.now(), "files": {str(p): io.sha(p) for p in paths}})
    (ROOT / "fit-ledger.jsonl").touch(exist_ok=False)
    return {"at": plan["at"], "candidates": 4, "supervised_fit_budget": 0}


def check_freeze():
    assert all(io.sha(Path(p)) == h for p, h in io.read(ROOT / "combination-freeze.json")["files"].items())
    assert (ROOT / "fit-ledger.jsonl").stat().st_size == 0


def actuals(rows, nav):
    """从2025共同日期独立计算涨跌标签，日期越界立即拒绝。"""
    if any(r["target"] > "2025-12-31" for r in rows) or max(nav) > "2025-12-31":
        raise ValueError("POST2025_LABEL_FORBIDDEN")
    values = []
    for r in rows:
        a, b = Decimal(nav[r["base"]]["unit_nav"]), Decimal(nav[r["target"]]["unit_nav"])
        values.append("UP" if b > a else "DOWN" if b < a else "FLAT")
    return values


def score(actual, predictions):
    correct = sum(a == p for a, p in zip(actual, predictions, strict=True))
    return {
        "correct": correct,
        "dates": len(actual),
        "accuracy": correct / len(actual),
        "classes": {
            label: {
                "actual": actual.count(label),
                "correct": sum(a == p == label for a, p in zip(actual, predictions, strict=True)),
            }
            for label in CLASSES
        },
    }


def run():
    old.prior.engine.access_guard()
    check_freeze()
    if (ROOT / "combination-results.json").exists():
        return io.read(ROOT / "combination-results.json")
    plan = io.read(ROOT / "combination-protocol.json")
    source = {k: io.lines(Path(p)) for k, p in plan["source_predictions"].items()}
    rows = [r for r in io.lines(old.ROOT / "inputs.jsonl") if r["target"].startswith("2025")]
    labels = actuals(rows, io.read(old.ROOT / "nav-through-2025.json"))
    splits = io.read(old.prior.ROOT / "splits.json")
    refs = io.read(old.ROOT / "baseline-and-reuse.json")
    bases = {
        "N": [p for f in refs["N"] for p in io.lines(Path(f))],
        "FIXED_NNE": io.lines(Path(refs["fixed_nne"])),
        "EVERY20_RF": source["RF_D4"],
        "ALWAYS_UP": [{"target": r["target"], "predicted": "UP"} for r in rows],
    }
    expected = {"N": 124, "FIXED_NNE": 129, "EVERY20_RF": 130, "ALWAYS_UP": 128}
    for name, b in bases.items():
        assert [p["target"] for p in b] == [r["target"] for r in rows]
        assert score(labels, [p["predicted"] for p in b])["correct"] == expected[name]
    results = []
    for name, members in plan["candidates"].items():
        io.append(
            ROOT / "combination-ledger.jsonl",
            {
                "at": io.now(),
                "id": name,
                "supervised_fits": 0,
                "protocol_sha256": io.sha(ROOT / "combination-protocol.json"),
            },
        )
        joined = []
        disagreement, ties = 0, 0
        for i, row in enumerate(rows):
            probabilities = np.mean([source[k][i]["probabilities"] for k in members], axis=0)
            disagreement += len({source[k][i]["predicted"] for k in members}) > 1
            ties += sum(probabilities == max(probabilities)) > 1
            joined.append(
                {
                    "target": row["target"],
                    "as_of": row["as_of"],
                    "members": members,
                    "model_ids": {k: source[k][i]["model_id"] for k in members},
                    "max_training_label_mature_at": max(source[k][i]["max_training_label_mature_at"] for k in members),
                    "probabilities": probabilities.tolist(),
                    "predicted": CLASSES[int(np.argmax(probabilities))],
                }
            )
        io.save_lines(ROOT / "combination-predictions" / (name + ".jsonl"), joined)
        folds = []
        for fold in old.prior.engine.FOLDS:
            indices = [i for i, r in enumerate(rows) if r["target"] in splits[fold]["evaluation_dates"]]
            folds.append(score([labels[i] for i in indices], [joined[i]["predicted"] for i in indices]))
        comparisons = {}
        for baseline, values in bases.items():
            delta = [
                int(j["predicted"] == a) - int(b["predicted"] == a)
                for j, b, a in zip(joined, values, labels, strict=True)
            ]
            comparisons[baseline] = {
                "net_correct_days": sum(delta),
                "candidate_only_correct": delta.count(1),
                "baseline_only_correct": delta.count(-1),
                "direction_disagreement_days": sum(
                    j["predicted"] != b["predicted"] for j, b in zip(joined, values, strict=True)
                ),
                "interval": old.prior.engine.kernel.impact.stats.block_interval(
                    delta, [r["session_index"] for r in rows]
                ),
            }
        results.append(
            {
                "id": name,
                "members": members,
                "score": score(labels, [j["predicted"] for j in joined]),
                "fold_scores": folds,
                "member_disagreement_days": disagreement,
                "exact_probability_tie_days": ties,
                "comparisons": comparisons,
                "min_fold_accuracy": min(f["accuracy"] for f in folds),
            }
        )
    ranked = sorted(
        results, key=lambda v: (-v["score"]["correct"], -v["min_fold_accuracy"], len(v["members"]), v["id"])
    )
    outcome = {
        "at": io.now(),
        "results": results,
        "ranking": [v["id"] for v in ranked],
        "winner": ranked[0],
        "supervised_fits": 0,
        "new_2026_scores": None,
        "adopted": False,
    }
    io.save(ROOT / "combination-results.json", outcome)
    return {"winner": ranked[0]["id"], "correct": ranked[0]["score"]["correct"], "candidates": 4, "supervised_fits": 0}


def verify():
    """用十进制重新求等权概率、回算方向和各分段；不运行任何监督训练。"""
    old.prior.engine.access_guard()
    check_freeze()
    plan = io.read(ROOT / "combination-protocol.json")
    source = {k: io.lines(Path(p)) for k, p in plan["source_predictions"].items()}
    rows = [r for r in io.lines(old.ROOT / "inputs.jsonl") if r["target"].startswith("2025")]
    truth = dict(
        zip([r["target"] for r in rows], actuals(rows, io.read(old.ROOT / "nav-through-2025.json")), strict=True)
    )
    results = io.read(ROOT / "combination-results.json")
    splits = io.read(old.prior.ROOT / "splits.json")
    checked = 0
    for result in results["results"]:
        name, members = result["id"], plan["candidates"][result["id"]]
        predictions = io.lines(ROOT / "combination-predictions" / (name + ".jsonl"))
        assert len(predictions) == len(rows) == 243
        for index, (r, p) in enumerate(zip(rows, predictions, strict=True)):
            assert r["target"] == p["target"] and r["as_of"] == p["as_of"]
            expected = [
                float(sum(Decimal(str(source[k][index]["probabilities"][j])) for k in members) / Decimal(len(members)))
                for j in range(3)
            ]
            np.testing.assert_allclose(p["probabilities"], expected, atol=1e-13, rtol=0)
            assert p["predicted"] == CLASSES[max(range(3), key=lambda j: expected[j])]
            assert p["max_training_label_mature_at"] < p["as_of"]
            assert p["model_ids"] == {k: source[k][index]["model_id"] for k in members}
        assert sum(p["predicted"] == truth[p["target"]] for p in predictions) == result["score"]["correct"]
        for fold, score_value in zip(old.prior.engine.FOLDS, result["fold_scores"], strict=True):
            selected = [p for p in predictions if p["target"] in splits[fold]["evaluation_dates"]]
            assert sum(p["predicted"] == truth[p["target"]] for p in selected) == score_value["correct"]
        for label in CLASSES:
            assert (
                sum(p["predicted"] == truth[p["target"]] == label for p in predictions)
                == result["score"]["classes"][label]["correct"]
            )
        checked += len(predictions)
    assert len(io.lines(ROOT / "combination-ledger.jsonl")) == 4
    assert all(e["at"] > plan["at"] and e["supervised_fits"] == 0 for e in io.lines(ROOT / "combination-ledger.jsonl"))
    outcome = {
        "at": io.now(),
        "probabilities_and_directions_recomputed": checked,
        "supervised_fits": 0,
        "fixed_candidates": 4,
        "old_probability_files_preserved": True,
        "new_2026_scores": None,
        "script_sha256": io.sha(Path(__file__)),
    }
    io.save(ROOT / "combination-independent-verification.json", outcome)
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "run": run, "verify": verify}[args.command](), ensure_ascii=False, indent=2))
