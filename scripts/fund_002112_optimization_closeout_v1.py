"""只读回算已完成批次的选模与分数，并新增封存证据；不重新训练或访问外部服务。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import fund_002112_event_optimization_audit_v1 as audit
from scripts import fund_002112_next_day_search_v1 as next_day

io = audit.io


def selection_boundary(root: Path) -> dict:
    """从逐日开发预测独立算排序和等权组合，核对保存的选择先于后段拟合和评分。"""
    rows = io.lines(root / "inputs.jsonl")
    splits = io.read(root / "splits.json")
    nav = io.read(audit.event.OLD / "snapshot/nav-facts.json")
    plan = io.read(root / "plan.json")
    selection = io.read(root / "selection.json")
    candidates, probabilities = [], {}
    development = [r for r in rows if r["target"].startswith("2025-")]
    actual = audit.truth(development, nav)
    for recorded in selection["all_development"]:
        if recorded["failed"]:
            raise AssertionError("REGISTERED_CANDIDATE_FAILED")
        group, recipe = recorded["group"], recorded["recipe"]
        predictions, fold_accuracies = [], []
        for fold in ("V1", "V2", "V3"):
            saved = io.lines(root / "models" / f"{group}__{recipe}__{fold}" / "predictions.jsonl")
            evaluation = [rows[i] for i in splits[fold]["evaluation"]]
            assert [p["target"] for p in saved] == [r["target"] for r in evaluation]
            truth = audit.truth(evaluation, nav)
            correct = sum(p["predicted"] == y for p, y in zip(saved, truth, strict=True))
            fold_accuracies.append(correct / len(saved))
            predictions.extend(saved)
        assert [p["target"] for p in predictions] == [r["target"] for r in development]
        correct = sum(p["predicted"] == y for p, y in zip(predictions, actual, strict=True))
        assert correct == recorded["score"]["correct"]
        assert min(fold_accuracies) == recorded["min_fold_accuracy"]
        candidates.append(((-correct, -min(fold_accuracies), recorded["features"], group, recipe), group, recipe))
        probabilities[(group, recipe)] = np.asarray([p["probabilities"] for p in predictions])
    candidates.sort()
    top, families = [], set()
    for _, group, recipe in candidates:
        family = plan["recipes"][recipe]["family"]
        if family not in families:
            families.add(family)
            top.append((group, recipe))
        if len(top) == 3:
            break
    mean = np.mean([probabilities[key] for key in top], axis=0)
    labels = [audit.search.CLASSES[int(i)] for i in np.argmax(mean, axis=1)]
    combined_correct = sum(a == b for a, b in zip(actual, labels, strict=True))
    assert combined_correct == selection["ensemble_dev_score"]["correct"]
    use_ensemble = combined_correct > -candidates[0][0][0]
    expected = top if use_ensemble else [(candidates[0][1], candidates[0][2])]
    assert expected == [(m["group"], m["recipe"]) for m in selection["members"]]
    assert selection["mode"] == ("ENSEMBLE" if use_ensemble else "SINGLE")
    assert selection["selected_dev_score"]["correct"] == (
        combined_correct if use_ensemble else -candidates[0][0][0]
    )
    ledger = io.lines(root / "fit-ledger.jsonl")
    accesses = io.lines(root / "audit-access-ledger.jsonl")
    assert len(accesses) == 1
    assert all(r["at"] > selection["at"] for r in ledger if r["fold"] in ("T", "FINAL"))
    assert accesses[0]["at"] > selection["at"]
    assert accesses[0]["selection_sha256"] == io.sha(root / "selection.json")
    result = {
        "at": io.now(),
        "candidate_count": len(candidates),
        "development_predictions_recounted": len(actual) * len(candidates),
        "selection_reproduced": True,
        "selected_before_audit_fits_and_scores": True,
        "selection_sha256": io.sha(root / "selection.json"),
        "selected_members": expected,
        "historical_years_previously_seen": True,
        "interpretation": "Verifies this batch sequence, not a claim of globally unseen historical labels",
    }
    io.save(root / "selection-boundary-audit.json", result)
    return result


def event_scores() -> dict:
    """按原始净值小数重新计分，包含沿用的旧基线，防止汇总表错抄。"""
    root, old = audit.impact.ROOT, audit.event.OLD
    rows = {r["target"]: r for r in io.lines(root / "feature-rows.jsonl")}
    nav = io.read(old / "snapshot/nav-facts.json")
    split, comparison = io.read(old / "split-manifest.json"), io.read(root / "comparison.json")
    result = {}
    for fold in ("V2023", "C2024", "C2025", "C2026"):
        dates = split[fold]["evaluation_dates"]
        actual = audit.truth([rows[d] for d in dates], nav)
        models = {"B0": old / "models" / f"{fold}_main_B0", "B2": old / "models" / f"{fold}_aux_B2"}
        models.update({m: root / "models" / f"{fold}_aux_{m}" for m in ("M1", "M2", "M3")})
        result[fold] = {}
        for name, folder in models.items():
            saved = io.lines(folder / "predictions.jsonl")
            assert [p["target"] for p in saved] == dates
            correct = sum(p["predicted"] == y for p, y in zip(saved, actual, strict=True))
            assert comparison["folds"][fold]["scores"][name]["correct"] == correct
            result[fold][name] = correct
    evidence = {"at": io.now(), "folds": result, "source_nav_sha256": io.sha(old / "snapshot/nav-facts.json")}
    io.save(root / "comparison-score-readback.json", evidence)
    return evidence


def main() -> None:
    """本命令新增带时间戳的不可覆盖证据，只对尚未封存的批次运行一次。"""
    results = {"event": event_scores()}
    for name, root in (("morning", audit.search.ROOT), ("next_day", next_day.ROOT)):
        results[name] = selection_boundary(root)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
