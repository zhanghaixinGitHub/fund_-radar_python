"""只读排查002112阶段性失效：分支影响、训练样本、披露持仓、缺失与模型漂移。

不调用fit、不修改模型/配比，不按诊断成绩选新方案。固定旧输入探针只描述模型变化，
其中样本曾参与训练，严禁将探针输出当作预测成绩或未来验证。
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from scripts import fund_002112_evening_attribution_v1 as m
from scripts.fund_002112_evening_verify_v2 import manual_probabilities

old = m.old
ROOT = old.RESEARCH / "evening-failure-audit/20261003-v1"
PERIODS = ["2025", "2026", "ALL"] + [f"{y}Q{q}" for y in ("2025", "2026") for q in range(1, 5)]
METHODS = ("information", "market", "history", "full", "without_information", "without_market", "without_history")


def forbid_fit(*args, **kwargs):
    raise RuntimeError("DIAGNOSTIC_ONLY_NO_TRAINING_ALLOWED")


def top_ten(report):
    """用股票代码比较已披露前十，避免名称更改及半年报披露数量变化造成伪换仓。"""
    if report is None:
        return None
    rows = sorted(report["holdings"], key=lambda h: int(h["reported_rank"]))[:10]
    raw = {h["stock_code"]: float(h["nav_weight_pct"]) / 100 for h in rows}
    total = sum(raw.values())
    return {code: weight / total for code, weight in raw.items()} if total > 0 else None


def overlap(a, b):
    """归一化前十持仓的重合度；未披露时期保持None，不将其当作零重合。"""
    if a is None or b is None:
        return None
    return sum(min(w, b.get(code, 0)) for code, w in a.items())


def probabilities(info, market, history):
    i, p, h = map(np.asarray, (info, market, history))
    return {
        "information": i,
        "market": p,
        "history": h,
        "full": 0.4 * i + 0.4 * p + 0.2 * h,
        "without_information": (2 * p + h) / 3,
        "without_market": (2 * i + h) / 3,
        "without_history": (i + p) / 2,
    }


def metrics(rows, method):
    values = np.array([r["probabilities"][method] for r in rows])
    truth = np.array([old.CLASSES.index(r["label"]) for r in rows])
    guess = values.argmax(axis=1)
    return {
        "n": len(rows),
        "correct": int((guess == truth).sum()),
        "accuracy": float((guess == truth).mean()),
        "predicted_up": int((guess == 2).sum()),
        "actual_up": int((truth == 2).sum()),
        "mean_up_probability": float(values[:, 2].mean()),
        "auc": float(roc_auc_score(truth == 2, values[:, 2])) if len(set(truth == 2)) > 1 else None,
        "down_recall": float((guess[truth == 0] == 0).mean()) if (truth == 0).any() else None,
        "brier": float(np.square(values - np.eye(3)[truth]).sum(axis=1).mean()),
        "near_half_days": int(((values[:, 2] >= 0.45) & (values[:, 2] <= 0.55)).sum()),
    }


def flips(rows, before, after):
    result = Counter()
    for row in rows:
        truth = old.CLASSES.index(row["label"])
        a, b = [int(np.argmax(row["probabilities"][key])) for key in (before, after)]
        key = (
            "UNCHANGED" if a == b else "CORRECTED" if b == truth else "SPOILED" if a == truth else "BOTH_WRONG_CHANGED"
        )
        result[key] += 1
    return {
        "corrected": result["CORRECTED"],
        "spoiled": result["SPOILED"],
        "unchanged": result["UNCHANGED"],
        "both_wrong_changed": result["BOTH_WRONG_CHANGED"],
        "net_correct": result["CORRECTED"] - result["SPOILED"],
    }


def main():
    RandomForestClassifier.fit = forbid_fit
    m.verify_hashes(m.ROOT)
    assert old.read(m.ROOT / "independent-verification.json")["passed"]
    if ROOT.exists():
        raise ValueError("NEW_DIAGNOSTIC_DIRECTORY_REQUIRED")
    ROOT.mkdir(parents=True)
    old.write(ROOT / "protection-before.json", old.git_snapshot())
    protected = {
        str(p): old.sha(p) for parent in (old.ROOT, m.prior.ROOT, m.ROOT) for p in parent.rglob("*") if p.is_file()
    }
    old.write(ROOT / "source-manifest.json", {"files": protected})
    old.write(
        ROOT / "scope.json",
        {
            "at": old.now(),
            "new_fits": 0,
            "candidate_search": False,
            "purpose": "诊断已知历史中的错误；不修改预测，不据此直接选择新方案",
            "fixed_probe": "所有模型输入同一批首个2025模型截点之前的60个旧样本，仅诊断模型变化，不能作为验证成绩",
            "holdings": "只比较当时可见的已披露前十持仓，按代码归一化；无法还原真实每日完整持仓",
            "limits": "删除分支后重新归一化只是已训练模型的算术对照；不能认定因果，也不能代替删分支重训",
        },
    )
    source = old.read(old.ROOT / "sources.json")
    sessions = source["sessions"]
    joint = old.lines(m.ROOT / "BOTH-inputs.jsonl")
    joint_map = {r["target"]: r for r in joint}
    all_history = {r["target"]: r for r in old.lines(old.ROOT / "dataset.jsonl")}
    base = {r["target"]: r for r in old.lines(old.ROOT / "historical-predictions.jsonl")}
    selected = [r for r in old.lines(m.ROOT / "weight-predictions.jsonl") if r["policy"] == "FIXED_40"]
    lineage = {r["target"]: r for r in old.lines(m.prior.ROOT / "feature-lineage.jsonl")}
    quality = {r["target"]: r for r in old.lines(m.ROOT / "quality-evidence.jsonl")}
    # 披露持仓逐日匹配缓存；训练样本始终使用该样本当时可见的报告。
    report_by_day = {d: old.choose_report(source["reports"], row["as_of"]) for d, row in all_history.items()}
    vectors = {d: top_ten(report) for d, report in report_by_day.items()}
    daily = []
    for row in selected:
        d, ref = row["target"], base[row["target"]]
        ps = probabilities(row["information"], ref["branches"]["market"], ref["branches"]["history"])
        assert np.allclose(ps["full"], row["probabilities"], atol=1e-12, rtol=0)
        truth = old.CLASSES.index(row["label"])
        item = {
            "target": d,
            "as_of": row["as_of"],
            "label": row["label"],
            "fit_prefix": row["fit_prefix"],
            "probabilities": {k: p.tolist() for k, p in ps.items()},
            "quality": quality[d]["state"],
            "report_end": report_by_day[d]["report_end"],
            "report_available_at": report_by_day[d]["available_at"],
            "full_correct": int(ps["full"].argmax()) == truth,
            "all_branches_wrong": all(int(ps[k].argmax()) != truth for k in ("information", "market", "history")),
            "all_branches_agree": len({int(ps[k].argmax()) for k in ("information", "market", "history")}) == 1,
            "public_event_ids": [e["id"] for e in lineage[d]["public_events"]],
            "company_event_ids": [e["id"] for e in lineage[d]["company_events"]],
            "weighted_up_minus_down": {
                k: w * float(ps[k][2] - ps[k][0]) for k, w in [("information", 0.4), ("market", 0.4), ("history", 0.2)]
            },
        }
        daily.append(item)
    old.write_lines(ROOT / "daily-diagnosis.jsonl", daily)
    periods = {}
    for period in PERIODS:
        rows = [r for r in daily if m.matches(r["target"], period)]
        if not rows:
            continue
        periods[period] = {
            "metrics": {key: metrics(rows, key) for key in METHODS},
            "removal": {key: flips(rows, "full", "without_" + key) for key in ("information", "market", "history")},
            "all_branches_wrong": sum(r["all_branches_wrong"] for r in rows),
            "all_branches_agree": sum(r["all_branches_agree"] for r in rows),
        }
    old.write(ROOT / "branch-diagnosis.json", periods)
    prefixes = list(dict.fromkeys(r["fit_prefix"] for r in selected))
    first_cutoff = next(r["as_of"] for r in selected if r["fit_prefix"] == prefixes[0])
    probe = [r for r in joint if r["label_mature_at"] < first_cutoff and r["target"] < first_cutoff[:10]][-60:]
    assert len(probe) == 60
    old.write(
        ROOT / "probe-definition.json",
        {"targets": [r["target"] for r in probe], "first_cutoff": first_cutoff, "in_sample_probe_not_validation": True},
    )
    folds, model_readbacks, max_delta = [], 0, 0.0
    range_daily = {r["target"]: {} for r in daily}
    for prefix in prefixes:
        evaluation = [r for r in selected if r["fit_prefix"] == prefix]
        first = evaluation[0]
        current_report = report_by_day[first["target"]]
        fold = {
            "prefix": prefix,
            "as_of": first["as_of"],
            "evaluation_start": first["target"],
            "evaluation_end": evaluation[-1]["target"],
            "current_report_end": current_report["report_end"],
            "branches": {},
        }
        for branch in ("information", "market", "history"):
            path = (
                m.prior.ROOT / "models" / ("C_EVENT_" + prefix + "information.joblib")
                if branch == "information"
                else old.ROOT / "models" / (prefix + branch + ".joblib")
            )
            saved = joblib.load(path)
            receipt = old.read(path.parent.parent / "fits" / (path.stem + ".json"))
            assert old.sha(path) == receipt["model_sha256"]
            training = [all_history[d] for d in receipt["training_targets"]]
            if branch == "information":
                training = [joint_map[d] for d in receipt["training_targets"]]
            assert all(r["label_mature_at"] < saved["cutoff"] and r["target"] < saved["cutoff"][:10] for r in training)
            targets = [r["target"] for r in evaluation]
            values = manual_probabilities(saved, [joint_map[d] for d in targets], branch)
            expected = np.array(
                [
                    r["information"] if branch == "information" else base[r["target"]]["branches"][branch]
                    for r in evaluation
                ]
            )
            delta = float(np.abs(values - expected).max())
            assert delta <= 1e-12
            max_delta = max(max_delta, delta)
            probe_values = manual_probabilities(saved, probe, branch)
            similarity = [overlap(vectors[d], top_ten(current_report)) for d in receipt["training_targets"]]
            known = [v for v in similarity if v is not None]
            older = sum(
                bisect.bisect_left(sessions, first["target"]) - bisect.bisect_left(sessions, r["target"]) > 252
                for r in training
            )
            counts = Counter(r["target"][:4] for r in training)
            matrix = np.array([r["groups"][branch] for r in training], dtype=float)
            active = saved["preprocessing"]["keep"]
            for row in evaluation:
                x = np.array(joint_map[row["target"]]["groups"][branch], dtype=float)
                present = ~np.isnan(x)
                indices = np.flatnonzero(active & present)
                outliers = [
                    int(j)
                    for j in indices
                    if x[j] < np.nanmin(matrix[:, j]) - 1e-12 or x[j] > np.nanmax(matrix[:, j]) + 1e-12
                ]
                range_daily[row["target"]][branch] = {
                    "retained_columns": int(active.sum()),
                    "missing_retained": int((active & ~present).sum()),
                    "outside_training_range": outliers,
                    "present_but_training_all_missing": [int(j) for j in np.flatnonzero(~active & present)],
                }
            fold["branches"][branch] = {
                "training_count": len(training),
                "training_start": training[0]["target"],
                "training_end": training[-1]["target"],
                "training_year_counts": dict(counts),
                "training_up_fraction": sum(r["label"] == "UP" for r in training) / len(training),
                "older_than_252_sessions": older,
                "pre_2024_04_17_samples": sum(r["as_of"] < "2024-04-17T00:00:00+08:00" for r in training),
                "known_holdings_rows": len(known),
                "holdings_unknown_rows": len(similarity) - len(known),
                "mean_top10_overlap": float(np.mean(known)) if known else None,
                "overlap_under_half_rows": sum(v < 0.5 for v in known),
                "retained_columns": int(active.sum()),
                "fixed_probe_mean_p_up": float(probe_values[:, 2].mean()),
                "fixed_probe_predicted_up": int((probe_values.argmax(axis=1) == 2).sum()),
                "model_path": str(path),
            }
            model_readbacks += 1
        folds.append(fold)
    old.write_lines(ROOT / "training-and-probe-diagnosis.jsonl", folds)
    old.write(ROOT / "input-range-diagnosis.json", range_daily)
    input_summary = {}
    for period in PERIODS:
        rows = [r for r in daily if m.matches(r["target"], period)]
        if not rows:
            continue
        n = len(rows)
        indices = [joint_map[r["target"]]["groups"]["information"] for r in rows]
        summary = {
            "n": n,
            "related_public_days": sum(bool(r["public_event_ids"]) for r in rows),
            "qualified_company_days": sum(bool(r["company_event_ids"]) for r in rows),
            "numeric_revenue_days": sum(x[2] is not None for x in indices),
            "numeric_profit_days": sum(x[3] is not None for x in indices),
            "numeric_order_days": sum(x[4] is not None for x in indices),
            "evidence_states": dict(Counter(r["quality"] for r in rows)),
            "branches": {},
        }
        for branch in ("information", "market", "history"):
            parts = [range_daily[r["target"]][branch] for r in rows]
            summary["branches"][branch] = {
                "mean_missing_fraction_retained": float(
                    np.mean([p["missing_retained"] / p["retained_columns"] for p in parts])
                ),
                "any_range_exceedance_days": sum(bool(p["outside_training_range"]) for p in parts),
                "new_present_column_dropped_days": sum(bool(p["present_but_training_all_missing"]) for p in parts),
                "range_exceedance_counts": dict(Counter(j for p in parts for j in p["outside_training_range"])),
            }
        input_summary[period] = summary
    old.write(ROOT / "input-coverage-diagnosis.json", input_summary)
    reports = []
    previous = None
    for report in sorted(source["reports"], key=lambda r: r["available_at"]):
        if old.choose_report(source["reports"], report["available_at"])["raw"]["sha256"] != report["raw"]["sha256"]:
            continue
        item = {
            "end": report["report_end"],
            "available_at": report["available_at"],
            "overlap_previous_visible_top10": overlap(top_ten(report), top_ten(previous)),
            "holdings": [
                {"code": h["stock_code"], "name": h["stock_name"], "weight_pct": h["nav_weight_pct"]}
                for h in sorted(report["holdings"], key=lambda h: int(h["reported_rank"]))[:10]
            ],
            "source_hash": report["raw"]["sha256"],
        }
        reports.append(item)
        previous = report
    old.write(ROOT / "holdings-timeline.json", reports)
    # 两套独立实现交叉核对计数和算术重组；不训练任何探针或解释模型。
    for period, results in periods.items():
        rows = [r for r in daily if m.matches(r["target"], period)]
        for method, result in results["metrics"].items():
            manual = sum(
                old.CLASSES[max(range(3), key=lambda j: r["probabilities"][method][j])] == r["label"] for r in rows
            )
            assert manual == result["correct"]
            assert all(math.isclose(sum(r["probabilities"][method]), 1, abs_tol=1e-12) for r in rows)
        for branch, item in results["removal"].items():
            assert (
                item["net_correct"]
                == results["metrics"]["without_" + branch]["correct"] - results["metrics"]["full"]["correct"]
            )
    before = old.read(ROOT / "protection-before.json")
    changed = [
        p
        for repo in before.values()
        for p, digest in repo["files"].items()
        if not Path(p).exists() or old.sha(Path(p)) != digest
    ]
    assert not changed, changed
    assert all(old.sha(Path(p)) == digest for p, digest in protected.items())
    old.write(
        ROOT / "verification.json",
        {
            "passed": True,
            "at": old.now(),
            "new_fits": 0,
            "models_read_back": model_readbacks,
            "probability_rows_recomputed": 3 * len(daily),
            "maximum_error": max_delta,
            "fixed_probe_inputs": len(probe),
            "probe_is_validation": False,
            "existing_files_protected": sum(len(v["files"]) for v in before.values()),
            "old_workspace_changes": changed,
            "source_hashes_unchanged": True,
            "code_sha256": old.sha(Path(__file__)),
        },
    )
    print(
        old.json.dumps(
            {
                "root": str(ROOT),
                "branch_scores": {p: v for p, v in periods.items() if p in ("2026Q2", "2026Q3", "ALL")},
                "first_fold": folds[0],
                "last_fold": folds[-1],
                "coverage": {p: input_summary[p] for p in ("2025", "2026")},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
