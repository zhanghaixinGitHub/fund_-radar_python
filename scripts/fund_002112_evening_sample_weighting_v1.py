"""002112单因素历史对照：只改变训练样本权重，固定材料、特征、算法和融合比例。

三个方案全部使用同一批合格训练日期。原方案直接复用；近期和持仓方案各做23期、
每期3分支，共138次拟合。既有2025/2026结果已经观察，不能称为盲测或未来验证。
"""

from __future__ import annotations

import argparse
import bisect
import math
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from scripts import fund_002112_evening_failure_audit_v1 as audit

m, old = audit.m, audit.old
ROOT = old.RESEARCH / "evening-sample-weighting/20261003-v1"
POLICIES = ("ALL", "RECENT", "SIMILAR")
WEIGHTS = {"information": 0.4, "market": 0.4, "history": 0.2}
FIT_CAP = 138
HALF_LIFE = 126
SIMILARITY_BOOST = 3.0


def training_rows(rows: list, branch: str, cutoff: str) -> list:
    """只允许截点前已经成熟的标签进入训练；保留原分支全部合格日期。"""
    return [r for r in rows if branch in r["groups"] and r["label_mature_at"] < cutoff and r["target"] < cutoff[:10]]


def sample_weights(policy: str, rows: list, cutoff: str, source: dict) -> tuple[np.ndarray, dict]:
    """按训练截点计算权重；不读取评估期标签、未来持仓或未来涨跌。

    近期权重每126交易日减半。持仓按各样本当时可见的前十持仓与训练截点可见的
    前十持仓计算重合度s，原始权重为1+3s。未披露持仓用已知样本原始权重的均值，
    归一化后恰为1，既不假定不相似，也不奖励未知；已知权重最大最小比不超过4。
    """
    if policy not in POLICIES or not rows:
        raise ValueError("UNKNOWN_POLICY_OR_EMPTY_TRAINING")
    if any(r["as_of"] > cutoff or r["target"] >= cutoff[:10] or r["label_mature_at"] >= cutoff for r in rows):
        raise ValueError("TRAINING_TIME_LEAKAGE")
    anchor = bisect.bisect_right(source["sessions"], cutoff[:10]) - 1
    ages = np.array([anchor - bisect.bisect_left(source["sessions"], r["target"]) for r in rows])
    assert (ages >= 0).all()
    current = old.choose_report(source["reports"], cutoff)
    overlaps, reports = [], []
    for row in rows:
        report = old.choose_report(source["reports"], row["as_of"])
        reports.append(report)
        overlaps.append(audit.overlap(audit.top_ten(report), audit.top_ten(current)))
    if policy == "RECENT":
        raw = np.exp2(-ages / HALF_LIFE)
    elif policy == "SIMILAR":
        known = [1 + SIMILARITY_BOOST * v for v in overlaps if v is not None]
        neutral = float(np.mean(known)) if known else 1.0
        raw = np.array([neutral if v is None else 1 + SIMILARITY_BOOST * v for v in overlaps])
    else:
        raw = np.ones(len(rows))
    weights = raw / raw.mean()
    assert np.isfinite(weights).all() and (weights > 0).all()
    old_mask = ages > 252
    unknown = np.array([v is None for v in overlaps])
    proof = {
        "policy": policy,
        "cutoff": cutoff,
        "current_report_end": current["report_end"] if current else None,
        "current_report_available_at": current["available_at"] if current else None,
        "current_report_hash": current["raw"]["sha256"] if current else None,
        "rows": [
            {
                "target": row["target"],
                "age_sessions": int(age),
                "top10_overlap": overlap,
                "sample_report_end": report["report_end"] if report else None,
                "sample_report_available_at": report["available_at"] if report else None,
                "sample_report_hash": report["raw"]["sha256"] if report else None,
                "weight": float(weight),
            }
            for row, age, overlap, report, weight in zip(rows, ages, overlaps, reports, weights, strict=True)
        ],
        "summary": {
            "n": len(rows),
            "older_than_year_count": int(old_mask.sum()),
            "older_than_year_weight_share": float(weights[old_mask].sum() / weights.sum()),
            "unknown_holdings_count": int(unknown.sum()),
            "unknown_holdings_weight_share": float(weights[unknown].sum() / weights.sum()),
            "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "weighted_up_fraction": float(
                sum(w for r, w in zip(rows, weights, strict=True) if r["label"] == "UP") / weights.sum()
            ),
        },
    }
    return weights, proof


def baseline_path(prefix: str, branch: str) -> Path:
    return (
        m.prior.ROOT / "models" / ("C_EVENT_" + prefix + "information.joblib")
        if branch == "information"
        else old.ROOT / "models" / (prefix + branch + ".joblib")
    )


def inputs():
    joint = old.lines(m.ROOT / "BOTH-inputs.jsonl")
    history = old.lines(old.ROOT / "dataset.jsonl")
    baseline = [r for r in old.lines(m.ROOT / "weight-predictions.jsonl") if r["policy"] == "FIXED_40"]
    return joint, history, baseline, old.read(old.ROOT / "sources.json")


def verify_freeze(root: Path) -> None:
    for name in ("freeze.json", "prior-artifacts.json"):
        for p, digest in old.read(root / name)["files"].items():
            if old.sha(Path(p)) != digest:
                raise ValueError("FROZEN_INPUT_CHANGED:" + p)


def prepare(root: Path) -> None:
    """先冻结规则及来源，禁止见到本轮分数后改半衰期、持仓倍率或判定门槛。"""
    m.verify_hashes(m.ROOT)
    assert old.read(audit.ROOT / "verification.json")["passed"]
    if root.exists():
        raise ValueError("FRESH_RUN_DIRECTORY_REQUIRED")
    joint, history, baseline, _ = inputs()
    prefixes = list(dict.fromkeys(r["fit_prefix"] for r in baseline))
    assert len(prefixes) == 23 and len(baseline) == 424 and len(joint) == 665
    root.mkdir(parents=True)
    old.write(root / "protection-before.json", old.git_snapshot())
    protected = {
        str(p): old.sha(p)
        for parent in (old.ROOT, m.prior.ROOT, m.ROOT, audit.ROOT)
        for p in parent.rglob("*")
        if p.is_file()
    }
    old.write(root / "prior-artifacts.json", {"files": protected})
    old.write(
        root / "protocol.json",
        {
            "at": old.now(),
            "policies": POLICIES,
            "weights": WEIGHTS,
            "recipe": old.RECIPE,
            "recent_half_life_sessions": HALF_LIFE,
            "similarity_raw_weight": "1+3*normalized_top10_overlap",
            "unknown_holdings": "raw mean among known training rows; normalized weight=1; preserve null overlap",
            "retained_training_rows": "exact same dates as each existing branch; no sample deletion",
            "preprocessing": "unchanged unweighted training medians and missing indicators, verified against original",
            "reference_holdings": "latest available report at training cutoff; fixed during that 20-session block",
            "sample_holdings": "report available at the sample's own historical prediction cutoff; stock-code top10",
            "as_of": "target trading day's previous calendar day 23:00 Asia/Shanghai",
            "fit_cap": FIT_CAP,
            "reused_models": 69,
            "new_fits": 138,
            "new_full_fit": False,
            "periods": audit.PERIODS,
            "joint_rows": len(joint),
            "history_rows": len(history),
            "selection": "no winner selection or deployment; fixed paired historical comparisons only",
            "historical_gate": (
                "strictly more correct and lower Brier in BOTH 2025/2026 against BOTH ALL and prior A_FIX; "
                "down recall loss<=5pp; at least half quarters nonworse per year"
            ),
            "uncertainty": (
                "paired circular moving-block bootstrap length20,4000 draws,seed20261003; "
                "descriptive only, no unseen-test significance claim"
            ),
            "already_observed_years": [2025, 2026],
            "future_validation": False,
            "historical_first_seen_proven": False,
            "new_data": False,
            "paid_calls": 0,
            "production_eligible": False,
            "sklearn": sklearn.__version__,
            "sampling_note": (
                "installed sklearn1.9 implements sample_weight as weighted bootstrap; "
                "same estimator recipe/seed; one seed is not robustness proof"
            ),
        },
    )
    code = [
        Path(__file__),
        Path(audit.__file__),
        Path(m.__file__),
        Path(m.prior.__file__),
        Path(old.__file__),
        old.PY / "scripts/fund_002112_evening_sample_weighting_verify_v1.py",
        old.PY / "scripts/test_fund_002112_evening_sample_weighting_v1.py",
    ]
    (root / "code").mkdir()
    for path in code:
        (root / "code" / path.name).write_bytes(path.read_bytes())
    paths = code + [root / "protocol.json", root / "prior-artifacts.json", root / "protection-before.json"]
    old.write(root / "freeze.json", {"at": old.now(), "files": {str(p): old.sha(p) for p in paths}})
    print(old.json.dumps({"prepared": str(root), "fit_cap": FIT_CAP, "policies": POLICIES}), flush=True)


def fit(root: Path, name: str, branch: str, policy: str, rows: list, cutoff: str, block: list, source: dict):
    """真实拟合前记账；每个分支只新增样本权重，保存逐样本权重和可复算回执。"""
    selected = training_rows(rows, branch, cutoff)
    assert len(selected) >= 120
    weights, proof = sample_weights(policy, selected, cutoff, source)
    ledger = root / "fit-ledger.jsonl"
    entries = old.lines(ledger) if ledger.exists() else []
    if len(entries) >= FIT_CAP or any(r["id"] == name for r in entries):
        raise ValueError("FIT_CAP_OR_DUPLICATE")
    path = root / "models" / (name + ".joblib")
    if path.exists():
        raise ValueError("MODEL_ALREADY_EXISTS")
    weight_path = root / "sample-weights" / (name + ".json")
    old.write(weight_path, proof)
    entry = {
        "id": name,
        "branch": branch,
        "policy": policy,
        "at": old.now(),
        "cutoff": cutoff,
        "training_targets": [r["target"] for r in selected],
        "count": len(selected),
        "weight_file": str(weight_path),
        "weight_sha256": old.sha(weight_path),
    }
    x, preprocessing = old.preprocess(np.array([r["groups"][branch] for r in selected], dtype=float))
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(old.json.dumps(entry, ensure_ascii=False) + "\n")
    model = RandomForestClassifier(**old.RECIPE).fit(x, [r["label"] for r in selected], sample_weight=weights)
    saved = {
        "model": model,
        "preprocessing": preprocessing,
        "branch": branch,
        "cutoff": cutoff,
        "row_count": len(selected),
        "policy": policy,
        "research_only": True,
        "historical_first_seen_proven": False,
        "production_eligible": False,
        "freeze_sha256": old.sha(root / "freeze.json"),
        "weight_sha256": entry["weight_sha256"],
    }
    path.parent.mkdir(exist_ok=True)
    joblib.dump(saved, path)
    probabilities = old.predict_expert(saved, block, branch)
    old.write(
        root / "fits" / (name + ".json"),
        {
            **entry,
            "model_sha256": old.sha(path),
            "probabilities": probabilities.tolist(),
            "evaluation_targets": [r["target"] for r in block],
            "weight_summary": proof["summary"],
            "finished_at": old.now(),
        },
    )
    return probabilities


def paired_comparison(rows: list, policy: str) -> dict:
    """同日对照、按连续20日成块重抽，保留时间相关性；只是已观察历史的波动范围。"""
    base = {r["target"]: r for r in rows if r["policy"] == "ALL"}
    other = [r for r in rows if r["policy"] == policy]
    difference, corrected, spoiled = [], 0, 0
    for row in other:
        truth = old.CLASSES.index(row["label"])
        a = int(np.argmax(base[row["target"]]["probabilities"])) == truth
        b = int(np.argmax(row["probabilities"])) == truth
        corrected += int(b and not a)
        spoiled += int(a and not b)
        difference.append(int(b) - int(a))
    difference = np.array(difference)
    n = len(difference)
    rng = np.random.default_rng(20261003)
    starts = rng.integers(0, n, size=(4000, math.ceil(n / 20)))
    indices = ((starts[:, :, None] + np.arange(20)) % n).reshape(4000, -1)[:, :n]
    interval = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
    return {
        "n": n,
        "corrected": corrected,
        "spoiled": spoiled,
        "net_correct": corrected - spoiled,
        "accuracy_delta": float(difference.mean()),
        "descriptive_block_interval95": interval.tolist(),
    }


def passes_gate(scores: dict, policy: str, baseline: str) -> dict:
    adapted = {p: {policy: s[policy], "A_FIX": s[baseline]} for p, s in scores.items()}
    return m.prior.gate(adapted, policy)


def summarize(root: Path, output: list) -> None:
    previous = old.read(m.ROOT / "scores.json")
    scores, comparisons = {}, {}
    for period in audit.PERIODS:
        rows = [r for r in output if m.matches(r["target"], period)]
        if not rows:
            continue
        scores[period] = {p: old.metrics([r for r in rows if r["policy"] == p], "probabilities") for p in POLICIES}
        for p in POLICIES:
            subset = [r for r in rows if r["policy"] == p]
            scores[period][p]["predicted_up"] = sum(int(np.argmax(r["probabilities"])) == 2 for r in subset)
            scores[period][p]["branches"] = {
                b: old.metrics([{**r, "p": r["branches"][b]} for r in subset], "p") for b in WEIGHTS
            }
        for control in ("A_FIX", "NAV", "PRICE", "ALWAYS_UP"):
            scores[period][control] = previous[period][control]
        comparisons[period] = {p: paired_comparison(rows, p) for p in POLICIES[1:]}
    old.write(root / "scores.json", scores)
    old.write(root / "paired-comparisons.json", comparisons)
    gates = {p: {b: passes_gate(scores, p, b) for b in ("ALL", "A_FIX")} for p in POLICIES[1:]}
    old.write(
        root / "decision.json",
        {
            "at": old.now(),
            "new_fits": len(old.lines(root / "fit-ledger.jsonl")),
            "historical_gates": gates,
            "passed_all": {p: all(g["passed"] for g in gs.values()) for p, gs in gates.items()},
            "selected": None,
            "adopted": False,
            "future_validation": False,
            "production_eligible": False,
            "reason": "固定两项历史对照已完成；不继续调参、不自动选用，不把重复观察历史当未来验证",
        },
    )


def train(root: Path) -> None:
    verify_freeze(root)
    if (root / "fit-ledger.jsonl").exists():
        raise ValueError("TRAINING_ALREADY_STARTED_NO_IMPLICIT_RERUN")
    joint, history, baseline, source = inputs()
    joint_map = {r["target"]: r for r in joint}
    old_predictions = {r["target"]: r for r in old.lines(old.ROOT / "historical-predictions.jsonl")}
    prefixes = list(dict.fromkeys(r["fit_prefix"] for r in baseline))
    output, reused = [], {}
    for index, prefix in enumerate(prefixes):
        ref = [r for r in baseline if r["fit_prefix"] == prefix]
        block = [joint_map[r["target"]] for r in ref]
        cutoff = block[0]["as_of"]
        for branch in WEIGHTS:
            path = baseline_path(prefix, branch)
            reused[str(path)] = old.sha(path)
        for r in ref:
            branches = {
                "information": r["information"],
                **{b: old_predictions[r["target"]]["branches"][b] for b in ("market", "history")},
            }
            output.append(
                {
                    "policy": "ALL",
                    "target": r["target"],
                    "as_of": r["as_of"],
                    "label": r["label"],
                    "fit_prefix": prefix,
                    "branches": branches,
                    "probabilities": r["probabilities"],
                }
            )
        for policy in POLICIES[1:]:
            branches = {
                branch: fit(
                    root,
                    policy + "_" + prefix + branch,
                    branch,
                    policy,
                    joint if branch == "information" else history,
                    cutoff,
                    block,
                    source,
                )
                for branch in WEIGHTS
            }
            combined = old.blend(branches, WEIGHTS)
            for j, row in enumerate(block):
                output.append(
                    {
                        "policy": policy,
                        "target": row["target"],
                        "as_of": row["as_of"],
                        "label": row["label"],
                        "fit_prefix": prefix,
                        "probabilities": combined[j].tolist(),
                        "branches": {b: values[j].tolist() for b, values in branches.items()},
                    }
                )
        # 每块单独落盘，失败时已完成的结果仍可检查；不静默重跑已拟合部分。
        old.write_lines(root / "blocks" / (prefix + ".jsonl"), [r for r in output if r["fit_prefix"] == prefix])
        print(
            old.json.dumps({"completed_blocks": index + 1, "total_blocks": len(prefixes), "new_fits": (index + 1) * 6}),
            flush=True,
        )
    old.write(root / "reused-models.json", {"files": reused})
    old.write_lines(root / "predictions.jsonl", output)
    assert len(old.lines(root / "fit-ledger.jsonl")) == FIT_CAP
    summarize(root, output)
    verify_freeze(root)
    print(old.json.dumps(old.read(root / "decision.json"), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    {"prepare": prepare, "train": train}[args.stage](args.root)
