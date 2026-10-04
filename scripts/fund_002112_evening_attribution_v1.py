"""固定四组信息表达、五种融合规则，分开检验内容/时效/价格反应及权重。

只使用上一轮的冻结材料和预测，两个已有方案不重训；本轮最多48次新拟合。
2025仅用于选择，2026已被观察，只作历史诊断。所有输出均为研究产物。
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from scripts import fund_002112_evening_refine_v1 as prior

old = prior.old
ROOT = old.RESEARCH / "evening-attribution-weights/20261003-v1"
GROUPS = ("CONTENT", "TIMING", "REACTION", "BOTH")
NEW_GROUPS = ("TIMING", "REACTION")
REUSED = {"CONTENT": "B_LINK", "BOTH": "C_EVENT"}
LABELS = {
    "CONTENT": "内容",
    "TIMING": "内容＋新增与时效",
    "REACTION": "内容＋价格反应",
    "BOTH": "内容＋新增时效＋价格反应",
}
REACTION_FIELDS = (
    "public_observed_reaction",
    "public_reaction_coverage",
    "public_no_complete_session_weight",
    "company_observed_reaction",
    "company_reaction_coverage",
    "company_no_complete_session_weight",
)
FIXED = {"FIXED_25": 0.25, "FIXED_40": 0.40, "FIXED_55": 0.55, "FIXED_70": 0.70}
POLICIES = (*FIXED, "QUALITY_GATE")
QUALITY_WEIGHTS = {"STRONG": 0.55, "CONTEXT": 0.25, "NONE": 0.0}
MAX_FITS = 48


def feature_names(group: str) -> list[str]:
    base, both = prior.feature_names("B_LINK"), prior.feature_names("C_EVENT")
    if group == "CONTENT":
        return base
    if group == "TIMING":
        return [name for name in both if name not in REACTION_FIELDS]
    if group == "REACTION":
        return base + list(REACTION_FIELDS)
    if group == "BOTH":
        return both
    raise ValueError("UNKNOWN_INFORMATION_GROUP")


def project_row(base: dict, both: dict, group: str) -> dict:
    """只改变信息列；同日市场、净值、标签和可用时间必须逐项保持相同。"""
    assert {k: v for k, v in base.items() if k != "groups"} == {k: v for k, v in both.items() if k != "groups"}
    assert base["groups"]["market"] == both["groups"]["market"]
    assert base["groups"]["history"] == both["groups"]["history"]
    names = prior.feature_names("C_EVENT")
    values = both["groups"]["information"]
    if group == "CONTENT":
        information = base["groups"]["information"][:]
    elif group == "REACTION":
        information = base["groups"]["information"] + [values[names.index(name)] for name in REACTION_FIELDS]
    elif group in ("TIMING", "BOTH"):
        information = [values[names.index(name)] for name in feature_names(group)]
    else:
        raise ValueError("UNKNOWN_INFORMATION_GROUP")
    return {**base, "groups": {**base["groups"], "information": information}}


def quality(row: dict, proof: dict, companies: dict, sessions: list[str]) -> dict:
    """只由截点前已保存的事实划分证据状态，不使用涨跌标签或消息后的价格。

    强证据：两个交易预测窗口内，已核实的营收/利润/订单数值覆盖至少3%已披露持仓。
    其他合格公告或有关联的公开材料为背景；缺证据时降低融合占比，原始缺值仍为None。
    """
    assert row["target"] == proof["target"] and row["as_of"] == proof["as_of"]
    evenings = old.prediction_evenings(sessions)
    index = sessions.index(row["base"])
    weights, reasons = {}, []
    for link in proof["company_events"]:
        event = companies[link["id"]]
        stamp = event["version_available_at"]
        assert stamp <= row["as_of"] and link["available_at"] == stamp
        age = index - bisect.bisect_left(evenings, stamp)
        metrics = [
            name
            for name in ("revenue_yoy", "profit_yoy", "order_ratio")
            if name in event["numeric"]
            and event["numeric"][name].get("value") is not None
            and event["numeric"][name].get("quote")
        ]
        if event["status"] == "QUALIFIED" and 0 <= age < 2 and metrics:
            weights[event["code"]] = max(weights.get(event["code"], 0), link["weight"])
            reasons.append(
                {
                    "id": event["id"],
                    "code": event["code"],
                    "age": age,
                    "weight": link["weight"],
                    "metrics": metrics,
                    "available_at": stamp,
                }
            )
    for event in proof["public_events"]:
        assert event["available_at"] <= row["as_of"]
    coverage = sum(weights.values())
    state = "STRONG" if coverage >= 0.03 else "CONTEXT" if proof["public_events"] or proof["company_events"] else "NONE"
    return {
        "target": row["target"],
        "as_of": row["as_of"],
        "state": state,
        "strong_coverage": coverage,
        "strong_events": reasons,
        "public_events": len(proof["public_events"]),
        "company_events": len(proof["company_events"]),
        "information_weight": QUALITY_WEIGHTS[state],
    }


def weights_for(policy: str, evidence: dict) -> dict:
    """仅调整信息分支占比，剩余市场/净值仍为2:1，避免同时改变两个问题。"""
    if policy == "QUALITY_GATE":
        alpha = QUALITY_WEIGHTS[evidence["state"]]
    elif policy in FIXED:
        alpha = FIXED[policy]
    else:
        raise ValueError("UNKNOWN_FUSION_POLICY")
    return {"information": alpha, "market": (1 - alpha) * 2 / 3, "history": (1 - alpha) / 3}


def blend(info, market, history, policy: str, evidence: dict) -> list:
    w = weights_for(policy, evidence)
    result = sum(w[key] * np.asarray(p) for key, p in [("information", info), ("market", market), ("history", history)])
    if not np.isfinite(result).all() or not np.isclose(result.sum(), 1) or (result < 0).any():
        raise ValueError("INVALID_FUSION_PROBABILITIES")
    return result.tolist()


def choose(candidates: dict, complexity: dict) -> str:
    """选择规则先冻结：正确数优先，其次概率误差，再按复杂度和名字确定性打破平局。"""
    return min(
        candidates, key=lambda name: (-candidates[name]["correct"], candidates[name]["brier"], complexity[name], name)
    )


def metric(rows, field="probabilities"):
    return old.metrics(rows, field)


def prepare(root: Path) -> None:
    if root.exists():
        raise ValueError("NEW_RUN_DIRECTORY_REQUIRED")
    prior.check_freeze(prior.ROOT)
    assert old.read(prior.ROOT / "independent-verification.json")["passed"]
    root.mkdir(parents=True)
    old.write(root / "protection-before.json", old.git_snapshot())
    protected = {
        str(path): old.sha(path) for parent in (old.ROOT, prior.ROOT) for path in parent.rglob("*") if path.is_file()
    }
    old.write(root / "prior-artifacts.json", {"files": protected})
    base = old.lines(prior.ROOT / "B_LINK-inputs.jsonl")
    both = old.lines(prior.ROOT / "C_EVENT-inputs.jsonl")
    for group in GROUPS:
        old.write_lines(
            root / (group + "-inputs.jsonl"), [project_row(a, b, group) for a, b in zip(base, both, strict=True)]
        )
    old.write(root / "feature-names.json", {g: feature_names(g) for g in GROUPS})
    source = old.read(old.ROOT / "sources.json")
    companies = {e["id"]: e for e in source["company_events"]}
    proofs = old.lines(prior.ROOT / "feature-lineage.jsonl")
    evidence = [quality(row, p, companies, source["sessions"]) for row, p in zip(base, proofs, strict=True)]
    old.write_lines(root / "quality-evidence.jsonl", evidence)
    coverage = {
        year: dict(Counter(e["state"] for e in evidence if year == "ALL" or e["target"].startswith(year)))
        for year in ("2024", "2025", "2026", "ALL")
    }
    old.write(
        root / "coverage.json",
        {
            "rows": len(base),
            "quality": coverage,
            "feature_counts": {g: len(feature_names(g)) for g in GROUPS},
            "new_paid_calls": 0,
            "new_external_data": 0,
        },
    )
    print(old.json.dumps(old.read(root / "coverage.json"), ensure_ascii=False), flush=True)


def freeze(root: Path) -> None:
    assert old.read(root / "tests.json")["passed"]
    protocol = {
        "at": old.now(),
        "fund": "002112",
        "data_scope": "上一轮665个相同日期及原有冻结证据；不新增外部数据",
        "groups": LABELS,
        "new_groups": NEW_GROUPS,
        "reused_groups": REUSED,
        "reaction_fields": REACTION_FIELDS,
        "timing_bundle": "公开信息新增比例、5交易日半衰期及非价格的时效/覆盖字段；组内因素本轮不再细拆",
        "recipe": old.RECIPE,
        "fixed_weights": FIXED,
        "quality_weights": QUALITY_WEIGHTS,
        "quality_rule": (
            "营收/利润/订单原文数值，发布后两个预测窗口，持仓覆盖>=3%为强；"
            "其他关联信息为背景；余为无可用证据"
        ),
        "baseline_market_history_ratio": "2:1",
        "max_fits": MAX_FITS,
        "rolling_fits": 46,
        "full_and_replay_max_fits": 2,
        "selection": "只用2025：先在55%融合下选四种信息表达之一；再仅对所选表达比较四个固定权重和一个证据规则",
        "tie_break": "正确数降序、Brier升序、少特征/固定优于条件规则、名称",
        "2025_status": "已多次使用的选择期，所选成绩有选择偏差，不宣称未见检验",
        "2026_status": "已见历史诊断；不得根据其成绩反选表达或权重",
        "selection_order": "保存两步选择回执后再运行2026新拟合",
        "historical_gate": (
            "2025和2026均比上一轮修正数字对照A正确更多且Brier更低；"
            "下跌识别下降<=5个百分点；各年至少半数季度不差"
        ),
        "formal_adoption": False,
        "adoption_reason": "历史首见存档不完整，未完成事前预测的未来验证",
        "limitations": "小样本及重复使用历史；强证据不等于消息必然影响价格；两种因素可能相互作用，不能把增益当因果贡献",
    }
    old.write(root / "protocol.json", protocol)
    paths = list(root.glob("*.json")) + list(root.glob("*.jsonl"))
    code = [
        Path(__file__),
        old.PY / "scripts/fund_002112_evening_attribution_verify_v1.py",
        old.PY / "scripts/test_fund_002112_evening_attribution_v1.py",
        Path(prior.__file__),
        Path(old.__file__),
    ]
    (root / "code").mkdir()
    for path in code:
        (root / "code" / path.name).write_bytes(path.read_bytes())
    old.write(root / "freeze.json", {"files": {str(p.resolve()): old.sha(p) for p in paths + code}})


def verify_hashes(root: Path) -> None:
    for name in ("freeze.json", "prior-artifacts.json"):
        for path, digest in old.read(root / name)["files"].items():
            if old.sha(Path(path)) != digest:
                raise ValueError("FROZEN_ARTIFACT_CHANGED:" + path)


def fit(root, name, rows, cutoff, evaluate):
    ledger = root / "fit-ledger.jsonl"
    if ledger.exists() and len(old.lines(ledger)) >= MAX_FITS:
        raise ValueError("FIT_CAP_REACHED")
    return old.fit_expert(root, name, "information", rows, cutoff, evaluate)


def full_bundle(root, selected, policy, datasets, reference):
    """最终研究包明确保存融合规则；条件权重不得伪装成固定55%。"""
    if selected == "BOTH":
        expert = joblib.load(prior.ROOT / "models/FULL_C_EVENT.joblib")
        full_origin = str(prior.ROOT / "models/FULL_C_EVENT.joblib")
        replay_error = None
    else:
        cutoff = old.now()
        expert = fit(root, "FULL_" + selected, datasets[selected], cutoff, [])
        replay = fit(root, "REPLAY_" + selected, datasets[selected], cutoff, datasets[selected][-8:])
        replay_error = float(
            np.abs(
                old.predict_expert(expert, datasets[selected][-8:], "information")
                - old.predict_expert(replay, datasets[selected][-8:], "information")
            ).max()
        )
        assert replay_error <= 1e-12
        full_origin = str(root / "models" / ("FULL_" + selected + ".joblib"))
    original = joblib.load(old.ROOT / "002112-evening-1d-news55.joblib")
    bundle = {
        "schema": "002112_EVENING_ATTRIBUTION_RESEARCH_V1",
        "fund": "002112",
        "trained_at": old.now(),
        "experts": {**original["experts"], "information": expert},
        "information_group": selected,
        "fusion_policy": policy,
        "fixed_weights": weights_for(policy, {}) if policy != "QUALITY_GATE" else None,
        "quality_weights": QUALITY_WEIGHTS,
        "features": {**original["features"], "information": feature_names(selected)},
        "classes": old.CLASSES,
        "as_of_rule": original["as_of_rule"],
        "label_end": original["label_end"],
        "research_only": True,
        "historical_first_seen_proven": False,
        "production_eligible": False,
        "freeze_sha256": old.sha(root / "freeze.json"),
        "full_information_origin": full_origin,
        "inference_entrypoint": "scripts.fund_002112_evening_attribution_v1.predict_bundle",
        "feature_builder": "project_row: frozen B_LINK/C_EVENT inputs; quality: as-of evidence only",
        "calibrated": False,
    }
    joblib.dump(bundle, root / "002112-evening-attribution.joblib")
    if selected == "BOTH":
        reference[full_origin] = old.sha(Path(full_origin))
    for branch in ("market", "history"):
        path = old.ROOT / "models" / ("FULL_" + branch + ".joblib")
        reference[str(path)] = old.sha(path)
    return replay_error


def predict_bundle(bundle: dict, rows: list, evidence: dict) -> list:
    """输入须按保存的信息表达构造；逐行使用证据状态选择融合占比。"""
    branches = {key: old.predict_expert(expert, rows, key) for key, expert in bundle["experts"].items()}
    return [
        blend(
            branches["information"][i],
            branches["market"][i],
            branches["history"][i],
            bundle["fusion_policy"],
            evidence[row["target"]],
        )
        for i, row in enumerate(rows)
    ]


def train(root: Path) -> None:
    verify_hashes(root)
    if (root / "fit-ledger.jsonl").exists():
        raise ValueError("TRAINING_ALREADY_STARTED")
    datasets = {g: old.lines(root / (g + "-inputs.jsonl")) for g in GROUPS}
    evidence = {r["target"]: r for r in old.lines(root / "quality-evidence.jsonl")}
    baseline = {r["target"]: r for r in old.lines(old.ROOT / "historical-predictions.jsonl")}
    previous = {(r["group"], r["target"]): r for r in old.lines(prior.ROOT / "predictions.jsonl")}
    output, reference, selections = [], {}, {}
    for year in ("2025", "2026"):
        for group in GROUPS:
            evaluation = [r for r in datasets[group] if r["target"].startswith(year)]
            for start in range(0, len(evaluation), 20):
                block = evaluation[start : start + 20]
                prefix = year + f"_{start:03d}_"
                if group in REUSED:
                    identifier = REUSED[group] + "_" + prefix + "information"
                    path = prior.ROOT / "models" / (identifier + ".joblib")
                    info = np.array([previous[(REUSED[group], r["target"])]["information"] for r in block])
                    fit_meta = old.read(prior.ROOT / "fits" / (identifier + ".json"))
                    assert old.sha(path) == fit_meta["model_sha256"]
                    reference[str(path)] = fit_meta["model_sha256"]
                else:
                    identifier = group + "_" + prefix + "information"
                    model = fit(root, identifier, datasets[group], block[0]["as_of"], block)
                    path = root / "models" / (identifier + ".joblib")
                    info = old.predict_expert(model, block, "information")
                for row, p in zip(block, info, strict=True):
                    ref = baseline[row["target"]]
                    assert ref["as_of"] == row["as_of"] and ref["fit_prefix"] == prefix
                    q = evidence[row["target"]]
                    output.append(
                        {
                            "group": group,
                            "target": row["target"],
                            "as_of": row["as_of"],
                            "label": row["label"],
                            "information": p.tolist(),
                            "model_path": str(path),
                            "fit_prefix": prefix,
                            "probabilities": blend(
                                p, ref["branches"]["market"], ref["branches"]["history"], "FIXED_55", q
                            ),
                        }
                    )
                for branch in ("market", "history"):
                    path = old.ROOT / "models" / (prefix + branch + ".joblib")
                    reference[str(path)] = old.sha(path)
            print(f"完成 {year} {group}：{len(evaluation)}日，{'复用' if group in REUSED else '新拟合'}", flush=True)
        if year == "2025":
            scores = {g: metric([r for r in output if r["group"] == g]) for g in GROUPS}
            selected = choose(scores, {g: len(feature_names(g)) for g in GROUPS})
            weight_rows = []
            for row in output:
                if row["group"] != selected:
                    continue
                ref, q = baseline[row["target"]], evidence[row["target"]]
                for policy in POLICIES:
                    weight_rows.append(
                        {
                            **row,
                            "policy": policy,
                            "probabilities": blend(
                                row["information"], ref["branches"]["market"], ref["branches"]["history"], policy, q
                            ),
                        }
                    )
            weight_scores = {p: metric([r for r in weight_rows if r["policy"] == p]) for p in POLICIES}
            selected_policy = choose(weight_scores, {p: int(p == "QUALITY_GATE") for p in POLICIES})
            selections = {
                "at": old.now(),
                "group": selected,
                "policy": selected_policy,
                "representation_scores_2025": scores,
                "weight_scores_2025": weight_scores,
                "selected_before_2026_new_fits": True,
                "selection_uses_2026": False,
            }
            old.write(root / "selection.json", selections)
    old.write_lines(root / "factorial-predictions.jsonl", output)
    weighted = []
    for row in output:
        if row["group"] == selections["group"]:
            ref, q = baseline[row["target"]], evidence[row["target"]]
            for policy in POLICIES:
                weighted.append(
                    {
                        **row,
                        "policy": policy,
                        "evidence_state": q["state"],
                        "weights": weights_for(policy, q),
                        "probabilities": blend(
                            row["information"], ref["branches"]["market"], ref["branches"]["history"], policy, q
                        ),
                    }
                )
    old.write_lines(root / "weight-predictions.jsonl", weighted)
    scores, factorial, weights = {}, {}, {}
    periods = ["2025", "2026", "ALL"] + [f"{year}Q{q}" for year in ("2025", "2026") for q in range(1, 5)]
    corrected = {(r["group"], r["target"]): r for r in old.lines(prior.ROOT / "predictions.jsonl")}
    for period in periods:
        rows = [r for r in output if matches(r["target"], period)]
        if not rows:
            continue
        factorial[period] = {g: metric([r for r in rows if r["group"] == g]) for g in GROUPS}
        wrows = [r for r in weighted if matches(r["target"], period)]
        weights[period] = {p: metric([r for r in wrows if r["policy"] == p]) for p in POLICIES}
        oldrows = [r for r in baseline.values() if matches(r["target"], period)]
        scores[period] = {
            "SELECTED": weights[period][selections["policy"]],
            "A_FIX": metric([corrected[("A_FIX", r["target"])] for r in oldrows]),
            "PRIOR_C": metric([corrected[("C_EVENT", r["target"])] for r in oldrows]),
            "PRICE": metric(oldrows, "price_history"),
            "NAV": metric(oldrows, "history_only"),
            "ALWAYS_UP": metric(oldrows, "always_up"),
        }
    old.write(root / "factorial-scores.json", factorial)
    old.write(root / "weight-scores.json", weights)
    old.write(root / "scores.json", scores)
    replay_error = full_bundle(root, selections["group"], selections["policy"], datasets, reference)
    old.write(root / "reused-models.json", {"files": reference})
    ledger = old.lines(root / "fit-ledger.jsonl")
    decision = {
        "at": old.now(),
        "selected_group": selections["group"],
        "selected_policy": selections["policy"],
        "historical_gate": prior.gate(scores, "SELECTED"),
        "actual_new_fits": len(ledger),
        "full_model_reused": selections["group"] == "BOTH",
        "replay_error": replay_error,
        "adopted": False,
        "production_eligible": False,
        "historical_first_seen_proven": False,
        "reason": "研究试验结束；历史检验结果单独报告，未见未来验证和历史首见存档仍不足",
    }
    old.write(root / "decision.json", decision)
    print(
        old.json.dumps(
            {"selection": selections, "2026": scores["2026"], "ALL": scores["ALL"], "decision": decision},
            ensure_ascii=False,
        ),
        flush=True,
    )


def matches(target: str, period: str) -> bool:
    return period == "ALL" or (
        target.startswith(period[:4]) and (len(period) == 4 or (int(target[5:7]) - 1) // 3 + 1 == int(period[-1]))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "freeze", "train"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "train": train}[args.stage](args.root)
