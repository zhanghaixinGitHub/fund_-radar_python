"""真正的下一交易日口径：预测日D早08:00，只用已知输入预测下一交易日U的净值方向。

独立于此前“目标日08:00判断当天”的研究。复用相同固定候选与算法，重新构造
输入日期/目标日期和标签成熟边界；不借下一交易日早晨的消息预测前一天。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

from scripts import fund_002112_all_information_search_v1 as kernel
from scripts import fund_002112_holding_impact_v1 as event

io = event.io
ROOT = io.RESEARCH / "next-trading-day-optimization/20260930-v1"


def dependencies(root: Path, label: str) -> dict:
    """按实际时点补充依赖快照，不把事后补充伪装成第一批fit之前的冻结。"""
    modules = (
        "fund_002112_existing_data_inputs_v1",
        "fund_002112_existing_events_sources_v1",
        "fund_002112_existing_events_features_v1",
        "fund_002112_existing_events_fit_v1",
        "fund_002112_holding_impact_v1",
        "fund_002112_holding_impact_fit_v1",
        "fund_002112_all_information_search_v1",
        "fund_002112_next_day_search_v1",
    )
    sources = {}
    target = root / "dependency-snapshot" / "scripts"
    target.mkdir(parents=True, exist_ok=True)
    for name in modules:
        src, dst = io.PY / "scripts" / (name + ".py"), target / (name + ".py")
        if dst.exists() and io.sha(dst) != io.sha(src):
            raise ValueError("DEPENDENCY_SNAPSHOT_CHANGED")
        if not dst.exists():
            shutil.copyfile(src, dst)
        sources[str(src)] = {"sha256": io.sha(src), "snapshot": str(dst)}
    manifest = {
        "at": io.now(),
        "phase": label,
        "not_claimed_as_earlier_freeze": True,
        "sources": sources,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "httpx", "pydantic")
        },
        "existing_environment": str(io.PY / ".venv/Scripts/python.exe"),
        "secrets_or_environment_values_saved": False,
    }
    io.save(root / "dependency-manifest.json", manifest)
    return manifest


def prepare(root: Path = ROOT) -> dict:
    """在晨间当日批次选择和2026评分之前登记新口径，避免用该评分调整新范围。"""
    if (root / "plan.json").exists():
        return io.read(root / "plan.json")
    previous = io.read(kernel.ROOT / "plan.json")
    plan = {
        **previous,
        "version": "TRUE_NEXT_TRADING_DAY_V1",
        "registered_at": io.now(),
        "prediction_cutoff": "PREVIOUS_TRADING_DATE_D_08:00_ASIA_SHANGHAI",
        "target": "NEXT_TRADING_DATE_U_UNIT_NAV_VS_D_UNIT_NAV_UP_FLAT_DOWN",
        "prior_morning_plan_sha256": io.sha(kernel.ROOT / "plan.json"),
        "new_comparison": "Separate horizon; no uplift claim against morning same-day results",
        "source_features_as_of": "D_08:00; never U_08:00",
        "training_cutoff": "FIRST_EVALUATION_TARGET_PREVIOUS_TRADING_DATE_08:00_EXCLUSIVE",
        "initial_row_boundary": "First target needs earlier predictor row; exact dates frozen at build",
        "simple_baselines": ["N_LR_C1", "ALWAYS_UP", "TRAINING_FOLD_MAJORITY"],
        "constraints": ["NO_NEW_DATA", "NO_SERVICE_RESTART", "NO_MODEL_ACTIVATION", "NO_FINAL_SCORE_TUNING"],
    }
    io.save(root / "plan.json", plan)
    dependencies(root, "BEFORE_ANY_TRUE_NEXT_DAY_FIT")
    return plan


def build(root: Path = ROOT) -> dict:
    """D的输入行与U的标签行分开拼接；训练可用标签按D的真实预测时点过滤。"""
    morning = io.lines(kernel.ROOT / "inputs.jsonl")
    by_date = {r["target"]: r for r in morning}
    rows, excluded = [], []
    for label_row in morning:
        origin = label_row["base"]
        source = by_date.get(origin)
        if source is None:
            excluded.append(
                {
                    "target": label_row["target"],
                    "origin": origin,
                    "reason": "NO_FROZEN_FEATURE_ROW_FOR_ORIGIN; common boundary for all candidates",
                }
            )
            continue
        if source["as_of"] != origin + "T08:00:00+08:00" or source["target"] >= label_row["target"]:
            raise ValueError("NEXT_DAY_INPUT_TIME_INVALID")
        rows.append(
            {
                **source,
                "target": label_row["target"],
                "base": origin,
                "prediction_origin": origin,
                "as_of": source["as_of"],
                "label_mature_at": label_row["label_mature_at"],
                "session_index": label_row["session_index"],
                "input_row_target": source["target"],
            }
        )
    splits = {}
    for fold in ("V1", "V2", "V3", "T", "FINAL"):
        start, end = kernel.old.EVAL_RANGE.get(fold, ("", ""))
        er = [i for i, r in enumerate(rows) if start and start <= r["target"] <= end]
        cutoff = rows[er[0]]["as_of"] if er else io.read(root / "plan.json")["registered_at"]
        tr = [
            i
            for i, r in enumerate(rows)
            if r["target"] <= kernel.old.TRAIN_END[fold] and r["label_mature_at"] < cutoff and r["as_of"] < cutoff
        ]
        if set(tr) & set(er):
            raise ValueError("NEXT_DAY_TRAIN_EVAL_OVERLAP")
        splits[fold] = {
            "training": tr,
            "evaluation": er,
            "cutoff_exclusive": cutoff,
            "train_dates": [rows[i]["target"] for i in tr],
            "evaluation_dates": [rows[i]["target"] for i in er],
        }
    io.save_lines(root / "inputs.jsonl", rows)
    io.save(root / "splits.json", splits)
    io.save(root / "structural-boundary-exclusions.json", excluded)
    io.save(
        root / "freeze.json",
        {
            "at": io.now(),
            "files": {
                str(p): io.sha(p)
                for p in (
                    root / "inputs.jsonl",
                    root / "splits.json",
                    root / "plan.json",
                    root / "dependency-manifest.json",
                    Path(__file__),
                    Path(kernel.__file__),
                    Path(kernel.impact.__file__),
                    Path(event.__file__),
                    kernel.ROOT / "inputs.jsonl",
                    sources_file(),
                )
            },
        },
    )
    return {
        "dates": len(rows),
        "structural_exclusions": excluded,
        "first": rows[0]["target"],
        "last": rows[-1]["target"],
        "first_audit_origin": rows[splits["T"]["evaluation"][0]]["as_of"],
    }


def sources_file():
    return event.OLD / "snapshot/nav-facts.json"


def baselines(root: Path):
    """固定规则只从对应训练折选择多数类；逐段保存，不能看评估答案再决定方向。"""
    rows, splits = io.lines(root / "inputs.jsonl"), io.read(root / "splits.json")
    nav = io.read(sources_file())
    report = {}
    for fold, split in splits.items():
        if not split["evaluation"]:
            continue
        tr, er = [rows[i] for i in split["training"]], [rows[i] for i in split["evaluation"]]
        labels = kernel.impact.labels(tr, nav)
        majority = max(kernel.CLASSES, key=lambda c: (labels.count(c), c))
        actual = kernel.impact.labels(er, nav)
        report[fold] = {
            "training_majority": majority,
            "majority": kernel.impact.stats.score(actual, [majority] * len(er)),
            "always_up": kernel.impact.stats.score(actual, ["UP"] * len(er)),
        }
    io.save(root / "simple-baselines.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "build", "search", "audit"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "build":
        result = build()
    elif args.command == "search":
        result = kernel.search(ROOT)
    else:
        result = kernel.audit(ROOT)
        baselines(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
