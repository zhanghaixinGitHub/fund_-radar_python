"""002112固定事件实验编排：独立预算、断点恢复、分阶段标签、真实重拟合与研究报告。"""

# 线程环境必须先于NumPy导入。只新增本实验产物，不调用旧snapshot/采集/登记入口。
# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from collections import Counter
from decimal import Decimal
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import sklearn
from threadpoolctl import threadpool_info

from scripts import fund_002112_existing_events_features_v1 as f
from scripts import fund_002112_existing_events_fit_v1 as fit
from scripts import fund_002112_existing_events_sources_v1 as s

FOLDS = {
    "V2023": ("2022-12-31", "2023-01-01", "2023-12-31"),
    "C2024": ("2023-12-31", "2024-01-01", "2024-12-31"),
    "C2025": ("2024-12-31", "2025-01-01", "2025-12-31"),
    "C2026": ("2025-12-31", "2026-01-01", "2026-09-29"),
    "FINAL": ("2026-09-29", None, None),
}
BUCKETS = {"main": 12, "aux": 8, "final": 3, "replay": 2, "retry": 3}
CODE = [
    s.PY / "scripts" / name
    for name in (
        "fund_002112_existing_events_sources_v1.py",
        "fund_002112_existing_events_features_v1.py",
        "fund_002112_existing_events_fit_v1.py",
        "fund_002112_existing_events_v1.py",
        "fund_002112_existing_events_revision_v1.py",
        "test_fund_002112_existing_events_v1.py",
    )
]


def budget(root):
    original = s.experiment_root(root)
    attempts = [r for r in s.lines(original / "fit-ledger.jsonl") if r["action"] == "RESERVED"]
    limit, buckets = 28, BUCKETS
    approval = original / "continuation-authorization.json"
    if approval.exists():
        authorized = s.read(approval)
        if authorized.get("execution_allowed") is True:
            limit = authorized["max_real_fits"]
            buckets = authorized["bucket_limits"]
    return {
        "limit": limit,
        "consumed": len(attempts),
        "remaining": limit - len(attempts),
        "buckets": dict(Counter(r["bucket"] for r in attempts)),
        "limits": buckets,
        "ledger_root": str(original),
    }


def reserve(root, run_id, bucket, parameters):
    """调用者持有操作系统排他锁；fsync完成后才允许真实监督fit开始。"""
    current = budget(root)
    if current["consumed"] >= current["limit"] or current["buckets"].get(bucket, 0) >= current["limits"][bucket]:
        raise ValueError("SUPERVISED_FIT_BUDGET_EXHAUSTED")
    attempt = current["consumed"] + 1
    s.append(
        s.experiment_root(root) / "fit-ledger.jsonl",
        {
            "action": "RESERVED",
            "attempt": attempt,
            "run_id": run_id,
            "bucket": bucket,
            "at": s.now(),
            "pid": os.getpid(),
            "parameters": parameters,
            "real_supervised_fit": True,
            "revision": str(root),
        },
    )
    s.replace(s.experiment_root(root) / "budget.json", budget(root))
    return attempt


def progress(root, stage, state, detail=""):
    entry = {"at": s.now(), "stage": stage, "state": state, "detail": detail, "budget": budget(root)}
    s.append(root / "stages.jsonl", entry)
    s.replace(root / "status.json", entry)
    if s.experiment_root(root) == s.ROOT.resolve():
        s.replace(s.COORD / "handoff.json", {**entry, "experiment_root": str(root), "external_requests": 0})
        (s.COORD / "status.md").write_text(
            f"{stage} {state}：{detail}\n\n"
            f"实际拟合 {entry['budget']['consumed']}/{entry['budget']['limit']}；外部请求 0。\n",
            "utf-8",
        )
    print(s.canonical(entry), flush=True)


def load(root):
    manifest = s.read(root / "input-manifest.json")
    for path, expected in manifest["artifacts"].items():
        if s.sha(root / path) != expected:
            raise ValueError("FROZEN_INPUT_CHANGED:" + path)
    rows = s.lines(root / "daily-inputs.jsonl")
    members = s.lines(root / "event-membership.jsonl")
    events = s.lines(root / "events.jsonl")
    if [r["target"] for r in rows] != [r["target"] for r in members]:
        raise ValueError("MEMBERSHIP_DATE_MISMATCH")
    return rows, members, events


def frozen_check(root):
    protocol = s.read(root / "protocol.json")
    if protocol["experiment"] != s.EXPERIMENT or protocol["doc_sha256"] != s.DOC_SHA:
        raise ValueError("PROTOCOL_IDENTITY_MISMATCH")
    for path, hashed in s.read(root / "code-manifest.json")["files"].items():
        if s.sha(path) != hashed:
            raise ValueError("FROZEN_CODE_CHANGED:" + path)
    if s.sha(root / "source-inventory.json") != protocol["source_inventory_sha256"]:
        raise ValueError("SOURCE_INVENTORY_CHANGED")
    for name in ("input-manifest", "split-manifest", "code-manifest"):
        if s.sha(root / (name + ".json")) != protocol[name.replace("-", "_") + "_sha256"]:
            raise ValueError("FROZEN_MANIFEST_CHANGED:" + name)


def freeze(root):
    if (root / "protocol.json").exists():
        frozen_check(root)
        return
    rows, _, _ = load(root)
    evidence = s.read(root / "input-manifest.json")["evidence"]
    for timing in f.TIMINGS:
        for kind in f.TYPES:
            item = evidence[timing][kind]
            if not (item["varying_statistic"] and item["text_characters"] > 0 and item["dates_with_information"] > 0):
                raise ValueError("REQUIRED_EVENT_CLASS_NOT_PRESENT:" + kind)
    checks = [
        ("pytest", [sys.executable, "-m", "pytest", str(CODE[-1]), "-q"]),
        ("ruff", [sys.executable, "-m", "ruff", "check", *map(str, CODE)]),
    ]
    results = {}
    for name, command in checks:
        env = {
            **os.environ,
            "FUND_EVENTS_SYNTHETIC_AUDIT": str(root / "synthetic-fit-ledger.jsonl"),
            "FUND_EVENTS_TEST_ROOT": str(root),
        }
        run = subprocess.run(command, cwd=s.PY, capture_output=True, encoding="utf-8", errors="replace", env=env)
        results[name] = {"returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    s.replace(root / "tests-before-freeze.json", results)
    if any(v["returncode"] != 0 for v in results.values()):
        raise ValueError("PRE_FREEZE_TESTS_FAILED")
    inventory = s.read(root / "source-inventory.json")
    sessions = s.read(root / "sessions.json")
    splits = {}
    for stage, (train_end, start, end) in FOLDS.items():
        cutoff = next(d for d in sessions if d >= start) + "T15:00:00+08:00" if start else inventory["frozen_at"]
        train_ids = [i for i, r in enumerate(rows) if r["target"] <= train_end and r["label_mature_at"] < cutoff]
        eval_ids = [
            i
            for i, r in enumerate(rows)
            if start and start <= r["target"] <= end and r["label_mature_at"] <= inventory["frozen_at"]
        ]
        splits[stage] = {
            "train_end": train_end,
            "fit_cutoff_exclusive": cutoff,
            "train_indices": train_ids,
            "evaluation_indices": eval_ids,
            "train_dates": [rows[i]["target"] for i in train_ids],
            "evaluation_dates": [rows[i]["target"] for i in eval_ids],
            "immature_training_dates": [
                r["target"] for r in rows if r["target"] <= train_end and r["label_mature_at"] >= cutoff
            ],
        }
    s.save(root / "split-manifest.json", splits)
    s.save(
        root / "code-manifest.json",
        {
            "files": {str(p): s.sha(p) for p in CODE},
            "python": sys.version,
            "sklearn": sklearn.__version__,
            "threads": threadpool_info(),
        },
    )
    s.save(
        root / "protocol.json",
        {
            "experiment": s.EXPERIMENT,
            "at": s.now(),
            "doc_sha256": s.DOC_SHA,
            "source_inventory_sha256": s.sha(root / "source-inventory.json"),
            "input_manifest_sha256": s.sha(root / "input-manifest.json"),
            "split_manifest_sha256": s.sha(root / "split-manifest.json"),
            "code_manifest_sha256": s.sha(root / "code-manifest.json"),
            "fit_limit": 28,
            "buckets": BUCKETS,
            "recipe": fit.RECIPE,
            "timings": {"main": "PUBLICATION_DATE_1500", "aux": "DATE_ONLY_NEXT_SESSION"},
            "frozen_source_scope_regex": f.UNRELATED_RE.pattern,
            "research_status": "HISTORICAL_RESEARCH_NOT_NEW_BLIND_TEST",
            "production_adoption_allowed": False,
            "external_requests": 0,
        },
    )


def labels(root, stage, purpose, rows):
    """答案仅在冻结计划允许的阶段访问；评估前必须已有全部指定模型的未评分预测。"""
    frozen_check(root)
    if purpose not in ("train", "evaluate") or stage not in FOLDS:
        raise ValueError("LABEL_STAGE_OR_PURPOSE_INVALID")
    split = s.read(root / "split-manifest.json")[stage]
    allowed = split["train_dates"] if purpose == "train" else split["evaluation_dates"]
    if [r["target"] for r in rows] != allowed:
        raise ValueError("LABEL_STAGE_DATE_BOUNDARY")
    if purpose == "evaluate":
        for timing, models in [("main", ("B0", "B1", "B2")), ("aux", ("B1", "B2"))]:
            for model in models:
                validate_completed(root, f"{stage}_{timing}_{model}")
    if stage == "FINAL" and not (root / "comparison.json").exists():
        raise ValueError("FINAL_REQUIRES_FIXED_COMPARISON")
    s.append(
        root / "label-access-ledger.jsonl",
        {
            "at": s.now(),
            "stage": stage,
            "purpose": purpose,
            "date_count": len(allowed),
            "dates_sha256": s.digest(allowed),
            "protocol_sha256": s.sha(root / "protocol.json"),
        },
    )
    nav = s.read(root / "snapshot/nav-facts.json")
    result = []
    for row in rows:
        a, b = Decimal(nav[row["base"]]["unit_nav"]), Decimal(nav[row["target"]]["unit_nav"])
        if not (a.is_finite() and b.is_finite() and a > 0 and b > 0):
            raise ValueError("INVALID_LABEL_NAV")
        result.append("UP" if b > a else "DOWN" if b < a else "FLAT")
    return result


def run_path(root, run_id):
    return (
        root / ("models/final" if run_id.startswith("FINAL_") and not run_id.endswith("_REPLAY") else "models") / run_id
    )


def validate_completed(root, run_id):
    directory = run_path(root, run_id)
    marker = s.read(directory / "complete.json")
    for name, expected in marker["files"].items():
        if s.sha(directory / name) != expected:
            raise ValueError("COMPLETED_MODEL_CORRUPTED:" + run_id)
    if marker.get("reused_from_previous_revision"):
        proof = s.read(directory / "reuse-equivalence.json")
        if proof["new_protocol_sha256"] != s.sha(root / "protocol.json") or not proof["equivalent"]:
            raise ValueError("REUSE_EQUIVALENCE_NOT_BOUND_TO_REVISION")
    return marker


def execute_fit(root, stage, timing, name, replay=False):
    frozen_check(root)
    require_training_authorization(root)
    run_id = f"{stage}_{timing}_{name}" + ("_REPLAY" if replay else "")
    directory = run_path(root, run_id)
    if (directory / "complete.json").exists():
        return validate_completed(root, run_id)
    rows, members, events = load(root)
    split = s.read(root / "split-manifest.json")[stage]
    ti = split["train_indices"]
    ei = split["evaluation_indices"] if stage != "FINAL" else ti
    tr, tm = [rows[i] for i in ti], [members[i] for i in ti]
    er, em = [rows[i] for i in ei], [members[i] for i in ei]
    y = labels(root, stage, "train", tr)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"{s.now()} {run_id} preparing fold-local matrices", flush=True)
    prep = fit.fit_preprocessor(tr, tm, events, name, timing)
    matrix = fit.transform(prep, tr, tm, events)
    evaluation = fit.transform(prep, er, em, events)
    old_attempts = [
        a
        for a in s.lines(s.experiment_root(root) / "fit-ledger.jsonl")
        if a["action"] == "RESERVED"
        and a["run_id"] == run_id
        and a.get("revision", str(s.experiment_root(root))) == str(root)
    ]
    if len(old_attempts) >= 2:
        raise ValueError("SINGLE_RETRY_ALREADY_USED:" + run_id)
    bucket = "replay" if replay else "final" if stage == "FINAL" else timing
    for retry in range(len(old_attempts), 2):
        consumed = []

        def before_fit(params, retry_index=retry, attempt_list=consumed):
            attempt = reserve(root, run_id, "retry" if retry_index else bucket, params)
            attempt_list.append(attempt)
            return attempt

        try:
            model, info = fit.train(prep, matrix, y, before_fit, max_iter=2000 if retry else 1000)
            s.append(
                s.experiment_root(root) / "fit-ledger.jsonl",
                {"action": "FIT_RETURNED", "run_id": run_id, "revision": str(root), "at": s.now(), **info},
            )
            if not info["converged"]:
                if retry:
                    raise ValueError("NON_CONVERGENCE_AFTER_FIXED_RETRY")
                continue
            predicted = fit.predict(model, evaluation, er)
            model_file = directory / "model.joblib"
            joblib.dump(model, model_file)
            reloaded = joblib.load(model_file)
            predictions_loaded = fit.predict(reloaded, evaluation, er)
            delta = probability_error(predicted, predictions_loaded)
            if delta > 1e-10 or [p["predicted"] for p in predicted] != [p["predicted"] for p in predictions_loaded]:
                raise ValueError("SERIALIZATION_REPLAY_FAILED")
            s.save_lines(directory / "predictions.jsonl", predicted)
            majority = max(sorted(set(y)), key=lambda c: y.count(c))
            info.update(
                {
                    "run_id": run_id,
                    "stage": stage,
                    "timing": timing,
                    "model": name,
                    "pid": os.getpid(),
                    "supervised_refit": True,
                    "replay": replay,
                    "train_dates": len(tr),
                    "prediction_dates": len(er),
                    "train_majority": majority,
                    "serialization_max_probability_error": delta,
                    "evidence": fit.text_evidence(prep, tr, tm, events),
                    "prediction_purpose": "IN_SAMPLE_REPLAY_ONLY" if stage == "FINAL" else "UNSCORED_HISTORY",
                    "protocol_sha256": s.sha(root / "protocol.json"),
                }
            )
            s.save(directory / "training.json", info)
            complete = {
                "run_id": run_id,
                "at": s.now(),
                "attempt": info["attempt"],
                "files": {p: s.sha(directory / p) for p in ("model.joblib", "predictions.jsonl", "training.json")},
            }
            s.save(directory / "complete.json", complete)
            print(f"{s.now()} {run_id} complete; fit {info['attempt']}/{budget(root)['limit']}", flush=True)
            return complete
        except Exception as exc:
            s.append(
                s.experiment_root(root) / "fit-ledger.jsonl",
                {
                    "action": "FAILED",
                    "run_id": run_id,
                    "at": s.now(),
                    "attempt": consumed[-1] if consumed else None,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
    raise ValueError("UNEXPECTED_RETRY_EXHAUSTED")


def require_training_authorization(root):
    """数据修订一旦触及总预算，未获明确补充授权前连第一次额外fit也不启动。"""
    original = s.experiment_root(root)
    if root == original:
        return
    path = original / "continuation-authorization.json"
    if not path.exists():
        raise ValueError("REVISION_TRAINING_HELD_PENDING_EXPLICIT_BUDGET_AUTHORIZATION")
    approval = s.read(path)
    if not (
        approval.get("execution_allowed") is True
        and approval.get("authorization_evidence")
        and approval.get("revision_protocol_sha256") == s.sha(root / "protocol.json")
    ):
        raise ValueError("REVISION_AUTHORIZATION_NOT_BOUND_TO_FROZEN_PROTOCOL")
    if approval["max_real_fits"] < 31:
        raise ValueError("REVISION_MINIMUM_TOTAL_31_NOT_AUTHORIZED")


def probability_error(a, b):
    if [r["target"] for r in a] != [r["target"] for r in b]:
        raise ValueError("REPLAY_DATE_ORDER_MISMATCH")
    return max(
        (
            abs(x["probabilities"][c] - y["probabilities"][c])
            for x, y in zip(a, b, strict=True)
            for c in fit.CLASSES
            if x["probabilities"][c] is not None and y["probabilities"][c] is not None
        ),
        default=0.0,
    )


def train_all(root):
    for stage in list(FOLDS)[:-1]:
        for timing, models in [("main", ("B0", "B1", "B2")), ("aux", ("B1", "B2"))]:
            for name in models:
                execute_fit(root, stage, timing, name)


def evaluate(root):
    if (root / "comparison.json").exists():
        return s.read(root / "comparison.json")
    rows, _, _ = load(root)
    splits = s.read(root / "split-manifest.json")
    comparisons, timing_result, daily = {}, {}, []
    for stage in list(FOLDS)[:-1]:
        ids = splits[stage]["evaluation_indices"]
        er = [rows[i] for i in ids]
        y = labels(root, stage, "evaluate", er)
        for timing in f.TIMINGS:
            predictions = {}
            for name in ("B0", "B1", "B2"):
                run_id = f"{stage}_{'main' if name == 'B0' else timing}_{name}"
                validate_completed(root, run_id)
                values = s.lines(run_path(root, run_id) / "predictions.jsonl")
                if [r["target"] for r in values] != [r["target"] for r in er]:
                    raise ValueError("PAIR_DATES_NOT_IDENTICAL")
                predictions[name] = [v["predicted"] for v in values]
            majority = s.read(run_path(root, f"{stage}_main_B0") / "training.json")["train_majority"]
            comparisons[f"{stage}_{timing}"] = fit.compare(er, y, predictions, timing, majority)
            for i, r in enumerate(er):
                daily.append(
                    {
                        "stage": stage,
                        "timing": timing,
                        "target": r["target"],
                        "actual": y[i],
                        **{name: predictions[name][i] for name in predictions},
                        "B2_minus_B0_correct": int(predictions["B2"][i] == y[i]) - int(predictions["B0"][i] == y[i]),
                    }
                )
        timing_result[stage] = {
            t: {name: comparisons[f"{stage}_{t}"]["all"]["models"][name] for name in ("B0", "B1", "B2")}
            for t in f.TIMINGS
        }
    s.save_lines(root / "paired-daily.jsonl", daily)
    s.save(root / "timing-sensitivity.json", timing_result)
    s.save(root / "comparison.json", comparisons)
    return comparisons


def fit_final(root):
    if not (root / "comparison.json").exists():
        raise ValueError("E05_NOT_COMPLETE")
    for name in ("B0", "B1", "B2"):
        execute_fit(root, "FINAL", "main", name)


def verify_replays(root):
    # 父进程此时不能占写锁；子进程持锁重新拟合，每次单独PID、分别记预算。
    results = {}
    for stage in ("C2026", "FINAL"):
        command = [
            sys.executable,
            "-m",
            "scripts.fund_002112_existing_events_v1",
            "replay-one",
            "--root",
            str(root),
            "--stage",
            stage,
        ]
        result = subprocess.run(command, cwd=s.PY, capture_output=True, encoding="utf-8", errors="replace")
        (root / f"replay-{stage}.log").write_text(result.stdout + result.stderr, "utf-8")
        if result.returncode:
            raise ValueError("INDEPENDENT_REFIT_FAILED:" + stage)
        original = run_path(root, f"{stage}_main_B2")
        replay = run_path(root, f"{stage}_main_B2_REPLAY")
        a, b = s.lines(original / "predictions.jsonl"), s.lines(replay / "predictions.jsonl")
        error = probability_error(a, b)
        first, second = s.read(original / "training.json"), s.read(replay / "training.json")
        same = [p["predicted"] for p in a] == [p["predicted"] for p in b]
        results[stage] = {
            "max_probability_error": error,
            "same_classes": same,
            "original_pid": first["pid"],
            "replay_pid": second["pid"],
            "separate_process": first["pid"] != second["pid"],
            "supervised_refit": second["supervised_refit"],
            "attempt": second["attempt"],
        }
    s.save(root / "replay.json", results)
    if any(
        r["max_probability_error"] > 1e-10 or not r["same_classes"] or not r["separate_process"]
        for r in results.values()
    ):
        raise ValueError("INDEPENDENT_REFIT_COMPARISON_FAILED")
    return results


def acceptance(root):
    frozen_check(root)
    load(root)
    if (root / "acceptance.json").exists():
        return s.read(root / "acceptance.json")
    expected = [
        f"{stage}_{timing}_{name}"
        for stage in list(FOLDS)[:-1]
        for timing, names in [("main", ("B0", "B1", "B2")), ("aux", ("B1", "B2"))]
        for name in names
    ]
    expected += [f"FINAL_main_{name}" for name in ("B0", "B1", "B2")]
    expected += [f"{stage}_main_B2_REPLAY" for stage in ("C2026", "FINAL")]
    for run_id in expected:
        validate_completed(root, run_id)
    b2 = s.read(run_path(root, "FINAL_main_B2") / "training.json")
    if not all(
        e["documents"] and e["training_dates"] and e["vocabulary_size"] and e["statistic_varies"]
        for e in b2["evidence"].values()
    ):
        raise ValueError("THREE_TYPES_NOT_ACTUALLY_TRAINED")
    inventory = s.read(root / "source-inventory.json")
    changed, snapshot_errors = [], []
    for source in inventory["sources"]:
        if s.sha(s.experiment_root(root) / source["snapshot_path"]) != source["sha256"]:
            snapshot_errors.append(source["source_path"])
        current = s.sha(source["source_path"])
        if current != source["sha256"]:
            changed.append(
                {"path": source["source_path"], "frozen": source["sha256"], "current": current, "at": s.now()}
            )
    protected = s.read(root / "protection-before.json")["protected"]
    protected_errors = [p for p, hashed in protected.items() if s.sha(p) != hashed]
    s.save(
        root / "protection-after.json",
        {
            "at": s.now(),
            "old_artifact_changes": protected_errors,
            "source_changes": changed,
            "snapshot_errors": snapshot_errors,
            "note": "来源并行更新只记录；本实验继续使用冻结副本，不回滚其他任务",
        },
    )
    replay = s.read(root / "replay.json")
    passed = not snapshot_errors and not protected_errors and budget(root)["consumed"] <= budget(root)["limit"]
    passed = passed and all(v["max_probability_error"] <= 1e-10 and v["same_classes"] for v in replay.values())
    result = {
        "at": s.now(),
        "engineering_complete": passed,
        "models_complete": len(expected),
        "three_types_actual_training": b2["evidence"],
        "budget": budget(root),
        "external_requests": 0,
        "business_writes": 0,
        "production_adopted": False,
        "replay": replay,
        "old_artifacts_preserved": not protected_errors,
        "frozen_sources_intact": not snapshot_errors,
        "source_changes_after_freeze": len(changed),
        "tests": s.read(root / "tests-before-freeze.json"),
        "limits": [
            "发布日期重建，非真实盘中回放",
            "日期级主同日假设",
            "资料库覆盖不等于全市场完整",
            "各年历史已查看，不是全新盲测",
            "最终拟合仅研究，不登记或替换现用模型",
        ],
    }
    s.save(root / "acceptance.json", result)
    if not passed:
        raise ValueError("FINAL_ACCEPTANCE_FAILED")
    return result


def report(root):
    accepted = s.read(root / "acceptance.json")
    summary = s.read(root / "event-summary.json")
    comparisons = s.read(root / "comparison.json")
    lines = [
        "# 002112 现有事件训练结果",
        "",
        "本轮完成新闻、政策、公告三类统计和真实文字入模，固定比较、最终研究模型及两次独立进程重新拟合。",
        "这是按发布日期重建的历史再研究，不是全新盲测，也不是线上采用。",
        "",
        "## 实际使用的资料",
        "",
        "| 类别 | 去重文档 | 发布日期 | 只有标题 | FINAL实际训练文档 | 训练日期 | 文字词表 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for kind, label in [("NEWS", "新闻"), ("POLICY", "政策"), ("ANNOUNCEMENT", "公告")]:
        v, e = summary["types"][kind], accepted["three_types_actual_training"][kind]
        lines.append(
            f"| {label} | {v['documents']} | {v['start']}—{v['end']} | {v['title_only']} | "
            f"{e['documents']} | {e['training_dates']} | {e['vocabulary_size']} |"
        )
    date_only = sum(v["date_only"] for v in summary["types"].values())
    lines += [
        "",
        f"共 {date_only} 条仅精确到日期。主口径当天使用并保留假设；辅助口径移到下一交易日。"
        "明确15:00及以后发布的消息不能进入当天。已知盘后结果标题也延后，净值输入只取可得历史记录。",
        "资料库计数为0只表示未收集到，不证明当时没有事件；来源处理错误不能当0。",
        "",
        "## 相同日期的比较",
        "",
        "| 阶段 | 日期数 | 不加事件B0 | 统计B1 | 文字B2 | B2比B0多对 | 辅助B1 | 辅助B2 | 辅助B2比B0多对 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    totals = Counter()
    for stage in list(FOLDS)[:-1]:
        a = comparisons[f"{stage}_main"]["all"]["models"]
        b = comparisons[f"{stage}_aux"]["all"]["models"]
        lines.append(
            f"| {stage} | {a['B0']['dates']} | {a['B0']['correct']} | {a['B1']['correct']} | "
            f"{a['B2']['correct']} | {a['B2']['correct'] - a['B0']['correct']:+d} | {b['B1']['correct']} | "
            f"{b['B2']['correct']} | {b['B2']['correct'] - a['B0']['correct']:+d} |"
        )
        for name in ("B0", "B1", "B2"):
            totals[name] += a[name]["correct"]
        totals["aux_B2"] += b["B2"]["correct"]
        totals["days"] += a["B0"]["dates"]
    lines += [
        "",
        f"合计 {totals['days']} 个比较日：B0判对 {totals['B0']} 天，B1判对 {totals['B1']} 天，"
        f"B2判对 {totals['B2']} 天（比B0 {totals['B2'] - totals['B0']:+d} 天）；"
        f"日期级消息延后后B2判对 {totals['aux_B2']} 天（比B0 {totals['aux_B2'] - totals['B0']:+d} 天）。",
        "",
        "## 上涨、下跌与季度稳定性",
        "",
    ]
    for stage in list(FOLDS)[:-1]:
        result = comparisons[f"{stage}_main"]
        models = result["all"]["models"]
        pieces = []
        for kind, label in [("UP", "上涨"), ("DOWN", "下跌"), ("FLAT", "持平")]:
            a, b = models["B0"]["classes"][kind], models["B2"]["classes"][kind]
            pieces.append(f"{label}{a['actual']}天，B0/B2分别判对{a['correct']}/{b['correct']}天")
        quarters = [f"{k} {v['pairs']['B2-B0']['net_correct_days']:+d}天" for k, v in result.items() if "-Q" in k]
        interval = result["all"]["pairs"]["B2-B0"]["accuracy_difference_95pct_block_interval"]
        lines += [
            f"- {stage}：{'；'.join(pieces)}。季度净变化：{'、'.join(quarters)}。"
            f"5日块重采样差值区间 [{interval['low']:.2%}, {interval['high']:.2%}]。"
        ]
    lines += [
        "",
        "## 资料分布与限制",
        "",
        "| 年份 | 新闻条数/发布日 | 政策条数/发布日 | 公告条数/发布日 |",
        "|---|---:|---:|---:|",
    ]
    for year in map(str, range(2016, 2027)):
        vals = [summary["types"][k]["yearly"][year] for k in f.TYPES]
        lines.append("| " + year + " | " + " | ".join(f"{v['documents']}/{v['publication_days']}" for v in vals) + " |")
    lines += [
        "",
        "各年资料密度不同，当前持仓未用于倒推历史；公司关系按目标日前已披露报告核对，"
        "无关系证明时作为固定库背景。未采集到的资料不能由本轮评价其价值。",
        "完整配对四格、各类别召回、季度、有/无消息窗口、三类非空窗口及训练多数类对照见 comparison.json。"
        "区间采用5个连续交易日为块、2000次固定种子重采样，并保留日历缺口。",
        "",
        "## 拟合与保存",
        "",
        f"真实监督拟合消耗 {accepted['budget']['consumed']}/{accepted['budget']['limit']}，"
        f"分项 {accepted['budget']['buckets']}。"
        "失败、重试、独立复现均计入；没有增加候选。人工样例测试独立执行，不使用真实数据子集。",
        "最终模型：models/final/FINAL_main_B0、FINAL_main_B1、FINAL_main_B2 下的 model.joblib；"
        "含标准化器、三类词表/IDF和分类器。FINAL训练内预测仅供序列化复核，不作为效果证据。",
        f"两次独立进程重新拟合结果：{s.canonical(accepted['replay'])}",
        "",
        "外部采集、网页、外部大模型请求均为0；业务库只读，未启停服务、未登记换模、未提交推送部署。",
        "本轮可证明三类已实际训练及所列历史比较结果，不能证明盘中真实可得、交易收益提高或适合替换现用系统。",
        "",
        "证据：source-inventory.json、event-summary.json、input-manifest.json、models/*/training.json、"
        "fit-ledger.jsonl、comparison.json、timing-sensitivity.json、replay.json、acceptance.json。",
    ]
    (root / "result.md").write_text("\n".join(lines) + "\n", "utf-8")
    if s.experiment_root(root) == s.ROOT.resolve():
        (s.COORD / "delivery.md").write_text("\n".join(lines) + "\n\n实验目录：" + str(root) + "\n", "utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "build-inputs",
            "freeze",
            "train",
            "evaluate",
            "fit-final",
            "verify",
            "report",
            "status",
            "resume",
            "replay-one",
        ],
    )
    parser.add_argument("--root", type=Path, default=s.ROOT)
    parser.add_argument("--stage", choices=["C2026", "FINAL"])
    args = parser.parse_args()
    root = s.active_root(args.root.resolve())
    if args.command == "status":
        print(
            s.canonical(
                {
                    "status": s.read(root / "status.json") if (root / "status.json").exists() else None,
                    "budget": budget(root),
                }
            )
        )
        return
    commands = ["prepare", "build-inputs", "freeze", "train", "evaluate", "fit-final", "verify", "report"]
    todo = commands if args.command == "resume" else [args.command]
    stages = {
        "prepare": "E00",
        "build-inputs": "E01",
        "freeze": "E03",
        "train": "E04",
        "evaluate": "E05",
        "fit-final": "E06",
        "verify": "E06",
        "report": "E07",
    }
    for command in todo:
        if command == "verify":
            progress(root, "E06", "REPLAY_RUNNING", "两个独立进程重新拟合")
            verify_replays(root)
            with s.writer_lock(root), s.offline_guard():
                acceptance(root)
            progress(root, "E06", "COMPLETE", "FINAL及独立进程重拟合已回读")
            continue
        with s.writer_lock(root):
            if command == "prepare":
                progress(root, "E00", "RUNNING", "固定指定既有来源")
                s.prepare(root)
                progress(root, "E00", "COMPLETE", "来源已固定")
                continue
            with s.offline_guard() as attempts:
                if command == "replay-one":
                    if not args.stage:
                        raise ValueError("REPLAY_STAGE_REQUIRED")
                    execute_fit(root, args.stage, "main", "B2", replay=True)
                    return
                stage = stages[command]
                progress(root, stage, "RUNNING", command)
                if command == "build-inputs":
                    f.build_inputs(root)
                    progress(root, "E02", "COMPLETE", "主辅每日输入及三类文字引用完成")
                elif command == "freeze":
                    freeze(root)
                elif command == "train":
                    train_all(root)
                elif command == "evaluate":
                    evaluate(root)
                elif command == "fit-final":
                    fit_final(root)
                elif command == "report":
                    report(root)
                s.append(
                    root / "network-audit.jsonl",
                    {"at": s.now(), "command": command, "external_network_attempts": len(attempts), "pid": os.getpid()},
                )
                progress(root, stage, "COMPLETE", command)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

# G_REVIEW_001: 已归档冻结源码；触发下一阶段摘要边界停止，等待分类修复审查。
