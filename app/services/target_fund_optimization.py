"""002112 第二轮受控离线运行；固定目录、预算、恢复和原系统保护，无自动训练入口。"""

from datetime import datetime

from app.services import fund_002112_training as original
from app.services import target_fund_diagnostics as diagnostics
from app.services import target_fund_experiments as experiment
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import PROJECT, ROOT, _publish, now, read, save

STORE = ROOT / "optimization-runs/round-20260925-v1"
BASE = original.checked_directory(diagnostics.BASE_RUN)


def code_manifest():
    paths = [*PROJECT.glob("app/services/target_fund_*.py"), PROJECT / "scripts/target_fund_optimization.py"]
    return {p.relative_to(PROJECT).as_posix(): original.file_hash(p) for p in sorted(paths)}


def baseline_files():
    """前轮全部成果文件不可改写；新结果只在独立目录登记。"""
    return {p.relative_to(BASE).as_posix(): original.file_hash(p) for p in sorted(BASE.rglob("*")) if p.is_file()}


def invariant_snapshot():
    return {"baseline_files": baseline_files(), "system": original.protection()}


def event(state, name, **details):
    state["events"].append({"at": now().isoformat(), "event": name, **details})
    state["fits"] = len(list((STORE / "attempts").glob("*.json")))
    save(STORE / "state.json", state, replace=True)


def put(name, value):
    original.put_once(STORE / name, value)


def publish_text(name, content):
    path = STORE / name
    raw = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("OPT_TEXT_ARTIFACT_CHANGED")
    else:
        _publish(path, raw)


def prepare(data):
    """首次拟合前固定代码、环境、方案、所有分段日期及权重；原 ready 指针不参与后续恢复。"""
    folds = experiment.build_folds(data)
    fitting = [f for f in folds if f["eligible"]]
    maximum = len(fitting) * len(experiment.CONFIGS) * 2 + len(experiment.CANDIDATES) * 2
    if maximum != experiment.PLAN["maximum_fits"]:
        raise ValueError("OPT_EXPECTED_FOLD_BUDGET_CHANGED")
    spec = {
        "plan": experiment.PLAN,
        "base_run": BASE.name,
        "data_hash": digest(data),
        "source_protocol_hash": digest(read(BASE / "protocol.json")),
        "code": code_manifest(),
        "environment": original.environment(),
        "maximum_fits": maximum,
    }
    if (STORE / "protocol.json").exists():
        if read(STORE / "protocol.json") != spec:
            raise ValueError("OPT_PROTOCOL_CHANGED_NO_NEW_BUDGET")
        return folds
    # 诊断不含随机性；已有零拟合成果必须与当前冻结资料一致。
    diagnostic = diagnostics.diagnose(data, BASE)
    put("diagnostics.json", diagnostic)
    publish_text("diagnostics.md", diagnostics.report(diagnostic))
    fold_manifest = []
    for fold in folds:
        item = {key: value for key, value in fold.items() if key not in {"train", "exam"}}
        item.update(
            {
                "training_rows": [
                    {
                        "fund_code": r["fund_code"],
                        "target": r["target"],
                        "label": r["actual_direction"],
                        "hash": digest(r),
                    }
                    for r in fold["train"]
                ],
                "exam_rows": [
                    {"target": r["target"], "label": r["actual_direction"], "hash": digest(r)} for r in fold["exam"]
                ],
                "weights": {
                    config: experiment.sample_weights(fold["train"], config).tolist() for config in experiment.CONFIGS
                }
                if fold["eligible"]
                else {},
            }
        )
        fold_manifest.append(item)
    put("folds.json", fold_manifest)
    put("protection-before.json", invariant_snapshot())
    put(
        "state.json",
        {
            "status": "FROZEN",
            "created_at": now().isoformat(),
            "events": [],
            "fits": 0,
            "failures": {},
            "completed_stages": [],
        },
    )
    put("protocol.json", spec)
    return folds


def load():
    data = original.load_frozen(BASE)
    spec = read(STORE / "protocol.json")
    if (
        spec["plan"] != experiment.PLAN
        or spec["data_hash"] != digest(data)
        or spec["code"] != code_manifest()
        or spec["environment"] != original.environment()
        or spec["source_protocol_hash"] != digest(read(BASE / "protocol.json"))
    ):
        raise ValueError("OPT_SOURCE_CODE_ENV_CHANGED")
    if read(STORE / "protection-before.json")["baseline_files"] != baseline_files():
        raise ValueError("OPT_BASELINE_ARTIFACT_CHANGED")
    folds = experiment.build_folds(data)
    saved = read(STORE / "folds.json")
    for fold, manifest in zip(folds, saved, strict=True):
        if any(fold[k] != manifest[k] for k in ("name", "start", "end", "train_hash", "exam_hash", "gate", "eligible")):
            raise ValueError("OPT_FOLD_CHANGED")
        if fold["eligible"] and any(
            experiment.sample_weights(fold["train"], config).tolist() != manifest["weights"][config]
            for config in experiment.CONFIGS
        ):
            raise ValueError("OPT_FROZEN_WEIGHT_CHANGED")
    return data, folds


def stage(state, train, exam, name, config, interrupt_after=None):
    """每个阶段两份不可变尝试；技术失败只留下原始原因，不重置该阶段的预算。"""
    stage_id = f"{name}-{config}"
    outputs = {}
    for phase in ("main", "replay"):
        identity = f"{stage_id}-{phase}"
        attempt = STORE / "attempts" / f"{identity}.json"
        checkpoint = STORE / "checkpoints" / f"{identity}.json"
        attempt_value = {
            "stage": stage_id,
            "phase": phase,
            "train_hash": digest(train),
            "exam_hash": digest(exam),
            "weight_hash": digest(experiment.sample_weights(train, config).tolist()),
            "protocol_hash": digest(read(STORE / "protocol.json")),
        }
        if attempt.exists() and read(attempt) != attempt_value:
            raise ValueError("OPT_ATTEMPT_IDENTITY_CHANGED")
        if checkpoint.exists():
            if not attempt.exists():
                raise ValueError("OPT_UNACCOUNTED_CHECKPOINT")
            outputs[phase] = read(checkpoint)
            event(state, "REUSED", stage=stage_id, phase=phase)
            continue
        if attempt.exists():
            raise ValueError("OPT_INTERRUPTED_FIT_BUDGET_CONSUMED")
        if len(list((STORE / "attempts").glob("*.json"))) >= experiment.PLAN["maximum_fits"]:
            raise ValueError("OPT_BUDGET_EXHAUSTED")
        save(attempt, attempt_value)
        event(state, "FIT_ENTERED", stage=stage_id, phase=phase)
        output = experiment.fit(train, exam, config, name)
        save(checkpoint, output)
        if interrupt_after == identity:
            import os

            os._exit(75)
        outputs[phase] = output
        event(state, "FIT_SAVED", stage=stage_id, phase=phase, model_hash=digest(output["model"]))
    main, replay = outputs["main"], outputs["replay"]
    if main["model"] != replay["model"]:
        raise ValueError("OPT_REPLAY_CHANGED")
    model = main["model"]
    if (
        model["config"] != config
        or model["stage"] != name
        or model["train_hash"] != digest(train)
        or model["weight_hash"] != digest(experiment.sample_weights(train, config).tolist())
    ):
        raise ValueError("OPT_MODEL_STAGE_CHANGED")
    predictions = experiment.predict_rows(model, exam)
    put(f"models/{stage_id}.json", model)
    put(
        f"replay/{stage_id}.json",
        {
            "parameters_equal": True,
            "main_hash": digest(main["model"]),
            "replay_hash": digest(replay["model"]),
            "main_restore": main["restore"],
            "replay_restore": replay["restore"],
            "main_iterations": main["iterations"],
            "replay_iterations": replay["iterations"],
        },
    )
    put(f"predictions/{stage_id}.json", predictions)
    if stage_id not in state["completed_stages"]:
        state["completed_stages"].append(stage_id)
    event(state, "STAGE_COMPLETE", stage=stage_id)
    return predictions


def attempt_stage(state, train, exam, name, config, interrupt_after=None):
    try:
        return stage(state, train, exam, name, config, interrupt_after)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) and str(exc).startswith("OPT_") else type(exc).__name__
        state["failures"].setdefault(f"{name}-{config}", reason)
        event(state, "STAGE_FAILED", stage=f"{name}-{config}", reason=reason)
        return None


def observation_readiness(data, final):
    """第四步只核对已有输入元数据，不读取 2025 标签或将 2026 历史输入补成预测。

    没有合格候选时结束于真实的准入结论。已有 input-only 文件的空方向永远不改写，
    自然到期和未来持平样本不能由离线代码模拟完成。
    """
    records = []
    for path in sorted((ROOT / "forward-inputs").glob("*.json")):
        # 年份先从路径筛选；禁止打开封存年份的任何前向文件。
        if not path.stem.startswith("2026-"):
            continue
        value = read(path)
        records.append(
            {
                "file": path.name,
                "hash": digest(value),
                "kind": value.get("kind"),
                "target": value["window"]["target_nav_date"],
                "generated_at": value["generated_at"],
                "has_candidate_prediction": value.get("candidate_direction") is not None,
                "training_eligible": value.get("training_eligible") is True,
                "expired": datetime.fromisoformat(value["expires_at"]) <= now(),
            }
        )
    control = read(ROOT / "runtime-control.json") if (ROOT / "runtime-control.json").exists() else {}
    historical = [r for r in data["train"] if r["actual_direction"] == "FLAT"]
    return {
        "status": "CANDIDATE_PASSED_REQUIRES_LIVE_INPUT_VALIDATION" if final["selected"] else "NO_QUALIFIED_CANDIDATE",
        "candidate": final["selected"],
        "existing_input_only": records,
        "own_train_flat_dates": len({r["target"] for r in historical if r["fund_code"] == "002112"}),
        "pooled_train_flat_dates": len({r["target"] for r in historical}),
        "development_flat_dates": sum(r["actual_direction"] == "FLAT" for r in data["development"]),
        "existing_maintenance_enabled": control.get("enabled", False),
        "maintenance_control_changed": False,
        "new_forward_predictions": 0,
        "new_mature_labels": 0,
        "future_wait_required": True,
        "adopted": False,
    }


def comparison_report(result):
    lines = [
        "# 002112 第二轮有限优化结果",
        "",
        "本轮为探索开发，不是独立盲测；所有模型保持三分类，未修改正式预测。",
        "",
        f"本轮真实拟合 {result['fits']} 次，上限 28；第一轮原 6 次单独保留。",
        "",
        "## 时间分段检查",
    ]
    for title, comparison in [
        ("2023 三个合格季度合计", result["historical"]),
        *result["folds"].items(),
        ("2024 原 230 日开发检查", result["development"]),
    ]:
        lines += [
            "",
            f"### {title}",
            "",
            "| 做法 | 正确/总数 | 上涨正确/实际 | 持平正确/实际 | 下跌正确/实际 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        ordered = [(k, comparison["models"][k]) for k in experiment.CONFIGS if k in comparison["models"]]
        ordered += [
            (f"始终{k}", comparison["constants"][k]) for k in experiment.CLASSES if k in comparison["constants"]
        ]
        for name, metric in ordered:
            cells = [
                f"{metric['class_correct'][k]}/{metric['actual'][k]}" if metric["actual"][k] else "无此类样本"
                for k in ("UP", "FLAT", "DOWN")
            ]
            lines.append(f"| {name} | {metric['correct']}/{metric['count']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 2024 季度正确天数",
        "",
        "| 季度 | 日期数 | A7 | C20 | W20 | I24 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for quarter, models in result["development"].get("quarters", {}).items():
        cells = [str(models[c]["correct"]) if c in models else "未完成" for c in experiment.CONFIGS]
        lines.append(f"| {quarter} | {next(iter(models.values()))['count']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 新做法相对原 C20 的逐日得失",
        "",
        "| 做法 | 新增判断正确 | 原来正确、现在错误 | 净变化 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for candidate in experiment.CANDIDATES:
        pair = result["development"].get("differences", {}).get(f"{candidate}_vs_C20")
        if pair:
            won, lost = pair["counts"]["candidate_only"], pair["counts"]["baseline_only"]
            lines.append(f"| {candidate} | {won} | {lost} | {won - lost:+d} |")
    lines += [
        "",
        "## 固定规则的选择结果",
        "",
        f"历史先选：{result['historical_decision']['selected'] or '无'}。",
        f"允许进入真实观察准备：{result['final_gate']['selected'] or '无'}。",
        "2024 不用于替换历史选择；完整规则逐项布尔结果见 comparison.json。",
        "",
        "## 限制与保护",
        "",
        "2023 Q1 因训练持平仅 29 个独立日期而跳过，未降低 30 日期门槛。",
        "2023 合格检查段仅 8 个目标持平日，2024 仅 2 个，不能证明持平能力稳定。",
        "原研究、第一轮模型和正式登记/预测保护结果见 protection-after.json；本轮未改关注或服务。",
        "未来样本需要真实接收、首次输出和自然到期，既有 input-only 不算预测。",
        "没有满足规则就停止本轮，不追加参数、基金、日期或组合实验。",
        "",
    ]
    return "\n".join(lines)


def run(interrupt_after=None):
    with original.run_lock():
        if not (STORE / "protocol.json").exists():
            data = original.load_frozen(BASE)
            prepare(data)
        data, folds = load()
        state = read(STORE / "state.json")
        state["status"] = "RUNNING"
        event(state, "START_OR_RESUME")
        comparisons, all_rows = {}, {k: [] for k in experiment.CONFIGS}
        for fold in folds:
            if not fold["eligible"]:
                event(state, "FOLD_SKIPPED_GATE", fold=fold["name"], gate=fold["gate"])
                continue
            predictions = {}
            for config in experiment.CONFIGS:
                rows = attempt_stage(state, fold["train"], fold["exam"], fold["name"], config, interrupt_after)
                if rows is not None:
                    predictions[config] = rows
                    all_rows[config].extend(rows)
            comparisons[fold["name"]] = experiment.compare(predictions)
        needed = sum(len(f["exam"]) for f in folds if f["eligible"])
        all_rows = {k: rows for k, rows in all_rows.items() if len(rows) == needed}
        historical = experiment.compare(all_rows)
        decision = experiment.select_historical(historical, list(comparisons.values()))
        put("historical-comparison.json", {"overall": historical, "folds": comparisons})
        put("historical-decision.json", decision)
        event(state, "HISTORICAL_SELECTION_FROZEN", selected=decision["selected"])
        # 时间筛选已永久保存后才拟合两个预定完整模型；不用 2024 反选比例或交互项。
        development_rows = {
            "A7": read(BASE / "predictions/NAV7.json"),
            "C20": read(BASE / "predictions/NAV7_HOLDINGS_MARKET.json"),
        }
        for config in experiment.CANDIDATES:
            rows = attempt_stage(state, data["train"], data["development"], "FULL", config, interrupt_after)
            if rows is not None:
                development_rows[config] = rows
        development = experiment.compare(development_rows)
        final = experiment.final_gate(decision, development)
        after = invariant_snapshot()
        unchanged = after == read(STORE / "protection-before.json")
        put("protection-after.json", after)
        if not unchanged:
            state["failures"]["protection"] = "OPT_PROTECTED_STATE_CHANGED"
        if state["failures"]:
            final = {**final, "selected": None, "technical_failures_block_observation": True}
        result = {
            "fits": len(list((STORE / "attempts").glob("*.json"))),
            "historical": historical,
            "folds": comparisons,
            "historical_decision": decision,
            "development": development,
            "final_gate": final,
            "failures": state["failures"],
            "protected_unchanged": unchanged,
            "adopted": False,
        }
        put("comparison.json", result)
        publish_text("report.md", comparison_report(result))
        if not (STORE / "observation-readiness.json").exists():
            put("observation-readiness.json", observation_readiness(data, final))
        state["status"] = "COMPLETED" if not state["failures"] else "PARTIAL"
        event(state, "FINISHED", status=state["status"])
        return {
            "status": state["status"],
            "fits": result["fits"],
            "candidate": final["selected"],
            "protected_unchanged": unchanged,
            "directory": str(STORE),
        }


def status():
    state = read(STORE / "state.json")
    effective = state["status"]
    if effective == "RUNNING":
        try:
            with original.run_lock():
                effective = "INTERRUPTED"
        except ValueError as exc:
            if str(exc) != "OFFLINE_TRAINING_BUSY":
                raise
    return {
        "status": effective,
        "fits": len(list((STORE / "attempts").glob("*.json"))),
        "failures": state["failures"],
        "completed_stages": state["completed_stages"],
        "directory": str(STORE),
    }


def verify_report():
    """从全部主模型恢复预测，重算历史选择及最终准入；没有 fit 调用。"""
    with original.run_lock():
        data, folds = load()
        comparisons, combined = {}, {c: [] for c in experiment.CONFIGS}
        for fold in folds:
            if not fold["eligible"]:
                continue
            predictions = {}
            for config in experiment.CONFIGS:
                name = f"{fold['name']}-{config}"
                model = read(STORE / f"models/{name}.json")
                rows = experiment.predict_rows(model, fold["exam"])
                if rows != read(STORE / f"predictions/{name}.json"):
                    raise ValueError("OPT_PREDICTION_RECOMPUTE_CHANGED")
                predictions[config] = rows
                combined[config].extend(rows)
            comparisons[fold["name"]] = experiment.compare(predictions)
        historical = experiment.compare(combined)
        decision = experiment.select_historical(historical, list(comparisons.values()))
        development_rows = {
            "A7": read(BASE / "predictions/NAV7.json"),
            "C20": read(BASE / "predictions/NAV7_HOLDINGS_MARKET.json"),
        }
        for config in experiment.CANDIDATES:
            rows = experiment.predict_rows(read(STORE / f"models/FULL-{config}.json"), data["development"])
            if rows != read(STORE / f"predictions/FULL-{config}.json"):
                raise ValueError("OPT_FULL_PREDICTION_RECOMPUTE_CHANGED")
            development_rows[config] = rows
        development = experiment.compare(development_rows)
        saved = read(STORE / "comparison.json")
        if any(
            saved[k] != v
            for k, v in {
                "historical": historical,
                "folds": comparisons,
                "historical_decision": decision,
                "development": development,
                "final_gate": experiment.final_gate(decision, development),
            }.items()
        ):
            raise ValueError("OPT_REPORT_RECOMPUTE_CHANGED")
        if (STORE / "report.md").read_text(encoding="utf-8") != comparison_report(saved):
            raise ValueError("OPT_TEXT_RECOMPUTE_CHANGED")
        return {"verified": True, "new_fits": 0, "recorded_fits": len(list((STORE / "attempts").glob("*.json")))}
