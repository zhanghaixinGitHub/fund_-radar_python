"""把四个已训练候选独立接到R116同题输入，不依赖未训练的R117。

独立运行计划和预测目录明确区分旧串行接入草案。训练模型、特征转换、
实际来源采集以及原R116的全部校验保持原实现；本模块不训练、不补历史。
"""

from collections import defaultdict
from datetime import datetime
from time import perf_counter

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_context_reuse_runtime as shared
from app.services import direction_1d_sprint_core_forward as core
from app.services import direction_1d_sprint_futures_member_live as member_live
from app.services import direction_1d_sprint_futures_member_runtime as member_runtime
from app.services import direction_1d_sprint_market_futures_member as member_model
from app.services import direction_1d_sprint_market_stock_exchange_return as return_model
from app.services import direction_1d_sprint_market_stock_exchange_spread as spread_model
from app.services import direction_1d_sprint_market_stock_moneyflow_v2 as moneyflow_model
from app.services import direction_1d_sprint_stock_exchange_return_live as return_live
from app.services import direction_1d_sprint_stock_exchange_return_runtime as return_runtime
from app.services import direction_1d_sprint_stock_exchange_spread_live as spread_live
from app.services import direction_1d_sprint_stock_exchange_spread_runtime as spread_runtime
from app.services import direction_1d_sprint_stock_moneyflow_binding_v2 as moneyflow_runtime
from app.services import direction_1d_sprint_stock_moneyflow_live_v2 as moneyflow_live
from app.services import direction_1d_sprint_stock_tail_forward as parent
from app.services.direction_1d_sprint_forward_context_reuse import Executor

ROUNDS = (118, 119, 120, 121)
SPECS = {
    118: (spread_model, spread_live, spread_runtime),
    119: (return_model, return_live, return_runtime),
    120: (member_model, member_live, member_runtime),
    121: (moneyflow_model, moneyflow_live, moneyflow_runtime),
}
BRANCHES = 39
FIRST_TARGET = "2026-09-17"


def root():
    return b.ROOT / "independent-candidates-r118-r121-v2"


def require(ok, reason):
    if not ok:
        raise ValueError("INDEPENDENT_CANDIDATES_" + reason)


def bindings():
    """读取已经冻结的独立计划；不会创建旧候选原生runtime-plan或绕过其模型校验。"""
    plan = b.read(root() / "plan.json")
    require(
        plan["rounds"] == list(ROUNDS)
        and plan["parent_round"] == 116
        and plan["same_question_branches"] == BRANCHES
        and plan["new_fits"] == 0
        and plan["new_source_request_budget"] == 2
        and plan["status"] == "MODEL_NOT_RELEASED",
        "PLAN_SCOPE_CHANGED",
    )
    for name, expected in plan["code_hashes"].items():
        require(core.sha(b.PROJECT / name) == expected, "CODE_CHANGED")
    for name, expected in plan["artifact_hashes"].items():
        require(b.digest(b.read(b.ROOT / name)) == expected, "ARTIFACT_CHANGED")
    require(core.sha(root() / "design-before-implementation.md") == plan["design_sha256"], "DESIGN_CHANGED")
    native, terminal = shared.bindings(116)
    require(b.digest(native) == plan["parent_shared_plan_hash"] and terminal is parent, "PARENT_PLAN_CHANGED")
    loaded = {}
    for number, (model, _, runtime) in SPECS.items():
        # verify_intent只验证该候选自己的冻结实现；models另外逐项校验训练指纹、文件和模型头。
        intent = runtime.verify_intent()
        manifest, bundle = model.models()
        expected = plan["models"][str(number)]
        require(
            b.digest(intent) == expected["original_runtime_intent_hash"]
            and b.digest(manifest) == expected["result_hash"]
            and manifest["plan_hash"] == expected["plan_hash"]
            and manifest["model_sha256"] == expected["model_sha256"],
            "MODEL_CHANGED",
        )
        loaded[number] = (manifest, bundle)
    return plan, loaded


def candidate_answers(source, question, histories, loaded):
    """只组合四个原answers的候选分支，不选最高分、不改缺失回退或概率数值。"""
    result = {}
    for number, (model, _, _) in SPECS.items():
        market = model.data.extend(source["market"], question["t"], question["u"], histories[number]["points"])
        original = model.answers(market, question["group"], loaded[number][1])
        require(len(model.CANDIDATES) == 1, "CANDIDATE_COUNT_CHANGED")
        for name in model.CANDIDATES:
            require(name not in result and original[name]["prediction"] in (0, 1), "CANDIDATE_INVALID")
            result[name] = original[name]
    require(len(result) == 4, "CANDIDATE_COUNT_CHANGED")
    return result


def combine(old, new):
    require(len(old) == 35 and len(new) == 4 and set(old).isdisjoint(new), "BRANCH_COLLISION_OR_MISSING")
    value = old | new
    require(len(value) == BRANCHES, "BRANCH_COUNT_CHANGED")
    return value


def early_capture(loaded):
    """有时限的唯一新排名请求优先执行，沿用原R95预约槽与原R120失败不重试语义。"""
    t, u = "2026-09-16", FIRST_TARGET
    if not member_live.in_window(t, u, b.now()):
        return
    quote_plan = b.digest(b.read(b.ROOT / "round-95/plan.json"))
    member_live.quote_live.capture(t, u, quote_plan)
    if (member_live.quote_live.live_root() / u / "snapshot.json").is_file():
        member_live.capture(t, u, loaded[120][0]["plan_hash"])
    # 同一个R109报价请求槽先得到实收，再只预约一次R121订单数据；父链以后复用同一快照。
    stock_plan = b.digest(b.read(b.ROOT / "round-109/plan.json"))
    moneyflow_live.quote_live.capture(t, u, stock_plan)
    moneyflow_live.capture(t, u, loaded[121][0]["plan_hash"])


def actual_histories(question, loaded):
    """每题只接收真实目标日快照；未采集的来源保持未就绪，绝不用训练历史填补。"""
    t, u = question["t"], question["u"]
    if not all(live.ready(u) for _, live, _ in SPECS.values()):
        return None
    result = {}
    for number, (_, live, _) in SPECS.items():
        value = live.load_live(t, u, loaded[number][0]["plan_hash"])
        require(live.in_window(t, u, datetime.fromisoformat(value["at"])), "SOURCE_OUTSIDE_WINDOW")
        result[number] = value
    return result


def model_identities(loaded):
    return {
        str(number): {k: manifest[k] for k in ("plan_hash", "model_sha256")} for number, (manifest, _) in loaded.items()
    }


def validate(value, receipt, question, source, tail, old_answers, histories, loaded, plan):
    """保存及以后核对均按同一契约重验时点、输入、父回读、来源、模型和39个完整答案。"""
    created, readback = datetime.fromisoformat(value["at"]), datetime.fromisoformat(receipt["readback_at"])
    deadline = min(core.data.deadline(question["u"]), core.evidence.sprint_end())
    require(value["u"] == FIRST_TARGET and question["u"] == FIRST_TARGET, "TARGET_CHANGED")
    require(all(value[k] == question[k] for k in ("code", "family", "group", "t", "u")), "QUESTION_CHANGED")
    require(
        datetime.fromisoformat(plan["at"]) <= created <= readback < deadline
        and datetime.fromisoformat(question["at"]) <= created,
        "FORECAST_LATE",
    )
    for number, (manifest, _) in loaded.items():
        require(datetime.fromisoformat(manifest["at"]) <= created, "MODEL_CREATED_LATE")
        require(datetime.fromisoformat(histories[number]["at"]) <= created, "SOURCE_CREATED_LATE")
    core_receipt = b.read(core.root() / "receipts" / question["u"] / f"{question['code']}.json")
    tail_receipt = b.read(parent.model.root() / "receipts" / question["u"] / f"{question['code']}.json")
    for record, saved_receipt in ((question, core_receipt), (tail, tail_receipt)):
        require(
            saved_receipt["status"] == "VERIFIED"
            and saved_receipt["forecast_hash"] == b.digest(record)
            and datetime.fromisoformat(saved_receipt["readback_at"]) <= created,
            "PARENT_RECEIPT_CHANGED_OR_LATE",
        )
    require(
        value["runtime_plan_hash"] == b.digest(plan)
        and value["parent_hash"] == b.digest(question)
        and value["r116_forecast_hash"] == b.digest(tail),
        "PLAN_OR_PARENT_CHANGED",
    )
    require(value["input_hash"] == question["input_hash"] == b.digest(source), "INPUT_CHANGED")
    require(value["models"] == model_identities(loaded), "FORECAST_MODEL_CHANGED")
    require(value["source_hashes"] == {str(n): b.digest(h) for n, h in histories.items()}, "SOURCE_CHANGED")
    require(
        value["answers"] == combine(old_answers, candidate_answers(source, question, histories, loaded)),
        "ANSWERS_CHANGED",
    )
    require(receipt["forecast_hash"] == b.digest(value) and receipt["status"] == "VERIFIED", "RECEIPT_CHANGED")
    require(value["status"] == "MODEL_NOT_RELEASED" and value["contract"] == core.CONTRACT, "CONTRACT_CHANGED")


def ready_parents(context):
    """R116完整run后的上下文仍逐题调用其原validate，不能仅凭文件存在接入候选。"""
    manifest, bundle, ctx, candidates, extras, histories = context
    sources, result = {}, {}
    for key, question in candidates.items():
        if question["u"] != FIRST_TARGET:
            continue
        path = parent.model.root() / "forward" / question["u"] / f"{question['code']}.json"
        if not path.is_file():
            continue
        if question["u"] not in sources:
            sources[question["u"]] = core.load_input(question["u"], ctx)
        tail, receipt = b.read(path), b.read(parent.model.root() / "receipts" / question["u"] / path.name)
        source = sources[question["u"]]
        parent.validate(tail, receipt, question, source, manifest, bundle, extras[key], histories[question["u"]])
        old = parent.combined_answers(question, extras[key], tail)
        require(len(old) == 35, "PARENT_BRANCHES_MISSING")
        result[key] = (question, source, tail, old)
    return result


def write_and_report(plan, loaded, parents):
    """同一题一次性保存39分支，回读后才可计数；后续只核对成熟的真实结果。"""
    histories_by_day = {}
    for question, source, tail, old in parents.values():
        u = question["u"]
        if u not in histories_by_day:
            histories_by_day[u] = actual_histories(question, loaded)
        histories = histories_by_day[u]
        if histories is None:
            continue
        path = root() / "forward" / u / f"{question['code']}.json"
        if path.exists() or b.now() >= min(core.data.deadline(u), core.evidence.sprint_end()):
            continue
        value = {k: question[k] for k in ("code", "family", "group", "t", "u")} | {
            "at": b.now().isoformat(),
            "runtime_plan_hash": b.digest(plan),
            "parent_hash": b.digest(question),
            "r116_forecast_hash": b.digest(tail),
            "input_hash": question["input_hash"],
            "models": model_identities(loaded),
            "source_hashes": {str(n): b.digest(h) for n, h in histories.items()},
            "answers": combine(old, candidate_answers(source, question, histories, loaded)),
            "status": "MODEL_NOT_RELEASED",
            "contract": core.CONTRACT,
        }
        if b.now() >= min(core.data.deadline(u), core.evidence.sprint_end()):
            continue
        b.save(path, value)
        checked, at = b.read(path), b.now()
        receipt = {
            "readback_at": at.isoformat(),
            "forecast_hash": b.digest(checked),
            "status": "VERIFIED" if checked == value and at < core.data.deadline(u) else "LATE_OR_INVALID",
        }
        b.save(root() / "receipts" / u / path.name, receipt)
        # 再读实际来源收据/原文，防止把内存快照当作已落盘证据。
        fresh = actual_histories(question, loaded)
        require(fresh == histories, "SOURCE_CHANGED_DURING_WRITE")
        validate(checked, receipt, question, source, tail, old, fresh, loaded, plan)
    paired, cache, valid, revised = defaultdict(list), {}, [], set()
    fallback_counts = {str(n): 0 for n in ROUNDS}
    at = b.now()
    for path in sorted((root() / "forward").glob("*/*.json")):
        value = b.read(path)
        require(path.stem == value["code"] and path.parent.name == value["u"], "FORECAST_PATH_CHANGED")
        key = value["code"], value["u"]
        require(key in parents, "VALID_PARENT_MISSING")
        question, source, tail, old = parents[key]
        histories = actual_histories(question, loaded)
        require(histories is not None, "SAVED_ACTUAL_SOURCE_MISSING")
        receipt = b.read(root() / "receipts" / value["u"] / path.name)
        validate(value, receipt, question, source, tail, old, histories, loaded, plan)
        valid.append(value)
        for n in ROUNDS:
            fallback_counts[str(n)] += int(not histories[n]["available"])
        outcome = core.root() / "outcomes" / value["u"] / path.name
        if not outcome.exists():
            continue
        first = b.read(outcome)
        pair = core.evidence.validated_outcome(question, first, cache, at)
        for revision in (core.root() / "revisions" / value["u"] / value["code"]).glob("*.json"):
            changed = b.read(revision)
            core.evidence.validated_outcome(question, changed, cache, at)
            require(changed["first_outcome_hash"] == b.digest(first), "OUTCOME_REVISION_CHANGED")
            revised.add(key)
        for name, answer in value["answers"].items():
            paired[name].append(question | pair | {"prediction": answer["prediction"]})
    closed = core.data.deadline(FIRST_TARGET) <= min(at, core.evidence.sprint_end())
    matched = len(next(iter(paired.values()), []))
    report = {
        "at": at.isoformat(),
        "verified_forecasts": len(valid),
        "matched": matched,
        "pending": len(valid) - matched,
        "same_question_branches": BRANCHES,
        "excluded_untrained_rounds": [117],
        "matched_forward_metrics": {name: b.metrics(rows) for name, rows in paired.items()},
        "missing_due_predictions": 30 - len(valid) if closed else 0,
        "eligible_coverage": len(valid) / 30 if closed else None,
        "whole_watchlist_coverage": len(valid) / 43 if closed else None,
        "revised_questions": len(revised),
        "status": "MODEL_NOT_RELEASED",
        "new_feature_fallback_forecasts": fallback_counts,
    }
    b.save(root() / "report.json", report, replace=True)
    return report


def preflight(loaded):
    """仅离线推理，不读标签计算成绩；完整输出逐份与原函数一致才能验收。"""
    ctx = core.context()
    questions = core.day16(ctx)
    require(len(questions) == 30, "PREFLIGHT_QUESTION_COUNT")
    source = b.read(b.ROOT / "core-market-independent-forward-v1/input.json")
    histories = {n: {"points": model.data.history()["snapshot"]["rows"]} for n, (model, _, _) in SPECS.items()}
    checks = []
    for question in questions:
        combined = candidate_answers(source, question, histories, loaded)
        for n, (model, _, _) in SPECS.items():
            market = model.data.extend(source["market"], question["t"], question["u"], histories[n]["points"])
            native = model.answers(market, question["group"], loaded[n][1])
            name = model.CANDIDATES[0]
            require(combined[name] == native[name], "NATIVE_INFERENCE_MISMATCH")
            checks.append({"round": n, "code": question["code"], "answer_hash": b.digest(native[name])})
    return {"kind": "DRY_RUN_ONLY_NOT_ACTUAL_FORWARD", "checks": len(checks), "answers": checks, "new_forecasts": 0}


def execute(action):
    require(action in ("preflight", "run"), "UNKNOWN_ACTION")
    started = perf_counter()
    plan, loaded = bindings()
    executor = Executor(parent, audit_only=action == "preflight")
    require([m.__name__ for m in executor.modules] == plan["parent_module_chain"], "PARENT_GRAPH_CHANGED")
    if action == "preflight":
        executor.audit_contexts()
        result = preflight(loaded)
        require(result["checks"] == 120, "PREFLIGHT_COUNT_CHANGED")
    else:
        early_capture(loaded)
        executor.run()
        require(all(s["phase"] == "COMPLETED" for s in executor.states.values()), "PARENT_NOT_COMPLETED")
        context = executor.states[parent.__name__]["context"]
        result = write_and_report(plan, loaded, ready_parents(context))
    after, _ = bindings()
    require(after == plan, "PLAN_CHANGED_DURING_RUN")
    receipt = {
        "at": b.now().isoformat(),
        "action": action,
        "plan_hash": b.digest(plan),
        "elapsed_seconds": perf_counter() - started,
        "status": "SUCCEEDED",
        "same_question_branches": BRANCHES,
        "excluded_untrained_rounds": [117],
        "new_fits": 0,
        "result_hash": b.digest(result),
        "context_calls": {m.model.root().name: executor.states[m.__name__]["context_calls"] for m in executor.modules},
        "result": result
        if action == "preflight"
        else {k: result[k] for k in ("verified_forecasts", "matched", "pending")},
        "future_deadline_performance_proven": False,
    }
    b.save(root() / ("preflight.json" if action == "preflight" else "last-run.json"), receipt, replace=True)
    return receipt
