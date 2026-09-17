"""第116轮前向运行独立绑定：不改已冻结训练代码，待R115完成后一次性冻结接入关系。"""

import shutil
from datetime import datetime

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_core_forward as core
from app.services import direction_1d_sprint_market_stock_activity as parent
from app.services import direction_1d_sprint_market_stock_tail as model
from app.services import direction_1d_sprint_stock_activity_runtime as parent_runtime


def verify_intent():
    """只核对本层文件和已完成的训练，不要求尚未完成的父模型，也不启动任何网络读取。"""
    intent = base.read(model.root() / "runtime-intent.json")
    model.require(
        intent["model_plan_hash"] == base.digest(base.read(model.root() / "plan.json"))
        and intent["model_result_hash"] == base.digest(base.read(model.root() / "result.json"))
        and intent["parent_round"] == 115
        and intent["same_question_branches"] == 35
        and intent["new_request_budget"] == 0
        and intent["first_target"] == model.FIRST_TARGET,
        "RUNTIME_INTENT_CHANGED",
    )
    for name, expected in intent["code_hashes"].items():
        model.require(core.sha(base.PROJECT / name) == expected, "RUNTIME_CODE_CHANGED")
    model.require(core.sha(model.root() / "runtime-design.md") == intent["design_sha256"], "RUNTIME_DESIGN_CHANGED")
    return intent


def bindings():
    """运行入口和逐份预测共用绑定；已冻结的接入计划不能借重新打包结果替换模型。"""
    intent = verify_intent()
    value = base.read(model.root() / "runtime-plan.json")
    model.require(value["intent_hash"] == base.digest(intent), "RUNTIME_PLAN_CHANGED")
    for name, expected in value["code_hashes"].items():
        model.require(core.sha(base.PROJECT / name) == expected, "RUNTIME_DEPENDENCY_CHANGED")
    for name, expected in value["artifact_hashes"].items():
        model.require(base.digest(base.read(base.ROOT / name)) == expected, "RUNTIME_ARTIFACT_CHANGED")
    return value


def plan():
    """R116训练与R115运行链均完成后才绑定；父运行层未完成时拒绝占位接入。"""
    intent = verify_intent()
    trained, _ = model.models()
    prior, _ = parent.models()
    parent_run = parent_runtime.plan()
    code = dict(trained["fingerprint"]["code"])
    for mapping in (parent_run["code_hashes"], prior["fingerprint"]["code"], intent["code_hashes"]):
        for name, digest in mapping.items():
            model.require(name not in code or code[name] == digest, "RUNTIME_CODE_CONFLICT")
            code[name] = digest
    context = {
        "intent_hash": base.digest(intent),
        "code_hashes": code,
        "artifact_hashes": {
            "round-116/plan.json": trained["plan_hash"],
            "round-116/result.json": base.digest(trained),
            "round-115/plan.json": prior["plan_hash"],
            "round-115/result.json": base.digest(prior),
            "round-115/runtime-plan.json": base.digest(parent_run),
        },
    }
    path = model.root() / "runtime-plan.json"
    if path.exists():
        value = bindings()
        model.require(all(value[k] == v for k, v in context.items()), "RUNTIME_CONTEXT_CHANGED")
        return value
    model.active()
    value = context | {"at": base.now().isoformat(), "status": "MODEL_NOT_RELEASED"}
    base.save(path, value)
    for name in code:
        destination = model.root() / "runtime-code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, destination)
    return bindings()


def checked_models():
    plan()
    return model.models()


def validate_forecast_binding(value):
    frozen = bindings()
    model.require(
        value["runtime_plan_hash"] == base.digest(frozen)
        and datetime.fromisoformat(frozen["at"]) <= datetime.fromisoformat(value["at"]),
        "RUNTIME_FORECAST_LATE_OR_CHANGED",
    )
