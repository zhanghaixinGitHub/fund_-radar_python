"""R121仅绑定自己已训练的资金流模型；不把失败R117放进任何运行父链。"""

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_core_forward as core
from app.services import direction_1d_sprint_market_stock_moneyflow_v2 as model


def verify_intent():
    """必须已完成27拟合及固定分析；核对本轮新入口实现、模型和来源，不能使用占位模型。"""
    intent = base.read(model.root() / "runtime-intent.json")
    trained = base.read(model.root() / "result.json")
    completion = base.read(model.root() / "training-completion.json")
    model.require(
        intent["model_plan_hash"] == base.digest(base.read(model.root() / "plan.json")) == trained["plan_hash"]
        and intent["model_result_hash"] == base.digest(trained)
        and intent["training_completion_hash"] == base.digest(completion)
        and trained["development_fits"] == 24
        and trained["current_fits"] == 3
        and completion["status"] == "TRAINED_ANALYZED_INDEPENDENT_RUNTIME_PENDING"
        and intent["same_question_branches"] == 39
        and intent["first_target"] == model.FIRST_TARGET
        and intent["new_request_budget"] == 1,
        "RUNTIME_INTENT_CHANGED",
    )
    for name, expected in intent["code_hashes"].items():
        model.require(core.sha(base.PROJECT / name) == expected, "RUNTIME_CODE_CHANGED")
    model.require(core.sha(model.root() / "runtime-design.md") == intent["design_sha256"], "RUNTIME_DESIGN_CHANGED")
    return intent
