"""每日入口直接衔接严格资料冻结、准备、实际审阅核验和发布凭据。"""

from pathlib import Path

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save


def prepare_daily_review(directory, request_packet):
    """每基金每材料版本共用一份四周期审阅包；不自动编写经济结论。"""
    from app.commands.fund_prediction_v2_sources import freeze_sources
    from app.services.prediction_trial_v2 import evaluate_trial, prepare_trial, storage_check

    request = read(Path(request_packet["path"]))
    identity = {
        key: request[key]
        for key in ("owner_hash", "fund_code", "window", "policy_hash", "materials_input_hash", "materials_hash")
    }
    identity["nav_input_hash"] = request.get("nav_input_hash")
    base = Path(directory) / "analysis" / request["fund_code"] / digest(identity)
    sources, trial = base / "sources", base / "trial"
    instant = now()
    guard = storage_check(instant.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
    if (sources / "freeze-summary.json").exists():
        frozen = read(sources / "freeze-summary.json")
        # 此目录由资料、净值和规则身份决定；同一输入的失败保留原原因。
        # 真正补齐后身份改变，自然使用新的sources目录，不能重写旧失败目录。
    else:
        frozen = freeze_sources(sources, analysis_requests=[Path(request_packet["path"])])
    if not frozen["frozen_count"]:
        return {**request_packet, "preparation_status": "WAITING_SOURCE_REPAIR", "source_summary": frozen}
    if not (trial / "manifest.json").exists():
        prepare_trial(trial, Path(frozen["selection_path"]))
    reviews = trial / "reviews.json"
    evaluated = None
    if reviews.exists():
        # 仍逐项重验真实审阅，不因某次evaluate成功而跳过后来被改动的文件。
        evaluated = evaluate_trial(trial, reviews)
        versioned_save(Path(request_packet["review_path"]), {"trial_directory": str(trial.resolve())})
    versioned_save(base / "storage-check.json", guard)
    return {
        **request_packet,
        "preparation_status": "EVALUATED" if evaluated else "READ_AND_JUDGE",
        "trial_directory": str(trial.resolve()),
        "prepared_input": str(trial / "inputs" / (request["fund_code"] + ".json")),
        "reviews_file": str(reviews),
        "evaluation": evaluated,
        "next_action": (
            "实际阅读prepared_input及其原文，独立撰写四周期reviews_file；"
            "继续原每日入口自动核验、发布、读回及生成建议。"
        ),
    }
