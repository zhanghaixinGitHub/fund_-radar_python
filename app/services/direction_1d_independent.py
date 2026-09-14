"""固定候选的未来对照协议：共同输入、原子留档、截止前恢复和首次答案保护。

本模块没有训练入口，不调用业务预测接口，也不写业务数据库。生产调用只使用系统
时钟；clock 参数用于测试断电、跨截止及晚到场景，CLI 不提供回拨时钟参数。
"""

import json
import math
import os
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.services import direction_1d_activity_study as activity
from app.services.direction_1d_protocol import ZONE, canonical, digest, features, label, score

PROTOCOL = "DIRECTION_1D_INDEPENDENT_V1"
POLICY = {
    "candidate": "ACTIVITY12_FINAL",
    "reference": "REGISTERED_FIXED7_BY_GROUP",
    "new_training_budget": 0,
    "threshold": 0.5,
    "threshold_comparison": "STRICTLY_GREATER",
    "target_days": 120,
    "grace_days": 10,
    "checkpoints": [20, 60, 120],
    "segments": [30, 30, 30, 30],
    "minimum_gain": 0.02,
    "bootstrap_block_days": 5,
    "bootstrap_repetitions": 10000,
    "bootstrap_seed": 20260913,
    "bootstrap_interval": "PERCENTILE_LINEAR_0.025_0.975_MOVING_NONCIRCULAR",
    "minimum_positive_segments": 3,
    "minimum_improved_fund_fraction": 2 / 3,
    "maximum_fund_loss": 0.05,
    "minimum_coverage": 0.95,
    "minimum_direction_target_days": 20,
    "minimum_fund_common_answers": 20,
    "minimum_fund_common_fraction": 0.95,
    "weighting": "ORIGINAL_FAMILY_DATE",
    "baselines": ["ALWAYS_UP", "ALWAYS_NON_UP", "INITIAL_MAJORITY", "MOMENTUM"],
    "prediction_slots": ["18:30", "22:30", "07:30", "08:10", "08:20"],
    "outcome_slots": ["18:30", "22:30", "07:30"],
    "deadline": "08:30",
    "window_open": "18:00",
    "index_amount_unit": "THOUSAND_CNY",
    "brier_bins": [i / 10 for i in range(11)],
    "log_loss_clip": 1e-6,
    "model_released": False,
    "protected_years": [2025],
    "retrospective_training_2026": False,
    "slot_lateness_seconds": 300,
    "max_outcome_queries_per_tick": 60,
}


def now() -> datetime:
    return datetime.now(ZONE)


def instant(value) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED")
    return parsed.astimezone(ZONE)


def read(path: Path):
    if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("EVIDENCE_PATH_OR_SIZE_INVALID")
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value) -> None:
    """先写同目录临时文件并 fsync，再用不覆盖的硬链接发布完整文件。

    进程中断只可能留下临时文件或完整目标文件；不会把半个 JSON 当成成功结果。
    同题竞争时 FileExistsError 交由调用方读取胜出的首次记录，绝不覆盖。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical(value).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal(path: Path, value) -> None:
    write_once(path, {"payload": value, "hash": digest(value)})


def unseal(path: Path):
    envelope = read(path)
    if set(envelope) != {"payload", "hash"} or digest(envelope["payload"]) != envelope["hash"]:
        raise ValueError("EVIDENCE_HASH_MISMATCH")
    return envelope["payload"]


def model_score(model: dict, x: list[float]) -> float:
    """12项树的严格数值推理；不改7项接口，不反序列化 pickle，不允许缺项补零。"""
    activity.validate_model(model)
    if len(x) != 12 or not all(math.isfinite(v) for v in x):
        raise ValueError("CANDIDATE_INPUT_INVALID")
    values = [(v - m) / s for v, m, s in zip(x, model["mean"], model["scale"], strict=True)]
    raw = model["baseline"]
    for nodes in model["trees"]:
        pos = 0
        while not nodes[pos]["leaf"]:
            node = nodes[pos]
            pos = node["left"] if values[node["feature"]] <= node["threshold"] else node["right"]
        raw += nodes[pos]["value"]
    # 与已冻结研究恢复器保持相同的裁剪和 numpy 指数实现。
    return float(1 / (1 + np.exp(-np.clip(raw, -700, 700))))


def schedule(sessions: list[str], first: str) -> dict:
    """清单来自已核实日历；缺少120日及10日宽限时只返回缺口，不补工作日。"""
    if sessions != sorted(set(sessions)) or first not in sessions:
        raise ValueError("CALENDAR_OR_FIRST_TARGET_INVALID")
    pos = sessions.index(first)
    if pos < 61:
        raise ValueError("CALENDAR_HISTORY_MISSING")
    following = sessions[pos : pos + POLICY["target_days"] + POLICY["grace_days"]]
    return {
        "first_target": first,
        "targets": following[:120],
        "grace_targets": following[120:130],
        "complete": len(following) == 130,
        "missing_target_days": max(0, 120 - len(following)),
        "missing_grace_days": 10 - len(following[120:130]),
    }


def bounds(sessions: list[str], target: str) -> tuple[str, datetime, datetime]:
    if target not in sessions or sessions.index(target) < 61:
        raise ValueError("CALENDAR_TARGET_MISSING")
    base = sessions[sessions.index(target) - 1]
    return (
        base,
        datetime.combine(date.fromisoformat(base), time(18), ZONE),
        datetime.combine(date.fromisoformat(target), time(8, 30), ZONE),
    )


def input_vector(
    snapshot: dict, sessions: list[str], member: dict, at: datetime
) -> tuple[list[float], list[float] | None]:
    """原始61日单位净值、6日指数收盘和21日千元成交额，均严格截至T。

    snapshot 的 observed_at 是本机实际读取时间；updated_at 仅代表数据库版本，
    不能冒充供应商首次公布时间。两个模型的7项净值特征来自同一份原始数据。
    """
    at = instant(at)
    base, opened, deadline = bounds(sessions, snapshot["target"])
    if snapshot["fund_code"] != member["fund_code"] or snapshot["base"] != base:
        raise ValueError("SNAPSHOT_IDENTITY_MISMATCH")
    if snapshot["family"] != member["family"] or snapshot["group"] != member["group"]:
        raise ValueError("SNAPSHOT_SCOPE_MISMATCH")
    if not opened <= instant(snapshot["observed_at"]) <= at < deadline:
        raise ValueError("INPUT_OUTSIDE_PREDICTION_WINDOW")
    source = snapshot["source"]
    if source["enabled"] is not True or at >= instant(source["expires_at"]):
        raise ValueError("SOURCE_EXPIRED_OR_DISABLED")
    if source["source_id"] != member["source_id"]:
        raise ValueError("SOURCE_IDENTITY_CHANGED")
    i = sessions.index(base)
    nav = snapshot["nav"]
    if [r["date"] for r in nav] != sessions[i - 60 : i + 1]:
        raise ValueError("NAV_HISTORY_INCOMPLETE")
    for row in nav:
        if not row.get("content_hash") or instant(row["updated_at"]) > instant(snapshot["observed_at"]):
            raise ValueError("NAV_VERSION_OR_TIME_INVALID")
        if row.get("ann_date") and row["ann_date"] > instant(snapshot["observed_at"]).date().isoformat():
            raise ValueError("NAV_NOT_ANNOUNCED")
    if snapshot["events_status"] not in ("KNOWN_EVENT", "UNKNOWN") or not isinstance(snapshot["events"], list):
        raise ValueError("EVENT_EVIDENCE_MISSING")
    x = features([r["unit_nav"] for r in nav])
    if snapshot["market"] is None:
        if not snapshot.get("candidate_missing_reason"):
            raise ValueError("MISSING_REASON_REQUIRED")
        return x, None
    market = snapshot["market"]
    if market["amount_unit"] != POLICY["index_amount_unit"] or market["mapping_hash"] != member["mapping_hash"]:
        raise ValueError("AMOUNT_UNIT_OR_MAPPING_CHANGED")
    if not opened <= instant(market["received_at"]) <= instant(market["persisted_at"]) <= at:
        raise ValueError("MARKET_RECEIPT_TIME_INVALID")
    if at >= instant(market["expires_at"]):
        raise ValueError("MARKET_SOURCE_EXPIRED")
    series = market["series"]
    returns = {}
    for index in {"000300.SH", member["index_code"]}:
        rows = series.get(index, [])
        wanted = sessions[i - (20 if index == "000300.SH" else 5) : i + 1]
        if [r["date"] for r in rows] != wanted:
            raise ValueError("INDEX_HISTORY_INCOMPLETE")
        values = [float(r["close"]) for r in rows]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("INDEX_VALUE_INVALID")
        returns[index] = [values[-1] / values[-2] - 1, values[-1] / values[-6] - 1]
    amounts = np.asarray([float(r["amount"]) for r in series["000300.SH"]])
    if not np.isfinite(amounts).all() or min(amounts) <= 0:
        raise ValueError("AMOUNT_INVALID")
    shared, own = returns["000300.SH"], returns[member["index_code"]]
    return x, [
        *x,
        *shared,
        *[a - b for a, b in zip(own, shared, strict=True)],
        float(amounts[-1] / amounts[:-1].mean() - 1),
    ]


def answers(snapshot: dict, sessions: list[str], member: dict, models: dict, at: datetime) -> dict:
    x, full = input_vector(snapshot, sessions, member, at)
    fixed = models["fixed"][member["fixed_model_id"]]
    if fixed["group_id"] != member["group"] or instant(fixed["train_as_of"]) > instant(at):
        raise ValueError("FIXED_MODEL_SCOPE_OR_TIME_INVALID")
    values = {"FIXED7": score(fixed, x), "ACTIVITY12": model_score(models["candidate"], full) if full else None}
    return {
        "scores": values,
        "directions": {
            **{k: int(v > 0.5) if v is not None else None for k, v in values.items()},
            "ALWAYS_UP": 1,
            "ALWAYS_NON_UP": 0,
            "INITIAL_MAJORITY": fixed["majority"],
            "MOMENTUM": int(Decimal(snapshot["nav"][-1]["unit_nav"]) > Decimal(snapshot["nav"][-2]["unit_nav"])),
        },
        "candidate_missing_reason": snapshot.get("candidate_missing_reason"),
    }


def archive(root: Path, contract: dict, snapshot: dict, models: dict, *, clock=now) -> dict:
    """首份输入和答案不可替换；恢复、回执落盘及回执再次读回都必须早于截止。

    缺回执的完整答案可以在截止前恢复；截止后不补造成功。生产调用方须先完成
    审计及启动验收。本函数是供测试验收的协议内核，当前CLI尚不开放正式预测。
    """
    code, target = snapshot["fund_code"], snapshot["target"]
    if code not in contract["members"] or target not in contract["schedule"]["targets"]:
        raise ValueError("OUTSIDE_FROZEN_SCOPE")
    folder = root / "questions" / target / code
    if (folder / "readback.json").exists():
        return verify_question(folder, contract, models)
    _, _, deadline = bounds(contract["sessions"], target)
    at = instant(clock())
    if at >= deadline:
        return {"status": "MISSED_DEADLINE", "fund_code": code, "target": target}
    for expires in (
        contract["candidate_expires_at"],
        contract["fixed_expires_at"][contract["members"][code]["fixed_model_id"]],
    ):
        if at >= instant(expires):
            raise ValueError("MODEL_SOURCE_EXPIRED")
    member = contract["members"][code]
    # 在发布首次快照前验证，避免无效数据永久占住幂等键。
    if not (folder / "input.json").exists():
        input_vector(snapshot, contract["sessions"], member, at)
        try:
            seal(folder / "input.json", snapshot)
        except FileExistsError:
            pass
    saved = unseal(folder / "input.json")
    result = answers(saved, contract["sessions"], member, models, instant(clock()))
    if not (folder / "prediction.json").exists():
        prediction = {
            "protocol": PROTOCOL,
            "contract_hash": digest(contract),
            "input_hash": digest(saved),
            "model_hash": digest(models),
            "generated_at": instant(clock()).isoformat(),
            "fund_code": code,
            "target": target,
            "base": saved["base"],
            "family": member["family"],
            "kind": "FORWARD_ORIGINAL",
            "model_released": False,
            "up_probability": None,
            **result,
        }
        try:
            seal(folder / "prediction.json", prediction)
        except FileExistsError:
            pass
    prediction = unseal(folder / "prediction.json")
    if any(prediction[k] != result[k] for k in result) or prediction["input_hash"] != digest(saved):
        raise ValueError("PREDICTION_RESTORE_MISMATCH")
    restored_at = instant(clock())
    receipt = {"prediction_hash": digest(prediction), "restored_at": restored_at.isoformat()}
    try:
        seal(folder / "receipt.json", receipt)
    except FileExistsError:
        pass
    receipt = unseal(folder / "receipt.json")
    confirmed = instant(clock())
    confirmation = {
        "receipt_hash": digest(receipt),
        "confirmed_at": confirmed.isoformat(),
        "status": "VERIFIED"
        if max(instant(prediction["generated_at"]), instant(receipt["restored_at"]), confirmed) < deadline
        else "LATE_ARCHIVE",
    }
    try:
        seal(folder / "confirmation.json", confirmation)
    except FileExistsError:
        pass
    final = unseal(folder / "confirmation.json")
    # 最后一份确认文件也要完成截止前读回；其时间单独持久化，不能只凭写入时间。
    read_back_at = instant(clock())
    try:
        seal(folder / "readback.json", {"confirmation_hash": digest(final), "at": read_back_at.isoformat()})
    except FileExistsError:
        pass
    return verify_question(folder, contract, models)


def verify_question(folder: Path, contract: dict, models: dict) -> dict:
    saved = unseal(folder / "input.json")
    prediction = unseal(folder / "prediction.json")
    receipt = unseal(folder / "receipt.json")
    confirmation = unseal(folder / "confirmation.json")
    readback = unseal(folder / "readback.json")
    code, target = saved["fund_code"], saved["target"]
    if code not in contract["members"] or target not in contract["schedule"]["targets"]:
        raise ValueError("OUTSIDE_FROZEN_SCOPE")
    _, opened, deadline = bounds(contract["sessions"], target)
    times = [
        instant(prediction["generated_at"]),
        instant(receipt["restored_at"]),
        instant(confirmation["confirmed_at"]),
        instant(readback["at"]),
    ]
    if times != sorted(times) or times[0] < opened:
        raise ValueError("ARCHIVE_TIME_ORDER_INVALID")
    if prediction["contract_hash"] != digest(contract) or prediction["model_hash"] != digest(models):
        raise ValueError("PREDICTION_CONTRACT_CHANGED")
    if (
        prediction["protocol"] != PROTOCOL
        or prediction["kind"] != "FORWARD_ORIGINAL"
        or prediction["model_released"] is not False
        or prediction["up_probability"] is not None
        or (prediction["fund_code"], prediction["target"], prediction["base"], prediction["family"])
        != (code, target, saved["base"], saved["family"])
    ):
        raise ValueError("PREDICTION_IDENTITY_OR_KIND_CHANGED")
    if prediction["input_hash"] != digest(saved) or receipt["prediction_hash"] != digest(prediction):
        raise ValueError("PREDICTION_CHAIN_CHANGED")
    if confirmation["receipt_hash"] != digest(receipt) or readback["confirmation_hash"] != digest(confirmation):
        raise ValueError("RECEIPT_CHAIN_CHANGED")
    if max(times) >= deadline or confirmation["status"] != "VERIFIED":
        return {"status": "LATE_ARCHIVE", "fund_code": code, "target": target}
    expected = answers(saved, contract["sessions"], contract["members"][code], models, times[0])
    if any(prediction[k] != expected[k] for k in expected):
        raise ValueError("PREDICTION_RESTORE_MISMATCH")
    return {"status": "VERIFIED", "prediction": prediction}


def record_outcome(folder: Path, contract: dict, models: dict, observation: dict, *, clock=now) -> dict:
    """真实U净值到达后才能核对。首次T采用原输入，修订另存且不改变首次成绩。"""
    verified = verify_question(folder, contract, models)
    if verified["status"] != "VERIFIED":
        raise ValueError("FORECAST_NOT_VERIFIED")
    snapshot = unseal(folder / "input.json")
    at, target = instant(clock()), snapshot["target"]
    mature = datetime.combine(date.fromisoformat(target), time(18), ZONE)
    if at < mature:
        return {"status": "PENDING_TARGET"}
    if contract["schedule"].get("grace_targets") and at > datetime.combine(
        date.fromisoformat(contract["schedule"]["grace_targets"][-1]), time(23, 59, 59), ZONE
    ):
        raise ValueError("STAGE_ALREADY_CLOSED")
    if at >= instant(snapshot["source"]["expires_at"]):
        raise ValueError("INPUT_EVIDENCE_EXPIRED")
    if observation["date"] != target or observation["fund_code"] != snapshot["fund_code"]:
        raise ValueError("OUTCOME_IDENTITY_MISMATCH")
    if not mature <= instant(observation["observed_at"]) <= at or instant(observation["updated_at"]) > instant(
        observation["observed_at"]
    ):
        raise ValueError("OUTCOME_OBSERVATION_TIME_INVALID")
    if observation.get("ann_date") and observation["ann_date"] > at.date().isoformat():
        return {"status": "PENDING_ANNOUNCEMENT"}
    if observation["source_id"] != snapshot["source"]["source_id"] or at >= instant(observation["expires_at"]):
        raise ValueError("OUTCOME_SOURCE_INVALID")
    answer = label(snapshot["nav"][-1]["unit_nav"], observation["unit_nav"])
    base_revision = observation.get("base_revision")
    if base_revision is not None:
        if (
            base_revision["date"] != snapshot["base"]
            or base_revision["source_id"] != observation["source_id"]
            or instant(base_revision["observed_at"]) > at
            or instant(base_revision["updated_at"]) > instant(base_revision["observed_at"])
            or at >= instant(base_revision["expires_at"])
        ):
            raise ValueError("BASE_REVISION_INVALID")
        revised_answer = label(base_revision["unit_nav"], observation["unit_nav"])
    else:
        revised_answer = None
    version = digest(
        {
            **{k: observation[k] for k in ("date", "unit_nav", "content_hash", "source_id")},
            "base_revision_hash": base_revision.get("content_hash") if base_revision else None,
            "event_hash": digest(observation.get("events", [])),
        }
    )
    evidence = {
        "prediction_hash": digest(verified["prediction"]),
        "observation": observation,
        "answer": answer,
        "version": version,
        "base_revision_answer_diagnostic_only": revised_answer,
    }
    version_path = folder / "outcomes" / (version + ".json")
    if not version_path.exists():
        seal(version_path, evidence)
    else:
        unseal(version_path)
    try:
        seal(folder / "first-outcome.json", {"version": version, "evidence_hash": digest(unseal(version_path))})
    except FileExistsError:
        pass
    first = unseal(folder / "first-outcome.json")
    original = unseal(folder / "outcomes" / (first["version"] + ".json"))
    if digest(original) != first["evidence_hash"]:
        raise ValueError("OUTCOME_CHAIN_CHANGED")
    return {
        "status": "VERIFIED" if first["version"] == version else "REVISION_RECORDED",
        "first_answer": original["answer"],
        "current_answer": answer,
    }
