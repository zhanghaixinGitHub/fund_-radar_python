"""已审阅但暂不判断的变化观察；只登记重审待办，不生成方向、不关闭新一期任务。"""

import json
from datetime import datetime, time
from pathlib import Path

from sqlalchemy import text

from app.db.session import get_engine
from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from app.services.fund_prediction_snapshot_compat import materials_input_identity
from app.services.prediction_trial_v2 import PROJECT, storage_check

ROOT = PROJECT / ".local-runs" / "prediction-review-watch"


def reviewed_material_identity(archive, code, horizon=None):
    """只从正式存档的来源链找已审阅材料，不能用首次观察到的材料冒充已读基线。"""
    cases = [
        c
        for c in archive.get("cases", [])
        if c["case"]["fund_code"] == code and (horizon is None or c["case"]["horizon"] == horizon)
    ]
    strict = {c.get("workflow_identity", {}).get("materials_input_hash") for c in cases}
    strict.discard(None)
    if archive.get("publication_type") == "STRICT_V2_TRIAL":
        if len(strict) != 1:
            raise ValueError("WATCH_REVIEWED_MATERIAL_IDENTITY_MISSING")
        return strict.pop()
    # 原窗口继续研究显式绑定本次实际审阅的材料后，才能确认新版本已经读过。
    bound = []
    for case in cases:
        source_hash = case.get("source_hash") or case["case"].get("source", {}).get("hash")
        stored = archive.get("source_packets", {}).get(source_hash)
        if stored:
            packet = json.loads(stored["canonical_json"])
            if digest(packet) != source_hash:
                raise ValueError("WATCH_ARCHIVED_RESEARCH_CHANGED")
            binding = packet.get("reviewed_materials", {}).get(code)
            if binding:
                path = Path(binding["materials_path"]).resolve()
                if not path.is_relative_to(PROJECT.resolve()):
                    raise ValueError("WATCH_MATERIAL_PATH_OUTSIDE_PROJECT")
                material = read(path)
                if material.get("fund_code") != code or digest(material) != binding["materials_hash"]:
                    raise ValueError("WATCH_REVIEWED_MATERIAL_CHANGED")
                bound.append(materials_input_identity(material))
    if bound:
        if len(bound) != len(cases) or len(set(bound)) != 1:
            raise ValueError("WATCH_REVIEWED_MATERIALS_CONFLICT")
        return bound[0]
    # 首次采用及其继续研究保留了原严格输入的摘要；按原试验目录核对整条链。
    from app.services.prediction_trial_v2 import local_path

    manifest = archive["manifest"]
    summary_path = local_path(manifest["original_summary_path"])
    summary = read(summary_path)
    if digest(summary) != manifest["original_summary_hash"]:
        raise ValueError("WATCH_ORIGINAL_SUMMARY_CHANGED")
    packet = read(summary_path.parent.parent / "inputs" / (code + ".json"))
    if not cases or any(c["original_record"]["packet_hash"] != digest(packet) for c in cases):
        raise ValueError("WATCH_ORIGINAL_PACKET_CHANGED")
    source = read(local_path(packet["source_path"]))
    if digest(source) != packet["source_hash"] or digest(source["input"]) != source["input_hash"]:
        raise ValueError("WATCH_ORIGINAL_SOURCE_CHANGED")
    data = source["input"]
    material_path = Path(data["materials_path"]).resolve()
    if not material_path.is_relative_to(PROJECT.resolve()):
        raise ValueError("WATCH_MATERIAL_PATH_OUTSIDE_PROJECT")
    material = read(material_path)
    if material.get("fund_code") != code or digest(material) != data["materials_hash"]:
        raise ValueError("WATCH_ORIGINAL_MATERIAL_CHANGED")
    return materials_input_identity(material)


def watch_transition(owner_hash, code, period, baseline, handoff, material, *, at, previous=None):
    """同一期同资料只产生一个待办；新研究的材料绑定才能消除旧待办。"""
    if material.get("fund_code") != code or digest(material) != handoff.get("materials_hash"):
        raise ValueError("WATCH_CURRENT_MATERIAL_CHANGED")
    observed = materials_input_identity(material)
    review_id = digest(
        {
            k: period.get(k)
            for k in (
                "horizonId",
                "analysisAt",
                "baseNavDate",
                "startDate",
                "endDate",
                "nominalEndDate",
                "decision",
                "primaryReason",
                "counterReason",
                "conditions",
            )
        }
    )
    end = period.get("endDate")
    expired = bool(end and at >= datetime.combine(datetime.fromisoformat(end).date(), time(15), ZONE))
    status = (
        "WINDOW_ENDED"
        if expired
        else "WAITING_REVIEW_BASELINE"
        if not baseline
        else "REVIEW_REQUIRED"
        if baseline != observed
        else "WATCHING"
    )
    # 截止日未知时不得用名义日期伪造实际终点或永久观察已过期区间。
    nominal = period.get("nominalEndDate")
    if not end and nominal and at.date().isoformat() >= nominal:
        status = "WAITING_OFFICIAL_END_DATE"
    identity = {
        "owner_hash": owner_hash,
        "fund_code": code,
        "horizon": period["horizonId"],
        "review_id": review_id,
        "reviewed_input_hash": baseline,
        "observed_input_hash": observed,
        "status": status,
        "review_complete": True,
        "direction_available": False,
        "review_required": status in {"REVIEW_REQUIRED", "WAITING_REVIEW_BASELINE"},
        "analysis_at": period["analysisAt"],
        "original_period": {k: period.get(k) for k in ("baseNavDate", "startDate", "endDate", "nominalEndDate")},
        "change_conditions": period.get("conditions", []),
        "next_action": "CONTINUE_ORIGINAL_WINDOW_REVIEW"
        if status in {"REVIEW_REQUIRED", "WAITING_REVIEW_BASELINE"}
        else None,
        "instructions": "阅读本次变化及主要对向依据，沿用原基准和结束日期；另存实际分析时间。"
        "材料未变不重复分析；暂不判断也算已审阅，不能算有方向；旧区间不得关闭新一天任务。",
        "current_materials_path": handoff["materials_path"],
        "current_materials_hash": handoff["materials_hash"],
        "completes_new_daily_window": False,
        "generates_direction": False,
    }
    # 审计路径、完整包摘要可保留，但不凭用途批准时间的变化重新打开经济判断。
    same = previous and all(
        previous.get(k) == v
        for k, v in identity.items()
        if k not in {"current_materials_path", "current_materials_hash"}
    )
    if same:
        return previous, False
    return {
        **identity,
        "changed_at": at.isoformat(),
        "work_item_id": digest(
            {
                "owner": owner_hash,
                "code": code,
                "horizon": period["horizonId"],
                "review": review_id,
                "observed": observed,
                "baseline": baseline,
            }
        ),
    }, True


def stored_reviews(owner_hash, codes):
    """读取每只基金最新正式档案；本人校验在观察入口再次执行。"""
    if not codes:
        return []
    result, archives = [], {}
    with get_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        if connection.scalar(text("SELECT to_regclass('prediction_review_v2_release')")) is None:
            return []
        for code in sorted(set(codes)):
            row = (
                connection.execute(
                    text("""SELECT release_id,views -> :code AS view,
                archive -> 'owner_audit' ->> 'owner_hash' AS owner_hash FROM prediction_review_v2_release
                WHERE views ? :code ORDER BY adopted_at DESC,release_id DESC LIMIT 1"""),
                    {"code": code},
                )
                .mappings()
                .first()
            )
            if row is None:
                continue
            if row["owner_hash"] != owner_hash:
                raise ValueError("WATCH_REVIEW_OWNER_MISMATCH")
            periods = [p for p in row["view"]["periods"] if p["decision"] == "ABSTAIN"]
            if periods:
                release = row["release_id"]
                if release not in archives:
                    archives[release] = connection.execute(
                        text("SELECT archive FROM prediction_review_v2_release WHERE release_id=:id"), {"id": release}
                    ).scalar_one()
                result.append((code, periods, archives[release]))
    return result


def watch_report(owner, codes, *, at, root=None):
    """接在现有每日检查后；无新版本不写观察文件，不额外采集，不新增定时器。"""
    from app.commands.fund_prediction_followthrough import owner_scope
    from app.core.config import get_settings
    from app.services.fund_workflow_materials import registered_materials

    owner_hash = digest(owner)
    # 每轮动态读取，观察中已清仓的基金不再产生新待办，旧证据不删除。
    held = set(owner_scope(get_settings(), owner))
    codes = sorted(set(codes) & held)
    folder = Path(root or ROOT) / owner_hash
    rows, failures = [], []
    for code, periods, archive in stored_reviews(owner_hash, codes):
        try:
            handoff = registered_materials(code, at=at)
            if not handoff:
                raise ValueError("WATCH_CURRENT_MATERIALS_NOT_REGISTERED")
            material = read(Path(handoff["materials_path"]))
            for period in periods:
                baseline = reviewed_material_identity(archive, code, period["horizonId"])
                path = folder / code / (period["horizonId"] + ".json")
                previous = read(path) if path.exists() else None
                state, changed = watch_transition(
                    owner_hash, code, period, baseline, handoff, material, at=at, previous=previous
                )
                if changed:
                    storage_check(at.isoformat())
                    versioned_save(path, state)
                rows.append(state)
        except (ValueError, KeyError, FileNotFoundError) as error:
            # 材料缺失不是“没有新变化”；原待办文件保留，不自动关闭。
            failures.append(
                {"fund_code": code, "status": "CHECK_FAILED", "reason": type(error).__name__ + ":" + str(error)}
            )
    return {
        "checked_at": at.isoformat(),
        "status": "CHECK_FAILED"
        if failures
        else "REVIEW_REQUIRED"
        if any(r["review_required"] for r in rows)
        else "WATCHING",
        "reviewed_abstain_count": len(rows),
        "reassessment_pending_count": sum(r["review_required"] for r in rows),
        "watching_count": sum(r["status"] == "WATCHING" for r in rows),
        "failure_count": len(failures),
        "items": rows,
        "failures": failures,
        "completes_new_daily_window": False,
        "directions_generated": 0,
    }
