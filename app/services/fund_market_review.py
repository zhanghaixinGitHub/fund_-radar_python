"""本机审阅回执：绑定具体公告正文版本，不能因下次采集相同内容自动解除复核。"""

import re
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.services.fund_exposure_common import now, read
from app.services.fund_market_preparation import validate_codes
from app.services.fund_materials_store import versioned_save


def review_root():
    return Path(get_settings().fund_insight_directory) / "market-workflow" / "reviews"


def pending_events(events: list[dict], *, root=None):
    """仅有明确的无评级影响结论才解除候选事项；需重评的事项一直保留待办。"""
    root = root or review_root()
    pending = []
    for event in events:
        validate_codes([event["fund_code"]])
        sha = event["source_hash"]
        if not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError("MARKET_EVENT_HASH")
        path = root / event["fund_code"] / (sha + ".json")
        review = read(path) if path.exists() else None
        if (
            not review
            or review.get("event") != event
            or review.get("decision") not in {"NO_RATING_IMPACT", "RATING_REVIEW_COMPLETED"}
        ):
            pending.append(event)
    return pending


def rating_review_required(code: str):
    """普通评级查询只读本机事项游标，不采集、不重评；历史引用保持原样。"""
    from app.services.fund_market_news import directory

    validate_codes([code])
    path = directory() / "funds" / (code + ".json")
    return bool(pending_events(read(path).get("candidate_events", []))) if path.exists() else False


def record_event_review(
    code: str, article: str, sha: str, decision: str, note: str, *, root=None, news_root=None, rating_batch_id=None
):
    """仅供本机审阅命令使用。原文必须已归档，结论附具体依据，标题不能替代审阅。"""
    from app.services.fund_market_news import directory

    validate_codes([code])
    if (
        not re.fullmatch(r"[a-f0-9]{64}", sha)
        or not re.fullmatch(r"AN\d{18}", article)
        or decision not in {"NO_RATING_IMPACT", "REQUIRES_RATING_REVIEW", "RATING_REVIEW_COMPLETED"}
        or len(note.strip()) < 80
    ):
        raise ValueError("MARKET_EVENT_REVIEW_INVALID")
    state = read((news_root or directory()) / "funds" / (code + ".json"))
    item = state["items"][article]
    if item["source"]["sha256"] != sha or not item["body"] or not item["candidate_material_event"]:
        raise ValueError("MARKET_EVENT_REVIEW_SOURCE_MISMATCH")
    if datetime.fromisoformat(item["received_at"]) > now():
        raise ValueError("MARKET_EVENT_REVIEW_TIME")
    if decision == "RATING_REVIEW_COMPLETED":
        from sqlalchemy import text

        from app.db.session import get_engine

        if not isinstance(rating_batch_id, str) or not re.fullmatch(r"[a-f0-9]{64}", rating_batch_id):
            raise ValueError("MARKET_EVENT_RATING_RECEIPT_REQUIRED")
        with get_engine().connect() as connection:
            row = (
                connection.execute(
                    text("""
                SELECT b.input_snapshot,b.as_of_date,r.status FROM fund_rating_current c
                JOIN fund_rating_batch b USING(batch_id) JOIN fund_rating_result r USING(batch_id)
                WHERE b.batch_id=:batch AND r.fund_code=:code AND b.status='PUBLISHED' AND NOT b.withdrawn
            """),
                    {"batch": rating_batch_id, "code": code},
                )
                .mappings()
                .one_or_none()
            )
        member = next((m for m in row["input_snapshot"]["members"] if m["fund_code"] == code), None) if row else None
        # 必须是包含该公告原文的新正式批次；不能用同基金的旧月末评级作为本次复核回执。
        import json

        if (
            not row
            or row["status"] != "RATED"
            or str(row["as_of_date"]) < item["entry"]["published"]
            or not member
            or sha not in json.dumps(member.get("bundle", {}).get("sources", []))
        ):
            raise ValueError("MARKET_EVENT_RATING_RECEIPT_MISMATCH")
    value = {
        "event": {"fund_code": code, "article_id": article, "source_hash": sha},
        "decision": decision,
        "note": note.strip(),
        "reviewed_at": now().isoformat(),
        "rating_batch_id": rating_batch_id,
    }
    versioned_save((root or review_root()) / code / (sha + ".json"), value)
    return value


def approve_comprehensive(code: str, note: str, *, root=None, materials_path=None, resolutions=None):
    """把已通过原综合校验的本基金资料与本轮来源绑定；不生成判断，不放宽时间/原文检查。"""
    from app.services.direction_1d_protocol import digest
    from app.services.fund_information_snapshot import prepare
    from app.services.fund_workflow_materials import input_gaps, inspect_materials, receipt_path, registered_materials

    validate_codes([code])
    if len(note.strip()) < 80:
        raise ValueError("MARKET_COMPREHENSIVE_REVIEW_REQUIRED")
    root = root or Path(get_settings().fund_insight_directory) / "market-workflow"
    material = read(root / "materials" / (code + ".json"))
    request_hash = (
        digest({"materials": read(Path(materials_path)), "note": note.strip(), "resolutions": resolutions or []})
        if materials_path is not None
        else None
    )
    if request_hash is not None:
        try:
            previous = registered_materials(code, root=root, at=now())
        except ValueError as error:
            if str(error) != "MARKET_COMPREHENSIVE_HANDOFF_CHANGED":
                raise
            previous = None
        if previous and previous.get("reviewed") and previous.get("review_request_hash") == request_hash:
            # 原回执可复用不等于旧行情跨期仍可用；用当前真实窗口重新核对。
            if input_gaps(prepare(now(), code, previous["materials_path"])):
                raise ValueError("MARKET_COMPREHENSIVE_GAPS_REMAIN")
            return {
                "fund_code": code,
                "source_hash": material["source_hash"],
                "reviewed": True,
                "materials_path": previous["materials_path"],
                "reused": True,
            }
    handoff = (
        inspect_materials(code, materials_path, root=root, resolutions=resolutions)
        if materials_path is not None
        else registered_materials(code, root=root, at=now())
    )
    gaps = handoff["gaps"] if handoff is not None else material["gaps"]
    if gaps or pending_events(material["candidate_events"]):
        raise ValueError("MARKET_COMPREHENSIVE_GAPS_REMAIN")
    packet = prepare(now(), code, handoff["materials_path"] if handoff else None)
    if input_gaps(packet):
        raise ValueError("MARKET_COMPREHENSIVE_GAPS_REMAIN")
    path = Path(packet["materials_path"])
    approved = read(path)
    approved.update(
        workflow_source_hash=material["source_hash"],
        workflow_review_note=note.strip(),
        workflow_reviewed_at=now().isoformat(),
    )
    # 审阅元数据写入独立内容寻址包，不能改写已经归档的原资料或其他冻结输入。
    approved_path = root / "comprehensive-inputs" / "packages" / (digest(approved) + ".json")
    if not approved_path.exists():
        from app.services.fund_exposure_common import save

        save(approved_path, approved)
    elif read(approved_path) != approved:
        raise ValueError("MARKET_COMPREHENSIVE_PACKAGE_CHANGED")
    handoff = handoff or inspect_materials(code, path, root=root)
    handoff.update(
        original_materials_path=handoff["materials_path"],
        materials_path=str(approved_path.resolve()),
        materials_hash=digest(approved),
        reviewed=True,
        checked_at=now().isoformat(),
        review_request_hash=request_hash,
    )
    versioned_save(receipt_path(code, root), handoff)
    return {
        "fund_code": code,
        "source_hash": material["source_hash"],
        "reviewed": True,
        "materials_path": str(approved_path.resolve()),
    }
