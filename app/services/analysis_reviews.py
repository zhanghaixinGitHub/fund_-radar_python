"""将本机冻结的综合事后回看追加留档；与事前预测和到期对错统计完全分开。"""

import hashlib
import json
import logging
import re
from datetime import date, datetime, time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.services.direction_1d_protocol import ZONE, canonical, digest

logger = logging.getLogger(__name__)
KIND = "COMPREHENSIVE_ASOF_REVIEW"
MAX_BYTES = 1_048_576


def _url(value):
    """公开来源只允许普通网页地址；不向页面传递本机路径、凭据或执行性链接。"""
    parsed = urlparse(value or "")
    return (
        value
        if parsed.scheme in {"https", "http"} and parsed.hostname and not parsed.username and not parsed.password
        else None
    )


def _stamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("ANALYSIS_REVIEW_TIMEZONE_REQUIRED")
    return stamp.astimezone(ZONE)


def prepare_archive(directory: Path, fund_code: str, target_date: str) -> tuple[dict, dict]:
    """读取既有原件并复核基金、摘要、时刻、逐字引文和PDF回执；不会重新分析或取数。

    本入口只供本机命令使用。晚取得的原文允许作为明确标记的事后重建资料保存，
    不把 available_at 倒填为 information_cutoff，也不写正式预测、净值或结果核验表。
    """
    directory = Path(directory).resolve()
    files = {}
    for name in ("analysis-input.json", "analysis-result.json", "verification.json"):
        raw = (directory / name).read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError("ANALYSIS_REVIEW_SIZE_LIMIT")
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "value": json.loads(raw.decode("utf-8"))}
    packet, result, verification = (files[name]["value"] for name in files)
    inputs = packet["input"]
    if (
        not re.fullmatch(r"[0-9]{6}", fund_code)
        or inputs["fund_code"] != fund_code
        or result["fund_code"] != fund_code
        or inputs["target_nav_date"] != target_date
        or result["target_nav_date"] != target_date
        or inputs["base_nav_date"] != result["base_nav_date"]
        or inputs["information_cutoff"] != result["information_cutoff"]
        or digest(inputs) != packet["input_hash"]
        or packet["input_hash"] != result["input_hash"]
        or result["kind"] != "CODEX_COMPREHENSIVE_ASOF_REPLAY"
        or result["direction"] not in {"UP", "DOWN"}
        or result["confidence"] != "LOW"
        or result.get("up_probability") is not None
        or result.get("validation_status") != "UNVALIDATED"
    ):
        raise ValueError("ANALYSIS_REVIEW_IDENTITY_INVALID")
    target, base = date.fromisoformat(target_date), date.fromisoformat(result["base_nav_date"])
    cutoff, generated = _stamp(result["information_cutoff"]), _stamp(result["actual_generated_at"])
    if (
        base >= target
        or cutoff.date() != target
        or cutoff >= datetime.combine(target, time(15), ZONE)
        or generated < datetime.combine(target, time(15), ZONE)
        or generated > repo.clock()
        or generated != _stamp(inputs["retrieved_at"])
        or generated != _stamp(verification["checked_at"])
        or inputs["metrics"]["quote_date"] > str(base)
        or any(
            verification.get(key) is not True
            for key in (
                "all_quotes_recomputed",
                "all_document_quotes_exact",
                "frozen_input_hash_matches",
                "target_day_quotes_not_used",
            )
        )
    ):
        raise ValueError("ANALYSIS_REVIEW_BOUNDARY_INVALID")
    sources = {}
    for document in inputs["documents"]:
        if document.get("published_date", str(base)) > str(cutoff.date()):
            raise ValueError("ANALYSIS_REVIEW_FUTURE_SOURCE")
        body = document["body"]
        if document.get("quote") and document["quote"] not in body:
            raise ValueError("ANALYSIS_REVIEW_QUOTE_MISMATCH")
        for span in document["read_ranges"]:
            if not 0 <= span["start"] < span["end"] <= len(body):
                raise ValueError("ANALYSIS_REVIEW_RANGE_INVALID")
        receipt = document["receipt"]
        original = directory / "sources" / Path(receipt["file"]).name
        if hashlib.sha256(original.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("ANALYSIS_REVIEW_SOURCE_CHANGED")
        sources[document["id"]] = {
            "title": "公告原文" if document["review_scope"] == "FULL_TEXT" else "财务报告原文（采用片段）",
            "url": _url(receipt["url"]),
        }
    news = inputs["overnight_news"]
    if _stamp(news["published_at"]) > cutoff:
        raise ValueError("ANALYSIS_REVIEW_FUTURE_SOURCE")
    sources["overnight-news"] = {"title": "隔夜市场报道", "url": _url(news["url"])}
    sources["apple_primary"] = {"title": "苹果发布原文", "url": _url(inputs["apple_primary"]["url"])}
    # 其他被引用资料只从同一目录的原始来源回执读取；标识不能构成本机任意路径。
    for item in [*result["supporting_evidence"], *result["counter_evidence"]]:
        for source_id in item["sources"]:
            if source_id in sources:
                continue
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", source_id):
                raise ValueError("ANALYSIS_REVIEW_SOURCE_INVALID")
            receipt_path = directory / "sources" / (source_id + ".receipt.json")
            path = receipt_path if receipt_path.is_file() else directory / "sources" / (source_id + ".json")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            files["source:" + source_id] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "value": receipt}
            if receipt_path.is_file():
                original = directory / "sources" / Path(receipt["file"]).name
                if hashlib.sha256(original.read_bytes()).hexdigest() != receipt["sha256"]:
                    raise ValueError("ANALYSIS_REVIEW_SOURCE_CHANGED")
            sources[source_id] = {
                "title": "披露报告原文" if receipt_path.is_file() else "已留存行情回执",
                "url": _url(receipt.get("url")),
            }
    archive = {"files": files, "sources": sources}
    if len(canonical(archive).encode("utf-8")) > MAX_BYTES:
        raise ValueError("ANALYSIS_REVIEW_SIZE_LIMIT")

    def evidence(items):
        return [
            {
                "fact": item["fact"],
                "interpretation": item["interpretation"],
                "sources": [sources[key] for key in item["sources"]],
            }
            for item in items
        ]

    view = {
        "recordType": KIND,
        "fundCode": fund_code,
        "targetNavDate": target_date,
        "baseNavDate": str(base),
        "informationCutoff": cutoff.isoformat(),
        "actualGeneratedAt": generated.isoformat(),
        "direction": result["direction"],
        "confidence": "LOW",
        "summary": result["summary"],
        "supportingEvidence": evidence(result["supporting_evidence"]),
        "counterEvidence": evidence(result["counter_evidence"]),
        "changeConditions": result["change_conditions"],
        "limitations": result["limitations"],
        "excludedFromAccuracy": True,
    }
    archive["view"] = view
    if len(canonical(archive).encode("utf-8")) > MAX_BYTES:
        raise ValueError("ANALYSIS_REVIEW_SIZE_LIMIT")
    return archive, view


def import_archive(directory: Path, fund_code: str, target_date: str, *, commit=False) -> dict:
    """默认仅预览；显式提交只追加独立回看，相同原件复用原编号和原生成时刻。"""
    archive, view = prepare_archive(directory, fund_code, target_date)
    content_hash = digest(archive)
    if not commit:
        return {"status": "PREVIEW", "contentHash": content_hash, "review": view}
    with get_engine().begin() as connection:
        if not repo.profiles(connection, [fund_code]):
            raise ValueError("SOURCE_UNAVAILABLE")
        review_id = uuid4()
        connection.execute(
            text("""INSERT INTO fund_analysis_review
          (review_id,fund_code,target_nav_date,base_nav_date,information_cutoff,actual_generated_at,
           input_hash,content_hash,archive,view_payload)
          VALUES(:id,:code,:target,:base,:cutoff,:generated,:input,:hash,CAST(:archive AS jsonb),CAST(:view AS jsonb))
          ON CONFLICT(content_hash) DO NOTHING"""),
            {
                "id": review_id,
                "code": fund_code,
                "target": target_date,
                "base": view["baseNavDate"],
                "cutoff": view["informationCutoff"],
                "generated": view["actualGeneratedAt"],
                "input": archive["files"]["analysis-input.json"]["value"]["input_hash"],
                "hash": content_hash,
                "archive": canonical(archive),
                "view": canonical(view),
            },
        )
        stored = (
            connection.execute(
                text(
                    "SELECT review_id,archive,view_payload,imported_at "
                    "FROM fund_analysis_review WHERE content_hash=:hash"
                ),
                {"hash": content_hash},
            )
            .mappings()
            .one()
        )
        if digest(stored["archive"]) != content_hash or stored["view_payload"] != view:
            raise ValueError("ANALYSIS_REVIEW_ARCHIVE_MISMATCH")
    logger.info(
        "analysis_reviews.import_archive   >>> fund_code=%s target=%s review_id=%s",
        fund_code,
        target_date,
        stored["review_id"],
    )
    return {
        "status": "IMPORTED",
        "reviewId": str(stored["review_id"]),
        "reused": stored["review_id"] != review_id,
        "actualGeneratedAt": view["actualGeneratedAt"],
        "importedAt": stored["imported_at"].isoformat(),
    }


def list_reviews(fund_code: str, *, before=None, before_id=None, limit=20) -> dict:
    """按真实生成时间独立翻页；读取不会生成预测、补结果或触发外部资料调用。"""
    if (before is None) != (before_id is None) or (before is not None and before.tzinfo is None):
        raise ValueError("ANALYSIS_REVIEW_CURSOR_INVALID")
    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text("""SELECT review_id,archive,content_hash,view_payload,imported_at
          FROM fund_analysis_review WHERE fund_code=:code AND
            (CAST(:before AS timestamptz) IS NULL OR (actual_generated_at,review_id) < (:before,CAST(:id AS uuid)))
          ORDER BY actual_generated_at DESC,review_id DESC LIMIT :limit"""),
                {"code": fund_code, "before": before, "id": before_id, "limit": min(50, max(1, limit)) + 1},
            )
            .mappings()
            .all()
        )
    items = []
    for row in rows[:limit]:
        view = row["view_payload"]
        if (
            digest(row["archive"]) != row["content_hash"]
            or view != row["archive"]["view"]
            or view["fundCode"] != fund_code
            or view["recordType"] != KIND
            or view["excludedFromAccuracy"] is not True
        ):
            raise ValueError("ANALYSIS_REVIEW_ARCHIVE_MISMATCH")
        items.append({**view, "reviewId": str(row["review_id"]), "importedAt": row["imported_at"].isoformat()})
    cursor = (
        {"before": items[-1]["actualGeneratedAt"], "beforeId": items[-1]["reviewId"]} if len(rows) > limit else None
    )
    return {"items": items, "nextCursor": cursor}
