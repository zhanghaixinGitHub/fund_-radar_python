"""正式002112预测的增量资料解析：先查不可变旧结果，只请求新增原文版本。"""

import json
import logging
from pathlib import Path

from app.core.config import get_settings
from app.schemas.fund_information_analysis import quote_catalog, validate_events
from app.services.fund_document_parse_cache import DocumentParseCache

logger = logging.getLogger(__name__)
EVENT_FIELDS = ("title", "assessment", "stage", "quotes", "meaning", "mechanism", "caveat")
# 旧本地档案把审计索引附在事项上；复用时用当前原文重建，不把它们当模型业务字段。
CACHED_AUDIT_FIELDS = {"document", "id", "quote_spans", "quote_groups", "event_selection", "quote_normalization"}


def parse_documents(
    data: dict, prompt: str, budget, request, *, skipped_documents: list[dict] | None = None
) -> list[dict]:
    """逐份复用或首次解析；不可用材料只跳过自身，其余材料继续。

    材料身份来自来源编号、原文哈希及完整性，不含基金仓位、预测日期或模型名。
    每个新版本只保存首次尝试，失败不会靠再次点击或切换模型自动产生付费重试。
    skipped_documents 接收未采用材料的身份及原因，供综合判断和留档说明真实缺口；
    不把这些材料当成没有事件，不更改旧解析，也不因一份失败中止整批。
    配置、鉴权和每日费用额度等全局拒绝仍向上抛出，不能绕过费用或用途保护。
    """
    if data.get("fund_code", "002112") != "002112":
        raise ValueError("DEEPSEEK_SCOPE_DENIED")
    documents = data["documents"]
    if not documents:
        return []
    settings = get_settings()
    cache = DocumentParseCache(Path(settings.fund_parse_cache_path))
    decisions = [_lookup(cache, doc, data["as_of"]) for doc in documents]
    if getattr(settings, "fund_parse_import_legacy", False) and any(row["state"] == "PENDING" for row in decisions):
        from app.services.fund_material_parse_history import import_legacy

        try:
            receipt = import_legacy(documents, data["as_of"], prompt, cache)
        except ValueError as error:
            if str(error) not in {"ANALYSIS_DOCUMENT_CACHE_BLOCKED", "ANALYSIS_LEGACY_SNAPSHOT_MISMATCH"}:
                raise
            # 旧档案未能完整核对时，只暂停身份未确认的材料；已可复用的资料继续使用。
            # 不把查不到的旧材料误当成新材料付费，也不让它拖停已有可用资料的预测。
            logger.exception("fund_material_parsing.parse_documents   >>> historical lookup incomplete")
            decisions = [
                {"state": "BLOCKED", "reason": "LEGACY_HISTORY_UNVERIFIED"} if row["state"] == "PENDING" else row
                for row in decisions
            ]
        else:
            logger.info(
                "fund_material_parsing.parse_documents   >>> historicalImported=%d historicalBlocked=%d",
                receipt["imported"],
                receipt["blocked"],
            )
            decisions = [_lookup(cache, doc, data["as_of"]) for doc in documents]
    parsed, skipped = [], []
    reused, created = 0, 0

    def skip(document: dict, reason: str, issues: list | None = None) -> None:
        """只留材料身份和原因；正文及首次回答继续由原档案保存。"""
        skipped.append(
            {
                "documentId": document.get("id"),
                "title": document.get("title"),
                "kind": document.get("kind"),
                "reasonCode": reason,
                "issues": list(issues or []),
            }
        )

    for document, decision in zip(documents, decisions, strict=True):
        if document.get("body_scope") != "FULL_TEXT" or document.get("body_truncated"):
            skip(document, "PARTIAL_SOURCE_NOT_FULL_PARSE")
            continue
        if decision["state"] not in {"REUSE", "PENDING"}:
            skip(document, decision.get("reason", "DOCUMENT_PARSE_UNAVAILABLE"), decision.get("issues"))
            continue
        new_material = decision["state"] == "PENDING"
        if decision["state"] == "PENDING":
            # 留出一次综合和一次核验的额度；达到单次上限后跳过尚未开始的材料，
            # 后续可复用资料仍须继续读取，不能因为新资料较多而整体放弃已有证据。
            if budget is not None and budget.calls + 3 > budget.max_calls:
                skip(document, "ANALYSIS_CALL_LIMIT")
                continue
            provenance = {
                "origin": "LIVE_002112_T1",
                "model": "deepseek-flash",
                "target_nav_date": data["window"]["target_nav_date"],
            }
            if not cache.claim(document, data["as_of"], provenance):
                # 另一执行者可能已经完成或正在解析；只查原结果，不能再次发请求。
                decision = _lookup(cache, document, data["as_of"])
                if decision["state"] != "REUSE":
                    skip(document, decision.get("reason", "FIRST_ATTEMPT_RUNNING_OR_OUTCOME_UNKNOWN"))
                    continue
                new_material = False
            else:
                decision = _parse_new(cache, document, data, prompt, budget, request)
                if decision["state"] != "REUSE":
                    skip(document, decision.get("reason", "DOCUMENT_PARSE_FAILED"), decision.get("issues"))
                    continue
        try:
            cached_result = {
                **decision["result"],
                "events": [
                    {key: value for key, value in event.items() if key not in CACHED_AUDIT_FIELDS}
                    for event in decision["result"]["events"]
                ],
            }
            # 仅剥离已知内部审计字段，未知业务字段仍按契约拒绝；首次回答保持原样。
            # 文档身份、逐字引句及字符位置重新来自本次原文，不能相信旧附带的document。
            organized = validate_events({"items": [cached_result]}, [document])
        except (ValueError, TypeError, KeyError):
            # 来源检查通过的旧记录仍可能不满足当前事项契约，保留原记录并仅跳过该份。
            logger.exception(
                "fund_material_parsing.parse_documents   >>> cached result rejected, document_id=%s", document["id"]
            )
            skip(document, "CACHED_RESULT_VALIDATION_FAILED")
            continue
        if decision.get("event_selection"):
            for event in organized:
                event["event_selection"] = dict(decision["event_selection"])
        parsed.extend(organized)
        created += int(new_material)
        reused += int(not new_material)
    if skipped_documents is not None:
        skipped_documents.extend(skipped)
    logger.info(
        "fund_material_parsing.parse_documents   >>> fund=002112 reused=%d new=%d skipped=%d",
        reused,
        created,
        len(skipped),
    )
    return parsed


def _lookup(cache: DocumentParseCache, document: dict, cutoff: str) -> dict:
    """单份身份、时间或缓存内容损坏只影响该份；存储基础设施故障仍由调用方处理。"""
    try:
        return cache.lookup(document, cutoff)
    except (ValueError, TypeError, KeyError):
        logger.exception(
            "fund_material_parsing._lookup   >>> document metadata rejected, document_id=%s", document.get("id")
        )
        return {"state": "BLOCKED", "reason": "DOCUMENT_SOURCE_METADATA_INVALID"}


def _parse_new(cache: DocumentParseCache, document: dict, data: dict, prompt: str, budget, request) -> dict:
    """每个已领取的新版本只请求一次；单份响应错误保留首次结果并交由上层跳过。"""
    raw = None
    try:
        raw = request(
            prompt,
            {
                "fund_code": "002112",
                "target_nav_date": data["window"]["target_nav_date"],
                "documents": [document],
                "quote_catalogs": {document["id"]: quote_catalog(document["body"])},
            },
            "EVENT",
            budget,
        )
        events = validate_events(raw, [document])
        result = {"id": document["id"], "events": [{k: event[k] for k in EVENT_FIELDS} for event in events]}
    except Exception as error:
        if isinstance(error, ValueError) and str(error) in {
            "DEEPSEEK_DAILY_BUDGET",
            "ANALYSIS_CALL_LIMIT",
            "ANALYSIS_NOT_CONFIGURED",
            "DEEPSEEK_DISABLED",
            "DEEPSEEK_SCOPE_DENIED",
            "DEEPSEEK_MATERIAL_SCOPE_DENIED",
        }:
            # 明确尚未发HTTP的拒绝不是一次模型解析，解除本次占用，保留材料等待处理。
            cache.release_unstarted(document)
            raise
        # 保留第一次完整返回或传输未知；同一文件失败后不自动重发。
        cache.finish(document, None, {"first_response": raw}, failure="UNKNOWN" if raw is None else "FAILED")
        logger.exception("fund_material_parsing._parse_new   >>> first parse failed, document_id=%s", document["id"])
        return {"state": "BLOCKED", "reason": "DOCUMENT_PARSE_FAILED"}
    # 只将前十个事项用于判断；完整首次回答及未采用条数另存，跨日复用也保留覆盖局限。
    cache.finish(document, result, {"first_response": raw, "event_selection": events[0]["event_selection"]})
    return _lookup(cache, document, data["as_of"])


def require_new_material(value: dict) -> None:
    """付费传输层再核对一次：仅允许已领取且没有任何历史结果的新原文版本。"""
    documents = value.get("documents")
    if not isinstance(documents, list) or len(documents) != 1:
        raise ValueError("DEEPSEEK_MATERIAL_SCOPE_DENIED")
    cache = DocumentParseCache(Path(get_settings().fund_parse_cache_path))
    key = cache.key(documents[0])
    with cache.connection() as connection:
        row = connection.execute("SELECT provenance_json FROM parse_attempt WHERE cache_key=?", (key,)).fetchone()
        finished = connection.execute("SELECT 1 FROM parse_result WHERE cache_key=?", (key,)).fetchone()
    provenance = json.loads(row["provenance_json"]) if row else {}
    if (
        finished
        or provenance.get("origin") != "LIVE_002112_T1"
        or provenance.get("target_nav_date") != value.get("target_nav_date")
    ):
        raise ValueError("DEEPSEEK_MATERIAL_SCOPE_DENIED")
