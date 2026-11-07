"""008888参考资料的重要信息文本政策；不声称复原所有外观或未公开精确数据。

正式发布需覆盖冻结的全部区域：重要内容有经核对的文字/数值，非重要内容有
基于上下文的排除理由。正文、旧政策和原件不覆盖，D盘原件仍长期保留。
"""

from __future__ import annotations

import json

from sqlalchemy import select

from app.models.material_content import MaterialDocument
from app.services import material_content as store
from app.services.material_text_pilot import _part, _save, _text_only

POLICY = "IMPORTANT_TEXT_008888_V1"
COMMON_POLICY = "IMPORTANT_TEXT_V1"
POLICIES = {POLICY, COMMON_POLICY}
PAGE_ENGINE = "IMPORTANT_REGION_TEXT_V1"
RELEASE_ENGINE = "IMPORTANT_RELEASE_V1"
DECISIONS = {"TEXTUALIZED_IMPORTANT", "REUSED_TEXT", "EXCLUDED_LOW_IMPORTANCE"}


def validate(document):
    """逐区域验证取舍及必要字段，不接受只靠非空摘要、OCR分数或完成布尔值发布。"""
    _text_only(document)
    if document.get("schema") not in POLICIES or document.get("coverage") != "IMPORTANT_INFORMATION":
        raise ValueError("IMPORTANT_TEXT_POLICY_REQUIRED")
    expected = document.get("expected_region_ids", [])
    regions = document.get("regions", [])
    identities = [r["region_id"] for r in regions]
    if len(identities) != len(set(identities)) or set(identities) != set(expected):
        raise ValueError("IMPORTANT_TEXT_REGION_COVERAGE_MISMATCH")
    for region in regions:
        if region.get("decision") not in DECISIONS or not region.get("reason") or not region.get("review"):
            raise ValueError("IMPORTANT_TEXT_REGION_DECISION_REQUIRED")
        if region["decision"] == "EXCLUDED_LOW_IMPORTANCE":
            if region.get("uncertain") or region.get("contains_unique_business_fact") is not False:
                raise ValueError("IMPORTANT_TEXT_UNCERTAIN_REGION_MUST_BE_RETAINED")
        elif not region.get("text", "").strip():
            raise ValueError("IMPORTANT_TEXT_BUSINESS_TEXT_REQUIRED")
        if (
            region.get("kind") in {"asset_allocation", "annual_returns"}
            and region["decision"] != "EXCLUDED_LOW_IMPORTANCE"
        ):
            table = region.get("data", {})
            if not table.get("rows") or not table.get("columns") or not table.get("unit"):
                raise ValueError("IMPORTANT_TEXT_EXACT_TABLE_REQUIRED")
        if region.get("kind") == "performance_curve":
            data = region.get("data", {})
            if not data.get("series") or not data.get("period") or not data.get("features"):
                raise ValueError("IMPORTANT_TEXT_CURVE_FACTS_REQUIRED")
            if data.get("values_are_source_precision_intervals") is not True:
                raise ValueError("IMPORTANT_TEXT_CURVE_PRECISION_REQUIRED")
    if not document.get("source_review") or document.get("remaining"):
        raise ValueError("IMPORTANT_TEXT_REVIEW_NOT_FINISHED")


def publish(document, *, repo=None):
    """先核取舍，再追加文本片段/版本，独立读回后公开供普通资料读取器使用。"""
    repo = repo or store.repository()
    validate(document)
    value = repo.get(source_hash=document["source_hash"], fresh=True)
    if value is None or value.content_hash != document["base_content_hash"]:
        raise ValueError("IMPORTANT_TEXT_BASE_CHANGED")
    pages = {}
    for region in document["regions"]:
        if region["decision"] != "EXCLUDED_LOW_IMPORTANCE" and region.get("text"):
            pages.setdefault(str(region.get("page", 1)), []).append(region)
    references = {}
    for page, regions in pages.items():
        record = {"source_hash": value.source_hash, "policy": document["schema"], "page": int(page), "regions": regions}
        references[page] = _part(repo, value.source_hash, PAGE_ENGINE, int(page) - 1, record)
    # 正文/数据只在片段保存一次；版本头保留取舍索引，避免再堆一套完整副本。
    decisions = [{k: v for k, v in region.items() if k not in {"text", "data"}} for region in document["regions"]]
    result = _save(repo, {**document, "regions": decisions, "ready": True, "content": {"important_pages": references}})
    # 发布标记最后追加；未正式核验的暂存片段绝不被旧计算器顺手采用。
    _part(
        repo,
        value.source_hash,
        RELEASE_ENGINE,
        0,
        {
            "policy": document["schema"],
            "source_hash": value.source_hash,
            "document_hash": result["document_hash"],
            "coverage": "IMPORTANT_INFORMATION",
            "pages": references,
        },
    )
    return result


def selected_policy(source_hash, repo):
    published = release(source_hash, repo)
    if published is None:
        return None
    with repo.engine.connect() as c:
        digest = c.scalar(
            select(MaterialDocument.document_hash).where(
                MaterialDocument.source_hash == source_hash,
                MaterialDocument.policy == published["policy"],
                MaterialDocument.ready.is_(True),
            )
        )
    if digest != published["document_hash"]:
        raise ValueError("IMPORTANT_TEXT_RELEASE_HEADER_MISMATCH")
    return published["policy"]


def release(source_hash, repo):
    """只读数据库标记，不检查D盘在线状态，也不在读取请求内补提。"""
    raw = repo.page(source_hash, RELEASE_ENGINE, 0)
    if raw is None:
        return None
    result = json.loads(raw)
    if result.get("policy") not in POLICIES or result.get("source_hash") != source_hash:
        raise ValueError("IMPORTANT_TEXT_RELEASE_IDENTITY_MISMATCH")
    return result


def page_text(source_hash, page_index, repo, published):
    if published is None:
        return ""
    reference = published["pages"].get(str(page_index + 1))
    if reference is None:
        return ""
    raw = repo.page(source_hash, reference["engine"], reference["index"])
    if raw is None or store.checksum(raw.encode()) != reference["hash"]:
        raise ValueError("IMPORTANT_TEXT_PAGE_MISSING_OR_CHANGED")
    record = json.loads(raw)
    if (record["source_hash"] != source_hash or record["page"] != page_index + 1
            or record["policy"] != published["policy"]):
        raise ValueError("IMPORTANT_TEXT_PAGE_IDENTITY_MISMATCH")
    return "\n".join(
        r["text"]
        + ("\n重要数据：" + json.dumps(r["data"], ensure_ascii=False, separators=(",", ":")) if r.get("data") else "")
        for r in record["regions"]
    )


def restore(document, base, repo):
    """通用文档入口提供全部已有正文及新核验的重要事实，始终标明本次覆盖口径。"""
    published = release(base.source_hash, repo)
    if published is None or published["document_hash"] != document["document_hash"]:
        raise ValueError("IMPORTANT_TEXT_RELEASE_NOT_PUBLISHED")
    if base.kind == "PDF":
        document["content"]["pages"] = [
            {
                "page": i + 1,
                "native_text": native,
                "text": native + ("\n" + extra if extra else ""),
                "layout_text": repo.page(base.source_hash, "PYPDF_LAYOUT_V1", i),
            }
            for i, native in enumerate(base.extracted["pdfium"])
            for extra in [page_text(base.source_hash, i, repo, published)]
        ]
    else:
        native = base.extracted.get("plain", base.extracted.get("text", ""))
        extra = page_text(base.source_hash, 0, repo, published)
        document["content"]["text"] = native + ("\n" + extra if extra else "")
    return document
