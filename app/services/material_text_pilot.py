"""008888纯文本试点：正文完整保存，视觉信息逐项核验，未完成资料禁止清理。

候选和正式底稿采用不同的不可变政策。正式版本只能引用已核实的文本，
OCR仅作为候选；图片、SVG、字形及原容器均不写入本模块的长期载荷。
"""

from __future__ import annotations

import json
import re
import zlib
from collections import Counter
from contextlib import closing
from pathlib import Path

from sqlalchemy import select

from app.models.material_content import MaterialDocument
from app.services import material_content as store

FUND = "008888"
CANDIDATE = "TEXT_ONLY_008888_C1"
POLICY = "TEXT_ONLY_008888_V1"
PAGE_ENGINE = "TEXT_ONLY_PAGE_V1"
REVIEW_ENGINE = "TEXT_ONLY_REVIEW_V1"
MAX_TEMP_BYTES = 256 * 1024 * 1024
MAX_PART_BYTES = 16 * 1024 * 1024


def _normal(text):
    # 比较时仅删除排版空白，原字符、原顺序另存，不能用归一化覆盖正文。
    return re.sub(r"\s+", "", text)


def _text_only(value):
    """拒绝把图像换个JSON外壳当成文本；位置/表格结构仍是合法业务信息。"""
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"base64", "data_base64", "database64", "svg", "glyphs", "pixels", "image_bytes",
                               "imagebytes", "pixeldata", "base64data"}:
                raise ValueError("TEXT_PILOT_BINARY_OR_VISUAL_PAYLOAD")
            _text_only(child)
    elif isinstance(value, list):
        for child in value:
            _text_only(child)
    elif isinstance(value, bytes):
        raise ValueError("TEXT_PILOT_BINARY_OR_VISUAL_PAYLOAD")
    elif isinstance(value, str) and re.search(r"data:image/|<svg\b", value, re.I):
        raise ValueError("TEXT_PILOT_ENCODED_VISUAL_PAYLOAD")


def _part(repo, source_hash, engine, index, value):
    _text_only(value)
    body = store.encoded(value)
    if len(body) > MAX_PART_BYTES:
        raise ValueError("TEXT_PILOT_PART_TOO_LARGE")
    repo.page(source_hash, engine, index, body.decode("utf-8"))
    return {"engine": engine, "index": index, "hash": store.checksum(body)}


def _save(repo, document):
    """只追加，冲突即失败；独立事务读回，不修改旧底稿和历史结论。"""
    _text_only(document)
    raw = store.encoded(document)
    if repo.engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    with repo.engine.begin() as connection:
        statement = insert(MaterialDocument).values(
            source_hash=document["source_hash"],
            policy=document["schema"],
            base_content_hash=document["base_content_hash"],
            document_hash=store.checksum(raw),
            document_data=zlib.compress(raw, 6),
            ready=document["ready"],
        )
        connection.execute(statement.on_conflict_do_nothing(index_elements=["source_hash", "policy"]))
    from app.services.material_document import get_review

    actual = get_review(document["source_hash"], repo=repo, policy=document["schema"])
    if actual is None or actual["document_hash"] != store.checksum(raw):
        raise ValueError("TEXT_PILOT_IMMUTABLE_DOCUMENT_CONFLICT")
    return actual


def selected_policy(source_hash, repo):
    """只选择已完成的本试点版本；明确指定旧政策的历史引用不受此选择影响。"""
    with repo.engine.connect() as connection:
        ready = connection.scalar(
            select(MaterialDocument.ready).where(
                MaterialDocument.source_hash == source_hash,
                MaterialDocument.policy == POLICY,
            )
        )
    return POLICY if ready else None


def read_candidate(source_hash, *, repo):
    """库内读取全部正文及视觉候选；明确区分未核实内容，不能当作正式完成底稿。"""
    from app.services.material_document import read_document

    document = read_document(source_hash, repo=repo, policy=CANDIDATE)
    if selected_policy(source_hash, repo):
        # 旧候选只供追溯，不把已被正式底稿替代的估读结果重新当作待采用事实。
        document["superseded_by_policy"] = POLICY
    for page in document["content"]["pages"]:
        for region in page["visual_regions"]:
            index = (page["page"] - 1) * 10000 + region["region"]
            for engine, name in [("TEXT_ONLY_OCR_V1", "ocr_candidate"), ("TEXT_ONLY_CURVE_V1", "curve_candidate")]:
                raw = repo.page(source_hash, engine, index)
                if raw is None:
                    continue
                value = json.loads(raw)
                if (
                    value["source_hash"] != source_hash
                    or value["page"] != page["page"]
                    or value["region_hash"] != region["region_hash"]
                ):
                    raise ValueError("TEXT_PILOT_VISUAL_CANDIDATE_IDENTITY_MISMATCH")
                region[name] = value
    return document


def _validate_review_data(region):
    """完成凭据必须有全部行列/系列信息，不能用一个非空的趋势字段冒充数据。"""
    kind, data = region.get("kind"), region.get("data")
    if kind in {"pie_chart", "bar_chart", "table"}:
        if not isinstance(data, dict):
            raise ValueError("TEXT_PILOT_FULL_TABLE_REQUIRED")
        tables = data.get("tables", [data])
        if not tables:
            raise ValueError("TEXT_PILOT_FULL_TABLE_REQUIRED")
        for table in tables:
            columns, rows = table.get("columns"), table.get("rows")
            if (
                not isinstance(columns, list)
                or not columns
                or not isinstance(rows, list)
                or not rows
                or any(not isinstance(row, list) or len(row) != len(columns) for row in rows)
            ):
                raise ValueError("TEXT_PILOT_FULL_TABLE_REQUIRED")
    if isinstance(data, dict) and data.get("semantic_review_required"):
        raise ValueError("TEXT_PILOT_CHART_CANDIDATE_NOT_VERIFIED")
    if isinstance(data, dict) and data.get("kind") == "SOURCE_VISIBLE_CURVE_TABLE":
        proof = data.get("verification", {})
        if (
            proof.get("passed") is not True
            or not proof.get("manual_business_review")
            or proof.get("region_hash") != region.get("region_hash")
            or not proof.get("original_image", {}).get("encoded_image_sha256")
            or any(p.get("uncovered_points") != 0 for p in proof.get("series", []))
            or len(proof.get("series", [])) != len(data.get("data", []))
        ):
            raise ValueError("TEXT_PILOT_INDEPENDENT_CHART_PROOF_REQUIRED")
        for series in data["data"]:
            if not series.get("samples") or series.get("unit") != "%":
                raise ValueError("TEXT_PILOT_FULL_CURVE_TABLE_REQUIRED")


def _rect(value):
    return [float(v) for v in value]


def extract_pdf(path, *, repo, temp_root):
    """完整提取所有页，临时渲染仅供逐图核对，绝不自动授予图像删除资格。

    PyMuPDF由离线命令环境提供；正式读库不依赖该工具。图表、扫描图、
    微小栅格边框及非正交矢量均逐项登记，不能凭面积阈值忽略图中文字。
    """
    import pymupdf
    import pypdfium2 as pdfium
    from pypdf import PdfReader

    from app.core.pdf import PDF_LOCK
    from app.services.material_document import get_review

    path, temp_root = Path(path), Path(temp_root).resolve()
    raw = path.read_bytes()
    source_hash = store.checksum(raw)
    base = repo.get(source_hash=source_hash, fresh=True)
    if base is None or base.kind != "PDF":
        raise ValueError("TEXT_PILOT_BASE_PDF_REQUIRED")
    cached = get_review(source_hash, repo=repo, policy=CANDIDATE)
    if cached is not None:
        return cached
    reader = PdfReader(path)
    issues, pages = [], []
    if reader.is_encrypted or reader.attachments:
        issues.append({"code": "PDF_ENCRYPTION_OR_ATTACHMENTS_REQUIRE_REVIEW"})
    with PDF_LOCK, pdfium.PdfDocument(raw) as native_doc, pymupdf.open(stream=raw, filetype="pdf") as doc:
        if len(doc) != len(native_doc) or len(doc) != len(reader.pages):
            raise ValueError("TEXT_PILOT_PAGE_COUNT_MISMATCH")
        if len(doc) != len(base.extracted["pdfium"]):
            raise ValueError("TEXT_PILOT_BASE_PAGE_COUNT_MISMATCH")
        for index, page in enumerate(doc):
            with closing(native_doc[index]) as native_page, closing(native_page.get_textpage()) as textpage:
                native = textpage.get_text_range()
            if native != base.extracted["pdfium"][index]:
                raise ValueError("TEXT_PILOT_BASE_TEXT_CHANGED")
            independent = page.get_text("text", sort=False)
            other = reader.pages[index].extract_text() or ""
            layout = reader.pages[index].extract_text(extraction_mode="layout") or ""
            repo.page(source_hash, "PYPDF_TEXT_V1", index, other)
            repo.page(source_hash, "PYPDF_LAYOUT_V1", index, layout)
            # 三套全文与文字坐标一并保留；顺序差异继续列缺口，不能靠字符集合放行。
            agreed = _normal(native) == _normal(independent) == _normal(other)
            missing = dict(Counter(_normal(native)) - Counter(_normal(independent)))
            extra = dict(Counter(_normal(independent)) - Counter(_normal(native)))
            page_issues = []
            if not agreed:
                page_issues.append(
                    {"code": "TEXT_ENGINE_DIFFERENCE", "native_only": missing, "independent_only": extra}
                )
            if "\ufffd" in native + independent + other:
                page_issues.append({"code": "TEXT_ENCODING_REVIEW_REQUIRED"})
            annotations = []
            for annotation in page.annots() or []:
                annotations.append({"type": annotation.type, "rect": _rect(annotation.rect), "info": annotation.info})
            widgets = [
                {"name": w.field_name, "value": w.field_value, "type": w.field_type_string, "rect": _rect(w.rect)}
                for w in page.widgets() or []
            ]
            if annotations or widgets:
                page_issues.append({"code": "ANNOTATIONS_OR_FORMS_REQUIRE_REVIEW"})
            tables = []
            for table in page.find_tables().tables:
                tables.append(
                    {
                        "bbox": _rect(table.bbox),
                        "rows": table.row_count,
                        "columns": table.col_count,
                        "cells": table.extract(),
                        "cell_bounds": [None if r is None else _rect(r) for r in table.cells],
                    }
                )
            regions = []
            image_info = page.get_image_info(hashes=True)
            tiny = [i for i in image_info if i["width"] <= 6 and i["height"] <= 6]
            for info in image_info:
                if info in tiny:
                    continue
                regions.append(
                    {
                        "kind": "raster",
                        "bbox": _rect(info["bbox"]),
                        "source_dimensions": [info["width"], info["height"]],
                        "source_pixel_digest": info["digest"].hex(),
                    }
                )
            if tiny:
                # 旧PDF可能用数百张1×6小图绘制表格边线，不能冒称装饰而跳过。
                regions.append(
                    {
                        "kind": "raster_grid",
                        "items": len(tiny),
                        "bbox": [
                            min(i["bbox"][0] for i in tiny),
                            min(i["bbox"][1] for i in tiny),
                            max(i["bbox"][2] for i in tiny),
                            max(i["bbox"][3] for i in tiny),
                        ],
                    }
                )
            for drawing in page.get_drawings():
                complex_path = any(
                    item[0] not in {"l", "re", "qu"}
                    or item[0] == "l"
                    and abs(item[1].x - item[2].x) > 0.1
                    and abs(item[1].y - item[2].y) > 0.1
                    for item in drawing["items"]
                )
                if complex_path:
                    regions.append({"kind": "vector", "bbox": _rect(drawing["rect"])})
            if regions or page_issues:
                target = temp_root / source_hash / f"page-{index + 1}.png"
                if not target.is_relative_to(temp_root):
                    raise ValueError("TEXT_PILOT_TEMP_PATH_ESCAPE")
                target.parent.mkdir(parents=True, exist_ok=True)
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8), alpha=False)
                preview = pixmap.tobytes("png")
                used = sum(p.stat().st_size for p in temp_root.rglob("*.png"))
                if used + len(preview) > MAX_TEMP_BYTES:
                    raise ValueError("TEXT_PILOT_TEMP_BUDGET_REACHED")
                target.write_bytes(preview)
                # 临时图只在本地，库内保存摘要与位置用于后续验收绑定，不保存文件字节。
                preview_hash = store.checksum(preview)
            else:
                preview_hash = None
            for number, region in enumerate(regions):
                region.update(region=number, status="UNVERIFIED", region_hash=store.checksum(store.encoded(region)))
            record = {
                "page": index + 1,
                "native_page_index": index,
                "native_text": native,
                "independent_text": independent,
                "third_text": other,
                "layout_engine": "PYPDF_LAYOUT_V1",
                "layout_hash": store.checksum(layout.encode()),
                "independent_text_hash": store.checksum(other.encode()),
                "size": _rect(page.rect)[2:],
                "words": [list(w) for w in page.get_text("words")],
                "tables": tables,
                "annotations": annotations,
                "widgets": widgets,
                "links": [
                    {k: v for k, v in link.items() if k in {"kind", "uri", "page", "nameddest"}}
                    for link in page.get_links()
                ],
                "visual_regions": regions,
                "text_agreement": agreed,
                "preview_sha256": preview_hash,
                "issues": page_issues,
            }
            pages.append(_part(repo, source_hash, PAGE_ENGINE, index, record))
            if regions or page_issues:
                issues.append({"page": index + 1, "regions": len(regions), "issues": page_issues})
    return _save(
        repo,
        {
            "schema": CANDIDATE,
            "source_hash": source_hash,
            "fund_code": FUND,
            "base_content_hash": base.content_hash,
            "format": "PDF",
            "ready": False,
            "purpose": "COMPLETE_REUSABLE_TEXT_CANDIDATE",
            "extractor": pymupdf.VersionBind,
            "content": {"pages": pages, "page_count": len(pages)},
            "issues": issues,
        },
    )


def publish(source_hash, *, repo, reviews=None):
    """核对所有页及每个区域后才发布。无审阅、只给摘要或曲线缺数据均阻止通过。"""
    from app.services.material_document import get_review, read_document

    if not reviews:
        completed = get_review(source_hash, repo=repo, policy=POLICY)
        if completed is not None and completed["ready"]:
            return completed
    candidate = read_document(source_hash, repo=repo, policy=CANDIDATE)
    if any("code" in issue for issue in candidate.get("issues", [])):
        raise ValueError("TEXT_PILOT_DOCUMENT_LEVEL_ISSUE_NOT_REVIEWED")
    references = get_review(source_hash, repo=repo, policy=CANDIDATE)["content"]["pages"]
    reviews = reviews or {}
    parts, pending = [], []
    for page in candidate["content"]["pages"]:
        review = reviews.get(str(page["page"]))
        needs_review = page["visual_regions"] or page["issues"]
        if needs_review:
            if (
                not review
                or review.get("source_hash") != source_hash
                or review.get("page") != page["page"]
                or review.get("preview_sha256") != page["preview_sha256"]
                or review.get("remaining") != []
            ):
                pending.append({"page": page["page"], "code": "PAGE_REVIEW_INCOMPLETE"})
                continue
            regions = review.get("regions", [])
            expected = {r["region_hash"] for r in page["visual_regions"]}
            if len(regions) != len(expected) or {r.get("region_hash") for r in regions} != expected:
                raise ValueError("TEXT_PILOT_VISUAL_COVERAGE_MISMATCH")
            for region in regions:
                if region.get("status") != "VERIFIED" or region.get("remaining") != []:
                    raise ValueError("TEXT_PILOT_UNVERIFIED_REGION")
                if not region.get("complete_transcription"):
                    raise ValueError("TEXT_PILOT_TRANSCRIPTION_REQUIRED")
                if region.get("kind") in {"line_chart", "bar_chart", "pie_chart"}:
                    if not region.get("data") or not region.get("axes") or not region.get("series"):
                        raise ValueError("TEXT_PILOT_CHART_SUMMARY_IS_INSUFFICIENT")
                _validate_review_data(region)
            if page["issues"] and not review.get("text_and_structure_verification"):
                raise ValueError("TEXT_PILOT_TEXT_DIFFERENCE_NOT_REVIEWED")
            reference = _part(repo, source_hash, REVIEW_ENGINE, page["page"] - 1, review)
        else:
            reference = None
        parts.append({"page_content": references[page["page"] - 1], "review": reference})
    if pending:
        return {"ready": False, "source_hash": source_hash, "pending": pending}
    # 正式底稿引用候选的不可变逐页片段；不再复制整篇正文，也不覆盖候选或旧政策。
    return _save(
        repo,
        {
            "schema": POLICY,
            "source_hash": source_hash,
            "fund_code": FUND,
            "base_content_hash": candidate["base_content_hash"],
            "format": "TEXT_DOCUMENT",
            "ready": True,
            "purpose": "COMPLETE_REUSABLE_TEXT",
            "content": {"pages": parts, "page_count": len(parts)},
            "issues": [],
        },
    )
