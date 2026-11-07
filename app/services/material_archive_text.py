"""D盘原件的全基金文本补提；原生完整、重要图表待核和正式发布分别记账。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from app.services import material_content as store
from app.services import material_important_text as important
from app.services.material_text_pilot import _part

PDF_ENGINE = "NATIVE_PDF_TEXT_V2"
HTML_ENGINE = "NATIVE_HTML_TEXT_V1"
STATUS_ENGINE = "TEXT_CAPTURE_STATUS_V2"
HTML_PART_ENGINE = "NATIVE_HTML_PART_V1"


def normal(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def save_html_text(repo, source_hash, payload):
    """大网页完整分片，保留正文、每个表格单元和链接，不截断成摘要。

    字符串按Unicode字符切分而非字节切分；片段与整体摘要独立读回后登记索引。
    业务已有HTML正文继续从基础文本读取，这里补充可再分析的逻辑表格表示。
    """
    body = store.encoded(payload)
    if len(body) <= 16*1024*1024:
        return _part(repo, source_hash, HTML_ENGINE, 0, payload)
    text, references = body.decode("utf-8"), []
    for index, offset in enumerate(range(0, len(text), 256*1024)):
        references.append(_part(repo, source_hash, HTML_PART_ENGINE, index,
                                {"source_hash": source_hash, "part": index, "text": text[offset:offset+256*1024]}))
    rebuilt = "".join(json.loads(repo.page(source_hash, HTML_PART_ENGINE, i))["text"]
                      for i in range(len(references))).encode("utf-8")
    if store.checksum(rebuilt) != store.checksum(body):
        raise ValueError("MATERIAL_HTML_TEXT_PART_READBACK_MISMATCH")
    return _part(repo, source_hash, HTML_ENGINE, 0,
                 {"source_hash": source_hash, "encoding": "UTF8_JSON_TEXT_PARTS", "bytes": len(body),
                  "content_sha256": store.checksum(body), "parts": references})


def read_html_text(source_hash, *, repo=None):
    """只从数据库读回全部网页文字/表格；分片缺失、次序或摘要变化均拒绝返回。"""
    repo = repo or store.repository()
    header = json.loads(repo.page(source_hash, HTML_ENGINE, 0))
    if header.get("encoding") != "UTF8_JSON_TEXT_PARTS":
        return header
    text = []
    for index, reference in enumerate(header["parts"]):
        raw = repo.page(source_hash, reference["engine"], reference["index"])
        if raw is None or store.checksum(raw.encode()) != reference["hash"]:
            raise ValueError("MATERIAL_HTML_TEXT_PART_CHANGED")
        part = json.loads(raw)
        if part["source_hash"] != source_hash or part["part"] != index:
            raise ValueError("MATERIAL_HTML_TEXT_PART_IDENTITY_CHANGED")
        text.append(part["text"])
    body = "".join(text).encode()
    if len(body) != header["bytes"] or store.checksum(body) != header["content_sha256"]:
        raise ValueError("MATERIAL_HTML_TEXT_PARTS_INCOMPLETE")
    return json.loads(body)


def _publish_plain(value, repo, review):
    """没有未解决视觉/外链区域且正文原值逐项核对时发布；不以摘要替代原文。"""
    document = {
        "schema": important.COMMON_POLICY, "source_hash": value.source_hash,
        "base_content_hash": value.content_hash, "format": value.kind, "coverage": "IMPORTANT_INFORMATION",
        "expected_region_ids": [], "regions": [], "remaining": [], "source_review": review,
    }
    return important.publish(document, repo=repo)


def capture(target, *, repo=None):
    """显式从D盘原件补提；只存文本，未核图表/外链输出待处理项，不自动放行。

    PDF逐页保存独立文字、位置和逻辑表格；与已有PDFium原文核对。栅格文字、
    曲线或彩色矢量待按重要性复核，不以对象面积或OCR分数排除。JSON保留数字词元。
    """
    repo = repo or store.repository()
    target = Path(target)
    from app.services.material_source_archive import archive_path

    if not archive_path(target):
        raise ValueError("MATERIAL_TEXT_CAPTURE_MUST_READ_EXTERNAL_ORIGINAL")
    value = repo.get(target, fresh=True)
    if value is None:
        raise ValueError("MATERIAL_TEXT_CAPTURE_SOURCE_NOT_INDEXED")
    with target.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != value.source_hash:
            raise ValueError("MATERIAL_TEXT_CAPTURE_SOURCE_CHANGED")
    attachments = []
    if value.kind == "PDF":
        from pypdf import PdfReader

        reader = PdfReader(target)
        try:
            metadata = {"is_encrypted": reader.is_encrypted,
                        "metadata": {str(k): str(v) for k, v in (reader.metadata or {}).items()}}
            if repo.page(value.source_hash, "PDF_METADATA_TEXT_V1", 0) is None:
                repo.page(value.source_hash, "PDF_METADATA_TEXT_V1", 0, store.encoded(metadata).decode())
            attachments = list(reader.attachments)
            for index, page in enumerate(reader.pages):
                for engine, layout in (("PYPDF_TEXT_V1", False), ("PYPDF_LAYOUT_V1", True)):
                    if repo.page(value.source_hash, engine, index) is None:
                        repo.page(value.source_hash, engine, index, store.pypdf_page_text(page, layout=layout))
        finally:
            reader.close()
    elif value.kind == "DOCX" and "tables" not in value.extracted:
        # 部分旧Word底稿仅有段落；即便已有审核版本，业务适配器仍需要独立的表格表示。
        # 显式从D盘补齐，不覆盖旧基础摘要，也不等待服务请求回读原件时才修补。
        if repo.page(value.source_hash, "DOCX_STRUCTURE_V1", 0) is None:
            raw = target.read_bytes()
            if store.checksum(raw) != value.source_hash:
                raise ValueError("MATERIAL_DOCX_STRUCTURE_SOURCE_CHANGED")
            _part(repo, value.source_hash, "DOCX_STRUCTURE_V1", 0, store.extract_docx(raw))
    previous = important.release(value.source_hash, repo)
    if previous is not None:
        important.selected_policy(value.source_hash, repo)
        return {"status": "IMPORTANT_TEXT_VERIFIED", "policy": previous["policy"], "reused": True}
    pending, pages, tables_count = [], 0, 0
    if value.kind in {"JSON", "TEXT", "HTML"}:
        raw = target.read_bytes()
        try:
            body = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            body = raw.decode("gb18030")
        if value.kind == "JSON":
            if not value.complete or store.compact_json(body) != value.decode():
                raise ValueError("MATERIAL_TEXT_CAPTURE_JSON_LEXEMES_CHANGED")
            from app.services.material_text_pilot import _text_only

            has_visual_payload = False
            try:
                _text_only(json.loads(body))
            except ValueError:
                has_visual_payload = True
            if has_visual_payload:
                pending.append({"reason": "JSON_VISUAL_PAYLOAD_REQUIRES_TEXT_ADAPTER"})
            else:
                _publish_plain(value, repo, "D盘原件与库内全部JSON文字及数字词元一致，未经浮点重序列化，无视觉载荷")
        elif value.kind == "TEXT":
            if body != value.decode():
                raise ValueError("MATERIAL_TEXT_CAPTURE_FULL_TEXT_CHANGED")
            if not value.complete or re.search(r"data:image/|<svg\b", body, re.I):
                pending.append({"reason": "TEXT_SOURCE_FORMAT_OR_VISUAL_PAYLOAD_REVIEW_REQUIRED", "issue": value.issue})
            else:
                _publish_plain(value, repo, "D盘原件与库内文字逐字核对一致，保留全部数值、条件和原文")
        else:
            from bs4 import BeautifulSoup

            original, stored = BeautifulSoup(body, "html.parser"), BeautifulSoup(value.decode(), "html.parser")
            if normal(original.get_text()) != normal(stored.get_text()):
                raise ValueError("MATERIAL_TEXT_CAPTURE_HTML_BODY_CHANGED")
            logical_tables = [
                [[c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])] for row in table.find_all("tr")]
                for table in original.find_all("table")
            ]
            tables_count = len(logical_tables)
            regions = original.find_all(["img", "svg", "canvas", "iframe", "object", "embed"])
            pending = [{"region": i, "tag": element.name,
                        "label": element.get("alt") or element.get("title"),
                        "source": element.get("src") if not str(element.get("src", "")).startswith("data:") else None,
                        "reason": "外链或视觉区域尚待按上下文确认重要性，不能凭名称或尺寸排除"}
                       for i, element in enumerate(regions)]
            links = [{"text": a.get_text(" ", strip=True), "target": a["href"]}
                     for a in original.find_all("a", href=True) if not a["href"].startswith("data:")]
            save_html_text(repo, value.source_hash,
                           {"source_hash": value.source_hash, "plain": original.get_text("\n"),
                            "tables": logical_tables, "links": links, "pending": pending})
            if value.issue and value.issue != "HTML_EMBEDDED_VISUAL_REQUIRES_TEXT_EXTRACTION":
                pending.append({"reason": "HTML_ORIGINAL_EXTRACTION_ISSUE_REQUIRES_REVIEW", "issue": value.issue})
            if "urn:schemas-microsoft-com:office:spreadsheet" in body:
                from app.services.material_spreadsheet_text import capture_spreadsheetml

                workbook = capture_spreadsheetml(body, value.source_hash, repo)
                tables_count += workbook["tables"]
                pending.extend(workbook["pending"])
            if not pending:
                _publish_plain(value, repo, "独立DOM核对全部正文；逻辑表格及链接已另存文本，页面无待核视觉/外链区域")
    elif value.kind == "PDF":
        import pymupdf

        native = value.extracted["pdfium"]
        if not value.complete:
            pending.append({"reason": "ORIGINAL_EXTRACTION_ISSUE_REQUIRES_REVIEW", "issue": value.issue})
        if attachments:
            pending.append({"reason": "PDF_EMBEDDED_ATTACHMENTS_REQUIRE_TEXT_EXTRACTION", "names": attachments})
        with pymupdf.open(target) as document:
            if len(document) != len(native):
                raise ValueError("MATERIAL_TEXT_CAPTURE_PDF_PAGE_COUNT_CHANGED")
            for index, page in enumerate(document):
                text = page.get_text()
                words = [list(word) for word in page.get_text("words")]
                logical_tables = []
                try:
                    logical_tables = [t.extract() for t in page.find_tables().tables]
                except Exception as error:
                    pending.append({"page": index + 1, "reason": "TABLE_STRUCTURE_REVIEW_REQUIRED",
                                    "error": type(error).__name__})
                if normal(text) != normal(native[index]):
                    pending.append({"page": index + 1, "reason": "NATIVE_TEXT_ORDER_OR_CONTENT_REVIEW_REQUIRED"})
                visuals = [{"kind": "raster", "bbox": list(image["bbox"]),
                            "pixel_digest": image["digest"].hex()}
                           for image in page.get_image_info(hashes=True)]
                for drawing in page.get_drawings():
                    colored = any(c and max(c) - min(c) > 0.08 for c in [drawing["color"], drawing["fill"]])
                    curved = any(item[0] == "c" or item[0] == "l" and abs(item[1].x-item[2].x) > 0.2
                                 and abs(item[1].y-item[2].y) > 0.2 for item in drawing["items"])
                    # 黑白柱图也可能只由填充矩形构成，不能仅凭没有彩色/曲线就判为纯文字。
                    if colored or curved or drawing["fill"] is not None:
                        visuals.append({"kind": "vector", "bbox": list(drawing["rect"])})
                if visuals:
                    pending.append({"page": index + 1, "reason": "IMPORTANT_VISUAL_TEXT_REVIEW_REQUIRED",
                                    "regions": visuals})
                annotations = [{"type": list(a.type), "info": {str(k): str(v) for k, v in a.info.items()},
                                "bbox": list(a.rect)} for a in page.annots() or []]
                widgets = [{"name": w.field_name, "value": w.field_value, "label": w.field_label,
                            "type": w.field_type_string, "bbox": list(w.rect)} for w in page.widgets() or []]
                links = page.get_links()
                if annotations or widgets or any(link.get("uri") or link.get("file") for link in links):
                    pending.append({"page": index+1, "reason": "PDF_ANNOTATIONS_FORMS_OR_LINKS_REVIEW_REQUIRED"})
                _part(repo, value.source_hash, PDF_ENGINE, index,
                      {"source_hash": value.source_hash, "page": index + 1, "text": text,
                       "words": words, "tables": logical_tables, "visual_regions": visuals,
                       "native_agreement": normal(text) == normal(native[index]),
                       "annotations": annotations, "widgets": widgets,
                       "links": [{k: (list(v) if hasattr(v, "x0") else str(v) if hasattr(v, "x") else v)
                                  for k, v in link.items()} for link in links]})
                pages += 1
                tables_count += len(logical_tables)
        if not pending:
            _publish_plain(value, repo, "PDFium与MuPDF逐页全部文字核对一致，逻辑表格已提取，未发现待核栅格或图表区域")
    elif value.kind in {"DOCX", "ARCHIVE", "LEGACY_DOC"}:
        from app.services.material_document import get_review, review_source

        # 沿用已有的Word正文、脚注/尾注、批注、逻辑表格及ZIP成员审查；历史政策
        # 只读回、不覆盖。新片段仍由统一文本守卫拒绝图片/SVG/Base64。
        reviewed = get_review(value.source_hash, repo=repo)
        if reviewed is None:
            reviewed = review_source(target, repo=repo)
        if not reviewed["ready"]:
            pending = [{"reason": "STRUCTURED_DOCUMENT_REVIEW_REQUIRED", "details": reviewed.get("issues", [])}]
    else:
        pending = [{"reason": "FORMAT_TEXT_ADAPTER_REQUIRES_COMPLETION", "format": value.kind, "issue": value.issue}]
    result = {"status": "IMPORTANT_TEXT_VERIFIED" if not pending else "NATIVE_TEXT_SAVED_REVIEW_PENDING",
              "source_hash": value.source_hash, "pages": pages, "tables": tables_count,
              "pending": pending, "original_retained_on_external_drive": True, "binary_payload_saved": False}
    _part(repo, value.source_hash, STATUS_ENGINE, 0, result)
    return result
