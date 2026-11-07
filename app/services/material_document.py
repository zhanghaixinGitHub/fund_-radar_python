"""用途无关的文档底稿及逐文件内容核验；不读取评分、预测或其结果。

原文页序、文字排版、表格/附注和图文缺口是验收对象。两个解析器一致及内容
逐项读回可放行普通文字资料；视觉信息未完成核准、解码不明等情况保留原件。
"""

from __future__ import annotations

import base64
import io
import json
import re
import struct
import unicodedata
import zipfile
import zlib
from contextlib import closing
from pathlib import Path
from xml.etree import ElementTree as ET

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.material_content import MaterialContent, MaterialDocument, MaterialLocation
from app.services import material_content as content
from app.services.material_svg import ASSET_FORMAT, ASSET_KIND, glyph_payload, restore_svg

POLICY = "GENERAL_CONTENT_V1"
MAX_VISUAL_PIXELS = 16_000_000
MAX_PART_BYTES = 20_000_000
MAX_DOCUMENT_VISUAL_BYTES = 4_000_000
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def normalized(text):
    """只归一Unicode等价形式与空白；不能丢掉数字、标点、否定或条件词。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _piece(repo, digest, engine, index, value):
    from app.services.material_text_pilot import _text_only

    # 新底稿只接受可再分析的文本/数值/逻辑表格；历史图文版本仍可读，不再追加图片资产。
    _text_only(value)
    body = content.encoded(value).decode("utf-8")
    if len(body.encode("utf-8")) > MAX_PART_BYTES:
        raise ValueError("MATERIAL_CONTENT_PART_SIZE_LIMIT")
    repo.page(digest, engine, index, body)
    return {"engine": engine, "index": index, "hash": content.checksum(body.encode("utf-8"))}


def _decode(raw):
    try:
        return raw.decode("utf-8-sig"), None
    except UnicodeDecodeError:
        return raw.decode("gb18030"), "ENCODING_REVIEW_REQUIRED"


def _png(page):
    """无额外图像依赖的页面内容预览；不是已完成OCR，也不据预览自动删除。"""
    width, height = page.get_size()
    scale = min(2.0, (MAX_VISUAL_PIXELS / max(width * height, 1)) ** 0.5)
    with closing(page.render(scale=scale, rev_byteorder=True)) as bitmap:
        pixels = bitmap.to_numpy()
        h, w, channels = pixels.shape
        if channels != 3:
            raise ValueError("MATERIAL_VISUAL_CHANNELS_UNSUPPORTED")
        rows = b"".join(b"\x00" + pixels[y].tobytes() for y in range(h))

        def chunk(kind, body):
            return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b"")
        )


def _asset(repo, source_hash, index, name, raw, media_type):
    if len(raw) > MAX_PART_BYTES:
        raise ValueError("MATERIAL_ASSET_SIZE_LIMIT")
    value = {
        "name": name,
        "media_type": media_type,
        "sha256": content.checksum(raw),
        "bytes": len(raw),
        "base64": base64.b64encode(raw).decode("ascii"),
    }
    return _piece(repo, source_hash, "GENERAL_ASSET_V1", index, value)


def _pdf(repo, value, raw, issues):
    import pypdfium2 as pdfium
    from pypdf import PdfReader

    from app.core.pdf import PDF_LOCK

    result = {"pages": [], "attachments": []}
    reader = PdfReader(io.BytesIO(raw))
    if reader.is_encrypted:
        issues.append({"code": "PDF_ENCRYPTION_REVIEW_REQUIRED"})
    result["attachments"] = list(reader.attachments)
    if result["attachments"]:
        issues.append({"code": "PDF_ATTACHMENTS_NOT_EXTRACTED"})
    with PDF_LOCK, pdfium.PdfDocument(raw) as document:
        native_pages = value.extracted.get("pdfium", [])
        if len(document) != len(reader.pages) or len(document) != len(native_pages):
            raise ValueError("MATERIAL_DOCUMENT_PAGE_COUNT_MISMATCH")
        for index in range(len(document)):
            with closing(document[index]) as page, closing(page.get_textpage()) as textpage:
                native = textpage.get_text_range()
                if native != native_pages[index]:
                    raise ValueError("MATERIAL_DOCUMENT_NATIVE_TEXT_MISMATCH")
                independent = reader.pages[index].extract_text() or ""
                layout = reader.pages[index].extract_text(extraction_mode="layout") or ""
                repo.page(value.source_hash, "PYPDF_TEXT_V1", index, independent)
                repo.page(value.source_hash, "PYPDF_LAYOUT_V1", index, layout)
                agreed = normalized(native) == normalized(independent)
                if not agreed:
                    issues.append({"code": "PDF_TEXT_ORDER_OR_CONTENT_DISAGREEMENT", "page": index + 1})
                if "\ufffd" in native or "\ufffd" in independent:
                    issues.append({"code": "PDF_TEXT_ENCODING_REVIEW_REQUIRED", "page": index + 1})
                runs, visual = [], False
                for number, obj in enumerate(page.get_objects(textpage=textpage)):
                    if number >= 100_000:
                        raise ValueError("MATERIAL_PAGE_OBJECT_LIMIT")
                    if obj.type == 1:
                        runs.append({"text": obj.extract(), "bounds": list(obj.get_bounds()), "level": obj.level})
                        if obj.level:
                            issues.append({"code": "PDF_FORM_LAYOUT_REVIEW_REQUIRED", "page": index + 1})
                    elif obj.type in {2, 3, 4}:
                        visual = True
                annotations = []
                for reference in reader.pages[index].get("/Annots", []):
                    annotation = reference.get_object()
                    annotations.append(
                        {
                            key: str(annotation[key])
                            for key in ("/Subtype", "/Contents", "/T", "/V")
                            if key in annotation
                        }
                    )
                    if annotation.get("/Subtype") == "/Widget":
                        issues.append({"code": "PDF_FORM_VALUE_REVIEW_REQUIRED", "page": index + 1})
                record = {
                    "page": index + 1,
                    "size": list(page.get_size()),
                    "native_page_index": index,
                    "layout_engine": "PYPDF_LAYOUT_V1",
                    "layout_hash": content.checksum(layout.encode("utf-8")),
                    "independent_text_hash": content.checksum(independent.encode("utf-8")),
                    "text_runs": runs,
                    "annotations": annotations,
                    "independent_text_agreement": agreed,
                }
                if visual:
                    # 图表尚未转文字时明确留缺口，D盘保留原件；不能用整页PNG冒充解析。
                    issues.append({"code": "PDF_VISUAL_CONTENT_REVIEW_REQUIRED", "page": index + 1})
                result["pages"].append(_piece(repo, value.source_hash, "GENERAL_PDF_PAGE_V1", index, record))
    result["page_count"] = len(result["pages"])
    result["content_basis"] = "FULL_PAGE_TEXT_LAYOUT_AND_POSITIONS"
    return result


def _paragraph(node):
    tokens = []
    for item in node.iter():
        tag = item.tag.rsplit("}", 1)[-1]
        if tag in {"t", "instrText", "delText"}:
            tokens.append({"kind": tag, "text": item.text or ""})
        elif tag in {"footnoteReference", "endnoteReference", "commentReference"}:
            tokens.append({"kind": tag, "id": item.get("{" + NS["w"] + "}id")})
    return {"text": "".join(t.get("text", "") for t in tokens), "tokens": tokens}


def _word(repo, value, raw, issues):
    """正文、表格、页眉页脚、脚注尾注和批注逐部件保留，修订/域代码明确标注。"""
    result = {"parts": [], "media": [], "links": []}
    media_bytes = 0
    note_ids = {"footnote": set(), "endnote": set(), "comment": set()}
    required_notes = {"footnote": set(), "endnote": set(), "comment": set()}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(i.file_size for i in archive.infolist()) > content.MAX_BYTES:
            raise ValueError("MATERIAL_DOCUMENT_ARCHIVE_SIZE_LIMIT")
        names = sorted(archive.namelist())
        for name in names:
            if not name.endswith(".xml") or not name.startswith("word/"):
                continue
            body = archive.read(name)
            if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
                raise ValueError("MATERIAL_DOCUMENT_XML_ENTITY")
            tree = ET.fromstring(body)
            source_paragraphs = tree.findall(".//w:p", NS)
            paragraph_index = {id(p): i for i, p in enumerate(source_paragraphs)}
            paragraphs = [_paragraph(p) for p in source_paragraphs]
            if not paragraphs:
                continue
            tables = []
            for table in tree.findall(".//w:tbl", NS):
                tables.append(
                    [
                        [
                            {
                                "paragraphs": [_paragraph(p) for p in cell.findall("w:p", NS)],
                                "grid_span": [dict(x.attrib) for x in cell.findall("w:tcPr/w:gridSpan", NS)],
                                "vertical_merge": [dict(x.attrib) for x in cell.findall("w:tcPr/w:vMerge", NS)],
                            }
                            for cell in row.findall("w:tc", NS)
                        ]
                        for row in table.findall("w:tr", NS)
                    ]
                )
            source_texts = [
                e.text or "" for e in tree.iter() if e.tag.rsplit("}", 1)[-1] in {"t", "instrText", "delText"}
            ]
            stored_texts = [t["text"] for p in paragraphs for t in p["tokens"] if "text" in t]
            if source_texts != stored_texts:
                issues.append({"code": "WORD_TEXT_OUTSIDE_PARAGRAPH_REVIEW_REQUIRED", "part": name})
            visual = any(
                e.tag.rsplit("}", 1)[-1] in {"drawing", "pict", "object", "oMath", "altChunk"} for e in tree.iter()
            )
            if visual:
                issues.append({"code": "WORD_VISUAL_OR_EMBEDDED_CONTENT_REVIEW_REQUIRED", "part": name})
            containers = []
            for item in tree.iter():
                tag = item.tag.rsplit("}", 1)[-1]
                if tag in note_ids:
                    note_ids[tag].add(item.get("{" + NS["w"] + "}id"))
                    containers.append(
                        {
                            "kind": tag,
                            "attributes": dict(item.attrib),
                            "paragraph_indices": [paragraph_index[id(p)] for p in item.findall(".//w:p", NS)],
                        }
                    )
                if tag in {"footnoteReference", "endnoteReference", "commentReference"}:
                    required_notes[tag.removesuffix("Reference")].add(item.get("{" + NS["w"] + "}id"))
                if tag in {"ins", "del", "moveFrom", "moveTo"}:
                    issues.append({"code": "WORD_REVISIONS_REVIEW_REQUIRED", "part": name})
            result["parts"].append(
                _piece(
                    repo,
                    value.source_hash,
                    "GENERAL_WORD_PART_V1",
                    len(result["parts"]),
                    {"name": name, "paragraphs": paragraphs, "tables": tables, "note_containers": containers},
                )
            )
        for name in names:
            if name.startswith("word/media/") and not name.endswith("/"):
                if archive.getinfo(name).file_size + media_bytes > MAX_DOCUMENT_VISUAL_BYTES:
                    issues.append({"code": "WORD_MEDIA_SIZE_LIMIT", "part": name})
                    continue
                media_bytes += archive.getinfo(name).file_size
                result["media"].append({"name": name, "bytes": archive.getinfo(name).file_size,
                                        "status": "TEXT_EXTRACTION_REQUIRED"})
                issues.append({"code": "WORD_MEDIA_TEXT_EXTRACTION_REQUIRED", "part": name})
            if name.startswith("word/embeddings/"):
                issues.append({"code": "WORD_EMBEDDED_OBJECT_NOT_EXTRACTED", "part": name})
            if name.endswith(".rels"):
                body = archive.read(name)
                if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
                    raise ValueError("MATERIAL_DOCUMENT_XML_ENTITY")
                for relationship in ET.fromstring(body):
                    result["links"].append({"part": name, **dict(relationship.attrib)})
    if not result["parts"]:
        issues.append({"code": "WORD_CONTENT_EMPTY"})
    for kind, ids in required_notes.items():
        if ids - note_ids[kind]:
            issues.append({"code": "WORD_NOTE_REFERENCE_NOT_CLOSED", "kind": kind, "ids": sorted(ids - note_ids[kind])})
    return result


def _text(repo, value, raw, issues):
    text, encoding_issue = _decode(raw)
    if encoding_issue:
        issues.append({"code": encoding_issue})
    if value.kind == "JSON":

        def invalid_constant(token):
            raise ValueError("NON_STANDARD_JSON_CONSTANT")

        json.loads(text, parse_constant=invalid_constant)
        if content.compact_json(text) != value.decode():
            raise ValueError("MATERIAL_DOCUMENT_JSON_LEXEMES_CHANGED")
        return {"body_ref": "BASE_TEXT", "numeric_lexemes_preserved": True}
    if value.kind == "HTML":
        from bs4 import BeautifulSoup

        source, stored = BeautifulSoup(text, "html.parser"), BeautifulSoup(value.decode(), "html.parser")
        if normalized(source.get_text()) != normalized(stored.get_text()):
            raise ValueError("MATERIAL_DOCUMENT_HTML_TEXT_CHANGED")
        # 老版本语义DOM可能改变字符位置；额外保留精确文字，后续新判断不受旧排版约束。
        exact = _piece(repo, value.source_hash, "GENERAL_EXACT_TEXT_V1", 0, {"text": text})
        return {
            "exact_text": exact,
            "headings": [h.get_text() for h in source.find_all(re.compile(r"^h[1-6]$"))],
            "tables": [
                [[c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])] for r in t.find_all("tr")]
                for t in source.find_all("table")
            ],
            "links": [{"text": a.get_text(), "target": a.get("href")} for a in source.find_all("a", href=True)],
        }
    if text != value.decode():
        raise ValueError("MATERIAL_DOCUMENT_TEXT_CHANGED")
    return {"body_ref": "BASE_TEXT"}


def get_review(source_hash, *, repo=None, verify_parts=True, policy=POLICY, _shared_assets=None):
    """按提取政策只读库内底稿并核验全部引用；默认仍读取既有通用底稿。

    历史资料补提采用追加政策版本，不能覆盖旧底稿或借版本切换消除原审核缺口。
    引用的子底稿也必须按其明确政策读取，不能悄悄替换成最新版本。
    """
    repo = repo or content.repository()
    with Session(repo.engine) as session:
        row = session.get(MaterialDocument, (source_hash, policy))
        if row is None:
            return None
        raw = zlib.decompress(row.document_data)
        if content.checksum(raw) != row.document_hash:
            raise ValueError("MATERIAL_DOCUMENT_DATABASE_HASH_MISMATCH")
        document = json.loads(raw)
        if document.get("schema") != policy:
            raise ValueError("MATERIAL_DOCUMENT_POLICY_MISMATCH")
        base = repo.get(source_hash=source_hash, fresh=True)
        if (
            base is None
            or base.content_hash != row.base_content_hash
            or document["base_content_hash"] != base.content_hash
        ):
            raise ValueError("MATERIAL_DOCUMENT_BASE_VERSION_MISMATCH")
        if row.ready != document["ready"]:
            raise ValueError("MATERIAL_DOCUMENT_REVIEW_STATE_MISMATCH")
        digest = row.document_hash
    if verify_parts:
        seen = set()
        shared = {}
        shared_kinds = {}

        def walk(node):
            if isinstance(node, dict):
                if set(node) == {"glyph_id"}:
                    index = node["glyph_id"]
                    catalog = document.get("shared_glyph_catalog", [])
                    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(catalog):
                        raise ValueError("MATERIAL_DOCUMENT_GLYPH_CATALOG_REFERENCE_INVALID")
                    walk(catalog[index])
                    return
                if set(node) == {"asset_kind", "source_hash", "content_hash"}:
                    from app.services import material_raster

                    if node["asset_kind"] not in {ASSET_KIND, material_raster.ASSET_KIND}:
                        raise ValueError("MATERIAL_SHARED_ASSET_KIND_UNSUPPORTED")
                    if shared_kinds.setdefault(node["source_hash"], node["asset_kind"]) != node["asset_kind"]:
                        raise ValueError("MATERIAL_SHARED_ASSET_KIND_CONFLICT")
                    expected = shared.setdefault(node["source_hash"], node["content_hash"])
                    if expected != node["content_hash"]:
                        raise ValueError("MATERIAL_SHARED_ASSET_VERSION_CONFLICT")
                    return
                if set(node) == {"document_source_hash", "document_hash", "policy"}:
                    child = get_review(
                        node["document_source_hash"], repo=repo, policy=node["policy"], _shared_assets=_shared_assets
                    )
                    if (
                        child is None
                        or child["document_hash"] != node["document_hash"]
                        or child["schema"] != node["policy"]
                    ):
                        raise ValueError("MATERIAL_DOCUMENT_ARCHIVE_CHILD_MISSING")
                    return
                if set(node) == {"engine", "index", "hash"}:
                    key = (node["engine"], node["index"], node["hash"])
                    if key in seen:
                        return
                    seen.add(key)
                    body = repo.page(source_hash, node["engine"], node["index"])
                    if body is None or content.checksum(body.encode("utf-8")) != node["hash"]:
                        raise ValueError("MATERIAL_DOCUMENT_PART_MISSING_OR_CHANGED")
                    walk(json.loads(body))
                else:
                    if "layout_engine" in node:
                        for engine, expected in [
                            (node["layout_engine"], node["layout_hash"]),
                            (node.get("independent_text_engine", "PYPDF_TEXT_V1"), node["independent_text_hash"]),
                        ]:
                            body = repo.page(source_hash, engine, node["page"] - 1)
                            if body is None or content.checksum(body.encode("utf-8")) != expected:
                                raise ValueError("MATERIAL_DOCUMENT_LAYOUT_MISSING_OR_CHANGED")
                    for child in node.values():
                        walk(child)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(document)
        # 字形在一份长报告中会被反复引用；一次批量校验，避免每个字符单独查询。
        if shared:
            found = set()
            with Session(repo.engine) as session:
                rows = session.execute(
                    select(MaterialContent.source_hash, MaterialContent.content_hash,
                           MaterialContent.format, MaterialContent.extracted_data)
                    .where(MaterialContent.source_hash.in_(list(shared)))
                ).yield_per(200)
                for identity, expected, kind, compressed in rows:
                    from app.services import material_raster

                    if shared_kinds[identity] == material_raster.ASSET_KIND:
                        if kind != material_raster.ASSET_FORMAT or expected != shared[identity]:
                            raise ValueError("MATERIAL_SHARED_RASTER_METADATA_MISMATCH")
                        value = material_raster.validate_asset(identity, expected, compressed)
                        found.add(identity)
                        if _shared_assets is not None:
                            _shared_assets[identity] = value
                        continue
                    body = zlib.decompress(compressed)
                    if kind != ASSET_FORMAT or expected != shared[identity] or content.checksum(body) != expected:
                        raise ValueError("MATERIAL_SHARED_ASSET_CONTENT_MISMATCH")
                    value = json.loads(body)
                    reference, _ = glyph_payload(value["glyph_node"])
                    if reference["source_hash"] != identity or reference["content_hash"] != expected:
                        raise ValueError("MATERIAL_SHARED_ASSET_IDENTITY_MISMATCH")
                    found.add(identity)
                    if _shared_assets is not None:
                        _shared_assets[identity] = value
            if found != set(shared):
                raise ValueError("MATERIAL_SHARED_ASSET_MISSING")
    return {"document_hash": digest, **document}


def review_source(path, *, repo=None, _raw=None, _source_hash=None, _depth=0):
    """对原件做独立内容提取与核验；通过不代表评分/预测完成，也不修改历史计算。"""
    repo = repo or content.repository()
    path = Path(path)
    value = repo.get(source_hash=_source_hash, fresh=True) if _source_hash else repo.get(path, fresh=True)
    if value is None:
        if _source_hash:
            raise ValueError("MATERIAL_ARCHIVE_CHILD_NOT_STORED")
        value = repo.ingest(path)
    raw = None
    if value.source_size <= content.MAX_BYTES:
        raw = _raw if _raw is not None else path.read_bytes()
        if content.checksum(raw) != value.source_hash:
            raise ValueError("MATERIAL_DOCUMENT_SOURCE_CHANGED")
    cached = get_review(value.source_hash, repo=repo)
    if cached is not None:
        return cached
    issues = []
    if not value.complete:
        issues.append({"code": "BASE_EXTRACTION_REQUIRES_REVIEW", "detail": value.issue})
    document = {
        "schema": POLICY,
        "source_hash": value.source_hash,
        "base_content_hash": value.content_hash,
        "format": value.kind,
        "source_size": value.source_size,
        "purpose": "GENERAL_REUSABLE_CONTENT",
        "content": {},
        "issues": issues,
        "ready": False,
    }
    if value.source_size > content.MAX_BYTES:
        issues.append({"code": "SOURCE_SIZE_REQUIRES_REVIEW"})
    else:
        try:
            if value.kind == "PDF":
                document["content"] = _pdf(repo, value, raw, issues)
            elif value.kind == "DOCX":
                document["content"] = _word(repo, value, raw, issues)
            elif value.kind in {"JSON", "TEXT", "HTML"}:
                document["content"] = _text(repo, value, raw, issues)
            elif value.kind == "LEGACY_DOC":
                document["content"] = {"body_ref": "BASE_TEXT"}
                issues.append({"code": "LEGACY_WORD_MAIN_BODY_ONLY"})
            elif value.kind == "ARCHIVE":
                if _depth >= 3:
                    raise ValueError("MATERIAL_DOCUMENT_ARCHIVE_DEPTH")
                members = {}
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    entries = [i for i in archive.infolist() if not i.is_dir()]
                    if len(entries) > 2048 or sum(i.file_size for i in entries) > content.MAX_BYTES:
                        raise ValueError("MATERIAL_DOCUMENT_ARCHIVE_EXPANSION_LIMIT")
                    if {i.filename for i in entries} != set(value.extracted["members"]):
                        raise ValueError("MATERIAL_DOCUMENT_ARCHIVE_MEMBER_COVERAGE")
                    for item in entries:
                        h = value.extracted["members"][item.filename]["source_hash"]
                        child = review_source(
                            path, repo=repo, _raw=archive.read(item), _source_hash=h, _depth=_depth + 1
                        )
                        members[item.filename] = {
                            "document_source_hash": h,
                            "document_hash": child["document_hash"],
                            "policy": POLICY,
                        }
                        if not child["ready"]:
                            issues.append({"code": "ARCHIVE_CHILD_REVIEW_REQUIRED", "member": item.filename})
                document["content"] = {"members": members}
            else:
                issues.append({"code": "FORMAT_REQUIRES_REVIEW"})
        except Exception as error:
            # 已提取的片段保留，但任何解析异常均不能获得删除资格。
            issues.append({"code": "CONTENT_EXTRACTION_FAILED", "error": type(error).__name__})
    # 一个含多层表单的页面可能重复报告同一缺口；按业务原因与位置去重，不堆积日志。
    unique = {content.encoded(issue): issue for issue in issues}
    issues[:] = list(unique.values())
    document["ready"] = not issues
    data = content.encoded(document)
    digest = content.checksum(data)
    if repo.engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    with repo.engine.begin() as connection:
        statement = insert(MaterialDocument).values(
            source_hash=value.source_hash,
            policy=POLICY,
            base_content_hash=value.content_hash,
            document_hash=digest,
            document_data=zlib.compress(data),
            ready=not issues,
        )
        connection.execute(
            statement.on_conflict_do_nothing(index_elements=[MaterialDocument.source_hash, MaterialDocument.policy])
        )
    stored = get_review(value.source_hash, repo=repo)
    if stored is None or stored["document_hash"] != digest:
        raise ValueError("MATERIAL_DOCUMENT_IMMUTABLE_REVIEW_CONFLICT")
    return stored


def read_document(source_hash, *, repo=None, include_assets=False, policy=None):
    """按明确政策读取完整底稿；全文/表格/脚注可再分析，不依赖PDF或Word原件。"""
    repo = repo or content.repository()
    if policy is None:
        # 优先完整文本，其次采用已正式发布的重要文本；历史引用显式传政策时保持原版本。
        from app.services.material_important_text import selected_policy as important_policy
        from app.services.material_text_pilot import selected_policy

        policy = selected_policy(source_hash, repo) or important_policy(source_hash, repo) or POLICY
    shared_assets = {}
    review = get_review(source_hash, repo=repo, policy=policy, _shared_assets=shared_assets if include_assets else None)
    if review is None:
        raise ValueError("MATERIAL_DOCUMENT_NOT_REVIEWED")
    base = repo.get(source_hash=source_hash)

    def expand(node):
        if isinstance(node, dict):
            if set(node) == {"glyph_id"}:
                # 同一文档的各页共用字形目录，避免每页重复保存两套完整摘要。
                index = node["glyph_id"]
                catalog = review.get("shared_glyph_catalog", [])
                if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(catalog):
                    raise ValueError("MATERIAL_DOCUMENT_GLYPH_CATALOG_REFERENCE_INVALID")
                return expand(catalog[index])
            if set(node) == {"asset_kind", "source_hash", "content_hash"}:
                if not include_assets:
                    return {"shared_asset_ref": node}
                return {**node, **shared_assets[node["source_hash"]]}
            if set(node) == {"document_source_hash", "document_hash", "policy"}:
                return read_document(
                    node["document_source_hash"], repo=repo, include_assets=include_assets, policy=node["policy"]
                )
            if set(node) == {"engine", "index", "hash"}:
                # 历史逐页图文可能较大，默认只返回资产引用；调用方明确需要时才展开。
                if (
                    node["engine"] == "GENERAL_ASSET_V1"
                    or node["engine"].startswith(("HISTORY_SVG_", "HISTORY_PAGE_PNG_", "HISTORY_ATTACHMENT_",
                                                "HISTORY_ARCHIVE_ASSET_", "HISTORY_IMAGE_", "TEXT_FIRST_SVG_"))
                ) and not include_assets:
                    return {"asset_ref": node}
                return expand(json.loads(repo.page(source_hash, node["engine"], node["index"])))
            return {k: expand(v) for k, v in node.items()}
        if isinstance(node, list):
            return [expand(v) for v in node]
        return node

    result = expand(review)
    from app.services.material_important_text import POLICIES as IMPORTANT_POLICIES

    if result.get("schema") in IMPORTANT_POLICIES:
        from app.services.material_important_text import restore

        return restore(result, base, repo)

    def restore_nested(node):
        """内嵌PDF与顶层页面使用相同的内容校验，压缩包中的页面也能独立恢复。"""
        if isinstance(node, dict):
            if "native_text" in node and "native_page_index" in node:
                node["text"] = node["native_text"]
            visual = node.get("visual", {})
            if include_assets and visual.get("format") in {"SVG_SHARED_GLYPHS_V1", "SVG_RASTER_ASSETS_V1"}:
                packed = "".join(part["text"] for part in visual["parts"])
                if len(packed) != visual["characters"] or content.checksum(packed.encode()) != visual["sha256"]:
                    raise ValueError("MATERIAL_DOCUMENT_COMPACT_SVG_MISMATCH")
                visual["svg"] = restore_svg(packed, visual["glyphs"], visual["tree_sha256"], lambda asset: asset)
                if visual.get("format") == "SVG_RASTER_ASSETS_V1":
                    from xml.etree import ElementTree as ET

                    tree = ET.fromstring(visual["svg"])
                    media = {item["placeholder"]: item["asset"] for item in visual["raster_bindings"]}
                    for element in tree.iter():
                        for key, value in list(element.attrib.items()):
                            if value in media:
                                asset = media[value]
                                element.attrib[key] = "data:" + asset["media_type"] + ";base64," + asset["data_base64"]
                    visual["svg"] = ET.tostring(tree, encoding="unicode")
            for child in list(node.values()):
                restore_nested(child)
        elif isinstance(node, list):
            for child in node:
                restore_nested(child)

    restore_nested(result)
    if result.get("schema") == "TEXT_ONLY_008888_V1":
        for item in result["content"]["pages"]:
            page = item["page_content"]
            review = item["review"]
            additions = []
            for region in [] if review is None else review["regions"]:
                figure_data = region.get("data")
                if isinstance(figure_data, dict):
                    # 其他文档只作交叉核对，不能把较晚公开的值混进本原件正文。
                    # 原始复核记录仍随底稿保留，来源哈希/页码明确供审计追溯。
                    figure_data = {
                        k: v
                        for k, v in figure_data.items()
                        if k not in {"verification", "exact_values_from_other_verified_text"}
                    }
                    external = region["data"].get("exact_values_from_other_verified_text")
                    if external is not None:
                        region["cross_source_verification"] = {
                            "role": "VERIFICATION_ONLY_NOT_ORIGINAL_SOURCE_CONTENT",
                            "point_in_time_use_requires_own_publication_cutoff": True,
                            "references": external,
                        }
                additions.append(
                    region["complete_transcription"]
                    + (
                        "\n完整图表数据：\n" + json.dumps(figure_data, ensure_ascii=False, separators=(",", ":"))
                        if figure_data is not None
                        else ""
                    )
                )
            # 日常正文入口可以取得图中全部核验文字；精确表格、图表数据和来源同时保留。
            full_text = page["native_text"] if review is None else review.get("corrected_text", page["native_text"])
            page["text"] = full_text + ("\n" + "\n".join(additions) if additions else "")
            page["layout_text"] = repo.page(source_hash, page["layout_engine"], page["page"] - 1)
        return result
    if result.get("format") == "PDF":
        for page in result["content"].get("pages", []):
            # 追加的历史底稿可以携带本次逐页核对的全文与结构化布局；旧底稿仍沿用原表示。
            page["text"] = (
                page["native_text"] if "native_text" in page else base.extracted["pdfium"][page["native_page_index"]]
            )
            if "layout_engine" in page:
                page["layout_text"] = repo.page(source_hash, page["layout_engine"], page["page"] - 1)
    elif result["content"].get("body_ref") == "BASE_TEXT":
        result["content"]["text"] = base.decode()
    elif "full_text" in result["content"]:
        # 历史大文本以分块保存，返回全文时核对拼接顺序与长度，保留所有字段和正文。
        full = result["content"]["full_text"]
        body = "".join(part["text"] for part in full["parts"])
        if len(body) != full["characters"] or content.checksum(body.encode("utf-8")) != full["sha256"]:
            raise ValueError("MATERIAL_DOCUMENT_FULL_TEXT_MISMATCH")
        result["content"]["text"] = body
    return result


def finish_source(path, *, repo=None):
    """新增资料通过独立验收后清理；存量暂缓标记和内容缺口都保持原件。"""
    repo = repo or content.repository()
    review = review_source(path, repo=repo)
    with Session(repo.engine) as session:
        alias = session.get(MaterialLocation, str(content.resolved(str(path))))
        held = alias is None or alias.retention_hold
    removed = 0
    if review["ready"] and not held and not content.cleanup_on_hold():
        repo.certify({str(path): review["source_hash"]}, review["document_hash"])
        removed = content.delete_verified(path, root=content.PROJECT, verification=review["document_hash"], repo=repo)
    return {
        "source_hash": review["source_hash"],
        "ready": review["ready"],
        "retention_hold": held,
        "deleted_bytes": removed,
        "issues": review["issues"],
    }
