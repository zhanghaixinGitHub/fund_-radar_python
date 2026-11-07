"""资料混合读取，迁移原件和已删除原件分别兼容。

同版本已有解析结果可复用，缺少所需表示时按需读D盘原件；可变业务文件按实际文件读取。
保存、读取和业务结束不再自动执行空间优化入库或清理，历史来源身份继续保留。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import lzma
import os
import re
import zlib
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime
from functools import lru_cache, wraps
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from sqlalchemy import case, or_, select
from sqlalchemy.orm import Session

from app.models.material_content import MaterialContent, MaterialLocation, MaterialPageRepresentation

VERSION = "EXTRACTED_MATERIAL_V1"
PROJECT = Path(__file__).resolve().parents[2]
MAX_BYTES = 150_000_000
PAGE_XZ_MAGIC = b"MATERIAL_PAGE_XZ1\x00"
_override: ContextVar = ContextVar("material_repository", default=None)
_database_only: ContextVar = ContextVar("material_database_only", default=False)
_original_only: ContextVar = ContextVar("material_original_validation", default=False)
_operation: ContextVar = ContextVar("material_operation", default=None)
_read_cache: ContextVar = ContextVar("material_read_cache", default=None)
_verification_cache: ContextVar = ContextVar("material_calculation_verification", default=None)
logger = logging.getLogger(__name__)


def compress_page(data):
    """历史大批量图文可选无损XZ压缩；摘要始终绑定解压后的完整内容。

    只追加新片段，不重写既有zlib片段。低预设限制编码/解码字典内存，不引入额外依赖。
    """
    return PAGE_XZ_MAGIC + lzma.compress(data, preset=1)


def decompress_page(data):
    """兼容既有zlib与明确标记的XZ片段；拒绝截断、多余尾流和超限输出。"""
    if not data.startswith(PAGE_XZ_MAGIC):
        return zlib.decompress(data)
    decoder = lzma.LZMADecompressor(memlimit=32 * 1024 * 1024)
    body = decoder.decompress(data[len(PAGE_XZ_MAGIC):], max_length=64 * 1024 * 1024 + 1)
    if len(body) > 64 * 1024 * 1024 or not decoder.eof or decoder.unused_data:
        raise ValueError("MATERIAL_PAGE_COMPRESSED_STREAM_INVALID")
    return body


@lru_cache(maxsize=16384)
def resolved(path: str):
    """读索引时复用稳定路径解析；删除前仍单独重新resolve和校验真实目标。"""
    return Path(path).resolve()


def original_location(path):
    path = Path(path)
    return any(p in {"sources", "raw", "company-sources"} for p in path.parts)


def persistent_location(path):
    path = resolved(str(path))
    from app.core.workspace_paths import runtime_material_path

    path = runtime_material_path(path, project_root=PROJECT)
    return original_location(path) or any(
        path.is_relative_to(PROJECT / "data" / name) for name in ("fund-materials", "fund-insights")
    )


def checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pypdf_page_text(page, *, layout=False):
    """合法空白页可能没有/Contents；两种文字表示均为空，不触发pypdf排版异常。

    页面上的图像、批注或表单是否重要由独立内容审查处理，此处只负责原生文字。
    """
    if page.get("/Contents") is None:
        return ""
    return page.extract_text(**({"extraction_mode": "layout"} if layout else {})) or ""


def encoded(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compact_json(text):
    """只去除字符串外空白，完整保留数字词元、转义和字符串内容。

    大材料可能含数百万个分隔空白。不能为每一段建立一个长期存活的
    Python字符串和列表项；StringIO分批合并写入，临时对象不随段数累积。
    """
    inside, start = False, 0
    pieces = io.StringIO()
    for match in re.finditer(r'"|\s+', text):
        if match[0] == '"':
            index, escapes = match.start() - 1, 0
            while index >= 0 and text[index] == "\\":
                escapes += 1
                index -= 1
            if escapes % 2 == 0:
                inside = not inside
        elif not inside:
            pieces.write(text[start : match.start()])
            start = match.end()
    pieces.write(text[start:])
    return pieces.getvalue()


class SourceContent(bytes):
    """提取后内容；PDF 的底层字节为空，防止意外把文本冒充 PDF 再解析。"""

    def __new__(cls, source_hash, size, kind, extracted, content_hash, complete=True, issue=None):
        value = super().__new__(cls, extracted.get("text", "").encode("utf-8"))
        value.source_hash, value.source_size, value.kind = source_hash, size, kind
        value.extracted, value.content_hash = extracted, content_hash
        value.complete, value.issue = complete, issue
        return value

    def __len__(self):
        # 历史回执比较的是原响应长度，不能误用提取后文字长度改变业务契约。
        return self.source_size

    def __hash__(self):
        return hash(self.cache_identity())

    def cache_identity(self):
        """缓存同时绑定所采用的重要文本版本，基础摘要不变时也不会继续复用旧缺图文本。"""
        published = getattr(self, "important_release", None)
        return self.source_hash, self.content_hash, published.get("document_hash") if published else None

    def __eq__(self, other):
        # PDF不携带字节；缓存身份必须用材料版本，不能让不同PDF都命中空字节键。
        return isinstance(other, SourceContent) and self.cache_identity() == other.cache_identity()

    def __ne__(self, other):
        return not self == other

    def decode(self, encoding="utf-8", errors="strict"):
        # 原编码已在提取时处理，库内文字统一UTF-8；旧适配器传gb18030也不会二次误解码。
        if self.kind == "PDF":
            raise ValueError("MATERIAL_PDF_REQUIRES_EXTRACTED_PAGES")
        return super().decode("utf-8", errors)

    def startswith(self, prefix, *args):
        if self.kind in {"PDF", "DOCX", "ARCHIVE", "LEGACY_DOC"}:
            prefixes = prefix if isinstance(prefix, tuple) else (prefix,)
            signatures = (
                (b"%PDF", b"%PDF-")
                if self.kind == "PDF"
                else ((bytes.fromhex("d0cf11e0a1b11ae1"),) if self.kind == "LEGACY_DOC" else (b"PK\x03\x04",))
            )
            # 旧解析器有的只检查PK两字节，有的检查完整ZIP签名；两种都是对
            # 同一来源格式的识别，不能因原件已转换为文字而误走JSON解析分支。
            return any(signature.startswith(p) for p in prefixes for signature in signatures) and not args
        return super().startswith(prefix, *args)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def content_digest(value: bytes) -> str:
    """库内数据已核提取摘要；返回其来源版本身份供既有回执校验。"""
    return value.source_hash if isinstance(value, SourceContent) else checksum(value)


def extract_docx(raw):
    """正文顺序、表格行列、单位和注释一起保留；不执行宏或外部关系。"""
    import zipfile
    from xml.etree import ElementTree as ET

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if "word/document.xml" not in archive.namelist():
            raise ValueError("MATERIAL_ARCHIVE_REQUIRES_EXPLICIT_EXTRACTION")
        if sum(x.file_size for x in archive.infolist()) > MAX_BYTES:
            raise ValueError("MATERIAL_EXPANDED_SIZE")
        xml = archive.read("word/document.xml")
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        raise ValueError("MATERIAL_XML_ENTITY")
    tree = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = ["".join(p.itertext()) for p in tree.findall(".//w:p", ns)]
    tables, blocks = [], []
    body = tree.find("w:body", ns)
    if body is None:
        raise ValueError("MATERIAL_DOCX_BODY_MISSING")
    for child in body:
        if child.tag.endswith("}tbl"):
            rows = [["".join(cell.itertext()) for cell in row.findall("w:tc", ns)] for row in child.findall("w:tr", ns)]
            tables.append(rows)
            blocks.extend(" ".join(row) for row in rows)
        else:
            blocks.append("".join(child.itertext()))
    return {
        "text": "\n".join(blocks),
        "paragraphs": paragraphs,
        "tables": tables,
        "blocks": blocks,
        "unread_visuals": bool(tree.findall(".//w:drawing", ns) or tree.findall(".//w:pict", ns)),
    }


def opaque_content(raw, suffix, issue):
    """未能完整提取的二进制只登记版本和异常，不入库原字节，也不授予删除资格。"""
    data = {"unsupported_format": suffix, "extraction_issue": issue}
    return SourceContent(checksum(raw), len(raw), "UNSUPPORTED", data, checksum(encoded(data)), False, issue)


def known_legacy_word(source_hash):
    """只重试项目既有三份已核准Word；未知OLE继续保留原件并报告未适配。"""
    from app.services.fund_rating_yy_manufacturing_legacy_word import KNOWN_ORIGINALS

    return source_hash in KNOWN_ORIGINALS


def pdf_page_issue(page, body):
    """带标题的扫描表格也可能无法提取，不能只检查整页文字是否为空。"""
    if "\ufffd" in body:
        return "TEXT_ENCODING_REQUIRES_REVIEW"
    if not body.strip() and any(page.get_objects()):
        return "NON_TEXT_PAGE_REQUIRES_REVIEW"
    width, height = page.get_size()
    area = max(width * height, 1)
    image_area = 0
    for image in page.get_objects(filter=[3]):  # PDFium公开常量FPDF_PAGEOBJ_IMAGE=3。
        left, bottom, right, top = image.get_bounds()
        image_area += max(0, min(width, right) - max(0, left)) * max(0, min(height, top) - max(0, bottom))
    if image_area / area >= 0.2:
        return "LARGE_IMAGE_CONTENT_REQUIRES_REVIEW"
    return None


def extract(raw: bytes, suffix: str, *, _depth=0) -> SourceContent:
    """保留数值、日期、单位、表格排版及完整页序；不丢弃正文或生成主观摘要。"""
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("MATERIAL_SOURCE_SIZE")
    if suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".svg"}:
        # 图片原件只在D盘；尚未完成OCR/图表核验时只存待提取说明，不能把像素作为解析结果。
        return opaque_content(raw, suffix, "IMAGE_REQUIRES_VERIFIED_TEXT_EXTRACTION")
    complete, issue = True, None
    if raw.startswith(b"%PDF"):
        from contextlib import closing

        import pypdfium2 as pdfium

        from app.core.pdf import PDF_LOCK

        # 全篇正文统一提取；旧算法确实使用的另一种排版逐页另存，不为每份报告
        # 反复执行三套全篇解析。所有文字页均保留，绝不只取前几页或主观摘要。
        native = []
        with PDF_LOCK, pdfium.PdfDocument(raw) as doc:
            if not 0 < len(doc) <= 2500:
                raise ValueError("MATERIAL_PAGE_LIMIT")
            for i in range(len(doc)):
                with closing(doc[i]) as page, closing(page.get_textpage()) as text:
                    body = text.get_text_range()
                    native.append(body)
                    page_issue = pdf_page_issue(page, body)
                    if page_issue:
                        complete, issue = False, page_issue
        if not any(p.strip() for p in native):
            complete, issue = False, "EMPTY_OR_SCANNED_DOCUMENT"
        kind, value = (
            "PDF",
            {"canonical_engine": "PDFIUM_TEXT_V1", "pdfium": native, "completeness_policy": "PDF_RASTER_AREA_V1"},
        )
    elif raw.startswith(b"PK\x03\x04"):
        import zipfile

        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = [item for item in archive.infolist() if not item.is_dir()]
            if "word/document.xml" in archive.namelist():
                kind, value = "DOCX", extract_docx(raw)
                if value["unread_visuals"]:
                    complete, issue = False, "DOCX_VISUALS_REQUIRE_REVIEW"
            else:
                if _depth >= 3 or len(entries) > 2048 or sum(i.file_size for i in entries) > MAX_BYTES:
                    raise ValueError("MATERIAL_ARCHIVE_EXPANSION_LIMIT")
                members, descriptions = {}, {}
                for item in entries:
                    name = PurePosixPath(item.filename.replace("\\", "/"))
                    if name.is_absolute() or ".." in name.parts or ":" in item.filename:
                        raise ValueError("MATERIAL_ARCHIVE_MEMBER_SCOPE")
                    body = archive.read(item)
                    try:
                        member = extract(body, name.suffix.lower(), _depth=_depth + 1)
                    except Exception as error:
                        member = opaque_content(body, name.suffix.lower(), type(error).__name__)
                    members[item.filename] = member
                    descriptions[item.filename] = {
                        "source_hash": member.source_hash,
                        "size": item.file_size,
                        "flags": item.flag_bits,
                        "crc": item.CRC,
                        "kind": member.kind,
                    }
                value = {"members": descriptions}
                result = SourceContent(
                    checksum(raw),
                    len(raw),
                    "ARCHIVE",
                    value,
                    checksum(encoded(value)),
                    all(m.complete for m in members.values()),
                    None if all(m.complete for m in members.values()) else "ARCHIVE_MEMBER_INCOMPLETE",
                )
                result.members = members
                return result
    elif raw.startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
        from app.services.fund_rating_yy_manufacturing_legacy_word import extract_body

        # 只支持原有解析器已经独立核准的三份旧Word，不扩展到未知OLE或执行宏。
        kind, value = "LEGACY_DOC", {"text": extract_body(raw), "extractor": "REVIEWED_CFB_DOC_BODY_V1"}
        complete, issue = False, "LEGACY_WORD_MAIN_BODY_ONLY_REQUIRES_REVIEW"
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gb18030")
        looks_json = text.lstrip().startswith(("{", "[", '"')) or text.strip() in {"true", "false", "null"}
        if looks_json:
            # 只移除字符串外空白，保留精确数字词元，Decimal 计算不会经 float 舍入。
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # 返回过非标准正文的旧响应仍可保留版本身份，但不能当作完整结构化资料清理。
                complete, issue = False, "INVALID_JSON_REQUIRES_REVIEW"
                kind, value = "TEXT", {"text": text}
            else:
                from app.services.material_text_pilot import _text_only

                # JSON字符串不能充当图片像素/字形/Base64的外壳。先检查解码后的
                # 全部字段和字符串（含Unicode转义），数字仍按原词元存储而不重序列化。
                _text_only(parsed)
                del parsed
                text = compact_json(text)
                kind, value = "JSON", {"text": text}
        elif suffix.lower() in {".html", ".htm"} or text.lstrip().startswith("<"):
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            if "urn:schemas-microsoft-com:office:spreadsheet" in text and any(
                re.search(r"image|picture|chart|drawing|object", element.name.rsplit(":", 1)[-1], re.I)
                for element in soup.find_all()
            ):
                # XML工作簿也可能内嵌图像字节；不能把整段XML作为HTML文字绕过像素禁存。
                # 未核图形原件只留D盘，底稿登记缺口，显式文本适配器再提取业务事实。
                return opaque_content(raw, suffix, "SPREADSHEETML_VISUAL_REQUIRES_TEXT_EXTRACTION")
            visuals = []
            for element in list(soup.find_all("svg")) + list(soup.find_all(src=re.compile(r"^data:image/", re.I))):
                if element.parent is None:
                    continue
                visuals.append({
                    "tag": element.name,
                    "label": element.get("alt") or element.get("title") or element.get_text(" ", strip=True),
                })
                element.decompose()
            if visuals:
                # HTML可嵌入整张图片/SVG；移除像素载荷并保留缺口，其他正文与表格继续入库。
                text = str(soup)
                complete, issue = False, "HTML_EMBEDDED_VISUAL_REQUIRES_TEXT_EXTRACTION"
            # 保存解析后的语义 DOM 和表格；原 HTML 文件及响应字节不入库。
            kind, value = (
                "HTML",
                {
                    "text": text,
                    "exact_html_text": not visuals,
                    **({"visuals_requiring_text": visuals} if visuals else {}),
                    "plain": soup.get_text("\n", strip=True),
                    "tables": [
                        [[c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])] for r in t.find_all("tr")]
                        for t in soup.find_all("table")
                    ],
                },
            )
        elif suffix.lower() in {".txt", ".csv", ".js", ".json", ".xml", ".md", ".markdown"}:
            kind, value = "TEXT", {"text": text}
        else:
            raise ValueError("MATERIAL_UNSUPPORTED_FORMAT")
        if suffix.lower() == ".json" and kind != "JSON" and issue is None:
            complete, issue = False, "SOURCE_FORMAT_DIFFERS_FROM_JSON_SUFFIX"
        if suffix.lower() == ".pdf" and kind != "PDF":
            complete, issue = False, "SOURCE_FORMAT_DIFFERS_FROM_PDF_SUFFIX"
    from app.services.material_text_pilot import _text_only

    _text_only(value)
    data = encoded(value)
    return SourceContent(checksum(raw), len(raw), kind, value, checksum(data), complete, issue)


class MaterialRepository:
    """提取内容不可变；同路径的新来源版本只切换索引，旧版本继续可读。"""

    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def location_candidates(path):
        """迁移后按有界候选查询历史别名，保持精确路径版本优先，不触碰源文件。"""
        from app.core.config import get_settings
        from app.core.workspace_paths import material_location_aliases

        return material_location_aliases(
            resolved(str(path)), project_root=PROJECT, archive_root=get_settings().fund_source_archive_root
        )

    def metadata(self, path=None, source_hash=None):
        """存在性、原长度和重复来源登记只取索引，避免反复传输整篇正文。"""
        with Session(self.engine) as session:
            query = select(
                MaterialContent.source_hash,
                MaterialContent.content_hash,
                MaterialContent.format,
                MaterialContent.source_size,
                MaterialContent.complete,
                MaterialContent.issue,
            )
            if path is not None:
                names = self.location_candidates(path)
                query = query.join(MaterialLocation, MaterialLocation.source_hash == MaterialContent.source_hash).where(
                    MaterialLocation.location.in_(names)
                ).order_by(case({name: index for index, name in enumerate(names)}, value=MaterialLocation.location))
            else:
                query = query.where(MaterialContent.source_hash == source_hash)
            row = session.execute(query).first()
            if row is None:
                return None
            return SimpleNamespace(
                source_hash=row[0], content_hash=row[1], kind=row[2], source_size=row[3], complete=row[4], issue=row[5]
            )

    def link(self, path, source_hash):
        """重复原件只登记来源版本，不复制提取内容；别名事务提交后独立读回。"""
        if self.engine.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            from sqlalchemy.dialects.sqlite import insert
        location = str(resolved(str(path)))
        with self.engine.begin() as connection:
            statement = insert(MaterialLocation).values(location=location, source_hash=source_hash)
            statement = statement.on_conflict_do_update(
                index_elements=[MaterialLocation.location],
                set_={
                    "source_hash": source_hash,
                    "deleted_at": None,
                    "verification": None,
                    "verified_at": datetime.now(UTC),
                },
                where=MaterialLocation.source_hash != source_hash,
            )
            connection.execute(statement)
        stored = self.metadata(path=path)
        if stored is None or stored.source_hash != source_hash:
            raise ValueError("MATERIAL_LOCATION_READBACK_FAILED")
        return stored

    def get(self, path=None, source_hash=None, *, fresh=False):
        cache = _read_cache.get()
        cache_key = (id(self.engine), str(resolved(str(path))) if path is not None else source_hash)
        if cache is not None and fresh:
            # 独立读回可能把小型失败占位恢复为超过缓存上限的完整正文。
            # 新正文不缓存时也必须移除旧占位，否则同一业务操作会继续读到失败。
            old_cached = cache["values"].pop(cache_key, None)
            if old_cached is not None:
                cache["size"] -= old_cached[1]
        if cache is not None and not fresh and cache_key in cache["values"]:
            cache["values"].move_to_end(cache_key)
            return cache["values"][cache_key][0]
        with Session(self.engine) as session:
            if source_hash is None and path is not None:
                names = self.location_candidates(path)
                source_hash = session.scalar(
                    select(MaterialLocation.source_hash).where(MaterialLocation.location.in_(names)).order_by(
                        case({name: index for index, name in enumerate(names)}, value=MaterialLocation.location)
                    ).limit(1)
                )
            row = session.get(MaterialContent, source_hash) if source_hash else None
            if row is None:
                return None
            if cache is not None and fresh:
                # 同一原文也可按摘要读取；恢复占位时路径和摘要两种缓存不能分叉。
                source_key = (id(self.engine), source_hash)
                old_source = cache["values"].pop(source_key, None)
                if old_source is not None:
                    cache["size"] -= old_source[1]
            data = zlib.decompress(row.extracted_data)
            if checksum(data) != row.content_hash or row.extractor_version != VERSION:
                raise ValueError("MATERIAL_DATABASE_CONTENT_MISMATCH")
            result = SourceContent(
                row.source_hash,
                row.source_size,
                row.format,
                json.loads(data),
                row.content_hash,
                row.complete,
                row.issue,
            )
            if cache is not None and len(data) <= 32 * 1024 * 1024:
                old_cached = cache["values"].pop(cache_key, None)
                if old_cached is not None:
                    cache["size"] -= old_cached[1]
                cache["values"][cache_key] = (result, len(data))
                cache["size"] += len(data)
                while cache["size"] > 64 * 1024 * 1024:
                    _, (_, size) = cache["values"].popitem(last=False)
                    cache["size"] -= size
            return result

    def put(self, path, value: SourceContent):
        for member in getattr(value, "members", {}).values():
            # 压缩包只保留成员清单与成员提取信息，绝不把整个ZIP搬到数据库。
            virtual = PROJECT / "data/fund-materials/archive-members" / member.source_hash
            existing = self.get(source_hash=member.source_hash)
            # 空成员的SourceContent长度为0，布尔值为False，但仍是有效的已存版本。
            # 必须按None判断是否缺记录，避免不同扩展名的空成员覆盖既有提取身份。
            selected = member if existing is None else existing
            stored_member = self.put(virtual, selected)
            if not stored_member.complete:
                value.complete, value.issue = False, "ARCHIVE_MEMBER_INCOMPLETE"
        data = encoded(value.extracted)
        if checksum(data) != value.content_hash:
            raise ValueError("MATERIAL_EXTRACTION_HASH_MISMATCH")
        if self.engine.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            from sqlalchemy.dialects.sqlite import insert
        # 并发取得同一材料时只插入一次；既有提取内容不能被第二个执行者覆盖。
        with self.engine.begin() as connection:
            statement = insert(MaterialContent).values(
                source_hash=value.source_hash,
                content_hash=value.content_hash,
                format=value.kind,
                source_size=value.source_size,
                extractor_version=VERSION,
                extracted_data=zlib.compress(data, 6),
                complete=value.complete,
                issue=value.issue,
            )
            connection.execute(statement.on_conflict_do_nothing(index_elements=[MaterialContent.source_hash]))
            actual = connection.scalar(
                select(MaterialContent.content_hash).where(MaterialContent.source_hash == value.source_hash)
            )
            if actual != value.content_hash:
                raise ValueError("MATERIAL_IMMUTABLE_EXTRACTION_CONFLICT")
        self.link(path, value.source_hash)
        # 独立事务读回；写入函数未抛错不等于资料已经完整可用。
        stored = self.get(path, fresh=True)
        if stored is None or stored.content_hash != value.content_hash:
            raise ValueError("MATERIAL_READBACK_FAILED")
        return stored

    def ingest(self, path, *, metadata_only=False):
        path = Path(path)
        if path.stat().st_size > MAX_BYTES:
            digest, size = hashlib.sha256(), 0
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                    size += len(block)
            data = {"unsupported_format": path.suffix, "extraction_issue": "SOURCE_SIZE_LIMIT"}
            value = SourceContent(
                digest.hexdigest(), size, "UNSUPPORTED", data, checksum(encoded(data)), False, "SOURCE_SIZE_LIMIT"
            )
            return self.put(path, value)
        raw = path.read_bytes()
        digest = checksum(raw)
        metadata = self.metadata(source_hash=digest)
        if (
            metadata is not None
            and metadata.kind == "UNSUPPORTED"
            and (
                path.suffix.lower() in {".md", ".markdown"}
                or known_legacy_word(digest)
                or (path.suffix.lower() == ".json" and metadata.issue == "MemoryError")
            )
        ):
            # 首轮未识别的文本/已核旧Word只登记了失败元数据，没有可用正文。仅允许补齐
            # 这种占位记录，并把首次失败信息完整保留；已存在的提取正文仍不可改写。
            parsed = extract(raw, path.suffix)
            if path.suffix.lower() == ".json" and (parsed.kind != "JSON" or not parsed.complete):
                # 修复内存占位只接受完整有效JSON；损坏文本不能覆盖首次失败，
                # 更不能借此放行清理或预测。来源摘要仍由本次真实原件决定。
                raise ValueError("MATERIAL_JSON_RECOVERY_REQUIRES_COMPLETE_CONTENT")
            with Session(self.engine) as session, session.begin():
                row = session.scalar(
                    select(MaterialContent).where(MaterialContent.source_hash == digest).with_for_update()
                )
                if row.format == "UNSUPPORTED":
                    previous = json.loads(zlib.decompress(row.extracted_data))
                    if set(previous) != {"unsupported_format", "extraction_issue"}:
                        raise ValueError("MATERIAL_UNSUPPORTED_PLACEHOLDER_NOT_EMPTY")
                    parsed.extracted["initial_extraction"] = {
                        "format": row.format,
                        "content_hash": row.content_hash,
                        "metadata": previous,
                        "extractor_version": row.extractor_version,
                    }
                    data = encoded(parsed.extracted)
                    row.format, row.content_hash = parsed.kind, checksum(data)
                    row.extracted_data = zlib.compress(data, 6)
                    row.complete, row.issue = parsed.complete, parsed.issue
            self.link(path, digest)
            return self.get(path, fresh=True)
        if metadata_only:
            if metadata is not None:
                result = self.link(path, digest)
                if result.kind == "DOCX":
                    self.page(digest, "DOCX_STRUCTURE_V1", 0, json.dumps(extract_docx(raw), ensure_ascii=False))
                return result
        cached = self.get(source_hash=digest)
        if cached is None:
            try:
                cached = extract(raw, path.suffix)
            except Exception as error:
                cached = opaque_content(raw, path.suffix, type(error).__name__)
        result = self.put(path, cached)
        if result.kind == "DOCX":
            self.page(digest, "DOCX_STRUCTURE_V1", 0, json.dumps(extract_docx(raw), ensure_ascii=False))
        return result

    def certify(self, paths, verification):
        """批量核来源版本后登记凭据；全事务提交，不逐行自动刷新放大数据库负担。"""
        if not re.fullmatch(r"[0-9a-f]{64}", verification):
            raise ValueError("MATERIAL_VERIFICATION_ID")
        from sqlalchemy import update

        items = [(str(resolved(str(p))), paths[p] if isinstance(paths, dict) else None) for p in paths]
        with Session(self.engine) as session, session.begin():
            for offset in range(0, len(items), 1000):
                batch = items[offset : offset + 1000]
                names = [p for p, _ in batch]
                stored = dict(
                    session.execute(
                        select(MaterialLocation.location, MaterialLocation.source_hash).where(
                            MaterialLocation.location.in_(names)
                        )
                    ).all()
                )
                for path, expected in batch:
                    if path not in stored:
                        raise ValueError("MATERIAL_VERIFICATION_LOCATION_MISSING")
                    if expected is not None and stored[path] != expected:
                        raise ValueError("MATERIAL_VERIFICATION_SOURCE_CHANGED")
                session.execute(
                    update(MaterialLocation)
                    .where(MaterialLocation.location.in_(names))
                    .values(verification=verification)
                )

    def page(self, source_hash, engine, page_index, value=None):
        """页面表示只追加一次；并发首读也不能覆盖另一执行者已核验的文本。"""
        key = {"source_hash": source_hash, "engine": engine, "page_index": page_index}
        if value is not None:
            if self.engine.dialect.name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert
            else:
                from sqlalchemy.dialects.sqlite import insert
            data = value.encode("utf-8")
            with self.engine.begin() as connection:
                statement = insert(MaterialPageRepresentation).values(
                    **key, content_hash=checksum(data), extracted_text=zlib.compress(data)
                )
                connection.execute(
                    statement.on_conflict_do_nothing(
                        index_elements=[
                            MaterialPageRepresentation.source_hash,
                            MaterialPageRepresentation.engine,
                            MaterialPageRepresentation.page_index,
                        ]
                    )
                )
        with Session(self.engine) as session:
            row = session.get(MaterialPageRepresentation, tuple(key.values()))
            if row is None:
                return None
            data = decompress_page(row.extracted_text)
            if checksum(data) != row.content_hash or value is not None and data.decode("utf-8") != value:
                raise ValueError("MATERIAL_PAGE_CONTENT_MISMATCH")
            return data.decode("utf-8")


def repository():
    if _override.get() is not None:
        return _override.get()
    from app.db.session import get_engine

    return MaterialRepository(get_engine())


def material_read_cache(function):
    """单次业务读取内有界复用已核内容，结束即释放，不跨请求缓存业务结论。"""

    @wraps(function)
    def run(*args, **kwargs):
        if _read_cache.get() is not None:
            return function(*args, **kwargs)
        token = _read_cache.set({"values": OrderedDict(), "size": 0})
        try:
            return function(*args, **kwargs)
        finally:
            _read_cache.reset(token)

    return run


def verification_memo(function):
    """仅批量验收复用相同冻结输入的复算；原件/库内/禁原件三种条件分别计算。"""

    @wraps(function)
    def run(*args, **kwargs):
        cache = _verification_cache.get()
        if cache is None:
            return function(*args, **kwargs)
        identity = json.dumps([args, kwargs], ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        key = (
            function.__module__,
            function.__qualname__,
            checksum(identity),
            _original_only.get(),
            _database_only.get(),
        )
        if key in cache["values"]:
            cache["hits"] += 1
            return deepcopy(cache["values"][key])
        result = function(*args, **kwargs)
        cache["executions"] += 1
        if len(cache["values"]) < 512:
            cache["values"][key] = deepcopy(result)
        return result

    return run


@contextmanager
def bulk_verification():
    """验收期间有界复用资料；不跨验收运行保存结论，结束释放全部缓存。"""
    memo = {"values": {}, "hits": 0, "executions": 0}
    math_token = _verification_cache.set(memo)
    read_token = _read_cache.set({"values": OrderedDict(), "size": 0})
    try:
        yield memo
    finally:
        _read_cache.reset(read_token)
        _verification_cache.reset(math_token)


@contextmanager
def using_repository(repo):
    """为隔离测试提供真实事务库，不改全局生产配置。"""
    token = _override.set(repo)
    try:
        yield
    finally:
        _override.reset(token)


def managed(path):
    path = resolved(str(path))
    if _override.get() is not None:
        return True
    from app.core.workspace_paths import runtime_material_path
    from app.services.material_source_archive import archive_path

    current = runtime_material_path(path, project_root=PROJECT)
    return archive_path(path) or any(current.is_relative_to(PROJECT / name) for name in ("data", ".local-runs"))


def source_bytes(path):
    """默认混合读取；显式数据库验收只走历史表示，且不自动补提。"""
    if not _database_only.get() and not _original_only.get():
        from app.services.material_file_access import read_material

        return read_material(path)
    path = Path(path)
    from app.core.config import get_settings
    from app.core.workspace_paths import runtime_material_path

    physical = runtime_material_path(path, project_root=PROJECT, archive_root=get_settings().fund_source_archive_root)
    if _original_only.get():
        return physical.read_bytes()
    from app.services.material_source_archive import archive_path, reviewed_fund_for

    # 仅显式数据库验收沿用旧别名约束；普通业务已由上方混合读取分支处理。
    strict_source = bool(original_location(path) or archive_path(path))
    if not managed(path):
        return physical.read_bytes()
    repo = repository()
    value = repo.get(path)
    if value is not None and not strict_source:
        # 共享路径可能不含基金代码。确定已归档的来源同样只读库，缺失的表示不能回读D/C原件。
        strict_source = bool(reviewed_fund_for(path, value.source_hash, repo))
    if (
        value is not None
        and value.kind == "UNSUPPORTED"
        and (path.suffix.lower() in {".md", ".markdown"} or known_legacy_word(value.source_hash))
    ):
        if _database_only.get() or strict_source:
            raise ValueError("MATERIAL_TEXT_NOT_EXTRACTED")
        value = repo.ingest(physical)
    if value is None and strict_source:
        raise ValueError("MATERIAL_DATABASE_MIGRATION_MISSING:" + str(path))
    if value is not None and strict_source and value.kind == "UNSUPPORTED":
        # 库内只有待提取登记时，不能把空字符串当正文交给业务，更不能回读原件掩盖缺口。
        raise ValueError("MATERIAL_TEXT_EXTRACTION_REQUIRED:" + str(value.issue))
    if value is None and not persistent_location(path):
        return physical.read_bytes()
    if value is None and _database_only.get():
        raise ValueError("MATERIAL_DATABASE_MIGRATION_MISSING:" + str(path))
    value = value if value is not None else repo.ingest(physical)
    value.database_only_source = strict_source
    if value.kind in {"PDF", "HTML"}:
        from app.services.material_important_text import release

        published = release(value.source_hash, repo)
        if published is not None:
            # 不修改可能已作为LRU键的旧对象；以相同底稿创建绑定新政策的读取视图。
            value = SourceContent(value.source_hash, value.source_size, value.kind, value.extracted,
                                  value.content_hash, value.complete, value.issue)
            value.database_only_source = strict_source
        value.important_release = published
    value.source_path = str(resolved(str(path)))
    operation = _operation.get()
    if operation is not None and original_location(path):
        operation[str(resolved(str(path)))] = value.source_hash
    return value


@contextmanager
def database_only():
    """验收时禁止从原件补提；未迁移的依赖必须显式失败。"""
    clear_calculation_caches()
    token = _database_only.set(True)
    try:
        yield
    finally:
        _database_only.reset(token)
        clear_calculation_caches()


@contextmanager
def original_materials():
    """迁移验收专用：以真实原件建立基准，不能用两次库内读取冒充原件对照。"""
    clear_calculation_caches()
    token = _original_only.set(True)
    try:
        yield
    finally:
        _original_only.reset(token)
        # 普通预读不能命中上一次从原件计算出来的旧缓存，否则会掩盖资料库漏读。
        clear_calculation_caches()


def clear_calculation_caches():
    """仅验收上下文隔离业务LRU；不重置连接池、配置或库内已核验的资料版本。"""
    import sys

    seen = set()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("app.services.") or name == __name__ or module is None:
            continue
        for value in tuple(vars(module).values()):
            origin = getattr(value, "__module__", "")
            if not isinstance(origin, str) or not origin.startswith("app.services.") or origin == __name__:
                continue
            clear = getattr(value, "cache_clear", None)
            if callable(clear) and id(value) not in seen:
                clear()
                seen.add(id(value))


def source_exists(path):
    path = Path(path)
    from app.core.config import get_settings
    from app.core.workspace_paths import runtime_material_path

    current = runtime_material_path(path, project_root=PROJECT, archive_root=get_settings().fund_source_archive_root)
    if _original_only.get():
        return current.is_file()
    if not _database_only.get() and current.is_file():
        return True
    if not _database_only.get() and not persistent_location(path):
        return False
    return (managed(path) and repository().metadata(path=path) is not None) or current.is_file()


def location_exists(path):
    """目录检查维持文件系统语义，历史资料文件也可由库内索引证明存在。"""
    path = Path(path)
    from app.core.config import get_settings
    from app.core.workspace_paths import runtime_material_path

    current = runtime_material_path(path, project_root=PROJECT, archive_root=get_settings().fund_source_archive_root)
    if _original_only.get():
        return current.exists()
    if not _database_only.get() and current.exists():
        return True
    if not _database_only.get() and not persistent_location(path):
        return False
    return (managed(path) and repository().metadata(path=path) is not None) or current.exists()


def source_glob(path, pattern, *, recursive=False):
    """合并当前目录和库内新旧别名，返回调用者根目录下的虚拟路径，不扫描旧工程。

    仅有旧数据库位置记录时也能在新目录发现资料；database_only验收不访问实体目录。
    新旧目录的同名条目合并后，正文读取仍由精确路径优先规则选择对应来源版本。
    """
    from app.core.config import get_settings
    from app.core.workspace_paths import runtime_material_path

    root = resolved(str(path))
    current = runtime_material_path(root, project_root=PROJECT, archive_root=get_settings().fund_source_archive_root)
    actual = () if _database_only.get() and not _original_only.get() else (
        current.rglob(pattern) if recursive else current.glob(pattern)
    )
    found = {str(root / p.relative_to(current)): root / p.relative_to(current) for p in actual}
    if _original_only.get():
        return iter(found.values())
    if managed(root):
        from sqlalchemy.exc import SQLAlchemyError

        try:
            repo = repository()
            roots = tuple(Path(p) for p in repo.location_candidates(root))
            with Session(repo.engine) as session:
                query = select(MaterialLocation.location).where(
                    or_(*(MaterialLocation.location.startswith(str(p) + os.sep, autoescape=True) for p in roots))
                )
                for name in session.scalars(query).yield_per(1000):
                    candidate = Path(name)
                    matched = next(p for p in roots if candidate.is_relative_to(p))
                    relative = candidate.relative_to(matched)
                    if relative.match(pattern) and (recursive or len(relative.parts) == 1 or "/" in pattern):
                        virtual = root / relative
                        found[str(virtual)] = virtual
        except SQLAlchemyError:
            if _database_only.get() or not found:
                raise
            logger.warning("material_content.source_glob   >>> 资料索引不可用，本次仅返回实际文件")
    return iter(found.values())


def source_size(path):
    from app.core.config import get_settings
    from app.core.workspace_paths import runtime_material_path

    current = runtime_material_path(path, project_root=PROJECT, archive_root=get_settings().fund_source_archive_root)
    if _original_only.get():
        return current.stat().st_size
    if not _database_only.get():
        try:
            return current.stat().st_size
        except FileNotFoundError:
            pass
    metadata = repository().metadata(path=path) if managed(path) else None
    if metadata is not None:
        return metadata.source_size
    from app.services.material_file_cleanup import physical_path

    return physical_path(current).stat().st_size


def source_text(path, encoding="utf-8", **kwargs):
    value = source_bytes(path)
    # 入库文本统一 UTF-8；来源编码只在首次提取时处理。
    body = value.decode("utf-8" if isinstance(value, SourceContent) else encoding, **kwargs)
    if isinstance(value, SourceContent) and value.kind == "HTML" and getattr(value, "important_release", None):
        from app.services.material_important_text import page_text

        extra = page_text(value.source_hash, 0, repository(), value.important_release)
        if extra:
            body += "\n" + extra
    return body


def exact_source_text(value):
    """需要原文字符位置的旧证据使用精确解码表示，不以重新排版HTML替换。

    历史提取只有语义DOM时，原件还在的普通预读阶段补存精确文字；严格库读
    阶段只接受已经保存的表示，不偷偷打开原件，也不修改原来的字符范围。
    """
    if isinstance(value, SourceContent):
        if value.kind != "HTML" or value.extracted.get("exact_html_text"):
            return value.decode()
        repo = repository()
        stored = repo.page(value.source_hash, "EXACT_HTML_TEXT_V1", 0)
        if stored is not None:
            return stored
        if _database_only.get() or getattr(value, "database_only_source", False):
            raise ValueError("MATERIAL_EXACT_HTML_TEXT_NOT_CAPTURED")
        path = getattr(value, "source_path", None)
        if path is None:
            raise ValueError("MATERIAL_EXACT_HTML_SOURCE_PATH_MISSING")
        from app.services.material_file_access import original_bytes

        raw = original_bytes(path, expected_hash=value.source_hash)
        if checksum(raw) != value.source_hash:
            raise ValueError("MATERIAL_EXACT_HTML_SOURCE_CHANGED")
    else:
        raw = value
    try:
        body = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        body = raw.decode("gb18030")
    if isinstance(value, SourceContent):
        return repo.page(value.source_hash, "EXACT_HTML_TEXT_V1", 0, body)
    return body


def docx_structure(value):
    """读取已持久化的Word正文及表格，兼容迁移早期只有段落的资料。"""
    if "tables" in value.extracted:
        return value.extracted
    repo = repository()
    stored = repo.page(value.source_hash, "DOCX_STRUCTURE_V1", 0)
    path = getattr(value, "source_path", None)
    if stored is None and path and not (_database_only.get() or getattr(value, "database_only_source", False)):
        from app.services.material_file_access import original_bytes

        raw = original_bytes(path, expected_hash=value.source_hash)
        if checksum(raw) != value.source_hash:
            raise ValueError("MATERIAL_DOCX_SOURCE_CHANGED")
        stored = repo.page(value.source_hash, "DOCX_STRUCTURE_V1", 0, json.dumps(extract_docx(raw), ensure_ascii=False))
    if stored is None:
        raise ValueError("MATERIAL_DOCX_STRUCTURE_NOT_CAPTURED")
    return json.loads(stored)


def archive_member(value, name):
    """压缩包成员通过资料编号读取提取信息，不重新打开或解压原ZIP。"""
    if value.kind != "ARCHIVE" or name not in value.extracted["members"]:
        raise ValueError("MATERIAL_ARCHIVE_MEMBER_MISSING")
    member = repository().get(source_hash=value.extracted["members"][name]["source_hash"])
    if member is None:
        raise ValueError("MATERIAL_ARCHIVE_MEMBER_NOT_STORED")
    member.source_archive_path = getattr(value, "source_path", None)
    member.source_archive_member = name
    return member


def register_content(path, value, *, replace=False):
    """转存历史引用只登记库内内容，不复制/重建已经删除的原件。"""
    if isinstance(value, SourceContent):
        old = repository().metadata(path=path)
        if old is not None and old.source_hash != value.source_hash and not replace:
            raise FileExistsError("MATERIAL_IMMUTABLE_LOCATION_CONFLICT")
        if value.kind == "DOCX":
            docx_structure(value)
        repository().put(path, value)
        return True
    return False


def persist_written(path, raw):
    """兼容旧保存调用：文件保存后不再自动解析入库、归档或删除原件。"""
    return None


def ensure_publishable(path, raw, *, replace=False):
    """保持不可变历史别名的冲突保护；只查身份索引，不提取或写入正文。"""
    if not replace and managed(path) and persistent_location(path):
        old = repository().metadata(path=path)
        if old is not None:
            if old.source_hash != content_digest(raw):
                raise FileExistsError("MATERIAL_IMMUTABLE_LOCATION_CONFLICT")
            raise FileExistsError("MATERIAL_IMMUTABLE_LOCATION_EXISTS")
    return True


def same_content(left, right):
    return content_digest(left) == content_digest(right)


def write_source(path, value):
    """原始字节写回调用方D盘目录；仅有历史库内表示时保留兼容引用，不重建伪原件。"""
    from app.services.material_file_access import physical_path

    if isinstance(value, (bytearray, memoryview)):
        value = bytes(value)
    if not register_content(path, value):
        from app.services.fund_exposure_common import _publish

        physical_path(path).parent.mkdir(parents=True, exist_ok=True)
        _publish(Path(path), value)


def material_operation(function):
    """兼容既有装饰器导入；业务操作不再触发全文补提、归档或结束清理。"""
    return function


def finalize_originals(used):
    """空间优化流程已撤回，保留签名兼容旧调用，不进行自动内容处理或清理。"""
    return None


def _finalize_originals(used):
    """旧内部入口同样停止，避免绕过公共兼容入口继续清理。"""
    return None


class StoredPage:
    def __init__(self, text, layout=None, document=None, index=0):
        self.text, self.layout = text, layout if layout is not None else text
        self.document, self.index = document, index

    def extract_text(self, *args, **kwargs):
        if self.document is not None:
            return self.document.page_text(self.index, layout=kwargs.get("extraction_mode") == "layout")
        return self.layout if kwargs.get("extraction_mode") == "layout" else self.text

    def get_textpage(self):
        return self

    def get_text_range(self, *args, **kwargs):
        return self.text

    def close(self):
        pass


class StoredDocument:
    def __init__(self, value, native=False):
        self.value, self.original_reader = value, None
        self.important_release = getattr(value, "important_release", None)
        self.important_page_cache = {}
        pages = value.extracted["pdfium"] if native else value.extracted.get("pypdf", value.extracted["pdfium"])
        layout = value.extracted.get("layout", pages)
        self.pages = [StoredPage(self.with_important_text(i, t), layout[i], None if native else self, i)
                      for i, t in enumerate(pages)]

    def with_important_text(self, index, body):
        """普通文字读取带上新核验的重要信息；历史排版表示保持原样，精确坐标不被追加内容改变。"""
        if self.important_release is None:
            return body
        from app.services.material_important_text import page_text

        if index not in self.important_page_cache:
            self.important_page_cache[index] = page_text(
                self.value.source_hash, index, repository(), self.important_release)
        extra = self.important_page_cache[index]
        return body + ("\n" + extra if extra else "")

    def _stored_metadata(self):
        """复用已有元数据；普通业务缺失时允许按需读取同版原件，显式读库验收不回读。"""
        raw = repository().page(self.value.source_hash, "PDF_METADATA_TEXT_V1", 0)
        if raw is None and not (_database_only.get() or getattr(self.value, "database_only_source", False)):
            from pypdf import PdfReader

            from app.services.material_file_access import original_bytes

            original = original_bytes(self.value.source_path, expected_hash=self.value.source_hash)
            reader = PdfReader(io.BytesIO(original))
            raw = encoded({"is_encrypted": reader.is_encrypted,
                           "metadata": {str(k): str(v) for k, v in (reader.metadata or {}).items()}}).decode()
            raw = repository().page(self.value.source_hash, "PDF_METADATA_TEXT_V1", 0, raw)
        if raw is None:
            raise ValueError("MATERIAL_PDF_METADATA_NOT_CAPTURED")
        return json.loads(raw)

    @property
    def metadata(self):
        return self._stored_metadata()["metadata"]

    @property
    def is_encrypted(self):
        return self._stored_metadata()["is_encrypted"]

    def get_metadata_dict(self):
        """兼容旧PDFium调用的八项字段；已确认原始Info字典缺项才返回空字符串。"""
        metadata = self.metadata
        return {name: metadata.get("/" + name, "") for name in (
            "Title", "Author", "Subject", "Keywords", "Creator", "Producer", "CreationDate", "ModDate",
        )}

    def page_text(self, index, *, layout=False):
        key = "layout" if layout else "pypdf"
        if key in self.value.extracted:
            body = self.value.extracted[key][index]
            return body if layout else self.with_important_text(index, body)
        engine = "PYPDF_LAYOUT_V1" if layout else "PYPDF_TEXT_V1"
        path = getattr(self.value, "source_path", None)
        transient = getattr(self.value, "transient_source", None)
        archive_path = getattr(self.value, "source_archive_path", None)
        if path is None and transient is None and archive_path is None and _override.get() is None:
            if layout:
                raise ValueError("MATERIAL_LAYOUT_REPRESENTATION_NOT_CAPTURED")
            return self.with_important_text(index, self.value.extracted["pdfium"][index])
        repo = repository()
        stored = repo.page(self.value.source_hash, engine, index)
        if stored is not None:
            return stored if layout else self.with_important_text(index, stored)
        path = getattr(self.value, "source_path", None)
        # 原逻辑路径可能已迁空，同版原件可由受控路由位于04-SourceFile；不以is_file提前屏蔽。
        has_original = transient is not None or path is not None or archive_path is not None
        if not (_database_only.get() or getattr(self.value, "database_only_source", False)) and has_original:
            if self.original_reader is None:
                from pypdf import PdfReader

                if transient is not None:
                    raw = transient
                elif archive_path:
                    import zipfile

                    with zipfile.ZipFile(archive_path) as archive:
                        raw = archive.read(self.value.source_archive_member)
                else:
                    from app.services.material_file_access import original_bytes

                    try:
                        raw = original_bytes(path, expected_hash=self.value.source_hash)
                    except FileNotFoundError:
                        if layout:
                            raise ValueError("MATERIAL_LAYOUT_REPRESENTATION_NOT_CAPTURED") from None
                        return self.with_important_text(index, self.value.extracted["pdfium"][index])
                if checksum(raw) != self.value.source_hash:
                    raise ValueError("MATERIAL_PAGE_SOURCE_CHANGED")
                self.original_reader = PdfReader(io.BytesIO(raw))
            page = self.original_reader.pages[index]
            # 合法空白页可能没有/Contents；复用已有兼容读取，不能因缺键阻断按需解析。
            result = pypdf_page_text(page, layout=layout)
            return repo.page(self.value.source_hash, engine, index, result or "")
        if layout:
            raise ValueError("MATERIAL_LAYOUT_REPRESENTATION_NOT_CAPTURED")
        return self.with_important_text(index, self.value.extracted["pdfium"][index])

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, i):
        return self.pages[i]

    def __iter__(self):
        return iter(self.pages)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.original_reader is not None:
            self.original_reader.close()
            self.original_reader = None
        self.value.transient_source = None


def document_reader(value, **kwargs):
    """原PDF直接交给原解析器；已删原件才使用历史页面，不在解析时自动写资料库。"""
    if isinstance(value, (str, Path)):
        value = source_bytes(value)
    if isinstance(value, SourceContent) and value.kind == "PDF":
        return StoredDocument(value)
    from pypdf import PdfReader

    return PdfReader(io.BytesIO(value) if isinstance(value, bytes) else value, **kwargs)


def document_stream(value):
    """仅兼容旧计算器的 with BytesIO 写法，不复制或重建 PDF。"""
    return value if isinstance(value, SourceContent) and value.kind == "PDF" else io.BytesIO(value)


def document_pdfium(value, **kwargs):
    if isinstance(value, (str, Path)):
        value = source_bytes(value)
    if isinstance(value, SourceContent) and value.kind == "PDF":
        return StoredDocument(value, native=True)
    import pypdfium2 as pdfium

    return pdfium.PdfDocument(value, **kwargs)


def source_pages(path, *, limit=None):
    from contextlib import closing

    from app.core.pdf import PDF_LOCK

    with PDF_LOCK, document_pdfium(path) as document:
        count = len(document) if limit is None else min(limit, len(document))
        result = []
        for i in range(count):
            with closing(document[i]) as page, closing(page.get_textpage()) as text:
                result.append(text.get_text_range())
        return result


def validate_completeness(content, path, *, repo, raw=None):
    """清理前复核历史提取的图像缺口；追加核验记录，不改写原有提取或评分。"""
    if not content.complete:
        raise ValueError("MATERIAL_DELETE_EXTRACTION_INCOMPLETE")
    if raw is not None and checksum(raw) != content.source_hash:
        raise ValueError("MATERIAL_DELETE_SOURCE_CHANGED")
    if content.kind == "DOCX":
        content.source_path = str(path)
        if raw is not None:
            structure = extract_docx(raw)
            repo.page(content.source_hash, "DOCX_STRUCTURE_V1", 0, json.dumps(structure, ensure_ascii=False))
        else:
            structure = docx_structure(content)
        if structure.get("unread_visuals"):
            raise ValueError("MATERIAL_DELETE_DOCX_VISUALS_INCOMPLETE")
    if content.kind == "PDF" and "completeness_policy" not in content.extracted:
        record = repo.page(content.source_hash, "PDF_COMPLETENESS_V2", 0)
        if record is None:
            from contextlib import closing

            import pypdfium2 as pdfium

            from app.core.pdf import PDF_LOCK

            raw = raw if raw is not None else Path(path).read_bytes()
            if checksum(raw) != content.source_hash:
                raise ValueError("MATERIAL_DELETE_SOURCE_CHANGED")
            issues = []
            with PDF_LOCK, pdfium.PdfDocument(raw) as doc:
                if len(doc) != len(content.extracted["pdfium"]):
                    raise ValueError("MATERIAL_PAGE_COUNT_MISMATCH")
                for index in range(len(doc)):
                    with closing(doc[index]) as page:
                        issue = pdf_page_issue(page, content.extracted["pdfium"][index])
                        if issue:
                            issues.append({"page": index + 1, "issue": issue})
            record = repo.page(
                content.source_hash,
                "PDF_COMPLETENESS_V2",
                0,
                json.dumps({"complete": not issues, "issues": issues}, sort_keys=True),
            )
        if not json.loads(record)["complete"]:
            raise ValueError("MATERIAL_DELETE_UNEXTRACTED_IMAGE_OR_TEXT")
    if content.kind == "ARCHIVE":
        import zipfile

        raw = raw if raw is not None else Path(path).read_bytes()
        if checksum(raw) != content.source_hash:
            raise ValueError("MATERIAL_DELETE_SOURCE_CHANGED")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for name, metadata in content.extracted["members"].items():
                member = repo.get(source_hash=metadata["source_hash"])
                if member is None:
                    raise ValueError("MATERIAL_ARCHIVE_MEMBER_NOT_STORED")
                validate_completeness(member, path, repo=repo, raw=archive.read(name))


def cleanup_on_hold():
    """删除默认暂停；原件提取质量尚无独立验收凭据，不能由评分回归自动放行。"""
    marker = PROJECT / "data/material-original-cleanup-hold.json"
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("hold") is not False
    except (OSError, ValueError, TypeError):
        return True


def delete_verified(path, *, root, verification, repo=None):
    """原子移入同目录暂存后核字节再删除，避免误删并发写入的新文件；支持中断恢复。"""
    from uuid import uuid4

    from app.services.material_file_access import legacy_workflow_error

    # 该入口属于本轮已撤回的空间优化，不再消费历史凭据执行任何改名或删除。
    legacy_workflow_error()

    repo = repo or repository()
    target, root = Path(path).resolve(), Path(root).resolve()
    if not verification or not target.is_relative_to(root) or target == root or Path(path).is_symlink():
        raise ValueError("MATERIAL_DELETE_SCOPE_OR_VERIFICATION")
    if root == PROJECT.resolve() and not original_location(target):
        raise ValueError("MATERIAL_DELETE_NOT_COLLECTION_DIRECTORY")
    if any(part in {"tests", "fixtures", ".git", ".venv", "node_modules"} for part in target.parts):
        raise ValueError("MATERIAL_DELETE_PROTECTED_PATH")
    if target.is_relative_to(PROJECT.resolve()) and cleanup_on_hold():
        raise ValueError("MATERIAL_DELETE_ON_USER_HOLD")
    content = repo.get(target, fresh=True)
    if content is None or not content.complete:
        raise ValueError("MATERIAL_DELETE_EXTRACTION_INCOMPLETE")
    if target.is_relative_to(PROJECT.resolve()):
        from app.services.material_document import get_review

        review = get_review(content.source_hash, repo=repo)
        if review is None or not review["ready"] or review["document_hash"] != verification:
            raise ValueError("MATERIAL_DELETE_CONTENT_REVIEW_REQUIRED")
    with Session(repo.engine) as session, session.begin():
        alias = session.scalar(
            select(MaterialLocation).where(MaterialLocation.location == str(target)).with_for_update()
        )
        if alias is None or alias.verification != verification:
            raise ValueError("MATERIAL_DELETE_NOT_BUSINESS_VERIFIED")
        if alias.source_hash != content.source_hash:
            raise ValueError("MATERIAL_DELETE_LOCATION_CHANGED")
        if target.is_relative_to(PROJECT.resolve()) and alias.retention_hold:
            raise ValueError("MATERIAL_DELETE_EXISTING_BATCH_HOLD")
        staged = None
        try:
            try:
                before = target.stat()
            except FileNotFoundError:
                # 上次进程可能在原子改名后退出。暂存名携带来源摘要，恢复仍逐字节核对。
                candidates = sorted(target.parent.glob(".material-delete-" + content.source_hash + "-*.tmp"))
                if not candidates:
                    alias.deleted_at = datetime.now(UTC)
                    return 0
                staged = candidates[0].resolve()
                before = staged.stat()
            physical = staged or target
            validate_completeness(content, physical, repo=repo)
            if staged is None:
                staged = target.parent / (".material-delete-" + content.source_hash + "-" + uuid4().hex + ".tmp")
                staged = staged.resolve()
                if not staged.is_relative_to(root) or staged.parent != target.parent:
                    raise ValueError("MATERIAL_DELETE_STAGE_SCOPE")
                target.rename(staged)
            if not staged.resolve().is_relative_to(root):
                raise ValueError("MATERIAL_DELETE_STAGE_SCOPE")
            actual, size = hashlib.sha256(), 0
            with staged.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    actual.update(block)
                    size += len(block)
            after = staged.stat()
            if actual.hexdigest() != content.source_hash or size != content.source_size:
                raise ValueError("MATERIAL_DELETE_SOURCE_CHANGED")
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("MATERIAL_DELETE_CONCURRENT_CHANGE")
            # 只删除已核验的暂存实体；原位置此时即使出现新文件，也绝不删除它。
            staged.unlink()
            if not target.exists():
                alias.deleted_at = datetime.now(UTC)
            return size
        except Exception:
            # 普通异常尽量恢复原名；新文件已占据原名时保留暂存件，绝不覆盖新数据。
            if staged is not None and staged.exists() and not target.exists():
                if staged.resolve().is_relative_to(root) and target.resolve().is_relative_to(root):
                    staged.rename(target)
            raise
