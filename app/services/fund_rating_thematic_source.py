"""完整主题包的受限原件读取；只读显式安全根、校验原字节并保留来源种类。"""

import io
import json
import re
import zipfile
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from app.services.material_content import content_digest, source_bytes, source_exists


def require(ok, code):
    if not ok:
        raise ValueError("R5_THEME_RAW_" + code)


def iso(value):
    require(isinstance(value, str) and re.fullmatch(r"(?:\d{8}|\d{4}-\d{2}-\d{2})", value), "DATE_SCHEMA")
    return date.fromisoformat(value).isoformat()


def plain(text):
    from app.services.fund_rating_market_review import normalized_text

    # PDF字形F052与勾选符仅做相同“勾”的格式归一；未勾选方框不改变。
    return normalized_text(text.replace("\uf052", "√").replace("☑", "√"))


class RawArchive:
    """每轮校字节；body只处理明确格式，研究底稿与外部原件不能混为一类。"""

    def __init__(self, inputs, root):
        self.root = Path(root).resolve()
        self.sources = inputs["sources"]
        self.asof, self.research = iso(inputs["as_of_date"]), iso(inputs["research_on"])
        require(self.asof <= self.research, "RESEARCH_DATE")
        require(1 <= len(self.sources) <= 1800, "SOURCE_COUNT")
        self.raw, self.cache, self.used = {}, {}, set()
        self.pages_cache = {}
        self.json_cache = {}
        for sha, ref in self.sources.items():
            path = (self.root / ref["file"]).resolve()
            require(path.is_relative_to(self.root / "sources") and source_exists(path), "SOURCE_PATH")
            data = source_bytes(path)
            require(len(data) <= 50000000 and content_digest(data) == sha == ref["sha256"], "SOURCE_HASH")
            require(ref.get("archive_kind") in {"ORIGINAL", "DERIVED"}, "SOURCE_KIND")
            if ref.get("published_on"):
                boundary = self.research if ref.get("economic_role") == "HISTORICAL_PRICE" else self.asof
                require(iso(ref["published_on"]) <= boundary, "SOURCE_AFTER_KNOWLEDGE_BOUNDARY")
            self.raw[sha] = data

    def json(self, sha):
        require(sha in self.raw, "SOURCE_NOT_ARCHIVED")
        self.used.add(sha)
        # 保留外部JSON精确十进制；需要JSON序列化时在结果边界转成文本。
        from decimal import Decimal

        if sha not in self.json_cache:
            self.json_cache[sha] = json.loads(self.raw[sha], parse_float=Decimal)
        return self.json_cache[sha]

    def body(self, sha):
        require(sha in self.raw, "SOURCE_NOT_ARCHIVED")
        self.used.add(sha)
        if sha in self.cache:
            return self.cache[sha]
        ref, data = self.sources[sha], self.raw[sha]
        kind = ref["format"]
        if kind == "pdf":
            text = "\n".join(self.pages(sha))
        elif kind == "docx":
            from app.services.material_content import SourceContent

            if isinstance(data, SourceContent) and data.kind == "DOCX":
                self.cache[sha] = data.decode("utf-8")
                return self.cache[sha]
            with zipfile.ZipFile(io.BytesIO(data)) as document:
                info = document.getinfo("word/document.xml")
                require(info.file_size <= 25_000_000, "DOCX_BODY_SIZE")
                tree = ElementTree.fromstring(document.read(info))
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            text = "\n".join("".join(p.itertext()) for p in tree.findall(".//w:p", ns))
        elif kind == "json":
            obj = self.json(sha)
            require(isinstance(obj.get("data", {}).get("notice_content"), str), "NOTICE_BODY_REQUIRED")
            text = BeautifulSoup(obj["data"]["notice_content"], "html.parser").get_text("\n", strip=True)
        elif kind in {"html", "text", "javascript"}:
            encoding = ref.get("encoding", "utf-8")
            require(encoding in {"utf-8", "utf-8-sig", "gb18030"}, "SOURCE_ENCODING")
            text = data.decode(encoding, errors="strict")
            if kind == "html":
                soup = BeautifulSoup(text, "html.parser")
                body = soup.select_one("#detail-body") or soup
                # 虹点原件的公告标题/公开日与正文是同级节点；双栏正文可能
                # 字符交错，保留真实标题，不以网站通用title或目录标题补正文。
                heading = soup.select_one(".detail-c > .title-c") if body is not soup else None
                text = ((heading.get_text("\n", strip=True) + "\n") if heading else "") + body.get_text(
                    "\n", strip=True
                )
        else:
            raise ValueError("R5_THEME_RAW_UNSUPPORTED_FORMAT")
        require(len(text) >= 50, "SOURCE_BODY_INCOMPLETE")
        self.cache[sha] = text
        return text

    def pages(self, sha):
        self.used.add(sha)
        if sha not in self.pages_cache:
            if self.sources[sha]["format"] == "pdf":
                from app.services.fund_rating_pdf import text_pages

                self.pages_cache[sha] = text_pages(self.root / self.sources[sha]["file"])
            else:
                self.pages_cache[sha] = [self.body(sha)]
        return self.pages_cache[sha]

    def plain(self, sha):
        return plain(self.body(sha))

    def claim(self, item):
        sha, quote = item["source_hash"], plain(item["quote"])
        require(
            self.sources[sha]["archive_kind"] == "ORIGINAL" and quote and quote in plain(self.body(sha)),
            "ORIGINAL_QUOTE",
        )
        if item.get("published_on"):
            require(iso(item["published_on"]) == iso(self.sources[sha]["published_on"]), "QUOTE_PUBLICATION_BINDING")

    def table(self, sha, *, symbol=None, allow_more=False):
        obj = self.json(sha)
        require(
            type(obj.get("code")) is int and obj["code"] == 0 and isinstance(obj.get("data"), dict), "PROVIDER_RESPONSE"
        )
        data = obj["data"]
        fields, items = data.get("fields"), data.get("items")
        require(
            isinstance(fields, list) and len(fields) == len(set(fields)) and isinstance(items, list), "PROVIDER_SCHEMA"
        )
        require(
            type(data.get("has_more")) is bool and (allow_more or data["has_more"] is False),
            "PROVIDER_PAGE_NOT_CLOSED",
        )
        rows = []
        for item in items:
            require(len(item) == len(fields), "PROVIDER_ROW_WIDTH")
            row = dict(zip(fields, item, strict=True))
            if symbol:
                require(row.get("ts_code") == symbol, "PROVIDER_WRONG_SECURITY")
            rows.append(row)
        return rows

    def original(self, sha):
        require(sha in self.sources and self.sources[sha]["archive_kind"] == "ORIGINAL", "ORIGINAL_REQUIRED")
        self.used.add(sha)
        return self.sources[sha]

    def provider_rows(self, sha, fields, *, allow_more=False):
        """同一个严格响应入口适配固定市场目录，不省略完整页和字段检查。"""
        self.original(sha)
        obj = self.json(sha)
        require(set(fields) <= set(obj.get("data", {}).get("fields", [])), "PROVIDER_REQUIRED_FIELDS")
        return self.table(sha, allow_more=allow_more)
