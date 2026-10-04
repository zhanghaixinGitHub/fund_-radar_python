"""补读固定官方目录的正文、TXT/PDF附件，保留旧提取和失败证据。"""

from __future__ import annotations

import concurrent.futures
import io as bytesio
import re
import threading
import zipfile
from collections import Counter
from urllib.parse import unquote, urljoin
from xml.etree import ElementTree

import pypdfium2 as pdfium
from bs4 import BeautifulSoup

from scripts import fund_002112_semantic_extract_v1 as sem
from scripts import fund_002112_semantic_public_v1 as public

io, ROOT = sem.io, sem.ROOT
PDF_LOCK = threading.Lock()


def decode(raw):
    """优先采用下载文件的UTF-8，旧政府TXT按GB18030读取，不能带替换符入训。"""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError("TEXT_ENCODING_UNVERIFIED")


def recover(path, folder="public-recovered"):
    original = io.read(path)
    target = ROOT / folder / path.name
    if target.exists():
        return io.read(target)
    row = original["entry"]
    result = {
        "id": original["id"],
        "entry": row,
        "admitted": False,
        "previous_path": str(path),
        "previous_sha256": io.sha(path),
        "assets": [],
    }
    try:
        raw, receipt = public.fetch(row["url"])
        assert receipt["status"] == 200, "PUBLIC_HTTP_NOT_200"
        soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
        meta = {str(x.get("name", "")).lower(): x.get("content", "") for x in soup.select("meta[name]")}
        title = meta.get("articletitle") or meta.get("title") or row["title"]
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True).split("_")[0]
        node = (
            soup.select_one("#zoom")
            or soup.select_one("#UCAP-CONTENT")
            or soup.select_one(".article-content")
            or soup.select_one(".ccontent")
            or soup.select_one(".content")
            or soup.select_one(".pages_content")
        )
        # 国务院客户端这个已固定URL的正文直接位于body；不把未知网站全页当正文。
        if node is None and row["url"] == public.TECH[3]:
            node = soup.body
        if node is None or not title:
            raise ValueError("ARTICLE_BODY_OR_TITLE_MISSING")
        stamp = next(
            (meta[k] for k in ("pubdate", "pubdateformat", "publishtime", "firstpublishedtime") if meta.get(k)), ""
        )
        if not stamp:
            # 只针对已知客户端布局，采用来源旁显示的发布日，绝不采用政策成文日期。
            text = soup.get_text(" ", strip=True)
            pattern = (
                r"工业和信息化部网站\s+(20\d{2}-\d{2}-\d{2})"
                if row["url"] == public.TECH[3]
                else r"(?:发布时间|发布日期)\s*[：:]?\s*(20\d{2}[-年/]\d{1,2}[-月/]\d{1,2})"
            )
            match = re.search(pattern, text)
            stamp = match[1] if match else ""
        parts = re.match(r"(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})", stamp)
        if not parts:
            raise ValueError("ARTICLE_PUBLICATION_DATE_UNVERIFIED")
        day = "-".join((parts[1], parts[2].zfill(2), parts[3].zfill(2)))
        if not "2024-01-01" <= day <= "2026-09-29":
            raise ValueError("ARTICLE_OUTSIDE_RANGE")
        dates = [x["display_date"] for x in row["aliases"] if x["display_date"]]
        available_day = max([day, min(dates)] if dates else [day])
        modified = meta.get("lastmodifiedtime", "")[:10]
        if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", modified):
            available_day = max(available_day, modified)
        for element in node.select("script,style"):
            element.decompose()
        sections = [node.get_text("\n", strip=True)]
        links = {}
        for a in node.select("a[href]"):
            url = urljoin(row["url"], a["href"])
            if re.search(r"\.(txt|pdf|docx?|xlsx?|zip)(?:$|[?&#])", unquote(url), re.I):
                links[url] = a.get_text(" ", strip=True)
        for url, label in links.items():
            asset = {"url": url, "label": label, "admitted": False}
            try:
                content, proof = public.fetch(url)
                asset["receipt"] = proof
                if proof["status"] != 200:
                    raise ValueError("ATTACHMENT_HTTP_NOT_200")
                if re.search(r"\.txt(?:$|[?&#])", unquote(url), re.I):
                    text = decode(content)
                elif re.search(r"\.pdf(?:$|[?&#])", unquote(url), re.I):
                    # PDFium不是线程安全的；网络可并行，解析必须串行并显式关闭页面。
                    with PDF_LOCK, pdfium.PdfDocument(content) as pdf:
                        pages = []
                        for number in range(len(pdf)):
                            page = pdf[number]
                            textpage = page.get_textpage()
                            try:
                                pages.append(textpage.get_text_range())
                            finally:
                                textpage.close()
                                page.close()
                        text = "\n".join(pages)
                        metadata = pdf.get_metadata_dict()
                    asset["pdf_metadata"] = metadata
                    for name in ("CreationDate", "ModDate"):
                        m = re.match(r"(?:D:)?(20\d{2})(\d{2})(\d{2})", metadata.get(name, ""))
                        if m:
                            available_day = max(available_day, "-".join(m.groups()))
                elif re.search(r"\.docx(?:$|[?&#])", unquote(url), re.I):
                    with zipfile.ZipFile(bytesio.BytesIO(content)) as archive:
                        tree = ElementTree.fromstring(archive.read("word/document.xml"))
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    text = "\n".join("".join(p.itertext()) for p in tree.findall(".//w:p", ns))
                else:
                    raise ValueError("NON_TEXT_ATTACHMENT_NOT_PARSED")
                if len(text.strip()) < 100 or "<html" in text[:200].lower():
                    raise ValueError("ATTACHMENT_NO_VERIFIED_TEXT")
                asset.update({"admitted": True, "characters": len(text), "text_sha256": io.digest(text)})
                # 来源标签和正文明确分段，禁止模型跨附件拼接数值和期间。
                sections.append("【附件正文开始】\n" + text + "\n【附件正文结束】")
            except Exception as exc:
                asset["reason"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            result["assets"].append(asset)
        body = "\n【省略】\n".join(sections)
        if len(body) < 100:
            raise ValueError("IMAGE_OR_SHORT_BODY_WITHOUT_VERIFIED_TEXT")
        result.update(
            {
                "admitted": True,
                "kind": row["kind"],
                "title": title,
                "published_date": day,
                "source_url": row["url"],
                "topic": row["topic"],
                "title_available_at": sem.core.previous.next_midnight(available_day),
                "body_available_at": sem.core.previous.next_midnight(available_day),
                "body": body,
                "body_sha256": io.digest(body),
                "receipt": receipt,
                "unread_images": len(node.select("img")),
                "historical_first_seen_proven": False,
                "scope": "网页及可验证文本附件；图片和不支持附件不编造正文",
            }
        )
    except Exception as exc:
        result["reason"] = str(exc) if isinstance(exc, (ValueError, AssertionError)) else type(exc).__name__
    io.save(target, result)
    return result


def run():
    docs = [(p, io.read(p)) for p in sorted((ROOT / "public-documents").glob("*.json"))]
    retry = (ROOT / "recovery-completion.json").exists()
    paths = [p for p, d in docs if d["entry"]["kind"] == "policy" or not d["admitted"]]
    if retry:
        paths = [
            p
            for p in paths
            if any(
                a.get("reason") in ("TypeError", "NON_TEXT_ATTACHMENT_NOT_PARSED")
                or a.get("reason", "").startswith("IMMUTABLE_CONFLICT")
                for a in io.read(ROOT / "public-recovered" / p.name)["assets"]
            )
        ]
    suffix = "-retry" if retry else ""
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        recovered = list(pool.map(lambda p: recover(p, "public-recovered" + suffix), paths))
    previous_ids = {d["id"]: d for d in sem.prepare()}
    extras = []
    for d in recovered:
        if not d["admitted"] or d["body_available_at"] > "2026-09-29T08:00:00+08:00":
            continue
        if d["id"] in previous_ids and d["body_sha256"] == previous_ids[d["id"]]["body_sha256"]:
            continue
        text, selection = sem.selected_text(d)
        extras.append(
            {
                **d,
                "event_id": io.digest([d["body_sha256"], []]),
                "aliases": [d["id"]],
                "issuer_codes": [],
                "type": "POLICY" if d["kind"] == "policy" else "NEWS",
                "text": text,
                "text_sha256": io.digest(text),
                "text_level": "BODY",
                "selection": selection,
            }
        )
    io.save_lines(ROOT / ("supplement" + suffix + "-documents.jsonl"), extras)
    result = {
        "at": io.now(),
        "attempted": len(paths),
        "admitted": sum(d["admitted"] for d in recovered),
        "new_or_revised_extractions": len(extras),
        "asset_success": sum(a["admitted"] for d in recovered for a in d["assets"]),
        "asset_failures": dict(Counter(a["reason"] for d in recovered for a in d["assets"] if not a["admitted"])),
        "unavailable": dict(Counter(d["reason"] for d in recovered if not d["admitted"])),
    }
    io.save(ROOT / ("recovery" + suffix + "-completion.json"), result)
    print(io.canonical(result), flush=True)
    # 扩充内容沿用已经审阅的提示词及同一请求账本，不扩大模型训练搜索。
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(sem.extract, extras))
    io.save(
        ROOT / ("supplement" + suffix + "-completion.json"),
        {"at": io.now(), "documents": len(extras), "status": dict(Counter(r["status"] for r in results))},
    )


if __name__ == "__main__":
    run()
