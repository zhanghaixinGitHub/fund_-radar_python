"""补本轮相关新闻政策正文；固定目录与关键词，不按涨跌结果挑新闻。"""

from __future__ import annotations

import concurrent.futures
import hashlib
import re
import threading
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from scripts import fund_002112_semantic_upgrade_v1 as core

io, ROOT = core.io, core.ROOT
LOCK = threading.Lock()
HEALTH = r"药|器械|耗材|医保目录|医疗服务价格|集采|集中采购|支付|创新"
TECH = [
    "https://www.gov.cn/zhengce/content/202508/content_7037861.htm",
    "https://www.miit.gov.cn/zwgk/zcwj/wjfb/tz/art/2024/art_b0927b0d4cba4ff89652bb64f74899d1.html",
    "https://www.miit.gov.cn/jgsj/txs/wlfz/art/2024/art_647ad43f4a474e2b832b31099e9f36a6.html",
    "https://app.www.gov.cn/govdata/gov/202505/28/529970/article.html",
    "https://gdca.miit.gov.cn/zwgk/zcwj/wjfb/art/2024/art_8d6908d847994bc4a2f0c9eb3517e780.html",
    "https://ahca.miit.gov.cn/xxgk/zfxxgkzl/fdzdgknr/art/2024/art_915cff99961249b797f79ca718a3ca75.html",
    "https://cqca.miit.gov.cn/zwgk/zcwj/wjfb/art/2024/art_ac147ad753b24997bd31f84fe52b94fd.html",
    "https://zjca.miit.gov.cn/zwgk/zcwj/wjfb/art/2024/art_a4b448ac56d14d69b49e86d30ca5beb5.html",
    "https://jsca.miit.gov.cn/zwgk/zcwj/wjfb/art/2024/art_8bbbcc1d7ce04d4a8810c47424bb285c.html",
    "https://gdca.miit.gov.cn/zwgk/tzgg/art/2024/art_589acdbd6f864f02a8047ff24761259a.html",
]


def inventory():
    path = ROOT / "public-inventory.json"
    if path.exists():
        return io.read(path)
    closure = io.RESEARCH / "closure/20260929-v1"
    sources = [closure / "public-catalogs/result.json", closure / "public-news-current-snapshot/result.json"]
    catalogs = io.read(sources[0])["columns"]
    rows = []
    for cat in catalogs:
        if cat["column"] not in (104, 46):
            continue
        rows += [{**r, "kind": "policy" if cat["column"] == 104 else "news", "topic": "HEALTH"}
                 for r in cat["all_entries"] if "2024" <= r["display_date"] <= "2026-09-29"]
    rows += [{**r, "kind": "news", "topic": "HEALTH"}
             for r in io.read(sources[1])["all_entries"]
             if "2024" <= r["display_date"] <= "2026-09-29" and re.search(HEALTH, r["title"])]
    rows += [{"url": u, "title": None, "display_date": None, "kind": "policy", "topic": "TECH"} for u in TECH]
    unique = {}
    for row in rows:
        key = row["url"].replace("http://www.nhsa.gov.cn/", "https://www.nhsa.gov.cn/")
        if key not in unique:
            unique[key] = {**row, "url": key, "aliases": []}
        unique[key]["aliases"].append({"title": row["title"], "display_date": row["display_date"]})
    result = {"at": io.now(), "entries": sorted(unique.values(), key=lambda x: x["url"]),
              "catalog_hashes": {str(p): io.sha(p) for p in sources}, "title_filter": HEALTH,
              "coverage": "固定目录相关条目及固定官方技术政策；不声称覆盖全网"}
    io.save(path, result)
    return result


def fetch(url):
    """只抓取公开HTTPS网页，原始响应与失败留档；不更改原缓存。"""
    host = urlparse(url).hostname or ""
    if not url.startswith("https://") or not (host.endswith(".gov.cn") or host.endswith(".nhsa.gov.cn")):
        raise ValueError("SOURCE_REQUIRES_SEPARATE_ADAPTER")
    key = io.digest(url)
    receipt_path = ROOT / "public-receipts" / (key + ".json")
    if receipt_path.exists():
        receipt = io.read(receipt_path)
        raw = (ROOT / receipt["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        return raw, receipt
    with LOCK:
        count = sum(1 for _ in (ROOT / "public-requests").glob("*.json"))
        if count >= core.plan()["public_requests_max"]:
            raise ValueError("PUBLIC_REQUEST_BUDGET_REACHED")
        request_path = ROOT / "public-requests" / (key + ".json")
        if request_path.exists():
            request_path = ROOT / "public-requests" / (key + "-retry2.json")
            if request_path.exists():
                raise ValueError("PUBLIC_TWO_ATTEMPTS_EXHAUSTED")
        io.save(request_path, {"url": url, "at": io.now()})
    with httpx.Client(timeout=25, trust_env=False, follow_redirects=False) as client:
        response = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = response.content
    digest = hashlib.sha256(raw).hexdigest()
    dest = ROOT / "public-raw" / (digest + ".html")
    with LOCK:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as f:
                f.write(raw)
    receipt = {"at": io.now(), "url": url, "status": response.status_code,
               "sha256": digest, "file": str(dest.relative_to(ROOT))}
    io.save(receipt_path, receipt)
    return raw, receipt


def collect(row):
    key = io.digest(row["url"])
    path = ROOT / "public-documents" / (key + ".json")
    if path.exists():
        return io.read(path)
    result = {"id": "public-" + key, "entry": row, "admitted": False}
    try:
        raw, receipt = fetch(row["url"])
        result["receipt"] = receipt
        if receipt["status"] != 200:
            raise ValueError("PUBLIC_HTTP_" + str(receipt["status"]))
        soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
        meta = {str(x.get("name", "")).lower(): x.get("content", "") for x in soup.select("meta[name]")}
        title = meta.get("articletitle") or meta.get("title")
        if not title:
            node = soup.select_one("h1") or soup.select_one(".title")
            title = node.get_text(" ", strip=True) if node else row["title"]
        content = (soup.select_one("#zoom") or soup.select_one("#UCAP-CONTENT")
                   or soup.select_one(".article-content") or soup.select_one(".ccontent")
                   or soup.select_one(".content") or soup.select_one(".pages_content"))
        if not content or not title:
            raise ValueError("ARTICLE_BODY_OR_TITLE_MISSING")
        stamp = meta.get("pubdate") or meta.get("pubdateformat") or meta.get("publishtime")
        if not stamp:
            match = re.search(r"(?:发布时间|发布日期|来源[：:]).{0,40}?(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})",
                              soup.get_text(" ", strip=True))
            stamp = "-".join((match[1], match[2].zfill(2), match[3].zfill(2))) if match else None
        if not stamp or not re.match(r"20\d{2}-\d{2}-\d{2}", stamp):
            raise ValueError("ARTICLE_PUBLICATION_DATE_UNVERIFIED")
        day = stamp[:10]
        if not "2024-01-01" <= day <= "2026-09-29":
            raise ValueError("ARTICLE_OUTSIDE_RANGE")
        text = content.get_text("\n", strip=True)
        if len(text) < 100:
            raise ValueError("ARTICLE_BODY_TOO_SHORT")
        # 转载列表日期可能比原站晚，采用较晚日期；目录别名不会重复产生事件。
        dates = [x["display_date"] for x in row["aliases"] if x["display_date"]]
        available_day = max([day, min(dates)] if dates else [day])
        attachments = [urljoin(row["url"], a["href"]) for a in content.select("a[href]")
                       if re.search(r"\.(pdf|docx?|xlsx?|zip)(?:$|\?)", a["href"], re.I)]
        result.update({"admitted": True, "kind": row["kind"], "title": title,
                       "published_date": day, "source_url": row["url"], "topic": row["topic"],
                       "title_available_at": core.previous.next_midnight(available_day),
                       "body_available_at": core.previous.next_midnight(available_day),
                       "body": text, "body_sha256": io.digest(text), "attachments": attachments,
                       "scope": "页面正文；未读取附件的内容不会当成已提取事实",
                       "historical_first_seen_proven": False})
    except Exception as exc:
        result["reason"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
    io.save(path, result)
    return result


def run():
    core.plan()
    rows = inventory()["entries"]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i, result in enumerate(pool.map(collect, rows), 1):
            results.append(result)
            if i % 50 == 0:
                print(f"公开资料 {i}/{len(rows)}，正文通过 {sum(r['admitted'] for r in results)}", flush=True)
    from collections import Counter
    summary = {"at": io.now(), "catalog_entries": len(rows), "admitted": sum(r["admitted"] for r in results),
               "unavailable": dict(Counter(r["reason"] for r in results if not r["admitted"])),
               "year_kind": dict(Counter(r["published_date"][:4] + "_" + r["kind"]
                                         for r in results if r["admitted"]))}
    io.save(ROOT / "public-completion.json", summary)
    print(io.canonical(summary), flush=True)


if __name__ == "__main__":
    run()
