"""为已确认无法公开读取的微信外链补充公开转载；保留原链接，不冒充微信原文。"""

import re

import httpx
from bs4 import BeautifulSoup

from app.integrations.dbfund_reports import bounded_get
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, blob, now, read, save
from app.services.fund_exposure_supplement import SUPPLEMENT, verified_bytes

# 每个地址都由本次标题检索和页面核对得到；这是有限补缺清单，不是任意网址采集入口。
MIRRORS = {
    "f3316d35ddd342f1a4d3737464d1c6b4": (
        "https://finance.sina.com.cn/money/fund/jjgsgd/2026-06-02/doc-inhzzumi1443185.shtml",
        "#artibody",
        "以成长为锚",
        "SINA_PUBLIC_REPUBLICATION",
    ),
    "545dd9eca0164190a7d61d287e12a716": (
        "https://www.cs.com.cn/ssgs/gsxl/202603/t20260313_6541151.html",
        ".Custom_UnionStyle",
        "擘画高质量发展新蓝图",
        "CHINA_SECURITIES_JOURNAL_PUBLIC",
    ),
    "3a6ee6991e8d4b779d9ae78be44f0e18": (
        "https://app-web.chnfund.com/YHHEediting/202501/t20250122_4440011.html",
        ".RdsText",
        "德邦基金荣获上海金融职工立功竞赛创新三等奖",
        "CHNFUND_PUBLIC_REPUBLICATION",
    ),
    "9db490ecd52e4e51a4616515f4c58fe2": (
        "https://app-web.chnfund.com/jx/202409/t20240929_4400870.html",
        ".RdsText",
        "拥抱新质生产力，同心共绘新时代",
        "CHNFUND_PUBLIC_REPUBLICATION",
    ),
    "7579d97fd6bb43d5994b2b5ab4f0f29e": (
        "https://app-web.chnfund.com/YHHEediting/202409/t20240911_4395434.html",
        ".RdsText",
        "德邦基金投教讲师团官宣",
        "CHNFUND_PUBLIC_REPUBLICATION",
    ),
    "a4f0daf9ef86481caf5f2856cdba3a06": (
        "https://app-web.chnfund.com/YHHEediting/202410/t20241009_4402871.html",
        ".RdsText",
        "推动金融高质量发展行稳致远",
        "CHNFUND_PUBLIC_REPUBLICATION",
    ),
    "f242c461213340c282e82aa9cebd7d30": (
        "https://finance.sina.com.cn/money/fund/jjzl/2024-08-28/doc-incmehfs3738435.shtml",
        "#artibody",
        "上海对外经贸大学师生走进德邦基金",
        "SINA_PUBLIC_REPUBLICATION",
    ),
    "40b5f9f740eb4eaf8beab1f62dc27578": (
        "https://www.samacn.org.cn/d_3006_86132.html",
        ".detail",
        "德邦基金党支部参与上海基金行业党纪学习教育活动",
        "SAMACN_PUBLIC_REPUBLICATION",
    ),
}


def mirror_text(raw, selector, title_fragment):
    """核对文章标题并仅提取正文，避免把侧栏实时新闻当成这篇历史文章。"""
    soup = BeautifulSoup(raw, "html.parser")
    titles = " ".join(n.get_text(" ", strip=True) for n in soup.select("title,h1,h2,h3,.utit"))
    if re.sub(r"\s+", "", title_fragment) not in re.sub(r"\s+", "", titles):
        raise ValueError("EXPOSURE_NEWS_MIRROR_TITLE_MISMATCH")
    body = soup.select_one(selector)
    if body is None:
        raise ValueError("EXPOSURE_NEWS_MIRROR_BODY_MISSING")
    for element in body.select("script,style,iframe"):
        element.decompose()
    content = body.get_text("\n", strip=True)
    if len(content) < 200 or "德邦" not in content:
        raise ValueError("EXPOSURE_NEWS_MIRROR_BODY_INVALID")
    return content


def recover_news_mirrors():
    """为八条已核对的文章补存转载；首次公布时间、原版一致性仍不作未经证明的保证。"""
    items = read(SUPPLEMENT / "public-catalog.json")["items"]
    public = read(SUPPLEMENT / "public-result.json")
    entries = {item["content_id"]: item for item in public["documents"]}
    result = {"at": now().isoformat(), "recovered": [], "errors": [], "original_wechat_access_unchanged": True}
    with httpx.Client(timeout=httpx.Timeout(30, connect=5), headers={"User-Agent": "Mozilla/5.0"}) as client:
        for key, (url, selector, title_fragment, source) in MIRRORS.items():
            try:
                path = SUPPLEMENT / "documents" / (digest(key) + ".json")
                if path.exists():
                    document = read(path)
                    verified_bytes(document["receipt"])
                else:
                    raw = bounded_get(client, url, 4_000_000)
                    content = mirror_text(raw, selector, title_fragment)
                    sha, filename = blob(raw, "html")
                    document = {
                        **items[key],
                        "url": url,
                        "receipt": {
                            "url": url,
                            "sha256": sha,
                            "file": filename,
                            "received_at": now().isoformat(),
                            "source_code": source,
                        },
                        "pages": [content],
                        "page_count": 1,
                        "text_status": "TEXT_EXTRACTED",
                        "relevance": "FUND_MANAGER_CONTEXT" if key.startswith("f3316") else "COMPANY_CONTEXT",
                        "original_official_url": items[key]["catalog"]["url"],
                        "is_official_site_copy": False,
                        "original_text_equivalence_verified": False,
                        "historical_first_seen_verified": False,
                        "training_eligible": False,
                    }
                    save(path, document)
                entries[key] = {
                    "content_id": key,
                    "title": items[key]["catalog"]["title"],
                    "file": path.relative_to(ROOT).as_posix(),
                    "sha256": document["receipt"]["sha256"],
                    "text_status": document["text_status"],
                    "relevance": document["relevance"],
                    "scopes": document["scopes"],
                }
                result["recovered"].append({"content_id": key, "url": url, "receipt": document["receipt"]})
            except Exception as exc:
                result["errors"].append(
                    {"content_id": key, "reason": str(exc) if str(exc).startswith("EXPOSURE_") else type(exc).__name__}
                )
    public["documents"] = list(entries.values())
    public["news_mirrors_updated_at"] = now().isoformat()
    save(SUPPLEMENT / "public-result.json", public, replace=True)
    save(SUPPLEMENT / "news-mirror-result.json", result, replace=True)
    return {"recovered": len(result["recovered"]), "errors": result["errors"], "public_documents": len(entries)}
