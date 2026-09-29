"""按国家医保局公开页面原有分页参数核目录；保留正文日期和附件缺口。"""

import re
from collections import Counter
from datetime import datetime
from urllib.parse import urljoin

from app.services.fund_information_history_v1 import OUT, ZONE, BoundedPublicReader, digest, read, save
from bs4 import BeautifulSoup

ORIGIN = "https://www.nhsa.gov.cn"
PROXY = ORIGIN + "/module/web/jpage/dataproxy.jsp"
PUBLIC = OUT / "public-history-v2"


def append_group(all_rows, rows, group, total):
    """官网带一条前瞻记录；只接纳逐页精确匹配的单条边界重叠，不任意去重。"""
    expected = min(45 if group == 0 else 46, total - group * 45)
    if len(rows) != expected or len({r["url"] for r in rows}) != len(rows):
        raise ValueError("PUBLIC_CATALOG_PAGE_SIZE_OR_UNIQUENESS_MISMATCH")
    if group >= 2:
        if not all_rows or rows[0] != all_rows[-1]:
            raise ValueError("PUBLIC_CATALOG_BOUNDARY_NOT_IDENTICAL")
        rows = rows[1:]
    if set(r["url"] for r in rows) & set(r["url"] for r in all_rows):
        raise ValueError("PUBLIC_CATALOG_DUPLICATE_OUTSIDE_BOUNDARY")
    all_rows.extend(rows)


def config(html):
    """只解析官网声明的普通分页参数，不运行网页脚本。"""
    match = re.search(r"var param_\d+ = \{([^}]+)\}", html)
    if not match:
        raise ValueError("COLUMN_PAGINATION_CONFIG_MISSING")
    params = {m[1]: m[2] if m[2] is not None else m[3] for m in re.finditer(r"(\w+):(?:'([^']*)'|(\d+))", match[1])}
    totals = re.findall(r"totalRecord:(\d+)", html)
    if not totals:
        raise ValueError("COLUMN_TOTAL_MISSING")
    return params, int(totals[-1])


def records(raw):
    """CDATА 内的标题、显示日期与文章 URL 一起保存，不用 URL 日期冒充发布日期。"""
    entries = []
    for fragment in re.findall(r"<record><!\[CDATA\[(.*?)\]\]></record>", raw, re.S):
        soup = BeautifulSoup(fragment, "html.parser")
        link = soup.find("a", href=True)
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", soup.get_text(" ", strip=True))
        if not link or not dates:
            raise ValueError("COLUMN_ENTRY_WITHOUT_URL_OR_DISPLAY_DATE")
        url = urljoin(ORIGIN, link["href"]).replace("http://www.nhsa.gov.cn/", ORIGIN + "/")
        entries.append({"url": url, "title": link.get("title") or link.get_text(strip=True), "display_date": dates[-1]})
    return entries


def collect_column(client, column, max_groups):
    raw, receipt = client.fetch(f"{ORIGIN}/col/col{column}/index.html", group="public")
    html = raw.decode("utf-8")
    params, total = config(html)
    all_rows, receipts, issues = [], [receipt], []
    groups = (total + 44) // 45
    for group in range(min(groups, max_groups)):
        try:
            if group == 0:
                rows = records(html)
            else:
                url = f"{PROXY}?startrecord={group * 45 + 1}&endrecord={(group + 1) * 45}&perpage=15"
                data, entry = client.fetch(url, params=params, group="public")
                receipts.append(entry)
                text = data.decode("utf-8")
                found = re.search(r"<totalrecord>(\d+)</totalrecord>", text)
                if not found or int(found[1]) != total:
                    raise ValueError("PUBLIC_CATALOG_TOTAL_CHANGED")
                rows = records(text)
            append_group(all_rows, rows, group, total)
        except ValueError as exc:
            issues.append(str(exc))
            break
    if len(all_rows) != total:
        issues.append("COLUMN_GROUP_LIMIT_OR_REQUEST_BUDGET")
    result = {
        "column": column,
        "source_declared_total": total,
        "catalog_entries": len(all_rows),
        "catalog_complete": not issues and len(all_rows) == total,
        "issues": issues,
        "range": ["2022-12-01", "2023-12-31"],
        "receipts": receipts,
        "entries_in_range": [r for r in all_rows if "2022-12-01" <= r["display_date"] <= "2023-12-31"],
        "all_catalog_entries": all_rows,
        "body_complete": False,
        "training_history_complete": False,
    }
    save(PUBLIC / f"catalog-{column}.json", result)
    return result


def collect_article(client, row):
    path = PUBLIC / "articles" / (digest(row["url"]) + ".json")
    if path.exists():
        return read(path)
    try:
        raw, receipt = client.fetch(row["url"], group="public")
        soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
        title = soup.find("meta", attrs={"name": "ArticleTitle"})
        pub = soup.find("meta", attrs={"name": "PubDate"})
        content = soup.select_one("#zoom") or soup.select_one(".article-content")
        if not content or not pub or not title:
            raise ValueError("ARTICLE_TITLE_DATE_OR_BODY_UNRESOLVED")
        displayed = pub.get("content", "")[:10]
        if displayed != row["display_date"]:
            raise ValueError("ARTICLE_CATALOG_DATE_CONFLICT")
        if re.sub(r"\s+", "", title.get("content", "")) != re.sub(r"\s+", "", row["title"]):
            raise ValueError("ARTICLE_CATALOG_TITLE_CONFLICT")
        attachments = [urljoin(row["url"], a["href"]) for a in content.select("a[href]")]
        text = content.get_text("\n", strip=True)
        result = {
            "ok": True,
            "row": row,
            "receipt": receipt,
            "published_date": displayed,
            "content_text": text,
            "content_sha256": digest(text),
            "attachments_or_links": attachments,
            "attachment_review_complete": not attachments,
            "semantic_verified": False,
            "effective_date": None,
            "effective_date_reason": "Not inferred from publication or signature",
            "redistribution_authorized": False,
            "publisher_training_license_verified": False,
        }
    except ValueError as exc:
        result = {"ok": False, "row": row, "reason": str(exc)}
    save(path, result)
    return result


def main():
    plan = read(OUT / "plan.json")
    save(
        OUT / "public-plan-amendment-v2.json",
        {
            "reason": "Live navigation shows media reports is col46; col15 is local medical-insurance news",
            "evidence": read(OUT / "public-columns/14.json")["receipt"],
            "corrected_columns": {"104": "政策法规", "14": "医保动态", "46": "媒体报道"},
            "unchanged_date_range": ["2022-12-01", "2023-12-31"],
            "unchanged_request_limit": 40,
            "allocation": (
                "Entire policy and media catalogs, first 6 official-news groups; dated articles to shared cap"
            ),
            "pagination_resolution": "One identical lookahead boundary item verified in raw adjacent pages",
            "prior_failed_probe_preserved": True,
            "not_a_training_protocol": True,
        },
    )
    lock = OUT / ".operation-lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("public-history")
    client = BoundedPublicReader(OUT, plan["requests"])
    try:
        catalogs = []
        for column, groups in ((104, 10), (14, 6), (46, 20)):
            p = PUBLIC / f"catalog-{column}.json"
            result = read(p) if p.exists() else collect_column(client, column, groups)
            catalogs.append(result)
            print(
                {
                    "column": column,
                    "entries": result["catalog_entries"],
                    "complete": result["catalog_complete"],
                    "in_range": len(result["entries_in_range"]),
                },
                flush=True,
            )
        articles = []
        policy = sorted(catalogs[0]["entries_in_range"], key=lambda r: (r["display_date"], r["url"]))
        media = sorted(catalogs[2]["entries_in_range"], key=lambda r: (r["display_date"], r["url"]))
        # 先留五篇政策正文，再读媒体正文；仅取最早日期，不参考任何标签或预测得失。
        for row in policy[:5] + media + policy[5:]:
            result = collect_article(client, row)
            articles.append(result)
        summary = {
            "created_at": datetime.now(ZONE).isoformat(),
            "catalogs": [
                {k: c[k] for k in ("column", "source_declared_total", "catalog_entries", "catalog_complete", "issues")}
                for c in catalogs
            ],
            "policy_entries_in_range": len(catalogs[0]["entries_in_range"]),
            "articles_checked": len(articles),
            "articles_saved_and_identity_passed": sum(a["ok"] for a in articles),
            "article_failures": dict(Counter(a["reason"] for a in articles if not a["ok"])),
            "policy_complete_for_training": False,
            "news_complete_for_training": False,
            "fit_count": 0,
            "status": "STOP_PREPARATION_HISTORY_OR_BODY_INCOMPLETE",
            "catalog_only_does_not_prove_all_relevant_policies": True,
        }
        save(PUBLIC / "result.json", summary)
        print(summary)
    finally:
        client.close()
        lock.unlink()


if __name__ == "__main__":
    main()
