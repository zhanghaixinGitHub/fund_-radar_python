"""冻结历史公司目录后定向补资料；本地正文优先，相同原件不重下、不重复请求大模型。"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pypdfium2 as pdfium

from scripts import fund_002112_signal_common_v1 as c

KEYWORDS = ("业绩预告", "回购", "重大合同")
RELEVANT = re.compile(r"业绩预告|业绩预增|业绩快报|季度报告|半年度报告|年度报告|回购|重大合同|订单|停产|违约|立案|处罚")
EXCLUDE = re.compile(r"说明会|制度|法律意见|核查意见|审计报告|限制性股票|员工持股|摘要的更正|不存在|未被|风险持续评估")


def existing_docs():
    docs = {d["id"]: d for d in c.lines(c.OLD / "extraction-documents.jsonl")}
    for name in ("supplement-documents.jsonl", "supplement-retry-documents.jsonl"):
        docs.update({d["id"]: d for d in c.lines(c.OLD / name)})
    return docs


def freeze_catalog(root):
    dest = root / "collection-catalog.json"
    if dest.exists():
        return c.io.read(dest)
    reports = c.io.read(c.OLD / "reports.json")
    weights, names = defaultdict(Counter), defaultdict(set)
    rows = c.original_rows()
    for row in rows:
        report = c.choose_report(reports, row["as_of"])
        if not report:
            continue
        for h in report["holdings"]:
            weights[row["base"][:4]][h["stock_code"]] += float(h["nav_weight_pct"])
            names[h["stock_code"]].add(h["stock_name"])
    identities = c.io.payload(c.io.read(c.io.RESEARCH / "supplement/cninfo-stock-map.json"))["rows"]
    queries = []
    for year in ("2024", "2025", "2026"):
        for code, _ in weights[year].most_common(3):
            identity = identities.get(code[:6])
            if not identity:
                continue
            for keyword in KEYWORDS:
                queries.append(
                    {
                        "code": code,
                        "year": year,
                        "keyword": keyword,
                        "identity": identity,
                        "start": year + "-01-01",
                        "end": "2026-09-29" if year == "2026" else year + "-12-31",
                    }
                )
    result = {
        "at": c.io.now(),
        "companies_by_year": {k: dict(v) for k, v in weights.items()},
        "historical_names": {k: sorted(v) for k, v in names.items()},
        "queries": queries,
        "public_hosts": ["www.cninfo.com.cn", "static.cninfo.com.cn"],
        "selection": "每年历史已披露权重累计前三家公司，固定三个业务关键词，每项最多三页；本地相关原件优先。",
        "policy_news": "复用上一轮官方政策和可信媒体原件，背景或场景关系不进入公司事实；旧33页及3附件逐项保留缺证。",
        "limits": c.LIMITS,
        "maximum_pages_per_query": 3,
        "source_hashes": {str(c.OLD / "reports.json"): c.io.sha(c.OLD / "reports.json")},
    }
    c.io.save(dest, result)
    return result


def fetch(root, url, data=None):
    """每次真实公开读取先记账，账号错误或持续限流停止该来源，无自动换账号。"""
    host = urlparse(url).hostname
    if host not in ("www.cninfo.com.cn", "static.cninfo.com.cn") or not url.startswith("https://"):
        raise ValueError("SOURCE_NOT_FROZEN")
    key = c.io.digest([url, data])
    receipt = root / "public-receipts" / (key + ".json")
    if receipt.exists():
        meta = c.io.read(receipt)
        if meta.get("status") != 200:
            raise ValueError("PREVIOUS_SOURCE_FAILURE")
        path = root / meta["file"]
        assert c.io.sha(path) == meta["sha256"]
        return path.read_bytes(), meta
    if (root / ("source-stopped-" + host + ".json")).exists():
        raise ValueError("SOURCE_STOPPED")
    c.reserve(root, "public", key, {"url": url, "data": data})
    meta = {"at": c.io.now(), "url": url}
    try:
        with httpx.Client(timeout=25, trust_env=False, follow_redirects=False) as client:
            response = client.request(
                "POST" if data else "GET",
                url,
                data=data,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"},
            )
        raw = response.content
        digest = hashlib.sha256(raw).hexdigest()
        suffix = ".pdf" if raw.startswith(b"%PDF") else ".json"
        path = root / "public-raw" / (digest + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as f:
                f.write(raw)
        meta.update(status=response.status_code, file=str(path.relative_to(root)), sha256=digest)
        c.io.save(receipt, meta)
        if response.status_code in (401, 402, 403, 429):
            c.io.save(root / ("source-stopped-" + host + ".json"), meta)
        response.raise_for_status()
        return raw, meta
    except Exception as exc:
        if not receipt.exists():
            c.io.save(receipt, {**meta, "error": type(exc).__name__})
        raise


def pdf_document(root, entry, raw, receipt):
    """交易所原件校验身份与版本时间；只有日期时次日可用，较晚PDF版本不能回填。"""
    pdf = pdfium.PdfDocument(raw)
    try:
        pages = []
        for page in pdf:
            textpage = page.get_textpage()
            try:
                pages.append(textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
        metadata = pdf.get_metadata_dict()
    finally:
        pdf.close()
    body = "\n".join(pages)
    code = entry["stock_code"]
    if code[:6] not in body[:5000] or len(body.strip()) < 100:
        raise ValueError("BODY_OR_ISSUER_UNVERIFIED")
    day = entry["published_date"]
    constraints = [c.bodies.next_midnight(day)]
    for key in ("CreationDate", "ModDate"):
        date = c.bodies.pdf_day(metadata.get(key))
        if date and date > day:
            constraints.append(c.bodies.next_midnight(date))
    return {
        **entry,
        "kind": "company",
        "body": body,
        "body_sha256": receipt["sha256"],
        "raw_path": str(root / receipt["file"]),
        "raw_hash_verified": True,
        "body_available_at": max(constraints),
        "title_available_at": constraints[0],
        "body_exclusions": [],
        "receipt": receipt,
        "pdf_dates": metadata,
        "historical_first_seen_proven": False,
        "new_source": True,
    }


def run(root, online=False):
    root = Path(root)
    if (root / "collection-completion.json").exists():
        return c.io.read(root / "collection-completion.json")
    catalog = freeze_catalog(root)
    old = existing_docs()
    local = c.lines(c.bodies.ROOT / "documents.jsonl")
    additions, failures, duplicate = [], [], []
    known_hashes = {(d.get("stock_code"), d.get("body_sha256")) for d in old.values()}
    company_codes = {code for by_year in catalog["companies_by_year"].values() for code in by_year}
    selected = [
        d
        for d in local
        if d["id"] not in old
        and d.get("stock_code") in company_codes
        and RELEVANT.search(d["title"])
        and not EXCLUDE.search(d["title"])
    ]
    for d in sorted(selected, key=lambda x: (x["published_date"], x["id"]))[:500]:
        if d.get("body_exclusions") or not d.get("body_available_at"):
            failures.append({"id": d["id"], "reason": d.get("body_exclusions") or "NO_VERSION_TIME"})
            continue
        if (d.get("stock_code"), d.get("body_sha256")) in known_hashes:
            duplicate.append({"id": d["id"], "body_hash": d.get("body_sha256"), "cache": "OLD_BODY"})
            continue
        additions.append({**d, "new_source": False, "new_to_this_input_pool": True})
        known_hashes.add((d.get("stock_code"), d.get("body_sha256")))
    c.io.save(
        root / "local-supplement-selection.json",
        {
            "selected_ids": [d["id"] for d in selected],
            "kept_ids": [d["id"] for d in additions],
            "failures": failures,
            "duplicates": duplicate,
        },
    )
    entries = {}
    if online:
        for query in catalog["queries"]:
            for page in range(1, 4):
                identity = query["identity"]
                data = {
                    "pageNum": str(page),
                    "pageSize": "30",
                    "column": "szse",
                    "tabName": "fulltext",
                    "plate": "",
                    "stock": identity["code"] + "," + identity["orgId"],
                    "searchkey": query["keyword"],
                    "secid": "",
                    "category": "",
                    "trade": "",
                    "seDate": query["start"] + "~" + query["end"],
                    "sortName": "time",
                    "sortType": "asc",
                    "isHLtitle": "true",
                }
                try:
                    import json

                    raw, _ = fetch(root, "https://www.cninfo.com.cn/new/hisAnnouncement/query", data)
                    result = json.loads(raw)
                    for item in result.get("announcements") or []:
                        identifier = "company-" + str(item["announcementId"])
                        title = re.sub("<[^>]+>", "", item["announcementTitle"])
                        if identifier in old or any(d["id"] == identifier for d in additions):
                            continue
                        if not RELEVANT.search(title) or EXCLUDE.search(title):
                            continue
                        day = datetime.fromtimestamp(item["announcementTime"] / 1000, c.io.ZONE).date().isoformat()
                        if not query["start"] <= day <= query["end"] or item.get("secCode") != query["code"][:6]:
                            continue
                        entries[identifier] = {
                            "id": identifier,
                            "title": title,
                            "stock_code": query["code"],
                            "published_date": day,
                            "source_url": "https://static.cninfo.com.cn/" + item["adjunctUrl"],
                        }
                    if not result.get("hasMore"):
                        break
                except Exception as exc:
                    failures.append({"query": query, "page": page, "reason": type(exc).__name__})
                    break
        c.io.save(
            root / "external-document-selection.json", {"entries": sorted(entries.values(), key=lambda x: x["id"])}
        )
        for entry in sorted(entries.values(), key=lambda x: (x["published_date"], x["id"]))[: 500 - len(additions)]:
            try:
                raw, receipt = fetch(root, entry["source_url"])
                if (entry["stock_code"], receipt["sha256"]) in known_hashes:
                    duplicate.append({"id": entry["id"], "body_hash": receipt["sha256"], "cache": "CONTENT_MATCH"})
                    continue
                c.reserve(root, "new_body", entry["id"], {"source_hash": receipt["sha256"]})
                doc = pdf_document(root, entry, raw, receipt)
                additions.append(doc)
                known_hashes.add((entry["stock_code"], receipt["sha256"]))
            except Exception as exc:
                failures.append({"id": entry["id"], "reason": type(exc).__name__})
    # 旧公共材料缺口逐项归档，不将没拿到正文的标题转成中性事件。
    gaps = []
    for folder in ("public-documents", "public-recovered", "public-recovered-retry"):
        for p in (c.OLD / folder).glob("*.json"):
            d = c.io.read(p)
            if not d.get("admitted"):
                gaps.append(
                    {
                        "source": str(p),
                        "sha256": c.io.sha(p),
                        "reason": d.get("reason", d.get("error")),
                        "decision": "保持缺证；仅背景/无法验证直接持仓业务事实，不放宽正文准入",
                    }
                )
    c.io.save_lines(root / "supplement-documents.jsonl", additions)
    c.io.save(
        root / "collection-failures.json", {"new_failures": failures, "old_public_gaps": gaps, "duplicates": duplicate}
    )
    summary = {
        "at": c.io.now(),
        "cached_original_documents": len(old),
        "supplement_documents": len(additions),
        "local_supplement": sum(not d.get("new_source") for d in additions),
        "downloaded_documents": sum(bool(d.get("new_source")) for d in additions),
        "public_requests": len(c.lines(root / "public-ledger.jsonl")),
        "llm_requests": 0,
        "by_year": dict(Counter(d["published_date"][:4] for d in additions)),
        "failures": len(failures),
        "old_gap_records": len(gaps),
        "new_subscription": False,
    }
    c.io.save(root / "collection-completion.json", summary)
    c.stage(
        root,
        "定向资料补充",
        "COMPLETE_WITH_GAPS",
        ["collection-catalog.json", "collection-completion.json", "collection-failures.json"],
        ["不能验证的正文、首次公开版本和非直接公司政策关联保持隔离"],
        "来源与最小输入联合核验后冻结",
    )
    print(c.io.canonical(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    with c.io.writer_lock(args.root):
        run(args.root, args.online)
