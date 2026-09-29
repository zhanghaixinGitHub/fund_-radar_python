"""把既有历史公告及策略文字接入独立事实库，并核查原训练日期的真实覆盖。"""

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import httpx
from app.services.fund_002112_information_admission import BASE, RUN, save_once
from app.services.fund_002112_information_admission import OUT as EARLY
from app.services.fund_002112_round3_data import FEATURES
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json
from app.services.fund_information_evidence import ZONE, classification, make_fact, store_bundle, window_covered
from bs4 import BeautifulSoup

OUT = RUN / "event-admission"
STYLE = ROOT / "style-training-coverage/20260928-v1"
PUBLIC = (
    {
        "key": "policy-tcm-registration",
        "url": "https://mpa.gd.gov.cn/gkmlpt/content/4/4093/post_4093450.html",
        "category": "INDUSTRY_POLICY",
        "title": "国家药监局关于发布《中药注册管理专门规定》的公告（2023年第20号）",
        "published": "2023-02-13",
        "effective": "2023-07-01",
        "anchor": "自2023年7月1日起施行",
        "origin": "https://www.nmpa.gov.cn/xxgk/ggtg/qtggtg/20230210173401120.html",
    },
    {
        "key": "news-national-drug-procurement-8",
        "url": "https://app.www.gov.cn/govdata/gov/202303/30/498668/article.html",
        "category": "NEWS",
        "title": "国家组织药品集采开标：39种平均降价56%",
        "published": "2023-03-30",
        "effective": None,
        "anchor": "产生拟中选结果",
        "origin": "新华社，政府网公开转载",
    },
)


def checked(path, sources):
    """记录实际读取的版本，读取前后哈希一致；冻结后不静默跟随更新。"""
    path = Path(path)
    sha = file_hash(path)
    value = read_json(path)
    if file_hash(path) != sha:
        raise ValueError("SOURCE_CHANGED_DURING_READ")
    if str(path) in sources and sources[str(path)] != sha:
        raise ValueError("SOURCE_CHANGED_WITHIN_IMPORT")
    sources[str(path)] = sha
    return value


def pin_raw(receipt, sources):
    p = Path(receipt["path"]) if receipt.get("path") else ROOT / receipt["file"]
    if file_hash(p) != receipt["sha256"]:
        raise ValueError("RAW_SOURCE_CHANGED")
    sources[str(p)] = receipt["sha256"]
    return p


def source_record(path, url, *, received, kind):
    """明确个人研究的用途依据，不伪称获得来源新闻训练/转载商业授权。"""
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "url": url,
        "kind": kind,
        "received_at": received,
        "use_basis": "USER_AUTHORIZED_LOCAL_RESEARCH_OF_PUBLIC_DISCLOSURE_OR_SHORT_EXCERPT",
        "retention_basis": "LOCAL_RESEARCH_EVIDENCE; no publisher-specific duration asserted",
        "redistribution_authorized": False,
        "publisher_training_license_verified": False,
    }


def verify_catalog(catalog, sources):
    """回放所有分页，核总数、股票、日期、页码；不因已有索引就假定覆盖完整。"""
    groups = defaultdict(list)
    for receipt in catalog.get("receipts", []):
        start, end = receipt["params"]["seDate"].split("~")
        if start > "2023-12-31":
            continue
        path = pin_raw(receipt, sources)
        raw = checked(path, sources)
        groups[(start, end)].append((receipt["params"]["pageNum"], raw))
    intervals, failures = [], []
    for (start, end), pages in groups.items():
        pages.sort(key=lambda x: x[0])
        total = int(pages[0][1]["totalAnnouncement"])
        items = [r for _, p in pages for r in p.get("announcements") or []]
        ids = [r["announcementId"] for r in items]
        expected = list(range(1, max(1, (total + 29) // 30) + 1))
        valid = [n for n, _ in pages] == expected and all(int(p["totalAnnouncement"]) == total for _, p in pages)
        valid = valid and len(items) == total and len(set(ids)) == total
        valid = valid and all(catalog["code"].split(".")[0] in r["secCode"].split(",") for r in items)
        valid = valid and all(
            start <= datetime.fromtimestamp(r["announcementTime"] / 1000, ZONE).date().isoformat() <= end for r in items
        )
        if valid:
            intervals.append([max(start, "2016-01-01"), min(end, "2023-12-31")])
        else:
            failures.append(
                {"stock": catalog["code"], "window": [start, end], "reason": "CATALOG_PAGINATION_INCOMPLETE"}
            )
    return intervals, failures


def company_facts(sources):
    """限读 2023 年底之前的三类公告正文，其余年份只略过目录元数据。"""
    records, coverage, rejected = [], {}, []
    catalog_paths = sorted((ROOT / "supplement/company-announcements").glob("*.json"))
    for number, path in enumerate(catalog_paths, 1):
        catalog = checked(path, sources)
        if not catalog.get("receipts"):
            continue
        intervals, failures = verify_catalog(catalog, sources)
        coverage[catalog["code"].split(".")[0]] = intervals
        rejected.extend(failures)
        for row in catalog.get("rows", []):
            day = row["announced_at_source"][:10]
            if not "2016-01-01" <= day <= "2023-12-31":
                continue
            typed = classification(row["title_plain"])
            if typed is None:
                continue
            item_id = row["announcementId"]
            doc_path = ROOT / "supplement/company-documents" / (digest(item_id) + ".json")
            if not doc_path.exists():
                rejected.append({"id": item_id, "stock": row["secCode"], "day": day, "reason": "DOCUMENT_NOT_SAVED"})
                continue
            doc = checked(doc_path, sources)
            if (
                doc["announcement_id"] != item_id
                or doc["stock_code"] != row["secCode"]
                or doc["title"] != row["title_plain"]
                or doc["announced_at_source"] != row["announced_at_source"]
            ):
                raise ValueError("DOCUMENT_CATALOG_IDENTITY_MISMATCH")
            raw = pin_raw(doc["receipt"], sources)
            catalog_raw = pin_raw(row["receipt"], sources)
            original_rows = [
                r for r in read_json(catalog_raw).get("announcements", []) if r["announcementId"] == item_id
            ]
            if len(original_rows) != 1:
                raise ValueError("CATALOG_RAW_ANNOUNCEMENT_IDENTITY")
            original = original_rows[0]
            if (
                BeautifulSoup(original["announcementTitle"], "html.parser").get_text() != row["title_plain"]
                or original["secCode"] != row["secCode"]
                or original["adjunctUrl"] != row["adjunctUrl"]
                or datetime.fromtimestamp(original["announcementTime"] / 1000, ZONE).isoformat()
                != row["announced_at_source"]
            ):
                raise ValueError("CATALOG_RAW_TITLE_OR_TIME_CHANGED")
            body = [(i + 1, re.sub(r"\s+", "", text)) for i, text in enumerate(doc["pages"])]
            anchors = [
                {
                    "kind": "CATALOG_TITLE",
                    "text": row["title_plain"],
                    "path": str(catalog_raw),
                    "sha256": file_hash(catalog_raw),
                    "announcement_id": item_id,
                }
            ]
            # 保存指标段落原文与单位文字，未逐项复核的数字不填到结构化数量字段。
            pattern = r"[^。；]{0,70}(?:净利润|营业收入|同比|回购金额|合同金额)[^。；]{0,120}"
            excerpts = []
            for page, text in body:
                for match in re.finditer(pattern, text):
                    excerpts.append(
                        {
                            "page": page,
                            "text": match[0],
                            "normalized_offset": match.start(),
                            "meaning": "SOURCE_QUOTE_NOT_VERIFIED_QUANTITY",
                        }
                    )
                    if len(excerpts) == 3:
                        break
                if len(excerpts) == 3:
                    break
            records.append(
                make_fact(
                    category=typed[0],
                    stage=typed[1],
                    entity_type="COMPANY",
                    entity_id=row["secCode"],
                    document_id="CNINFO:" + item_id,
                    title=row["title_plain"],
                    published=day,
                    source=source_record(
                        raw,
                        urljoin("https://static.cninfo.com.cn/", row["adjunctUrl"]),
                        received=doc["receipt"]["received_at"],
                        kind="CNINFO_DISCLOSURE",
                    ),
                    anchors=anchors,
                    body_evidence=excerpts,
                    text_status=doc["text_status"],
                    revision_unresolved=typed[1] == "CORRECTION" or bool(row.get("source_changed")),
                    fact_status="TITLE_AND_SOURCE_VERIFIED; QUANTITIES_REQUIRE_REVIEW",
                )
            )
        if number % 100 == 0:
            print(f"公司目录已核 {number}/{len(catalog_paths)}，已接入披露 {len(records)}", flush=True)
    return records, coverage, rejected


def strategy_facts(sources):
    sections = checked(STYLE / "sections-reviewed.json", sources)
    tags = {r["report_sha256"]: r for r in checked(STYLE / "reviewed-tags.json", sources)}
    records, rejected = [], []
    for entry in sections:
        spec, section = entry["source"], entry["section"]
        sha = spec["report_sha256"]
        if spec["published_date"] > "2023-12-31":
            continue
        raw = Path(spec["raw_path"])
        if file_hash(raw) != sha:
            raise ValueError("STRATEGY_SOURCE_CHANGED")
        sources[str(raw)] = sha
        tag = tags[sha]
        if section["status"] != "EXTRACTED" or tag["evidence"] not in section["text"]:
            rejected.append({"report": sha, "reason": "STRATEGY_ANCHOR_UNRESOLVED"})
            continue
        records.append(
            make_fact(
                category="MANAGER_STRATEGY",
                stage="REPORT_NARRATIVE",
                entity_type="FUND",
                entity_id=spec["fund_code"],
                document_id=sha,
                title=spec["title"],
                published=max(spec["effective_publication_dates"]),
                source=source_record(raw, spec["raw_url"], received=None, kind="FUND_PERIODIC_REPORT"),
                anchors=[
                    {
                        "kind": "REPORT_STRATEGY_SECTION",
                        "pages": section["pages"],
                        "text": tag["evidence"],
                        "section_sha256": digest(section["text"]),
                        "offset": tag["section_quote_offset"],
                    }
                ],
                strategy_tag=tag["tag"],
                interpretation=tag["reason"],
                report_end=spec["report_end"],
                semantic_limit="REPORT_PERIOD_NARRATIVE; NOT_PROOF_OF_DAILY_ACTUAL_HOLDINGS",
                fact_status="SOURCE_SECTION_AND_PRIOR_REVIEW_VERIFIED",
            )
        )
    return records, rejected


def public_facts(sources):
    """有限公开补查：保存原网页、显示日期和短原文锚点，不回填更早的文件落款日。"""
    records, rejected = [], []
    save_once(
        OUT / "public-source-plan.json",
        {
            "maximum_requests": len(PUBLIC),
            "documents": list(PUBLIC),
            "coverage_claim": "SELECTED_DOCUMENTS_ONLY_NOT_EXHAUSTIVE_HISTORY",
        },
    )
    with httpx.Client(timeout=httpx.Timeout(25, connect=5), follow_redirects=False) as client:
        for spec in PUBLIC:
            receipt_path = OUT / "public" / (spec["key"] + ".json")
            if receipt_path.exists():
                receipt = read_json(receipt_path)
            else:
                request = OUT / "public" / (spec["key"] + "-request.json")
                if request.exists():
                    rejected.append({"id": spec["key"], "reason": "INCOMPLETE_REQUEST_NO_AUTORETRY"})
                    continue
                save_once(request, {"url": spec["url"], "at": datetime.now().astimezone().isoformat()})
                try:
                    response = client.get(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
                    response.raise_for_status()
                    if len(response.content) > 4_000_000:
                        raise ValueError("SOURCE_PAGE_TOO_LARGE")
                    raw = OUT / "public" / (spec["key"] + ".html")
                    with raw.open("xb") as stream:
                        stream.write(response.content)
                    receipt = {
                        "passed": True,
                        "path": str(raw),
                        "sha256": file_hash(raw),
                        "received_at": datetime.now().astimezone().isoformat(),
                    }
                except Exception as exc:
                    receipt = {"passed": False, "reason": type(exc).__name__}
                save_once(receipt_path, receipt)
            if not receipt["passed"]:
                rejected.append({"id": spec["key"], "reason": receipt["reason"]})
                continue
            raw = Path(receipt["path"])
            if file_hash(raw) != receipt["sha256"]:
                raise ValueError("PUBLIC_PAGE_CHANGED")
            sources[str(raw)] = receipt["sha256"]
            soup = BeautifulSoup(raw.read_bytes(), "html.parser", from_encoding="utf-8")
            text = re.sub(r"\s+", "", soup.get_text())
            full = raw.read_text(encoding="utf-8")
            if (
                re.sub(r"\s+", "", spec["title"]) not in text
                or spec["anchor"] not in text
                or spec["published"] not in full
            ):
                rejected.append({"id": spec["key"], "reason": "DISPLAY_DATE_OR_TEXT_ANCHOR_MISMATCH"})
                continue
            records.append(
                make_fact(
                    category=spec["category"],
                    stage="POLICY_PUBLISHED" if spec["category"] == "INDUSTRY_POLICY" else "DISCLOSED",
                    entity_type="INDUSTRY",
                    entity_id="MEDICINE",
                    document_id=spec["key"],
                    title=spec["title"],
                    published=spec["published"],
                    source=source_record(
                        raw, spec["url"], received=receipt["received_at"], kind="OFFICIAL_PUBLIC_PAGE"
                    ),
                    anchors=[
                        {"kind": "PAGE_TEXT", "text": spec["anchor"], "normalized_offset": text.index(spec["anchor"])}
                    ],
                    effective_date=spec["effective"],
                    original_source=spec["origin"],
                    fact_status="DISPLAY_DATE_AND_SHORT_EXCERPT_VERIFIED",
                    topic="MEDICINE",
                )
            )
    return records, rejected


def coverage_audit(records, company_coverage, sources):
    """在原名单全部日期检查覆盖；缺口清单按公司和连续区间合并，供有限补采复用。"""
    reports = {r["raw"]["sha256"]: r for r in checked(STYLE / "reports.json", sources)}
    pool = checked(EARLY / "inputs.json", sources)["train"]
    base_folds = checked(BASE / "folds.json", sources)
    strategy = {r["source"]["sha256"] for r in records if r["category"] == "MANAGER_STRATEGY"}
    rows, missing = [], defaultdict(list)
    for row in pool:
        report = reports.get(row["report_sha256"])
        start = (date.fromisoformat(row["target"]) - timedelta(days=30)).isoformat()
        end = (date.fromisoformat(row["target"]) - timedelta(days=1)).isoformat()
        codes = [h["stock_code"].split(".")[0] for h in report["holdings"]] if report else []
        gaps = [c for c in codes if not window_covered(company_coverage.get(c, []), start, end)]
        for code in gaps:
            missing[code].append([start, end])
        rows.append(
            {
                "fund_code": row["fund_code"],
                "target": row["target"],
                "report_sha256": row["report_sha256"],
                "company_coverage_passed": report is not None and bool(codes) and not gaps,
                "missing_companies": gaps,
                "strategy_passed": row["report_sha256"] in strategy,
                "policy_coverage_passed": False,
                "news_coverage_passed": False,
                "reason": "POLICY_NEWS_HAVE_DOCUMENTS_BUT_NOT_COMPLETE_REGISTERED_HISTORY",
            }
        )
    merged = []
    for code, intervals in sorted(missing.items()):
        spans = []
        for a, b in sorted(intervals):
            if spans and a <= (date.fromisoformat(spans[-1][1]) + timedelta(days=1)).isoformat():
                spans[-1][1] = max(b, spans[-1][1])
            else:
                spans.append([a, b])
        merged.extend({"stock": code, "start": a, "end": b} for a, b in spans)
    index = {(r["fund_code"], r["target"]): r for r in rows}
    folds = {}
    for f in base_folds:
        if f["name"] == "FULL":
            continue
        selected = [index[tuple(k)] for k in f["train_ids"]]
        exams = [index[r["fund_code"], r["target"]] for r in f["exam"]]
        folds[f["name"]] = {
            "training_rows_checked": len(selected),
            "company_complete": sum(r["company_coverage_passed"] for r in selected),
            "strategy_complete": sum(r["strategy_passed"] for r in selected),
            "exam_days": len(exams),
            "exam_company_complete": sum(r["company_coverage_passed"] for r in exams),
            "exam_strategy_complete": sum(r["strategy_passed"] for r in exams),
        }
    save_once(OUT / "row-coverage.json", rows)
    save_once(OUT / "company-gap-worklist.json", merged)
    return {
        "training_pool_rows": len(pool),
        "company_complete": sum(r["company_coverage_passed"] for r in rows),
        "strategy_complete": sum(r["strategy_passed"] for r in rows),
        "company_gap_companies": len(missing),
        "company_gap_intervals": len(merged),
        "folds": folds,
        "policy_complete": 0,
        "news_complete": 0,
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    feature_names = [
        "disclosure_30d_" + n for n in ("performance", "buyback", "major_contract", "industry_policy", "news")
    ]
    feature_names += [
        "strategy_" + n for n in ("medicine", "ai_compute", "other_explicit", "diversified", "unspecified")
    ]
    save_once(
        OUT / "feature-contract.json",
        {
            "features": list(FEATURES) + feature_names,
            "candidate": "H2_EVENT_INFORMATION",
            "base_pool": "early-admission/inputs.json",
            "fixed_algorithm": "Original logistic recipe",
            "window": "30 calendar days before target; date-only disclosures usable next day 08:00",
            "count_unit": "DISCLOSURE; no guessed economic-event merging or monetary summation",
            "coverage_required": (
                "Complete declared company catalog windows, "
                "registered policy/news source history, exact report strategy"
            ),
            "missing_behavior": "BLOCK_NO_ZERO_FILL",
            "source_roster": ["CNINFO", "NMPA_AND_OFFICIAL_REPRINT", "GOV_CN_XINHUA", "FUND_REPORT_STRATEGY"],
            "old_samples": "Original 12 samples untouched; no claim they cover Q2",
            "real_fits_before_freeze": 0,
        },
    )
    sources = {}
    company, coverage, blocked_company = company_facts(sources)
    strategy, blocked_strategy = strategy_facts(sources)
    public, blocked_public = public_facts(sources)
    records = company + strategy + public
    rejected = blocked_company + blocked_strategy + blocked_public
    report = coverage_audit(records, coverage, sources)
    manifest = store_bundle(OUT, records, {"company": coverage, "policy": [], "news": []}, sources)
    save_once(OUT / "bundle-manifest.json", manifest)
    save_once(OUT / "source-rejections.json", rejected)
    save_once(OUT / "coverage-summary.json", report)
    save_once(OUT / "sources.json", {"files": sources})
    result = {
        "status": "STOP_PREPARATION_COVERAGE_INCOMPLETE",
        "material_ready": False,
        "new_fits": 0,
        "facts_persisted": manifest["facts"],
        "by_category": dict(Counter(r["category"] for r in records)),
        "public_docs": len(public),
        "blocked_source_items": len(rejected),
        "business_published": False,
        "adopted": False,
        "reasons": ["COMPANY_HISTORY_GAPS", "POLICY_NEWS_HISTORY_NOT_EXHAUSTIVE"],
        "next_evidence": (
            "company-gap-worklist.json plus complete dated policy/news source catalogs; "
            "do not spend fits on missing features"
        ),
    }
    save_once(OUT / "decision.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return result


def verify():
    manifest = read_json(OUT / "bundle-manifest.json")
    if file_hash(manifest["path"]) != manifest["sha256"]:
        raise ValueError("FACT_BUNDLE_CHANGED")
    body = read_json(manifest["path"])
    if digest(body) != manifest["content_sha256"]:
        raise ValueError("FACT_BUNDLE_CONTENT_CHANGED")
    for path, sha in body["sources"].items():
        if file_hash(path) != sha:
            raise ValueError("FACT_SOURCE_CHANGED")
    result = {
        "passed": True,
        "facts": len(body["facts"]),
        "sources_unchanged": len(body["sources"]),
        "training_preparation_passed": False,
        "real_fits": 0,
        "scope": "Fact storage and source integrity, not model readiness",
    }
    save_once(OUT / "fact-storage-verification.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    args = parser.parse_args()
    prepare() if args.command == "prepare" else verify()
