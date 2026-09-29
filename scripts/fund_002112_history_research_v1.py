"""002112 公告历史与语义补齐：只读旧包，全部新产物写入独立 V1 目录。"""

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pypdfium2 as pdfium
from app.services.fund_information_history_v1 import (
    OLD,
    OUT,
    QUERY,
    ROOT,
    ZONE,
    BoundedPublicReader,
    buyback_candidates,
    catalog_rows,
    classify,
    covered,
    digest,
    normalize,
    pdf_revision,
    read,
    save,
    sha,
)

PYTHON = Path(__file__).resolve().parents[1]
FRONT = Path("C:/WebStormProject/workSpace05")
JAVA = Path("C:/ideaProject/workSpace12")


def old_bundle():
    manifest = read(OLD / "event-admission/bundle-manifest.json")
    if sha(manifest["path"]) != manifest["sha256"]:
        raise ValueError("OLD_FACT_BUNDLE_CHANGED")
    return read(manifest["path"])


def git_snapshot():
    return {
        str(p): {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p, text=True).strip(),
            "status": subprocess.check_output(["git", "status", "--porcelain=v1", "-uall"], cwd=p, text=True),
        }
        for p in (PYTHON, FRONT, JAVA)
    }


def prepare():
    """批次选择只依赖基金、报告、日期与缺口，完全不读取预测或答案列。"""
    if (OUT / "plan.json").exists():
        return read(OUT / "plan.json")
    rows = read(OLD / "event-admission/row-coverage.json")
    groups = defaultdict(list)
    for row in rows:
        if row["fund_code"] == "002112" and row["missing_companies"]:
            groups[row["report_sha256"]].append(row)
    selected = sorted(groups, key=lambda key: min(r["target"] for r in groups[key]))[:2]
    windows = []
    for key in selected:
        members = groups[key]
        for code in sorted({s for r in members for s in r["missing_companies"]}):
            dates = [r["target"] for r in members if code in r["missing_companies"]]
            windows.append(
                {
                    "stock": code,
                    "start": str(date.fromisoformat(min(dates)) - timedelta(days=30)),
                    "end": str(date.fromisoformat(max(dates)) - timedelta(days=1)),
                    "report_sha256": key,
                    "target_records": len(dates),
                }
            )
    validation = read(OLD / "validation-final.json")
    protect = {str(p): sha(p) for p in OLD.rglob("*") if p.is_file()}
    protect.update(validation["new_code_files"])
    protect[validation["document"]["path"]] = validation["document"]["sha256"]
    for path, expected in read(OLD / "delivery-manifest.json")["files"].items():
        if sha(path) != expected:
            raise ValueError("OLD_DELIVERY_CHANGED:" + path)
    for path, expected in old_bundle()["sources"].items():
        if sha(path) != expected:
            raise ValueError("OLD_SOURCE_CHANGED:" + path)
        protect[path] = expected
    for path, expected in protect.items():
        if sha(path) != expected:
            raise ValueError("OLD_FILE_CHANGED:" + path)
    save(OUT / "protection-before.json", {"files": protect, "git": git_snapshot()})
    value = {
        "schema": "002112_HISTORY_PREPARATION_V1",
        "created_at": datetime.now(ZONE).isoformat(),
        "mode": "DATA_PREPARATION_ONLY_NO_TRAINING_PROTOCOL",
        "new_fit_budget": 0,
        "old_fits": 70,
        "old_remaining_fits": 12,
        "old_stopped_hypotheses_remain_stopped": True,
        "selection": "First two incomplete 002112 report groups ordered by earliest target, no outcomes read",
        "target_rows": sum(len(groups[k]) for k in selected),
        "windows": windows,
        "requests": {"company": 220, "public": 40},
        "maximum_pages_per_window": 10,
        "maximum_new_documents": 100,
        "maximum_pdf_bytes": 25_000_000,
        "maximum_pdf_pages": 500,
        "public_source_plan": {
            "source": "国家医疗保障局",
            "policy_column": "政策法规 col104",
            "news_columns": ["医保动态 col14", "媒体报道 col15"],
            "start": "2022-12-01",
            "end": "2023-12-31",
            "scope": "Bounded source/column history feasibility, not all industries or all market news",
            "limits": "40 requests shared by metadata, pagination and article reads; no search-as-coverage",
            "complete_requires": "Stable count, all dated entries in registered interval, article/attachment audit",
        },
        "use_basis": "User-authorized local study of free public disclosures; no redistribution license asserted",
        "stops": [
            "Budget or response limits stop dependent collection",
            "Identity/date/page conflict blocks window",
            "Unexplained revisions block semantic use",
            "Missing data never encode as zero",
            "No new hypothesis or fit until complete preparation and valid authorization",
        ],
        "source_sha256": {
            str(OLD / "event-admission/company-gap-worklist.json"): sha(
                OLD / "event-admission/company-gap-worklist.json"
            ),
            str(OLD / "event-admission/row-coverage.json"): sha(OLD / "event-admission/row-coverage.json"),
        },
    }
    save(OUT / "plan.json", value)
    return value


def collect_company():
    """有限补齐原始目录，并下载范围内三类原件；一个窗口失败不伪造完整。"""
    plan = read(OUT / "plan.json")
    stocks = read(ROOT / "supplement/cninfo-stock-map.json")["rows"]
    client = BoundedPublicReader(OUT, plan["requests"])
    try:
        for index, window in enumerate(plan["windows"], 1):
            output = OUT / "company-catalogs" / (digest(window) + ".json")
            if output.exists():
                continue
            receipts, pages = [], []
            try:
                stock = stocks.get(window["stock"])
                if not stock:
                    raise ValueError("MISSING_PUBLIC_STOCK_ID")
                page, maximum = 1, 1
                while page <= maximum:
                    params = {
                        "pageNum": page,
                        "pageSize": 30,
                        "column": "szse",
                        "tabName": "fulltext",
                        "plate": "",
                        "stock": stock["code"] + "," + stock["orgId"],
                        "searchkey": "",
                        "secid": "",
                        "category": "",
                        "trade": "",
                        "sortName": "",
                        "sortType": "",
                        "seDate": window["start"] + "~" + window["end"],
                        "isHLtitle": "true",
                    }
                    raw, receipt = client.fetch(QUERY, params=params)
                    value = json.loads(raw)
                    maximum = max(1, (int(value["totalAnnouncement"]) + 29) // 30)
                    if maximum > plan["maximum_pages_per_window"]:
                        raise ValueError("CATALOG_PAGE_LIMIT")
                    pages.append((page, value))
                    receipts.append(receipt)
                    page += 1
                rows = catalog_rows(pages, window["stock"], window["start"], window["end"])
                result = {"window": window, "catalog_complete": True, "rows": rows, "receipts": receipts}
            except (ValueError, KeyError, TypeError) as exc:
                result = {"window": window, "catalog_complete": False, "reason": str(exc), "receipts": receipts}
            save(output, result)
            print(
                json.dumps(
                    {
                        "catalog": index,
                        "total": len(plan["windows"]),
                        "passed": result["catalog_complete"],
                        "rows": len(result.get("rows", [])),
                    }
                ),
                flush=True,
            )
        entries = {}
        for path in sorted((OUT / "company-catalogs").glob("*.json")):
            cat = read(path)
            if cat["catalog_complete"]:
                for row in cat["rows"]:
                    if row["category"]:
                        previous = entries.get(row["announcementId"])
                        if previous and previous != row:
                            raise ValueError("DUPLICATE_DOCUMENT_METADATA_CONFLICT")
                        entries[row["announcementId"]] = row
        for index, row in enumerate(
            sorted(entries.values(), key=lambda r: (r["published_date"], r["announcementId"])), 1
        ):
            output = OUT / "company-documents" / (row["announcementId"] + ".json")
            if output.exists():
                continue
            try:
                if index > plan["maximum_new_documents"]:
                    raise ValueError("DOCUMENT_COUNT_LIMIT")
                url = "https://static.cninfo.com.cn/" + row["adjunctUrl"]
                if not url.lower().endswith(".pdf"):
                    raise ValueError("NOT_A_PDF_URL")
                raw, receipt = client.fetch(url, maximum_bytes=plan["maximum_pdf_bytes"])
                if not raw.startswith(b"%PDF"):
                    raise ValueError("NOT_PDF_CONTENT")
                with pdfium.PdfDocument(raw) as document:
                    if len(document) > plan["maximum_pdf_pages"]:
                        raise ValueError("PDF_PAGE_LIMIT")
                    metadata, pages = document.get_metadata_dict(), []
                    for page in document:
                        textpage = page.get_textpage()
                        try:
                            pages.append(textpage.get_text_range())
                        finally:
                            textpage.close()
                            page.close()
                issues = pdf_revision(metadata, row["published_date"])
                first = normalize("".join(pages[:2]))
                identity = any(s in first for s in row["secCode"].split(","))
                title_found = normalize(row["title_plain"]) in first
                # 名称加前缀的目录标题允许留待复核，不能只凭 PDF 可解析就声称原件身份核对完成。
                result = {
                    "row": row,
                    "receipt": receipt,
                    "metadata": metadata,
                    "pages": pages,
                    "body_saved": True,
                    "identity_stock_verified": identity,
                    "title_exact_in_first_two_pages": title_found,
                    "revision_issues": issues,
                    "source_identity_passed": identity and title_found,
                    "semantic_verified": False,
                }
            except (ValueError, pdfium.PdfiumError) as exc:
                result = {"row": row, "body_saved": False, "reason": str(exc), "semantic_verified": False}
            save(output, result)
            if index % 5 == 0 or index == len(entries):
                print(json.dumps({"document": index, "total": len(entries), "saved": result["body_saved"]}), flush=True)
    finally:
        client.close()
    return coverage()


def coverage():
    """分别统计目录与正文状态，不把扩大目录覆盖误报为训练材料就绪。"""
    old = old_bundle()
    intervals = {k: list(v) for k, v in old["coverage"]["company"].items()}
    catalog = [read(p) for p in sorted((OUT / "company-catalogs").glob("*.json"))]
    for cat in catalog:
        if cat["catalog_complete"]:
            w = cat["window"]
            intervals.setdefault(w["stock"], []).append([w["start"], w["end"]])
    rows, missing = [], []
    for row in read(OLD / "event-admission/row-coverage.json"):
        start = str(date.fromisoformat(row["target"]) - timedelta(days=30))
        end = str(date.fromisoformat(row["target"]) - timedelta(days=1))
        gaps = [s for s in row["missing_companies"] if not covered(intervals.get(s, []), start, end)]
        complete = row["company_coverage_passed"] or (bool(row["missing_companies"]) and not gaps)
        rows.append(
            {
                "fund_code": row["fund_code"],
                "target": row["target"],
                "catalog_complete": complete,
                "remaining_companies": gaps,
            }
        )
        missing.extend({"stock": s, "start": start, "end": end} for s in gaps)
    docs = [read(p) for p in sorted((OUT / "company-documents").glob("*.json"))]
    result = {
        "new_fits": 0,
        "cumulative_fits": 70,
        "catalog_windows_attempted": len(catalog),
        "catalog_windows_passed": sum(c["catalog_complete"] for c in catalog),
        "catalog_rows": sum(len(c.get("rows", [])) for c in catalog),
        "new_documents": len(docs),
        "bodies_saved": sum(d["body_saved"] for d in docs),
        "source_identity_passed": sum(d.get("source_identity_passed", False) for d in docs),
        "revision_issues": sum(bool(d.get("revision_issues")) for d in docs),
        "old_catalog_complete": 705,
        "new_catalog_complete": sum(r["catalog_complete"] for r in rows),
        "target_catalog_complete": sum(r["catalog_complete"] for r in rows if r["fund_code"] == "002112"),
        "training_ready": False,
        "semantic_complete_training_rows": None,
        "limitation": "Catalog is complete only for registered source/title scope; body/semantics separate",
    }
    save(OUT / "row-coverage.json", rows)
    # 保留逐行缺口，不通过聚合丢失基金/日期对应关系；合并视图用于后续批次。
    grouped = defaultdict(list)
    for item in missing:
        grouped[item["stock"]].append([item["start"], item["end"]])
    merged = []
    for code, spans in sorted(grouped.items()):
        intervals_for_code = []
        for lo, hi in sorted(spans):
            if intervals_for_code and lo <= str(date.fromisoformat(intervals_for_code[-1][1]) + timedelta(days=1)):
                intervals_for_code[-1][1] = max(hi, intervals_for_code[-1][1])
            else:
                intervals_for_code.append([lo, hi])
        merged.extend({"stock": code, "start": lo, "end": hi} for lo, hi in intervals_for_code)
    result["remaining_gap_companies"] = len(grouped)
    result["remaining_gap_intervals"] = len(merged)
    save(OUT / "company-gap-worklist.json", merged)
    save(OUT / "company-result.json", result)
    return result


def public_probe():
    """栏目入口探测有单独 40 次总上限；没有分页证据时不声称历史完整。"""
    client = BoundedPublicReader(OUT, read(OUT / "plan.json")["requests"])
    results = []
    try:
        for column in (104, 14, 15):
            url = f"https://www.nhsa.gov.cn/col/col{column}/index.html"
            try:
                raw, receipt = client.fetch(url, group="public")
                html = raw.decode("utf-8", errors="replace")
                result = {
                    "column": column,
                    "receipt": receipt,
                    "pagination_hints": re.findall(
                        r".{0,180}(?:dataproxy|createPage|recordCount|totalRecord|unitid|webid|colid|perpage).{0,250}",
                        html,
                    ),
                    "history_complete": False,
                }
            except ValueError as exc:
                result = {"column": column, "history_complete": False, "reason": str(exc)}
            save(OUT / "public-columns" / (str(column) + ".json"), result)
            results.append(result)
    finally:
        client.close()
    return results


def semantic_inventory():
    """旧事实只读，生成语义新版候选；结构化候选不自动等于审核通过。"""
    sources, values = {}, []
    for fact in old_bundle()["facts"]:
        if fact["category"] not in ("PERFORMANCE", "BUYBACK", "MAJOR_CONTRACT"):
            continue
        item_id = fact["document_id"].split(":", 1)[1]
        # 历史文件名使用旧 digest 的紧凑 JSON 定义，不与新版摘要混用。
        import hashlib

        name = hashlib.sha256(json.dumps(item_id, separators=(",", ":")).encode()).hexdigest()
        path = ROOT / "supplement/company-documents" / (name + ".json")
        document = read(path)
        if sha(fact["source"]["path"]) != fact["source"]["sha256"]:
            raise ValueError("SEMANTIC_PDF_CHANGED")
        sources[str(path)] = sha(path)
        typed = classify(fact["title"])
        values.append(
            {
                "document_id": item_id,
                "entity_id": fact["entity_id"],
                "title": fact["title"],
                "published_date": fact["published_date"],
                "old_stage": fact["stage"],
                **typed,
                "source": fact["source"],
                "text_path": str(path),
                "quantities": buyback_candidates(document["pages"]) if typed["category"] == "BUYBACK" else [],
                "semantic_status": "UNREVIEWED",
                "old_revision_unresolved": fact["revision_unresolved"],
                "no_prediction_direction": True,
            }
        )
    save(OUT / "semantic-inventory.json", values)
    save(OUT / "semantic-sources.json", sources)
    result = {
        "documents": len(values),
        "classification": dict(Counter(v["stage"] for v in values)),
        "cumulative_buyback_candidates": sum(len(v["quantities"]) for v in values),
        "reviewed_quantities": 0,
        "new_fits": 0,
    }
    save(OUT / "semantic-inventory-summary.json", result)
    return result


def verify():
    protected = read(OUT / "protection-before.json")["files"]
    failures = [p for p, expected in protected.items() if not Path(p).is_file() or sha(p) != expected]
    if failures:
        raise ValueError("PROTECTED_FILES_CHANGED:" + repr(failures))
    checked = 0
    for path in (OUT / "receipts").glob("*.json"):
        value = read(path)
        if value["ok"]:
            if sha(value["path"]) != value["sha256"]:
                raise ValueError("NEW_SOURCE_CHANGED")
            checked += 1
    budgets = Counter(read(p)["group"] for p in (OUT / "requests").glob("*.json"))
    limits = read(OUT / "plan.json")["requests"]
    if any(n > limits[k] for k, n in budgets.items()):
        raise ValueError("REQUEST_BUDGET_EXCEEDED")
    ledger = list((OLD / "fit-ledger").glob("*.json"))
    if len(ledger) != 12:
        raise ValueError("OLD_LEDGER_CHANGED")
    result = {
        "passed": True,
        "protected_files": len(protected),
        "new_source_receipts": checked,
        "requests": dict(budgets),
        "request_limits": limits,
        "new_fits": 0,
        "cumulative_fits": 70,
        "old_ledger_entries": 12,
        "git": git_snapshot(),
    }
    # Git 状态会增加交付文件，因此按内容寻址保存每次核验，避免篡改前次时点。
    save(OUT / "verification" / (digest(result) + ".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "company", "public-probe", "semantics", "verify"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / ".operation-lock"
    # 正常退出释放自己的锁；中断留下锁，须确认没有运行进程后人工恢复，不自动清除。
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(args.command)
    try:
        action = {
            "prepare": prepare,
            "company": collect_company,
            "public-probe": public_probe,
            "semantics": semantic_inventory,
            "verify": verify,
        }[args.command]
        result = action()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
