"""有限业绩披露补采：冻结排序、独立目录、断点复用，永不调用拟合。"""

import argparse
import json
from collections import Counter
from datetime import date, datetime, timedelta

import pypdfium2 as pdfium
from app.services.fund_earnings_evidence_v1 import earnings_kind, plan_groups
from app.services.fund_information_history_v1 import (
    OLD,
    QUERY,
    ROOT,
    ZONE,
    BoundedPublicReader,
    catalog_rows,
    covered,
    digest,
    normalize,
    pdf_revision,
    read,
    save,
    sha,
)
from app.services.fund_information_history_v1 import (
    OUT as PREVIOUS,
)

OUT = ROOT / "information-research/20260928-earnings-v1"


def prepare():
    """先核旧冻结文件，再写有限计划；新版本不改前一批清单与失败记录。"""
    if (OUT / "plan.json").exists():
        return read(OUT / "plan.json")
    protected = dict(read(PREVIOUS / "protection-before.json")["files"])
    protected.update(read(PREVIOUS / "delivery-manifest.json")["files"])
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError("PROTECTED_EVIDENCE_CHANGED:" + path)
    for path in PREVIOUS.rglob("*"):
        if path.is_file():
            protected.setdefault(str(path), sha(path))
    save(OUT / "protection-before.json", {"files": protected})
    plan = plan_groups(read(OLD / "event-admission/row-coverage.json"), read(PREVIOUS / "row-coverage.json"))
    plan.update(
        {
            "schema": "EARNINGS_HISTORY_PREPARATION_V1",
            "created_at": datetime.now(ZONE).isoformat(),
            "scope": "DATA_PREPARATION_ONLY",
            "fits_authorized_here": 0,
            "cumulative_fits": 70,
            "prior_protocol_unused_fits": 12,
            "selection_basis": "Complete rows per missing company window; one group per fund; no outcome reads",
            "limits": {"catalog": 90, "body": 60},
            "max_catalog_pages": 10,
            "max_pdf_bytes": 25_000_000,
            "max_pdf_pages": 500,
            "document_order": "Forecasts then preliminary results then other earnings, each by date and id",
            "coverage_baseline": 797,
            "training_ready": False,
            "stops": [
                "No automatic retry of failed requests",
                "Catalog identity/date/pagination conflicts block window",
                "PDF identity or revision conflicts block facts",
                "Request/document limits stop that collection",
                "No new hypothesis registered and no fitting",
            ],
            "context_windows": [
                {
                    "stock": "002422",
                    "start": "2022-04-26",
                    "end": "2022-04-28",
                    "purpose": "ORIGINAL_FORECAST_EXPLICITLY_REFERENCED_BY_1213972395",
                }
            ],
            "semantic_case": {
                "issuer": "002422",
                "period": "2022-01-01/2022-06-30",
                "correction": "1213972395",
                "actual": "1214445963",
                "scope": "Resolve original forecast, correction and actual; no label selection",
            },
            "source_hashes": {
                str(p): sha(p)
                for p in (
                    OLD / "event-admission/row-coverage.json",
                    PREVIOUS / "row-coverage.json",
                    PREVIOUS / "semantic-inventory.json",
                    ROOT / "supplement/cninfo-stock-map.json",
                )
            },
        }
    )
    save(OUT / "plan.json", plan)
    return {
        "selected": plan["selected"],
        "target_rows": sum(g["rows"] for g in plan["selected"]),
        "catalog_windows": len(plan["windows"]) + len(plan["context_windows"]),
        "limits": plan["limits"],
    }


def params_for(window, stock, page):
    return {
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


def collect():
    """目录与正文各用独立硬上限。每个完成或失败结果排他保存，恢复不重取。"""
    plan = read(OUT / "plan.json")
    for p, expected in plan["source_hashes"].items():
        if sha(p) != expected:
            raise ValueError("PLAN_SOURCE_CHANGED")
    stocks = read(ROOT / "supplement/cninfo-stock-map.json")["rows"]
    client = BoundedPublicReader(OUT, plan["limits"])
    try:
        for i, window in enumerate(plan["windows"] + plan["context_windows"], 1):
            output = OUT / "catalogs" / (digest(window) + ".json")
            if output.exists():
                continue
            pages, receipts = [], []
            try:
                stock = stocks[window["stock"]]
                page, maximum = 1, 1
                while page <= maximum:
                    raw, receipt = client.fetch(QUERY, params=params_for(window, stock, page), group="catalog")
                    value = json.loads(raw)
                    receipts.append(receipt)
                    maximum = max(1, (int(value["totalAnnouncement"]) + 29) // 30)
                    if maximum > plan["max_catalog_pages"]:
                        raise ValueError("CATALOG_PAGE_LIMIT")
                    pages.append((page, value))
                    page += 1
                rows = catalog_rows(pages, window["stock"], window["start"], window["end"])
                result = {"window": window, "catalog_complete": True, "rows": rows, "receipts": receipts}
            except (ValueError, KeyError, TypeError) as exc:
                result = {"window": window, "catalog_complete": False, "reason": str(exc), "receipts": receipts}
            save(output, result)
            print(
                json.dumps({"catalog": i, "passed": result["catalog_complete"], "rows": len(result.get("rows", []))}),
                flush=True,
            )
        entries = {}
        for path in sorted((OUT / "catalogs").glob("*.json")):
            catalog = read(path)
            for row in catalog.get("rows", []) if catalog["catalog_complete"] else []:
                kind = earnings_kind(row["title_plain"])
                if kind:
                    item = {**row, "earnings_kind": kind}
                    key = row["announcementId"]
                    # 同一原件被多个窗口检出时，关键身份必须一致。
                    if key in entries and any(
                        entries[key][k] != item[k] for k in ("secCode", "adjunctUrl", "published_date", "title_plain")
                    ):
                        raise ValueError("DOCUMENT_METADATA_CONFLICT")
                    entries[key] = item
        rank = {"FORECAST": 0, "PRELIMINARY_RESULT": 1, "CORRECTION_NOTICE": 2, "REPORTED_RESULT": 3}
        ordered = sorted(
            entries.values(), key=lambda r: (rank[r["earnings_kind"]], r["published_date"], r["announcementId"])
        )
        save(OUT / "body-worklist.json", ordered)
        for i, row in enumerate(ordered, 1):
            output = OUT / "documents" / (row["announcementId"] + ".json")
            if output.exists():
                continue
            try:
                raw, receipt = client.fetch(
                    "https://static.cninfo.com.cn/" + row["adjunctUrl"],
                    group="body",
                    maximum_bytes=plan["max_pdf_bytes"],
                )
                if not raw.startswith(b"%PDF"):
                    raise ValueError("NOT_PDF_CONTENT")
                with pdfium.PdfDocument(raw) as document:
                    if len(document) > plan["max_pdf_pages"]:
                        raise ValueError("PDF_PAGE_LIMIT")
                    metadata, pages = document.get_metadata_dict(), []
                    for page in document:
                        text = page.get_textpage()
                        try:
                            pages.append(text.get_text_range())
                        finally:
                            text.close()
                            page.close()
                front = normalize("".join(pages[:2]))
                exact = normalize(row["title_plain"]) in front and row["secCode"] in front
                result = {
                    "row": row,
                    "receipt": receipt,
                    "metadata": metadata,
                    "pages": pages,
                    "body_saved": True,
                    "source_identity_exact": exact,
                    "revision_issues": pdf_revision(metadata, row["published_date"]),
                    "semantic_verified": False,
                }
            except (ValueError, pdfium.PdfiumError) as exc:
                result = {"row": row, "body_saved": False, "reason": str(exc), "semantic_verified": False}
            save(output, result)
            print(json.dumps({"body": i, "total": len(ordered), "saved": result["body_saved"]}), flush=True)
            if result.get("reason", "").startswith("PUBLIC_REQUEST_LIMIT"):
                break
    finally:
        client.close()
    return coverage()


def coverage():
    """复算完整目录覆盖；资料内容准入另算，不能混成可训练记录。"""
    bundle = read(read(OLD / "event-admission/bundle-manifest.json")["path"])
    intervals = {k: list(v) for k, v in bundle["coverage"]["company"].items()}
    for path in list((PREVIOUS / "company-catalogs").glob("*.json")) + list((OUT / "catalogs").glob("*.json")):
        catalog = read(path)
        if catalog["catalog_complete"]:
            w = catalog["window"]
            intervals.setdefault(w["stock"], []).append([w["start"], w["end"]])
    rows = []
    for row in read(OLD / "event-admission/row-coverage.json"):
        start = str(date.fromisoformat(row["target"]) - timedelta(days=30))
        end = str(date.fromisoformat(row["target"]) - timedelta(days=1))
        gaps = [s for s in row["missing_companies"] if not covered(intervals.get(s, []), start, end)]
        rows.append(
            {
                "fund_code": row["fund_code"],
                "target": row["target"],
                "catalog_complete": row["company_coverage_passed"] or (bool(row["missing_companies"]) and not gaps),
                "remaining_companies": gaps,
            }
        )
    save(OUT / "row-coverage.json", rows)
    docs = [read(p) for p in (OUT / "documents").glob("*.json")]
    cats = [read(p) for p in (OUT / "catalogs").glob("*.json")]
    result = {
        "training_ready": False,
        "new_fits": 0,
        "cumulative_fits": 70,
        "baseline_complete": 797,
        "catalog_complete": sum(r["catalog_complete"] for r in rows),
        "by_fund": dict(Counter(r["fund_code"] for r in rows if r["catalog_complete"])),
        "catalog_windows_passed": sum(c["catalog_complete"] for c in cats),
        "catalog_windows_attempted": len(cats),
        "catalog_rows": sum(len(c.get("rows", [])) for c in cats),
        "bodies_saved": sum(d["body_saved"] for d in docs),
        "bodies_in_worklist": len(read(OUT / "body-worklist.json")),
        "exact_identity_passed": sum(d.get("source_identity_exact", False) for d in docs),
        "metadata_conflicts": sum(bool(d.get("revision_issues")) for d in docs),
        "requests": dict(Counter(read(p)["group"] for p in (OUT / "requests").glob("*.json"))),
    }
    save(OUT / "collection-result.json", result)
    return result


def verify():
    """独立进程核文件、请求上限和原拟合台账，绝不改旧验证历史。"""
    files = read(OUT / "protection-before.json")["files"]
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError("PROTECTED_EVIDENCE_CHANGED:" + path)
    receipts = [read(p) for p in (OUT / "receipts").glob("*.json")]
    for receipt in receipts:
        if receipt["ok"] and sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("NEW_RAW_CHANGED")
    used = Counter(read(p)["group"] for p in (OUT / "requests").glob("*.json"))
    limits = read(OUT / "plan.json")["limits"]
    if any(v > limits[k] for k, v in used.items()):
        raise ValueError("REQUEST_BUDGET_EXCEEDED")
    ledger = list((OLD / "fit-ledger").glob("*.json"))
    if len(ledger) != 12:
        raise ValueError("ORIGINAL_FIT_LEDGER_COUNT_CHANGED")
    result = {
        "passed": True,
        "protected_files": len(files),
        "requests": dict(used),
        "limits": limits,
        "successful_receipts": sum(r["ok"] for r in receipts),
        "new_fits": 0,
        "cumulative_fits": 70,
        "original_fit_ledger_entries": len(ledger),
    }
    save(OUT / "verification" / (digest(result) + ".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "verify"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / ".operation-lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(args.command)
    try:
        print(
            json.dumps(
                {"prepare": prepare, "collect": collect, "verify": verify}[args.command](), ensure_ascii=False, indent=2
            )
        )
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
