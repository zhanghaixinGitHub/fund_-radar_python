"""第十二批的三家年报摘要参考批次；限定检索，不计为全公司历史目录覆盖。"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.services.fund_earnings_batch_v4 import PYTHON, RUNS, extract_pdf, issuer_identity
from app.services.fund_earnings_reference_v2 import select_annual_summary
from app.services.fund_information_history_v1 import (
    QUERY,
    ROOT,
    ZONE,
    BoundedPublicReader,
    catalog_rows,
    pdf_revision,
    read,
    save,
    sha,
)

OUT = RUNS / "20260929-earnings-reference-v2"
PARENT = RUNS / "20260929-earnings-v12"
SOURCES = {
    "688083": ["1215684451", "1215989547"],
    "688023": ["1215723579", "1215990412"],
    "300676": ["1215725876", "1215989715"],
}


def prepare():
    if (OUT / "plan.json").exists():
        return check_plan()
    source_paths = []
    for stock, identities in SOURCES.items():
        for key in identities:
            path = PARENT / "documents" / f"{key}.json"
            d = read(path)
            if d["row"]["secCode"] != stock or not d["identity"]["passed"] or d["revision_issues"]:
                raise ValueError("REFERENCE_TRIGGER_NOT_VERIFIED")
            source_paths.append(path)
    code = [
        PYTHON / n
        for n in (
            "app/services/fund_earnings_reference_v2.py",
            "scripts/fund_002112_earnings_reference_v2.py",
            "tests/test_fund_earnings_reference_v2.py",
        )
    ]
    inherited = read(PARENT / "plan.json")["code_hashes"]
    plan = {
        "created_at": datetime.now(ZONE).isoformat(),
        "year": 2022,
        "stocks": sorted(SOURCES),
        "start": "2023-03-01",
        "end": "2023-04-30",
        "searchkey": "年度报告摘要",
        "maximum_pages_per_stock": 2,
        "limits": {"catalog": 6, "body": 3},
        "maximum_documents": 3,
        "maximum_pdf_bytes": 25000000,
        "maximum_pdf_pages": 500,
        "scope": (
            "Three independently published annual summaries only; "
            "targeted queries are not complete company-history coverage"
        ),
        "selection_basis": (
            "Current registered batch has forecast and preliminary results for these same three issuers/year; "
            "no labels or outcomes used"
        ),
        "purpose": "Compare successive disclosures; never backdate annual figures into forecast/preliminary queries",
        "new_fits": 0,
        "cumulative_fits": 70,
        "stops": [
            "No retries or range widening",
            "Unique same-issuer/year unrevised summary required",
            "Stop source on identity/date/size conflict",
            "No use before public-date-plus-one-day 08:00",
            "No new training",
        ],
        "code_hashes": {**inherited, **{str(p): sha(p) for p in code}},
        "source_hashes": {str(p): sha(p) for p in [*source_paths, ROOT / "supplement/cninfo-stock-map.json"]},
    }
    save(OUT / "plan.json", plan)
    return {"stocks": plan["stocks"], "limits": plan["limits"], "new_fits": 0}


def check_plan():
    p = read(OUT / "plan.json")
    if any(sha(path) != h for path, h in {**p["code_hashes"], **p["source_hashes"]}.items()):
        raise ValueError("FROZEN_REFERENCE_DEPENDENCY_CHANGED")
    return p


def collect():
    plan = check_plan()
    if (OUT / "collection-stop.json").exists():
        raise ValueError("REFERENCE_BATCH_STOPPED_USE_AUDIT")
    mapping = read(ROOT / "supplement/cninfo-stock-map.json")["rows"]
    reader = BoundedPublicReader(OUT, plan["limits"])
    try:
        for stock in plan["stocks"]:
            failure, target = OUT / f"failure-{stock}.json", OUT / "documents" / f"{stock}.json"
            if target.exists() or failure.exists():
                continue
            try:
                catalog_path = OUT / "catalogs" / f"{stock}.json"
                if not catalog_path.exists():
                    pages, receipts, number, total = [], [], 1, 1
                    while number <= total:
                        params = {
                            "pageNum": number,
                            "pageSize": 30,
                            "column": "szse",
                            "tabName": "fulltext",
                            "plate": "",
                            "stock": mapping[stock]["code"] + "," + mapping[stock]["orgId"],
                            "searchkey": plan["searchkey"],
                            "secid": "",
                            "category": "",
                            "trade": "",
                            "sortName": "",
                            "sortType": "",
                            "seDate": plan["start"] + "~" + plan["end"],
                            "isHLtitle": "true",
                        }
                        raw, receipt = reader.fetch(QUERY, params=params, group="catalog")
                        data = json.loads(raw)
                        total = max(1, (int(data["totalAnnouncement"]) + 29) // 30)
                        if total > plan["maximum_pages_per_stock"]:
                            raise ValueError("REFERENCE_PAGE_LIMIT")
                        pages.append((number, data))
                        receipts.append(receipt)
                        number += 1
                    rows = catalog_rows(pages, stock, plan["start"], plan["end"])
                    save(catalog_path, {"rows": rows, "receipts": receipts, "targeted_search_only": True})
                row = select_annual_summary(read(catalog_path)["rows"], stock, plan["year"])
                raw, receipt = reader.fetch(
                    "https://static.cninfo.com.cn/" + row["adjunctUrl"],
                    group="body",
                    maximum_bytes=plan["maximum_pdf_bytes"],
                )
                pages, metadata = extract_pdf(raw, plan["maximum_pdf_pages"])
                save(
                    target,
                    {
                        "row": row,
                        "pages": pages,
                        "metadata": metadata,
                        "receipt": receipt,
                        "identity": issuer_identity(row, pages),
                        "revision_issues": pdf_revision(metadata, row["published_date"]),
                        "body_saved": True,
                        "semantic_verified": False,
                    },
                )
            except ValueError as exc:
                save(failure, {"stock": stock, "reason": str(exc), "do_not_retry": True, "new_fits": 0})
    finally:
        reader.close()
    save(
        OUT / "collection-stop.json",
        {"reason": "ALL_THREE_REFERENCE_POSITIONS_ACCOUNTED", "do_not_resume_collect": True, "new_fits": 0},
    )
    return audit()


def audit():
    plan = check_plan()
    saved = []
    for stock in plan["stocks"]:
        path = OUT / "documents" / f"{stock}.json"
        if not path.exists():
            if not (OUT / f"failure-{stock}.json").exists():
                raise ValueError("REFERENCE_POSITION_NOT_ACCOUNTED")
            continue
        d, catalog = read(path), read(OUT / "catalogs" / f"{stock}.json")
        for receipt in [*catalog["receipts"], d["receipt"]]:
            if sha(receipt["path"]) != receipt["sha256"]:
                raise ValueError("REFERENCE_BYTES_CHANGED")
        rows = catalog_rows(
            [(p["params"]["pageNum"], read(p["path"])) for p in catalog["receipts"]], stock, plan["start"], plan["end"]
        )
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), plan["maximum_pdf_pages"])
        if (
            rows != catalog["rows"]
            or select_annual_summary(rows, stock, plan["year"]) != d["row"]
            or pages != d["pages"]
            or metadata != d["metadata"]
            or issuer_identity(d["row"], pages) != d["identity"]
            or pdf_revision(metadata, d["row"]["published_date"]) != d["revision_issues"]
        ):
            raise ValueError("REFERENCE_REPLAY_CHANGED")
        saved.append(
            {
                "stock": stock,
                "document_id": d["row"]["announcementId"],
                "published_date": d["row"]["published_date"],
                "identity_passed": d["identity"]["passed"],
                "revision_issues": d["revision_issues"],
            }
        )
    used = dict(Counter(read(p)["group"] for p in (OUT / "requests").glob("*.json")))
    if any(count > plan["limits"][group] for group, count in used.items()):
        raise ValueError("REFERENCE_REQUEST_LIMIT_EXCEEDED")
    result = {
        "replay_passed": True,
        "saved": saved,
        "failures": [read(p) for p in OUT.glob("failure-*.json")],
        "requests": used,
        "limits": plan["limits"],
        "new_fits": 0,
        "training_ready": False,
        "complete_history_coverage_credit": 0,
    }
    save(OUT / "collection-and-audit-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "audit"))
    args = parser.parse_args()
    lock = RUNS / ".earnings-preparation.lock"
    owner = {"pid": os.getpid(), "run": OUT.name, "command": args.command, "at": datetime.now(ZONE).isoformat()}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
    try:
        print(json.dumps(globals()[args.command](), ensure_ascii=False, indent=2))
    finally:
        if read(lock) == owner:
            lock.unlink()


if __name__ == "__main__":
    main()
