"""独立窄范围免费核查：申昊 2022 年年度预告原件，最多三页目录及一份 PDF。"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.services.fund_earnings_batch_v4 import PYTHON, RUNS, extract_pdf
from app.services.fund_earnings_reference_v1 import reference_identity, select_reference
from app.services.fund_information_history_v1 import (
    QUERY,
    ROOT,
    ZONE,
    BoundedPublicReader,
    catalog_rows,
    normalize,
    pdf_revision,
    read,
    save,
    sha,
)

OUT = RUNS / "20260929-earnings-reference-v1"
PARENT = RUNS / "20260929-earnings-v8"


def prepare():
    if (OUT / "plan.json").exists():
        return check_plan()
    source = PARENT / "documents/1216443689.json"
    notice = read(source)
    if (
        not notice["identity"]["passed"]
        or notice["revision_issues"]
        or "2023年1月20日" not in normalize(notice["pages"][0])
        or "2023-008" not in notice["pages"][0]
    ):
        raise ValueError("EXPLICIT_REFERENCE_SOURCE_NOT_VERIFIED")
    names = [
        "app/services/fund_earnings_reference_v1.py",
        "scripts/fund_002112_earnings_reference_v1.py",
        "tests/test_fund_earnings_reference_v1.py",
        "app/services/fund_earnings_batch_v4.py",
        "app/services/fund_earnings_history_v2.py",
        "app/services/fund_information_history_v1.py",
        "app/services/fund_earnings_titles_v3.py",
        "app/services/fund_earnings_semantics_v2.py",
        "app/services/fund_earnings_evidence_v1.py",
    ]
    plan = {
        "created_at": datetime.now(ZONE).isoformat(),
        "schema": "EXPLICIT_REFERENCE_LINK_V1",
        "reference": {
            "stock": "300853",
            "published_date": "2023-01-20",
            "title": "2022年年度业绩预告",
            "notice_number": "2023-008",
        },
        "window": {"stock": "300853", "start": "2023-01-20", "end": "2023-01-20"},
        "source_notice": str(source),
        "limits": {"catalog": 3, "body": 1},
        "maximum_catalog_pages": 3,
        "maximum_pdf_bytes": 25_000_000,
        "maximum_pdf_pages": 500,
        "new_fit_budget": 0,
        "cumulative_fits": 70,
        "scope": "Exact referenced original only; does not extend V8 catalog/body/document limits",
        "reason": "Need independent previous disclosure before comparing profit forecast revision",
        "selection_basis": "Explicit reference in public correction; no prediction error or labels read",
        "acceptance": [
            "Full pagination within fixed single day",
            "Exactly one matching catalog item",
            "Issuer/title/date/notice number match",
            "No revision metadata conflict",
            "Byte and text replay",
            "No fit or production write",
        ],
        "stops": [
            "No retries",
            "No widened date interval",
            "No quoted-old-value substitution",
            "No budget increase",
            "No fitting",
        ],
        "code_hashes": {str(PYTHON / n): sha(PYTHON / n) for n in names},
        "source_hashes": {str(p): sha(p) for p in (source, ROOT / "supplement/cninfo-stock-map.json")},
    }
    save(OUT / "plan.json", plan)
    return {"limits": plan["limits"], "date": plan["window"]["start"], "new_fits": 0}


def check_plan():
    plan = read(OUT / "plan.json")
    for p, h in {**plan["code_hashes"], **plan["source_hashes"]}.items():
        if sha(p) != h:
            raise ValueError("FROZEN_REFERENCE_INPUT_CHANGED")
    return plan


def collect():
    plan = check_plan()
    if (OUT / "collection-stop.json").exists():
        raise ValueError("REFERENCE_ALREADY_STOPPED_USE_AUDIT")
    reader = BoundedPublicReader(OUT, plan["limits"])
    try:
        target = OUT / "catalog.json"
        if target.exists():
            catalog = read(target)
        else:
            stock = read(ROOT / "supplement/cninfo-stock-map.json")["rows"]["300853"]
            pages, receipts, number, total = [], [], 1, 1
            while number <= total:
                params = {
                    "pageNum": number,
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
                    "seDate": "2023-01-20~2023-01-20",
                    "isHLtitle": "true",
                }
                raw, receipt = reader.fetch(QUERY, params=params, group="catalog")
                response = json.loads(raw)
                total = max(1, (int(response["totalAnnouncement"]) + 29) // 30)
                if total > plan["maximum_catalog_pages"]:
                    raise ValueError("REFERENCE_PAGINATION_LIMIT")
                pages.append((number, response))
                receipts.append(receipt)
                number += 1
            rows = catalog_rows(pages, "300853", "2023-01-20", "2023-01-20")
            catalog = {"rows": rows, "receipts": receipts}
            save(target, catalog)
        row = select_reference(catalog["rows"], plan["reference"])
        target = OUT / "document.json"
        if not target.exists():
            raw, receipt = reader.fetch(
                "https://static.cninfo.com.cn/" + row["adjunctUrl"],
                group="body",
                maximum_bytes=plan["maximum_pdf_bytes"],
            )
            pages, metadata = extract_pdf(raw, plan["maximum_pdf_pages"])
            proof = reference_identity(row, pages, plan["reference"])
            issues = pdf_revision(metadata, row["published_date"])
            if issues:
                raise ValueError("REFERENCE_METADATA_DATE_CONFLICT")
            save(
                target,
                {
                    "row": row,
                    "pages": pages,
                    "metadata": metadata,
                    "receipt": receipt,
                    "identity": proof,
                    "revision_issues": issues,
                    "body_saved": True,
                },
            )
        save(
            OUT / "collection-stop.json",
            {"reason": "REGISTERED_REFERENCE_SOURCE_SAVED", "do_not_resume_collect": True, "new_fits": 0},
        )
    except ValueError as exc:
        save(
            OUT / "collection-stop.json",
            {"reason": str(exc), "passed": False, "do_not_resume_collect": True, "new_fits": 0},
        )
        raise
    finally:
        reader.close()
    return audit()


def audit():
    plan = check_plan()
    catalog, d = read(OUT / "catalog.json"), read(OUT / "document.json")
    for receipt in catalog["receipts"] + [d["receipt"]]:
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("REFERENCE_SOURCE_CHANGED")
    rows = catalog_rows(
        [(r["params"]["pageNum"], read(r["path"])) for r in catalog["receipts"]], "300853", "2023-01-20", "2023-01-20"
    )
    if rows != catalog["rows"] or select_reference(rows, plan["reference"]) != d["row"]:
        raise ValueError("REFERENCE_CATALOG_REPLAY_CHANGED")
    pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), plan["maximum_pdf_pages"])
    if (
        pages != d["pages"]
        or metadata != d["metadata"]
        or reference_identity(d["row"], pages, plan["reference"]) != d["identity"]
        or pdf_revision(metadata, d["row"]["published_date"]) != d["revision_issues"]
    ):
        raise ValueError("REFERENCE_BODY_REPLAY_CHANGED")
    used = dict(Counter(read(p)["group"] for p in (OUT / "requests").glob("*.json")))
    if any(n > plan["limits"][g] for g, n in used.items()):
        raise ValueError("REFERENCE_BUDGET_EXCEEDED")
    result = {
        "passed": True,
        "original_document_id": d["row"]["announcementId"],
        "published_date": d["row"]["published_date"],
        "catalog_rows": len(rows),
        "requests": used,
        "limits": plan["limits"],
        "new_fits": 0,
        "training_ready": False,
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
