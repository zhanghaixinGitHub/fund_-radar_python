"""第十八批：定向补两家公司被快报引用的 2018 三季报；零拟合。"""

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.services.fund_earnings_batch_v4 import PYTHON, RUNS, extract_pdf, issuer_identity
from app.services.fund_earnings_targeted_v1 import select_referenced_quarter
from app.services.fund_information_history_v1 import (
    OLD,
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

OUT = RUNS / "20260929-earnings-v18"
PREVIOUS = RUNS / "20260929-earnings-v17"
RESOURCE_STOP = RUNS / "20260929-earnings-v16/resource-stop.json"


def workspace():
    """只读记录三仓库的提交和工作区；不提交或改动已有文件。"""
    return [
        {
            "path": str(p),
            **{
                key: subprocess.check_output(["git", "-C", str(p), *args], text=True, encoding="utf-8").splitlines()
                for key, args in {"head": ["rev-parse", "HEAD"], "status": ["status", "--short"]}.items()
            },
        }
        for p in [Path("C:/WebStormProject/workSpace05"), PYTHON, Path("C:/ideaProject/workSpace12")]
    ]


def prepare():
    if (OUT / "plan.json").exists():
        return check_plan()
    preview = read(PREVIOUS / "next-targeted-source-preview.json")
    stops = read(PREVIOUS / "source-stop-registry.json")
    resource_rows = [p["row"] for p in read(RESOURCE_STOP)["positions"]]
    if any(
        w["stock"] == r["secCode"] and w["start"] <= r["published_date"] <= w["end"]
        for w in preview["windows"]
        for r in resource_rows
    ):
        raise ValueError("TARGET_WINDOW_INTERSECTS_OLD_RESOURCE_CAP")
    for entry in stops["entries"]:
        if any(
            w["stock"] == entry["stock"] and w["start"] <= entry["published_date"] <= w["end"]
            for w in preview["windows"]
        ):
            raise ValueError("TARGET_WINDOW_INTERSECTS_OLD_STOP")
    protected = dict(read(PREVIOUS / "protection-before.json")["files"])
    protected.update(read(PREVIOUS / "delivery-manifest.json")["files"])
    if any(sha(p) != h for p, h in protected.items()):
        raise ValueError("OLD_EVIDENCE_CHANGED")
    for p in PREVIOUS.rglob("*"):
        if p.is_file():
            protected.setdefault(str(p), sha(p))
    ledger = list((OLD / "fit-ledger").glob("*.json"))
    if len(ledger) != 12 or not all(read(p)["budget_consumed"] for p in ledger):
        raise ValueError("FIT_LEDGER_CHANGED")
    save(OUT / "protection-before.json", {"files": protected})
    save(
        OUT / "preflight.json",
        {
            "at": datetime.now(ZONE).isoformat(),
            "repositories": workspace(),
            "old_fit_ledger_files": {str(p): sha(p) for p in ledger},
            "new_fits": 0,
            "cumulative_fits": 70,
            "prior_stop_entries_preserved": len(stops["entries"]),
            "known_stop_collisions": [],
            "old_resource_positions": len(resource_rows),
            "resource_stop_collisions": [],
        },
    )
    code = dict(read(PREVIOUS / "semantic-review-plan.json")["code_hashes"])
    for name in [
        "scripts/fund_002112_earnings_targeted_v18.py",
        "app/services/fund_earnings_targeted_v1.py",
        "tests/test_fund_earnings_targeted_v1.py",
    ]:
        code[str(PYTHON / name)] = sha(PYTHON / name)
    plan = {
        "at": datetime.now(ZONE).isoformat(),
        "previous_run": str(PREVIOUS),
        "scope": "TWO_REFERENCED_ORIGINALS_ONLY_NOT_FULL_HISTORY",
        "windows": preview["windows"],
        "year": 2018,
        "searchkey": "",
        "limits": {"catalog": 6, "body": 4},
        "maximum_documents": 4,
        "maximum_pages_per_window": 3,
        "maximum_pdf_bytes": 25000000,
        "maximum_pdf_pages": 500,
        "new_fit_budget": 0,
        "cumulative_fits": 70,
        "old_protocol_unused_fits": 12,
        "selection_basis": preview["selection_basis"],
        "acceptance": preview["acceptance"],
        "code_hashes": code,
        "source_hashes": {
            str(p): sha(p)
            for p in [
                PREVIOUS / "next-targeted-source-preview.json",
                PREVIOUS / "source-stop-registry.json",
                PREVIOUS / "semantic-review-final.json",
                ROOT / "supplement/cninfo-stock-map.json",
                RESOURCE_STOP,
            ]
        },
        "catalog_coverage_credit": 0,
        "training_ready": False,
    }
    save(OUT / "plan.json", plan)
    return {"prepared": True, "windows": len(plan["windows"]), "limits": plan["limits"], "new_fits": 0}


def check_plan():
    plan = read(OUT / "plan.json")
    if any(sha(p) != h for p, h in {**plan["code_hashes"], **plan["source_hashes"]}.items()):
        raise ValueError("FROZEN_TARGETED_DEPENDENCY_CHANGED")
    return plan


def collect():
    plan = check_plan()
    if (OUT / "collection-stop.json").exists():
        raise ValueError("BATCH_STOPPED_USE_AUDIT")
    mapping = read(ROOT / "supplement/cninfo-stock-map.json")["rows"]
    stop_ids = {e["document_id"] for e in read(PREVIOUS / "source-stop-registry.json")["entries"]}
    stop_ids.update(p["row"]["announcementId"] for p in read(RESOURCE_STOP)["positions"])
    reader = BoundedPublicReader(OUT, plan["limits"])
    try:
        for window in plan["windows"]:
            stock = window["stock"]
            failure = OUT / "failures" / f"catalog-{stock}.json"
            if failure.exists():
                continue
            try:
                target = OUT / "catalogs" / f"{stock}.json"
                if not target.exists():
                    pages, receipts, number, total = [], [], 1, 1
                    while number <= total:
                        params = {
                            "pageNum": number,
                            "pageSize": 30,
                            "column": "szse",
                            "tabName": "fulltext",
                            "plate": "",
                            "stock": mapping[stock]["code"] + "," + mapping[stock]["orgId"],
                            "searchkey": "",
                            "secid": "",
                            "category": "",
                            "trade": "",
                            "sortName": "",
                            "sortType": "",
                            "seDate": window["start"] + "~" + window["end"],
                            "isHLtitle": "true",
                        }
                        raw, receipt = reader.fetch(QUERY, params=params, group="catalog")
                        data = json.loads(raw)
                        total = max(1, (int(data["totalAnnouncement"]) + 29) // 30)
                        if total > plan["maximum_pages_per_window"]:
                            raise ValueError("TARGET_CATALOG_PAGE_LIMIT")
                        pages.append((number, data))
                        receipts.append(receipt)
                        number += 1
                    rows = catalog_rows(pages, stock, window["start"], window["end"])
                    save(target, {"rows": rows, "receipts": receipts, "catalog_complete_in_window": True})
                rows = select_referenced_quarter(
                    read(target)["rows"], stock, plan["year"], window["referenced_public_date"]
                )
                if any(row["announcementId"] in stop_ids for row in rows):
                    raise ValueError("TARGET_ORIGINAL_INTERSECTS_OLD_STOP")
                save(OUT / "selections" / f"{stock}.json", rows)
            except ValueError as exc:
                save(failure, {"stock": stock, "reason": str(exc), "do_not_retry": True})
                continue
            for row in rows:
                key = row["announcementId"]
                target, failure = OUT / "documents" / f"{key}.json", OUT / "failures" / f"body-{key}.json"
                if target.exists() or failure.exists():
                    continue
                try:
                    if len(list((OUT / "documents").glob("*.json"))) >= plan["maximum_documents"]:
                        raise ValueError("TARGET_DOCUMENT_LIMIT")
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
                    save(
                        failure,
                        {"stock": stock, "document_id": key, "row": row, "reason": str(exc), "do_not_retry": True},
                    )
    finally:
        reader.close()
    save(
        OUT / "collection-stop.json",
        {"reason": "TARGETED_POSITIONS_ACCOUNTED", "do_not_resume_collect": True, "new_fits": 0},
    )
    return audit()


def audit():
    """独立进程从原始响应重算选择、身份、公开日和 PDF 字节；不发网络请求。"""
    plan = check_plan()
    selected, docs = [], []
    for window in plan["windows"]:
        stock = window["stock"]
        if (OUT / "failures" / f"catalog-{stock}.json").exists():
            continue
        catalog = read(OUT / "catalogs" / f"{stock}.json")
        for receipt in catalog["receipts"]:
            if sha(receipt["path"]) != receipt["sha256"]:
                raise ValueError("RAW_CATALOG_CHANGED")
        rows = catalog_rows(
            [(r["params"]["pageNum"], read(r["path"])) for r in catalog["receipts"]],
            stock,
            window["start"],
            window["end"],
        )
        if rows != catalog["rows"]:
            raise ValueError("CATALOG_REPLAY_CHANGED")
        selection = select_referenced_quarter(rows, stock, plan["year"], window["referenced_public_date"])
        if selection != read(OUT / "selections" / f"{stock}.json"):
            raise ValueError("SELECTION_REPLAY_CHANGED")
        selected.extend(selection)
    for row in selected:
        key = row["announcementId"]
        target = OUT / "documents" / f"{key}.json"
        if not target.exists():
            if not (OUT / "failures" / f"body-{key}.json").exists():
                raise ValueError("UNACCOUNTED_TARGET_BODY")
            continue
        d = read(target)
        if d["row"] != row or sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("TARGET_BODY_CHANGED")
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), plan["maximum_pdf_pages"])
        if (
            pages != d["pages"]
            or metadata != d["metadata"]
            or issuer_identity(row, pages) != d["identity"]
            or pdf_revision(metadata, row["published_date"]) != d["revision_issues"]
        ):
            raise ValueError("TARGET_BODY_REPLAY_CHANGED")
        docs.append(
            {
                "document_id": key,
                "stock": row["secCode"],
                "title": row["title_plain"],
                "published_date": row["published_date"],
                "identity_passed": d["identity"]["passed"],
                "revision_issues": d["revision_issues"],
                "pages": len(pages),
            }
        )
    used = dict(Counter(read(p)["group"] for p in (OUT / "requests").glob("*.json")))
    if any(count > plan["limits"][group] for group, count in used.items()) or len(docs) > plan["maximum_documents"]:
        raise ValueError("TARGET_BUDGET_EXCEEDED")
    result = {
        "replay_passed": True,
        "documents": docs,
        "requests": used,
        "limits": plan["limits"],
        "failures": [read(p) for p in sorted((OUT / "failures").glob("*.json"))],
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
