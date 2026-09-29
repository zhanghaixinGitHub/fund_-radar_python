"""登记第二批有限业绩历史补采；复用冻结 V1 工具，不修改 V1 文件或研究结果。"""

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pypdfium2 as pdfium
from app.services.fund_earnings_evidence_v1 import earnings_kind, plan_groups
from app.services.fund_earnings_history_v2 import identity, merge_windows
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

from scripts.fund_002112_earnings_batch_v1 import params_for

PREVIOUS = ROOT / "information-research/20260928-earnings-v1"
HISTORY = ROOT / "information-research/20260928-history-v1"
OUT = ROOT / "information-research/20260928-earnings-v2"
PYTHON = Path(__file__).resolve().parents[1]
CODE = [
    PYTHON / p
    for p in (
        "app/services/fund_earnings_history_v2.py",
        "scripts/fund_002112_earnings_batch_v2.py",
        "tests/test_fund_earnings_history_v2.py",
    )
]


def prepare():
    """继承原范围和预算边界，冻结新增公司/时间/请求上限，只有资料准备授权。"""
    if (OUT / "plan.json").exists():
        return read(OUT / "plan.json")
    protected = dict(read(PREVIOUS / "protection-before.json")["files"])
    for p, expected in read(PREVIOUS / "delivery-manifest.json")["files"].items():
        if p in protected and protected[p] != expected:
            raise ValueError("PREVIOUS_MANIFEST_CONFLICT")
        protected[p] = expected
    for p, expected in protected.items():
        if sha(p) != expected:
            raise ValueError("PROTECTED_SOURCE_CHANGED:" + p)
    for p in PREVIOUS.rglob("*"):
        if p.is_file():
            protected.setdefault(str(p), sha(p))
    ledger = list((OLD / "fit-ledger").glob("*.json"))
    if len(ledger) != 12:
        raise ValueError("FIT_LEDGER_COUNT_CHANGED")
    save(OUT / "protection-before.json", {"files": protected})
    rows = read(PREVIOUS / "row-coverage.json")
    plan = plan_groups(read(OLD / "event-admission/row-coverage.json"), rows, 4, 40)
    plan["unmerged_windows"] = plan["windows"]
    plan["windows"] = merge_windows(plan["windows"])
    plan.update(
        {
            "schema": "EARNINGS_HISTORY_PREPARATION_V2",
            "created_at": datetime.now(ZONE).isoformat(),
            "scope": "DATA_ONLY_NO_FIT",
            "previous_run": str(PREVIOUS),
            "prior_catalog_runs": [str(HISTORY), str(PREVIOUS)],
            "baseline_complete": sum(r["catalog_complete"] for r in rows),
            "fits_authorized_here": 0,
            "cumulative_fits": 70,
            "old_protocol_remaining_fits": 12,
            "limits": {"catalog": 150, "body": 100},
            "max_pages_per_window": 10,
            "max_pdf_bytes": 25_000_000,
            "max_pdf_pages": 500,
            "title_scope": "FORECAST, PRELIMINARY_RESULT, REPORTED_RESULT, CORRECTION_NOTICE via frozen earnings_kind",
            "document_order": "Forecast then preliminary then correction then reported result; date/id ascending",
            "semantic_scope": "Original unit/period/basis/YOY preservation; field review separate from identity and coverage",
            "selection_basis": "Whole-report completion per missing company window, no outcome selection",
            "stops": [
                "Bounded free CNINFO queries and PDFs only",
                "No retry of failed or interrupted request",
                "Wrong identity/date/pagination blocks affected window",
                "Unresolved correction blocks semantic use",
                "No training/adoption; old candidates remain stopped",
            ],
            "code_hashes": {str(p): sha(p) for p in CODE},
            "source_hashes": {
                str(p): sha(p)
                for p in (
                    PREVIOUS / "row-coverage.json",
                    OLD / "event-admission/row-coverage.json",
                    ROOT / "supplement/cninfo-stock-map.json",
                )
            },
        }
    )
    save(OUT / "plan.json", plan)
    return {"windows": len(plan["windows"]), "selected": plan["selected"], "limits": plan["limits"], "fits": 0}


def check_plan():
    plan = read(OUT / "plan.json")
    for p, expected in {**plan["source_hashes"], **plan["code_hashes"]}.items():
        if sha(p) != expected:
            raise ValueError("FROZEN_PLAN_DEPENDENCY_CHANGED:" + p)
    return plan


def reused_sources():
    """缓存只按已验证的同一 URL 复用；冲突不任选一份，不覆盖原件。"""
    values = defaultdict(list)
    for directory in (HISTORY, PREVIOUS):
        for path in (directory / "receipts").glob("*.json"):
            item = read(path)
            if item["ok"] and item.get("params") is None:
                values[item["url"]].append(item)
    # 初始事实库也有可复用原件，避免在后续窗口重下同一披露。
    for item in read(HISTORY / "semantic-inventory.json"):
        source = item["source"]
        values[source["url"]].append({"ok": True, **source})
    return values


def collect():
    plan, reused = check_plan(), reused_sources()
    stocks = read(ROOT / "supplement/cninfo-stock-map.json")["rows"]
    client = BoundedPublicReader(OUT, plan["limits"])
    try:
        for i, window in enumerate(plan["windows"], 1):
            target = OUT / "catalogs" / (digest(window) + ".json")
            if target.exists():
                continue
            pages, receipts = [], []
            try:
                page, maximum = 1, 1
                while page <= maximum:
                    raw, receipt = client.fetch(
                        QUERY, params=params_for(window, stocks[window["stock"]], page), group="catalog"
                    )
                    value = json.loads(raw)
                    receipts.append(receipt)
                    maximum = max(1, (int(value["totalAnnouncement"]) + 29) // 30)
                    if maximum > plan["max_pages_per_window"]:
                        raise ValueError("WINDOW_PAGE_LIMIT")
                    pages.append((page, value))
                    page += 1
                rows = catalog_rows(pages, window["stock"], window["start"], window["end"])
                result = {"window": window, "catalog_complete": True, "rows": rows, "receipts": receipts}
            except (ValueError, KeyError, TypeError) as exc:
                result = {"window": window, "catalog_complete": False, "reason": str(exc), "receipts": receipts}
            save(target, result)
            print(
                json.dumps(
                    {
                        "catalog": i,
                        "total": len(plan["windows"]),
                        "passed": result["catalog_complete"],
                        "rows": len(result.get("rows", [])),
                    }
                ),
                flush=True,
            )
        entries = {}
        for path in sorted((OUT / "catalogs").glob("*.json")):
            catalog = read(path)
            if catalog["catalog_complete"]:
                for row in catalog["rows"]:
                    kind = earnings_kind(row["title_plain"])
                    if kind:
                        previous = entries.get(row["announcementId"])
                        if previous and any(
                            previous[k] != row[k] for k in ("secCode", "adjunctUrl", "published_date", "title_plain")
                        ):
                            raise ValueError("DOCUMENT_IDENTITY_CONFLICT")
                        entries[row["announcementId"]] = {**row, "earnings_kind": kind}
        rank = {"FORECAST": 0, "PRELIMINARY_RESULT": 1, "CORRECTION_NOTICE": 2, "REPORTED_RESULT": 3}
        items = sorted(
            entries.values(), key=lambda r: (rank[r["earnings_kind"]], r["published_date"], r["announcementId"])
        )
        save(OUT / "body-worklist.json", items)
        for i, row in enumerate(items, 1):
            target = OUT / "documents" / (row["announcementId"] + ".json")
            if target.exists():
                continue
            try:
                if i > 100:
                    raise ValueError("DOCUMENT_COUNT_LIMIT")
                url = "https://static.cninfo.com.cn/" + row["adjunctUrl"]
                cached = reused.get(url, [])
                if cached:
                    if len({c["sha256"] for c in cached}) != 1:
                        raise ValueError("CACHED_SOURCE_VERSION_CONFLICT")
                    receipt = cached[0]
                    if sha(receipt["path"]) != receipt["sha256"]:
                        raise ValueError("CACHED_SOURCE_HASH_CHANGED")
                    raw = Path(receipt["path"]).read_bytes()
                else:
                    raw, receipt = client.fetch(url, group="body", maximum_bytes=plan["max_pdf_bytes"])
                if not raw.startswith(b"%PDF") or len(raw) > plan["max_pdf_bytes"]:
                    raise ValueError("INVALID_OR_OVERSIZED_PDF")
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
                result = {
                    "row": row,
                    "receipt": receipt,
                    "reused": bool(cached),
                    "pages": pages,
                    "metadata": metadata,
                    "body_saved": True,
                    "identity": identity(row, pages),
                    "revision_issues": pdf_revision(metadata, row["published_date"]),
                    "semantic_verified": False,
                }
            except (ValueError, pdfium.PdfiumError) as exc:
                result = {"row": row, "body_saved": False, "reason": str(exc), "semantic_verified": False}
            save(target, result)
            print(
                json.dumps(
                    {
                        "body": i,
                        "total": len(items),
                        "saved": result["body_saved"],
                        "reused": result.get("reused", False),
                    }
                ),
                flush=True,
            )
            if result.get("reason", "").startswith(("PUBLIC_REQUEST_LIMIT", "DOCUMENT_COUNT_LIMIT")):
                break
    finally:
        client.close()
    return audit()


def audit():
    """从原始分页复核目录，累计前批覆盖；独立重开每份 PDF 验原件与身份。"""
    plan = check_plan()
    for p, expected in read(OUT / "protection-before.json")["files"].items():
        if sha(p) != expected:
            raise ValueError("PROTECTED_FILE_CHANGED:" + p)
    requests = Counter(read(p)["group"] for p in (OUT / "requests").glob("*.json"))
    if any(v > plan["limits"][k] for k, v in requests.items()):
        raise ValueError("REQUEST_LIMIT_EXCEEDED")
    receipts = [read(p) for p in (OUT / "receipts").glob("*.json")]
    for r in receipts:
        if r["ok"] and sha(r["path"]) != r["sha256"]:
            raise ValueError("NEW_RAW_CHANGED")
    bundle = read(read(OLD / "event-admission/bundle-manifest.json")["path"])
    intervals = {k: list(v) for k, v in bundle["coverage"]["company"].items()}
    catalogs = [read(p) for p in (OUT / "catalogs").glob("*.json")]
    for c in catalogs:
        if c["catalog_complete"]:
            pages = [(r["params"]["pageNum"], read(r["path"])) for r in c["receipts"]]
            w = c["window"]
            if catalog_rows(pages, w["stock"], w["start"], w["end"]) != c["rows"]:
                raise ValueError("PAGINATION_REPLAY_MISMATCH")
    prior = [read(p) for p in (HISTORY / "company-catalogs").glob("*.json")]
    prior += [read(p) for p in (PREVIOUS / "catalogs").glob("*.json")]
    for c in prior + catalogs:
        if c["catalog_complete"]:
            w = c["window"]
            intervals.setdefault(w["stock"], []).append([w["start"], w["end"]])
    rows, gaps = [], defaultdict(list)
    for row in read(OLD / "event-admission/row-coverage.json"):
        lo, hi = [str(date.fromisoformat(row["target"]) - timedelta(days=n)) for n in (30, 1)]
        missing = [s for s in row["missing_companies"] if not covered(intervals.get(s, []), lo, hi)]
        rows.append(
            {
                "fund_code": row["fund_code"],
                "target": row["target"],
                "catalog_complete": row["company_coverage_passed"] or (bool(row["missing_companies"]) and not missing),
                "remaining_companies": missing,
            }
        )
        for stock in missing:
            gaps[stock].append([lo, hi])
    save(OUT / "row-coverage.json", rows)
    save(
        OUT / "company-gap-worklist.json",
        [{"stock": s, "intervals_unmerged": spans} for s, spans in sorted(gaps.items())],
    )
    before = {(r["fund_code"], r["target"]): r for r in read(PREVIOUS / "row-coverage.json")}
    after = {(r["fund_code"], r["target"]): r for r in rows}
    folds = []
    for fold in read(OLD / "early-admission/folds.json"):
        keys = [tuple(k) for k in fold["train_ids"]]
        folds.append(
            {
                "name": fold["name"],
                "train_rows": len(keys),
                "before_complete": sum(before[k]["catalog_complete"] for k in keys),
                "after_complete": sum(after[k]["catalog_complete"] for k in keys),
            }
        )
    save(OUT / "coverage-on-new-folds.json", {"folds": folds, "training_ready": False})
    documents = [read(p) for p in sorted((OUT / "documents").glob("*.json"))]
    identities, hints = [], []
    for item in documents:
        if not item["body_saved"]:
            continue
        if sha(item["receipt"]["path"]) != item["receipt"]["sha256"]:
            raise ValueError("BODY_SOURCE_CHANGED")
        with pdfium.PdfDocument(item["receipt"]["path"]) as document:
            pages = []
            for page in document:
                text = page.get_textpage()
                try:
                    pages.append(text.get_text_range())
                finally:
                    text.close()
                    page.close()
        if [normalize(p) for p in pages] != [normalize(p) for p in item["pages"]]:
            raise ValueError("BODY_REPLAY_MISMATCH")
        proof = identity(item["row"], pages)
        if proof != item["identity"]:
            raise ValueError("IDENTITY_REPLAY_MISMATCH")
        identities.append({"id": item["row"]["announcementId"], **proof})
        # 仅列正文候选页，不能以关键词出现就当有效预告或数值审核通过。
        selected_pages = [
            i + 1 for i, text in enumerate(pages) if "业绩预告" in text or "经营业绩的预计" in normalize(text)
        ]
        if selected_pages:
            hints.append(
                {
                    "id": item["row"]["announcementId"],
                    "stock": item["row"]["secCode"],
                    "title": item["row"]["title_plain"],
                    "pages": selected_pages,
                    "status": "UNREVIEWED_HINT",
                }
            )
    save(OUT / "identity-audit.json", identities)
    save(OUT / "body-forecast-hints.json", hints)
    result = {
        "new_fits": 0,
        "cumulative_fits": 70,
        "old_fit_ledger_entries": len(list((OLD / "fit-ledger").glob("*.json"))),
        "protected_files": len(read(OUT / "protection-before.json")["files"]),
        "requests": dict(requests),
        "request_limits": plan["limits"],
        "catalog_windows_passed": sum(c["catalog_complete"] for c in catalogs),
        "catalog_windows_attempted": len(catalogs),
        "catalog_rows": sum(len(c.get("rows", [])) for c in catalogs),
        "baseline_complete": plan["baseline_complete"],
        "catalog_complete": sum(r["catalog_complete"] for r in rows),
        "by_fund": dict(Counter(r["fund_code"] for r in rows if r["catalog_complete"])),
        "body_worklist": len(read(OUT / "body-worklist.json")),
        "bodies_saved": len(identities),
        "reused_bodies": sum(d.get("reused", False) for d in documents),
        "identity_passed": sum(i["passed"] for i in identities),
        "metadata_conflicts": sum(bool(d.get("revision_issues")) for d in documents),
        "remaining_gap_companies": len(gaps),
        "folds": folds,
        "training_ready": False,
    }
    if result["old_fit_ledger_entries"] != 12:
        raise ValueError("FIT_LEDGER_COUNT_CHANGED")
    save(OUT / "collection-and-audit-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "audit"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "information-research/.earnings-preparation.lock"
    owner = {"pid": os.getpid(), "command": args.command, "directory": str(OUT), "at": datetime.now(ZONE).isoformat()}
    save_lock = lock.open("x", encoding="utf-8")
    try:
        with save_lock:
            json.dump(owner, save_lock)
        print(
            json.dumps(
                {"prepare": prepare, "collect": collect, "audit": audit}[args.command](), ensure_ascii=False, indent=2
            )
        )
    finally:
        if read(lock) == owner:
            lock.unlink()


if __name__ == "__main__":
    main()
