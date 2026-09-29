"""登记并补齐完整目录范围内的公司事实原件；保持旧停止记录与三层准入分离。"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pypdfium2 as pdfium
from app.services.fund_earnings_batch_v4 import extract_pdf, issuer_identity
from app.services.fund_earnings_titles_v3 import classify_catalog_row
from app.services.fund_information_history_v1 import (
    ROOT,
    BoundedPublicReader,
    buyback_candidates,
    catalog_rows,
    pdf_revision,
    read,
    save,
    sha,
)

from scripts.fund_002112_closure_v1 import OUT

DEST = OUT / "company-bodies"


def prepare():
    """冻结全部已完整目录中的业绩、回购、重大合同原件，不再按小报告组择优取样。"""
    if (DEST / "plan.json").exists():
        return read(DEST / "plan.json")["summary"]
    if not (OUT / "catalog-reconciliation/result.json").exists():
        raise ValueError("WAIT_FOR_CATALOG_COLLECTION_END")
    files = [OUT / "catalog-result.json", OUT / "catalog-reconciliation/result.json"]
    files += list((ROOT / "information-research").glob("*/catalogs/*.json"))
    files += list((OUT / "catalogs").glob("*.json"))
    files += list((OUT / "catalog-reconciliation/catalogs").glob("*.json"))
    rows, variants = {}, defaultdict(list)
    for path in files[1:]:
        catalog = read(path)
        if not catalog.get("catalog_complete"):
            continue
        for row in catalog["rows"]:
            if row["published_date"] > "2023-12-31":
                continue
            key = row["announcementId"]
            identity = tuple(row[k] for k in ("secCode", "published_date", "title_plain", "adjunctUrl"))
            if identity not in variants[key]:
                variants[key].append(identity)
            rows[key] = row
    # 最早一批使用另一种目录封装。回放原始分页后合并，避免只补新批次而遗漏旧覆盖。
    for path in (ROOT / "supplement/company-announcements").glob("*.json"):
        catalog = read(path)
        grouped = defaultdict(list)
        files.append(path)
        for receipt in catalog.get("receipts", []):
            lo, hi = receipt["params"]["seDate"].split("~")
            if lo > "2023-12-31":
                continue
            raw_path = ROOT / receipt["file"]
            if sha(raw_path) != receipt["sha256"]:
                raise ValueError("ORIGINAL_CATALOG_CHANGED")
            files.append(raw_path)
            grouped[(lo, hi)].append((receipt["params"]["pageNum"], read(raw_path)))
        for (lo, hi), pages in grouped.items():
            try:
                verified = catalog_rows(sorted(pages), catalog["code"].split(".")[0], lo, hi)
            except ValueError:
                # 旧失败分页保持失败；本计划只登记能够从原始响应重放的记录。
                continue
            for row in verified:
                if row["published_date"] > "2023-12-31":
                    continue
                key = row["announcementId"]
                identity = tuple(row[k] for k in ("secCode", "published_date", "title_plain", "adjunctUrl"))
                if identity not in variants[key]:
                    variants[key].append(identity)
                rows[key] = row
    cached = defaultdict(list)
    failures = set()
    receipt_files = list((ROOT / "information-research").glob("*/receipts/*.json"))
    for path in receipt_files:
        receipt = read(path)
        if receipt.get("ok") and receipt.get("params") is None and receipt["url"].endswith((".PDF", ".pdf")):
            cached[receipt["url"]].append(receipt)
        elif not receipt.get("ok"):
            request_path = path.parent.parent / "requests" / path.name
            if request_path.exists():
                failures.add(read(request_path)["url"])
    inventory = ROOT / "information-research/20260928-history-v1/semantic-inventory.json"
    files.append(inventory)
    for item in read(inventory):
        source = item["source"]
        cached[source["url"]].append({"ok": True, **source})
    registry = ROOT / "information-research/20260929-earnings-v24/source-stop-registry.json"
    files.append(registry)
    stops = {e["document_id"]: e for e in read(registry)["entries"]}
    items = []
    for key, row in sorted(rows.items()):
        kind = classify_catalog_row(row)["kind"]
        category = "PERFORMANCE" if kind else row.get("category")
        if not kind and category not in {"BUYBACK", "MAJOR_CONTRACT"}:
            continue
        url = "https://static.cninfo.com.cn/" + row["adjunctUrl"]
        copies = cached.get(url, [])
        reasons = []
        if len(variants[key]) != 1:
            reasons.append("CATALOG_METADATA_VARIANTS_REQUIRE_REVIEW")
        if key in stops:
            reasons.append("PREVIOUS_SOURCE_STOP_PRESERVED")
        if url in failures:
            reasons.append("PREVIOUS_REQUEST_FAILURE_NO_RETRY")
        if len({r["sha256"] for r in copies}) > 1:
            reasons.append("CACHED_VERSION_CONFLICT")
        items.append({"row": row, "category": category, "earnings_kind": kind, "cached": copies[:1], "stops": reasons})
    requests = sum(not i["cached"] and not i["stops"] for i in items)
    plan = {
        "scope": "COMPANY_EARNINGS_BUYBACK_MAJOR_CONTRACT_ORIGINALS_ONLY",
        "items": items,
        "maximum_new_requests": requests,
        "maximum_pdf_bytes": 16_000_000,
        "maximum_pdf_pages": 500,
        "maximum_saved_bytes": 8_000_000_000,
        "source_hashes": {str(p): sha(p) for p in files + receipt_files},
        "code_hashes": {str(Path(__file__).resolve()): sha(__file__)},
        "summary": {
            "entries": len(rows),
            "selected_originals": len(items),
            "cached": sum(bool(i["cached"]) for i in items),
            "stopped": sum(bool(i["stops"]) for i in items),
            "maximum_new_requests": requests,
            "categories": dict(Counter(i["category"] for i in items)),
        },
        "no_fit_or_adoption": True,
        "no_policy_news_completeness_claim": True,
    }
    save(DEST / "plan.json", plan)
    return plan["summary"]


def collect():
    """逐份保留原件、全文、身份及日期检查；任何未知仍为未知，不生成模型数值。"""
    plan = read(DEST / "plan.json")
    for path, expected in {**plan["source_hashes"], **plan["code_hashes"]}.items():
        if sha(path) != expected:
            raise ValueError("FROZEN_SOURCE_CHANGED")
    reader = BoundedPublicReader(DEST, {"body": plan["maximum_new_requests"]})
    consecutive_failures = 0
    saved_bytes = sum(p.stat().st_size for p in (DEST / "raw").glob("*"))
    try:
        for number, item in enumerate(plan["items"], 1):
            row = item["row"]
            path = DEST / "documents" / (row["announcementId"] + ".json")
            if path.exists():
                continue
            result = {"row": row, "category": item["category"], "body_saved": False, "training_ready": False}
            try:
                if item["stops"]:
                    raise ValueError(";".join(item["stops"]))
                if saved_bytes >= plan["maximum_saved_bytes"]:
                    raise ValueError("TOTAL_SAVED_SIZE_LIMIT")
                if item["cached"]:
                    receipt = item["cached"][0]
                    if sha(receipt["path"]) != receipt["sha256"]:
                        raise ValueError("CACHED_ORIGINAL_CHANGED")
                    raw = Path(receipt["path"]).read_bytes()
                else:
                    raw, receipt = reader.fetch(
                        "https://static.cninfo.com.cn/" + row["adjunctUrl"],
                        group="body",
                        maximum_bytes=min(plan["maximum_pdf_bytes"], plan["maximum_saved_bytes"] - saved_bytes),
                    )
                    saved_bytes += len(raw)
                pages, metadata = extract_pdf(raw, plan["maximum_pdf_pages"])
                result.update(
                    body_saved=True,
                    receipt=receipt,
                    reused=bool(item["cached"]),
                    pages=pages,
                    metadata=metadata,
                    identity=issuer_identity(row, pages),
                    revision_issues=pdf_revision(metadata, row["published_date"]),
                    buyback_candidates=buyback_candidates(pages) if item["category"] == "BUYBACK" else [],
                    semantic_verified=False,
                )
                consecutive_failures = 0
            except (ValueError, pdfium.PdfiumError) as exc:
                result["reason"] = str(exc)
                if str(exc).startswith("PUBLIC_READ_STOP"):
                    consecutive_failures += 1
            save(path, result)
            if number % 25 == 0:
                print(
                    json.dumps({"body": number, "total": len(plan["items"]), "saved": result["body_saved"]}), flush=True
                )
            if consecutive_failures >= 3 or result.get("reason") == "TOTAL_SAVED_SIZE_LIMIT":
                save(DEST / "hard-stop.json", {"reason": result["reason"], "ordinal": number})
                break
    finally:
        reader.close()
    docs = [read(p) for p in (DEST / "documents").glob("*.json")]
    result = {
        "planned": len(plan["items"]),
        "processed": len(docs),
        "body_saved": sum(d["body_saved"] for d in docs),
        "identity_passed": sum(d.get("identity", {}).get("passed", False) for d in docs),
        "revision_conflicts": sum(bool(d.get("revision_issues")) for d in docs),
        "failures": dict(Counter(d.get("reason") for d in docs if not d["body_saved"])),
        "body_or_semantic_completeness_established": False,
        "new_fits": 0,
    }
    save(DEST / "collection-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "collect"])
    args = parser.parse_args()
    lock = OUT / ".company-bodies.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        print(json.dumps(prepare() if args.command == "prepare" else collect(), ensure_ascii=False), flush=True)
    finally:
        if lock.read_text(encoding="utf-8") == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
