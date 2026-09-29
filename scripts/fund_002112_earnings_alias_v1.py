"""在同批预算内补齐真实标题遗漏，保留原分类和初核结果，不扩大公司或时间。"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from app.services.fund_earnings_history_v2 import identity
from app.services.fund_earnings_semantics_v2 import earnings_title_kind
from app.services.fund_information_history_v1 import BoundedPublicReader, pdf_revision, read, save, sha

from scripts.fund_002112_earnings_batch_v2 import OUT, PYTHON, ROOT, check_plan, reused_sources

CODE = [
    PYTHON / p
    for p in (
        "app/services/fund_earnings_semantics_v2.py",
        "scripts/fund_002112_earnings_alias_v1.py",
        "tests/test_fund_earnings_semantics_v2.py",
    )
]


def prepare():
    """精确列出增补原件与排除项，理由来自公开标题，与预测结果无关。"""
    if (OUT / "title-amendment-v1.json").exists():
        return read(OUT / "title-amendment-v1.json")
    check_plan()
    existing = {r["announcementId"] for r in read(OUT / "body-worklist.json")}
    additions, excluded = {}, {}
    for path in sorted((OUT / "catalogs").glob("*.json")):
        catalog = read(path)
        if not catalog["catalog_complete"]:
            continue
        for r in catalog["rows"]:
            kind = earnings_title_kind(r["title_plain"])
            if kind and r["announcementId"] not in existing:
                additions[r["announcementId"]] = {**r, "earnings_kind": kind}
            elif not kind and r["announcementId"] in existing:
                excluded[r["announcementId"]] = {"row": r, "reason": "SPONSOR_REPORT_NOT_COMPANY_EARNINGS"}
    if len(existing) + len(additions) > 100:
        raise ValueError("SAME_BATCH_DOCUMENT_LIMIT")
    amendment = {
        "scope": "DATA_TITLE_ADMISSION_ONLY_NO_FIT",
        "parent_plan_sha256": sha(OUT / "plan.json"),
        "original_result_sha256": sha(OUT / "collection-and-audit-result.json"),
        "reason": "Observed pre-increase titles missed; sponsor annual report incorrectly selected",
        "added": sorted(additions.values(), key=lambda r: (r["published_date"], r["announcementId"])),
        "excluded_from_company_earnings": list(excluded.values()),
        "same_companies_and_windows": True,
        "request_limits_unchanged": True,
        "fits_authorized_here": 0,
        "code_hashes": {str(p): sha(p) for p in CODE},
    }
    save(OUT / "title-amendment-v1.json", amendment)
    return {"additions": len(additions), "supporting_documents": len(excluded), "request_limits_unchanged": True}


def collect():
    plan, amendment = check_plan(), read(OUT / "title-amendment-v1.json")
    if sha(OUT / "plan.json") != amendment["parent_plan_sha256"]:
        raise ValueError("AMENDMENT_PARENT_CHANGED")
    for p, expected in amendment["code_hashes"].items():
        if sha(p) != expected:
            raise ValueError("AMENDMENT_CODE_CHANGED")
    cached = reused_sources()
    reader = BoundedPublicReader(OUT, plan["limits"])
    try:
        for row in amendment["added"]:
            path = OUT / "additional-documents" / (row["announcementId"] + ".json")
            if path.exists():
                continue
            try:
                url = "https://static.cninfo.com.cn/" + row["adjunctUrl"]
                previous = cached.get(url, [])
                if previous:
                    if len({r["sha256"] for r in previous}) != 1:
                        raise ValueError("CACHED_SOURCE_CONFLICT")
                    receipt = previous[0]
                    if sha(receipt["path"]) != receipt["sha256"]:
                        raise ValueError("CACHED_SOURCE_CHANGED")
                    raw = Path(receipt["path"]).read_bytes()
                else:
                    raw, receipt = reader.fetch(url, group="body", maximum_bytes=plan["max_pdf_bytes"])
                if not raw.startswith(b"%PDF") or len(raw) > plan["max_pdf_bytes"]:
                    raise ValueError("INVALID_PDF_BYTES")
                with pdfium.PdfDocument(raw) as doc:
                    if len(doc) > plan["max_pdf_pages"]:
                        raise ValueError("PDF_PAGE_LIMIT")
                    pages, metadata = [], doc.get_metadata_dict()
                    for page in doc:
                        text = page.get_textpage()
                        try:
                            pages.append(text.get_text_range())
                        finally:
                            text.close()
                            page.close()
                value = {
                    "row": row,
                    "receipt": receipt,
                    "reused": bool(previous),
                    "body_saved": True,
                    "pages": pages,
                    "metadata": metadata,
                    "identity": identity(row, pages),
                    "revision_issues": pdf_revision(metadata, row["published_date"]),
                    "semantic_verified": False,
                }
            except (ValueError, pdfium.PdfiumError) as exc:
                value = {"row": row, "body_saved": False, "reason": str(exc), "semantic_verified": False}
            save(path, value)
    finally:
        reader.close()
    used = Counter(read(p)["group"] for p in (OUT / "requests").glob("*.json"))
    if any(v > plan["limits"][k] for k, v in used.items()):
        raise ValueError("SAME_BATCH_LIMIT_EXCEEDED")
    value = {
        "requests": dict(used),
        "new_fits": 0,
        "requests_within_original_limits": True,
        "additional_bodies": len(list((OUT / "additional-documents").glob("*.json"))),
    }
    save(OUT / "title-amendment-result.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect"))
    args = parser.parse_args()
    lock = ROOT / "information-research/.earnings-preparation.lock"
    owner = {"pid": os.getpid(), "command": "title-amendment-" + args.command}
    with lock.open("x", encoding="utf-8") as f:
        json.dump(owner, f)
    try:
        print(json.dumps({"prepare": prepare, "collect": collect}[args.command](), ensure_ascii=False, indent=2))
    finally:
        if read(lock) == owner:
            lock.unlink()


if __name__ == "__main__":
    main()
