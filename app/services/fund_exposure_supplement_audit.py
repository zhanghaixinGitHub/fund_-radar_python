"""补充数据覆盖核对：以落地文件与原文校验为准，不把计划数量当完成数量。"""

import json
from collections import Counter

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, now, read, save
from app.services.fund_exposure_quotes import reports
from app.services.fund_exposure_supplement import (
    FINANCIAL_APIS,
    OPERATING_APIS,
    SUPPLEMENT,
    plan,
    verified_bytes,
)


def audit_supplement():
    """逐类核对 469 家公司、文件及来源哈希；扫描件只算原文已取得，不算文字提取完整。"""
    scope = plan()
    checked_raw, failures = set(), []

    def verify(receipt):
        if receipt["file"] not in checked_raw:
            verified_bytes(receipt)
            checked_raw.add(receipt["file"])

    financial = {}
    for api in FINANCIAL_APIS + OPERATING_APIS:
        summary = {
            "companies": 0,
            "rows": 0,
            "distinct_rows": 0,
            "future_rows_quarantined": 0,
            "full_field_companies": 0,
            "additional_fields": [],
            "additional_fields_nonempty_rows": {},
            "empty_codes": [],
            "missing_codes": [],
        }
        for code in scope["codes"]:
            path = SUPPLEMENT / "financials" / code / (api + ".json")
            enriched = SUPPLEMENT / "enriched-financials" / code / (api + ".json")
            if enriched.exists():
                # “接口支持这个字段”不等于这批公司都有值，分别统计实际返回的非空数量。
                original = read(path)
                # 默认快照仍需逐段验收；统计时选完整字段版本，但不能因此跳过旧证据完整性。
                for receipt in original.get("all_receipts", [original["receipt"]]):
                    verify(receipt)
                default_fields = json.loads(verified_bytes(original["receipt"]))["data"]["fields"]
                path = enriched
            if not path.exists():
                summary["missing_codes"].append(code)
                continue
            value = read(path)
            if value["training_eligible"] is not False:
                raise ValueError("EXPOSURE_SUPPLEMENT_UNEXPECTED_TRAINING_ELIGIBILITY")
            for receipt in value.get("all_receipts", [value["receipt"]]):
                verify(receipt)
            summary["companies"] += 1
            summary["rows"] += len(value["rows"])
            summary["distinct_rows"] += len({digest(row) for row in value["rows"]})
            summary["future_rows_quarantined"] += len(value["future_rows"])
            if enriched.exists():
                extra_fields = sorted(set(value["fields"]) - set(default_fields))
                summary["full_field_companies"] += 1
                summary["additional_fields"] = sorted(set(summary["additional_fields"]) | set(extra_fields))
                for field in extra_fields:
                    counts = summary["additional_fields_nonempty_rows"]
                    counts[field] = counts.get(field, 0) + sum(
                        row.get(field) is not None and row.get(field) != "" for row in value["rows"]
                    )
            if not value["rows"]:
                summary["empty_codes"].append(code)
        financial[api] = summary
    announcements, ids = {}, set()
    for path in (SUPPLEMENT / "company-announcements").glob("*.json"):
        value = read(path)
        for receipt in value["receipts"]:
            verify(receipt)
        expected = 0
        for receipt in value["receipts"]:
            if receipt["params"]["pageNum"] == 1:
                expected += json.loads(verified_bytes(receipt))["totalAnnouncement"]
        if expected != len(value["rows"]):
            failures.append({"code": path.stem, "reason": "ANNOUNCEMENT_COUNT_MISMATCH"})
        announcements[path.stem] = len(value["rows"])
        ids.update(r["announcementId"] for r in value["rows"])
    document_ids, text_status, raw_bytes, total_pages = set(), Counter(), 0, 0
    for path in (SUPPLEMENT / "company-documents").glob("*.json"):
        value = read(path)
        verify(value["receipt"])
        document_ids.add(value["announcement_id"])
        text_status[value["text_status"]] += 1
        raw_bytes += (ROOT / value["receipt"]["file"]).stat().st_size
        total_pages += len(value["pages"])
    public = read(SUPPLEMENT / "public-result.json")
    sources, relevance, categories, fresh_hashes = Counter(), Counter(), Counter(), set()
    for entry in public["documents"]:
        value = read(ROOT / entry["file"])
        verify(value["receipt"])
        sources[value["receipt"]["source_code"]] += 1
        relevance[value["relevance"]] += 1
        categories.update(value["scopes"])
        fresh_hashes.add(value["receipt"]["sha256"])
    recovery = read(SUPPLEMENT / "public-recovery.json")
    available_public_ids = {entry["content_id"] for entry in public["documents"]}
    external_ids = {entry["content_id"] for entry in recovery["external"]}
    nav = read(SUPPLEMENT / "official-nav.json")
    for receipt in nav["receipts"]:
        verify(receipt)
    old_report_hashes = {r["raw"]["sha256"] for r in reports()}
    result = {
        "at": now().isoformat(),
        "cutoff": scope["end_date"],
        "financial": financial,
        "financial_total_rows": sum(r["rows"] for r in financial.values()),
        "company_announcements": {
            "companies": len(announcements),
            "rows": sum(announcements.values()),
            "unique_ids": len(ids),
            "raw_documents": len(document_ids),
            "raw_bytes": raw_bytes,
            "text_status": dict(text_status),
            "pages": total_pages,
            "missing_document_ids": sorted(ids - document_ids),
            "unmatched_document_ids": sorted(document_ids - ids),
        },
        "fund_public": {
            "catalog": public["catalog_count"],
            "saved": len(public["documents"]),
            "sources": dict(sources),
            "relevance": dict(relevance),
            "categories": dict(categories),
            "errors": public["errors"],
            "external_links": len(recovery["external"]),
            "external_text_unavailable": sum(r["status"] != "TEXT_AVAILABLE" for r in recovery["external"]),
            "external_links_with_public_mirror": len(external_ids & available_public_ids),
            "catalog_without_saved_text": public["catalog_count"] - len(available_public_ids),
            "external_ids_without_saved_text": sorted(external_ids - available_public_ids),
        },
        "official_nav": {
            "rows": len(nav["rows"]),
            "first": nav["rows"][0]["date"],
            "latest": nav["rows"][-1]["date"],
            "latest_unit_nav": nav["rows"][-1]["unit_nav"],
        },
        "old_report_raw_versions_reconfirmed": len(old_report_hashes & fresh_hashes),
        "old_report_raw_versions_not_in_new_public_documents": sorted(old_report_hashes - fresh_hashes),
        "raw_files_verified": len(checked_raw),
        "failures": failures,
        "new_purchase_cny": 0,
        "historical_first_publication_verified": False,
        "training_eligible": False,
        "long_term_auto_refresh": "EXISTING_QUOTES_REPORTS_AND_OFFICIAL_NAV_ONLY",
    }
    save(SUPPLEMENT / "audit.json", result, replace=True)
    return {
        "at": result["at"],
        "financial_rows": result["financial_total_rows"],
        "announcement_rows": sum(announcements.values()),
        "raw_documents": len(document_ids),
        "missing_documents": len(ids - document_ids),
        "failures": failures,
        "raw_files_verified": len(checked_raw),
    }
