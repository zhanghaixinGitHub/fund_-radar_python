"""第二批最终只读复验：保留增补前审计，用新文件合并预算、覆盖和语义审核。"""

import json
from collections import Counter, defaultdict
from datetime import date, timedelta

from app.services.fund_earnings_evidence_v1 import compare_claims, plan_groups, review_claim
from app.services.fund_earnings_history_v2 import review_yoy
from app.services.fund_earnings_semantics_v2 import review_table_yoy
from app.services.fund_information_history_v1 import OLD, catalog_rows, digest, normalize, read, save, sha

from scripts.fund_002112_earnings_batch_v2 import HISTORY, OUT, PREVIOUS, check_plan
from scripts.fund_002112_earnings_review_v1 import extract


def days(lo, hi):
    first, last = date.fromisoformat(lo), date.fromisoformat(hi)
    return {str(first + timedelta(days=n)) for n in range((last - first).days + 1)}


def run():
    plan = check_plan()
    protected = read(OUT / "protection-before.json")["files"]
    for p, expected in protected.items():
        if sha(p) != expected:
            raise ValueError("PROTECTED_EVIDENCE_CHANGED")
    amendment = read(OUT / "title-amendment-v1.json")
    for p, expected in amendment["code_hashes"].items():
        if sha(p) != expected:
            raise ValueError("AMENDMENT_CODE_CHANGED")
    if sha(OUT / "collection-and-audit-result.json") != amendment["original_result_sha256"]:
        raise ValueError("ORIGINAL_BATCH_AUDIT_CHANGED")
    ledger = list((OLD / "fit-ledger").glob("*.json"))
    if len(ledger) != 12:
        raise ValueError("FIT_LEDGER_CHANGED")
    used = Counter(read(p)["group"] for p in (OUT / "requests").glob("*.json"))
    if any(count > plan["limits"][group] for group, count in used.items()):
        raise ValueError("TOTAL_BATCH_REQUEST_BUDGET_EXCEEDED")
    for p in (OUT / "receipts").glob("*.json"):
        receipt = read(p)
        if receipt["ok"] and sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("PUBLIC_SOURCE_CHANGED")
    paths = sorted((OUT / "catalogs").glob("*.json"))
    if {p.stem for p in paths} != {digest(w) for w in plan["windows"]}:
        raise ValueError("CATALOG_SCOPE_MISMATCH")
    current = [read(p) for p in paths]
    for c in current:
        if c["catalog_complete"]:
            w = c["window"]
            pages = [(r["params"]["pageNum"], read(r["path"])) for r in c["receipts"]]
            if catalog_rows(pages, w["stock"], w["start"], w["end"]) != c["rows"]:
                raise ValueError("RAW_CATALOG_REPLAY_MISMATCH")
    bundle = read(read(OLD / "event-admission/bundle-manifest.json")["path"])
    ranges = {k: list(v) for k, v in bundle["coverage"]["company"].items()}
    catalogs = current + [read(p) for p in (HISTORY / "company-catalogs").glob("*.json")]
    catalogs += [read(p) for p in (PREVIOUS / "catalogs").glob("*.json")]
    for c in catalogs:
        if c["catalog_complete"]:
            w = c["window"]
            ranges.setdefault(w["stock"], []).append([w["start"], w["end"]])
    complete_days = {s: set().union(*(days(lo, hi) for lo, hi in spans)) for s, spans in ranges.items()}
    expected_rows, missing = [], defaultdict(list)
    for r in read(OLD / "event-admission/row-coverage.json"):
        lo, hi = [str(date.fromisoformat(r["target"]) - timedelta(days=n)) for n in (30, 1)]
        needed = days(lo, hi)
        gaps = [s for s in r["missing_companies"] if not needed.issubset(complete_days.get(s, set()))]
        expected_rows.append(
            {
                "fund_code": r["fund_code"],
                "target": r["target"],
                "catalog_complete": r["company_coverage_passed"] or (bool(r["missing_companies"]) and not gaps),
                "remaining_companies": gaps,
            }
        )
        for stock in gaps:
            missing[stock].append((lo, hi))
    if expected_rows != read(OUT / "row-coverage.json"):
        raise ValueError("INDEPENDENT_COVERAGE_MISMATCH")
    merged_gaps = []
    for stock, spans in sorted(missing.items()):
        merged = []
        for lo, hi in sorted(spans):
            if merged and lo <= str(date.fromisoformat(merged[-1][1]) + timedelta(days=1)):
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        merged_gaps.extend({"stock": stock, "start": lo, "end": hi} for lo, hi in merged)
    save(OUT / "company-gap-worklist-merged.json", merged_gaps)
    candidates = read(OUT / "review-candidate.json")
    visual = read(OUT / "visual-review-receipt.json")
    if not visual["all_passed"] or visual["candidate_sha256"] != sha(OUT / "review-candidate.json"):
        raise ValueError("VISUAL_REVIEW_NOT_VALID")
    if visual["reviewed_pages"] != candidates["renders"]:
        raise ValueError("VISUAL_PAGE_SET_MISMATCH")
    for page in candidates["renders"]:
        if sha(page["path"]) != page["sha256"]:
            raise ValueError("VISUAL_RENDER_CHANGED")
    docs = {
        d["row"]["announcementId"]: d
        for p in list((OUT / "documents").glob("*.json")) + list((OUT / "additional-documents").glob("*.json"))
        if (d := read(p))["body_saved"]
    }
    if len(docs) > 100:
        raise ValueError("TOTAL_BATCH_DOCUMENT_LIMIT_EXCEEDED")
    for item in docs.values():
        if sha(item["receipt"]["path"]) != item["receipt"]["sha256"]:
            raise ValueError("DOCUMENT_SOURCE_CHANGED")
    page_cache = {}
    for claim in candidates["claims"]:
        key = claim["document_id"]
        if key not in page_cache:
            page_cache[key] = extract(claim["source"]["path"])[0]
        replay = review_claim(claim, page_cache[key])
        if replay["money"] != claim["money"] or replay["anchor"] != claim["anchor"]:
            raise ValueError("MONEY_REVIEW_REPLAY_MISMATCH")
    for claim in candidates["yoy_fields"]:
        replay = (review_table_yoy if "row_quote" in claim else review_yoy)(claim, page_cache[claim["document_id"]])
        if replay != claim:
            raise ValueError("YOY_REPLAY_MISMATCH")
    claims = candidates["claims"]
    pairs = [
        compare_claims(claims[0], claims[2], claims[2]["available_at"]),
        compare_claims(claims[1], claims[3], claims[3]["available_at"]),
    ]
    if pairs != candidates["comparisons"] or not candidates["cross_period_check"]["rejected"]:
        raise ValueError("EARNINGS_COMPARISON_REPLAY_MISMATCH")
    identities = read(OUT / "identity-supplement-candidate.json")
    excluded = {v["id"] for v in identities["excluded"]}
    confirmed = {key for key, item in docs.items() if item["identity"]["passed"] and key not in excluded}
    for proof in identities["verified"]:
        item = docs[proof["id"]]
        if sha(item["receipt"]["path"]) != proof["source_sha256"]:
            raise ValueError("IDENTITY_PROOF_SOURCE_CHANGED")
        pages, _ = extract(item["receipt"]["path"])
        if "page" in proof:
            body = normalize(pages[proof["page"] - 1])
            if "股票代码" + proof["stock"] not in body or proof["company"] not in body:
                raise ValueError("IDENTITY_COMPANY_PAGE_REPLAY_MISMATCH")
        else:
            chinese = docs[proof["linked_chinese"]]
            cn_pages, _ = extract(chinese["receipt"]["path"])
            name = "SungrowPowerSupplyCo.,Ltd."
            if (
                name not in normalize(cn_pages[9])
                or name not in normalize(pages[0])
                or sha(chinese["receipt"]["path"]) != proof["linked_chinese_sha256"]
            ):
                raise ValueError("BILINGUAL_IDENTITY_REPLAY_MISMATCH")
        confirmed.add(proof["id"])
    save(
        OUT / "reviewed-earnings-facts.json",
        {
            **candidates,
            "visual_status": "PASSED_SPECIFIED_PAGES",
            "visual_receipt_sha256": sha(OUT / "visual-review-receipt.json"),
        },
    )
    save(
        OUT / "identity-final.json",
        {
            "confirmed_ids": sorted(confirmed),
            "excluded": identities["excluded"],
            "unresolved_ids": sorted(set(docs) - confirmed - excluded),
            "semantic_complete_not_implied": True,
        },
    )
    next_scope = plan_groups(read(OLD / "event-admission/row-coverage.json"), expected_rows, 4, 40)
    save(OUT / "remaining-priority-preview.json", {"status": "COST_PREVIEW_NOT_REGISTERED_BATCH", **next_scope})
    result = {
        "passed": True,
        "protected_files": len(protected),
        "requests": dict(used),
        "request_limits": plan["limits"],
        "documents": len(docs),
        "newly_fetched_bodies": used.get("body", 0),
        "reused_bodies": sum(d["reused"] for d in docs.values()),
        "identity_confirmed": len(confirmed),
        "excluded_from_parent_earnings": len(excluded),
        "catalog_complete": sum(r["catalog_complete"] for r in expected_rows),
        "increment": sum(r["catalog_complete"] for r in expected_rows) - plan["baseline_complete"],
        "independent_coverage_replay": True,
        "remaining_gap_companies": len(missing),
        "remaining_gap_intervals": len(merged_gaps),
        "money_fields": len(claims),
        "yoy_fields": len(candidates["yoy_fields"]),
        "comparison_pairs": len(pairs),
        "visual_pages": len(candidates["renders"]),
        "new_fits": 0,
        "cumulative_fits": 70,
        "old_fit_ledger_entries": len(ledger),
        "training_ready": False,
    }
    save(OUT / "independent-audit-final.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
