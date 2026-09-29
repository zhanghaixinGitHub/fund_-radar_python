"""离线独立重放本批原始分页、训练覆盖和已核字段；不发请求、不拟合。"""

import json
from collections import Counter, defaultdict
from datetime import date, timedelta

from app.services.fund_earnings_evidence_v1 import compare_claims, plan_groups, review_claim
from app.services.fund_information_history_v1 import OLD, catalog_rows, read, save, sha
from app.services.fund_information_history_v1 import OUT as PREVIOUS

from scripts.fund_002112_earnings_batch_v1 import OUT, verify
from scripts.fund_002112_earnings_review_v1 import extract


def days(start, end):
    current, finish = date.fromisoformat(start), date.fromisoformat(end)
    while current <= finish:
        yield str(current)
        current += timedelta(days=1)


def run():
    protection = verify()
    plan = read(OUT / "plan.json")
    expected_paths = {str(p) for p in (OUT / "catalogs").glob("*.json")}
    if len(expected_paths) != len(plan["windows"]) + len(plan["context_windows"]):
        raise ValueError("PLANNED_CATALOG_COUNT_MISMATCH")
    all_catalogs = [read(p) for p in sorted(expected_paths)]
    for catalog in all_catalogs:
        if not catalog["catalog_complete"]:
            continue
        pages = []
        for receipt in catalog["receipts"]:
            if sha(receipt["path"]) != receipt["sha256"]:
                raise ValueError("CATALOG_RAW_CHANGED")
            pages.append((receipt["params"]["pageNum"], read(receipt["path"])))
        window = catalog["window"]
        actual = catalog_rows(pages, window["stock"], window["start"], window["end"])
        if actual != catalog["rows"]:
            raise ValueError("RAW_CATALOG_REPLAY_MISMATCH")
    bundle = read(read(OLD / "event-admission/bundle-manifest.json")["path"])
    intervals = {k: list(v) for k, v in bundle["coverage"]["company"].items()}
    for catalog in all_catalogs + [read(p) for p in (PREVIOUS / "company-catalogs").glob("*.json")]:
        if catalog["catalog_complete"]:
            w = catalog["window"]
            intervals.setdefault(w["stock"], []).append([w["start"], w["end"]])
    # 与采集脚本的区间游标法分开，按自然日集合重算原有 30 天窗口。
    covered_days = {stock: {d for lo, hi in spans for d in days(lo, hi)} for stock, spans in intervals.items()}
    expected_rows = []
    merged_gaps = defaultdict(list)
    for row in read(OLD / "event-admission/row-coverage.json"):
        lo = str(date.fromisoformat(row["target"]) - timedelta(days=30))
        hi = str(date.fromisoformat(row["target"]) - timedelta(days=1))
        required = set(days(lo, hi))
        gaps = [s for s in row["missing_companies"] if not required.issubset(covered_days.get(s, set()))]
        expected_rows.append(
            {
                "fund_code": row["fund_code"],
                "target": row["target"],
                "catalog_complete": row["company_coverage_passed"] or (bool(row["missing_companies"]) and not gaps),
                "remaining_companies": gaps,
            }
        )
        for stock in gaps:
            merged_gaps[stock].append((lo, hi))
    if expected_rows != read(OUT / "row-coverage.json"):
        raise ValueError("INDEPENDENT_COVERAGE_MISMATCH")
    worklist = []
    for stock, spans in sorted(merged_gaps.items()):
        merged = []
        for lo, hi in sorted(spans):
            if merged and lo <= str(date.fromisoformat(merged[-1][1]) + timedelta(days=1)):
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        worklist.extend({"stock": stock, "start": lo, "end": hi} for lo, hi in merged)
    save(OUT / "company-gap-worklist.json", worklist)
    before = {(r["fund_code"], r["target"]): r for r in read(PREVIOUS / "row-coverage.json")}
    after = {(r["fund_code"], r["target"]): r for r in expected_rows}
    folds = []
    # 只投影原文件的名称和训练成员身份；不访问 exam、标签、权重或预测列。
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
    review = read(OUT / "earnings-review-candidate.json")
    visual = read(OUT / "visual-review-receipt.json")
    if visual["candidate_sha256"] != sha(OUT / "earnings-review-candidate.json"):
        raise ValueError("VISUAL_REVIEW_CANDIDATE_CHANGED")
    if visual["reviewed_pages"] != review["renders"] or not visual["all_passed"]:
        raise ValueError("VISUAL_REVIEW_NOT_COMPLETE")
    for page in review["renders"]:
        if sha(page["path"]) != page["sha256"]:
            raise ValueError("REVIEW_IMAGE_CHANGED")
    for claim in review["claims"]:
        if sha(claim["source"]["path"]) != claim["source"]["sha256"]:
            raise ValueError("CLAIM_PDF_CHANGED")
        pages, _ = extract(claim["source"]["path"])
        replay = review_claim(claim, pages)
        if replay["anchor"] != claim["anchor"] or replay["money"] != claim["money"]:
            raise ValueError("CLAIM_REPLAY_MISMATCH")
    claims = review["claims"]
    pairs = [
        compare_claims(claims[0], claims[1], claims[1]["available_at"]),
        compare_claims(claims[1], claims[2], claims[2]["available_at"]),
    ]
    if pairs != review["comparisons"]:
        raise ValueError("CLAIM_CHAIN_REPLAY_MISMATCH")
    prepared = plan_groups(read(OLD / "event-admission/row-coverage.json"), expected_rows)
    save(
        OUT / "remaining-priority-preview.json",
        {
            "status": "COST_RANKING_ONLY_NOT_AUTHORIZED_COLLECTION_OR_TRAINING_PROTOCOL",
            **prepared,
        },
    )
    result = {
        "passed": True,
        "protection": protection,
        "catalog_windows_replayed": len(all_catalogs),
        "independent_row_coverage_passed": True,
        "catalog_complete": sum(r["catalog_complete"] for r in expected_rows),
        "folds": folds,
        "remaining_gap_companies": len(merged_gaps),
        "remaining_gap_intervals": len(worklist),
        "by_fund": dict(Counter(r["fund_code"] for r in expected_rows if r["catalog_complete"])),
        "reviewed_claims": len(claims),
        "reviewed_chain_transitions": len(pairs),
        "visual_pages": len(review["renders"]),
        "training_ready": False,
        "new_fits": 0,
        "reason": "Historical catalog coverage and semantic coverage still incomplete; old two hypotheses stopped",
    }
    save(OUT / "independent-audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
