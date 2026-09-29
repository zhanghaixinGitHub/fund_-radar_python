"""第八批核验：亿/百万单位、单季与累计期间，以及有独立原件的预告修正链。"""

import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from app.services.fund_earnings_asof_v1 import asof_changes
from app.services.fund_earnings_batch_v4 import extract_pdf, run_path
from app.services.fund_earnings_evidence_v1 import compare_claims, review_claim
from app.services.fund_earnings_identity_v1 import occurrence, profile_proof, supplement_identity
from app.services.fund_earnings_signed_facts_v1 import review_loss_claim
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

OUT = run_path("20260929-earnings-v8")


def render(key, number, source):
    target = OUT / "review-renders" / f"{key}-p{number}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(
            [
                shutil.which("pdftoppm"),
                "-f",
                str(number),
                "-l",
                str(number),
                "-singlefile",
                "-scale-to",
                "1500",
                "-png",
                source,
                str(target.with_suffix("")),
            ],
            check=True,
            capture_output=True,
            timeout=45,
        )
    return {"document_id": key, "page": number, "path": str(target), "sha256": sha(target)}


def run():
    plan = read(OUT / "semantic-review-plan.json")
    for group in ("code_hashes", "document_hashes"):
        if any(sha(p) != h for p, h in plan[group].items()):
            raise ValueError("FROZEN_REVIEW_DEPENDENCY_CHANGED")
    sources = {}
    for path in sorted(plan["document_hashes"]):
        d = read(path)
        if not d["body_saved"]:
            continue
        if sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("REVIEW_SOURCE_CHANGED")
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), 500)
        if [normalize(p) for p in pages] != [normalize(p) for p in d["pages"]] or metadata != d["metadata"]:
            raise ValueError("REVIEW_TEXT_REPLAY_CHANGED")
        sources[d["row"]["announcementId"]] = d
    excluded = {"1209769630": "ANNUAL_REPORT_ERROR_ACCOUNTABILITY_POLICY_NOT_EARNINGS_DISCLOSURE"}
    refs = [{"document": d, "proof": profile_proof(d)} for d in sources.values() if profile_proof(d)]
    supplements = [
        supplement_identity(d, refs)
        for key, d in sources.items()
        if not d["identity"]["passed"] and key not in excluded
    ]
    by_id = {p["document_id"]: p for p in supplements}
    claims = []
    for spec in plan["money_specs"]:
        d = sources[spec["document_id"]]
        if not (d["identity"]["passed"] or by_id.get(spec["document_id"], {}).get("passed")):
            raise ValueError("MONEY_SOURCE_IDENTITY_NOT_PASSED")
        if pdf_revision(d["metadata"], d["row"]["published_date"]):
            raise ValueError("MONEY_SOURCE_REVISION_CONFLICT")
        spec = {**spec, "published_date": d["row"]["published_date"]}
        review = review_loss_claim if spec.get("loss_anchor") else review_claim
        claim = review(spec, d["pages"])
        claims.append(
            {
                **claim,
                "source": d["receipt"],
                "source_identity_verified": True,
                "revision_issues": [],
                "period_anchor": occurrence(d["pages"], spec["period_page"], spec["period_quote"]),
            }
        )
    snapshots = []
    for issuer in ("000651", "300853"):
        group = [c for c in claims if c["issuer"] == issuer]
        times = sorted({datetime.fromisoformat(c["available_at"]) for c in group})
        for instant in times[1:]:
            early = (instant - timedelta(seconds=1)).isoformat()
            earlier = [c for c in group if datetime.fromisoformat(c["available_at"]) < instant]
            if asof_changes(group, early) != asof_changes(earlier, early):
                raise ValueError("FUTURE_INFORMATION_CHANGED_PAST")
            snapshots.extend(
                {"issuer": issuer, "as_of": t, "facts": asof_changes(group, t)} for t in (early, instant.isoformat())
            )
    # 同一季报的单季与年初累计不是同一期间，不能因为标题相同就直接相减。
    gree = [c for c in claims if c["issuer"] == "000651" and c["kind"] == "REPORTED_RESULT"]
    try:
        compare_claims(gree[0], gree[1], gree[1]["available_at"])
    except ValueError as exc:
        if str(exc) != "INCOMPARABLE_EARNINGS_CLAIMS":
            raise
    else:
        raise ValueError("QUARTER_VERSUS_CUMULATIVE_WAS_COMPARED")
    bank = [c for c in claims if c["issuer"] == "601398"]
    bank_facts = asof_changes(bank, bank[0]["available_at"])
    if not all(f["status"] == "NO_PRIOR_COMPARABLE_DISCLOSURE" and f["change"] is None for f in bank_facts):
        raise ValueError("MISSING_BANK_PRIOR_WAS_FILLED")
    wrong = sources["1216517762"]
    conflict = {
        "document_id": "1216517762",
        "catalog_title": wrong["row"]["title_plain"],
        "published_date": wrong["row"]["published_date"],
        "body_title_anchor": occurrence(wrong["pages"], 1, "2022年年度报告摘要"),
        "status": "STOPPED_CATALOG_REPORT_YEAR_DIFFERS_FROM_BODY",
        "training_ready": False,
    }
    lower = sources["1205505074"]
    # “增加某金额以上”不是当前利润金额，也没有有限上界；只保存候选口径，不造区间。
    lower_pending = {
        "document_id": "1205505074",
        "issuer": "600176",
        "anchor": occurrence(lower["pages"], 1, "将增加31,030.73万元以上，同比增长20%以上"),
        "status": "ONE_SIDED_INCREASE_BOUND_NOT_ABSOLUTE_PROFIT_INTERVAL",
        "absolute_profit_amount": None,
        "finite_upper_bound": None,
        "training_ready": False,
    }
    result = {
        "claims": claims,
        "asof_snapshots": snapshots,
        "bank_facts_without_prior": bank_facts,
        "cross_period_comparison_rejected": True,
        "future_invariance_passed": True,
        "identity_supplements": supplements,
        "identity_supplement_counts": dict(Counter(s.get("rule", s.get("reason")) for s in supplements)),
        "identity_supplements_passed": sum(s["passed"] for s in supplements),
        "purpose_exclusions": excluded,
        "report_year_conflict": conflict,
        "one_sided_forecast_pending": lower_pending,
        "bodies_replayed": len(sources),
        "original_reference_source_independently_verified": "1215682799",
        "renders": [render(k, p, sources[k]["receipt"]["path"]) for k, p in plan["render_pages"]],
        "visual_status": "PENDING",
        "new_fits": 0,
        "training_ready": False,
    }
    save(OUT / "semantic-review-candidate.json", result)
    print(
        json.dumps(
            {
                "claims": len(claims),
                "snapshots": len(snapshots),
                "identity_passed": result["identity_supplements_passed"],
                "identity_pending": sum(not s["passed"] for s in supplements),
                "renders": len(result["renders"]),
            }
        )
    )


if __name__ == "__main__":
    run()
