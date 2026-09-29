"""第十四批离线复核：保留季度期间、披露阶段和来源限制，不解释为预测收益。"""

import json
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from app.services.fund_earnings_accounting_loss_v1 import review_explicit_profit_range, review_parenthesized_loss_row
from app.services.fund_earnings_asof_v1 import asof_changes
from app.services.fund_earnings_batch_v4 import extract_pdf, run_path
from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_identity_v1 import occurrence, profile_proof, supplement_identity
from app.services.fund_earnings_page_span_v1 import review_page_span_money_row
from app.services.fund_earnings_scoped_table_v1 import review_scoped_money_row
from app.services.fund_earnings_source_scope_v1 import review_non_profit_sources
from app.services.fund_earnings_table_facts_v1 import review_money_row
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

OUT = run_path("20260929-earnings-v14")


def render(key, page, source):
    """保存选定原件页供人工查看；重放保留已有图片字节。"""
    target = OUT / "review-renders" / f"{key}-p{page}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(
            [
                shutil.which("pdftoppm"),
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "-scale-to",
                "1500",
                "-png",
                source,
                str(target.with_suffix("")),
            ],
            capture_output=True,
            check=True,
            timeout=45,
        )
    return {"document_id": key, "page": page, "path": str(target), "sha256": sha(target)}


def run():
    plan = read(OUT / "semantic-review-plan.json")
    for group in ("code_hashes", "document_hashes"):
        if any(sha(p) != h for p, h in plan[group].items()):
            raise ValueError("FROZEN_DEPENDENCY_CHANGED")
    documents, failures = {}, []
    for path in plan["document_hashes"]:
        d = read(path)
        if not d["body_saved"]:
            failures.append({"document_id": d["row"]["announcementId"], "reason": d["reason"]})
            continue
        if sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("PDF_BYTES_CHANGED")
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), 500)
        if [normalize(p) for p in pages] != [normalize(p) for p in d["pages"]] or metadata != d["metadata"]:
            raise ValueError("PDF_REPLAY_CHANGED")
        if pdf_revision(metadata, d["row"]["published_date"]) != d["revision_issues"]:
            raise ValueError("REVISION_REPLAY_CHANGED")
        documents[d["row"]["announcementId"]] = d
    references = [{"document": d, "proof": profile_proof(d)} for d in documents.values() if profile_proof(d)]
    automatic = [supplement_identity(d, references) for d in documents.values() if not d["identity"]["passed"]]
    verified = {k for k, d in documents.items() if d["identity"]["passed"] and not d["revision_issues"]}
    verified.update(p["document_id"] for p in automatic if p["passed"])
    manual = []
    for spec in plan["identity_specs"]:
        d = documents[spec["document_id"]]
        if d["row"]["secCode"] != spec["stock"] or d["revision_issues"]:
            raise ValueError("IDENTITY_ISSUER_OR_DATE_CONFLICT")
        anchors = [occurrence(d["pages"], page, text) for page, text in spec["anchors"]]
        reference_proof = None
        if "reference_document_id" in spec:
            other = documents[spec["reference_document_id"]]
            if (
                other["row"]["announcementId"] not in verified
                or other["row"]["secCode"] != spec["stock"]
                or other["row"]["published_date"] > d["row"]["published_date"]
                or other["revision_issues"]
            ):
                raise ValueError("IDENTITY_REFERENCE_NOT_VALID_OR_LATER")
            legal = normalize(spec["legal_name"])
            if not legal or not any(legal in normalize(p) for p in d["pages"]):
                raise ValueError("TARGET_LEGAL_NAME_MISSING")
            if not any(legal in normalize(p) for p in other["pages"]):
                raise ValueError("REFERENCE_LEGAL_NAME_MISSING")
            reference_proof = {
                "document_id": other["row"]["announcementId"],
                "sha256": other["receipt"]["sha256"],
                "anchors": [occurrence(other["pages"], p, q) for p, q in spec["reference_anchors"]],
                "identity_only_not_an_independent_earnings_change": True,
            }
        manual.append(
            {
                **spec,
                "anchors": anchors,
                "reference_proof": reference_proof,
                "passed": True,
                "original_title": d["row"]["title_plain"],
                "source_sha256": d["receipt"]["sha256"],
                "published_date": d["row"]["published_date"],
                "version_history_verified": False,
                "training_ready": False,
            }
        )
        verified.add(spec["document_id"])
    # 身份通过仍不等于利润用途通过；先隔离监管报告和仅补薪酬的原件。
    non_profit_sources = review_non_profit_sources(
        plan["non_profit_sources"], documents, verified, plan["money_specs"]
    )
    claims = []
    for spec in plan["money_specs"]:
        key = spec["document_id"]
        if key not in verified:
            raise ValueError("MONEY_SOURCE_NOT_VERIFIED")
        d = documents[key]
        reviewer = (
            review_explicit_profit_range
            if spec.get("explicit_profit_range")
            else review_parenthesized_loss_row
            if spec.get("accounting_parentheses")
            else review_page_span_money_row
            if "page_segments" in spec
            else review_scoped_money_row
            if "scope_quote" in spec
            else review_money_row
            if "row_quote" in spec
            else review_claim
        )
        claim = reviewer({**spec, "published_date": d["row"]["published_date"]}, d["pages"])
        claims.append(
            {
                **claim,
                "source": d["receipt"],
                "source_identity_verified": True,
                "revision_issues": [],
                "period_anchor": occurrence(d["pages"], spec["period_page"], spec["period_quote"]),
                "audit_anchor": occurrence(d["pages"], spec["audit_page"], spec["audit_quote"]),
                "disclosure_limitations": [
                    occurrence(d["pages"], page, quote) for page, quote in spec.get("limitation_anchors", [])
                ],
            }
        )
    snapshots = []
    for issuer in sorted({c["issuer"] for c in claims}):
        group = [c for c in claims if c["issuer"] == issuer]
        for at in sorted({datetime.fromisoformat(c["available_at"]) for c in group}):
            before = (at - timedelta(seconds=1)).isoformat()
            for query in (before, at.isoformat()):
                available = [
                    c for c in group if datetime.fromisoformat(c["available_at"]) <= datetime.fromisoformat(query)
                ]
                if asof_changes(group, query) != asof_changes(available, query):
                    raise ValueError("FUTURE_DISCLOSURE_CHANGED_PAST")
                snapshots.append({"issuer": issuer, "as_of": query, "facts": asof_changes(group, query)})
    scopes = []
    for spec in plan["scope_specs"]:
        key = spec["document_id"]
        if key not in verified:
            raise ValueError("SCOPE_SOURCE_NOT_VERIFIED")
        scopes.append(
            {
                **spec,
                "anchors": [occurrence(documents[key]["pages"], page, quote) for page, quote in spec["anchors"]],
                "training_ready": False,
            }
        )
    result = {
        "bodies_replayed": len(documents),
        "failed_body_positions": failures,
        "identity_automatic": automatic,
        "identity_manual": manual,
        "identity_and_date_passed": len(verified),
        "pending_ids": sorted(set(documents) - verified),
        "claims": claims,
        "asof_snapshots": snapshots,
        "future_invariance_passed": True,
        "scope_reviews": scopes,
        "non_profit_sources": non_profit_sources,
        "renders": [render(k, p, documents[k]["receipt"]["path"]) for k, p in plan["render_pages"]],
        "visual_status": "PENDING",
        "training_ready": False,
        "new_fits": 0,
        "new_requests": 0,
    }
    save(OUT / "semantic-review-candidate.json", result)
    print(
        json.dumps(
            {
                "bodies": len(documents),
                "identity_passed": len(verified),
                "pending": result["pending_ids"],
                "claims": len(claims),
                "snapshots": len(snapshots),
            }
        )
    )


if __name__ == "__main__":
    run()
