"""第十批原件复核：表格列口径、前次预告与实际利润、非利润修订和时间冲突。"""

import json
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from app.services.fund_earnings_asof_v1 import asof_changes
from app.services.fund_earnings_batch_v4 import extract_pdf, run_path
from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_identity_v1 import occurrence, profile_proof, supplement_identity
from app.services.fund_earnings_table_facts_v1 import review_money_row
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

OUT = run_path("20260929-earnings-v10")


def render(key, number, source):
    """只生成计划内选定页面，独立复核时保持已有图片字节不变。"""
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
            capture_output=True,
            check=True,
            timeout=45,
        )
    return {"document_id": key, "page": number, "path": str(target), "sha256": sha(target)}


def run():
    plan = read(OUT / "semantic-review-plan.json")
    for group in ("code_hashes", "document_hashes"):
        if any(sha(p) != h for p, h in plan[group].items()):
            raise ValueError("FROZEN_REVIEW_DEPENDENCY_CHANGED")
    sources, conflicts = {}, []
    for path in plan["document_hashes"]:
        d = read(path)
        if sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("SOURCE_BYTES_CHANGED")
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), 500)
        issues = pdf_revision(metadata, d["row"]["published_date"])
        if (
            [normalize(p) for p in pages] != [normalize(p) for p in d["pages"]]
            or metadata != d["metadata"]
            or issues != d["revision_issues"]
        ):
            raise ValueError("SOURCE_TEXT_METADATA_REPLAY_CHANGED")
        key = d["row"]["announcementId"]
        sources[key] = d
        if issues:
            conflicts.append(
                {
                    "document_id": key,
                    "published_date": d["row"]["published_date"],
                    "receipt": d["receipt"],
                    "metadata": metadata,
                    "issues": issues,
                    "status": "STOP_SOURCE_DATE_CONFLICT",
                    "training_ready": False,
                }
            )
    refs = [{"document": d, "proof": profile_proof(d)} for d in sources.values() if profile_proof(d)]
    automatic = [supplement_identity(d, refs) for d in sources.values() if not d["identity"]["passed"]]
    verified = {k for k, d in sources.items() if d["identity"]["passed"] and not d["revision_issues"]}
    verified.update(p["document_id"] for p in automatic if p["passed"])
    manual = []
    for spec in plan["identity_specs"]:
        d = sources[spec["document_id"]]
        if d["row"]["secCode"] != spec["stock"] or d["revision_issues"]:
            raise ValueError("IDENTITY_ISSUER_OR_DATE_CONFLICT")
        reference = spec.get("reference_document")
        if reference:
            other = sources[reference]
            if (
                reference not in verified
                or other["row"]["secCode"] != spec["stock"]
                or other["row"]["published_date"] > d["row"]["published_date"]
            ):
                raise ValueError("IDENTITY_REFERENCE_NOT_AVAILABLE_OR_VERIFIED")
            if spec.get("same_day_required") and other["row"]["published_date"] != d["row"]["published_date"]:
                raise ValueError("PAIRED_REPORT_PUBLIC_DATE_DIFFERS")
        anchors = []
        for key, number, text in spec["anchors"]:
            source = sources[key]
            if source["row"]["secCode"] != spec["stock"]:
                raise ValueError("IDENTITY_ANCHOR_WRONG_ISSUER")
            anchors.append(
                {
                    "document_id": key,
                    "source_sha256": source["receipt"]["sha256"],
                    **occurrence(source["pages"], number, text),
                }
            )
        manual.append({**spec, "anchors": anchors, "passed": True, "semantic_verified": False, "training_ready": False})
        verified.add(spec["document_id"])
    claims = []
    for spec in plan["money_specs"]:
        key = spec["document_id"]
        if key not in verified:
            raise ValueError("MONEY_SOURCE_NOT_VERIFIED")
        d = sources[key]
        spec = {**spec, "published_date": d["row"]["published_date"]}
        review = review_money_row if "row_quote" in spec else review_claim
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
    for issuer in sorted({c["issuer"] for c in claims}):
        group = [c for c in claims if c["issuer"] == issuer]
        instant = max(datetime.fromisoformat(c["available_at"]) for c in group)
        earlier = [c for c in group if datetime.fromisoformat(c["available_at"]) < instant]
        early = (instant - timedelta(seconds=1)).isoformat()
        if asof_changes(group, early) != asof_changes(earlier, early):
            raise ValueError("FUTURE_DISCLOSURE_CHANGED_EARLIER_QUERY")
        snapshots.extend(
            {"issuer": issuer, "as_of": t, "facts": asof_changes(group, t)} for t in (early, instant.isoformat())
        )
    # 分类仅覆盖已经选定且逐页核验的段落；无文字或未对账部分仍未知。
    scope_reviews = []
    for spec in plan["scope_specs"]:
        anchors = []
        for key, page, quote in spec["anchors"]:
            if key not in verified:
                raise ValueError("SCOPE_SOURCE_NOT_VERIFIED")
            anchors.append(
                {
                    "document_id": key,
                    "published_date": sources[key]["row"]["published_date"],
                    **occurrence(sources[key]["pages"], page, quote),
                }
            )
        scope_reviews.append({**spec, "anchors": anchors, "training_ready": False})
    main_ids = {p.stem for p in (OUT / "documents").glob("*.json")}
    result = {
        "identity_automatic": automatic,
        "identity_manual": manual,
        "main_bodies": len(main_ids),
        "main_identity_and_date_passed": len(main_ids & verified),
        "main_pending_ids": sorted(main_ids - verified),
        "source_date_conflicts": conflicts,
        "claims": claims,
        "asof_snapshots": snapshots,
        "scope_reviews": scope_reviews,
        "future_invariance_passed": True,
        "bodies_replayed": len(sources),
        "renders": [render(k, p, sources[k]["receipt"]["path"]) for k, p in plan["render_pages"]],
        "visual_status": "PENDING",
        "new_fits": 0,
        "training_ready": False,
    }
    save(OUT / "semantic-review-candidate.json", result)
    print(
        json.dumps(
            {
                "identity_passed": result["main_identity_and_date_passed"],
                "pending": result["main_pending_ids"],
                "claims": len(claims),
                "snapshots": len(snapshots),
                "source_date_conflicts": len(conflicts),
            }
        )
    )


if __name__ == "__main__":
    run()
