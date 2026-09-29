"""第八批五份复杂标题补证：原文代码、双语公司简介或同日正文与全文对应。"""

import json
from pathlib import Path

from app.services.fund_earnings_batch_v4 import extract_pdf, run_path
from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_information_history_v1 import pdf_revision, read, save, sha

from scripts.fund_002112_earnings_review_v8 import render

OUT = run_path("20260929-earnings-v8")


def run():
    plan = read(OUT / "identity-review-plan.json")
    for group in ("code_hashes", "document_hashes"):
        if any(sha(p) != h for p, h in plan[group].items()):
            raise ValueError("IDENTITY_FROZEN_DEPENDENCY_CHANGED")
    documents = {}
    for path in plan["document_hashes"]:
        d = read(path)
        if sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("IDENTITY_SOURCE_BYTES_CHANGED")
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), 500)
        if pages != d["pages"] or metadata != d["metadata"] or pdf_revision(metadata, d["row"]["published_date"]):
            raise ValueError("IDENTITY_SOURCE_REPLAY_OR_DATE_CONFLICT")
        documents[d["row"]["announcementId"]] = d
    results, renders = [], {}
    for spec in plan["specs"]:
        d = documents[spec["document_id"]]
        if d["row"]["secCode"] != spec["stock"]:
            raise ValueError("IDENTITY_TARGET_ISSUER_MISMATCH")
        anchors = []
        for key, page, quote in spec["anchors"]:
            source = documents[key]
            anchors.append(
                {
                    "document_id": key,
                    "source_sha256": source["receipt"]["sha256"],
                    **occurrence(source["pages"], page, quote),
                }
            )
            renders[(key, page)] = render(key, page, source["receipt"]["path"])
        other = spec.get("paired_document")
        if other:
            paired = documents[other]
            if (
                not paired["identity"]["passed"]
                or paired["row"]["secCode"] != spec["stock"]
                or paired["row"]["published_date"] != d["row"]["published_date"]
            ):
                raise ValueError("SAME_DAY_PAIRED_REPORT_NOT_VERIFIED")
        results.append(
            {
                "document_id": spec["document_id"],
                "stock": spec["stock"],
                "published_date": d["row"]["published_date"],
                "original_title": d["row"]["title_plain"],
                "rule": spec["rule"],
                "anchors": anchors,
                "passed": True,
                "same_day_pair_does_not_imply_two_events": bool(other),
                "accounting_basis_not_merged_across_languages": True,
                "semantic_verified": False,
                "training_ready": False,
            }
        )
    save(
        OUT / "identity-review-candidate.json",
        {"items": results, "count": len(results), "renders": list(renders.values()), "visual_status": "PENDING"},
    )
    print(json.dumps({"identity_supplements": len(results), "renders": len(renders)}))


if __name__ == "__main__":
    run()
