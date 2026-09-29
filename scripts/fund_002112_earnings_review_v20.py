"""第二十批离线身份补证；完整保留初核失败，审计附件只作佐证。"""

import json
import os
import shutil
import subprocess
from pathlib import Path

from app.services.fund_earnings_batch_v4 import RUNS, extract_pdf, issuer_identity
from app.services.fund_earnings_identity_supplement_v2 import review_explicit_identity
from app.services.fund_earnings_identity_v1 import profile_proof, supplement_identity
from app.services.fund_information_history_v1 import pdf_revision, read, save, sha

OUT = RUNS / "20260929-earnings-v20"


def run():
    """重提原 PDF、逐项核对冻结页锚点，生成独立身份结果和待目视页面。"""
    plan = read(OUT / "semantic-review-plan.json")
    for group in ("code_hashes", "document_hashes", "source_hashes"):
        if any(sha(p) != h for p, h in plan[group].items()):
            raise ValueError("REVIEW_DEPENDENCY_CHANGED")
    documents = {}
    for path in plan["document_hashes"]:
        d = read(path)
        if sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("ORIGINAL_BYTES_CHANGED")
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), 500)
        if pages != d["pages"] or metadata != d["metadata"]:
            raise ValueError("ORIGINAL_REPLAY_CHANGED")
        if pdf_revision(metadata, d["row"]["published_date"]) or d["revision_issues"]:
            raise ValueError("REVISION_REVIEW_REQUIRED")
        key = d["row"]["announcementId"]
        if key in documents:
            raise ValueError("DUPLICATE_DOCUMENT_ID")
        documents[key] = d
    references = [{"document": d, "proof": p} for d in documents.values() if (p := profile_proof(d))]
    explicit = {}
    for spec in plan["identity_specs"]:
        key = spec["document_id"]
        explicit[key] = review_explicit_identity(spec, documents[key], documents.get(spec.get("reference_document_id")))
    identities = []
    for key, d in documents.items():
        initial = issuer_identity(d["row"], d["pages"])
        supplement = explicit.get(key)
        if not initial["passed"] and supplement is None:
            supplement = supplement_identity(d, references)
        identities.append(
            {
                "document_id": key,
                "initial": initial,
                "supplement": supplement,
                "passed": initial["passed"] or bool(supplement and supplement["passed"]),
                "current_collection": key in plan["current_document_ids"],
            }
        )
    renders = []
    for key, page in plan["render_pages"]:
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
                    documents[key]["receipt"]["path"],
                    str(target.with_suffix("")),
                ],
                capture_output=True,
                check=True,
                timeout=45,
            )
        renders.append({"document_id": key, "page": page, "path": str(target), "sha256": sha(target)})
    result = {
        "identities": identities,
        "explicit_supplements": explicit,
        "replayed_bodies": len(documents),
        "renders": renders,
        "visual_review_required": True,
        "source_dates_unchanged": True,
        "current_identity_passed": sum(i["current_collection"] and i["passed"] for i in identities),
        "current_identity_pending": [
            i["document_id"] for i in identities if i["current_collection"] and not i["passed"]
        ],
        "prior_pending_identity_resolved": [k for k in explicit if k not in plan["current_document_ids"]],
        "new_comparable_numeric_changes": 0,
        "new_fits": 0,
        "training_ready": False,
        "all_amounts_semantically_reviewed": False,
        "original_initial_results_preserved": True,
    }
    save(OUT / "semantic-review-candidate.json", result)
    return {k: v for k, v in result.items() if k not in {"identities", "explicit_supplements", "renders"}}


if __name__ == "__main__":
    lock = RUNS / ".earnings-preparation.lock"
    owner = {"pid": os.getpid(), "run": OUT.name, "command": "offline-identity-review"}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    finally:
        if read(lock) == owner:
            lock.unlink()
