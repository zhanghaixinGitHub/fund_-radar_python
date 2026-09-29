"""第24批离线核验：隔离日期冲突，核对现有英文身份和前次口径缺口。"""

import json
import os
import shutil
import subprocess
from pathlib import Path

from app.services.fund_earnings_batch_v4 import RUNS, extract_pdf, issuer_identity
from app.services.fund_earnings_english_identity_v1 import review_english_identity
from app.services.fund_earnings_identity_v1 import profile_proof, supplement_identity
from app.services.fund_earnings_quarterly_growth_v1 import (
    observe_same_period,
    review_percentage_forecast,
    review_q1_report,
)
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

OUT = RUNS / "20260929-earnings-v24"


def run():
    """所有原件都重放；已登记的日期异常只返回隔离证据，不参与身份准入。"""
    plan = read(OUT / "semantic-review-plan.json")
    for group in ("code_hashes", "document_hashes", "source_hashes"):
        if any(sha(p) != h for p, h in plan[group].items()):
            raise ValueError("SEMANTIC_DEPENDENCY_CHANGED")
    docs, conflicts = {}, []
    for path in plan["document_hashes"]:
        d = read(path)
        if sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("ORIGINAL_BYTES_CHANGED")
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), 500)
        if pages != d["pages"] or metadata != d["metadata"]:
            raise ValueError("ORIGINAL_REPLAY_CHANGED")
        issues = pdf_revision(metadata, d["row"]["published_date"])
        if issues != d["revision_issues"]:
            raise ValueError("REVISION_REPLAY_CHANGED")
        key = d["row"]["announcementId"]
        if issues:
            if plan["quarantined_date_conflicts"].get(key) != issues:
                raise ValueError("UNREGISTERED_SOURCE_CONFLICT")
            conflicts.append(
                {
                    "document_id": key,
                    "published_date": d["row"]["published_date"],
                    "metadata": metadata,
                    "issues": issues,
                    "receipt": d["receipt"],
                    "usable_at_catalog_date": False,
                    "semantic_admission": False,
                    "training_ready": False,
                }
            )
        docs[key] = d
    if set(plan["quarantined_date_conflicts"]) != {c["document_id"] for c in conflicts}:
        raise ValueError("REGISTERED_CONFLICT_DISAPPEARED")
    refs = [{"document": d, "proof": p} for d in docs.values() if (p := profile_proof(d))]
    identities = []
    for key in plan["current_document_ids"]:
        d = docs[key]
        direct = issuer_identity(d["row"], d["pages"])
        supplement = None if direct["passed"] else supplement_identity(d, refs)
        identities.append(
            {
                "document_id": key,
                "initial": direct,
                "supplement": supplement,
                "passed": not d["revision_issues"] and (direct["passed"] or bool(supplement and supplement["passed"])),
            }
        )
    specs = {s["document_id"]: s for s in plan["english_identity_specs"]}
    english = []
    for key, spec in specs.items():
        ref = spec.get("reference_id")
        english.append(
            review_english_identity(spec, docs[key], docs[ref] if ref else None, specs[ref] if ref else None)
        )
    # 重读前次大秦原件，不用新收集的其他公司准则替代其自身证据。
    facts, basis_checks = [], []
    for spec in plan["prior_daqin_claims"]:
        d = docs[spec["document_id"]]
        if d["revision_issues"] or not issuer_identity(d["row"], d["pages"])["passed"]:
            raise ValueError("DAQIN_ORIGINAL_CONFLICT")
        review = review_percentage_forecast if spec["kind"] == "FORECAST" else review_q1_report
        facts.append(review({**spec, "published_date": d["row"]["published_date"]}, d["pages"]))
        hits = [
            {"page": n, "marker": marker}
            for n, pg in enumerate(d["pages"], 1)
            for marker in plan["basis_search_markers"]
            if marker in normalize(pg)
        ]
        basis_checks.append(
            {
                "document_id": spec["document_id"],
                "source_sha256": d["receipt"]["sha256"],
                "pages_checked": len(d["pages"]),
                "marker_hits": hits,
                "absence_of_hits_is_not_proof_of_absence": True,
            }
        )
    observation = observe_same_period(facts[0], facts[1], facts[1]["available_at"])
    if observation != read(RUNS / "20260929-earnings-v23/semantic-review-final.json")["descriptive_observation"]:
        raise ValueError("PRIOR_OBSERVATION_CHANGED")
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
                    docs[key]["receipt"]["path"],
                    str(target.with_suffix("")),
                ],
                capture_output=True,
                check=True,
                timeout=45,
            )
        renders.append({"document_id": key, "page": page, "path": str(target), "sha256": sha(target)})
    result = {
        "replayed_bodies": len(docs),
        "current_replayed_bodies": len(identities),
        "identities": identities,
        "current_identity_passed": sum(i["passed"] for i in identities),
        "current_identity_pending": [i["document_id"] for i in identities if not i["passed"]],
        "quarantined_date_conflicts": conflicts,
        "prior_english_identities": english,
        "old_stops_removed": False,
        "prior_daqin_basis_checks": basis_checks,
        "prior_daqin_observation_unchanged": observation,
        "new_comparable_numeric_changes": 0,
        "daqin_basis_route_status": "EXPLICIT_BASIS_NOT_ESTABLISHED_FROM_EXISTING_ORIGINALS_NO_INFERENCE",
        "renders": renders,
        "visual_review_required": True,
        "new_fits": 0,
        "training_ready": False,
        "complete_version_history_claimed": False,
    }
    save(OUT / "semantic-review-candidate.json", result)
    return {
        k: result[k]
        for k in ("replayed_bodies", "current_identity_passed", "current_identity_pending", "daqin_basis_route_status")
    }


if __name__ == "__main__":
    lock = RUNS / ".earnings-preparation.lock"
    owner = {"pid": os.getpid(), "run": OUT.name, "command": "semantic-review"}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    finally:
        if read(lock) == owner:
            lock.unlink()
