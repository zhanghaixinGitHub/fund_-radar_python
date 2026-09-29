"""第 23 批离线核验：季报身份补证、同比原文及非利润更正；不启动拟合。"""

import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from app.services.fund_earnings_batch_v4 import RUNS, extract_pdf, issuer_identity
from app.services.fund_earnings_identity_v1 import profile_proof, supplement_identity
from app.services.fund_earnings_quarterly_growth_v1 import (
    observe_same_period,
    review_percentage_forecast,
    review_q1_report,
)
from app.services.fund_earnings_same_day_identity_v1 import review_same_day_quarterly_identity
from app.services.fund_earnings_yoy_amount_v1 import unique
from app.services.fund_information_history_v1 import pdf_revision, read, save, sha

OUT = RUNS / "20260929-earnings-v23"


def run():
    """原件重新抽取后复算；合格身份、已核事实、可训练比较分别返回。"""
    plan = read(OUT / "semantic-review-plan.json")
    for group in ("code_hashes", "document_hashes", "source_hashes"):
        if any(sha(p) != h for p, h in plan[group].items()):
            raise ValueError("SEMANTIC_DEPENDENCY_CHANGED")
    documents = {}
    for path in plan["document_hashes"]:
        d = read(path)
        if sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("ORIGINAL_BYTES_CHANGED")
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), 500)
        if pages != d["pages"] or metadata != d["metadata"]:
            raise ValueError("ORIGINAL_REPLAY_CHANGED")
        if pdf_revision(metadata, d["row"]["published_date"]) or d["revision_issues"]:
            raise ValueError("ORIGINAL_REVISION_REQUIRES_REVIEW")
        documents[d["row"]["announcementId"]] = d
    refs = [{"document": d, "proof": p} for d in documents.values() if (p := profile_proof(d))]
    bridges = [
        review_same_day_quarterly_identity(documents[b["target_id"]], documents[b["reference_id"]], b["legal_name"])
        for b in plan["identity_bridges"]
    ]
    bridge_map = {b["document_id"]: b for b in bridges}
    identities = []
    for key in plan["current_document_ids"]:
        d = documents[key]
        direct = issuer_identity(d["row"], d["pages"])
        supplement = None if direct["passed"] else supplement_identity(d, refs)
        if supplement and not supplement["passed"] and key in bridge_map:
            supplement = bridge_map[key]
        identities.append({"document_id": key, "initial": direct, "supplement": supplement})
    facts = []
    for spec in plan["claims"]:
        d = documents[spec["document_id"]]
        if not issuer_identity(d["row"], d["pages"])["passed"] or d["row"]["secCode"] != spec["issuer"]:
            raise ValueError("SEMANTIC_SOURCE_IDENTITY_NOT_VERIFIED")
        review = review_percentage_forecast if spec["kind"] == "FORECAST" else review_q1_report
        facts.append(
            review({**spec, "published_date": d["row"]["published_date"]}, d["pages"])
            | {"source": d["receipt"], "source_identity_verified": True}
        )
    forecast, report, half = facts
    observation = observe_same_period(forecast, report, report["available_at"])
    snapshots = []
    for fact in facts:
        at = datetime.fromisoformat(fact["available_at"])
        for cutoff in (at - timedelta(seconds=1), at):
            known = [f["document_id"] for f in facts if datetime.fromisoformat(f["available_at"]) <= cutoff]
            # 上半年预告不能追写一季报公开前的事实，也不能替代其同比比较数。
            before_half = [f["document_id"] for f in facts[:2] if datetime.fromisoformat(f["available_at"]) <= cutoff]
            if cutoff < datetime.fromisoformat(half["available_at"]) and known != before_half:
                raise ValueError("FUTURE_HALF_YEAR_CHANGED_PAST")
            snapshots.append(
                {
                    "as_of": cutoff.isoformat(),
                    "available_documents": known,
                    "q1_observation_available": all(f["document_id"] in known for f in facts[:2]),
                }
            )
    rejected_cross_period = False
    try:
        observe_same_period(half, report, half["available_at"])
    except ValueError as exc:
        rejected_cross_period = str(exc) == "SAME_PERIOD_AND_SAME_COMPARATIVE_REQUIRED"
    if not rejected_cross_period:
        raise ValueError("CROSS_PERIOD_MUST_BE_REJECTED")
    correction = plan["non_profit_correction"]
    d = documents[correction["document_id"]]
    if not issuer_identity(d["row"], d["pages"])["passed"] or d["row"]["secCode"] != "000538":
        raise ValueError("CORRECTION_IDENTITY_UNVERIFIED")
    correction_result = {
        **correction,
        "anchors": [unique(d["pages"], page, quote) for page, quote in correction["quotes"]],
        "published_date": d["row"]["published_date"],
        "profit_revision": False,
        "backdating_to_original_report_permitted": False,
        "training_ready": False,
    }
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
        "replayed_bodies": len(documents),
        "current_replayed_bodies": len(identities),
        "identities": identities,
        "current_identity_passed": sum(
            i["initial"]["passed"] or bool(i["supplement"] and i["supplement"]["passed"]) for i in identities
        ),
        "current_identity_pending": [
            i["document_id"] for i in identities if not i["initial"]["passed"] and not i["supplement"]["passed"]
        ],
        "same_day_identity_bridges": bridges,
        "old_stops_removed": False,
        "claims": facts,
        "descriptive_observation": observation,
        "cross_period_comparison_rejected": True,
        "non_profit_correction": correction_result,
        "asof_snapshots": snapshots,
        "future_invariance_passed": True,
        "renders": renders,
        "visual_review_required": True,
        "new_comparable_numeric_changes": 0,
        "new_fits": 0,
        "training_ready": False,
        "complete_version_history_claimed": False,
    }
    save(OUT / "semantic-review-candidate.json", result)
    return {
        k: result[k]
        for k in ("replayed_bodies", "current_identity_passed", "current_identity_pending", "descriptive_observation")
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
