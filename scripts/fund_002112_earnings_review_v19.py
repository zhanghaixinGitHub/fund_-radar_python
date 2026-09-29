"""第十九批离线语义重放：明示中国准则的归母预告与实际结果，不拟合。"""

import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from app.services.fund_earnings_annual_forecast_v1 import compare_verified_claims, comparison_readiness
from app.services.fund_earnings_batch_v4 import RUNS, extract_pdf, issuer_identity
from app.services.fund_earnings_cas_parent_v1 import review_cas_parent
from app.services.fund_earnings_identity_v1 import profile_proof, supplement_identity
from app.services.fund_information_history_v1 import pdf_revision, read, save, sha

OUT = RUNS / "20260929-earnings-v19"


def run():
    """重读冻结资料，独立复算身份、金额、可用时点，并生成待目视核对的页面。"""
    plan = read(OUT / "semantic-review-plan.json")
    for group in ("code_hashes", "document_hashes", "source_hashes"):
        if any(sha(p) != h for p, h in plan[group].items()):
            raise ValueError("SEMANTIC_DEPENDENCY_CHANGED")
    documents = {}
    for path in plan["document_hashes"]:
        d = read(path)
        if sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("ORIGINAL_BYTES_CHANGED")
        # 表格必须从原 PDF 重提，与保存文本及元数据逐项一致，不能只复用字段 JSON。
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), 500)
        if pages != d["pages"] or metadata != d["metadata"]:
            raise ValueError("ORIGINAL_REPLAY_CHANGED")
        if pdf_revision(metadata, d["row"]["published_date"]) or d["revision_issues"]:
            raise ValueError("ORIGINAL_REVISION_REQUIRES_REVIEW")
        documents[d["row"]["announcementId"]] = d
    refs = [{"document": d, "proof": p} for d in documents.values() if (p := profile_proof(d))]
    identities = []
    for key, d in documents.items():
        direct = issuer_identity(d["row"], d["pages"])
        identities.append(
            {
                "document_id": key,
                "initial": direct,
                "supplement": None if direct["passed"] else supplement_identity(d, refs),
            }
        )
    facts = []
    for spec in plan["claims"]:
        d = documents[spec["document_id"]]
        if not issuer_identity(d["row"], d["pages"])["passed"] or d["row"]["secCode"] != spec["issuer"]:
            raise ValueError("SEMANTIC_SOURCE_IDENTITY_NOT_VERIFIED")
        fact = review_cas_parent({**spec, "published_date": d["row"]["published_date"]}, d["pages"])
        facts.append({**fact, "source": d["receipt"], "source_identity_verified": True, "revision_issues": []})
    if len(facts) != 2:
        raise ValueError("REGISTERED_PAIR_REQUIRED")
    older, newer = facts
    ready = comparison_readiness(older, newer)
    comparison = compare_verified_claims(older, newer, newer["available_at"])
    snapshots = []
    for fact in facts:
        at = datetime.fromisoformat(fact["available_at"])
        for cutoff in (at - timedelta(seconds=1), at):
            available = [f["document_id"] for f in facts if datetime.fromisoformat(f["available_at"]) <= cutoff]
            old_only = [older["document_id"]] if datetime.fromisoformat(older["available_at"]) <= cutoff else []
            if cutoff < datetime.fromisoformat(newer["available_at"]) and available != old_only:
                raise ValueError("FUTURE_RESULT_CHANGED_PAST")
            snapshots.append(
                {
                    "as_of": cutoff.isoformat(),
                    "available_documents": available,
                    "comparison_available": len(available) == 2,
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
        "replayed_bodies": len(documents),
        "identities": identities,
        "claims": facts,
        "comparison_readiness": ready,
        "comparison": comparison,
        "asof_snapshots": snapshots,
        "future_invariance_passed": True,
        "renders": renders,
        "visual_review_required": True,
        "new_comparable_numeric_changes": 1,
        "new_fits": 0,
        "training_ready": False,
        "selection_basis": plan["selection_basis"],
        "complete_version_history_claimed": False,
    }
    save(OUT / "semantic-review-candidate.json", result)
    return {
        "replayed_bodies": len(documents),
        "facts": len(facts),
        "comparison": comparison,
        "identity_supplements_passed": sum(bool(i["supplement"] and i["supplement"]["passed"]) for i in identities),
        "identity_unresolved": [
            i["document_id"] for i in identities if not i["initial"]["passed"] and not i["supplement"]["passed"]
        ],
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
