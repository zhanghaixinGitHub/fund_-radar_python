"""跨批次复核原件身份停点；补证只引用同一公司当时已经公开的资料。"""

from pathlib import Path

from app.services.fund_earnings_batch_v4 import issuer_identity
from app.services.fund_earnings_identity_v1 import profile_proof, supplement_identity
from app.services.fund_information_history_v1 import ROOT, read, save, sha

from scripts.fund_002112_closure_v1 import OUT


def run():
    """复用已保存全文，核对哈希，输出可回放身份依据，不承诺金额或版本已准入。"""
    registry = ROOT / "information-research/20260929-earnings-v24/source-stop-registry.json"
    sources = {str(registry): sha(registry), str(Path(__file__).resolve()): sha(__file__)}
    stopped = read(registry)["entries"]
    docs, conflicts = {}, []
    for path in sorted((ROOT / "information-research").glob("*-earnings-v*/documents/*.json")):
        value = read(path)
        if not value.get("body_saved") or not value.get("pages"):
            continue
        raw = value["receipt"]
        if sha(raw["path"]) != raw["sha256"]:
            raise ValueError("OLD_ORIGINAL_CHANGED")
        sources[str(path)] = sha(path)
        sources[raw["path"]] = raw["sha256"]
        key = value["row"]["announcementId"]
        if key in docs and docs[key]["receipt"]["sha256"] != raw["sha256"]:
            conflicts.append(key)
        docs[key] = value
    references = [{"document": d, "proof": p} for d in docs.values() if (p := profile_proof(d))]
    results = []
    for entry in stopped:
        key = entry["document_id"]
        if "IDENTITY" not in entry.get("stage", "") and not any(
            "IDENTITY" in e.get("reason", "") for e in entry["evidence"]
        ):
            continue
        if key not in docs:
            results.append({"document_id": key, "passed": False, "reason": "NO_SAVED_BODY"})
            continue
        doc = docs[key]
        direct = issuer_identity(doc["row"], doc["pages"])
        extra = supplement_identity(doc, references)
        passed = bool((direct["passed"] or extra["passed"]) and not doc["revision_issues"] and key not in conflicts)
        results.append(
            {
                "document_id": key,
                "title": doc["row"]["title_plain"],
                "stock": doc["row"]["secCode"],
                "passed": passed,
                "direct": direct,
                "supplement": extra,
                "old_stop_unchanged": True,
                "training_ready": False,
            }
        )
    save(OUT / "identity-replay-sources.json", sources)
    save(
        OUT / "identity-replay-candidate.json",
        {
            "documents_indexed": len(docs),
            "intrinsic_profile_references": len(references),
            "body_version_conflicts": sorted(set(conflicts)),
            "identity_candidates": results,
            "sources_sha256": sha(OUT / "identity-replay-sources.json"),
            "visual_review_required_before_manual_admission": True,
            "new_requests": 0,
            "new_fits": 0,
        },
    )
    print(
        {
            "documents": len(docs),
            "references": len(references),
            "stops": len(results),
            "passed": sum(x["passed"] for x in results),
        }
    )
    print([{k: r[k] for k in ("document_id", "title", "stock")} for r in results if not r["passed"]])


if __name__ == "__main__":
    run()
