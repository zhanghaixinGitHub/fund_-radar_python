"""第七批只读重放：公司身份补证、同期间预告到实际及更正附件的时间边界。"""

import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal

from app.services.fund_earnings_asof_v1 import asof_changes
from app.services.fund_earnings_batch_v4 import extract_pdf, run_path
from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_identity_v1 import occurrence, profile_proof, supplement_identity
from app.services.fund_earnings_yoy_v3 import review_yoy
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

OUT = run_path("20260929-earnings-v7")


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
        document = read(path)
        if not document["body_saved"]:
            continue
        receipt = document["receipt"]
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("SOURCE_BYTES_CHANGED")
        with open(receipt["path"], "rb") as stream:
            pages, metadata = extract_pdf(stream.read(), 500)
        if [normalize(p) for p in pages] != [normalize(p) for p in document["pages"]] or metadata != document[
            "metadata"
        ]:
            raise ValueError("SOURCE_REPLAY_CHANGED")
        sources[document["row"]["announcementId"]] = document
    refs = [{"document": d, "proof": profile_proof(d)} for d in sources.values() if profile_proof(d)]
    supplements = [supplement_identity(d, refs) for d in sources.values() if not d["identity"]["passed"]]
    by_id = {s["document_id"]: s for s in supplements}
    claims, yoys = [], []
    for spec in plan["money_specs"]:
        d = sources[spec["document_id"]]
        if not (d["identity"]["passed"] or by_id.get(spec["document_id"], {}).get("passed")):
            raise ValueError("MONEY_SOURCE_IDENTITY_UNRESOLVED")
        if pdf_revision(d["metadata"], d["row"]["published_date"]):
            raise ValueError("MONEY_SOURCE_REVISION_CONFLICT")
        spec = {**spec, "published_date": d["row"]["published_date"]}
        claim = {
            **review_claim(spec, d["pages"]),
            "source": d["receipt"],
            "source_identity_verified": True,
            "revision_issues": [],
        }
        claims.append(claim)
        if spec.get("reported_yoy"):
            yoys.append(
                {
                    **review_yoy(
                        {
                            **spec,
                            "unit": "PERCENT",
                            "comparison": "YEAR_ON_YEAR",
                            "values": [spec["reported_yoy"]],
                            "direction_word": "原文有符号数值",
                            "base_state": "POSITIVE",
                        },
                        d["pages"],
                    ),
                    "source": d["receipt"],
                }
            )
    snapshots = []
    for issuer in ("002049", "300676"):
        selected = [c for c in claims if c["issuer"] == issuer]
        latest = max(datetime.fromisoformat(c["available_at"]) for c in selected)
        early = (latest - timedelta(seconds=1)).isoformat()
        previous = [c for c in selected if c["kind"] == "FORECAST"]
        if asof_changes(selected, early) != asof_changes(previous, early):
            raise ValueError("FUTURE_RESULT_CHANGED_PAST")
        snapshots.extend(
            {"issuer": issuer, "as_of": t, "facts": asof_changes(selected, t)} for t in (early, latest.isoformat())
        )
    correction = [c for c in claims if c["issuer"] == "300732"]
    # 两份附件同日公开；按文件编号排出先后或回填四月都会制造虚假的已知信息。
    if asof_changes(correction, "2021-04-29T08:00:00+08:00"):
        raise ValueError("REISSUED_ATTACHMENT_BACKDATED")
    ambiguous = asof_changes(correction, "2021-05-25T08:00:00+08:00")
    if not all(f["status"] == "AMBIGUOUS_DISCLOSURE_ORDER" for f in ambiguous):
        raise ValueError("SAME_DAY_VERSION_ORDER_GUESSED")
    differences = []
    for metric in ("PARENT_NET_PROFIT", "TOTAL_ASSETS"):
        before = next(c for c in correction if c["metric"] == metric and c["document_id"] == "1210071572")
        after = next(c for c in correction if c["metric"] == metric and c["document_id"] == "1210071573")
        differences.append(
            {
                "metric": metric,
                "before_cny": before["money"][0]["cny"],
                "after_cny": after["money"][0]["cny"],
                "change_cny": str(Decimal(after["money"][0]["cny"]) - Decimal(before["money"][0]["cny"])),
                "role": "SAME_DAY_CORRECTION_ATTACHMENT_COMPARISON_NOT_ORDERED_EVENT_SIGNAL",
            }
        )
    d = sources["1212767208"]
    conflict = {
        "document_id": "1212767208",
        "catalog_published_date": d["row"]["published_date"],
        "title_anchor": occurrence(d["pages"], 1, "2022年第一季度业绩预增公告"),
        "body_signature_anchor": occurrence(d["pages"], 2, "2021年4月1日"),
        "source": d["receipt"],
        "status": "STOPPED_BODY_SIGNATURE_YEAR_CONFLICT",
        "may_be_typographical_error_but_not_assumed": True,
        "training_ready": False,
        "new_requests": 0,
        "repeat_unchanged_source_search": False,
    }
    result = {
        "identity_supplements": supplements,
        "identity_supplement_counts": dict(Counter(s.get("rule", s.get("reason")) for s in supplements)),
        "identity_supplements_passed": sum(s["passed"] for s in supplements),
        "independently_reextracted_bodies": len(sources),
        "claims": claims,
        "reported_yoy": yoys,
        "asof_snapshots": snapshots,
        "future_invariance_passed": True,
        "correction_attachment_differences": differences,
        "same_day_order": ambiguous,
        "may_backfill_original_report": False,
        "source_date_conflict": conflict,
        "renders": [render(k, p, sources[k]["receipt"]["path"]) for k, p in plan["render_pages"]],
        "visual_status": "PENDING",
        "new_fits": 0,
        "training_ready": False,
    }
    save(OUT / "semantic-review-candidate.json", result)
    print(
        json.dumps(
            {
                "bodies_replayed": len(sources),
                "identity_supplement_passed": result["identity_supplements_passed"],
                "claims": len(claims),
                "yoy": len(yoys),
                "renders": len(result["renders"]),
            }
        )
    )


if __name__ == "__main__":
    run()
