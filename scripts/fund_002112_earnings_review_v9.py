"""第九批原件补齐复核：分隔代码、英文公司简介和年报修订前后同口径利润。"""

import json
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from app.services.fund_earnings_asof_v1 import asof_changes
from app.services.fund_earnings_batch_v4 import extract_pdf, run_path
from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

OUT = run_path("20260929-earnings-v9")


def render(key, number, source):
    """将计划选定的原件页面保存在本批次，复核时复用已有图片并校验其哈希。"""
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
    sources = {}
    for path in plan["document_hashes"]:
        d = read(path)
        if not d["body_saved"] or sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("REVIEW_SOURCE_BYTES_CHANGED")
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), 500)
        if pages != d["pages"] or metadata != d["metadata"] or pdf_revision(metadata, d["row"]["published_date"]):
            raise ValueError("REVIEW_SOURCE_REPLAY_OR_DATE_CONFLICT")
        sources[d["row"]["announcementId"]] = d
    supplements = []
    for spec in plan["identity_specs"]:
        d = sources[spec["document_id"]]
        if d["row"]["secCode"] != spec["stock"]:
            raise ValueError("IDENTITY_ISSUER_MISMATCH")
        supplements.append(
            {
                **spec,
                "anchors": [occurrence(d["pages"], p, q) for p, q in spec["anchors"]],
                "passed": True,
                "published_date": d["row"]["published_date"],
                "source_sha256": d["receipt"]["sha256"],
                "semantic_verified": False,
                "training_ready": False,
            }
        )
    supplemented = {s["document_id"] for s in supplements}
    claims = []
    for spec in plan["money_specs"]:
        d = sources[spec["document_id"]]
        if not d["identity"]["passed"] and spec["document_id"] not in supplemented:
            raise ValueError("MONEY_SOURCE_IDENTITY_NOT_PASSED")
        claim = review_claim({**spec, "published_date": d["row"]["published_date"]}, d["pages"])
        claims.append(
            {
                **claim,
                "source_identity_verified": True,
                "source": d["receipt"],
                "revision_issues": [],
                "period_anchor": occurrence(d["pages"], spec["period_page"], spec["period_quote"]),
            }
        )
    pair = [c for c in claims if c["issuer"] == "601233"]
    instant = datetime.fromisoformat(pair[-1]["available_at"])
    early = (instant - timedelta(seconds=1)).isoformat()
    if asof_changes(pair, early) != asof_changes(pair[:1], early):
        raise ValueError("REVISION_CHANGED_PAST_QUERY")
    snapshots = [{"as_of": t, "facts": asof_changes(pair, t)} for t in (early, instant.isoformat())]
    # 这里只确认选定的归母利润不变，不能据此声称整份修订没有其他财务变化。
    if snapshots[-1]["facts"][0]["change"]["midpoint_change_cny"] != "0.00":
        raise ValueError("REVIEWED_PROFIT_CHANGED")
    old, new = sources["1216517764"], sources["1217228154"]
    if len(old["pages"]) != len(new["pages"]):
        raise ValueError("REVISION_PAGE_ALIGNMENT_NEEDS_REVIEW")
    changed = [
        i + 1 for i, (a, b) in enumerate(zip(old["pages"], new["pages"], strict=True)) if normalize(a) != normalize(b)
    ]
    result = {
        "identity_supplements": supplements,
        "main_batch_bodies_identity_passed": sum(
            d["identity"]["passed"] or key in supplemented for key, d in sources.items() if key != "1216517764"
        ),
        "claims": claims,
        "asof_snapshots": snapshots,
        "future_invariance_passed": True,
        "revision_review": {
            "original": "1216517764",
            "revised": "1217228154",
            "same_report_period": "2022-01-01/2022-12-31",
            "reviewed_parent_profit_unchanged": True,
            "normalized_text_changed_pages": changed,
            "whole_report_changes_reviewed": False,
            "not_all_revisions_are_profit_changes": True,
            "catalog_year_conflicted_summary_1216517762_still_blocked": True,
        },
        "bodies_replayed": len(sources),
        "renders": [render(k, p, sources[k]["receipt"]["path"]) for k, p in plan["render_pages"]],
        "visual_status": "PENDING",
        "training_ready": False,
        "new_fits": 0,
    }
    save(OUT / "semantic-review-candidate.json", result)
    print(
        json.dumps(
            {
                "identity_passed": result["main_batch_bodies_identity_passed"],
                "claims": len(claims),
                "changed_pages": len(changed),
            }
        )
    )


if __name__ == "__main__":
    run()
