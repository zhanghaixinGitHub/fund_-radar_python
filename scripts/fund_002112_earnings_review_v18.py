"""第十八批原件字段重放：同日全文/正文去重，声明与计算变化分开。"""

import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from app.services.fund_earnings_annual_forecast_v1 import comparison_readiness, review_annual_forecast
from app.services.fund_earnings_batch_v4 import extract_pdf, issuer_identity
from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

from scripts.fund_002112_earnings_targeted_v18 import OUT, RUNS, check_plan


def render(key, page, source):
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
    check_plan()
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
    verified = {k for k, d in documents.items() if issuer_identity(d["row"], d["pages"])["passed"]}
    supplements = []
    for s in plan["identity_specs"]:
        d, ref = documents[s["document_id"]], documents[s["reference_document_id"]]
        if (
            s["reference_document_id"] not in verified
            or d["row"]["secCode"] != ref["row"]["secCode"]
            or d["row"]["published_date"] != ref["row"]["published_date"]
        ):
            raise ValueError("IDENTITY_REFERENCE_SCOPE_CONFLICT")
        anchors = [occurrence(d["pages"], p, q) for p, q in s["anchors"]]
        ref_anchors = [occurrence(ref["pages"], p, q) for p, q in s["reference_anchors"]]
        if any(normalize(s["legal_name"]) not in normalize(x["pages"][0]) for x in (d, ref)):
            raise ValueError("LEGAL_NAME_NOT_SHARED_ON_COVERS")
        supplements.append(
            {
                **s,
                "anchors": anchors,
                "reference_anchors": ref_anchors,
                "source_sha256": d["receipt"]["sha256"],
                "reference_sha256": ref["receipt"]["sha256"],
                "published_date": d["row"]["published_date"],
                "identity_passed": True,
                "original_initial_result_preserved": True,
                "version_history_verified": False,
            }
        )
        verified.add(s["document_id"])
    forecasts = []
    for s in plan["forecast_specs"]:
        d = documents[s["document_id"]]
        if s["document_id"] not in verified or d["row"]["secCode"] != s["issuer"]:
            raise ValueError("FORECAST_IDENTITY_NOT_VERIFIED")
        fact = review_annual_forecast(
            {**s, "source_title": d["row"]["title_plain"], "published_date": d["row"]["published_date"]}, d["pages"]
        )
        forecasts.append({**fact, "source": d["receipt"], "source_identity_verified": True, "revision_issues": []})
    unique, duplicate_groups = [], []
    for stock in sorted({f["issuer"] for f in forecasts}):
        group = [f for f in forecasts if f["issuer"] == stock]
        selected = [f for f in group if f["source_title"].endswith("正文")]
        if len(group) != 2 or len(selected) != 1:
            raise ValueError("FULL_SHORT_PAIR_REQUIRED")
        fields = (
            "period_start",
            "period_end",
            "amount_values_in_source_unit",
            "unit",
            "yoy_percent",
            "comparative_value",
            "published_date",
            "available_at",
        )
        if any(group[0][k] != group[1][k] for k in fields):
            raise ValueError("FULL_SHORT_FORECAST_CONFLICT")
        unique.append(selected[0])
        duplicate_groups.append(
            {
                "issuer": stock,
                "canonical_document_id": selected[0]["document_id"],
                "supporting_document_ids": sorted(f["document_id"] for f in group),
                "economic_fact_count": 1,
                "same_public_date_same_issuer_same_annual_metric_values": True,
            }
        )
    statements, readiness = [], []
    for s in plan["statement_specs"]:
        d = documents[s["document_id"]]
        forecast = next(f for f in unique if f["issuer"] == s["issuer"])
        current = next(c for c in read(plan["prior_current_claims"])["claims"] if c["document_id"] == s["document_id"])
        if (
            s["document_id"] not in verified
            or d["row"]["secCode"] != s["issuer"]
            or s["referenced_date"] != forecast["published_date"]
            or current["issuer"] != s["issuer"]
        ):
            raise ValueError("STATEMENT_REFERENCE_MISMATCH")
        anchor = occurrence(d["pages"], s["page"], s["quote"])
        if len(anchor["offsets"]) != 1 or "不存在差异" not in anchor["text"]:
            raise ValueError("EXPLICIT_STATEMENT_REQUIRED")
        y, m, day = map(int, s["referenced_date"].split("-"))
        if f"{y}年{m}月{day}日" not in anchor["text"] or f"{y}年第三季度报告" not in anchor["text"]:
            raise ValueError("STATEMENT_REFERENCE_ANCHOR_CONFLICT")
        statements.append(
            {
                **s,
                "anchor": anchor,
                "source": d["receipt"],
                "published_date": d["row"]["published_date"],
                "available_at": current["available_at"],
                "referenced_original_id": forecast["document_id"],
                "referenced_original_present": True,
                "statement_type": "ISSUER_REPORTS_NO_DIFFERENCE_FROM_PRIOR_FORECAST",
                "numeric_change": None,
                "is_independently_computed_change": False,
                "training_ready": False,
            }
        )
        readiness.append(
            {
                "issuer": s["issuer"],
                "older": forecast["document_id"],
                "newer": s["document_id"],
                **comparison_readiness(forecast, current),
                "numeric_change": None,
            }
        )
    snapshots = []
    for stock in sorted({f["issuer"] for f in unique}):
        facts = [
            {"kind": "ORIGINAL_FORECAST", "id": f["document_id"], "available_at": f["available_at"]}
            for f in unique
            if f["issuer"] == stock
        ]
        future = [
            {"kind": "ISSUER_COMPARISON_STATEMENT", "id": s["document_id"], "available_at": s["available_at"]}
            for s in statements
            if s["issuer"] == stock
        ]
        for item in [*facts, *future]:
            at = datetime.fromisoformat(item["available_at"])
            for query in (at - timedelta(seconds=1), at):
                before = [f for f in facts if datetime.fromisoformat(f["available_at"]) <= query]
                selected = [f for f in [*facts, *future] if datetime.fromisoformat(f["available_at"]) <= query]
                if query < datetime.fromisoformat(future[0]["available_at"]) and before != selected:
                    raise ValueError("FUTURE_STATEMENT_CHANGED_PAST")
                snapshots.append({"issuer": stock, "as_of": query.isoformat(), "available_facts": selected})
    renders = [
        render(s["document_id"], s["page"], documents[s["document_id"]]["receipt"]["path"])
        for s in plan["render_pages"]
    ]
    result = {
        "replayed_bodies": len(documents),
        "identity_passed": len(verified),
        "identity_supplements": supplements,
        "forecast_document_fields": forecasts,
        "unique_annual_forecasts": unique,
        "duplicate_groups": duplicate_groups,
        "issuer_comparison_statements": statements,
        "comparison_readiness": readiness,
        "asof_snapshots": snapshots,
        "future_invariance_passed": True,
        "renders": renders,
        "visual_review_required": True,
        "new_comparable_numeric_changes": 0,
        "new_fits": 0,
        "training_ready": False,
    }
    save(OUT / "semantic-review-candidate.json", result)
    return {
        "bodies": len(documents),
        "identity_passed": len(verified),
        "annual_forecasts": len(unique),
        "issuer_statements": len(statements),
        "new_comparable_numeric_changes": 0,
        "render_pages": len(renders),
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
