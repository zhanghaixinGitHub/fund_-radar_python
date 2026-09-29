"""第二十一批离线核验：双同比基数、同日顺序及非业绩更正，零网络零拟合。"""

import json
import os
import shutil
import subprocess
from pathlib import Path

from app.services.fund_earnings_batch_v4 import RUNS, extract_pdf, issuer_identity
from app.services.fund_earnings_dual_comparative_v1 import review_dual_comparative, same_day_order_status
from app.services.fund_earnings_identity_v1 import profile_proof, supplement_identity
from app.services.fund_earnings_yoy_amount_v1 import unique
from app.services.fund_information_history_v1 import normalize, read, save, sha

OUT = RUNS / "20260929-earnings-v21"


def run():
    """逐件重提已保存 PDF，绑定冻结原文、代码及图片，输出待目视结果。"""
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
        if pages != d["pages"] or metadata != d["metadata"] or d["revision_issues"]:
            raise ValueError("ORIGINAL_REPLAY_OR_REVISION_CONFLICT")
        documents[d["row"]["announcementId"]] = d
    refs = [{"document": d, "proof": p} for d in documents.values() if (p := profile_proof(d))]
    identities = []
    for key, d in documents.items():
        initial = issuer_identity(d["row"], d["pages"])
        supplement = None if initial["passed"] else supplement_identity(d, refs)
        # 仅处理本批两项逐字核过的标题表达，类型与年度保持不变。
        if key in plan["literal_identity_aliases"]:
            spec = plan["literal_identity_aliases"][key]
            allowed = {
                "1212240427": ("2021年年报业绩预告", "2021年度业绩预告", "000733"),
                "1212532881": ("2022-006中航重机2021年度业绩快报", "2021年年度业绩快报", "600765"),
            }
            expected = allowed[key]
            if (normalize(d["row"]["title_plain"]), normalize(spec["body_title"]), d["row"]["secCode"]) != expected:
                raise ValueError("LITERAL_ALIAS_SCOPE_MISMATCH")
            if not initial["stock_verified"] or initial.get("reason") == "EXPLICIT_BODY_ISSUER_DIFFERS_FROM_CATALOG":
                raise ValueError("LITERAL_ALIAS_ISSUER_CONFLICT")
            supplement = {
                "passed": True,
                "rule": "FROZEN_LITERAL_SAME_YEAR_AND_CARRIER_ALIAS",
                "anchors": [unique(d["pages"], 1, spec[k]) for k in ("body_title", "legal_name", "code_quote")],
                "semantic_verified": False,
                "version_history_verified": False,
            }
        identities.append(
            {
                "document_id": key,
                "initial": initial,
                "supplement": supplement,
                "passed": initial["passed"] or bool(supplement and supplement["passed"]),
            }
        )
    passed = {i["document_id"] for i in identities if i["passed"]}
    facts = []
    for spec in plan["dual_comparative_specs"]:
        d = documents[spec["document_id"]]
        if spec["document_id"] not in passed or any(
            spec[k] != d["row"][other] for k, other in (("issuer", "secCode"), ("published_date", "published_date"))
        ):
            raise ValueError("FACT_SOURCE_IDENTITY_MISMATCH")
        facts.append(review_dual_comparative(spec, d["pages"]))
    correction = plan["shareholder_correction"]
    d = documents[correction["document_id"]]
    if correction["document_id"] not in passed or d["row"]["secCode"] != "000423":
        raise ValueError("CORRECTION_ISSUER_UNVERIFIED")
    count_anchors = [
        unique(d["pages"], 1, f"更正{label}：{correction[key]}")
        for label, key in (("前", "old_count_text"), ("后", "new_count_text"))
    ]
    correction_result = {
        **correction,
        "anchors": [unique(d["pages"], 1, q) for q in correction["quotes"]],
        "count_anchors": count_anchors,
        "published_date": d["row"]["published_date"],
        "measure": "SHAREHOLDER_COUNT",
        "is_earnings_revision": False,
        "old_original_independently_verified": False,
        "do_not_backdate_updated_report": True,
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
        "identities": identities,
        "current_identity_passed": len(passed),
        "current_identity_pending": sorted(set(documents) - passed),
        "replayed_bodies": len(documents),
        "facts": facts,
        "same_day_order": same_day_order_status(*facts),
        "shareholder_correction": correction_result,
        "renders": renders,
        "visual_review_required": True,
        "new_comparable_numeric_changes": 0,
        "all_amounts_semantically_reviewed": False,
        "training_ready": False,
        "new_fits": 0,
    }
    save(OUT / "semantic-review-candidate.json", result)
    return {k: v for k, v in result.items() if k not in {"identities", "facts", "renders", "shareholder_correction"}}


if __name__ == "__main__":
    lock = RUNS / ".earnings-preparation.lock"
    owner = {"pid": os.getpid(), "run": OUT.name, "command": "offline-semantic-review"}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    finally:
        if read(lock) == owner:
            lock.unlink()
