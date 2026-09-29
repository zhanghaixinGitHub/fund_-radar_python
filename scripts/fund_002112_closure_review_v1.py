"""第25批已保存原件的独立身份核验；只生成候选结果，目视检查另记回执。"""

import shutil
import subprocess
from pathlib import Path

from app.services.fund_earnings_batch_v4 import extract_pdf, issuer_identity
from app.services.fund_earnings_english_identity_v1 import review_english_identity
from app.services.fund_earnings_identity_v1 import occurrence, profile_proof, supplement_identity
from app.services.fund_information_history_v1 import normalize, read, save, sha

from scripts.fund_002112_closure_v1 import OUT, PREVIOUS


def run():
    files = sorted((PREVIOUS / "documents").glob("*.json"))
    specs = [
        {
            "document_id": "1206062046",
            "stock": "000333",
            "year": 2018,
            "mode": "ANNUAL_PROFILE",
            "chinese_name": "美的集团股份有限公司",
            "english_name": "Midea Group Co., Ltd.",
            "cover_name_quote": "Midea Group Co., Ltd.",
            "title_quote": "The 2018 Annual Report",
            "profile_page": 9,
            "chinese_heading": "Name of the Company in Chinese",
            "english_heading": "Name of the Company in English (if any)",
            "code_page": 9,
            "code_quote": "Stock code 000333",
        },
        {
            "document_id": "1206232888",
            "stock": "000333",
            "year": 2019,
            "mode": "Q1_EARLIER_PROFILE",
            "chinese_name": "美的集团股份有限公司",
            "english_name": "Midea Group Co., Ltd.",
            "cover_name_quote": "Midea Group Co., Ltd.",
            "title_quote": "Interim Report for the First Quarter 2019",
        },
    ]
    plan = {
        "scope": "ALL_66_SAVED_V25_ORIGINALS_IDENTITY_ONLY",
        "document_hashes": {str(p): sha(p) for p in files},
        "code_hashes": {str(Path(__file__).resolve()): sha(__file__)},
        "english_specs": specs,
        "no_new_network_requests": True,
        "new_fits": 0,
        "zte_manual_anchors": {
            "1": ["二〇二三年半年度报告", "中兴通讯股份有限公司"],
            "7": ["法定中文名称中兴通讯股份有限公司", "股票代码：000063"],
        },
        "excluded_documents": {
            "1206100204": "同一控制下企业合并的历史报表追溯调整专项说明；非新的业绩报告",
            "1217616293": "保险公司偿付能力报告；非公司经营业绩更新",
        },
    }
    save(OUT / "v25-review-plan.json", plan)
    docs = {}
    for path in files:
        d = read(path)
        if sha(d["receipt"]["path"]) != d["receipt"]["sha256"]:
            raise ValueError("ORIGINAL_CHANGED")
        pages, metadata = extract_pdf(Path(d["receipt"]["path"]).read_bytes(), 500)
        if pages != d["pages"] or metadata != d["metadata"]:
            raise ValueError("INDEPENDENT_BODY_REPLAY_DIFFERENT")
        docs[d["row"]["announcementId"]] = d
    refs = [{"document": d, "proof": p} for d in docs.values() if (p := profile_proof(d))]
    proofs, render_pages = [], set()
    for key, d in sorted(docs.items()):
        direct = issuer_identity(d["row"], d["pages"])
        supplement = None if direct["passed"] else supplement_identity(d, refs)
        passed = not d["revision_issues"] and (direct["passed"] or bool(supplement and supplement["passed"]))
        proofs.append({"document_id": key, "direct": direct, "supplement": supplement, "identity_passed": passed})
        if supplement and supplement["passed"]:
            render_pages.add((key, 1))
            for a in supplement.get("anchors", []):
                render_pages.add((key, a["page"]))
            for r in supplement.get("references", []):
                render_pages.add((key, r["target_anchor"]["page"]))
    english = [review_english_identity(specs[0], docs[specs[0]["document_id"]])]
    english.append(
        review_english_identity(specs[1], docs[specs[1]["document_id"]], docs[specs[0]["document_id"]], specs[0])
    )
    render_pages.update({("1206062046", 1), ("1206062046", 9), ("1206232888", 1)})
    zte = docs["1217574510"]
    if zte["row"]["secCode"] != "000063" or zte["revision_issues"]:
        raise ValueError("ZTE_IDENTITY_CONFLICT")
    anchors = [
        occurrence(zte["pages"], int(page), q) for page, values in plan["zte_manual_anchors"].items() for q in values
    ]
    render_pages.update({("1217574510", 1), ("1217574510", 7), ("1217616293", 1)})
    for key in plan["excluded_documents"]:
        if not any(t in normalize(docs[key]["pages"][0]) for t in ["偿付能力", "追溯调整"]):
            raise ValueError("PURPOSE_EXCLUSION_NOT_IN_SOURCE")
        render_pages.add((key, 1))
    for proof in proofs:
        if proof["document_id"] in {"1206062046", "1206232888", "1217574510"}:
            proof["identity_passed"] = True
    renders = []
    for key, number in sorted(render_pages):
        target = OUT / "v25-renders" / f"{key}-p{number}.png"
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
                    "1800",
                    "-png",
                    docs[key]["receipt"]["path"],
                    str(target.with_suffix("")),
                ],
                check=True,
                capture_output=True,
                timeout=45,
            )
        renders.append({"document_id": key, "page": number, "path": str(target), "sha256": sha(target)})
    result = {
        "originals_independently_replayed": len(docs),
        "identities": proofs,
        "english_identity_proofs": english,
        "zte_anchors": anchors,
        "identity_passed": sum(p["identity_passed"] for p in proofs),
        "purpose_exclusions": plan["excluded_documents"],
        "version_restriction": {"1210551144": "Updated carrier remains at actual later publication date"},
        "renders": renders,
        "visual_review_pending": True,
        "full_semantics_verified": False,
        "new_fits": 0,
    }
    save(OUT / "v25-review-candidate.json", result)
    print({"replayed": len(docs), "identity_passed": result["identity_passed"], "renders": len(renders)})


if __name__ == "__main__":
    run()
