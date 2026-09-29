"""对登记的科伦药业业绩链重放原件、金额、期间与可用时间，输出审核候选。"""

import json
import shutil
import subprocess

import pypdfium2 as pdfium
from app.services.fund_earnings_evidence_v1 import compare_claims, review_claim
from app.services.fund_information_history_v1 import (
    OUT as PREVIOUS,
)
from app.services.fund_information_history_v1 import (
    normalize,
    pdf_revision,
    read,
    save,
    sha,
)

from scripts.fund_002112_earnings_batch_v1 import OUT

# 逐份对照目录与原件标题后登记的明确写法；不使用相似度阈值替代身份核查。
IDENTITY_TITLES = {
    "1211372225": "2021年第三季度报告",
    "1212686416": "二〇二一年度报告",
    "1213133672": "2022年第一季度报告",
    "1214884010": "2022年第三季度报告",
    "1214903105": "2022年第三季度报告",
    "1214938883": "2022年第三季度报告",
    "1214940220": "2022年第三季度报告",
    "1214945494": "2022年第三季度报告",
    "1214965153": "2022年第三季度报告",
    "1214966596": "2022年第三季度报告",
    "1214976769": "2022年第三季度报告",
}


def extract(path):
    """独立打开原 PDF，逐页提取并释放资源，不借用旧文字替代原件核查。"""
    with pdfium.PdfDocument(path) as document:
        pages, metadata = [], document.get_metadata_dict()
        for page in document:
            text = page.get_textpage()
            try:
                pages.append(text.get_text_range())
            finally:
                text.close()
                page.close()
    return pages, metadata


def identity_audit():
    results = []
    for path in sorted((OUT / "documents").glob("*.json")):
        item = read(path)
        if not item["body_saved"]:
            results.append({"path": str(path), "passed": False, "reason": item["reason"]})
            continue
        receipt, row = item["receipt"], item["row"]
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("IDENTITY_PDF_CHANGED")
        pages, metadata = extract(receipt["path"])
        if [normalize(p) for p in pages] != [normalize(p) for p in item["pages"]]:
            raise ValueError("BODY_REPLAY_MISMATCH")
        title = IDENTITY_TITLES.get(row["announcementId"], row["title_plain"])
        first = normalize("".join(pages[:2]))
        passed = row["secCode"] in first and normalize(title) in first
        results.append(
            {
                "document_id": row["announcementId"],
                "stock": row["secCode"],
                "catalog_title": row["title_plain"],
                "body_title_anchor": title,
                "passed": passed,
                "source_sha256": receipt["sha256"],
                "metadata_issues": pdf_revision(metadata, row["published_date"]),
                "body_replay_passed": True,
                "semantic_verified": False,
            }
        )
    save(OUT / "identity-audit.json", {"documents": results, "passed": sum(r["passed"] for r in results)})
    return results


def run():
    identities = {r["document_id"]: r for r in identity_audit() if "document_id" in r}
    inventory = {r["document_id"]: r for r in read(PREVIOUS / "semantic-inventory.json")}
    source0 = read(OUT / "documents/1213133672.json")
    sources = {
        "1213133672": {
            "source": source0["receipt"],
            "pages": source0["pages"],
            "published_date": source0["row"]["published_date"],
        },
    }
    for key in ("1213972395", "1214445963"):
        item = inventory[key]
        sources[key] = {
            "source": item["source"],
            "pages": read(item["text_path"])["pages"],
            "published_date": item["published_date"],
        }
    common = {
        "issuer": "002422",
        "period_start": "2022-01-01",
        "period_end": "2022-06-30",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_ATTRIBUTABLE_TO_PARENT",
        "currency": "CNY",
    }
    specs = [
        {
            **common,
            "document_id": "1213133672",
            "page": 4,
            "unit": "万元",
            "kind": "FORECAST",
            "values": ["66,530", "76,387"],
            "quote": "归属于上市公司股东的净利润（万元）66,530 -- 76,387 49,281.71 增长35.00% -- 55.00%",
            "period_anchor": "三、对2022年1-6月经营业绩的预计",
            "identity_title": "2022年第一季度报告",
        },
        {
            **common,
            "document_id": "1213972395",
            "page": 1,
            "unit": "万元",
            "kind": "FORECAST",
            "values": ["82,000", "89,000"],
            "quote": "盈利：82,000万元–89,000万元",
            "period_anchor": "业绩预告期间：2022年1月1日-2022年6月30日",
            "identity_title": "2022年半年度业绩预告修正公告",
            "revises": "1213133672",
            "revision_reference": "公司于2022年4月27日披露的《2022年第一季度报告》",
        },
        {
            **common,
            "document_id": "1214445963",
            "page": 1,
            "unit": "元",
            "kind": "REPORTED_RESULT",
            "values": ["868,078,939.00"],
            "quote": "归属于上市公司股东的净利润（元）868,078,939.00 492,817,087.00 76.15%",
            "period_anchor": "2022年半年度报告摘要",
            "identity_title": "2022年半年度报告摘要",
        },
    ]
    # 显式引用在原文中含括号，保留连续原文，避免由模糊匹配伪造引用关系。
    specs[1]["revision_reference"] = (
        "四川科伦药业股份有限公司(以下简称“公司”)于2022年4月27日披露的"
        "《2022年第一季度报告》中对2022年1-6月经营业绩的预计为："
    )
    save(OUT / "earnings-review-specs.json", specs)
    claims, rendered = [], []
    render_dir = OUT / "review-renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        key = spec["document_id"]
        source = sources[key]
        raw = source["source"]
        if sha(raw["path"]) != raw["sha256"]:
            raise ValueError("SEMANTIC_SOURCE_CHANGED")
        pages, metadata = extract(raw["path"])
        if [normalize(p) for p in pages] != [normalize(p) for p in source["pages"]]:
            raise ValueError("SEMANTIC_PAGE_REPLAY_MISMATCH")
        front = normalize("".join(pages[:2]))
        if "002422" not in front or normalize(spec["identity_title"]) not in front:
            raise ValueError("SEMANTIC_SOURCE_IDENTITY_FAILED")
        if key in identities and not identities[key]["passed"]:
            raise ValueError("CONTEXT_IDENTITY_FAILED")
        if normalize(spec["period_anchor"]) not in normalize(pages[spec["page"] - 1]):
            raise ValueError("PERIOD_ANCHOR_FAILED")
        if spec.get("revision_reference") and normalize(spec["revision_reference"]) not in front:
            raise ValueError("REVISION_REFERENCE_FAILED")
        claim = review_claim({**spec, "published_date": source["published_date"]}, pages)
        claim.update(
            {
                "source": raw,
                "source_identity_verified": True,
                "revision_issues": pdf_revision(metadata, source["published_date"]),
                "independent_page_replay_passed": True,
                "visual_review_status": "PENDING_EXPLICIT_REVIEW_RECEIPT",
            }
        )
        claims.append(claim)
        page_numbers = {1, spec["page"]} | ({2} if key == "1213972395" else set())
        for number in sorted(page_numbers):
            target = render_dir / f"{key}-p{number}.png"
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
                        raw["path"],
                        str(target.with_suffix("")),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=45,
                )
            rendered.append({"document_id": key, "page": number, "path": str(target), "sha256": sha(target)})
    # 两份原件分别支持旧预测与新预测；不将修正公告追述的旧值提前使用。
    pairs = [
        compare_claims(claims[0], claims[1], claims[1]["available_at"]),
        compare_claims(claims[1], claims[2], claims[2]["available_at"]),
    ]
    result = {
        "claims": claims,
        "comparisons": pairs,
        "renders": rendered,
        "new_fits": 0,
        "training_ready": False,
        "limitation": "One registered issuer-period chain; no general extraction precision or predictive gain claim",
    }
    save(OUT / "earnings-review-candidate.json", result)
    print(
        json.dumps(
            {"claims": len(claims), "comparisons": pairs, "rendered_pages": len(rendered)}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    run()
