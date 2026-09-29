"""核对本批真实遗漏的国联股份预告、正式结果及同比列，区分公司主体与报告性质。"""

import json
import shutil
import subprocess

from app.services.fund_earnings_evidence_v1 import compare_claims, review_claim
from app.services.fund_earnings_history_v2 import review_yoy
from app.services.fund_earnings_semantics_v2 import review_table_yoy
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

from scripts.fund_002112_earnings_batch_v2 import OUT, check_plan
from scripts.fund_002112_earnings_review_v1 import extract


def source(key):
    path = OUT / "documents" / (key + ".json")
    if not path.exists():
        path = OUT / "additional-documents" / (key + ".json")
    value = read(path)
    if not value["body_saved"] or sha(value["receipt"]["path"]) != value["receipt"]["sha256"]:
        raise ValueError("REVIEW_SOURCE_UNAVAILABLE")
    pages, metadata = extract(value["receipt"]["path"])
    if [normalize(p) for p in pages] != [normalize(p) for p in value["pages"]]:
        raise ValueError("REVIEW_PDF_REPLAY_MISMATCH")
    return value, pages, metadata


def render(key, page_number, source_path):
    target = OUT / "review-renders" / f"{key}-p{page_number}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(
            [
                shutil.which("pdftoppm"),
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-singlefile",
                "-scale-to",
                "1500",
                "-png",
                source_path,
                str(target.with_suffix("")),
            ],
            check=True,
            capture_output=True,
            timeout=45,
        )
    return {"document_id": key, "page": page_number, "path": str(target), "sha256": sha(target)}


def run():
    check_plan()
    common = {
        "issuer": "603613",
        "period_start": "2022-01-01",
        "period_end": "2022-03-31",
        "basis": "CONSOLIDATED_ATTRIBUTABLE_TO_PARENT",
        "currency": "CNY",
        "audit_status": "UNAUDITED",
    }
    header = "单位：元币种：人民币项目本报告期本报告期比上年同期增减变动幅度(%)"
    money_specs = [
        {
            **common,
            "document_id": "1212699002",
            "page": 1,
            "metric": "PARENT_NET_PROFIT",
            "kind": "FORECAST",
            "unit": "万元",
            "values": ["14,500.00", "15,300.00"],
            "quote": "1、经财务部门初步测算，公司预计2022年第一季度实现归属于上市公司股东的净利润为"
            "14,500.00万元到15,300.00万元，与上年同期（法定披露数据）相比，预计增加"
            "6,698.01万元到7,498.01万元，同比增加85.85%到96.10%。",
        },
        {
            **common,
            "document_id": "1212699002",
            "page": 2,
            "metric": "REVENUE",
            "basis": "CONSOLIDATED_REVENUE",
            "kind": "FORECAST",
            "unit": "万元",
            "values": ["1,160,000.00", "1,220,000.00"],
            "quote": "本期预计营业收入1,160,000.00万元至1,220,000.00万元，同比增加90.91%至100.78%。",
        },
        {
            **common,
            "document_id": "1213020054",
            "page": 1,
            "metric": "PARENT_NET_PROFIT",
            "kind": "REPORTED_RESULT",
            "unit": "元",
            "values": ["155,144,879.59"],
            "quote": header + "营业收入12,137,927,945.28 99.76 归属于上市公司股东的净利润155,144,879.59 98.85",
        },
        {
            **common,
            "document_id": "1213020054",
            "page": 1,
            "metric": "REVENUE",
            "basis": "CONSOLIDATED_REVENUE",
            "kind": "REPORTED_RESULT",
            "unit": "元",
            "values": ["12,137,927,945.28"],
            "quote": header + "营业收入12,137,927,945.28 99.76",
        },
        {
            **common,
            "period_end": "2022-06-30",
            "document_id": "1214025248",
            "page": 1,
            "metric": "PARENT_NET_PROFIT",
            "kind": "FORECAST",
            "unit": "万元",
            "values": ["41,800.00", "42,200.00"],
            "quote": "1、经财务部门初步测算，预计2022年上半年度实现归属于上市公司所有者的净利润为"
            "41,800.00万元到42,200.00万元，与上年同期相比，将增加20,182.97万元到20,582.97万元，"
            "同比增加93.37%到95.22%。",
        },
    ]
    save(OUT / "review-money-specs.json", money_specs)
    sources = {key: source(key) for key in ("1212699002", "1213020054", "1214025248")}
    claims, yoy_claims, renders = [], [], []
    for spec in money_specs:
        item, pages, metadata = sources[spec["document_id"]]
        if not item["identity"]["passed"]:
            raise ValueError("EARNINGS_IDENTITY_NOT_VERIFIED")
        claim = review_claim({**spec, "published_date": item["row"]["published_date"]}, pages)
        claim.update(
            {
                "source": item["receipt"],
                "source_identity_verified": True,
                "revision_issues": pdf_revision(metadata, item["row"]["published_date"]),
            }
        )
        claims.append(claim)
    for index, values in ((0, ["85.85", "96.10"]), (1, ["90.91", "100.78"]), (4, ["93.37", "95.22"])):
        spec = money_specs[index]
        item, pages, _ = sources[spec["document_id"]]
        yoy_claims.append(
            review_yoy(
                {
                    **spec,
                    "unit": "PERCENT",
                    "values": values,
                    "comparison": "YEAR_ON_YEAR",
                    "direction_word": "原文有符号数值",
                    "original_direction_word": "增加",
                    "base_state": "UNKNOWN",
                    "published_date": item["row"]["published_date"],
                },
                pages,
            )
        )
    for index, quote, values in (
        (2, "归属于上市公司股东的净利润155,144,879.59 98.85", ["155144879.59", "98.85"]),
        (3, "营业收入12,137,927,945.28 99.76", ["12137927945.28", "99.76"]),
    ):
        spec = money_specs[index]
        yoy_claims.append(
            review_table_yoy(
                {
                    **spec,
                    "unit": "PERCENT",
                    "comparison": "YEAR_ON_YEAR",
                    "header": "本报告期比上年同期增减变动幅度(%)",
                    "row_quote": quote,
                    "row_values": values,
                    "value_index": 1,
                    "value": values[1],
                },
                sources[spec["document_id"]][1],
            )
        )
    pairs = [
        compare_claims(claims[0], claims[2], claims[2]["available_at"]),
        compare_claims(claims[1], claims[3], claims[3]["available_at"]),
    ]
    try:
        compare_claims(claims[0], claims[4], claims[4]["available_at"])
    except ValueError as exc:
        period_check = {"rejected": str(exc) == "INCOMPARABLE_EARNINGS_CLAIMS", "reason": str(exc)}
    else:
        raise ValueError("CROSS_PERIOD_COMPARISON_NOT_REJECTED")
    for key, (item, _, _) in sources.items():
        for number in [1, 2] if key != "1213020054" else [1]:
            renders.append(render(key, number, item["receipt"]["path"]))
    # 两份中文年报在公司简介页才出现代码，分别用原文公司名称和代码共同确认。
    identities = []
    for key, number, company, code in (
        ("1212975100", 10, "阳光电源股份有限公司", "300274"),
        ("1213171977", 8, "重庆智飞生物制品股份有限公司", "300122"),
    ):
        item, pages, _ = source(key)
        target = normalize(pages[number - 1])
        if company not in normalize(pages[0]) or "股票代码" + code not in target or company not in target:
            raise ValueError("FULL_REPORT_COMPANY_IDENTITY_FAILED")
        identities.append(
            {
                "id": key,
                "passed": True,
                "stock": code,
                "page": number,
                "company": company,
                "source_sha256": item["receipt"]["sha256"],
                "proof": "SAME_COMPANY_ON_COVER_AND_CODE_IN_COMPANY_INFORMATION",
            }
        )
        renders.extend(render(key, n, item["receipt"]["path"]) for n in [1, number])
    chinese, chinese_pages, _ = source("1212975100")
    english, english_pages, _ = source("1213790493")
    english_name = "SungrowPowerSupplyCo.,Ltd."
    if english_name not in normalize(chinese_pages[9]) or english_name not in normalize(english_pages[0]):
        raise ValueError("ENGLISH_LEGAL_NAME_NOT_LINKED")
    if "2021AnnualReport(ConciseVersioninEnglish)" not in normalize(english_pages[0]):
        raise ValueError("ENGLISH_PERIOD_OR_DOCUMENT_TYPE_MISMATCH")
    identities.append(
        {
            "id": "1213790493",
            "passed": True,
            "stock": "300274",
            "source_sha256": english["receipt"]["sha256"],
            "linked_chinese": "1212975100",
            "linked_chinese_sha256": chinese["receipt"]["sha256"],
            "proof": "EXPLICIT_ENGLISH_LEGAL_NAME_IN_CODE_VERIFIED_CHINESE_ANNUAL_REPORT",
            "same_document_or_same_values_not_assumed": True,
        }
    )
    renders.append(render("1213790493", 1, english["receipt"]["path"]))
    save(
        OUT / "identity-supplement-candidate.json",
        {
            "verified": identities,
            "excluded": [
                {"id": "1212942451", "reason": "SPONSOR_SUPERVISION_REPORT_NOT_COMPANY_EARNINGS"},
                {
                    "id": "1213136108",
                    "catalog_stock": "601318",
                    "body_company": "平安银行股份有限公司",
                    "reason": "SUBSIDIARY_REPORT_NOT_PARENT_EARNINGS_DO_NOT_REASSIGN",
                },
            ],
            "visual_status": "PENDING",
        },
    )
    result = {
        "claims": claims,
        "yoy_fields": yoy_claims,
        "comparisons": pairs,
        "cross_period_check": period_check,
        "renders": renders,
        "identity_supplement": identities,
        "training_ready": False,
        "new_fits": 0,
        "visual_status": "PENDING",
        "limitation": "Selected fields only; no extraction accuracy or prediction improvement claim",
    }
    save(OUT / "review-candidate.json", result)
    print(
        json.dumps(
            {
                "money_fields": len(claims),
                "yoy_fields": len(yoy_claims),
                "comparisons": pairs,
                "renders": len(renders),
                "identity_supplements": len(identities),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
