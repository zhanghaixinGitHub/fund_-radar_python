"""只读重放第三批亏损、单季/累计和追溯调整字段；不训练、不请求网络。"""

import json
import shutil
import subprocess

from app.services.fund_earnings_batch_v3 import EarningsBatch
from app.services.fund_earnings_evidence_v1 import compare_claims, review_claim
from app.services.fund_earnings_history_v2 import review_yoy
from app.services.fund_earnings_semantics_v2 import review_table_yoy
from app.services.fund_earnings_signed_facts_v1 import review_loss_claim
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

from scripts.fund_002112_earnings_review_v1 import extract

BATCH = EarningsBatch("20260928-earnings-v3")
OUT = BATCH.out


def source(key):
    """校验原件哈希后在独立进程中重新取文，不用缓存正文充当重放。"""
    item = read(OUT / "documents" / (key + ".json"))
    if not item["body_saved"] or sha(item["receipt"]["path"]) != item["receipt"]["sha256"]:
        raise ValueError("REVIEW_SOURCE_UNAVAILABLE")
    pages, metadata = extract(item["receipt"]["path"])
    if [normalize(p) for p in pages] != [normalize(p) for p in item["pages"]]:
        raise ValueError("REVIEW_PDF_REPLAY_MISMATCH")
    if pdf_revision(metadata, item["row"]["published_date"]):
        raise ValueError("REVIEW_METADATA_CONFLICT")
    return item, pages


def anchored(pages, number, text):
    """保留明确的页和唯一连续引文，不以散落关键词组合代替原文。"""
    page, quote = normalize(pages[number - 1]), normalize(text)
    if not quote or page.count(quote) != 1:
        raise ValueError("SUPPLEMENT_ANCHOR_NOT_UNIQUE")
    return {"page": number, "offset": page.index(quote), "text": quote}


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
    BATCH.check_plan()
    common = {
        "period_start": "2022-01-01",
        "period_end": "2022-06-30",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_ATTRIBUTABLE_TO_PARENT",
        "currency": "CNY",
        "audit_status": "UNAUDITED",
    }
    actual_row = (
        "归属于上市公司股东的净利润（元） 1,318,723,734.41 836,162,558.07 836,712,748.16 57.61 "
        "5,854,967,785.02 2,299,254,791.95 2,299,694,538.33 154.60"
    )
    specs = [
        {
            **common,
            "issuer": "002714",
            "document_id": "1214043424",
            "page": 1,
            "kind": "FORECAST",
            "unit": "亿元",
            "values": ["63.00", "69.00"],
            "quote": "归属于上市公司股东的净利润亏损：63.00亿元—69.00亿元盈利：95.26亿元"
            "比上年同期下降：166.13%—172.43%",
            "loss_anchor": "亏损：63.00亿元—69.00亿元",
        },
        {
            **common,
            "issuer": "600026",
            "document_id": "1213983584",
            "page": 1,
            "kind": "FORECAST",
            "unit": "万元",
            "values": ["11,000", "18,000"],
            "quote": "2、中远海运能源运输股份有限公司（以下简称“本公司”，连同其附属公司，简称“本集团”）"
            "预计本集团二〇二二年上半年实现归属于上市公司股东的净利润为人民币"
            "11,000万元~18,000万元，同比下降66.9%~79.8%。",
        },
        {
            **common,
            "issuer": "000933",
            "period_end": "2022-09-30",
            "document_id": "1214741329",
            "page": 1,
            "kind": "FORECAST",
            "unit": "万元",
            "values": ["585,000.00"],
            "quote": "归属于上市公司股东的净利润盈利：585,000.00万元盈利：229,969.45万元比上年同期增加154.38%",
        },
        {
            **common,
            "issuer": "000933",
            "period_start": "2022-07-01",
            "period_end": "2022-09-30",
            "document_id": "1214741329",
            "page": 1,
            "kind": "FORECAST",
            "unit": "万元",
            "values": ["131,000.00"],
            "quote": "归属于上市公司股东的净利润盈利：131,000.00万元盈利：83,671.27万元比上年同期增加56.57%",
        },
        {
            **common,
            "issuer": "000933",
            "period_end": "2022-09-30",
            "document_id": "1214815239",
            "page": 2,
            "kind": "REPORTED_RESULT",
            "unit": "元",
            "values": ["5,854,967,785.02"],
            "quote": actual_row,
        },
        {
            **common,
            "issuer": "000933",
            "period_start": "2022-07-01",
            "period_end": "2022-09-30",
            "document_id": "1214815239",
            "page": 2,
            "kind": "REPORTED_RESULT",
            "unit": "元",
            "values": ["1,318,723,734.41"],
            "quote": actual_row,
        },
    ]
    sources = {key: source(key) for key in {s["document_id"] for s in specs}}
    claims, yoy, renders = [], [], []
    for spec in specs:
        item, pages = sources[spec["document_id"]]
        if not item["identity"]["passed"]:
            raise ValueError("SOURCE_IDENTITY_NOT_VERIFIED")
        spec = {**spec, "published_date": item["row"]["published_date"]}
        claim = (review_loss_claim if "loss_anchor" in spec else review_claim)(spec, pages)
        claims.append({**claim, "source": item["receipt"], "source_identity_verified": True, "revision_issues": []})
    for index, values, direction in (
        (0, ["166.13", "172.43"], "下降"),
        (1, ["66.9", "79.8"], "下降"),
        (2, ["154.38"], "原文有符号数值"),
        (3, ["56.57"], "原文有符号数值"),
    ):
        spec = specs[index]
        item, pages = sources[spec["document_id"]]
        # 前两项原文同时明确同比下降及上年盈利。后两项保留“增加”原词和所报值，
        # 不扩大旧核验器的方向词集合，不从金额自行计算同比。
        base_proof = (
            anchored(pages, 1, "盈利：95.26亿元")
            if index == 0
            else anchored(pages, 2, "（一）归属于上市公司股东的净利润：人民币54,363万元。")
            if index == 1
            else anchored(pages, 1, spec["quote"])
        )
        yoy.append(
            review_yoy(
                {
                    **spec,
                    "unit": "PERCENT",
                    "values": values,
                    "comparison": "YEAR_ON_YEAR",
                    "direction_word": direction,
                    "original_direction_word": "下降" if index < 2 else "增加",
                    "base_state": "POSITIVE",
                    "base_proof": base_proof,
                    "published_date": item["row"]["published_date"],
                },
                pages,
            )
        )
    actual_pages = sources["1214815239"][1]
    column_header = anchored(
        actual_pages,
        2,
        "项目本报告期上年同期本报告期比上年同期增减（%）年初至报告期末上年同期"
        "年初至报告期末比上年同期增减（%）调整前调整后调整后调整前调整后调整后",
    )
    restatement_proof = anchored(
        actual_pages,
        1,
        "公司是否需追溯调整或重述以前年度会计数据√是□否追溯调整或重述原因会计政策变更",
    )
    row_values = [
        "1318723734.41",
        "836162558.07",
        "836712748.16",
        "57.61",
        "5854967785.02",
        "2299254791.95",
        "2299694538.33",
        "154.60",
    ]
    restated_fields = []
    for index, offset, header in (
        (4, 4, "年初至报告期末比上年同期增减（%）"),
        (5, 0, "本报告期比上年同期增减（%）"),
    ):
        spec = specs[index]
        yoy.append(
            review_table_yoy(
                {
                    **spec,
                    "unit": "PERCENT",
                    "comparison": "YEAR_ON_YEAR",
                    "header": header,
                    "row_quote": actual_row,
                    "row_values": row_values,
                    "value_index": offset + 3,
                    "value": row_values[offset + 3],
                    "prior_year_basis": "AS_RESTATED_IN_CURRENT_DISCLOSURE",
                    "published_date": sources["1214815239"][0]["row"]["published_date"],
                },
                actual_pages,
            )
        )
        restated_fields.append(
            {
                "document_id": "1214815239",
                "issuer": "000933",
                "period_start": spec["period_start"],
                "period_end": spec["period_end"],
                "metric": spec["metric"],
                "basis": spec["basis"],
                "unit": "元",
                "currency": "CNY",
                "current": row_values[offset],
                "prior_year_unadjusted": row_values[offset + 1],
                "prior_year_adjusted": row_values[offset + 2],
                "reported_yoy_percent": row_values[offset + 3],
                "restatement_proof": restatement_proof,
                "column_header": column_header,
                "row_anchor": anchored(actual_pages, 2, actual_row),
                "available_at": claims[index]["available_at"],
                "backfill_prior_disclosures": False,
            }
        )
    pairs = [compare_claims(claims[a], claims[b], claims[b]["available_at"]) for a, b in ((2, 4), (3, 5))]
    try:
        compare_claims(claims[2], claims[5], claims[5]["available_at"])
    except ValueError as exc:
        if str(exc) != "INCOMPARABLE_EARNINGS_CLAIMS":
            raise
        period_check = {"rejected": True, "reason": str(exc)}
    else:
        raise ValueError("CROSS_PERIOD_COMPARISON_ACCEPTED")
    # 环比保留为另外的原文字段，不送入同比核验器，也不与半年度同比合并。
    qoq = {
        "document_id": "1213983584",
        "comparison": "QUARTER_ON_QUARTER",
        "kind": "FORECAST",
        "period_start": "2022-04-01",
        "period_end": "2022-06-30",
        "prior_period_start": "2022-01-01",
        "prior_period_end": "2022-03-31",
        "reported_percent_range": ["239.5", "519.1"],
        "anchor": anchored(
            sources["1213983584"][1],
            2,
            "其中2022年第二季度实现归属于上市公司股东的净利润预计比第一季度的人民币2,503万元"
            "增加人民币5,994万元~12,994万元，环比增幅239.5%~519.1%。",
        ),
        "training_ready": False,
    }
    for key in sorted(sources):
        for number in [1, 2]:
            renders.append(render(key, number, sources[key][0]["receipt"]["path"]))
    # 身份补证仍使用原件；英文、封面缺代码及摘要括号不改变报告数值语义。
    identity_specs = [
        ("1212755838", 1, ["StockCode:600519", "KWEICHOWMOUTAICO.,LTD.ANNUALREPORT2021"]),
        ("1213004638", 9, ["股票代码002299", "公司的中文名称福建圣农发展股份有限公司"]),
        ("1213171817", 7, ["股票代码002475", "公司的中文名称立讯精密工业股份有限公司"]),
        ("1213225065", 1, ["公司代码：601021", "春秋航空股份有限公司2021年年度报告摘要"]),
        ("1213254701", 1, ["StockCode:002714", "Summaryof2021AnnualReportofMuyuanFoodsCo.,Ltd.I．ImportantNotes"]),
        ("1213254727", 10, ["股票代码002714", "公司的中文名称牧原食品股份有限公司"]),
        ("1213308107", 7, ["Stockcode002475", "Chinesename立讯精密工业股份有限公司"]),
    ]
    identities = []
    for key, number, texts in identity_specs:
        item, pages = source(key)
        proofs = [anchored(pages, number, text) for text in texts]
        identities.append(
            {
                "document_id": key,
                "stock": item["row"]["secCode"],
                "passed": True,
                "source_sha256": item["receipt"]["sha256"],
                "anchors": proofs,
                "catalog_title": item["row"]["title_plain"],
                "same_values_as_chinese_not_assumed": True,
            }
        )
        renders.append(render(key, number, item["receipt"]["path"]))
    result = {
        "claims": claims,
        "yoy_fields": yoy,
        "restatement_fields": restated_fields,
        "qoq_separate_field": qoq,
        "comparisons": pairs,
        "cross_period_check": period_check,
        "identity_supplement": identities,
        "identity_pending": [{"document_id": "1214960791", "reason": "BODY_LEGAL_NAME_PRESENT_CODE_LINK_PENDING"}],
        "renders": renders,
        "new_fits": 0,
        "training_ready": False,
        "visual_status": "PENDING",
        "limitation": "Selected fields only; no population semantic readiness or prediction improvement claim",
    }
    save(OUT / "review-money-specs.json", specs)
    save(OUT / "semantic-review-candidate.json", result)
    print(
        json.dumps({"money_fields": len(claims), "yoy_fields": len(yoy), "renders": len(renders)}, ensure_ascii=False)
    )


if __name__ == "__main__":
    run()
