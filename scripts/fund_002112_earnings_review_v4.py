"""重放第四批亏损兑现与真实更正；资料不足时保持历史缺口，不制造旧报告。"""

import json
import shutil
import subprocess

from app.services.fund_earnings_asof_v1 import asof_changes, review_correction
from app.services.fund_earnings_batch_v3 import EarningsBatch
from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_signed_facts_v1 import review_loss_claim
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

from scripts.fund_002112_earnings_review_v1 import extract

BATCH = EarningsBatch("20260928-earnings-v4")
OUT = BATCH.out


def source(key):
    item = read(OUT / "documents" / (key + ".json"))
    if not item["body_saved"] or sha(item["receipt"]["path"]) != item["receipt"]["sha256"]:
        raise ValueError("REVIEW_SOURCE_UNAVAILABLE")
    pages, metadata = extract(item["receipt"]["path"])
    if [normalize(p) for p in pages] != [normalize(p) for p in item["pages"]]:
        raise ValueError("REVIEW_REPLAY_CHANGED")
    if pdf_revision(metadata, item["row"]["published_date"]):
        raise ValueError("SOURCE_REVISION_CONFLICT")
    return item, pages


def anchor(pages, number, quote):
    text, wanted = normalize(pages[number - 1]), normalize(quote)
    if not wanted or text.count(wanted) != 1:
        raise ValueError("UNIQUE_SOURCE_ANCHOR_REQUIRED:" + wanted)
    return {"page": number, "offset": text.index(wanted), "text": wanted}


def render(key, number, path):
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
                path,
                str(target.with_suffix("")),
            ],
            capture_output=True,
            check=True,
            timeout=45,
        )
    return {"document_id": key, "page": number, "path": str(target), "sha256": sha(target)}


def run():
    BATCH.check_plan()
    common = {
        "period_end": "2021-09-30",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_ATTRIBUTABLE_TO_PARENT",
        "currency": "CNY",
        "audit_status": "UNAUDITED",
    }
    specs = []
    enjie_row = "归属于上市公司股东的净利润（元）705,230,709.23 118.94% 1,755,423,408.66 172.79%"
    muyuan_row = "归属于上市公司股东的净利润（元）-821,714,334.22 -108.05% 8,704,266,223.85 -58.53%"
    for start, values, quote in (
        ("2021-07-01", ["65,000", "74,000"], "盈利：65,000万元—74,000万元"),
        ("2021-01-01", ["170,000", "179,000"], "盈利：170,000万元—179,000万元"),
    ):
        specs.append(
            {
                **common,
                "issuer": "002812",
                "document_id": "1211209632",
                "period_start": start,
                "page": 1,
                "kind": "FORECAST",
                "unit": "万元",
                "values": values,
                "quote": quote,
            }
        )
    for start, value in (("2021-07-01", "705,230,709.23"), ("2021-01-01", "1,755,423,408.66")):
        specs.append(
            {
                **common,
                "issuer": "002812",
                "document_id": "1212063755",
                "period_start": start,
                "page": 1,
                "kind": "REPORTED_RESULT",
                "unit": "元",
                "values": [value],
                "quote": enjie_row,
                "version": "CORRECTED_REPORT_PUBLISHED_2021_12_31",
            }
        )
    specs.extend(
        [
            {
                **common,
                "issuer": "002714",
                "document_id": "1211269603",
                "period_start": "2021-07-01",
                "page": 1,
                "kind": "FORECAST",
                "unit": "万元",
                "values": ["50,000.00", "100,000.00"],
                "quote": "亏损：50,000.00万元–100,000.00万元",
                "loss_anchor": "亏损：50,000.00万元–100,000.00万元",
            },
            {
                **common,
                "issuer": "002714",
                "document_id": "1211269603",
                "period_start": "2021-01-01",
                "page": 1,
                "kind": "FORECAST",
                "unit": "万元",
                "values": ["850,000.00", "900,000.00"],
                "quote": "盈利：850,000.00万元–900,000.00万元",
            },
        ]
    )
    for start, value in (("2021-07-01", "-821,714,334.22"), ("2021-01-01", "8,704,266,223.85")):
        specs.append(
            {
                **common,
                "issuer": "002714",
                "document_id": "1211316681",
                "period_start": start,
                "page": 2,
                "kind": "REPORTED_RESULT",
                "unit": "元",
                "values": [value],
                "quote": muyuan_row,
            }
        )
    sources = {key: source(key) for key in sorted({s["document_id"] for s in specs} | {"1212063759"})}
    identity_specs = [
        ("1211235338", 1, ["证券代码：603501", "上海韦尔半导体股份有限公司2021年前三季度业绩预增的公告"]),
        ("1211588655", 9, ["股票代码002460（A股）；01772（H股）", "公司的中文名称江西赣锋锂业股份有限公司"]),
        ("1212063755", 1, ["证券代码：002812证券简称：恩捷股份公告编号：2021-214"]),
        ("1212730520", 6, ["股票代码002594、01211", "公司的中文名称比亚迪股份有限公司"]),
        ("1212840230", 1, ["公司代码：601636", "株洲旗滨集团股份有限公司2021年年度报告"]),
        ("1213172272", 1, ["证券代码：603599证券简称：广信股份", "安徽广信农化股份有限公司2022年第一季度报告"]),
    ]
    identities, render_keys = [], set()
    for key, number, texts in identity_specs:
        item, pages = sources.get(key) or source(key)
        identities.append(
            {
                "document_id": key,
                "stock": item["row"]["secCode"],
                "passed": True,
                "source_sha256": item["receipt"]["sha256"],
                "anchors": [anchor(pages, number, text) for text in texts],
                "rule": "EXPLICIT_BODY_CODE_AND_LEGAL_NAME_OR_UNIQUE_ISSUER_HEADER",
            }
        )
        render_keys.add((key, number))
    supplemental = {i["document_id"] for i in identities}
    claims = []
    for spec in specs:
        item, pages = sources[spec["document_id"]]
        if not item["identity"]["passed"] and spec["document_id"] not in supplemental:
            raise ValueError("SOURCE_IDENTITY_NOT_VERIFIED")
        spec = {**spec, "published_date": item["row"]["published_date"]}
        value = (review_loss_claim if "loss_anchor" in spec else review_claim)(spec, pages)
        claims.append({**value, "source": item["receipt"], "source_identity_verified": True, "revision_issues": []})
        render_keys.add((spec["document_id"], spec["page"]))
    correction_item, correction_pages = sources["1212063759"]
    old_row = "归属于上市公司股东的净利润（元）700,522,734.07 117.48% 1,750,715,433.50 172.06%"
    corrections = []
    for start, old, new in (
        ("2021-07-01", "700,522,734.07", "705,230,709.23"),
        ("2021-01-01", "1,750,715,433.50", "1,755,423,408.66"),
    ):
        base = {
            **common,
            "issuer": "002812",
            "document_id": "1212063759",
            "period_start": start,
            "page": 2,
            "kind": "REPORTED_RESULT",
            "unit": "元",
            "published_date": correction_item["row"]["published_date"],
        }
        result = review_correction(
            {**base, "values": [old], "quote": old_row}, {**base, "values": [new], "quote": enjie_row}, correction_pages
        )
        result["source"] = correction_item["receipt"]
        result["original_reference"] = anchor(correction_pages, 1, "于2021年10月26日披露了《2021年第三季度报告》。")
        result["current_corrected_report"] = "1212063755"
        corrected = next(c for c in claims if c["document_id"] == "1212063755" and c["period_start"] == start)
        if corrected["money"] != result["after"]["money"]:
            raise ValueError("CORRECTION_AND_UPDATED_REPORT_MISMATCH")
        corrections.append(result)
    render_keys.update({("1212063759", 1), ("1212063759", 2), ("1211316681", 1)})
    enjie = [c for c in claims if c["issuer"] == "002812"]
    muyuan = [c for c in claims if c["issuer"] == "002714"]
    cutoff_before = "2021-11-01T08:00:00+08:00"
    early = [c for c in enjie if c["kind"] == "FORECAST"]
    if asof_changes(early, cutoff_before) != asof_changes(enjie, cutoff_before):
        raise ValueError("FUTURE_CORRECTION_CHANGED_PAST_STATE")
    snapshots = [
        {"as_of": cutoff, "issuer": "002812", "facts": asof_changes(enjie, cutoff)}
        for cutoff in (cutoff_before, "2022-01-01T07:59:59+08:00", "2022-01-01T08:00:00+08:00")
    ]
    snapshots.extend(
        {"as_of": cutoff, "issuer": "002714", "facts": asof_changes(muyuan, cutoff)}
        for cutoff in ("2021-10-20T07:59:59+08:00", "2021-10-20T08:00:00+08:00")
    )
    excluded = []
    for key, reason, quote in (
        ("1211338407", "PARENT_WRAPPER_FOR_SUBSIDIARY_REPORT", "本公司控股子公司平安银行股份有限公司"),
        ("1212174062", "PARENT_WRAPPER_FOR_SUBSIDIARY_REPORT", "本公司控股子公司平安银行股份有限公司"),
        ("1211338408", "SUBSIDIARY_REPORT_NOT_CATALOG_PARENT", "平安银行股份有限公司2021年第三季度报告"),
        ("1212174063", "SUBSIDIARY_REPORT_NOT_CATALOG_PARENT", "平安银行股份有限公司2021年度业绩快报"),
        (
            "1212684377",
            "INVESTOR_MEETING_NOTICE_NOT_EARNINGS_REPORT",
            "关于召开2021年年度报告网上说明会并征集问题的公告",
        ),
        ("1213688332", "CORPORATE_ANNUAL_RETURN_NOT_EARNINGS_REPORT", "展示文件2021年度企業年度報告書"),
    ):
        item, pages = source(key)
        excluded.append(
            {
                "document_id": key,
                "catalog_stock": item["row"]["secCode"],
                "reason": reason,
                "anchor": anchor(pages, 1, quote),
                "source_sha256": item["receipt"]["sha256"],
            }
        )
        render_keys.add((key, 1))
    renders = []
    for key, number in sorted(render_keys):
        item = read(OUT / "documents" / (key + ".json"))
        renders.append(render(key, number, item["receipt"]["path"]))
    result = {
        "claims": claims,
        "correction_pairs": corrections,
        "asof_snapshots": snapshots,
        "future_invariance_passed": True,
        "identity_supplement": identities,
        "excluded_documents": excluded,
        "renders": renders,
        "training_ready": False,
        "new_fits": 0,
        "visual_status": "PENDING",
        "original_enjie_report_missing": True,
        "limitation": "Only reviewed facts queried. Missing original report means prior-disclosure history "
        "is incomplete;"
        " correction-to-forecast comparison is not a complete adjacent-disclosure history.",
    }
    save(OUT / "review-money-specs.json", specs)
    save(OUT / "semantic-review-candidate.json", result)
    print(json.dumps({"claims": len(claims), "correction_pairs": len(corrections), "renders": len(renders)}))


if __name__ == "__main__":
    run()
