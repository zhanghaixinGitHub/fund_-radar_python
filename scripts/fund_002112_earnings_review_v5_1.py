"""核验第五批真实版本链：修订不自动等于利润变化，预告与实际分开记录。"""

import json
import shutil
import subprocess

from app.services.fund_earnings_asof_v1 import asof_changes
from app.services.fund_earnings_batch_v4 import EarningsBatch
from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_semantics_v2 import review_table_yoy
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

from scripts.fund_002112_earnings_review_v1 import extract

BATCH = EarningsBatch("20260929-earnings-v5")
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
    frozen = read(OUT / "semantic-review-plan-r2.json")
    for path, expected in frozen["code_hashes"].items():
        if sha(path) != expected:
            raise ValueError("SEMANTIC_CODE_CHANGED")
    sources = {key: source(key) for key in frozen["document_ids"]}
    identity_specs = [
        ("1211418794", 1, ["证券代码：603599", "安徽广信农化股份有限公司2021年三季度报告更正公告"]),
        ("1212240427", 1, ["证券代码：000733", "中国振华（集团）科技股份有限公司2021年度业绩预告"]),
        ("1212730375", 7, ["股票代码002241", "公司的中文名称歌尔股份有限公司"]),
        ("1212751615", 9, ["股票代码002460（A股）；01772（H股）", "公司的中文名称江西赣锋锂业股份有限公司"]),
        ("1212906454", 9, ["StockCode002812", "NameoftheCompanyinChinese云南恩捷新材料股份有限公司"]),
        ("1212921809", 11, ["股票代码002920", "公司的中文名称惠州市德赛西威汽车电子股份有限公司"]),
        ("1212960545", 2, ["公司代码：603501", "上海韦尔半导体股份有限公司2021年年度报告重要提示"]),
        ("1213027750", 7, ["股票代码300750", "公司的中文名称宁德时代新能源科技股份有限公司"]),
        ("1213200180", 8, ["Stockcode002241", "NameoftheCompanyinChinese歌尔股份有限公司"]),
    ]
    identities, render_keys = [], set()
    for key, number, texts in identity_specs:
        item, pages = sources[key]
        identities.append(
            {
                "document_id": key,
                "stock": item["row"]["secCode"],
                "passed": True,
                "anchors": [anchor(pages, number, text) for text in texts],
                "source_sha256": item["receipt"]["sha256"],
                "rule": "EXPLICIT_BODY_CODE_AND_LEGAL_NAME_OR_UNIQUE_ISSUER_HEADER",
                "semantic_verified": False,
            }
        )
        render_keys.add((key, number))
    common = {
        "period_end": "2021-09-30",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_ATTRIBUTABLE_TO_PARENT",
        "currency": "CNY",
        "audit_status": "UNAUDITED",
    }
    specs, yoy = [], []
    # 金额和列序由人工读原件指定。只从固定表头至固定行截取引用，不自动猜测字段。
    for key in ("1211400642", "1211418975"):
        item, pages = sources[key]
        page = pages[0]
        begin = page.index("单位：元")
        end = page.index("137.20", begin) + len("137.20")
        quote = page[begin:end]
        header = quote[: quote.index("营业收入")]
        row = "归属于上市公司股东的净利润 402,519,609.64 165.72 1,032,880,538.34 137.20"
        for start, value, percent, index in (
            ("2021-07-01", "402,519,609.64", "165.72", 1),
            ("2021-01-01", "1,032,880,538.34", "137.20", 3),
        ):
            base = {
                **common,
                "issuer": "603599",
                "document_id": key,
                "period_start": start,
                "kind": "REPORTED_RESULT",
                "page": 1,
                "published_date": item["row"]["published_date"],
            }
            specs.append({**base, "unit": "元", "values": [value], "quote": quote})
            yoy.append(
                {
                    **review_table_yoy(
                        {
                            **base,
                            "unit": "PERCENT",
                            "comparison": "YEAR_ON_YEAR",
                            "header": header,
                            "row_quote": row,
                            "row_values": ["402519609.64", "165.72", "1032880538.34", "137.20"],
                            "value_index": index,
                            "value": percent,
                        },
                        pages,
                    ),
                    "source": item["receipt"],
                }
            )
        render_keys.add((key, 1))
        render_keys.add((key, 2))
    for start, values, quote in (
        ("2021-01-01", ["133,489.45", "154,026.29"], "盈利：133,489.45万元–154,026.29万元"),
        ("2021-07-01", ["55,113.71", "63,592.74"], "盈利：55,113.71万元–63,592.74万元"),
    ):
        specs.append(
            {
                **common,
                "issuer": "002049",
                "document_id": "1211244116",
                "period_start": start,
                "kind": "FORECAST",
                "page": 1,
                "unit": "万元",
                "values": values,
                "quote": quote,
            }
        )
    for start, value in (("2021-01-01", "1,457,401,712.84"), ("2021-07-01", "581,848,759.92")):
        specs.append(
            {
                **common,
                "issuer": "002049",
                "document_id": "1211373372",
                "period_start": start,
                "kind": "REPORTED_RESULT",
                "page": 1,
                "unit": "元",
                "values": [value],
                "quote": "归属于上市公司股东的净利润（元）581,848,759.92 105.87% 1,457,401,712.84 112.90%",
            }
        )
    claims = []
    for spec in specs:
        item, pages = sources[spec["document_id"]]
        if not item["identity"]["passed"]:
            raise ValueError("MONEY_SOURCE_IDENTITY_NOT_PASSED")
        spec = {**spec, "published_date": item["row"]["published_date"]}
        value = review_claim(spec, pages)
        claims.append({**value, "source": item["receipt"], "source_identity_verified": True, "revision_issues": []})
        render_keys.add((spec["document_id"], spec["page"]))
    correction_item, correction_pages = sources["1211418794"]
    correction_scope = {
        "document_id": "1211418794",
        "source": correction_item["receipt"],
        "published_date": correction_item["row"]["published_date"],
        "original_document": "1211400642",
        "revised_document": "1211418975",
        "affected_sections": ["FINANCIAL_CHANGE_EXPLANATION", "SHAREHOLDER_INFORMATION"],
        "anchors": [
            anchor(correction_pages, 1, text)
            for text in (
                "于2021年10月28日在",
                "主要会计数据、财务指标发生变动的情况、原因填写错误",
                "前十名股东持股情况表填写错误",
            )
        ],
        "profit_field_comparison": "Q3_AND_YTD_PARENT_PROFIT_AND_REPORTED_YOY_UNCHANGED",
        "all_financial_fields_checked": False,
        "training_ready": False,
    }
    render_keys.update({("1211418794", 2), ("1211418794", 3)})
    snapshots = []
    for issuer, cutoffs in (
        ("603599", ("2021-10-29T08:00:00+08:00", "2021-10-30T07:59:59+08:00", "2021-10-30T08:00:00+08:00")),
        ("002049", ("2021-10-27T07:59:59+08:00", "2021-10-27T08:00:00+08:00")),
    ):
        selected = [c for c in claims if c["issuer"] == issuer]
        early = [c for c in selected if c["document_id"] in {"1211400642", "1211244116"}]
        if asof_changes(selected, cutoffs[0]) != asof_changes(early, cutoffs[0]):
            raise ValueError("FUTURE_DISCLOSURE_CHANGED_PAST_FACTS")
        for cutoff in cutoffs:
            snapshots.append({"issuer": issuer, "as_of": cutoff, "facts": asof_changes(selected, cutoff)})
    for snapshot in snapshots:
        if snapshot["issuer"] == "603599" and snapshot["as_of"] == "2021-10-30T08:00:00+08:00":
            if any(f["change"]["lower_change_cny"] != "0.00" for f in snapshot["facts"]):
                raise ValueError("REVISED_PROFIT_NOT_IDENTICAL")
    renders = [render(key, number, sources[key][0]["receipt"]["path"]) for key, number in sorted(render_keys)]
    result = {
        "claims": claims,
        "reported_yoy": yoy,
        "correction_scope": correction_scope,
        "identity_supplement": identities,
        "asof_snapshots": snapshots,
        "renders": renders,
        "future_invariance_passed": True,
        "visual_status": "PENDING",
        "training_ready": False,
        "new_fits": 0,
        "limitation": "Reviewed inputs only; full event history and global semantic admission remain incomplete",
    }
    save(OUT / "review-money-specs.json", specs)
    save(OUT / "semantic-review-candidate.json", result)
    print(json.dumps({"claims": len(claims), "yoy": len(yoy), "identities": len(identities), "renders": len(renders)}))


if __name__ == "__main__":
    run()
