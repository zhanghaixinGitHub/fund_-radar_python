"""核验第六批报告内嵌预告：区分未来期间、未给预告及同日重复正文。"""

import json
import shutil
import subprocess
from collections import Counter

from app.services.fund_earnings_asof_v1 import asof_changes
from app.services.fund_earnings_batch_v4 import EarningsBatch
from app.services.fund_earnings_embedded_v1 import embedded_forecasts
from app.services.fund_earnings_evidence_v1 import compare_claims, review_claim
from app.services.fund_earnings_yoy_v3 import review_yoy
from app.services.fund_information_history_v1 import normalize, pdf_revision, read, save, sha

from scripts.fund_002112_earnings_review_v1 import extract

BATCH = EarningsBatch("20260929-earnings-v6")
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
    """只读取已保存原件；遇到采集上限不调用采集入口，金额核验不增加请求。"""
    BATCH.check_plan()
    plan = read(OUT / "semantic-review-plan-r2.json")
    for key in ("code_hashes", "document_hashes"):
        if any(sha(path) != expected for path, expected in plan[key].items()):
            raise ValueError("SEMANTIC_FROZEN_INPUT_CHANGED")
    sources = {key: source(key) for key in plan["selected_document_ids"]}
    inventory = []
    for path in sorted(plan["document_hashes"]):
        document = read(path)
        if document["body_saved"]:
            inventory.extend(embedded_forecasts(document["row"], document["pages"]))
    common = {
        "period_start": "2021-01-01",
        "period_end": "2021-03-31",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_ATTRIBUTABLE_TO_PARENT",
        "currency": "CNY",
        "audit_status": "UNAUDITED",
    }
    specs = [
        {
            **common,
            "issuer": "002709",
            "document_id": "1209444876",
            "kind": "FORECAST",
            "page": 1,
            "unit": "万元",
            "values": ["25,000", "30,000"],
            "quote": "盈利：25,000万元–30,000万元",
        },
        {
            **common,
            "issuer": "002709",
            "document_id": "1209722788",
            "kind": "REPORTED_RESULT",
            "page": 3,
            "unit": "元",
            "values": ["286,855,117.10"],
            "quote": "归属于上市公司股东的净利润（元）286,855,117.10 41,504,117.44 591.15%",
        },
        {
            **common,
            "issuer": "002709",
            "period_end": "2021-06-30",
            "document_id": "1209722788",
            "kind": "FORECAST",
            "page": 8,
            "unit": "万元",
            "values": ["65,000", "75,000"],
            "quote": "归属于上市公司股东的净利润（万元）65,000 -- 75,000 31,168 增长108.55% -- 140.63%",
        },
        {
            **common,
            "issuer": "002600",
            "document_id": "1209491331",
            "kind": "FORECAST",
            "page": 1,
            "unit": "万元",
            "values": ["44,623.87", "46,937.70"],
            "quote": "盈利：44,623.87万元–46,937.70万元",
        },
        {
            **common,
            "issuer": "002600",
            "document_id": "1209829597",
            "kind": "REPORTED_RESULT",
            "page": 3,
            "unit": "元",
            "values": ["463,076,073.72"],
            "quote": "归属于上市公司股东的净利润（元）463,076,073.72 64,876,078.75 613.79%",
        },
    ]
    claims, yoy, render_keys = [], [], set()
    for spec in specs:
        item, pages = sources[spec["document_id"]]
        if not item["identity"]["passed"]:
            raise ValueError("SELECTED_MONEY_SOURCE_IDENTITY_NOT_PASSED")
        spec = {**spec, "published_date": item["row"]["published_date"]}
        value = review_claim(spec, pages)
        if spec["period_end"] == "2021-06-30":
            candidates = [
                r
                for r in inventory
                if r["document_id"] == spec["document_id"] and r["period_end"] == spec["period_end"]
            ]
            if len(candidates) != 1 or candidates[0]["status"] != "APPLICABLE_REQUIRES_FACT_REVIEW":
                raise ValueError("EMBEDDED_FORECAST_SECTION_NOT_ADMITTED")
            value["period_anchor"] = anchor(pages, 8, "六、对2021年1-6月经营业绩的预计")
        claims.append({**value, "source": item["receipt"], "source_identity_verified": True, "revision_issues": []})
        if spec["kind"] == "REPORTED_RESULT":
            percent = "591.15" if spec["issuer"] == "002709" else "613.79"
            yoy.append(
                {
                    **review_yoy(
                        {
                            **spec,
                            "unit": "PERCENT",
                            "comparison": "YEAR_ON_YEAR",
                            "values": [percent],
                            "direction_word": "原文有符号数值",
                            "base_state": "POSITIVE",
                        },
                        pages,
                    ),
                    "source": item["receipt"],
                }
            )
        render_keys.update({(spec["document_id"], 1), (spec["document_id"], spec["page"])})
    # 同一公告不同期间不能直接比较；半年预告不会冒充第一季度实际值。
    try:
        compare_claims(claims[1], claims[2], claims[2]["available_at"])
    except ValueError as exc:
        if str(exc) != "INCOMPARABLE_EARNINGS_CLAIMS":
            raise
    else:
        raise ValueError("CROSS_PERIOD_COMPARISON_NOT_REJECTED")
    duplicate = [r for r in inventory if r["document_id"] in {"1209722788", "1209722789"}]
    if len(duplicate) != 2 or len({r["content_group_key"] for r in duplicate}) != 1:
        raise ValueError("DUPLICATE_EMBEDDED_CONTENT_MISMATCH")
    negative = [r for r in inventory if r["document_id"] == "1209829597"]
    if len(negative) != 1 or negative[0]["status"] != "EXPLICIT_NOT_APPLICABLE_IN_THIS_SECTION":
        raise ValueError("NOT_APPLICABLE_SECTION_MISMATCH")
    snapshots = []
    for issuer, cutoffs in (
        ("002709", ("2021-04-21T07:59:59+08:00", "2021-04-21T08:00:00+08:00")),
        ("002600", ("2021-04-29T07:59:59+08:00", "2021-04-29T08:00:00+08:00")),
    ):
        selected = [c for c in claims if c["issuer"] == issuer]
        early = [c for c in selected if c["document_id"] in {"1209444876", "1209491331"}]
        if asof_changes(selected, cutoffs[0]) != asof_changes(early, cutoffs[0]):
            raise ValueError("FUTURE_DISCLOSURE_CHANGED_PAST_STATE")
        snapshots.extend(
            {"issuer": issuer, "as_of": cutoff, "facts": asof_changes(selected, cutoff)} for cutoff in cutoffs
        )
    render_keys.update({("1209722789", 8), ("1209829597", 10), ("1209852169", 10), ("1209852169", 11)})
    result = {
        "claims": claims,
        "reported_yoy": yoy,
        "asof_snapshots": snapshots,
        "embedded_inventory": inventory,
        "embedded_status_counts": dict(Counter(r["status"] for r in inventory)),
        "same_day_duplicate_sources_not_compared": [r["document_id"] for r in duplicate],
        "not_applicable_does_not_mean_zero": True,
        "cross_period_comparison_rejected": True,
        "future_invariance_passed": True,
        "new_fits": 0,
        "training_ready": False,
        "renders": [render(key, page, sources[key][0]["receipt"]["path"]) for key, page in sorted(render_keys)],
        "visual_status": "PENDING",
        "limitation": (
            "Hints from saved bodies only. Uncollected bodies, source identities, "
            "all facts and historical continuity remain incomplete"
        ),
    }
    save(OUT / "review-money-specs.json", specs)
    save(OUT / "semantic-review-candidate.json", result)
    print(
        json.dumps(
            {
                "claims": len(claims),
                "yoy": len(yoy),
                "hints": len(inventory),
                "status_counts": result["embedded_status_counts"],
                "renders": len(result["renders"]),
            }
        )
    )


if __name__ == "__main__":
    run()
