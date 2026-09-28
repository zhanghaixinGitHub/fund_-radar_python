"""已有 12 张人工核查新闻卡的格式整理及本地校验，不抓取、不生成方向、不训练。

本校验器验证这批已审样例的契约和反例，不声称自动理解任意公告，也不建立
消息到收益的打分公式。原文事实与原有机制分析分栏，未知值保持 None。
"""

import re
from decimal import Decimal

from app.services import fund_002112_round3_data as data

SOURCE = data.ROOT / "news-chain-checks/prior-day-20260925-v1"
# 数量来自既有分析卡，单位与经济含义显式保存；同一事项多份公告不相加。
AMOUNTS = {
    "N01": [("issued_shares", "355706", "SHARE", "HK_ORDINARY", "ACTUAL")],
    "N02": [
        ("buyback_plan_min", "150000000", "CNY", None, "PLAN"),
        ("buyback_plan_max", "300000000", "CNY", None, "PLAN"),
        ("paid", "14121888.36", "CNY", None, "ACTUAL_INCLUDING_FEES"),
    ],
    "N03": [
        ("buyback_plan_min", "250000000", "CNY", None, "PLAN"),
        ("buyback_plan_max", "500000000", "CNY", None, "PLAN"),
        ("paid", "4102048", "CNY", None, "ACTUAL_EXCLUDING_FEES"),
        ("repurchased_shares", "70100", "SHARE", "A_SHARE", "ACTUAL"),
    ],
    "N04": [
        ("option_issue_one", "433732", "SHARE", "HK_ORDINARY", "ACTUAL"),
        ("option_issue_two", "7800", "SHARE", "HK_ORDINARY", "ACTUAL"),
    ],
    "N05": [],
    "N06": [],
    "N07": [
        ("guarantee_ceiling", "606000000", "CNY", None, "PLANNED_LIMIT"),
        ("high_debt_ceiling", "409000000", "CNY", None, "SUBSET_OF_LIMIT"),
        ("guarantee_balance", "257000000", "CNY", None, "CONTINGENT_BALANCE_NOT_LOSS"),
    ],
    "N08": [],
    "N09": [
        ("contract_price", "575000000", "CNY", None, "CONTRACT_TOTAL_INCLUDING_TAX"),
        ("deposit", "170000000", "CNY", None, "INCLUDED_IN_CONTRACT_PRICE"),
    ],
    "N10": [("external_subscription", "28249996", "USD", None, "PI_HEALTH_FINANCING_NOT_ISSUER_REVENUE")],
    "N11": [],
    "N12": [
        ("monthly_net_shares", "953108", "SHARE", "ORDINARY_MONTHLY_SCOPE", "OVERLAPS_DAILY_NOT_ADDITIVE"),
        ("option_proceeds", "2238792.31", "USD", None, "CAPITAL_PROCEEDS_NOT_REVENUE"),
    ],
}
TOPICS = {
    "N01": ["BEIGENE_FEB_CAPITAL"],
    "N02": ["KANGYUAN_BUYBACK"],
    "N03": ["ANTU_BUYBACK"],
    "N04": ["BEIGENE_FEB_CAPITAL"],
    "N05": ["ZHIFEI_BUYBACK_PROCEDURE"],
    "N06": ["CR999_KUNYAO_GUARANTEE", "CR999_TECH_CENTER"],
    "N07": ["CR999_KUNYAO_GUARANTEE"],
    "N08": ["CR999_KUNYAO_GUARANTEE"],
    "N09": ["BEIGENE_SHANGHAI_PROPERTY"],
    "N10": ["BEIGENE_PI_HEALTH"],
    "N11": ["BEIGENE_PI_HEALTH"],
    "N12": ["BEIGENE_FEB_CAPITAL"],
}


def amounts(sample_id):
    return [
        dict(zip(("meaning", "value", "unit", "share_class", "status"), row, strict=True)) for row in AMOUNTS[sample_id]
    ]


def normalize(analysis, selection):
    selected = {r["sample_id"]: r for r in selection["selected"]}
    cards = []
    for r in analysis["records"]:
        s = selected[r["id"]]
        cards.append(
            {
                "id": r["id"],
                "announcement_id": s["announcement_id"],
                "evidence": {
                    "pdf_file": r["pdf_file"],
                    "pdf_sha256": r["pdf_sha256"],
                    "pages": r["source_pages"],
                    "anchors": r["anchors"],
                    "url": r["url"],
                },
                "published_date": r["published_date"],
                "target": r["target"],
                "company": {"stock_code": r["stock_code"], "name": r["stock_name"], "objects": r["objects"]},
                "facts": {"text": r["facts"], "amounts": amounts(r["id"])},
                "events": {
                    "ids": r["event_topics"],
                    "stage": r["event_stage"],
                    "old_and_new": r["novelty"],
                    "aggregation": "EVIDENCE_ONLY_NO_ADDITION",
                },
                "analysis": {
                    "positive_mechanism": r["positive_path"],
                    "negative_mechanism": r["negative_path"],
                    "horizon": r["horizon"],
                    "pitfall": r["pitfall"],
                    "kind": "EXISTING_HUMAN_ANALYSIS",
                },
                "holding": {
                    "report_end": r["report_end"],
                    "public_date": r["report_published_date"],
                    "nav_weight_pct": r["reported_weight_pct"],
                    "report_sha256": s["report_sha256"],
                    "meaning": "DISCLOSED_NOT_CURRENT_ACTUAL_HOLDING",
                },
                "unknowns": r["unknowns"],
                "uncovered": "其他持仓、当日实际仓位、市场预期和完整历史消息覆盖未知",
                "next_day_direction": None,
                "impact_magnitude": None,
                "prediction_eligible": False,
            }
        )
    return {
        "schema": "ROUND3_EXISTING_NEWS_CARDS_V1",
        "cards": cards,
        "daily_context": analysis["daily_context"],
        "gaps": analysis["limitations"],
        "fit_count": 0,
        "network_requests": 0,
        "external_llm_calls": 0,
    }


def validate(bundle):
    """固定样例的强约束：支持一文多事、一事多文和单位/阶段反例。"""
    cards = bundle["cards"]
    if [r["id"] for r in cards] != list(TOPICS) or len({r["announcement_id"] for r in cards}) != 12:
        raise ValueError("NEWS_FIXED_SAMPLE_IDENTITY_CHANGED")
    anchors = 0
    for r in cards:
        if not r["published_date"] < r["target"] or not r["holding"]["public_date"] < r["target"]:
            raise ValueError("NEWS_OR_HOLDING_NOT_PRIOR_PUBLIC")
        if r["events"]["ids"] != TOPICS[r["id"]] or r["events"]["aggregation"] != "EVIDENCE_ONLY_NO_ADDITION":
            raise ValueError("NEWS_DUPLICATE_OR_DISTINCT_EVENT_MERGED")
        if r["facts"]["amounts"] != amounts(r["id"]):
            raise ValueError("NEWS_AMOUNT_UNIT_OR_ECONOMIC_STAGE_CHANGED")
        if r["next_day_direction"] is not None or r["impact_magnitude"] is not None or r["prediction_eligible"]:
            raise ValueError("NEWS_UNKNOWN_MUST_NOT_BECOME_ZERO_OR_FLAT")
        if not r["unknowns"] or not r["analysis"]["horizon"] or not r["uncovered"]:
            raise ValueError("NEWS_UNCERTAINTY_MISSING")
        if Decimal(r["holding"]["nav_weight_pct"]) <= 0:
            raise ValueError("NEWS_HOLDING_WEIGHT_INVALID")
        anchors += len(r["evidence"]["anchors"])
    return {
        "cards": len(cards),
        "anchors": anchors,
        "companies": len({r["company"]["stock_code"] for r in cards}),
        "target_dates": len({r["target"] for r in cards}),
        "passed": True,
        "semantic_accuracy_measured": False,
        "prediction_accuracy_measured": False,
    }


def build(output):
    """核对原有 12 份文件与已抽取证据，不重新选择、增样或调用大模型。"""
    inputs = {
        n: data.read(SOURCE / (n + ".json"))
        for n in ("plan", "selection", "analysis", "verification", "extracted-pages")
    }
    if inputs["analysis"]["source_manifest"]["selection_sha256"] != data.file_hash(SOURCE / "selection.json"):
        raise ValueError("NEWS_SELECTION_SOURCE_CHANGED")
    bundle = normalize(inputs["analysis"], inputs["selection"])
    result = validate(bundle)
    texts = {r["sample_id"]: r for r in inputs["extracted-pages"]}
    selected = {r["sample_id"]: r for r in inputs["selection"]["selected"]}
    manifest = {str(SOURCE / (n + ".json")): data.file_hash(SOURCE / (n + ".json")) for n in inputs}
    for card in bundle["cards"]:
        evidence, text = card["evidence"], texts[card["id"]]
        path = data.ROOT / evidence["pdf_file"]
        if data.file_hash(path) != evidence["pdf_sha256"] or text["pdf_sha256"] != evidence["pdf_sha256"]:
            raise ValueError("NEWS_RAW_CHANGED")
        manifest[str(path)] = evidence["pdf_sha256"]
        for anchor in evidence["anchors"]:
            if re.sub(r"\s+", "", anchor["text"]) not in re.sub(r"\s+", "", text["pages"][anchor["page"] - 1]):
                raise ValueError("NEWS_SOURCE_ANCHOR_NOT_FOUND")
        s = selected[card["id"]]
        report = data.read(data.ROOT / s["report_file"])
        holding = next(h for h in report["holdings"] if h["stock_code"] == card["company"]["stock_code"])
        if (
            holding["nav_weight_pct"] != card["holding"]["nav_weight_pct"]
            or report["raw"]["sha256"] != s["report_sha256"]
        ):
            raise ValueError("NEWS_HOLDING_LINK_CHANGED")
        manifest[str(data.ROOT / s["report_file"])] = data.file_hash(data.ROOT / s["report_file"])
    data.write_once(output / "news-cards.json", bundle)
    data.write_once(
        output / "news-validation.json",
        {**result, "sources": manifest, "boundary": "LOCAL_FORMAT_AND_REVIEWED_SAMPLE_CHECKS_ONLY"},
    )
    return result
