"""产品预测唯一政策：综合审阅；旧净值算法仅留作历史证据恢复。"""

import json
from functools import lru_cache
from pathlib import Path

from app.services.direction_1d_protocol import digest


@lru_cache(maxsize=1)
def policy() -> dict:
    """规则由随代码发布的文件固定，浏览器、模型输出和旧任务不能覆盖。"""
    path = Path(__file__).resolve().parents[1] / "data/comprehensive_prediction_policy_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def comprehensive_only() -> bool:
    """没有回退旧预测的运行开关；政策变更需修改版本并重新验证。"""
    return True


def identity() -> dict:
    value = policy()
    return {
        "generationMethod": value["generation_method"],
        "analysisPolicyVersion": value["version"],
        "analysisPolicyHash": digest(value),
    }


def validate_review(data: dict, review: dict, facts: dict) -> None:
    """校验综合审阅的最小业务边界，不将形式校验宣称为语义正确或效果验证。

    必需维度的来源、无新增公告的检索回执，以及反证检查均须进入冻结资料。
    判断至少引用一个已核实的非净值事实；无法满足时保存缺口并暂不生成。
    """
    if (
        review.get("analysis_policy_version") != policy()["version"]
        or data.get("analysis_policy_version") != policy()["version"]
    ):
        raise ValueError("COMPREHENSIVE_POLICY_REQUIRED")
    coverage = data.get("comprehensive_coverage", {})
    source_ids = {document["id"] for document in data.get("documents", [])}
    source_ids.update(fact.get("fact_id", key) for key, fact in facts.items())
    if data.get("report", {}).get("sourceUrl"):
        source_ids.add(data["report"]["sourceUrl"])
    for dimension in policy()["required_dimensions"]:
        row = coverage.get(dimension, {})
        allowed = {"AVAILABLE", "VERIFIED_EMPTY"} if dimension == "DISCLOSURES" else {"AVAILABLE"}
        if row.get("status") not in allowed or not row.get("source_refs") or not set(row["source_refs"]) <= source_ids:
            raise ValueError("COMPREHENSIVE_EVIDENCE_REQUIRED")
        if dimension == "DISCLOSURES":
            documents = {d["id"]: d for d in data.get("documents", [])}
            kinds = {"ANNOUNCEMENT"} if row["status"] == "AVAILABLE" else {"DISCLOSURE_SEARCH"}
            if not all(ref in documents and documents[ref].get("kind") in kinds for ref in row["source_refs"]):
                raise ValueError("COMPREHENSIVE_DISCLOSURE_RECEIPT_REQUIRED")
    if review.get("counter_evidence_checked") is not True:
        raise ValueError("COMPREHENSIVE_COUNTER_REVIEW_REQUIRED")
    analysis = review.get("analysis", {})
    if not analysis.get("change_conditions"):
        raise ValueError("COMPREHENSIVE_CHANGE_CONDITIONS_REQUIRED")
    expected = "支持上涨" if analysis.get("direction") == "UP" else "支持下跌"
    directional = [
        facts[ref]
        for reason in analysis.get("reasons", [])
        if reason.get("role") == expected
        for ref in reason.get("refs", [])
        if ref in facts
    ]
    allowed_categories = {
        "持仓行情",
        "行业行情",
        "市场行情",
        "公司经营",
        "公告",
        "公司公告",
        "新闻",
        "政策",
        "财务",
        "MARKET",
        "DISCLOSURES",
        "FINANCIALS",
        "NEWS",
        "POLICY",
    }
    if not any(
        fact.get("category") in allowed_categories and fact.get("direction_eligible") is not False
        for fact in directional
    ):
        raise ValueError("NAV_ONLY_PREDICTION_DISABLED")
