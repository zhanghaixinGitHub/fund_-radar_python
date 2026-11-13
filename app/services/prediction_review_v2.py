"""第二版试验的分析格式与检查；核对依据，不用公式生成方向。"""

import json
import re
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.services.direction_1d_protocol import digest

Explanation = Annotated[str, StringConstraints(strip_whitespace=True, min_length=12)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Horizon = Literal["T1", "T5_V1", "T20_V1", "M6_V1"]


@lru_cache(maxsize=1)
def policy_v2():
    directory = Path(__file__).resolve().parents[1] / "data"
    policy = json.loads((directory / "comprehensive_prediction_policy_v2.json").read_text(encoding="utf-8"))
    files = policy["horizon_files"]
    if set(files) != {"T1", "T5_V1", "T20_V1", "M6_V1"}:
        raise ValueError("V2_FOUR_PERIOD_RULE_FILES_REQUIRED")
    rules = {}
    for horizon, name in files.items():
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("V2_PERIOD_RULE_PATH_INVALID")
        rule = json.loads((directory / name).read_text(encoding="utf-8"))
        if (
            rule["horizon"] != horizon
            or set(rule["asset_checks"]) != set(policy["asset_checks"])
            or set(rule["overlay_checks"]) != set(policy["overlay_checks"])
        ):
            raise ValueError("V2_PERIOD_RULE_IDENTITY_OR_TYPES_INVALID")
        rules[horizon] = rule
    if len({r["id"] for r in rules.values()}) != 4:
        raise ValueError("V2_PERIOD_RULE_IDS_NOT_DISTINCT")
    # 汇总后一起计算校验值，四份规则任何一份改动都必须另建试验。
    return {**policy, "horizons": rules}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Quote(Strict):
    kind: Literal["FACT", "DOCUMENT"]
    ref: Identifier
    text: Annotated[str, StringConstraints(min_length=8)]


class Classification(Strict):
    base_type: Identifier
    overlays: list[Identifier]
    proofs: list[Quote] = Field(min_length=1)
    explanation: Explanation


class Check(Strict):
    status: Literal["SUPPORTED", "LIMITED", "MISSING", "NOT_APPLICABLE"]
    refs: list[Identifier]
    note: Explanation
    material_to_direction: bool
    gap_effect: Literal["NONE", "QUALIFY", "BLOCK"] = "NONE"
    impact_reason: Explanation | None = None
    follow_up: Explanation | None = None


class Reading(Strict):
    kind: Literal["FACT", "DOCUMENT"]
    ref: Identifier
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class PriceObservation(Strict):
    observed_on: Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    quote: Quote


class Factor(Strict):
    id: Identifier
    event_group: Identifier
    role: Literal["SUPPORT_UP", "SUPPORT_DOWN", "MIXED", "BACKGROUND"]
    importance: Literal["MAIN", "AUXILIARY", "BACKGROUND"]
    quotes: list[Quote] = Field(min_length=1)
    fund_link: Explanation
    coverage: Literal["MAJOR", "SUBSTANTIAL", "PARTIAL", "UNKNOWN"]
    coverage_note: Explanation
    stage: Literal["REALIZED", "IN_PROGRESS", "PLANNED", "HISTORICAL"]
    novelty: Literal["NEW", "CHANGED", "BACKGROUND"]
    novelty_note: Explanation
    priced: Literal["PARTIAL", "NOT_OBSERVED", "UNKNOWN", "FULLY_REFLECTED"]
    pricing_note: Explanation
    time_bucket: Literal["BASE_REALIZED", "REALIZED_NOT_IN_NAV", "FUTURE", "ONGOING_EFFECT"]
    timing: Literal["MATCHED", "CONDITIONAL", "UNCLEAR", "OUTSIDE"]
    mechanism: Explanation
    window_path: Explanation
    invalidation: Explanation
    continuation_explanation: Explanation | None = None
    continuation_quotes: list[Quote] = Field(default_factory=list)
    coverage_basis: Literal[
        "UNKNOWN", "PRODUCT_MANDATE", "INDEX_REPRESENTATIVE", "REPORTED_PORTFOLIO", "BOUND_ANALYSIS"
    ] = "UNKNOWN"
    coverage_refs: list[Identifier] = Field(default_factory=list)
    driver_kind: Literal["EVENT", "RATES_POLICY", "PRICE_CONTINUATION", "OPERATING", "ASSET_INCOME"] | None = None
    price_observations: list[PriceObservation] = Field(default_factory=list)


class Scenario(Strict):
    kind: Literal["BASE", "FAVORABLE", "ADVERSE"]
    refs: list[Identifier] = Field(min_length=1)
    assumptions: Explanation
    outcome_path: Explanation
    verification: Explanation


class SensitivityCase(Strict):
    kind: Literal["EVIDENCE_GAP", "FUTURE_RISK"] = "EVIDENCE_GAP"
    condition: Explanation
    plausibility: Literal["PLAUSIBLE", "REMOTE"]
    direction_effect: Literal["SAME", "WEAKER", "REVERSES", "UNKNOWN"]
    refs: list[Identifier] = Field(min_length=1)
    explanation: Explanation


class PeriodAssessment(Strict):
    rule_id: Identifier
    primary_driver: Explanation
    window_reason: Explanation
    counter_tradeoff: Explanation
    missing_effect: Explanation


class Review(Strict):
    version: Identifier
    policy_hash: Identifier
    packet_hash: Identifier
    fund_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")]
    horizon: Horizon
    classification: Classification
    reading: list[Reading] = Field(min_length=1)
    checks: dict[str, Check]
    factors: list[Factor] = Field(min_length=1)
    decision: Literal["UP", "DOWN", "FLAT", "ABSTAIN", "BLOCKED"]
    summary: Explanation
    dominant_factors: list[Identifier]
    strongest_counter: list[Identifier]
    counter_search_result: Literal["FOUND", "NO_VERIFIABLE_COUNTER"]
    counter_search_scope: Explanation
    tradeoff: Explanation
    sensitivity: Explanation
    sensitive_to_unknowns: bool
    sensitivity_cases: list[SensitivityCase] = Field(default_factory=list)
    output_kind: Literal["DATED_ANALYSIS", "CONDITIONAL_OUTLOOK"] = "DATED_ANALYSIS"
    change_conditions: list[Explanation] = Field(min_length=1)
    abstention_reasons: list[Identifier]
    recovery_conditions: list[Explanation]
    scenarios: list[Scenario]
    cross_horizon: Explanation
    limitations: list[Explanation] = Field(min_length=1)
    period_assessment: PeriodAssessment


TYPE_MARKERS = {
    "BROAD_INDEX": ("指数",),
    "SECTOR_INDEX": ("指数",),
    "ACTIVE_EQUITY": ("股票型", "股票投资", "股票基金"),
    "MIXED": ("混合型", "灵活配置", "混合基金"),
    "BOND": ("债券基金", "债券型", "政府债券"),
    "CREDIT_BOND": ("信用债", "企业债", "公司债"),
    "CONVERTIBLE": ("可转债", "可转换债"),
    "GOLD": ("黄金",),
    "COMMODITY": ("商品", "原油", "期货"),
    "FOF": ("基金中基金", "FOF"),
    "REIT": ("基础设施",),
    "MONEY_MARKET": ("货币市场",),
    "FEEDER": ("联接",),
    "ENHANCED": ("增强",),
    "OVERSEAS": (
        "QDII",
        "境外",
    ),
    "PASSIVE": ("指数", "被动", "跟踪"),
    "THEMATIC": ("行业", "主题", "指数"),
}


def fact_text(fact):
    """冻结程序事实和原文引句分别保留，不声称程序整理文字就是原文。"""
    return "\n".join(dict.fromkeys(s for s in [fact.get("text", ""), *fact.get("quotes", [])] if s))


def catalog_v2(data):
    return {
        **{("FACT", key): fact_text(fact) for key, fact in data.get("facts", {}).items()},
        **{("DOCUMENT", doc["id"]): doc.get("body", "") for doc in data.get("documents", [])},
    }


def required_checks(classification, horizon):
    p = policy_v2()
    if classification.base_type not in p["asset_checks"]:
        raise ValueError("V2_ASSET_TYPE_UNKNOWN")
    if len(classification.overlays) != len(set(classification.overlays)):
        raise ValueError("V2_OVERLAY_DUPLICATE")
    result = {**p["common_checks"], **p["horizons"][horizon]["checks"]}
    result.update(p["horizons"][horizon]["asset_checks"][classification.base_type])
    for overlay in classification.overlays:
        if overlay not in p["overlay_checks"]:
            raise ValueError("V2_OVERLAY_UNKNOWN")
        result.update(p["horizons"][horizon]["overlay_checks"][overlay])
    return result


def validate_review_v2(packet, data, value, *, at=None):
    """校验单基金单周期；不把人工填写的实质说明当作机器证明的经济事实。"""
    r = Review.model_validate(value)
    p = policy_v2()
    period_rule = p["horizons"][r.horizon]
    if r.period_assessment.rule_id != period_rule["id"]:
        raise ValueError("V2_PERIOD_ASSESSMENT_RULE_MISMATCH")
    if (
        r.version != p["version"]
        or r.policy_hash != digest(p)
        or r.packet_hash != digest(packet)
        or r.fund_code != packet["fund_code"]
        or data["fund_code"] != r.fund_code
    ):
        raise ValueError("V2_REVIEW_IDENTITY_MISMATCH")
    if at is not None and datetime.fromisoformat(packet["captured_at"]) > at:
        raise ValueError("V2_FUTURE_PACKET")
    source_cutoff = datetime.fromisoformat(packet["source_cutoff"])
    captured = datetime.fromisoformat(packet["captured_at"])
    if source_cutoff.tzinfo is None or captured.tzinfo is None or source_cutoff > captured:
        raise ValueError("V2_CUTOFF_TIMEZONE_REQUIRED")
    documents = {doc["id"]: doc for doc in data.get("documents", [])}
    texts = catalog_v2(data)
    spans = {}

    def source_time_check(kind, ref):
        refs = [ref] if kind == "DOCUMENT" else data["facts"][ref].get("source_refs", [])
        for key in refs:
            doc = documents.get(key)
            if doc is None:
                continue
            for name in ("available_at", "published_at"):
                if doc.get(name):
                    stamp = datetime.fromisoformat(doc[name])
                    if stamp.tzinfo is None or stamp > source_cutoff:
                        raise ValueError("V2_SOURCE_AFTER_CUTOFF")
            if doc.get("published_date", "") > packet["source_cutoff"][:10]:
                raise ValueError("V2_SOURCE_AFTER_CUTOFF")

    for row in r.reading:
        key = (row.kind, row.ref)
        if key not in texts or not 0 <= row.start < row.end <= len(texts[key]):
            raise ValueError("V2_READING_RANGE_INVALID")
        source_time_check(row.kind, row.ref)
        spans.setdefault(key, []).append((row.start, row.end))

    def quote_check(q):
        key = (q.kind, q.ref)
        body = texts.get(key, "")
        # 重复引句取真正登记的任一连续位置，不能拼接或借未读同名文本通过。
        start = body.find(q.text)
        covered = False
        while start >= 0:
            covered |= any(a <= start and start + len(q.text) <= b for a, b in spans.get(key, []))
            start = body.find(q.text, start + 1)
        if not covered:
            raise ValueError("V2_QUOTE_OUTSIDE_READING")
        source_time_check(q.kind, q.ref)

    for proof in r.classification.proofs:
        quote_check(proof)
        doc = documents.get(proof.ref, {}) if proof.kind == "DOCUMENT" else {}
        if proof.kind != "DOCUMENT" or r.fund_code not in doc.get("codes", []):
            raise ValueError("V2_CLASSIFICATION_OWN_DOCUMENT_REQUIRED")
        if doc.get("kind") not in {"REPORT", "PROSPECTUS", "ANNOUNCEMENT"}:
            raise ValueError("V2_CLASSIFICATION_PRODUCT_DOCUMENT_REQUIRED")
    proof_text = "\n".join(q.text for q in r.classification.proofs)
    for label in [r.classification.base_type, *r.classification.overlays]:
        if label not in TYPE_MARKERS or not any(token in proof_text for token in TYPE_MARKERS[label]):
            raise ValueError("V2_CLASSIFICATION_EVIDENCE_REQUIRED:" + label)

    # 正式档案用于拦住明显错分；报告引用仍需实际阅读，不能仅凭基金名称批准类型。
    profile = packet.get("profile", {})
    compatible = {
        "STOCK": {"BROAD_INDEX", "SECTOR_INDEX", "ACTIVE_EQUITY"},
        "MIXED": {"MIXED", "FOF"},
        "BOND": {"BOND", "CREDIT_BOND", "CONVERTIBLE"},
        "OTHER": {"GOLD", "COMMODITY", "FOF", "REIT", "MONEY_MARKET"},
        "MONEY": {"MONEY_MARKET"},
    }
    if r.classification.base_type not in compatible.get(profile.get("fund_type"), set()):
        raise ValueError("V2_CLASSIFICATION_PROFILE_CONFLICT")
    name = profile.get("fund_name") or ""
    mandatory = []
    if "联接" in name:
        mandatory.append("FEEDER")
    if "QDII" in name.upper():
        mandatory.append("OVERSEAS")
    if "增强" in (profile.get("invest_type") or ""):
        mandatory.append("ENHANCED")
    if not set(mandatory) <= set(r.classification.overlays):
        raise ValueError("V2_REQUIRED_OVERLAY_OMITTED")
    if profile.get("fund_type") == "BOND" and "可转债" in name and r.classification.base_type != "CONVERTIBLE":
        raise ValueError("V2_CONVERTIBLE_CANNOT_USE_PLAIN_BOND_RULE")

    expected = required_checks(r.classification, r.horizon)
    if set(r.checks) != set(expected):
        raise ValueError("V2_REQUIRED_CHECKS_MISMATCH")
    known_refs = {ref for _, ref in spans}
    for check in r.checks.values():
        if not set(check.refs) <= known_refs:
            raise ValueError("V2_UNREAD_CHECK_REFERENCE")
        if check.status in {"SUPPORTED", "NOT_APPLICABLE"} and not check.refs:
            raise ValueError("V2_CHECK_EVIDENCE_REQUIRED")
        if check.status in {"MISSING", "LIMITED"}:
            if check.gap_effect == "NONE" or check.impact_reason is None or check.follow_up is None:
                raise ValueError("V2_GAP_IMPACT_AND_FOLLOWUP_REQUIRED")
            if check.material_to_direction != (check.gap_effect == "BLOCK"):
                raise ValueError("V2_GAP_IMPACT_FLAG_CONFLICT")
        elif check.gap_effect != "NONE":
            raise ValueError("V2_RESOLVED_CHECK_HAS_GAP")
    for key, check in r.checks.items():
        if check.status == "NOT_APPLICABLE":
            # 没有具体事件不等于没有持续驱动；只允许有阅读依据的事件项目说明不适用。
            if key not in p["event_checks_allowing_not_applicable"] or check.impact_reason is None:
                raise ValueError("V2_SELECTED_CHECK_CANNOT_BE_WAIVED")
            if not any(
                documents.get(ref, {}).get("kind") in {"DISCLOSURE_SEARCH", "REPORT", "PROSPECTUS"}
                for ref in check.refs
            ):
                raise ValueError("V2_NOT_APPLICABLE_EVIDENCE_REQUIRED")
    factors = {f.id: f for f in r.factors}
    if len(factors) != len(r.factors):
        raise ValueError("V2_FACTOR_ID_DUPLICATE")
    for f in r.factors:
        for q in [*f.quotes, *f.continuation_quotes, *(o.quote for o in f.price_observations)]:
            quote_check(q)
            if (
                q.kind == "FACT"
                and data["facts"][q.ref].get("direction_eligible") is False
                and f.role not in {"BACKGROUND", "MIXED"}
            ):
                raise ValueError("V2_INELIGIBLE_DIRECTION_FACT")
        if f.importance == "BACKGROUND" and f.role != "BACKGROUND":
            raise ValueError("V2_BACKGROUND_DIRECTION_CONFLICT")
        if not set(f.coverage_refs) <= known_refs:
            raise ValueError("V2_COVERAGE_REFERENCE_UNKNOWN")
    if not set(r.dominant_factors + r.strongest_counter) <= set(factors):
        raise ValueError("V2_FACTOR_REFERENCE_UNKNOWN")
    if set(r.dominant_factors) & set(r.strongest_counter):
        raise ValueError("V2_COUNTER_NOT_INDEPENDENT")
    if not set(r.abstention_reasons) <= set(p["abstention_reasons"]):
        raise ValueError("V2_ABSTENTION_REASON_UNKNOWN")
    for scenario in r.scenarios:
        if not set(scenario.refs) <= known_refs:
            raise ValueError("V2_SCENARIO_REFERENCE_UNKNOWN")
    for case in r.sensitivity_cases:
        if not set(case.refs) <= known_refs:
            raise ValueError("V2_SENSITIVITY_REFERENCE_UNKNOWN")
    could_reverse = any(
        c.kind == "EVIDENCE_GAP" and c.plausibility == "PLAUSIBLE" and c.direction_effect in {"REVERSES", "UNKNOWN"}
        for c in r.sensitivity_cases
    )
    if r.sensitivity_cases and r.sensitive_to_unknowns != could_reverse:
        raise ValueError("V2_SENSITIVITY_FLAG_CONFLICT")
    if r.output_kind == "CONDITIONAL_OUTLOOK" and r.horizon != "M6_V1":
        raise ValueError("V2_CONDITIONAL_OUTLOOK_ONLY_M6")
    if r.horizon == "M6_V1" and (
        len(r.scenarios) != 3 or {s.kind for s in r.scenarios} != {"BASE", "FAVORABLE", "ADVERSE"}
    ):
        raise ValueError("V2_M6_SCENARIOS_REQUIRED")

    if r.decision in {"UP", "DOWN", "FLAT"}:
        if r.classification.base_type == "MONEY_MARKET":
            raise ValueError("V2_SEPARATE_INCOME_TARGET_REQUIRED")
        if not packet["source_ready"] or (
            packet["periods"][r.horizon]["status"] != "RESOLVED" and r.output_kind != "CONDITIONAL_OUTLOOK"
        ):
            raise ValueError("V2_TARGET_OR_SOURCE_UNVERIFIED")
        if any(c.gap_effect == "BLOCK" for c in r.checks.values()):
            raise ValueError("V2_MATERIAL_GAP_REQUIRES_ABSTENTION")
        if r.sensitive_to_unknowns or r.abstention_reasons:
            raise ValueError("V2_UNSTABLE_DIRECTION")
        if r.decision == "FLAT":
            # 阈值是不可变政策的一部分，不能由审阅临时填写。
            if not p["flat_bands"].get(r.classification.base_type, {}).get(r.horizon):
                raise ValueError("V2_FLAT_BAND_UNREGISTERED")
        else:
            if not any(c.kind == "EVIDENCE_GAP" for c in r.sensitivity_cases):
                raise ValueError("V2_CONCRETE_SENSITIVITY_CASE_REQUIRED")
            expected_role = "SUPPORT_UP" if r.decision == "UP" else "SUPPORT_DOWN"
            main = [factors[key] for key in r.dominant_factors]
            if len({f.event_group for f in main}) != len(main):
                raise ValueError("V2_REPEATED_EVENT_AS_MAIN_DRIVER")
            if not main or any(
                f.role != expected_role
                or f.importance != "MAIN"
                or (
                    f.coverage not in {"MAJOR", "SUBSTANTIAL"}
                    and not (f.coverage == "PARTIAL" and f.coverage_basis != "UNKNOWN" and f.coverage_refs)
                )
                or f.timing not in {"MATCHED", "CONDITIONAL"}
                or f.time_bucket not in {"FUTURE", "ONGOING_EFFECT"}
                or f.priced == "FULLY_REFLECTED"
                or (f.novelty == "BACKGROUND" and f.time_bucket != "ONGOING_EFFECT")
                for f in main
            ):
                raise ValueError("V2_FUTURE_MAIN_DRIVER_REQUIRED")
            for factor in main:
                if factor.driver_kind not in period_rule["main_driver_kinds"]:
                    raise ValueError("V2_DRIVER_NOT_SUITABLE_FOR_PERIOD")
                if factor.driver_kind == "PRICE_CONTINUATION":
                    days = set()
                    for observation in factor.price_observations:
                        day = date.fromisoformat(observation.observed_on)
                        q = observation.quote
                        is_market = (
                            data["facts"][q.ref].get("category")
                            in {"市场行情", "持仓行情", "MARKET", "PRICES", "PRICE"}
                            if q.kind == "FACT"
                            else documents[q.ref].get("kind") == "MARKET_DATA"
                        )
                        # 日期必须出现在已读引句里，不能给同一条昨日行情随意标两个日期。
                        compact = re.sub(r"[-/\s]", "", q.text)
                        if not is_market or day.strftime("%Y%m%d") not in compact or day > source_cutoff.date():
                            raise ValueError("V2_PRICE_OBSERVATION_SOURCE_OR_DATE_INVALID")
                        days.add(day)
                    base = packet["periods"][r.horizon].get("base_date")
                    if (
                        len(days) < 2
                        or base is None
                        or max(days) != date.fromisoformat(base)
                        or min(days) < date.fromisoformat(base) - timedelta(days=period_rule["price_lookback_days"])
                    ):
                        raise ValueError("V2_MULTIDAY_PRICE_EVIDENCE_REQUIRED")
                if factor.time_bucket == "ONGOING_EFFECT":
                    # 用历史经营分析持续作用可以，用昨日收盘或产品名称独自外推不可以。
                    categories = {
                        data["facts"][q.ref].get("category") for q in factor.continuation_quotes if q.kind == "FACT"
                    }
                    direct_financial = any(
                        q.kind == "DOCUMENT"
                        and documents[q.ref].get("kind") in {"COMPANY_FINANCIAL", "ISSUER_FINANCIAL", "OFFICIAL_POLICY"}
                        for q in factor.continuation_quotes
                    )
                    if (
                        factor.continuation_explanation is None
                        or not factor.continuation_quotes
                        or (
                            factor.driver_kind != "PRICE_CONTINUATION"
                            and not direct_financial
                            and not categories
                            & {
                                "财务",
                                "公司经营",
                                "政策",
                                "公告",
                                "公司公告",
                                "新闻",
                                "FINANCIAL",
                                "FINANCIALS",
                                "POLICY",
                                "NEWS",
                            }
                        )
                    ):
                        raise ValueError("V2_ONGOING_EFFECT_EVIDENCE_REQUIRED")
            if all(
                all(q.kind == "FACT" and data["facts"][q.ref].get("category") in {"基金净值", "NAV"} for q in f.quotes)
                for f in main
            ):
                raise ValueError("V2_NAV_ONLY_DIRECTION_DISABLED")
            opposite = "SUPPORT_DOWN" if r.decision == "UP" else "SUPPORT_UP"
            if r.counter_search_result == "FOUND" and not r.strongest_counter:
                raise ValueError("V2_INDEPENDENT_COUNTER_REQUIRED")
            if any(factors[key].role not in {opposite, "MIXED"} for key in r.strongest_counter):
                raise ValueError("V2_INDEPENDENT_COUNTER_REQUIRED")
            if r.counter_search_result == "NO_VERIFIABLE_COUNTER":
                search = r.checks["counter_search"]
                if (
                    r.strongest_counter
                    or search.status != "SUPPORTED"
                    or not any(
                        documents.get(ref, {}).get("kind") in {"DISCLOSURE_SEARCH", "NEWS_SEARCH"}
                        for ref in search.refs
                    )
                ):
                    raise ValueError("V2_COUNTER_SEARCH_RECEIPT_REQUIRED")
            if any(
                factors[key].mechanism == f.mechanism and factors[key].quotes == f.quotes
                for key in r.strongest_counter
                for f in main
            ):
                raise ValueError("V2_COUNTER_COPIED_FROM_SUPPORT")
    else:
        if not r.abstention_reasons or not r.recovery_conditions:
            raise ValueError("V2_ABSTENTION_RECOVERY_REQUIRED")
        if r.dominant_factors:
            raise ValueError("V2_ABSTENTION_HAS_DIRECTION_DRIVER")
        if r.decision == "BLOCKED" and not (
            not packet["source_ready"] or packet["periods"][r.horizon]["status"] != "RESOLVED"
        ):
            raise ValueError("V2_BLOCKED_WITHOUT_INPUT_BLOCK")
    # 数值保留在绑定来源的引句；自由推断不得新增收益承诺或概率。
    text_fields = [r.summary, r.tradeoff, r.sensitivity, *r.change_conditions]
    if any(re.search(r"必涨|必跌|稳赚|保证收益|胜率\s*\d|上涨概率\s*\d", s) for s in text_fields):
        raise ValueError("V2_UNSUPPORTED_PROMISE")
    return r


def validate_cross_horizons(reviews):
    """同基金方向相反必须说清原因；只换文字不能证明分别做过分析。"""
    by_code = {}
    for r in reviews:
        by_code.setdefault(r.fund_code, []).append(r)
    for rows in by_code.values():
        if len({digest(r.classification.model_dump()) for r in rows}) != 1:
            raise ValueError("V2_CLASSIFICATION_CHANGED_BETWEEN_HORIZONS")
        assessments = [
            digest({k: v for k, v in r.period_assessment.model_dump().items() if k != "rule_id"}) for r in rows
        ]
        if len(assessments) != len(set(assessments)):
            raise ValueError("V2_COPIED_PERIOD_ASSESSMENT")
        if {r.decision for r in rows} >= {"UP", "DOWN"}:
            if any("无方向分歧" in r.cross_horizon for r in rows if r.decision in {"UP", "DOWN"}):
                raise ValueError("V2_CROSS_HORIZON_EXPLANATION_REQUIRED")
