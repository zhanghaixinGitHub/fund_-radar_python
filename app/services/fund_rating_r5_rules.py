"""R5正式计算使用的支付、责任接续及持有人子项。

这里的75仍是候选正常交付锚，不是安全概率。任何数值为零必须来自已披露
事实；必要金额、责任日期或用途尚未核准时返回异常，由总分保留为空。
"""

from datetime import date, timedelta
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services.material_content import content_digest, source_bytes


def amount(value):
    if value is None or isinstance(value, bool):
        raise ValueError("R4_REQUIRED_AMOUNT_MISSING")
    n = D(str(value))
    if not n.is_finite() or n < 0:
        raise ValueError("R4_INVALID_NONNEGATIVE_AMOUNT")
    return n


def check_refs(node, registry, asof):
    refs = node.get("source_refs")
    if not refs or not set(refs) <= set(registry):
        raise ValueError("R4_ORIGINAL_REFERENCES_MISSING")
    for h in refs:
        r = registry[h]
        if r.get("published_on") and r["published_on"] > asof:
            raise ValueError("R4_SOURCE_AFTER_CUTOFF")
        if content_digest(source_bytes(Path(r["path"]))) != h:
            raise ValueError("R4_SOURCE_BYTES_CHANGED")


def payment_score(node):
    """按用途分配已披露本金，证券账户只抵证券交收义务，避免跨用途高估。

    B：银行普通付款本金；T：证券结算专用本金；S：可由T支付的已核准交收
    义务；P：全部账面负债。先用T抵S，再用B抵余下义务，避免重复使用同一
    笔钱。只统计原件可确认的本金，不净掉应收款，不把保证金当一般现金。
    """
    if node.get("scope_closed") is not True or node.get("currency") != "CNY" or not node.get("as_of"):
        raise ValueError("R4_PAYMENT_SCOPE_NOT_CLOSED")
    if node.get("restriction_amount_pending") is not False:
        raise ValueError("R4_CASH_RESTRICTION_UNRESOLVED")
    # 银行扣限制后已覆盖全部负债时，原用途公式在合法未知T/S域恒定。
    # 由原件编译这个显式域，不给未知证券账户本金或负债用途补0。
    if "coverage_domain" in node:
        from app.services.fund_rating_payment_domain import payment_domain_score

        return payment_domain_score(node)
    b, t, s, p = [
        amount(node[k])
        for k in ["bank_principal", "broker_settlement_principal", "settlement_liabilities", "all_book_liabilities"]
    ]
    if s > p:
        raise ValueError("R4_SETTLEMENT_LIABILITIES_EXCEED_TOTAL")
    # 发现明确限制时必须有金额并扣除，不能用空金额当零。
    for r in node.get("disclosed_restrictions", []):
        x = amount(r["amount"])
        if r["account"] == "bank":
            b -= x
        elif r["account"] == "broker":
            t -= x
        else:
            raise ValueError("R4_UNKNOWN_RESTRICTION_ACCOUNT")
    if min(b, t) < 0:
        raise ValueError("R4_RESTRICTION_EXCEEDS_ACCOUNT_PRINCIPAL")
    with localcontext() as c:
        c.prec = 65
        via_broker = min(t, s)
        via_bank = min(b, p - via_broker)
        covered = via_broker + via_bank
        ratio = D(1) if p == 0 else covered / p
        value = D(75) * ratio
    return value, {
        "bank_principal": str(b),
        "broker_settlement_principal": str(t),
        "settlement_liabilities": str(s),
        "other_book_liabilities": str(p - s),
        "all_book_liabilities": str(p),
        "covered_via_broker": str(via_broker),
        "covered_via_bank": str(via_bank),
        "covered_total": str(covered),
        "uncovered_book_liabilities": str(p - covered),
        "covered_ratio": str(ratio),
        "point": str(value),
        "not_payment_failure_inferred": True,
        "not_future_redemption_safety": True,
        "basis": "公开报表本金与全部账面负债的保守静态用途匹配，不是逐笔到期清偿审计。",
    }


def continuity_score(node):
    """日期连续及当前履职证据同等处理，不奖励共管天数、不按短任期扣分。"""
    if (
        node.get("timeline_closed") is not True
        or node.get("current_role_verified") is not True
        or node.get("arrangements_review_closed") is not True
    ):
        raise ValueError("R4_CONTINUITY_EVIDENCE_NOT_CLOSED")
    start = date.fromisoformat(node["period_start"])
    end = date.fromisoformat(node["period_end"]) + timedelta(days=1)
    spans = []
    for r in node["intervals"]:
        if not r.get("name") or not r.get("source_refs"):
            raise ValueError("R4_RESPONSIBLE_PERSON_EVIDENCE_MISSING")
        a = date.fromisoformat(r["start"])
        b = date.fromisoformat(r["end"]) if r.get("end") else end
        if b <= a:
            raise ValueError("R4_INVALID_APPOINTMENT_INTERVAL")
        a = max(a, start)
        b = min(b, end)
        if b > a:
            spans.append((a, b))
    if not spans:
        raise ValueError("R4_EMPTY_APPOINTMENT_COVERAGE")
    cursor = start
    gaps = []
    for a, b in sorted(spans):
        if a > cursor:
            gaps.append([cursor.isoformat(), a.isoformat()])
        cursor = max(cursor, b)
    if cursor < end:
        gaps.append([cursor.isoformat(), end.isoformat()])
    events = node.get("confirmed_functional_defects")
    if events is None:
        raise ValueError("R4_CONTINUITY_EVENT_SCOPE_UNKNOWN")
    # 负面事实必须是已确认适用于该基金的实际影响，不把未披露内部文件当缺陷。
    mapping = {"SEVERE_FUNCTION_FAILURE": 0, "UNRESOLVED_OPERATIONAL_GAP": 25, "LIMITED_OPERATIONAL_DEFECT": 50}
    if events:
        if any(
            e.get("effect_confirmed") is not True or not e.get("source_refs") or e.get("classification") not in mapping
            for e in events
        ):
            raise ValueError("R4_DEFECT_NOT_VERIFIED")
        value = D(min(mapping[e["classification"]] for e in events))
    elif gaps:
        raise ValueError("R4_PUBLIC_DATE_GAP_NEEDS_REVIEW")
    else:
        value = D(75)
    # 100不从日期连续推定。此轮六例均未提交足够强化证据，不保守补满分。
    return value, {
        "coverage_start": start.isoformat(),
        "coverage_end_inclusive": (end - timedelta(days=1)).isoformat(),
        "public_date_gaps": gaps,
        "normal_handover_forms_equally_treated": True,
        "point": str(value),
        "scope": "公开责任及履职接续，不声称内部交接程序经过审计。",
    }


def holders_score(node):
    """结构、机制、实际履职三者统一核准；静态现金不足不重复门控。

    分散、集中、ETF及普通开放式均需相同层级的结构与实际执行证据。
    份额表真实零申赎不等于缺资料；正常只75，不把一季正常当压力验收。
    """
    if node.get("structure_condition_protocol"):
        from app.services.fund_rating_cmf_liquidity import PROTOCOL, invariant_holders_score

        if node["structure_condition_protocol"] != PROTOCOL:
            raise ValueError("R5_HOLDER_STRUCTURE_CONDITION_PROTOCOL")
        return invariant_holders_score(node, evaluator=holders_score)
    if any(
        node.get(k) is not True
        for k in ["structure_closed", "share_flow_closed", "actual_execution_verified", "redemption_mechanism_verified"]
    ):
        raise ValueError("R5_HOLDER_BASIC_EVIDENCE_MISSING")
    if node.get("structure_type") not in {"DISPERSED", "CONCENTRATED_ETF", "CONCENTRATED_OPEN_END", "CLOSED_PERIOD"}:
        raise ValueError("R5_HOLDER_STRUCTURE_NOT_REVIEWED")
    defects = node.get("confirmed_pressure_effects")
    if defects is None:
        raise ValueError("R5_HOLDER_EFFECT_SCOPE_UNKNOWN")
    mapping = {"LIMITED_ACTUAL_PRESSURE": 50, "MATERIAL_UNRESOLVED_PRESSURE": 25, "SEVERE_INDEPENDENT_HOLDER_HARM": 0}
    for event in defects:
        if (
            event.get("effect_confirmed") is not True
            or not event.get("source_refs")
            or event.get("classification") not in mapping
        ):
            raise ValueError("R5_HOLDER_DEFECT_NOT_VERIFIED")
        if event.get("only_static_cash_proxy") is not False:
            raise ValueError("R5_STATIC_CASH_IS_NOT_HOLDER_HARM")
    value = D(min(mapping[e["classification"]] for e in defects)) if defects else D(75)
    return value, {
        "point": str(value),
        "actual_redemption_shares": node.get("actual_redemption_shares"),
        "concentration_still_disclosed": True,
        "not_future_pressure_test": True,
        "payment_dependency_removed": True,
        "basis": "全结构统一核准完整结构、适配机制及实际履职；只凭现金覆盖或未来意向未知不另扣持有人分。",
    }
