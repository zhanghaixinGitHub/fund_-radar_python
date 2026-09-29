"""只按披露当时可用的同口径事实构建变化，不把更正公告追述值倒灌为旧原件。"""

from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal

from app.services.fund_earnings_evidence_v1 import compare_claims, review_claim
from app.services.fund_information_history_v1 import normalize

KEYS = ("issuer", "period_start", "period_end", "metric", "basis", "currency")


def asof_changes(claims, as_of):
    """在已核验输入中选择截止时刻前最近两次同公司、同期间、同口径披露。

    输入必须已经过原件字段核验。未知日期、来源未核准、重复原件数值冲突
    都拒绝；同一可用时刻的不同披露不靠文件名强行排先后。仅一份事实返回
    无前次披露，变化值保持 None。输出是事实变化，绝不输出基金涨跌标签。
    """
    cutoff = datetime.fromisoformat(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("AS_OF_TIMEZONE_REQUIRED")
    groups, seen = defaultdict(list), {}
    for claim in claims:
        available = datetime.fromisoformat(claim["available_at"])
        if available.tzinfo is None:
            raise ValueError("AVAILABLE_TIMEZONE_REQUIRED")
        # 先截断，再验证来源内容：未来新增披露不会改变过去的查询结果。
        if available > cutoff:
            continue
        published = date.fromisoformat(claim["published_date"])
        earliest = datetime.fromisoformat(str(published + timedelta(days=1)) + "T08:00:00+08:00")
        if available < earliest:
            raise ValueError("PUBLIC_DATE_BACKDATED")
        if claim.get("evidence_role", "CURRENT_DISCLOSURE") != "CURRENT_DISCLOSURE":
            raise ValueError("QUOTED_HISTORY_NOT_INDEPENDENT_DISCLOSURE")
        if not claim.get("source_identity_verified") or claim.get("revision_issues"):
            raise ValueError("SOURCE_REVIEW_NOT_PASSED")
        key = tuple(claim[k] for k in KEYS)
        identity = (*key, claim["document_id"])
        signature = (claim["kind"], claim["available_at"], tuple(m["cny"] for m in claim["money"]))
        if identity in seen:
            if seen[identity] != signature:
                raise ValueError("CONFLICTING_DUPLICATE_DISCLOSURE")
            continue
        seen[identity] = signature
        groups[key].append(claim)
    result = []
    for key, values in sorted(groups.items()):
        times = defaultdict(list)
        for value in values:
            times[datetime.fromisoformat(value["available_at"])].append(value)
        ordered = sorted(times)
        latest = times[ordered[-1]]
        previous = times[ordered[-2]] if len(ordered) > 1 else []
        common = {
            **dict(zip(KEYS, key, strict=True)),
            "as_of": as_of,
            "training_ready": False,
            "selection_scope": "REVIEWED_INPUT_FACTS_ONLY",
            "complete_history_admitted": False,
        }
        if len(latest) != 1 or len(previous) > 1:
            result.append(
                {
                    **common,
                    "status": "AMBIGUOUS_DISCLOSURE_ORDER",
                    "change": None,
                    "latest_ids": sorted(v["document_id"] for v in latest),
                    "previous_ids": sorted(v["document_id"] for v in previous),
                }
            )
        elif not previous:
            result.append(
                {
                    **common,
                    "status": "NO_PRIOR_COMPARABLE_DISCLOSURE",
                    "change": None,
                    "latest_ids": [latest[0]["document_id"]],
                    "previous_ids": [],
                }
            )
        else:
            result.append(
                {
                    **common,
                    "status": "COMPARABLE_DISCLOSURE_CHANGE",
                    "latest_ids": [latest[0]["document_id"]],
                    "previous_ids": [previous[0]["document_id"]],
                    "change": compare_claims(previous[0], latest[0], as_of),
                }
            )
    return result


def review_correction(before, after, pages):
    """核验一份更正公告中的更正前后金额；两者都只在更正公告公开后可用。

    before/after 具有相同公司、期间和口径，分别锚定“更正前”和“更正后”
    区块内的唯一字段。追述旧值可以解释本次更正幅度，但不能替代旧原件，
    也不能以原报告的落款日期制造一条更早已知事实。
    """
    if any(before[k] != after[k] for k in (*KEYS, "document_id", "published_date", "page")):
        raise ValueError("CORRECTION_SCOPE_MISMATCH")
    if before["kind"] != "REPORTED_RESULT" or after["kind"] != "REPORTED_RESULT":
        raise ValueError("CORRECTION_REPORTED_VALUES_REQUIRED")
    page = normalize(pages[before["page"] - 1])
    if page.count("更正前：") != 1 or page.count("更正后：") != 1:
        raise ValueError("CORRECTION_SECTIONS_NOT_UNIQUE")
    before_pos, after_pos = page.index("更正前："), page.index("更正后：")
    old, new = review_claim(before, pages), review_claim(after, pages)
    if not before_pos < old["anchor"]["offset"] < after_pos < new["anchor"]["offset"]:
        raise ValueError("CORRECTION_FIELD_IN_WRONG_SECTION")
    old["evidence_role"] = "QUOTED_PREVIOUS_VALUE"
    new["evidence_role"] = "QUOTED_CORRECTED_VALUE"
    return {
        "before": old,
        "after": new,
        "change_cny": str(Decimal(new["money"][0]["cny"]) - Decimal(old["money"][0]["cny"])),
        "available_at": new["available_at"],
        "original_report_source_present": False,
        "may_backfill_original_report": False,
        "training_ready": False,
    }
