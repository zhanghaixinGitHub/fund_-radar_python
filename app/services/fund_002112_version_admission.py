"""002112 的独立版本准入：逐次使用判定，不训练、不联网、不改旧资料。

来源审阅与日期计算分开。只有已审阅、摘要锁定的证据条目能进入计算；文件
存在、数值相等、调用者传入 eligible=True 都不是来源证明。历史第三方存档
可以今天取得，不要求本系统当年下载。真实来源适配后的条目应另存新协议，
不能向已冻结协议补一个通过标记。
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

from app.services.fund_002112_zero_fit_review import digest, file_hash

PEERS = {"017493", "160323"}
KINDS = {"PROVIDER_VERSION", "INDEPENDENT_ARCHIVE"}


def day(value):
    """只接受日期级公开时间；未知或不规范日期拒绝，不从接收时间倒推。"""
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("INVALID_PUBLIC_DATE")
    return value


def nav_value(value):
    """单位净值按原始十进制比较，禁止零、非有限值及浮点近似相等。"""
    result = Decimal(str(value))
    if not result.is_finite() or result <= 0:
        raise ValueError("INVALID_RAW_UNIT_NAV")
    return result


def load_evidence(entries, reviewed_digests):
    """验证人工/来源适配审阅后锁定的证据，不接受证据自己声称已经审阅。

    entries 中 facts 为审阅者从 source 原文抽取的基金、日期、原始值/报告
    摘要与版本时间；审阅说明必须写明原文定位及版本时间语义。来源原文和整条
    抽取结果分别验摘要，防止审阅后换字节、改数值或提前日期。该检查保障绑定
    一致性，不声称仅凭 SHA 能证明来源陈述真实。真实审阅清单为空时不会放行。
    """
    result = {}
    for entry in entries:
        if digest(entry) not in reviewed_digests:
            raise ValueError("EVIDENCE_NOT_IN_REVIEWED_PROTOCOL")
        source, facts = entry["source"], entry["facts"]
        if not entry.get("source_locator") or not entry.get("version_semantics"):
            raise ValueError("SOURCE_REVIEW_INCOMPLETE")
        if entry["kind"] not in KINDS:
            raise ValueError("DATED_CATALOG_OR_CURRENT_COPY_IS_NOT_VERSION_RECORD")
        if file_hash(Path(source["path"])) != source["sha256"]:
            raise ValueError("EVIDENCE_SOURCE_CHANGED")
        if facts["fund_code"] not in PEERS:
            raise ValueError("EVIDENCE_FUND_OUTSIDE_SCOPE")
        if facts["type"] == "NAV":
            if not "2021-01-01" <= day(facts["business_date"]) <= "2023-12-31":
                raise ValueError("NAV_OUTSIDE_ALLOWED_SCOPE")
            nav_value(facts["value"])
        elif facts["type"] == "REPORT":
            if not "2020-01-01" <= day(facts["business_date"]) <= "2023-12-31":
                raise ValueError("REPORT_OUTSIDE_ALLOWED_SCOPE")
            if len(facts["value"]) != 64 or any(c not in "0123456789abcdef" for c in facts["value"]):
                raise ValueError("REPORT_CONTENT_HASH_INVALID")
        else:
            raise ValueError("UNKNOWN_EVIDENCE_TYPE")
        # 原公告日保留；修订公开日或独立存档见证日只能推迟可用时间。
        dates = [day(facts[k]) for k in ("publication_date", "version_publication_date")]
        if dates[0] < facts["business_date"] or dates[1] < dates[0]:
            raise ValueError("VERSION_DATE_BEFORE_ORIGINAL_PUBLICATION")
        if entry["kind"] == "INDEPENDENT_ARCHIVE":
            witness = day(entry["witness_date"])
            if witness < dates[1]:
                raise ValueError("ARCHIVE_BEFORE_VERSION_PUBLICATION")
            dates.append(witness)
        result.setdefault(facts["key"], []).append({**facts, "known_on": max(dates), "evidence_sha256": digest(entry)})
    return result


def version_at(requirement, cutoff, index):
    """按实际使用截止日选择当时已知的最新版本；有更正时不能退回旧值。

    输入截止日为目标 U；标签截止日为原训练阶段开始日。均要求公开日在截止
    日之前。label 的额外成熟时间由调用者独立检查，不与输入时间混用。
    """
    day(cutoff)
    versions = index.get(requirement["key"], [])
    if not versions:
        return {"eligible": False, "reason": "HISTORICAL_VERSION_EVIDENCE_MISSING"}
    for row in versions:
        if any(row[k] != requirement[k] for k in ("fund_code", "business_date", "type")):
            return {"eligible": False, "reason": "VERSION_IDENTITY_MISMATCH"}
        if row["publication_date"] != requirement["publication_date"]:
            return {"eligible": False, "reason": "ORIGINAL_PUBLICATION_CHANGED"}
    available = [r for r in versions if r["known_on"] < cutoff]
    if not available:
        return {"eligible": False, "reason": "NO_VERSION_KNOWN_BEFORE_USE"}
    latest = max(r["version_publication_date"] for r in available)
    chosen = [r for r in available if r["version_publication_date"] == latest]
    norm = nav_value if requirement["type"] == "NAV" else str
    if len({norm(r["value"]) for r in chosen}) != 1:
        return {"eligible": False, "reason": "CONFLICTING_HISTORICAL_VERSIONS"}
    if norm(chosen[0]["value"]) != norm(requirement["value"]):
        return {"eligible": False, "reason": "FROZEN_VALUE_DIFFERS_FROM_KNOWN_VERSION"}
    return {
        "eligible": True,
        "reason": "VERSION_BOUND_BEFORE_USE",
        "evidence_sha256": sorted(r["evidence_sha256"] for r in chosen),
        "known_on": min(r["known_on"] for r in chosen),
    }


def row_at(row, cutoff, requirements, index):
    """完整输入与两个标签端点分别检查；返回具体失败依赖，不覆盖原候选。

    row 是已冻结的完整 20 项输入身份及其依赖。原成熟时间原样保留，额外验证
    两个标签净值在训练开始前均已公开。结果只称资料版本准入，不冒充全包门槛
    检查、训练授权或候选采用。
    """
    day(row["target"])
    day(cutoff)
    if len(set(row["input_keys"])) != 62 or len(set(row["label_keys"])) != 2:
        raise ValueError("EXPECTED_61_NAV_ONE_REPORT_AND_TWO_LABEL_ENDPOINTS")
    checks = []
    uses = [(key, row["target"], "INPUT") for key in row["input_keys"]]
    uses += [(key, cutoff, "LABEL") for key in row["label_keys"]]
    for key, limit, purpose in uses:
        if key not in requirements:
            check = {"eligible": False, "reason": "REQUIRED_DEPENDENCY_MISSING"}
        else:
            check = version_at(requirements[key], limit, index)
        if not check["eligible"]:
            checks.append({"key": key, "purpose": purpose, "before": limit, "reason": check["reason"]})
    time_ok = row["target"] < cutoff and row["mature_at"] < cutoff + "T00:00:00+08:00"
    return {
        "fund_code": row["fund_code"],
        "target": row["target"],
        "cutoff": cutoff,
        "time_eligible": time_ok,
        "version_eligible": not checks,
        "material_eligible": time_ok and not checks,
        "failed_dependencies": checks,
    }
