"""公告、政策、新闻与经理策略的独立事实层。

事实入库不等于预测有效。原样保存来源身份、原文锚点、日期精度和未知项；
披露事件按文档去重，不把回购计划当成交，也不把合同额当利润。
本层只写调用方指定的新研究目录，不改旧样例表、不发布业务事件。
"""

import re
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

from app.services.fund_002112_zero_fit_review import digest, file_hash, read_json, save_once

ZONE = timezone(timedelta(hours=8))
CATEGORIES = ("PERFORMANCE", "BUYBACK", "MAJOR_CONTRACT", "INDUSTRY_POLICY", "NEWS", "MANAGER_STRATEGY")
STAGES = {
    "FORECAST",
    "CORRECTION",
    "RESULT",
    "PLAN",
    "PROGRESS",
    "COMPLETED",
    "TERMINATED",
    "SIGNED",
    "DISCLOSED",
    "POLICY_PUBLISHED",
    "REPORT_NARRATIVE",
}


def classification(title):
    """只识别标题明确的披露类型和阶段，不推测利好、利空或交易方向。"""
    if "回购" in title and not any(w in title for w in ("限制性股票", "股权激励", "回购注销", "注销部分")):
        category = "BUYBACK"
    elif any(w in title for w in ("重大合同", "重大经营合同", "重大销售合同")):
        category = "MAJOR_CONTRACT"
    elif any(w in title for w in ("业绩预告", "业绩快报", "年度报告", "季度报告", "半年度报告")):
        if any(w in title for w in ("监事会", "独立董事", "审核意见", "提示性公告")):
            return None
        category = "PERFORMANCE"
    else:
        return None
    if any(w in title for w in ("更正", "修正", "修订")):
        stage = "CORRECTION"
    elif any(w in title for w in ("终止", "取消")):
        stage = "TERMINATED"
    elif any(w in title for w in ("完成", "完毕")):
        stage = "COMPLETED"
    elif "进展" in title:
        stage = "PROGRESS"
    elif "业绩预告" in title:
        stage = "FORECAST"
    elif category == "PERFORMANCE":
        stage = "RESULT"
    elif any(w in title for w in ("方案", "计划", "提议")):
        stage = "PLAN"
    elif category == "MAJOR_CONTRACT" and "签订" in title:
        stage = "SIGNED"
    else:
        stage = "DISCLOSED"
    return category, stage


def available_at(record):
    """历史来源只有日期时，沿用次日 08:00；未知修订时间不提前。"""
    published = record.get("published_date")
    if not published or record.get("revision_unresolved"):
        return None
    days = [date.fromisoformat(published)]
    if record.get("version_publication_date"):
        days.append(date.fromisoformat(record["version_publication_date"]))
    elif record.get("known_revision"):
        return None
    return datetime.combine(max(days) + timedelta(days=1), time(8), ZONE)


def validate(record):
    """校验事实契约；空金额保留未知，禁止将金额、比例和收益概率混用。"""
    if (
        record.get("schema") != "FUND_INFORMATION_FACT_V1"
        or record.get("category") not in CATEGORIES
        or record.get("stage") not in STAGES
    ):
        raise ValueError("FACT_TYPE_OR_STAGE_INVALID")
    if record.get("date_precision") != "DAY" or record.get("published_at") is not None:
        raise ValueError("UNPROVEN_INTRADAY_PUBLICATION")
    if not record.get("document_id") or not record.get("entity_id") or not record.get("title"):
        raise ValueError("FACT_IDENTITY_MISSING")
    if record.get("published_date") and date.fromisoformat(record["published_date"]).year >= 2025:
        raise ValueError("FACT_OUTSIDE_RESEARCH_SCOPE")
    source = record["source"]
    if urlparse(source["url"]).scheme != "https" or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
        raise ValueError("FACT_SOURCE_INVALID")
    if file_hash(source["path"]) != source["sha256"]:
        raise ValueError("FACT_SOURCE_CHANGED")
    if not source.get("use_basis") or not source.get("retention_basis"):
        raise ValueError("SOURCE_USE_BASIS_MISSING")
    if record.get("prediction_score") is not None or record.get("direction") is not None:
        raise ValueError("FACT_MUST_NOT_INVENT_PREDICTION")
    if record["dedup_unit"] != "DISCLOSURE" or not record["anchors"]:
        raise ValueError("SOURCE_ANCHOR_OR_DEDUP_MISSING")
    for number in record["quantities"]:
        value = Decimal(number["value"])
        if not value.is_finite() or number["unit"] not in ("CNY", "CNY_10000", "CNY_100000000", "PERCENT", "SHARES"):
            raise ValueError("QUANTITY_UNIT_OR_VALUE_INVALID")
        if not number["metric"] or not number["anchor"] or not number["basis"]:
            raise ValueError("QUANTITY_MEANING_MISSING")
    if "identity" in record:
        identity = digest([record["entity_type"], record["entity_id"], record["document_id"], record["category"]])
        if record["identity"] != identity:
            raise ValueError("FACT_IDENTITY_CHANGED")
    if "revision" in record and record["revision"] != digest({k: v for k, v in record.items() if k != "revision"}):
        raise ValueError("FACT_REVISION_CHANGED")
    available_at(record)
    return record


def make_fact(*, category, stage, entity_type, entity_id, document_id, title, published, source, anchors, **extra):
    """构造一版独立事实；没有证据的字段留空，不通过默认值制造业务结论。"""
    value = {
        "schema": "FUND_INFORMATION_FACT_V1",
        "category": category,
        "stage": stage,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "document_id": document_id,
        "title": title,
        "published_date": published,
        "published_at": None,
        "date_precision": "DAY",
        "version_publication_date": published,
        "known_revision": stage == "CORRECTION",
        "revision_unresolved": stage == "CORRECTION",
        "source": source,
        "anchors": anchors,
        "quantities": [],
        "unknown_fields": ["calibrated_predictive_effect", "event_lifecycle_identity"],
        "dedup_unit": "DISCLOSURE",
        "prediction_score": None,
        "direction": None,
        **extra,
    }
    validate(value)
    value["identity"] = digest([entity_type, entity_id, document_id, category])
    value["revision"] = digest(value)
    return value


def store_bundle(directory, records, coverage, sources):
    """版本按内容寻址，排他保存后独立读回；并发同内容可复用，不覆盖另一版事实。"""
    directory = Path(directory)
    unique = {}
    for item in records:
        validate(item)
        key = item["identity"]
        if key in unique and unique[key] != item:
            raise ValueError("CONFLICTING_DISCLOSURE_VERSIONS")
        unique[key] = item
    body = {
        "schema": "FUND_INFORMATION_BUNDLE_V1",
        "facts": sorted(unique.values(), key=lambda r: (r["published_date"] or "", r["identity"])),
        "coverage": coverage,
        "sources": sources,
        "business_published": False,
    }
    sha = digest(body)
    target = directory / "bundles" / (sha + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    save_once(target, body)
    if read_json(target) != body:
        raise ValueError("BUNDLE_READBACK_MISMATCH")
    return {
        "path": str(target),
        "sha256": file_hash(target),
        "content_sha256": sha,
        "facts": len(unique),
        "readback_passed": True,
    }


def window_covered(intervals, start, end):
    """检查完整目录查询区间的并集；区间中间缺一天也不能称没有事件。"""
    next_day = date.fromisoformat(start)
    for lower, upper in sorted(intervals):
        lo, hi = date.fromisoformat(lower), date.fromisoformat(upper)
        if hi < next_day:
            continue
        if lo > next_day:
            return False
        next_day = hi + timedelta(days=1)
        if next_day > date.fromisoformat(end):
            return True
    return False


def information_features(records, *, cutoff, companies, report_sha256, coverage):
    """固定 30 日披露计数加策略类别；缺覆盖返回阻断，绝不自动变成零事件。

    公司事件按当时披露的持仓公司关联，不声称预测日仍实际持有。计数单位是公告，
    不是累计回购金额或独立经济事件。政策、新闻只对预登记来源集合统计。
    """
    limit = datetime.fromisoformat(cutoff)
    if limit.tzinfo is None:
        raise ValueError("CUTOFF_TIMEZONE_REQUIRED")
    start = (limit.date() - timedelta(days=30)).isoformat()
    end = (limit.date() - timedelta(days=1)).isoformat()
    missing = [code for code in companies if not window_covered(coverage.get("company", {}).get(code, []), start, end)]
    if (
        missing
        or not window_covered(coverage.get("policy", []), start, end)
        or not window_covered(coverage.get("news", []), start, end)
    ):
        raise ValueError("INFORMATION_COVERAGE_INCOMPLETE")
    selected = []
    for r in records:
        stamp = available_at(r)
        if stamp and stamp <= limit and start <= r["published_date"] <= end:
            if r["entity_type"] == "COMPANY" and r["entity_id"] in companies:
                selected.append(r)
            elif r["category"] in ("NEWS", "INDUSTRY_POLICY"):
                selected.append(r)
    if any(
        r.get("revision_unresolved")
        for r in records
        if r["entity_id"] in companies and r.get("published_date") and start <= r["published_date"] <= end
    ):
        raise ValueError("EVENT_REVISION_UNRESOLVED")
    counts = [len({r["identity"] for r in selected if r["category"] == c}) for c in CATEGORIES[:5]]
    strategy = [
        r
        for r in records
        if r["category"] == "MANAGER_STRATEGY"
        and r["source"]["sha256"] == report_sha256
        and available_at(r)
        and available_at(r) <= limit
    ]
    if len(strategy) != 1 or strategy[0].get("strategy_tag") is None:
        raise ValueError("MANAGER_STRATEGY_UNAVAILABLE")
    tags = ("MEDICINE_FOCUS", "AI_COMPUTE_FOCUS", "OTHER_EXPLICIT_FOCUS", "DIVERSIFIED_OR_MULTI", "UNSPECIFIED")
    if strategy[0]["strategy_tag"] not in tags:
        raise ValueError("STRATEGY_TAG_NOT_REGISTERED")
    return counts + [int(strategy[0]["strategy_tag"] == t) for t in tags]
