"""冻结资料转逐日 N8、36项事件计数和文字引用；该模块从不生成或读取方向标签。"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from scripts import fund_002112_existing_events_sources_v1 as s

# 记录已加载的生成器源码；断点输入只能由同一份源码继续，结束时再核验磁盘没有中途改码。
GENERATOR_CODE = {str(Path(p).resolve()): s.sha(p) for p in (__file__, s.__file__)}
TYPES = ("NEWS", "POLICY", "ANNOUNCEMENT")
KINDS = ("EARNINGS", "BUYBACK", "CONTRACT", "MANAGER_STRATEGY", "POLICY", "INDUSTRY_NEWS")
TIMINGS = ("main", "aux")
N8 = [
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "max_drawdown_60d",
    "relative_position_60d",
    "consecutive_decline_days",
    "nav_lag_sessions",
]
STATS = (
    [f"{t}_count_{w}" for t in TYPES for w in (1, 5, 20)]
    + [f"kind_{t}_count_{w}" for t in KINDS for w in (1, 5, 20)]
    + [f"{t}_{k}_20" for k in ("date_only", "title_only", "related") for t in TYPES]
)
POLICY_RE = re.compile(r"通知|公告|指导意见|决定|办法|规定|规划|征求意见|实施方案|工作方案")
NEWS_RE = re.compile(r"解读|答记者问|发布会|采访|新闻|报道|一图|图解")
UNRELATED_RE = re.compile(r"党建|党史|主题教育|廉政|干部任免|人事任免|干部考察|机关招聘|公务员|巡视|巡察|机关食堂")
KIND_RE = {
    "EARNINGS": r"业绩|财务|利润|年度报告|季度报告|半年度报告|中期报告",
    "BUYBACK": r"回购",
    "CONTRACT": r"合同|中标|协议",
    "MANAGER_STRATEGY": r"基金经理|投资策略|经理变更",
}


def sessions_from(calendars):
    days = set()
    for c in calendars:
        closed = [pair for y in c["years"] for pair in y["closed_ranges"]]
        d, end = date.fromisoformat(c["coverage_start"]), date.fromisoformat(c["coverage_end"])
        while d <= end:
            day = str(d)
            if d.weekday() < 5 and not any(a <= day <= b for a, b in closed):
                days.add(day)
            d += timedelta(days=1)
    return sorted(days)


def effective_session(day, published_at, sessions, timing="main", revision_at=None, force_next=False):
    """15:00是严格边界；日期级主同日/辅次日，周末不会在次一交易日再多延一天。"""
    if not day:
        return None
    date.fromisoformat(day)
    point = datetime.fromisoformat(published_at).astimezone(s.ZONE) if published_at else None
    if revision_at:
        revised = datetime.fromisoformat(revision_at)
        revised = revised.replace(tzinfo=s.ZONE) if revised.tzinfo is None else revised.astimezone(s.ZONE)
        if point is None or revised > point:
            point = revised
    if point:
        day = str(point.date())
        after = point.time() >= time(15)
    else:
        after = timing == "aux"
    after = after or force_next
    i = bisect_right(sessions, day) if after else bisect_left(sessions, day)
    return sessions[i] if i < len(sessions) else None


def normalize_url(url):
    p = urlsplit(url or "")
    return urlunsplit(
        (
            p.scheme.lower().replace("http", "https", 1) if p.scheme == "http" else p.scheme.lower(),
            p.netloc.lower(),
            p.path,
            p.query,
            "",
        )
    )


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    text = re.sub(r"版权所有[^\n]*|本公司董事会及全体董事保证[^。]*。", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classification(title, default, semantic_kind="", *, url="", body="", own_numbers=()):
    """按当前可用原件证据分类；独立文号与发文结构优先于旧语义，文内引用文号不算自身文号。"""
    # 公告来源类别已由独立目录确认，无需对长篇财报重复执行政府公文解析。
    if default == "ANNOUNCEMENT":
        return default, {"rule": "ANNOUNCEMENT_SOURCE", "evidence_rank": 5}
    official = urlsplit(url).netloc.endswith(".gov.cn")
    official_column = bool(re.search(r"nhsa\.gov\.cn/art/.*/art_104_\d+\.html", url))
    # 仅剥离文末日期说明，保留正文标题中的范围、草案、修订等含义。
    heading = re.sub(r"[（(](?:(?:截至|截止)[^）)]*|\d{4}年)[）)]$", "", title).strip()
    formal_end = r"(?:意见|通告|通知|公告|办法|规定|方案|函|批复)"
    ordinary_form = re.match(
        r"^(?:关于|(?:中共中央|国务院|国家|中共|.{1,10}(?:省|市|自治区)|新疆生产建设兵团).{0,100}关于)"
        + r".+"
        + formal_end
        + r"[》]?$",
        heading,
    )
    # “印发《…》的通知”是明确发文标题；仅“印发《…》”仍可能是报道，不机械升为政策。
    issuance_notice = re.match(
        r"^(?:国务院|国家|中共中央|中共|.{1,10}(?:省|市|自治区)|新疆生产建设兵团)"
        r".{0,120}印发《.+》的通知$",
        heading,
    )
    issuer = heading.split("关于", 1)[0]
    reporting_issuer = bool(re.search(r"印发|发布|出台|介绍|解读", issuer)) and not issuance_notice
    formal_title = bool(issuance_notice or (ordinary_form and not reporting_issuer))
    explicit_commentary = bool(NEWS_RE.search(title)) and not formal_title
    # 只读取原件开头的独立文号行；括号里的“依据某文件”不能冒充本文文号。
    paragraphs = [clean_text(line) for line in str(body or "").splitlines() if clean_text(line)]
    own_lines = [
        {"paragraph": i + 1, "number": line}
        for i, line in enumerate(paragraphs[:20])
        if re.fullmatch(r"[\u4e00-\u9fff]{2,30}[〔﹝\[（(]\d{4}[〕﹞\]）)]\d+号", re.sub(r"\s+", "", line))
    ]
    addressed = any(
        re.search(r"(?:局|厅|政府|部门|单位|机构|委员会|党支部|企业)[^。]{0,12}[：:]$", p) for p in paragraphs[:15]
    )
    directive = bool(re.search(r"现.{0,40}(?:通知|提出|答复)如下|现提出以下|请.{0,30}执行|专此函达", clean_text(body)))
    signed = any(
        re.match(r"^(?:国家|国务院|中共中央|中共|.{1,10}(?:省|市|自治区)).{0,60}(?:局|厅|委|政府|党委|办公室)$", p)
        and i + 1 < len(paragraphs)
        and re.fullmatch(r"\d{4}年\d{1,2}月\d{1,2}日", paragraphs[i + 1])
        for i, p in enumerate(paragraphs)
    )
    body_original = official and bool(own_lines or (formal_title and addressed and (directive or signed)))
    reporting = bool(re.search(r"(?:日前|近日|近期).{0,60}(?:印发|发布|出台)", clean_text(body)[:240]))
    evidence = {
        "formal_title": formal_title,
        "standalone_document_numbers": own_lines,
        "addressed_recipients": addressed,
        "operative_document_structure": directive,
        "issuing_authority_and_document_date": signed,
        "reporting_another_document": reporting,
        "official_policy_column": official_column,
        "body_available_for_classification": bool(body),
    }

    def result(kind, rule, rank):
        return kind, {**evidence, "rule": rule, "evidence_rank": rank}

    if default == "ANNOUNCEMENT":
        return result(default, "ANNOUNCEMENT_SOURCE", 5)
    if title.startswith("【"):
        return result("NEWS", "EXPLICIT_MEDIA_REPORT_TITLE", 4)
    if explicit_commentary:
        return result("NEWS", "EXPLICIT_COMMENTARY_TITLE", 4)
    if body_original:
        return result("POLICY", "ORIGINAL_DOCUMENT_NUMBER_OR_OPERATIVE_STRUCTURE", 5)
    if reporting:
        return result("NEWS", "BODY_REPORTS_ANOTHER_DOCUMENT", 4)
    if "INTERPRETATION" in semantic_kind:
        return result("NEWS", "EXISTING_INTERPRETATION_EVIDENCE", 3)
    if semantic_kind == "VIDEO_OR_LIVE_REPORT":
        return result("NEWS", "EXISTING_VIDEO_EVIDENCE", 3)
    if semantic_kind in ("OFFICIAL_POLICY_OR_NOTICE", "DRAFT_OR_PUBLIC_CONSULTATION", "POLICY", "FORMAL_POLICY"):
        return result("POLICY", "EXISTING_FORMAL_SEMANTIC_EVIDENCE", 3)
    if body and semantic_kind == "PUBLIC_NEWS_OR_SERVICE_INFORMATION":
        return result("NEWS", "EXISTING_NEWS_BODY_WITHOUT_ORIGINAL_DOCUMENT_EVIDENCE", 3)
    if official and formal_title and not title.startswith("【"):
        return result("POLICY", "EXPLICIT_GOVERNMENT_DOCUMENT_TITLE_ONLY", 2)
    if official_column or default == "POLICY":
        return result("POLICY", "POLICY_SOURCE_COLUMN_WITHOUT_CONTRADICTING_BODY", 1)
    return result("NEWS", "NEWS_SOURCE_OR_TITLE_FALLBACK", 1)


def classify(title, default, semantic_kind="", *, url="", body="", own_numbers=()):
    return classification(title, default, semantic_kind, url=url, body=body, own_numbers=own_numbers)[0]


def publisher_timestamp(raw, document, semantic):
    """仅提取冻结原件中与已核实标题/日期一致的发布栏时刻；不使用正文事件时间或更新时刻。"""
    day = raw.get("published_date") or raw.get("display_date")
    url = raw.get("url") or raw.get("source_url") or ""
    if (
        not day
        or document.get("title_identity_verified") is not True
        or normalize_url(url) != normalize_url(document.get("url", ""))
        or semantic.get("body_version_updated_at")
        or document.get("body_version_updated_at")
    ):
        return None, None
    clock = (
        r"(?P<year>\d{4})[-年](?P<month>\d{1,2})[-月](?P<day>\d{1,2})日?\s+"
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
    )
    host = urlsplit(url).netloc
    patterns = {
        "tv.cctv.com": r"来源\s*[:：]\s*央视网\s*" + clock,
        "m.news.cctv.com": r"央视新闻客户端\s+" + clock,
        "www.kankanews.com": clock + r"\s+看看新闻Knews",
    }
    if host not in patterns:
        return None, None
    matches = list(re.finditer(patterns[host], document.get("text", "")))
    if len(matches) != 1:
        return None, None
    matched = matches[0]
    values = {k: int(v or 0) for k, v in matched.groupdict().items()}
    point = datetime(**values, tzinfo=s.ZONE)
    if str(point.date()) != str(day)[:10]:
        # 正文时刻与旧日期相冲突时不能任选其一，更不能保留“日期级同日”掩盖冲突。
        raise ValueError("FROZEN_PUBLISHER_TIME_DATE_CONFLICT:" + url)
    return point.isoformat(), {
        "basis": "FROZEN_ARTICLE_PUBLISHER_TIMESTAMP",
        "publisher_host": host,
        "quote": matched.group(0),
        "title_identity_verified": True,
        "date_matches_verified_publication_day": True,
    }


def event_record(raw, reference, default, body="", semantic=None):
    """只接受来源日期，不把旧available_at、抓取日或报告期当发布时间。"""
    semantic = semantic or {}
    title = clean_text(raw.get("title") or raw.get("announcementTitle"))
    day = raw.get("published_date") or raw.get("publishedDate") or raw.get("display_date")
    point = raw.get("source_published_at")
    day = str(day)[:10] if day else None
    precision = "TIMESTAMP" if point else "DATE"
    url = raw.get("source_url") or raw.get("sourceUrl") or raw.get("url") or ""
    entity = str(raw.get("company") or raw.get("stockCode") or raw.get("stock_code") or "").split(".")[0]
    issue = raw.get("old_version_issues") or []
    revision_at = semantic.get("body_version_updated_at") or raw.get("revision_at")
    reason = None
    if not day:
        reason = "NO_PUBLICATION_DATE"
    elif not "2016-01-01" <= day <= "2026-09-29":
        reason = "OUTSIDE_DATE_RANGE"
    elif not title:
        reason = "EMPTY_TITLE"
    elif raw.get("identity_verified") is False or any("IDENTITY" in str(x) for x in issue):
        reason = "EXPLICIT_IDENTITY_CONFLICT"
    elif issue:
        reason = "KNOWN_SOURCE_VERSION_CONFLICT"
    elif UNRELATED_RE.search(title) and default != "ANNOUNCEMENT":
        reason = "FROZEN_NON_MARKET_COLUMN_SCOPE"
    direct = raw.get("kind") == "fund" or any(v in title for v in ("002112", "001412", "德邦鑫星价值"))
    # 已知后续版本的正文及其衍生语义不能参与原始日期的分类。
    body_exclusion = raw.get("existing_text_exclusion")
    usable_body = "" if revision_at or body_exclusion else body
    usable_semantic = {} if revision_at or body_exclusion else semantic
    primary, classification_detail = classification(
        title,
        default,
        usable_semantic.get("kind", ""),
        url=url,
        body=usable_body,
        own_numbers=usable_semantic.get("own_document_numbers", ()),
    )
    body = clean_text(usable_body)
    # 已知较晚版本只用独立目录标题；不把后来正文混入原始日期。
    if revision_at:
        body = ""
    text_level = "TITLE_AND_EXISTING_TEXT" if body else "TITLE_ONLY"
    existing = body[:2000]
    kinds = [k for k, pattern in KIND_RE.items() if re.search(pattern, title)]
    if primary == "POLICY":
        kinds.append("POLICY")
    if primary == "NEWS":
        kinds.append("INDUSTRY_NEWS")
    # 明确的当日结果/盘后复盘即便只有日期，也只能次日进入；不删其后续历史信息。
    force_next = bool(re.search(r"收盘|盘后|复盘", title) or (direct and re.search(r"净值|涨跌|收益率", title)))
    return {
        "document_id": str(raw.get("id") or raw.get("news_id") or s.digest([url, title, day])),
        "event_id": "",
        "source_refs": [reference],
        "primary_type": primary,
        "classification_evidence": {
            "semantic_kind": semantic.get("kind"),
            "source_url": url,
            "default_source_type": default,
            "own_document_numbers": usable_semantic.get("own_document_numbers", []),
            "later_version_evidence_ignored": bool(revision_at),
            "unverified_body_evidence_ignored": bool(body_exclusion),
            **classification_detail,
        },
        "event_kind": kinds,
        "event_stage": "UPDATE" if re.search(r"进展|修订|更正|调整", title) else "DOCUMENT_SIGNAL",
        "published_date": day,
        "published_at": point,
        "date_precision": precision,
        "date_basis": (raw.get("publication_time_evidence") or {}).get("basis", "SOURCE_PUBLICATION_FIELD"),
        "publication_time_evidence": raw.get("publication_time_evidence"),
        "revision_at": revision_at,
        "version_note": "INDEPENDENT_CATALOG_TITLE_ONLY" if revision_at else "HISTORICAL_RECONSTRUCTION",
        "title": title,
        "existing_text": existing,
        "existing_text_exclusion": "LATER_BODY_VERSION" if revision_at else body_exclusion,
        "text_level": text_level,
        "text": title + (" " + existing if existing else ""),
        "full_existing_text_sha256": s.digest(body) if len(body) >= 100 else None,
        "url": normalize_url(url),
        "entity_codes": [entity] if entity else [],
        "relation_scope": "DIRECT_FUND" if direct else "BACKGROUND",
        "relation_basis": "FUND_IDENTITY" if direct else "FIXED_SOURCE_LIBRARY_NO_HISTORICAL_WEIGHT_ASSUMED",
        "old_training_eligible": raw.get("training_eligible"),
        "old_reasons": raw.get("unresolved") or raw.get("gaps") or [],
        "usable_for_count": reason is None,
        "usable_for_text": reason is None,
        "excluded_reason": reason,
        "date_only_assumed_same_day": not point,
        "force_next": force_next,
    }


def normalized_reprint_title(title):
    """只去媒体署名前缀、空白/标点；不删“进展/修订/更正”等版本含义。"""
    title = re.sub(r"^(?:【[^】]+】\s*)+", "", title)
    title = re.sub(r"^(?:央视新闻|新华社(?:客户端)?|人民日报(?:头版)?|环球时报(?:社评)?)[：:|｜]", "", title)
    return re.sub(r"\W", "", title)


def specific_document_key(url):
    """仅识别有稳定文章/视频/公告ID的URL；媒体首页和栏目页不充当文档身份。"""
    p = urlsplit(url)
    if re.search(r"/art/\d{4}/\d{1,2}/\d{1,2}/art_\d+_\d+\.html$", p.path):
        return p.netloc, p.path
    match = re.search(r"/finalpage/\d{4}-\d{2}-\d{2}/(\d+)\.PDF$", p.path, re.IGNORECASE)
    if match and p.netloc == "static.cninfo.com.cn":
        return "CNINFO", match.group(1)
    query = dict(parse_qsl(p.query))
    if p.netloc in ("app.cntv.cn", "w.yangshipin.cn") and (query.get("id") or query.get("vid")):
        return p.netloc, query.get("id") or query["vid"]
    return None


def deduplicate(records):
    """同一来源时点只选一次，后续转载不能补写早期正文；新进展标题保留自己的日期。"""
    # 同文同日已有明确发布时刻时，日期级目录副本不能借同日假设提前进入特征。
    clocks = defaultdict(list)
    for event in records:
        if event["published_at"] and event["url"]:
            key = (event["url"], normalized_reprint_title(event["title"]), event["published_date"])
            clocks[key].append(event)
    for event in records:
        key = (event["url"], normalized_reprint_title(event["title"]), event["published_date"])
        if not event["published_at"] and clocks.get(key):
            known = min(clocks[key], key=lambda item: item["published_at"])
            event["published_at"] = known["published_at"]
            event["date_precision"] = "TIMESTAMP"
            event["date_only_assumed_same_day"] = False
            event["date_basis"] = "SAME_DOCUMENT_VERIFIED_PUBLISHER_TIMESTAMP"
            event["publication_time_evidence"] = known.get("publication_time_evidence")
    # 同日同一原文已有语义/正文证据时，目录的标题回退不能反过来覆盖其分类。
    # 只在相同真实发布日期内传播，未来版本或转载绝不改变早期输入。
    strongest = {}
    for event in records:
        key = (event["url"], normalized_reprint_title(event["title"]), event["published_date"], event["published_at"])
        if event["url"]:
            previous = strongest.get(key)
            rank = event.get("classification_evidence", {}).get("evidence_rank", 0)
            if previous is None or rank > previous.get("classification_evidence", {}).get("evidence_rank", 0):
                strongest[key] = event
    for event in records:
        key = (event["url"], normalized_reprint_title(event["title"]), event["published_date"], event["published_at"])
        known = strongest.get(key)
        if (
            known is not None
            and known is not event
            and (
                known.get("classification_evidence", {}).get("evidence_rank", 0)
                > event.get("classification_evidence", {}).get("evidence_rank", 0)
            )
        ):
            event["primary_type"] = known["primary_type"]
            event["event_kind"] = [k for k in event["event_kind"] if k not in ("POLICY", "INDUSTRY_NEWS")]
            if event["primary_type"] in ("POLICY", "NEWS"):
                event["event_kind"].append("POLICY" if event["primary_type"] == "POLICY" else "INDUSTRY_NEWS")
            event["classification_evidence"] = {
                **known["classification_evidence"],
                "basis": "SAME_DOCUMENT_SAME_PUBLICATION_TIME_STRONGER_SOURCE_EVIDENCE",
            }
    ordered = sorted(
        records,
        key=lambda e: (
            e["published_date"] or "9999",
            e["published_at"] or "",
            {"POLICY": 0, "ANNOUNCEMENT": 1, "NEWS": 2}[e["primary_type"]],
            e["text_level"] == "TITLE_ONLY",
            e["document_id"],
        ),
    )
    seen, accepted, decisions = {}, [], []
    for e in ordered:
        if not e["usable_for_count"]:
            decisions.append(e)
            continue
        keys = [("identity", re.sub(r"\W", "", e["title"]), tuple(e["entity_codes"]), e["published_date"])]
        if e["url"]:
            # 保留既有“同URL且完整标题完全相同”的明确文档重复规则；首页不同标题绝不按URL合并。
            keys.append(("url_and_exact_title", e["url"], e["title"]))
        document_key = specific_document_key(e["url"])
        if document_key:
            keys.append(("specific_document", document_key, normalized_reprint_title(e["title"]), e["revision_at"]))
        if e.get("full_existing_text_sha256"):
            keys.append(("full_text", e["full_existing_text_sha256"], tuple(e["entity_codes"]), e["revision_at"]))
        # 审计必须在扩展别名之前记录真正命中的键；新注册键不是本次合并依据。
        matched_keys = [k for k in keys if k in seen]
        previous = seen[matched_keys[0]] if matched_keys else None
        if previous is not None:
            # 只增加原件引用；日期、文字、类型均不被后来转载改写。
            accepted[previous]["source_refs"].extend(e["source_refs"])
            # 当前已可见转载的具体URL/标题成为后续别名；不会补写过去的文字或发布日期。
            for key in keys:
                seen.setdefault(key, previous)
            decisions.append(
                {
                    "document_id": e["document_id"],
                    "excluded_reason": "EXACT_DOCUMENT_DUPLICATE",
                    "canonical_event_id": accepted[previous]["event_id"],
                    "source_refs": e["source_refs"],
                    "published_date": e["published_date"],
                    "title": e["title"],
                    "url": e["url"],
                    "matched_keys": matched_keys,
                }
            )
            continue
        e["event_id"] = s.digest(keys)
        for key in keys:
            seen[key] = len(accepted)
        accepted.append(e)
    return accepted, decisions


def adapt_sources(root=s.ROOT):
    frozen = s.Frozen(root)
    records = []
    for name in ("historical", "public"):
        for i, raw in enumerate(frozen.entry(name)["rows"]):
            body, semantic = "", {}
            if name == "historical":
                doc = frozen.get(raw["source_record"]["path"]) if raw.get("source_record") else {}
                body = "\n".join(doc.get("pages", []))
                # 原件的版本警报仍保留，日期推导字段不作为真实时刻。
                raw = {**raw, "old_version_issues": raw.get("old_version_issues") or doc.get("revision_issues") or []}
            else:
                doc = frozen.get(raw["source"]["path"]) if raw.get("source") else {}
                semantic = frozen.get(raw["semantic_source"]["path"]) if raw.get("semantic_source") else {}
                body = doc.get("text", "")
                # 视频导航、模板及身份未对上的文字不是核实的正文；仍保留独立目录的标题信号。
                if body and (doc.get("body_verified") is not True or doc.get("title_identity_verified") is not True):
                    raw = {**raw, "existing_text_exclusion": "SOURCE_BODY_OR_IDENTITY_UNVERIFIED"}
                point, time_evidence = publisher_timestamp(raw, doc, semantic)
                if point:
                    raw = {**raw, "source_published_at": point, "publication_time_evidence": time_evidence}
            reference = {
                "entry": name,
                "row": i,
                "source": frozen.inventory["entries"][name],
                "raw_refs": {
                    k: raw[k]
                    for k in ("original", "source_record", "source", "literal_evidence", "semantic_source")
                    if k in raw
                },
            }
            records.append(
                event_record(raw, reference, "ANNOUNCEMENT" if name == "historical" else "NEWS", body, semantic)
            )
    for i, column in enumerate(frozen.entry("catalogs")["columns"]):
        for j, raw in enumerate(column["all_entries"]):
            source_type = "POLICY" if str(column["column"]) == "104" else "NEWS"
            records.append(event_record(raw, {"entry": "catalogs", "column": i, "row": j}, source_type))
    for i, raw in enumerate(frozen.entry("news")["all_entries"]):
        records.append(event_record(raw, {"entry": "news", "row": i}, "NEWS"))
    material = s.payload(frozen.entry("materials"))
    if material["fundCode"] != "002112":
        raise ValueError("FUND_IDENTITY_MISMATCH")
    for i, raw in enumerate(material["documents"]):
        records.append(event_record(raw, {"entry": "materials", "row": i}, "ANNOUNCEMENT"))
    db = s.read(root / "snapshot/database.json")
    if db["status"] == "COMPLETE":
        for i, raw in enumerate(db["tables"].get("news_item", [])):
            point = datetime.fromisoformat(raw["published_at"]).astimezone(s.ZONE)
            # 业务字段未提供来源精度证明，午夜值不得冒充已核实的时分秒；统一日期级保留不确定性。
            row = {**raw, "published_date": str(point.date()), "source_published_at": None}
            records.append(event_record(row, {"entry": "database/news_item", "row": i}, "NEWS", raw.get("summary")))
        for i, raw in enumerate(db["tables"].get("news_research_card", [])):
            record = raw.get("record_payload") or {}
            # 只取已有来源标题/URL；analysis正文含事后解释，不进入文字模型。
            origin = record.get("analysis", {})
            row = {
                "title": origin.get("title"),
                "url": origin.get("url"),
                "id": raw["announcement_id"],
                "published_date": raw["published_date"],
                "stock_code": raw["stock_code"],
                "training_eligible": False,
            }
            records.append(event_record(row, {"entry": "database/news_research_card", "row": i}, "ANNOUNCEMENT"))
    events, decisions = deduplicate(records)
    sessions = sessions_from([frozen.entry(k) for k in s.CALENDARS])
    for e in events:
        for timing in TIMINGS:
            e[f"effective_session_{timing}"] = effective_session(
                e["published_date"], e["published_at"], sessions, timing, force_next=e["force_next"]
            )
    s.save_lines(root / "events.jsonl", events)
    s.save_lines(root / "source-decisions.jsonl", decisions)
    summary = {}
    for kind in TYPES:
        es = [e for e in events if e["primary_type"] == kind]
        summary[kind] = {
            "documents": len(es),
            "start": min((e["published_date"] for e in es), default=None),
            "end": max((e["published_date"] for e in es), default=None),
            "title_only": sum(e["text_level"] == "TITLE_ONLY" for e in es),
            "date_only": sum(e["date_precision"] == "DATE" for e in es),
            "yearly": {
                y: {
                    "documents": sum(e["published_date"].startswith(y) for e in es),
                    "publication_days": len({e["published_date"] for e in es if e["published_date"].startswith(y)}),
                }
                for y in map(str, range(2016, 2027))
            },
        }
    s.save(
        root / "event-summary.json",
        {
            "types": summary,
            "raw_records": len(records),
            "excluded": dict(Counter(e["excluded_reason"] for e in decisions)),
            "database_status": db["status"],
        },
    )
    return events, sessions


def nav_features(values):
    """原N7公式原样迁移：20日总体波动、60日窗口回撤与位置，不进行收益年化。"""
    if len(values) != 61:
        raise ValueError("HISTORY_TOO_SHORT")
    v = list(map(float, values))
    if any(not math.isfinite(x) or x <= 0 for x in v):
        raise ValueError("INVALID_NAV")
    low, high = min(v[1:]), max(v[1:])
    if low == high:
        raise ValueError("FLAT_FEATURE_WINDOW")
    returns = [v[i] / v[i - 1] - 1 for i in range(41, 61)]
    avg = sum(returns) / 20
    peak, drawdown = v[1], 0.0
    for point in v[1:]:
        peak = max(peak, point)
        drawdown = min(drawdown, point / peak - 1)
    decline = 0
    for i in range(60, 0, -1):
        if v[i] >= v[i - 1]:
            break
        decline += 1
    return [
        v[60] / v[55] - 1,
        v[60] / v[40] - 1,
        v[60] / v[0] - 1,
        math.sqrt(sum((r - avg) ** 2 for r in returns) / 20),
        drawdown,
        (v[60] - low) / (high - low),
        float(decline),
    ]


def nav_facts(frozen):
    facts = s.payload(frozen.entry("historical_facts"))["funds"]["002112"]
    if facts["family"] != "FUND_001412" or facts["nav"]["fund_code"] != "002112":
        raise ValueError("NAV_SHARE_MISMATCH")
    nav = {}
    for r in facts["nav"]["rows"]:
        ann = r.get("ann_date") or str(date.fromisoformat(r["date"]) + timedelta(days=1))
        # 延续已有净值公布日的保守次日08:00约定，不能借事件主口径提前净值。
        available = datetime.combine(date.fromisoformat(ann) + timedelta(days=1), time(8), s.ZONE).isoformat()
        nav[r["date"]] = {
            "unit_nav": r["nav"],
            "available_at": available,
            "ann_date": ann,
            "source_ref": "historical_facts/funds/002112/nav",
            "source_hash": r.get("source_hash"),
        }
    for d, r in frozen.entry("recent_nav").items():
        if d in nav and Decimal(nav[d]["unit_nav"]) != Decimal(r["unit_nav"]):
            raise ValueError(f"NAV_SOURCE_VALUE_CONFLICT:{d}")
        nav[d] = {**r, "source_ref": "recent_nav"}
    reports = list(facts["reports"])
    for r in s.payload(frozen.entry("materials"))["reports"]:
        reports.append(
            {
                "report_end": r["endDate"],
                "available_at": str(date.fromisoformat(r["publishedDate"]) + timedelta(days=1)) + "T08:00:00+08:00",
                "holdings": [{"stock_code": h["stockCode"]} for h in r["holdings"]],
            }
        )
    return nav, reports


def select_holdings(reports, target):
    available = [r for r in reports if r["available_at"] < target + "T15:00:00+08:00"]
    if not available:
        return set(), None
    report = max(available, key=lambda r: (r["report_end"], r["available_at"]))
    return {h["stock_code"].split(".")[0] for h in report["holdings"]}, report["report_end"]


def counts_and_members(events, memberships, held):
    counts = dict.fromkeys(STATS, 0)
    for index, age in memberships:
        if not 0 <= age < 20:
            continue
        event = events[index]
        kind = event["primary_type"]
        for w in (1, 5, 20):
            if age < w:
                counts[f"{kind}_count_{w}"] += 1
                for k in event["event_kind"]:
                    counts[f"kind_{k}_count_{w}"] += 1
        counts[f"{kind}_date_only_20"] += event["date_precision"] == "DATE"
        counts[f"{kind}_title_only_20"] += event["text_level"] == "TITLE_ONLY"
        counts[f"{kind}_related_20"] += event["relation_scope"] == "DIRECT_FUND" or bool(
            set(event["entity_codes"]) & held
        )
    return [math.log1p(counts[k]) for k in STATS]


def build_inputs(root=s.ROOT):
    if any(s.sha(p) != expected for p, expected in GENERATOR_CODE.items()):
        raise ValueError("INPUT_GENERATOR_CODE_CHANGED_DURING_PROCESS")
    generator_sha = s.digest(GENERATOR_CODE)
    if (root / "input-manifest.json").exists():
        existing = s.read(root / "input-manifest.json")
        if existing.get("generator_code_sha256") != generator_sha:
            raise ValueError("INPUT_GENERATOR_CODE_CHANGED_NEW_IMMUTABLE_REVISION_REQUIRED")
        if any(s.sha(root / p) != h for p, h in existing["artifacts"].items()):
            raise ValueError("INPUT_ARTIFACT_CHANGED")
        return existing
    # 即使中断发生在输入清单生成前，也必须绑定相同生成器和来源清单才能恢复。
    s.save(
        root / "input-generator.json",
        {
            "files": GENERATOR_CODE,
            "source_inventory_sha256": s.sha(root / "source-inventory.json"),
        },
    )
    events, sessions = adapt_sources(root)
    frozen = s.Frozen(root)
    nav, reports = nav_facts(frozen)
    s.save(root / "snapshot/nav-facts.json", nav)
    s.save(root / "sessions.json", sessions)
    buckets = {timing: defaultdict(list) for timing in TIMINGS}
    for index, e in enumerate(events):
        for timing in TIMINGS:
            day = e[f"effective_session_{timing}"]
            if day in sessions:
                start = sessions.index(day)
                for age in range(20):
                    if start + age < len(sessions):
                        buckets[timing][sessions[start + age]].append([index, age])
    rows, members, excluded = [], [], []
    for i, target in enumerate(sessions):
        if not "2016-01-01" <= target <= "2026-09-29":
            continue
        base = sessions[i - 1]
        if target not in nav or base not in nav:
            excluded.append({"target": target, "reason": "TARGET_OR_BASE_NAV_ABSENT"})
            continue
        n8 = None
        for lag in range(21):
            end = i - 1 - lag
            days = sessions[end - 60 : end + 1] if end >= 60 else []
            if len(days) != 61 or any(
                d not in nav or nav[d]["available_at"] >= target + "T15:00:00+08:00" for d in days
            ):
                continue
            try:
                n8 = nav_features([nav[d]["unit_nav"] for d in days]) + [float(lag)]
                break
            except ValueError:
                continue
        if n8 is None:
            excluded.append({"target": target, "reason": "NO_CONTIGUOUS_61_NAV_WINDOW_WITHIN_20_SESSIONS"})
            continue
        held, report_end = select_holdings(reports, target)
        row = {
            "target": target,
            "base": base,
            "session_index": i,
            "n8": n8,
            "input_nav_end": sessions[end],
            "label_mature_at": max(nav[target]["available_at"], nav[base]["available_at"]),
            "historical_holding_report_end": report_end,
        }
        member = {"target": target}
        for timing in TIMINGS:
            member[timing] = buckets[timing][target]
            row[timing] = counts_and_members(events, member[timing], held)
        rows.append(row)
        members.append(member)
    s.save_lines(root / "daily-inputs.jsonl", rows)
    s.save_lines(root / "event-membership.jsonl", members)
    s.save_lines(root / "excluded-dates.jsonl", excluded)
    s.save(
        root / "feature-spec.json",
        {
            "n8": N8,
            "statistics": STATS,
            "counts_transform": "log1p",
            "text": {
                "types": TYPES,
                "analyzer": "char",
                "ngram_range": [2, 4],
                "max_features_each": 1024,
                "min_df": 1,
                "max_df": 1.0,
                "norm": "l2",
                "weight": "0.9**age",
                "window": 20,
            },
            "zero_means": "冻结库该窗口没有已收集记录；不等于现实没有事件",
        },
    )
    evidence = {}
    for timing in TIMINGS:
        evidence[timing] = {}
        for k in TYPES:
            used = {i for row in members for i, _ in row[timing] if events[i]["primary_type"] == k}
            count_col = STATS.index(f"{k}_count_20")
            evidence[timing][k] = {
                "training_document_candidates": len(used),
                "dates_with_information": sum(row[timing][count_col] > 0 for row in rows),
                "varying_statistic": len({row[timing][count_col] for row in rows}) > 1,
                "text_characters": sum(len(events[i]["text"]) for i in used),
                "sample_event_ids": [events[i]["event_id"] for i in sorted(used)[:5]],
            }
    if any(s.sha(p) != expected for p, expected in GENERATOR_CODE.items()):
        raise ValueError("INPUT_GENERATOR_CODE_CHANGED_DURING_PROCESS")
    manifest = {
        "generator_code_sha256": generator_sha,
        "rows": len(rows),
        "start": rows[0]["target"],
        "end": rows[-1]["target"],
        "events": len(events),
        "excluded_dates": len(excluded),
        "evidence": evidence,
        "artifacts": {
            n: s.sha(root / n)
            for n in [
                "input-generator.json",
                "events.jsonl",
                "daily-inputs.jsonl",
                "event-membership.jsonl",
                "feature-spec.json",
                "sessions.json",
                "snapshot/nav-facts.json",
            ]
        },
    }
    s.save(root / "input-manifest.json", manifest)
    return manifest
