"""将有原文依据的事实按公开时点和当时持仓转为固定输入；不读取收益答案。"""

from __future__ import annotations

import bisect
import re
from collections import Counter, defaultdict
from datetime import date

from scripts import fund_002112_semantic_extract_v1 as sem

io, ROOT, core = sem.io, sem.ROOT, sem.core
FEATURE_ROOT = ROOT / "feature-revision-r3"
APPLICATION_TERMS = ("数据中心", "云计算", "通信网络", "5G", "光通信", "智能计算", "人工智能芯片")
WINDOWS = (1, 5, 20)
QUANTITIES = ("PROFIT_YOY", "REVENUE_YOY", "CONTRACT_AMOUNT", "BUYBACK_AMOUNT")
GENERIC = {
    "产品",
    "业务",
    "行业",
    "公司",
    "企业",
    "市场",
    "服务",
    "药品",
    "医疗",
    "医药",
    "生产",
    "研发",
    "数字化",
    "人工智能",
    "基础设施",
    "电子信息",
    "信息技术",
    "医疗服务",
    "药物",
    "技术",
}


def feature_names():
    names = []
    for window in WINDOWS:
        base = [
            "company_weight",
            "fund_count",
            "public_link_weight",
            "public_application_weight",
            "background_count",
            "quarantined_count",
        ]
        base += ["kind_" + v for v in sem.legacy.KINDS]
        base += ["stage_" + v for v in sem.legacy.STAGES]
        base += ["direction_" + v for v in sem.legacy.DIRECTIONS]
        base += ["channel_" + v for v in sem.legacy.CHANNELS]
        base += ["quantified_" + v for v in QUANTITIES]
        base += ["profit_yoy_mean", "profit_yoy_covered_weight", "revenue_yoy_mean", "revenue_yoy_covered_weight"]
        # 未关联持仓的行业资料单列背景，不能冒充某家公司的利润或持仓贡献。
        base += ["background_stage_" + v for v in sem.legacy.STAGES]
        base += ["background_kind_" + v for v in sem.legacy.KINDS]
        base += ["background_direction_" + v for v in sem.legacy.DIRECTIONS]
        names += [f"s{window}_{v}" for v in base]
    return names


def signed_yoy(quantity):
    """只接纳一个百分数或明确区间。币种金额不参与跨公司加总，增长率不是收益率。"""
    text = sem.legacy.quote_text(quantity["value_text"])
    numbers = re.findall(r"[-+−]?\d+(?:\.\d+)?\s*[%％]", text)
    if not 1 <= len(numbers) <= 2:
        return None
    if len(numbers) == 2 and not re.search(r"至|到|~|～|—", text):
        return None
    values = [float(n.replace("%", "").replace("％", "").replace("−", "-")) / 100 for n in numbers]
    sign_text = text if re.search(r"增|降|减|下滑", text) else sem.legacy.quote_text(quantity["quote"])
    if re.search(r"增长|增加|上升|增幅", sign_text) and re.search(r"下降|减少|下滑|降幅", sign_text):
        return None
    if all(v >= 0 for v in values) and re.search(r"下降|减少|下滑|降幅", sign_text):
        values = [-v for v in values]
    # 固定裁剪仅避免极端低基数值主导分裂；仍在证据中保留原文区间。
    return max(-10.0, min(10.0, sum(values) / len(values)))


def company_links(doc, result, reports, profiles):
    """政策只能经当时已知的持仓和公司自述业务关联；保留两端原文，禁止借未来报告。"""
    cutoff = doc["body_available_at"]
    report = core.previous.choose_report(reports, cutoff)
    weights = core.previous.weights(report)
    if doc["kind"] == "company":
        code = doc["stock_code"]
        return (
            [
                {
                    "code": code,
                    "at_event_weight": weights[code],
                    "basis": "ISSUER_ID",
                    "report_hash": report["raw"]["sha256"],
                }
            ]
            if code in weights
            else []
        )
    if doc["kind"] == "fund":
        return []
    links = []
    for code, weight in weights.items():
        matched = None
        for profile in reversed(profiles.get(code, [])):
            if profile["available_at"] > cutoff:
                continue
            if (date.fromisoformat(cutoff[:10]) - date.fromisoformat(profile["available_at"][:10])).days > 730:
                continue
            for product in profile["products"]:
                term = sem.legacy.quote_text(product["term"])
                if term in GENERIC:
                    continue
                for target in result.get("targets", []):
                    target_term = sem.legacy.quote_text(target["term"])
                    target_quote = sem.legacy.quote_text(target["quote"])
                    if target_term in GENERIC:
                        continue
                    own_quote = sem.legacy.quote_text(product["quote"])
                    direct = term == target_term or (
                        len(term) >= 4 and term in target_quote and target_term in own_quote
                    )
                    # 两端均明确提及使用场景才关联。场景相关不代表订单增加或政策利好。
                    application = next((t for t in APPLICATION_TERMS if t in own_quote and t in target_quote), None)
                    application = (
                        application
                        if (
                            re.search(r"公司.*(?:主营|生产|研发|从事)", own_quote)
                            and re.search(r"应用|运用|服务于|面向|客户|领域", own_quote)
                        )
                        else None
                    )
                    # “医保局大数据中心”是机构名，不能因包含“数据中心”就关联光模块厂商。
                    # 场景扩展限定固定的信息通信政策来源；医疗材料仍需直接产品证据。
                    if doc.get("topic") != "TECH":
                        application = None
                    if direct or application:
                        matched = {
                            "code": code,
                            "at_event_weight": weight,
                            "basis": "EXPLICIT_BUSINESS_TEXT" if direct else "APPLICATION_CONTEXT_ONLY",
                            "application_term": application if not direct else None,
                            "report_hash": report["raw"]["sha256"],
                            "product": product,
                            "target": target,
                            "profile_event_id": profile["event_id"],
                            "profile_available_at": profile["available_at"],
                        }
                        break
                if matched:
                    break
            if matched:
                break
        if matched:
            links.append(matched)
    return links


def prepare_events(documents, reports):
    profiles = defaultdict(list)
    seen_facts = set()
    events = []
    for doc in sorted(documents, key=lambda d: (d["body_available_at"], d["event_id"])):
        path = ROOT / "validated-v5" / (doc["event_id"] + ".json")
        if not path.exists():
            path = ROOT / "semantic-results" / (doc["event_id"] + ".json")
        checked = io.read(path)
        assert checked["text_sha256"] == doc["text_sha256"]
        result = sem.validate(checked["raw"], doc) if "raw" in checked else checked
        grounded = result["status"] == "SOURCE_GROUNDED"
        facts = result.get("facts", [])
        fingerprints = [
            io.digest([doc["issuer_codes"], result.get("kind"), result.get("stage"), sem.legacy.quote_text(fact)])
            for fact in facts
        ]
        repeated = bool(fingerprints) and all(key in seen_facts for key in fingerprints)
        if grounded:
            seen_facts.update(fingerprints)
        links = (
            company_links(doc, result, reports, profiles)
            if (grounded or doc["kind"] == "company") and not repeated
            else []
        )
        event = {
            "id": doc["event_id"],
            "document_id": doc["id"],
            "published_date": doc["published_date"],
            "available_at": doc["body_available_at"],
            "source_kind": doc["kind"],
            "title": doc["title"],
            "source_hash": doc["body_sha256"],
            "text_sha256": doc["text_sha256"],
            "repeated_facts": repeated,
            "links": links,
            "source_url": doc.get("source_url", doc.get("url")),
            "selection": doc["selection"],
            "status": result["status"],
            "facts": facts,
            **{k: result.get(k) for k in ("kind", "stage", "direction", "channel", "quantities", "field_warnings")},
        }
        events.append(event)
        if grounded and doc["kind"] == "company" and result.get("products"):
            profiles[doc["stock_code"]].append(
                {"available_at": doc["body_available_at"], "event_id": doc["event_id"], "products": result["products"]}
            )
    return events


def vector(row, events, reports, sessions):
    """每一行只用截至08:00已公开资料；权重取事件时与预测时的较小值。"""
    cutoff = row["as_of"]
    index = sessions.index(cutoff[:10])
    report = core.previous.choose_report(reports, cutoff)
    current = core.previous.weights(report)
    values = dict.fromkeys(feature_names(), 0.0)
    sums, amounts = Counter(), Counter()
    used = []
    for event in events:
        if event["available_at"] > cutoff:
            continue
        effective = bisect.bisect_left(sessions, event["available_at"][:10])
        if effective < len(sessions) and event["available_at"] > sessions[effective] + "T08:00:00+08:00":
            effective += 1
        age = index - effective
        if not 0 <= age < 20 or event["repeated_facts"]:
            continue
        weights = {
            link["code"]: min(link["at_event_weight"], current.get(link["code"], 0.0)) for link in event["links"]
        }
        weight = sum(weights.values())
        application_weight = sum(
            weights[link["code"]] for link in event["links"] if link["basis"] == "APPLICATION_CONTEXT_ONLY"
        )
        source = event["source_kind"]
        if source == "company" and weight <= 0:
            continue
        if source == "fund":
            weight = 1.0
        grounded = event["status"] == "SOURCE_GROUNDED"
        used.append(
            {
                "id": event["id"],
                "available_at": event["available_at"],
                "age_sessions": age,
                "linked_weight": weight,
                "status": event["status"],
                "weights": weights,
                "application_context_weight": application_weight,
                "background_only": source in ("news", "policy") and weight == 0,
            }
        )
        for window in WINDOWS:
            if age >= window:
                continue
            prefix = f"s{window}_"
            if not grounded:
                values[prefix + "quarantined_count"] += 1
                continue
            if weight == 0:
                values[prefix + "background_count"] += 1
                values[prefix + "background_stage_" + event["stage"]] += 1
                values[prefix + "background_kind_" + event["kind"]] += 1
                values[prefix + "background_direction_" + event["direction"]] += 1
                continue
            values[
                prefix
                + (
                    "company_weight"
                    if source == "company"
                    else "fund_count"
                    if source == "fund"
                    else "public_link_weight"
                )
            ] += weight
            values[prefix + "public_application_weight"] += application_weight
            for field in ("kind", "stage", "direction", "channel"):
                if field in ("direction", "channel") and application_weight:
                    # 已提取的行业方向不得冒充这家场景相关公司的确定影响。
                    values[prefix + field + "_" + event[field]] += weight - application_weight
                    values[prefix + field + "_" + ("UNKNOWN" if field == "direction" else "NONE")] += application_weight
                else:
                    values[prefix + field + "_" + event[field]] += weight
            # 同一公告同一指标只记一次，不能把区间上下限当两条消息。
            for metric in QUANTITIES:
                quantity = next((q for q in event["quantities"] if q["metric"] == metric), None)
                if quantity is None:
                    continue
                values[prefix + "quantified_" + metric] += weight
                if source == "company" and metric in ("PROFIT_YOY", "REVENUE_YOY"):
                    number = signed_yoy(quantity)
                    if number is not None:
                        key = prefix + ("profit" if metric == "PROFIT_YOY" else "revenue") + "_yoy"
                        sums[key] += weight * number
                        amounts[key] += weight
    for window in WINDOWS:
        for metric in ("profit", "revenue"):
            key = f"s{window}_{metric}_yoy"
            values[key + "_mean"] = sums[key] / amounts[key] if amounts[key] else None
            values[key + "_covered_weight"] = amounts[key]
    return [values[n] for n in feature_names()], used


def run():
    if (FEATURE_ROOT / "features-completion.json").exists():
        return io.read(FEATURE_ROOT / "features-completion.json")
    assert (ROOT / "semantic-completion.json").exists()
    documents = sem.prepare()
    # 附件修复替换同一来源的旧入口文，晚修订以更晚可用时点入场，不保留双份事件。
    by_id = {d["id"]: d for d in documents}
    for name in ("supplement-documents.jsonl", "supplement-retry-documents.jsonl"):
        for doc in core.previous.read_lines(ROOT / name):
            by_id[doc["id"]] = doc
    documents = list({d["event_id"]: d for d in by_id.values()}.values())
    reports = io.read(ROOT / "reports.json")
    events = prepare_events(documents, reports)
    io.save_lines(FEATURE_ROOT / "events.jsonl", events)
    rows = core.previous.read_lines(ROOT / "numeric-expanded.jsonl")
    sessions = io.read(core.previous.BUNDLE)["sessions"]
    lineage = []
    for row in rows:
        row["F"], used = vector(row, events, reports, sessions)
        lineage.append({"target": row["target"], "as_of": row["as_of"], "events": used})
    io.save_lines(FEATURE_ROOT / "inputs.jsonl", rows)
    io.save_lines(FEATURE_ROOT / "event-lineage.jsonl", lineage)
    io.save(
        FEATURE_ROOT / "feature-protocol.json",
        {
            "at": io.now(),
            "names": feature_names(),
            "code_hash": io.sha(__file__),
            "windows_sessions": WINDOWS,
            "product_matching": "公司业务直接匹配，或两端明确出现相同使用场景；场景关联的公司方向强制未知",
            "application_terms": APPLICATION_TERMS,
            "revision_reason": "原文审查发现产品名称与使用场景不同；尚未进行任何监督拟合，四组方案不变",
            "dedup": "按公开时点排序，同一主体、阶段、类别且全部事实引用已出现则不重复计数",
            "quantities": "有出处的同比百分比取区间均值，固定裁剪[-10,10]；金额只记录有具体金额，不与收益混算",
            "missing": "无可靠同比为None，覆盖权重为0；无事件计数为0；隔离资料另记覆盖数量",
        },
    )
    summary = {
        "at": io.now(),
        "rows": len(rows),
        "semantic_features": len(feature_names()),
        "documents": len(documents),
        "statuses": dict(Counter(e["status"] for e in events)),
        "deduplicated_events": sum(e["repeated_facts"] for e in events),
        "linked_public_events": sum(e["source_kind"] in ("news", "policy") and bool(e["links"]) for e in events),
        "documents_by_year_kind": dict(Counter(d["published_date"][:4] + "_" + d["kind"] for d in documents)),
        "partial_documents": sum(d["selection"]["partial"] for d in documents),
        "days_with_grounded_facts": sum(any(x["status"] == "SOURCE_GROUNDED" for x in r["events"]) for r in lineage),
    }
    io.save(FEATURE_ROOT / "features-completion.json", summary)
    return summary


if __name__ == "__main__":
    print(io.canonical(run()), flush=True)
