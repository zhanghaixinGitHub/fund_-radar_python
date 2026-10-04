"""晚间一日模型的信息改进：利润字段修复、持仓业务关联、事件新增量与反应窗口。

复用上一轮冻结的价格、净值、标签及模型；不改现用服务，不读新增答案。
只从已保存原文及核对过的业务引文补充信息。关联不等于收益因果，未知不变成利好。
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import joblib
import numpy as np

from scripts import fund_002112_evening_fusion_v2 as old

ROOT = old.RESEARCH / "evening-event-refinement/20261003-v2"
GROUPS = ("A_FIX", "B_LINK", "C_EVENT")
LABELS = {"A_FIX": "仅修正数字输入", "B_LINK": "修正＋业务关联", "C_EVENT": "关联＋新增内容与价格反应"}
LIMIT = 80
# 固定语义词用于辨认业务范围，不能由历史涨跌筛词。行业相关只进入方向未知的背景。
TOPICS = {
    "PHARMA": ("医药", "药品", "药物", "制药", "创新药", "中药", "化学药", "生物药", "仿制药", "疫苗", "血液制品"),
    "MED_DEVICE": ("医疗器械", "医用耗材", "诊断试剂", "医疗设备"),
    "MED_SERVICE": ("医疗服务", "医院", "护理服务", "体检"),
    "OPTICAL": ("光模块", "光器件", "光通信", "光芯片", "光互连"),
    "COMPUTE": ("算力", "人工智能", "数据中心", "服务器"),
    "SEMICONDUCTOR": ("集成电路", "半导体", "电子芯片", "芯片制造"),
    "CIRCUIT_BOARD": ("PCB", "印制电路板"),
}
GENERIC = {
    "产品",
    "服务",
    "企业",
    "技术",
    "业务",
    "公司",
    "市场",
    "行业",
    "药品",
    "药物",
    "医药",
    "医疗",
    "医疗服务",
    "人工智能",
    "芯片",
    "算力",
    "数据中心",
    "基础设施",
    "医疗器械",
}
EXTRA_NAMES = [
    "public_new_fact_weight",
    "public_repeated_fact_weight",
    "public_direct_weight",
    "public_context_weight",
    "public_observed_reaction",
    "public_reaction_coverage",
    "public_no_complete_session_weight",
    "public_source_time_unknown_weight",
    "company_observed_reaction",
    "company_reaction_coverage",
    "company_no_complete_session_weight",
    "company_newest_age",
    "guidance_revision_delta",
    "guidance_revision_coverage",
    "buyback_execution_delta",
    "buyback_delta_coverage",
] + ["context_exposure_" + topic for topic in TOPICS]


def clean(value) -> str:
    """仅清除空白，不用相似文本代替原文引文。"""
    return re.sub(r"\s+", "", value or "")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def topic_terms(text: str) -> dict:
    return {
        name: next(term for term in terms if term in text)
        for name, terms in TOPICS.items()
        if any(term in text for term in terms)
    }


def check_old() -> None:
    old.verify_freeze(old.ROOT)
    # 每个复用模型仍需在训练和验证阶段核对原拟合回执；这里保护上轮所有产物。
    assert old.read(old.ROOT / "independent-verification.json")["passed"]


def start(root: Path) -> None:
    if root.exists():
        raise ValueError("NEW_RUN_DIRECTORY_REQUIRED")
    check_old()
    root.mkdir(parents=True)
    old.write(root / "protection-before.json", old.git_snapshot())
    protected = {str(p): old.sha(p) for p in old.ROOT.rglob("*") if p.is_file()}
    old.write(root / "old-artifacts.json", {"files": protected})
    old.write(
        root / "intent.json",
        {
            "at": old.now(),
            "goal": "修正利润字段及回购累计口径，并检验新闻业务关联和事件表达的增量价值",
            "weights": old.WEIGHTS,
            "unchanged": ["665个信息样本", "市场/净值输入", "标签", "截点", "每20日更新", "模型配方"],
            "candidate_count": 3,
            "maximum_new_fits": LIMIT,
            "planned_fits": 71,
            "selection": "只用2025在B/C中按正确数、Brier、少特征选一个；2026仅诊断，不能反选",
            "new_paid_requests": 0,
            "production_adoption": False,
            "data_scope": "复用已冻结历史及已有正文/业务引文，不新增历史答案、不重跑旧模型",
        },
    )


def diagnose(root: Path, source: dict, rows: list) -> None:
    """对全部424天用同一规则诊断；52个改错日与50个改对日并列，不挑有利样本。"""
    predictions = old.lines(old.ROOT / "historical-predictions.jsonl")
    lineage = {r["target"]: r for r in old.lines(old.ROOT / "daily-lineage.jsonl")}
    by_id = {e["id"]: e for e in source["company_events"]}
    by_id.update({e["id"]: e for e in source["public_events"] if e["source_kind"] in ("news", "policy")})
    output = []
    for row in predictions:
        new = old.CLASSES[int(np.argmax(row["weighted_55"]))]
        prior = old.CLASSES[int(np.argmax(row["price_history"]))]
        category = "UNCHANGED" if new == prior else "CORRECTED" if new == row["label"] else "SPOILED"
        evidence = []
        for e in lineage[row["target"]]["information"]["events"]:
            original = by_id[e["id"]]
            evidence.append(
                {
                    **e,
                    "title": original["title"],
                    "url": original.get("source_url"),
                    "quotes": original.get("facts", original.get("quotes", [])),
                    "publication": original.get("published_date", original.get("published_at")),
                }
            )
        output.append(
            {
                "target": row["target"],
                "as_of": row["as_of"],
                "category": category,
                "actual": row["label"],
                "prior": prior,
                "information55": new,
                "events": evidence,
            }
        )
    stats = {}
    for category in ("SPOILED", "CORRECTED", "UNCHANGED"):
        subset = [r for r in output if r["category"] == category]
        events = [e for r in subset for e in r["events"] if e["kind"] in ("news", "policy")]
        stats[category] = {
            "days": len(subset),
            "public_links": len(events),
            "no_direct_holding_link": sum(e["weight"] == 0 for e in events),
        }
    original_rows = [r for r in rows if "information" in r["groups"]]
    old.write_lines(root / "all-day-diagnosis.jsonl", output)
    old.write(
        root / "diagnosis.json",
        {
            "groups": stats,
            "profit_source_events": sum(
                "profit_yoy" in e["numeric"] and e["status"] == "QUALIFIED" for e in source["company_events"]
            ),
            "old_profit_nonmissing_rows": sum(r["groups"]["information"][3] is not None for r in original_rows),
            "confirmed_problem": "来源profit_yoy被误读为parent_profit_yoy；原列全缺失；回购须有累计金额口径",
            "causal_limit": "所有改错和改对日均缺直接政策关联，不能据此将错误归因于某条新闻",
        },
    )


def supplement(root: Path, source: dict) -> dict:
    """逐字核对既有正文、产品引文、政策对象；只增加通过原文核对的业务证据。"""
    manifest, docs = {}, {}
    for name in ["extraction-documents.jsonl", "supplement-documents.jsonl", "supplement-retry-documents.jsonl"]:
        path = old.SEM / name
        if not path.exists():
            continue
        manifest[str(path)] = old.sha(path)
        for doc in old.lines(path):
            if doc.get("event_id"):
                docs[doc["event_id"]] = doc
    profiles, public, excluded = [], [], []
    for event in source["public_events"]:
        if event["status"] != "SOURCE_GROUNDED":
            continue
        doc = docs.get(event["id"])
        path = old.SEM / "validated-v5" / (event["id"] + ".json")
        if not path.exists():
            path = old.SEM / "semantic-results" / path.name
        if not doc or not path.exists():
            excluded.append({"id": event["id"], "reason": "NO_MATCHING_ORIGINAL_OR_VALIDATED_EXTRACTION"})
            continue
        result = old.read(path)
        manifest[str(path)] = old.sha(path)
        if (
            result.get("status") != "SOURCE_GROUNDED"
            or result["text_sha256"] != event["text_sha256"]
            or doc.get("text_sha256") != event["text_sha256"]
            or doc.get("body_exclusions")
        ):
            excluded.append({"id": event["id"], "reason": "VERSION_OR_EXTRACTION_MISMATCH"})
            continue
        text = clean(doc.get("text") or doc.get("body"))
        if any(clean(q) not in text for q in event["facts"]):
            excluded.append({"id": event["id"], "reason": "FACT_QUOTE_NOT_IN_ORIGINAL"})
            continue
        if event["source_kind"] == "company":
            for product in result.get("products", []):
                if not clean(product.get("quote")) or clean(product["quote"]) not in text:
                    continue
                # 工商经营范围可包含尚未经营的业务；本轮只保留实际业务描述作关联依据。
                if "经营范围" in clean(product["quote"]):
                    continue
                profiles.append(
                    {
                        "code": doc["stock_code"],
                        "available_at": event["available_at"],
                        "document_id": event["id"],
                        "term": clean(product["term"]),
                        "quote": product["quote"],
                        "source_hash": event["source_hash"],
                        "topics": topic_terms(clean(product["quote"])),
                    }
                )
        elif event["source_kind"] in ("news", "policy"):
            targets = [t for t in result.get("targets", []) if clean(t.get("quote")) and clean(t["quote"]) in text]
            impact = result.get("impact_quote") or ""
            direction = event["direction"] if impact and clean(impact) in text else "UNKNOWN"
            public.append(
                {
                    **event,
                    "targets": targets,
                    "direction": direction,
                    "impact_quote": impact,
                    "body_time_precision": "DATE_ONLY_OR_CONSERVATIVE_BOUND",
                    "facts_hash": digest(event["facts"]),
                }
            )
    mark_novelty(public)
    enriched = {"profiles": profiles, "public": public, "exclusions": excluded}
    old.write(root / "supplement-source-manifest.json", {"files": manifest})
    old.write(root / "verified-business-material.json", enriched)
    return enriched


def mark_novelty(public: list) -> None:
    """同句原文跨文重现只计一次；同一可得时刻以固定ID排序，不使用后续材料。"""
    seen = {}
    for event in sorted(public, key=lambda e: (e["available_at"], e["id"])):
        facts = list(dict.fromkeys(clean(f) for f in event["facts"] if clean(f)))
        old_ids = sorted({seen[f]["id"] for f in facts if f in seen})
        event["new_fraction"] = sum(f not in seen for f in facts) / len(facts) if facts else 0.0
        event["repeated_source_ids"] = old_ids
        for fact in facts:
            seen.setdefault(fact, {"id": event["id"], "available_at": event["available_at"]})


def company_rows(base: str, cutoff: str, source: dict, max_age=5) -> list:
    weights = old.holding_weights(old.choose_report(source["reports"], cutoff))
    index = source["sessions"].index(base)
    result = []
    for event in source["company_events"]:
        stamp = event["version_available_at"]
        if event["status"] != "QUALIFIED" or not stamp or stamp > cutoff or not event["links"]:
            continue
        age = index - event["evening_index"]
        weight = min(event["links"][0]["weight"], weights.get(event["code"], 0))
        if 0 <= age < max_age and weight > 0:
            result.append((event, weight, age))
    return sorted(result, key=lambda x: (x[0]["version_available_at"], x[0]["id"]))


def fixed_information(row: dict, source: dict) -> tuple[list, list]:
    values = list(row["groups"]["information"])
    latest = {}
    for event, weight, _ in company_rows(row["base"], row["as_of"], source):
        number = event["numeric"].get("profit_yoy")
        if number:
            if number.get("period") != event["period"] or not number.get("quote"):
                raise ValueError("PROFIT_PERIOD_OR_QUOTE_MISMATCH")
            latest[event["code"]] = (number["value"], weight, event["id"])
    coverage = sum(w for _, w, _ in latest.values())
    values[3] = sum(v * w for v, w, _ in latest.values()) / coverage if coverage else None
    # 原来没有可处理原文时保持None，不把没有信息变成无变化。
    if values[16] is not None or coverage:
        values[16] = coverage
    buyback_latest = {}
    for event, weight, _ in company_rows(row["base"], row["as_of"], source):
        if "buyback_ratio" in event["numeric"] and cumulative_buyback(event.get("buyback")):
            buyback_latest[event["code"]] = (event["numeric"]["buyback_ratio"]["value"], weight)
    buyback_cover = sum(w for _, w in buyback_latest.values())
    values[6] = sum(v * w for v, w in buyback_latest.values()) / buyback_cover if buyback_cover else None
    if values[18] is not None or buyback_cover:
        values[18] = buyback_cover
    return values, [identifier for _, _, identifier in latest.values()]


def relate(event: dict, report: dict | None, profiles: dict, cutoff: str) -> list:
    """发行人直指、具体产品和宽行业背景分层；后两者不自动继承公司利好方向。"""
    if not report:
        return []
    event_text = clean(event["title"] + "".join(event["facts"]))
    target_topics = {
        topic: (term, t["quote"]) for t in event["targets"] for topic, term in topic_terms(clean(t["term"])).items()
    }
    links = []
    # 使用事件当时可见且预测时仍然适用的业务资料，不借未来业务披露补历史关联。
    time_limit = min(cutoff, event["available_at"])
    for holding in report["holdings"]:
        code, name = holding["stock_code"], clean(holding["stock_name"])
        weight = float(holding["nav_weight_pct"]) / 100
        if weight <= 0:
            continue
        issuer = len(name) >= 3 and name in event_text
        link = {
            "code": code,
            "weight": weight,
            "report_hash": report["raw"]["sha256"],
            "report_available_at": report["available_at"],
            "basis": "ISSUER" if issuer else None,
            "topics": [],
            "profile": None,
            "source_quote": name if issuer else None,
            "direction": event["direction"] if issuer else "UNKNOWN",
        }
        for profile in reversed(profiles.get(code, [])):
            if profile["available_at"] > time_limit:
                continue
            if (date.fromisoformat(time_limit[:10]) - date.fromisoformat(profile["available_at"][:10])).days > 730:
                continue
            specific = profile["term"] not in GENERIC and len(profile["term"]) >= 4
            product_target = next(
                (
                    t
                    for t in event["targets"]
                    if specific and (profile["term"] == clean(t["term"]) or profile["term"] in clean(t["term"]))
                ),
                None,
            )
            common = sorted(set(profile["topics"]) & set(target_topics))
            if link["basis"] is None and (product_target or common):
                link.update(
                    {
                        "basis": "PRODUCT" if product_target else "INDUSTRY_CONTEXT",
                        "profile": profile,
                        "source_quote": product_target["quote"] if product_target else target_topics[common[0]][1],
                        "topics": common,
                        # 产品相符仍未证实供货、订单和利润结果，业务方向保持未知。
                        "direction": "UNKNOWN",
                    }
                )
            if link["basis"]:
                break
        if link["basis"]:
            links.append(link)
    return links


def observed_reaction(code: str, stamp: str, base: str, cutoff: str, source: dict):
    """只测首个完整交易日之后的价格反应，不把相伴涨跌认定为事件造成。"""
    sessions = source["sessions"]
    end = sessions.index(base)
    start = bisect.bisect_left(sessions, stamp[:10])
    # 日线无法拆分盘中消息前后涨跌；开盘后可得的消息从下个完整交易日才观察。
    if start < len(sessions) and stamp > sessions[start] + "T09:30:00+08:00":
        start += 1
    days = sessions[start : end + 1]
    if not days:
        return None, {"status": "NO_COMPLETE_SESSION_YET", "price_days": []}
    quotes = [source["stocks"].get(day, {}).get(code) for day in days]
    if any(not old.quote_ok(q, d, cutoff) for q, d in zip(quotes, days, strict=True)):
        return None, {"status": "QUOTE_UNAVAILABLE", "price_days": days}
    value = math.prod(1 + float(q["pct_chg"]) / 100 for q in quotes) - 1
    return value, {"status": "OBSERVED_NOT_CAUSAL", "price_days": days}


def revised_features(row: dict, source: dict, material: dict, profiles: dict) -> tuple[dict, dict]:
    base, cutoff = row["base"], row["as_of"]
    a, profit_ids = fixed_information(row, source)
    report = old.choose_report(source["reports"], cutoff)
    current = old.holding_weights(report)
    index = source["sessions"].index(base)
    used, ignored = [], []
    for event in material["public"]:
        if event["available_at"] > cutoff or event["repeated_facts"]:
            continue
        age = index - event["evening_index"]
        if not 0 <= age < 20:
            continue
        event_report = old.choose_report(source["reports"], event["available_at"])
        links = relate(event, event_report, profiles, cutoff)
        links = [dict(link, weight=min(link["weight"], current.get(link["code"], 0))) for link in links]
        links = [link for link in links if link["weight"] > 0]
        if not links:
            ignored.append(event["id"])
            continue
        used.append({"event": event, "links": links, "age": age, "weight": sum(x["weight"] for x in links)})
    vectors = {}
    for group in ("B_LINK", "C_EVENT"):
        values = a[:20]
        for window in (1, 5, 20):
            for kind in ("news", "policy"):
                selected = [x for x in used if x["age"] < window and x["event"]["source_kind"] == kind]
                weighted = []
                for item in selected:
                    factor = (item["event"]["new_fraction"] * 2 ** (-item["age"] / 5)) if group == "C_EVENT" else 1.0
                    weighted += [(item["event"], link, link["weight"] * factor) for link in item["links"] if factor > 0]
                values += [
                    sum(w for _, _, w in weighted),
                    sum(w for _, link, w in weighted if link["basis"] == "ISSUER") if weighted else None,
                ]
                for field, categories in old.PUBLIC_ENUMS.items():
                    for category in categories:
                        values.append(
                            sum(
                                w
                                for event, link, w in weighted
                                if (link["direction"] if field == "direction" else event[field]) == category
                            )
                            if weighted
                            else None
                        )
        vectors[group] = values
    extra = dict.fromkeys(EXTRA_NAMES, None)
    public_reaction, public_cover, private_reaction, private_cover = 0.0, 0.0, 0.0, 0.0
    public_pending = private_pending = 0.0
    if used:
        for name in EXTRA_NAMES[:4] + ["public_source_time_unknown_weight"] + ["context_exposure_" + t for t in TOPICS]:
            extra[name] = 0.0
    for item in used:
        event, weight = item["event"], item["weight"]
        extra["public_new_fact_weight"] += weight * event["new_fraction"]
        extra["public_repeated_fact_weight"] += weight * (1 - event["new_fraction"])
        extra["public_source_time_unknown_weight"] += weight
        for link in item["links"]:
            key = "public_direct_weight" if link["basis"] == "ISSUER" else "public_context_weight"
            extra[key] += link["weight"]
            for topic in link["topics"]:
                extra["context_exposure_" + topic] += link["weight"]
            reaction, proof = observed_reaction(link["code"], event["available_at"], base, cutoff, source)
            link["reaction"] = {**proof, "value": reaction}
            if reaction is not None:
                public_reaction += reaction * link["weight"]
                public_cover += link["weight"]
            if proof["status"] == "NO_COMPLETE_SESSION_YET":
                public_pending += link["weight"]
    extra["public_observed_reaction"] = public_reaction / public_cover if public_cover else None
    extra["public_reaction_coverage"] = public_cover if used else None
    extra["public_no_complete_session_weight"] = public_pending if used else None
    private = company_rows(base, cutoff, source)
    company_proof, buyback_delta, buyback_cover = [], 0.0, 0.0
    for event, weight, _age in private:
        reaction, proof = observed_reaction(event["code"], event["version_available_at"], base, cutoff, source)
        company_proof.append(
            {
                "id": event["id"],
                "available_at": event["version_available_at"],
                "weight": weight,
                "reaction": {**proof, "value": reaction},
            }
        )
        if reaction is not None:
            private_reaction += weight * reaction
            private_cover += weight
        if proof["status"] == "NO_COMPLETE_SESSION_YET":
            private_pending += weight
        # 同计划、同币种、同上限的执行进度差；没有配对原件时不造“新增回购”。
        buyback = event.get("buyback")
        if cumulative_buyback(buyback) and buyback.get("plan") and buyback_month(event):
            previous = [
                e
                for e in source["company_events"]
                if e["code"] == event["code"]
                and e["status"] == "QUALIFIED"
                and (e.get("buyback") or {}).get("plan") == buyback["plan"]
                and e["version_available_at"] < event["version_available_at"]
                and comparable_buyback(e.get("buyback"), buyback)
                and buyback_month(e)
                and buyback_month(e) < buyback_month(event)
            ]
            if previous:
                last = max(previous, key=lambda e: e["version_available_at"])
                delta = buyback["ratio"]["value"] - last["buyback"]["ratio"]["value"]
                if delta < 0:
                    company_proof[-1]["buyback_delta_missing"] = "CUMULATIVE_DECREASE_NEEDS_ORIGINAL_RECONCILIATION"
                    continue
                buyback_delta += delta * weight
                buyback_cover += weight
                company_proof[-1]["buyback_delta"] = {
                    "value": delta,
                    "previous_id": last["id"],
                    "previous_available_at": last["version_available_at"],
                }
    extra["company_observed_reaction"] = private_reaction / private_cover if private_cover else None
    extra["company_reaction_coverage"] = private_cover if private else None
    extra["company_no_complete_session_weight"] = private_pending if private else None
    extra["company_newest_age"] = min((age for _, _, age in private), default=None)
    extra["buyback_execution_delta"] = buyback_delta / buyback_cover if buyback_cover else None
    extra["buyback_delta_coverage"] = buyback_cover if private else None
    # 当前已核实的forecast事件没有同口径数值；两列保持未知，训练会删除全缺列。
    vectors["C_EVENT"] += [extra[name] for name in EXTRA_NAMES]
    proof = {
        "target": row["target"],
        "as_of": cutoff,
        "profit_events": profit_ids,
        "public_events": [
            {
                "id": x["event"]["id"],
                "title": x["event"]["title"],
                "available_at": x["event"]["available_at"],
                "kind": x["event"]["source_kind"],
                "new_fraction": x["event"]["new_fraction"],
                "age": x["age"],
                "links": x["links"],
            }
            for x in used
        ],
        "ignored_unlinked_events": ignored,
        "company_events": company_proof,
        "extras": extra,
    }
    return {"A_FIX": a, **vectors}, proof


def comparable_buyback(previous: dict | None, current: dict) -> bool:
    """执行比例可比需同币种、同金额上限；引文表述差异不等于计划金额变化。"""
    if not cumulative_buyback(previous) or not cumulative_buyback(current):
        return False
    a, b = previous.get("cap"), current.get("cap")
    return bool(a and b and a.get("value") == b.get("value") and a.get("currency") == b.get("currency"))


def cumulative_buyback(value: dict | None) -> bool:
    """累计执行金额必须由金额引文本身说明；孤立的单次支付金额不得当累计完成比例。"""
    if not value or not value.get("ratio") or not value.get("executed"):
        return False
    quote = clean(value["executed"].get("quote"))
    return "累计" in quote or "总金额" in quote


def buyback_month(event: dict) -> str | None:
    """只对有明确年月的进展公告计算连续进度；最终结果或混合期间暂不强行配对。"""
    match = re.search(r"(20\d{2})年(\d{1,2})月股份回购进展", event.get("title", ""))
    return f"{match[1]}-{int(match[2]):02d}" if match else None


def feature_names(group: str) -> list:
    names = list(old.INFO_NAMES)
    names[3] = "profit_yoy"
    if group != "A_FIX":
        names = [name.replace("_observed_count", "_observed_link_weight") for name in names]
    return names + (EXTRA_NAMES if group == "C_EVENT" else [])


def prepare(root: Path):
    start(root)
    source = old.read(old.ROOT / "sources.json")
    rows = old.lines(old.ROOT / "dataset.jsonl")
    diagnose(root, source, rows)
    material = supplement(root, source)
    profiles = defaultdict(list)
    for profile in sorted(material["profiles"], key=lambda p: (p["available_at"], p["document_id"])):
        profiles[profile["code"]].append(profile)
    evenings = old.prediction_evenings(source["sessions"])
    for pool, key in [(source["company_events"], "version_available_at"), (material["public"], "available_at")]:
        for event in pool:
            if event.get(key):
                event["evening_index"] = bisect.bisect_left(evenings, event[key])
    output = {group: [] for group in GROUPS}
    proofs = []
    for row in rows:
        if "information" not in row["groups"]:
            continue
        revised, proof = revised_features(row, source, material, profiles)
        for group in GROUPS:
            output[group].append({**row, "groups": {**row["groups"], "information": revised[group]}})
        proofs.append(proof)
    for group in GROUPS:
        old.write_lines(root / (group + "-inputs.jsonl"), output[group])
    old.write_lines(root / "feature-lineage.jsonl", proofs)
    old.write(root / "feature-names.json", {g: feature_names(g) for g in GROUPS})
    summary = {
        "rows": len(proofs),
        "business_profiles": len(material["profiles"]),
        "verified_public_documents": len(material["public"]),
        "material_exclusions": len(material["exclusions"]),
        "corrected_profit_rows": sum(r["groups"]["information"][3] is not None for r in output["A_FIX"]),
        "buyback_ratio_rows": sum(r["groups"]["information"][6] is not None for r in output["A_FIX"]),
        "public_related_days": sum(bool(p["public_events"]) for p in proofs),
        "related_news_days": sum(any(e["kind"] == "news" for e in p["public_events"]) for p in proofs),
        "related_policy_days": sum(any(e["kind"] == "policy" for e in p["public_events"]) for p in proofs),
        "public_link_basis": dict(
            Counter(link["basis"] for p in proofs for e in p["public_events"] for link in e["links"])
        ),
        "retained_public_unique": len({e["id"] for p in proofs for e in p["public_events"]}),
        "ignored_unlinked_unique": len({e for p in proofs for e in p["ignored_unlinked_events"]}),
        "guidance_revision_rows": 0,
        "reaction_days": sum(
            p["extras"]["public_observed_reaction"] is not None or p["extras"]["company_observed_reaction"] is not None
            for p in proofs
        ),
        "historical_first_seen_proven": False,
        "paid_calls": 0,
    }
    old.write(root / "coverage.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def freeze(root: Path):
    """准备和边界测试结束后固定代码/全部输入，禁止成绩出来后修改规则。"""
    assert old.read(root / "tests.json")["passed"]
    protocol = {
        **old.read(root / "intent.json"),
        "frozen_at": old.now(),
        "groups": GROUPS,
        "rule_B": "利润和回购累计口径修复；新闻按两时点持仓和已公开实际业务引文关联，行业关联方向未知",
        "rule_C": "B加原文新增比例、5交易日半衰期、完整交易日后价格反应，实际增量不当市场预期",
        "fit_recipe": old.RECIPE,
        "new_fits": 71,
        "historical_gate": "2025/2026均较修复对照A正确更多且Brier更低；下跌识别不降逾5个百分点；季度至少半数不差",
        "formal_gate": "历史首见与未见未来样本证据不足，任何历史改善均不得自动采用",
    }
    old.write(root / "protocol.json", protocol)
    paths = list(root.glob("*.json")) + list(root.glob("*.jsonl"))
    code = [
        Path(__file__),
        old.PY / "scripts/fund_002112_evening_refine_verify_v1.py",
        old.PY / "scripts/test_fund_002112_evening_refine_v1.py",
        Path(old.__file__),
    ]
    (root / "code").mkdir()
    for path in code:
        with (root / "code" / path.name).open("xb") as handle:
            handle.write(path.read_bytes())
    old.write(root / "freeze.json", {"files": {str(p.resolve()): old.sha(p) for p in paths + code}})


def check_freeze(root: Path):
    for name in ["freeze.json", "old-artifacts.json", "supplement-source-manifest.json"]:
        for path, expected in old.read(root / name)["files"].items():
            if old.sha(Path(path)) != expected:
                raise ValueError("SOURCE_CHANGED:" + path)


def score(records):
    return old.metrics([{**row, "value": row["probabilities"]} for row in records], "value")


def gate(scores: dict, selected: str):
    checks = []
    for year in ("2025", "2026"):
        a, b = scores[year][selected], scores[year]["A_FIX"]
        periods = [p for p in scores if p.startswith(year + "Q")]
        quarters = sum(scores[p][selected]["accuracy"] >= scores[p]["A_FIX"]["accuracy"] for p in periods)
        checks.append(
            {
                "year": year,
                "accuracy_better": a["correct"] > b["correct"],
                "brier_better": a["brier"] < b["brier"],
                "down_recall_protected": a["down_recall"] >= b["down_recall"] - 0.05,
                "quarterly_support": quarters >= math.ceil(len(periods) / 2),
            }
        )
    return {"checks": checks, "passed": all(all(v for k, v in check.items() if k != "year") for check in checks)}


def train(root: Path):
    check_freeze(root)
    if (root / "fit-ledger.jsonl").exists():
        raise ValueError("TRAINING_ALREADY_STARTED")
    datasets = {g: old.lines(root / (g + "-inputs.jsonl")) for g in GROUPS}
    previous = {r["target"]: r for r in old.lines(old.ROOT / "historical-predictions.jsonl")}
    output, reused = [], {}
    selected = None
    for year in ("2025", "2026"):
        for group in GROUPS:
            evaluation = [r for r in datasets[group] if r["target"].startswith(year)]
            for start_index in range(0, len(evaluation), 20):
                block = evaluation[start_index : start_index + 20]
                prefix = year + f"_{start_index:03d}_"
                if len(old.lines(root / "fit-ledger.jsonl") if (root / "fit-ledger.jsonl").exists() else []) >= LIMIT:
                    raise ValueError("FIT_CAP")
                saved = old.fit_expert(
                    root, group + "_" + prefix + "information", "information", datasets[group], block[0]["as_of"], block
                )
                info = old.predict_expert(saved, block, "information")
                for row, p in zip(block, info, strict=True):
                    ref = previous[row["target"]]
                    assert ref["fit_prefix"] == prefix and ref["as_of"] == row["as_of"]
                    probabilities = old.blend(
                        {
                            "information": p,
                            "market": np.array(ref["branches"]["market"]),
                            "history": np.array(ref["branches"]["history"]),
                        }
                    )
                    output.append(
                        {
                            "group": group,
                            "target": row["target"],
                            "as_of": row["as_of"],
                            "label": row["label"],
                            "probabilities": probabilities.tolist(),
                            "information": p.tolist(),
                            "baseline_fit_prefix": prefix,
                            "new_fit": group + "_" + prefix + "information",
                        }
                    )
                for branch in ("market", "history"):
                    path = old.ROOT / "models" / (prefix + branch + ".joblib")
                    expected = old.read(old.ROOT / "fits" / (prefix + branch + ".json"))["model_sha256"]
                    assert old.sha(path) == expected
                    reused[str(path)] = expected
            print(f"完成 {year} {group}：{len(evaluation)}个历史判断", flush=True)
        if year == "2025":
            candidates = {g: score([r for r in output if r["group"] == g]) for g in GROUPS}
            selected = min(
                ("B_LINK", "C_EVENT"),
                key=lambda g: (
                    -candidates[g]["correct"],
                    candidates[g]["brier"],
                    len(datasets[g][0]["groups"]["information"]),
                    g,
                ),
            )
            old.write(
                root / "selection.json",
                {
                    "at": old.now(),
                    "selected": selected,
                    "scores_2025": candidates,
                    "future_validation": False,
                    "adoption": False,
                    "selected_before_2026_new_fits": True,
                },
            )
    old.write(root / "reused-models.json", {"files": reused})
    old.write_lines(root / "predictions.jsonl", output)
    scores = {}
    periods = ["2025", "2026", "ALL"] + [y + "Q" + str(q) for y in ("2025", "2026") for q in range(1, 5)]
    for period in periods:

        def matches(r, period=period):
            return period == "ALL" or (
                r["target"].startswith(period[:4])
                and (len(period) == 4 or (int(r["target"][5:7]) - 1) // 3 + 1 == int(period[-1]))
            )

        subset = [r for r in output if matches(r)]
        if not subset:
            continue
        scores[period] = {g: score([r for r in subset if r["group"] == g]) for g in GROUPS}
        old_rows = [r for r in previous.values() if matches(r)]
        scores[period].update(
            {
                g: old.metrics(old_rows, field)
                for g, field in [
                    ("OLD", "weighted_55"),
                    ("PRICE", "price_history"),
                    ("NAV", "history_only"),
                    ("ALWAYS_UP", "always_up"),
                ]
            }
        )
    old.write(root / "scores.json", scores)
    cutoff = old.now()
    final = old.fit_expert(root, "FULL_" + selected, "information", datasets[selected], cutoff, [])
    original_package = joblib.load(old.ROOT / "002112-evening-1d-news55.joblib")
    package = {
        **original_package,
        "experts": {**original_package["experts"], "information": final},
        "information_recipe": selected,
        "trained_at": old.now(),
        "research_only": True,
        "features": {**original_package["features"], "information": old.read(root / "feature-names.json")[selected]},
        "freeze_sha256": old.sha(root / "freeze.json"),
    }
    joblib.dump(package, root / "002112-evening-refined.joblib")
    replay = old.fit_expert(
        root, "REPLAY_" + selected, "information", datasets[selected], cutoff, datasets[selected][-8:]
    )
    difference = float(
        np.max(
            np.abs(
                old.predict_expert(replay, datasets[selected][-8:], "information")
                - old.predict_expert(final, datasets[selected][-8:], "information")
            )
        )
    )
    assert difference <= 1e-12
    historical = gate(scores, selected)
    old.write(
        root / "decision.json",
        {
            "at": old.now(),
            "selected": selected,
            "historical_gate": historical,
            "training_complete": True,
            "adopted": False,
            "production_eligible": False,
            "reason": "仅历史重建；历史首见及未见未来检验未完成",
            "fits": len(old.lines(root / "fit-ledger.jsonl")),
            "replay_max_error": difference,
            "unchanged_market_history_weights": True,
        },
    )
    print(
        json.dumps(
            {
                "2025": scores["2025"],
                "2026": scores["2026"],
                "ALL": scores["ALL"],
                "decision": old.read(root / "decision.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "freeze", "train"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "train": train}[args.stage](args.root)
