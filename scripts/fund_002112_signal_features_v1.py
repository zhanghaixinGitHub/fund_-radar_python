"""固定20项最小事件输入：原文数值、两时点持仓、实质进展及缺失原因。

本模块不读取答案或成绩。所有数值保留引文，无法核对主体/期间/分母时保持未知。
"""

from __future__ import annotations

import argparse
import bisect
import re
from collections import Counter
from datetime import date
from pathlib import Path

from scripts import fund_002112_signal_common_v1 as c
from scripts import fund_002112_signal_sources_v1 as sources

FIELDS = [
    "guidance_up_weight",
    "guidance_down_weight",
    "revenue_yoy",
    "parent_profit_yoy",
    "order_to_annual_revenue",
    "order_cancel_weight",
    "buyback_executed_to_cap",
    "new_major_risk_weight",
    "resolved_major_risk_weight",
    "new_forecast_weight",
    "new_realized_earnings_weight",
    "material_change_weight",
    "qualified_event_count",
    "direct_disclosed_weight",
    "processed_body_count",
    "revenue_yoy_coverage_weight",
    "profit_yoy_coverage_weight",
    "order_ratio_coverage_weight",
    "buyback_ratio_coverage_weight",
    "holding_report_age_days",
]
WEIGHT_FIELDS = {
    "guidance_up": 0,
    "guidance_down": 1,
    "order_cancel": 5,
    "risk_new": 7,
    "risk_resolved": 8,
    "forecast": 9,
    "realized": 10,
    "material": 11,
}
NUM_FIELDS = {"revenue_yoy": (2, 15), "profit_yoy": (3, 16), "order_ratio": (4, 17), "buyback_ratio": (6, 18)}
MONEY = r"(-?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|元)"


def clean(text):
    return re.sub(r"\s+", "", str(text or "")).replace("，", ",").replace("％", "%")


def period(text):
    t = clean(text)
    year = re.search(r"(20\d{2})年?", t)
    if not year:
        return None
    suffix = (
        "Q3"
        if re.search(r"前三季|三季度|第三季度|1[-至]9月", t)
        else "H1"
        if re.search(r"半年度|半年度|半.?年|1[-至]6月", t)
        else "Q1"
        if re.search(r"第一季度|一季度|1[-至]3月", t)
        else "FY"
        if re.search(r"年度|年年报|1月1日.*12月31日", t)
        else None
    )
    return year[1] + suffix if suffix else None


def money_value(match):
    return float(match[1].replace(",", "")) * {"元": 1, "万元": 1e4, "亿元": 1e8}[match[2]]


def metric_yoy(quantity, metric, current_period):
    """只接收同报告期的指定同比列；归母/扣非/绝对金额/负基数不得混淆。"""
    q, val = clean(quantity["quote"]), clean(quantity["value_text"])
    if not current_period or period(quantity.get("period_text")) not in (None, current_period):
        return None
    noun = r"归属于(?:上市公司股东|母公司所有者|母公司股东)的?净利润" if metric == "profit_yoy" else r"营业(?:总)?收入"
    n = re.search(noun, q)
    if not n or re.search(r"扣除非经常|扣非|亏损|负值|负基数", q):
        return None
    # 一句中含收入和利润时，只取该指标后、下一个指标前的同比，避免跨列错配。
    segment = q[n.end() :]
    segment = re.split(r"归属于|营业(?:总)?收入|净利润|利润总额", segment)[0]
    percentages = re.findall(r"[-+−]?\d+(?:\.\d+)?%", val)
    if len(percentages) != 1 or percentages[0] not in segment:
        return None
    if not re.search(r"同比|上年同期|去年同期|降幅", segment):
        return None
    number = float(percentages[0].replace("%", "").replace("−", "-")) / 100
    before = segment[: segment.index(percentages[0])]
    if number >= 0 and re.search(r"减少|下降|下滑|降幅", before):
        number = -number
    return {
        "raw": number,
        "value": max(-10, min(10, number)),
        "quote": quantity["quote"],
        "period": current_period,
        "unit": "ratio",
        "source_value": quantity["value_text"],
    }


def guidance_range(body, current_period):
    """只取明确归母盈利金额区间，不以同比增长冒充指引上调；亏损范围保留未知。"""
    t = clean(body)
    pattern = (
        r"归属于(?:上市公司股东|母公司所有者)的?净利润.{0,12}?(?:盈利[:：]?|为)?" + MONEY + r"[至到~～—–-]+" + MONEY
    )
    m = re.search(pattern, t)
    if not m or not current_period or re.search(r"亏损|负", m[0]):
        return None
    low = float(m[1].replace(",", "")) * {"元": 1, "万元": 1e4, "亿元": 1e8}[m[2]]
    high = float(m[3].replace(",", "")) * {"元": 1, "万元": 1e4, "亿元": 1e8}[m[4]]
    if low <= 0 or high < low:
        return None
    return {"low": low, "high": high, "period": current_period, "currency": "CNY", "quote": m[0]}


def compare_guidance(previous, current):
    if not previous or not current or any(previous[k] != current[k] for k in ("period", "currency")):
        return None
    if current["low"] > previous["high"]:
        return "guidance_up"
    if current["high"] < previous["low"]:
        return "guidance_down"
    return None


def numeric_ratio(numerator, denominator, cap=10):
    if not numerator or not denominator or numerator["currency"] != denominator["currency"]:
        return None
    if denominator["value"] <= 0 or numerator["value"] < 0:
        return None
    raw = numerator["value"] / denominator["value"]
    return {"raw": raw, "value": min(cap, raw), "unit": "ratio", "numerator": numerator, "denominator": denominator}


def buyback(body):
    """执行额与计划上限必须在同一原件内关联；无上限不猜比例，不混股份注销。"""
    t = clean(body)
    execution = re.search(
        r"(?:已支付的总金额|支付的?(?:总)?金额|累计(?:已)?支付.{0,6}?金额|累计回购.{0,8}?金额)[为达合计：:]*(?:人民币)?"
        + MONEY,
        t,
    )
    # 上限只在金额字段内寻找；到“价格/股数”等下一个字段立即停止，禁止串表格列。
    amount_field = re.search(r"(?:回购资金总额|回购金额|回购股份的资金总额|回购资金总金额)(.{0,100})", t)
    amount_text = re.split(r"回购价格|价格上限|回购数量|股数|。|；", amount_field[0])[0] if amount_field else ""
    ceiling = re.search(r"(?:不超过|不高于|上限为?)(?:人民币)?" + MONEY, amount_text)
    amount_range = re.search(MONEY + r"[至到~～—–-]+" + MONEY, amount_text)
    plan = re.search(r"(20\d{2}年\d{1,2}月\d{1,2}日).{0,90}?(?:审议通过|通过了).{0,35}?回购", t)
    if not execution:
        return None
    # 港币和美元必须显式匹配；本期不做跨币种兑换。
    if re.search(r"港币|港元|美元|美金", execution[0]):
        return None
    n = {"value": money_value(execution), "currency": "CNY", "quote": execution[0]}
    d = {"value": money_value(ceiling), "currency": "CNY", "quote": amount_text} if ceiling else None
    if d is None and amount_range:
        value = float(amount_range[3].replace(",", "")) * {"元": 1, "万元": 1e4, "亿元": 1e8}[amount_range[4]]
        d = {"value": value, "currency": "CNY", "quote": amount_text}
    if d and d["value"] < n["value"]:
        d = None
    return {"executed": n, "cap": d, "ratio": numeric_ratio(n, d), "plan": plan[1] if plan else None}


def annual_revenue(body, current_period):
    """仅接收本期年度原件中明确叙述的合并营业收入，不从多列表格猜金额。"""
    if not current_period or not current_period.endswith("FY"):
        return None
    t = clean(body)
    m = re.search(
        r"(?:报告期内[,，]?公司(?:合并报表)?|公司"
        + current_period[:4]
        + r"年(?:度)?).{0,12}?实现营业(?:总)?收入(?:为)?"
        + MONEY,
        t,
    )
    if not m:
        return None
    return {"value": money_value(m), "currency": "CNY", "quote": m[0], "period": current_period}


def effective_index(sessions, stamp):
    i = bisect.bisect_left(sessions, stamp[:10])
    if i < len(sessions) and c.moment(stamp) > c.nav_inputs.at0800(sessions[i]):
        i += 1
    return i


def age_factor(days):
    return 1.0 if days <= 90 else 0.5 if days <= 180 else 0.0


def referenced_progress(event, previous_events, body, facts):
    """终止/解除必须点名此前事项原公告；只出现负面词或一般风险说明不构成进展。"""
    flags, references, quotes = [], [], []
    for previous in reversed(previous_events):
        if previous["code"] != event["code"] or previous["status"] != "QUALIFIED":
            continue
        if c.moment(previous["version_available_at"]) >= c.moment(event["version_available_at"]):
            continue
        title = clean(previous["title"])
        if len(title) < 10 or title not in body:
            continue
        if previous.get("order"):
            quote = next((f for f in facts if re.search(r"(?:终止|取消).{0,20}(?:合同|订单)", clean(f))), None)
            flag = "order_cancel"
        elif "risk_new" in previous["flags"]:
            quote = next((f for f in facts if re.search(r"(?:恢复生产|解除.{0,12}风险|终止调查|结案)", clean(f))), None)
            flag = "risk_resolved"
        else:
            continue
        if quote:
            flags.append(flag)
            references.append(previous["id"])
            quotes.append(quote)
            break
    return flags, references, quotes


def build_events(documents, cached, reports):
    events, previous_guidance, previous_buyback, annuals, fingerprints = [], {}, {}, {}, {}
    financial_seen = {}
    # 同一公开时刻的全文/摘要优先保留有核实数值的一份；不从未来全文把数字搬到较早摘要。
    for d in sorted(
        documents,
        key=lambda x: (
            x["body_available_at"] or "9999",
            -len(cached.get(x.get("event_id"), {}).get("quantities", [])),
            x["id"],
        ),
    ):
        old = cached.get(d.get("event_id"), {})
        body, title = clean(d.get("body")), clean(d["title"])
        code = d.get("stock_code")
        eid = d.get("event_id") or c.io.digest([d["id"], d.get("body_sha256")])
        e = {
            "id": eid,
            "document_id": d["id"],
            "title": d["title"],
            "code": code,
            "source_kind": d["kind"],
            "source_url": d.get("source_url", d.get("url")),
            "source_hash": d.get("body_sha256"),
            "raw_sha256": d.get("receipt", {}).get("sha256"),
            "published_at": d.get("title_available_at"),
            "version_available_at": d.get("body_available_at"),
            "fetched_at": d.get("receipt", {}).get("at", d.get("receipt", {}).get("received_at")),
            "period": period(title),
            "flags": [],
            "numeric": {},
            "quotes": [],
            "missing": {},
            "previous_event_ids": [],
            "status": "PROCESSED_NO_ELIGIBLE_EVENT",
            "repeated_of": None,
            "cache_reused": bool(old),
            "raw_path": d.get("raw_path"),
            "links": [],
        }
        if not body or not e["version_available_at"] or d.get("body_exclusions"):
            e["status"] = "QUARANTINED_BODY_OR_VERSION"
            events.append(e)
            continue
        if not code or d["kind"] != "company":
            e["status"] = "BACKGROUND_WITHOUT_DIRECT_COMPANY_FACT"
            events.append(e)
            continue
        report = c.choose_report(reports, e["version_available_at"])
        weight = c.bodies.weights(report).get(code, 0)
        if weight:
            e["links"] = [
                {
                    "code": code,
                    "weight": weight,
                    "report_hash": report["raw"]["sha256"],
                    "report_end": report["report_end"],
                    "report_available_at": report["available_at"],
                }
            ]
        # 旧协议隔离的结果不因新增关键词规则重新准入；新增原件用下面可审计的窄规则。
        if old and old.get("status") != "SOURCE_GROUNDED":
            e["status"] = "QUARANTINED_EXTRACTION"
            events.append(e)
            continue
        facts = [f for f in old.get("facts", []) if clean(f) in body]
        # 非缓存原件只作确定性字面提取，不追加大模型请求。
        if not old:
            facts = [
                m[0]
                for m in re.finditer(
                    r"[^。\n]{0,120}(?:营业收入|归属于.{0,20}净利润|回购|签订|立案|处罚)[^。\n]{0,220}", body
                )
            ][:12]
        financial = bool(
            re.search(r"年度报告|季度报告|半年度报告|业绩快报|主要财务数据|业绩预告|业绩预增|业绩预减", title)
        )
        forecast = bool(re.search(r"业绩预告|业绩预增|业绩预减", title))
        material_facts = [
            f
            for f in facts
            if re.search(r"营业(?:总)?收入|归属于.{0,20}净利润|净亏损", clean(f)) and re.search(r"\d", f)
        ]
        if financial and e["period"] and material_facts and not sources.EXCLUDE.search(title):
            e["flags"].append("forecast" if forecast else "realized")
            e["quotes"].extend(material_facts)
            if not forecast:
                for metric, source_metric in (("revenue_yoy", "REVENUE_YOY"), ("profit_yoy", "PROFIT_YOY")):
                    valid = [
                        metric_yoy(q, metric, e["period"])
                        for q in old.get("quantities", [])
                        if q["metric"] == source_metric and clean(q["quote"]) in body
                    ]
                    valid = [q for q in valid if q]
                    if len({q["value"] for q in valid}) == 1:
                        e["numeric"][metric] = valid[0]
                    else:
                        e["missing"][metric] = "NO_UNAMBIGUOUS_CURRENT_PERIOD_LITERAL_YOY"
                revenue = annual_revenue(body, e["period"])
                if revenue:
                    annuals[(code, e["period"])] = {
                        **revenue,
                        "event_id": eid,
                        "available_at": e["version_available_at"],
                    }
            else:
                current = guidance_range(body, e["period"])
                previous = previous_guidance.get((code, e["period"]))
                direction = compare_guidance(previous, current)
                if direction:
                    e["flags"].append(direction)
                    e["guidance"] = {"previous": previous, "current": current}
                    e["previous_event_ids"].append(previous["event_id"])
                if current:
                    previous_guidance[(code, e["period"])] = {
                        **current,
                        "event_id": eid,
                        "available_at": e["version_available_at"],
                    }
        if (
            re.search(r"(?:回购.*(?:进展|实施结果|完成)|首次回购)", title)
            and not sources.EXCLUDE.search(title)
            and "注销" not in title
        ):
            b = buyback(body)
            if b:
                key = (code, b["plan"])
                previous = previous_buyback.get(key)
                changed = b["executed"]["value"] > 0 and (
                    previous is None or previous["executed"]["value"] != b["executed"]["value"]
                )
                # 缺计划标识时只认首次明确执行，不串联不同年度的相似回购。
                if b["plan"] and changed:
                    e["flags"].append("buyback_execution")
                    e["quotes"].append(b["executed"]["quote"])
                    if b["ratio"]:
                        e["numeric"]["buyback_ratio"] = b["ratio"]
                        e["quotes"].append(b["cap"]["quote"])
                    else:
                        e["missing"]["buyback_ratio"] = "SAME_PLAN_AMOUNT_CAP_MISSING"
                    if previous:
                        e["previous_event_ids"].append(previous["event_id"])
                elif not b["plan"]:
                    e["missing"]["buyback_ratio"] = "PLAN_ID_UNVERIFIED"
                e["buyback"] = b
                if b["plan"]:
                    previous_buyback[key] = {**b, "event_id": eid}
        if re.search(r"重大.*(?:合同|订单)|签订.*(?:合同|订单)|中标", title) and not sources.EXCLUDE.search(title):
            order = next(
                (
                    q
                    for q in old.get("quantities", [])
                    if q["metric"] == "CONTRACT_AMOUNT"
                    and clean(q["quote"]) in body
                    and re.search(r"订单|合同|采购", q["quote"])
                ),
                None,
            )
            if order and not re.search(r"拟|尚未|意向", clean(order["quote"])):
                match = re.search(MONEY, clean(order["value_text"]))
                if match and not re.search(r"美元|港元|港币", clean(order["quote"])):
                    n = {"value": money_value(match), "currency": "CNY", "quote": order["quote"]}
                    # 上一完整年度按事件日期确定，且分母版本必须早于该事件。
                    year = str(int(e["version_available_at"][:4]) - 1) + "FY"
                    denom = annuals.get((code, year))
                    ratio = numeric_ratio(n, denom)
                    e["order"] = n
                    e["quotes"].append(order["quote"])
                    if ratio:
                        e["numeric"]["order_ratio"] = ratio
                        e["previous_event_ids"].append(denom["event_id"])
                    else:
                        e["missing"]["order_ratio"] = "SAME_ISSUER_LAST_FULL_YEAR_REVENUE_UNVERIFIED"
                    if "重大" in title or (ratio and ratio["raw"] >= 0.1):
                        e["flags"].append("major_order")
        # 公司/直接控股子公司风险与股东个人风险不同，本期不将股东个人行为算为公司事件。
        if re.search(r"停产|违约|立案|处罚", title) and not re.search(
            r"股东|控制人|董事|不存在|未被|最近五年|制度|评估", title
        ):
            quote = next(
                (
                    f
                    for f in facts
                    if re.search(
                        r"公司.{0,30}(?:收到.{0,20}(?:立案通知|处罚决定)|被立案|发生违约|已停产|停止生产)", clean(f)
                    )
                ),
                None,
            )
            if quote:
                e["flags"].append("risk_new")
                e["quotes"].append(quote)
        if re.search(r"终止|取消|解除|恢复生产|结案", title):
            flags, references, progress_quotes = referenced_progress(e, events, body, facts)
            e["flags"].extend(flags)
            e["previous_event_ids"].extend(references)
            e["quotes"].extend(progress_quotes)
        for key in ("guidance_up", "guidance_down", "order_cancel", "risk_resolved"):
            if key not in e["flags"]:
                e["missing"].setdefault(key, "NO_VERIFIED_COMPARABLE_PREVIOUS_EVENT_OR_PROGRESS")
        if e["flags"]:
            e["flags"].append("material")
            # 公司、事项、期间、实际金额/指标一起去重；旧执行金额的月度重复不形成新事件。
            fp = c.io.digest(
                [
                    code,
                    e["period"],
                    e["flags"],
                    {k: v["raw"] for k, v in e["numeric"].items()},
                    e.get("buyback"),
                    e.get("order"),
                    e.get("guidance"),
                    [clean(q) for q in e["quotes"]] if not e["period"] else None,
                ]
            )
            if fp in fingerprints:
                e["repeated_of"] = fingerprints[fp]
                e["status"] = "DUPLICATE_EVENT"
            else:
                fingerprints[fp] = eid
                e["status"] = "QUALIFIED"
            if financial:
                disclosure = "forecast" if forecast else "preliminary" if "业绩快报" in title else "report"
                key = (code, e["period"], disclosure)
                previous = financial_seen.get(key)
                changed = previous and any(
                    k in previous["numeric"] and v["raw"] != previous["numeric"][k]["raw"]
                    for k, v in e["numeric"].items()
                )
                changed = changed or bool(set(e["flags"]) & {"guidance_up", "guidance_down"})
                if previous and not changed:
                    e["status"] = "DUPLICATE_EVENT"
                    e["repeated_of"] = previous["id"]
                else:
                    if previous:
                        e["previous_event_ids"].append(previous["id"])
                    financial_seen[key] = e
        e["important"] = bool(
            set(e["flags"])
            & {
                "guidance_up",
                "guidance_down",
                "major_order",
                "order_cancel",
                "buyback_execution",
                "risk_new",
                "risk_resolved",
            }
        )
        for quote in e["quotes"]:
            assert clean(quote) in body, "NONCONTIGUOUS_QUOTE"
        events.append(e)
    return events


def vector(row, events, reports, sessions, decay=False):
    cutoff = row["as_of"]
    report = c.choose_report(reports, cutoff)
    if report is None:
        return [None] * 20, {"trigger": False, "reason": "NO_AVAILABLE_HOLDINGS", "events": []}
    current = c.bodies.weights(report)
    current_age = (date.fromisoformat(row["base"]) - date.fromisoformat(report["report_end"])).days
    i = sessions.index(row["base"])
    used, processed, eligible = [], [], []
    for event in events:
        stamp = event["version_available_at"]
        if not stamp or c.moment(stamp) > c.moment(cutoff):
            continue
        age = i - effective_index(sessions, stamp)
        if not 0 <= age < 5 or not event["links"]:
            continue
        link = event["links"][0]
        raw = min(link["weight"], current.get(link["code"], 0))
        if raw <= 0:
            continue
        event_report_age = (c.moment(stamp).date() - date.fromisoformat(link["report_end"])).days
        holding_age = max(event_report_age, current_age)
        factor = age_factor(holding_age) * (1 - 0.2 * age) if decay else 1
        detail = {
            "id": event["id"],
            "code": event["code"],
            "version_available_at": stamp,
            "event_report": link,
            "prediction_report": {
                "hash": report["raw"]["sha256"],
                "end": report["report_end"],
                "available_at": report["available_at"],
                "weight": current[event["code"]],
            },
            "original_weight": raw,
            "effective_weight": raw * factor,
            "holding_age": holding_age,
            "event_age_sessions": age,
            "factor": factor,
            "important": event.get("important", False),
            "status": event["status"],
        }
        used.append(detail)
        if event["status"] not in ("QUARANTINED_BODY_OR_VERSION", "QUARANTINED_EXTRACTION"):
            processed.append(detail)
        if event["status"] == "QUALIFIED" and factor > 0:
            eligible.append((event, detail))
    values = [None] * 20
    values[19] = float(current_age)
    if processed:
        for ix in (*WEIGHT_FIELDS.values(), 12, 13, 14, 15, 16, 17, 18):
            values[ix] = 0.0
        values[14] = float(len(processed))
        values[12] = float(sum(e["status"] == "QUALIFIED" for e in processed))
        for flag, ix in WEIGHT_FIELDS.items():
            company = {}
            for e, d in eligible:
                if flag in e["flags"]:
                    company[d["code"]] = max(company.get(d["code"], 0), d["effective_weight"])
            values[ix] = sum(company.values())
        direct = {}
        for _, d in eligible:
            direct[d["code"]] = max(direct.get(d["code"], 0), d["effective_weight"])
        values[13] = sum(direct.values())
        for metric, (ix, cover) in NUM_FIELDS.items():
            # 同公司多个事件只用最后一项有效数值，防止公告数量放大公司仓位。
            latest = {}
            for e, d in eligible:
                if metric in e["numeric"]:
                    latest[d["code"]] = (e["numeric"][metric]["value"], d["effective_weight"])
            weight = sum(w for _, w in latest.values())
            values[cover] = weight
            values[ix] = sum(v * w for v, w in latest.values()) / weight if weight > 0 else None
    trigger_companies = {}
    for e, d in eligible:
        if e["important"]:
            trigger_companies[d["code"]] = max(trigger_companies.get(d["code"], 0), d["original_weight"])
    trigger_weight = sum(trigger_companies.values())
    proof = {
        "trigger": trigger_weight >= 0.03,
        "trigger_original_weight": trigger_weight,
        "reason": "QUALIFIED_DIRECT_NEW_IMPORTANT_EVENTS" if trigger_weight >= 0.03 else "KEEP_BASELINE",
        "events": used,
        "missing": {
            FIELDS[k]: "NO_PROCESSED_DIRECT_BODY" if not processed else "NO_VERIFIED_NUMERIC_FACT"
            for k, v in enumerate(values)
            if v is None
        },
    }
    return values, proof


def run(root, revision="features-r1"):
    root = Path(root)
    out = root / revision
    if (root / "training-freeze.json").exists():
        raise ValueError("FEATURES_FROZEN_NO_REVISION_ALLOWED")
    docs = sources.existing_docs()
    docs.update({d["id"]: d for d in c.lines(root / "supplement-documents.jsonl")})
    cached = {e["id"]: e for e in c.lines(c.OLD / "feature-revision-r3/events.jsonl")}
    reports = c.io.read(c.OLD / "reports.json")
    events = build_events(list(docs.values()), cached, reports)
    sessions = c.bundle()["sessions"]
    rows, lineage = [], []
    for old in c.original_rows():
        row = {k: old[k] for k in ("target", "base", "as_of", "label_mature_at", "groups", "session_index")}
        row["E"], normal = vector(row, events, reports, sessions)
        row["ED"], decayed = vector(row, events, reports, sessions, True)
        row["trigger"], row["trigger_decay"] = normal["trigger"], decayed["trigger"]
        rows.append(row)
        lineage.append({"target": row["target"], "as_of": row["as_of"], "B1_B2": normal, "B3": decayed})
    c.io.save_lines(out / "events.jsonl", events)
    c.io.save_lines(out / "inputs.jsonl", rows)
    c.io.save_lines(out / "event-lineage.jsonl", lineage)
    protocol = {
        "at": c.io.now(),
        "fields": [{"id": f"F{i + 1:02d}", "name": v} for i, v in enumerate(FIELDS)],
        "baseline_fields": 15,
        "enhanced_business_fields": 35,
        "window_sessions": 5,
        "trigger_min_original_nav_weight": 0.03,
        "baseline_fraction": 0.75,
        "event_fraction": 0.25,
        "age_days": {"0_90": 1, "91_180": 0.5, "181_plus": 0},
        "time_decay": [1, 0.8, 0.6, 0.4, 0.2],
        "yoy_clip": [-10, 10],
        "new_ratio_clip": [0, 10],
        "major_order": "明确重大合同，或同主体上年已公开营收比例至少10%；拟签和无金额保持未知",
        "missing": "没有成功处理的直接持仓正文时事件字段为None；有正文但无观察事件时计数0；无数值仍None。",
        "preprocessing": "仅训练行拟合中位数、缺失指示与标准化；全空列固定0并带缺失指示，业务层仍为未知",
        "scope_contraction": (
            "股东个人处罚不作为公司重大风险；计划无法关联、指引不可比、订单取消/风险解除未配对均未知。"
        ),
        "policy_news": "无明确公司数值变化的政策/新闻只存来源，不能凭行业背景触发。",
    }
    c.io.save(out / "feature-protocol.json", protocol)
    summary = {
        "rows": len(rows),
        "events": len(events),
        "statuses": dict(Counter(e["status"] for e in events)),
        "flags": dict(Counter(f for e in events if e["status"] == "QUALIFIED" for f in e["flags"])),
        "nonempty_features": {name: sum(r["E"][i] is not None for r in rows) for i, name in enumerate(FIELDS)},
        "by_year": {
            year: {
                "rows": sum(r["target"].startswith(year) for r in rows),
                "trigger_B2": sum(r["trigger"] for r in rows if r["target"].startswith(year)),
                "trigger_B3": sum(r["trigger_decay"] for r in rows if r["target"].startswith(year)),
            }
            for year in ("2024", "2025", "2026")
        },
    }
    c.io.save(out / "feature-summary.json", summary)
    paths = [
        c.OLD / "feature-revision-r3/events.jsonl",
        c.OLD / "reports.json",
        c.OLD / "extraction-documents.jsonl",
        c.OLD / "supplement-documents.jsonl",
        c.OLD / "supplement-retry-documents.jsonl",
        c.bodies.ROOT / "documents.jsonl",
    ]
    for e in events:
        if e["status"] == "QUALIFIED" and e["raw_path"]:
            p = Path(e["raw_path"])
            assert c.io.sha(p) == e["raw_sha256"], "ORIGINAL_HASH_MISMATCH"
            paths.append(p)
    c.register_sources(out, paths, "event-source-manifest.json")
    c.io.replace(root / "active-inputs.json", {"directory": revision, "sha256": c.io.sha(out / "inputs.jsonl")})
    c.stage(
        root,
        "最小输入与条件修正",
        "BUILT_PENDING_REVIEW",
        ["feature-protocol.json", "feature-summary.json", "event-lineage.jsonl"],
        ["未取得可比前件或分母的字段保持未知"],
        "至少50个真实变化事件抽查与边界测试",
    )
    print(c.io.canonical(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", default="features-r1")
    args = parser.parse_args()
    run(args.root, args.revision)
