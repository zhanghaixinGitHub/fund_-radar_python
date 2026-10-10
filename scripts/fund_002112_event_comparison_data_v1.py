"""002112 信息对照的纯文件输入层；不采集、不连接数据库、不写现用预测。

所有候选使用同一目标日期、净值窗口和可用时间。公开日期重建仍不等于历史
首见存档；沿用旧证据中更晚的可用时间，不为了增加样本提前消息或净值。
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from app.services.direction_1d_protocol import FEATURES, features

from scripts import fund_002112_evening_fusion_v2 as legacy
from scripts import fund_002112_evening_refine_v1 as links

PY = Path(__file__).resolve().parents[1]
RESEARCH = PY / ".local-runs/fund-exposure-002112"
ROOT = RESEARCH / "event-input-comparison/20261009-v1"
SOURCE = RESEARCH / "evening-news-fusion/20261003-v2/sources.json"
MATERIAL = RESEARCH / "evening-event-refinement/20261003-v2/verified-business-material.json"
ZONE = ZoneInfo("Asia/Shanghai")
PHASES = ("EVENING_2300", "MORNING_0830")
GROUPS = ("A_NAV", "B_MARKET", "C_EVENTS")
KINDS = ("ANNOUNCEMENT", "NEWS", "POLICY")
EVENT_FIELDS = (
    "observed_count",
    "issuer_weight",
    "product_weight",
    "context_weight",
    "benefit_weight",
    "pressure_weight",
    "proposal_weight",
    "correction_weight",
    "risk_mention_weight",
)
EVENT_NAMES = [f"{kind}_{window}_{field}" for kind in KINDS for window in (5, 20) for field in EVENT_FIELDS]
EVENT_NAMES += ["profit_yoy", "profit_coverage", "revenue_yoy", "revenue_coverage"]
NAMES = {
    "A_NAV": list(FEATURES),
    "B_MARKET": list(FEATURES) + legacy.MARKET_NAMES,
    "C_EVENTS": list(FEATURES) + legacy.MARKET_NAMES + EVENT_NAMES,
}
RISK_TERMS = re.compile(r"出口管制|限制出口|禁止|制裁|处罚|违约|取消订单|供应中断|风险|限制措施")
UNCERTAIN_TERMS = re.compile(r"传闻|传言|假设|推演|或将|可能实施|拟议|征求意见")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value) -> None:
    """产物只新建；失败后重启可读取已完成阶段，不能覆盖旧账本或已冻结输入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)


def save_lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str) + "\n")


def moment(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED")
    return stamp.astimezone(ZONE)


def prediction_target(sessions: list[str], cutoff: str) -> tuple[str, str]:
    """15点前归当日，15点起归下一交易日；休市日归下一交易日。返回基日与目标日。"""
    stamp = moment(cutoff)
    day = stamp.date().isoformat()
    i = bisect.bisect_left(sessions, day)
    if i < len(sessions) and sessions[i] == day and stamp.hour >= 15:
        i += 1
    if not 0 < i < len(sessions):
        raise ValueError("CALENDAR_UNAVAILABLE")
    return sessions[i - 1], sessions[i]


def cutoff_for(target: str, phase: str) -> str:
    """固定两个评分截点；只是离线比较时刻，不创建定时任务。"""
    if phase == "MORNING_0830":
        return target + "T08:30:00+08:00"
    if phase == "EVENING_2300":
        return (date.fromisoformat(target) - timedelta(days=1)).isoformat() + "T23:00:00+08:00"
    raise ValueError("UNKNOWN_PHASE")


def strict_nav(source: dict, base: str, cutoff: str) -> tuple[list | None, dict]:
    """沿用现用入口所需的61条连续净值；基日未到齐就等待，不用更老窗口冒充基日。"""
    sessions, nav = source["sessions"], source["nav"]
    i = sessions.index(base)
    wanted = sessions[i - 60 : i + 1]
    if i < 60 or any(day not in nav for day in wanted):
        return None, {"reason": "NAV_GAP"}
    late = [day for day in wanted if moment(nav[day]["available_at"]) > moment(cutoff)]
    if late:
        return None, {"reason": "NAV_NOT_YET_AVAILABLE", "late_dates": late}
    try:
        values = features([nav[day]["unit_nav"] for day in wanted])
    except ValueError as exc:
        return None, {"reason": str(exc)}
    return values, {
        "dates": wanted,
        "values": [nav[day]["unit_nav"] for day in wanted],
        "max_available_at": max(nav[day]["available_at"] for day in wanted),
        "strict_base_ready": True,
    }


def load_sources() -> dict:
    """只读取此前保存的资料；源码中不调用旧采集、训练、数据库初始化入口。"""
    source = read(SOURCE)
    material = read(MATERIAL)
    profiles = defaultdict(list)
    for profile in sorted(material["profiles"], key=lambda item: (item["available_at"], item["document_id"])):
        profiles[profile["code"]].append(profile)
    # 使用已逐条核验业务引文的新闻/政策；公告引用旧已核验正文事实。
    public = [dict(event) for event in material["public"] if event["status"] == "SOURCE_GROUNDED"]
    company = [
        dict(event)
        for event in source["public_events"]
        if event["source_kind"] == "company" and event["status"] == "SOURCE_GROUNDED"
    ]
    documents = []
    for event in public + company:
        if event.get("repeated_facts") or not event.get("facts") or not event.get("available_at"):
            continue
        # 去除仅有泛化公司活动且没有风险事实的公告；规则不参考任何净值答案。
        text = event["title"] + " ".join(event["facts"])
        if event["source_kind"] == "company" and event["kind"] == "OTHER" and not RISK_TERMS.search(text):
            continue
        event["event_type"] = {"company": "ANNOUNCEMENT", "news": "NEWS", "policy": "POLICY"}[event["source_kind"]]
        event["uncertain_claim"] = bool(UNCERTAIN_TERMS.search(text))
        documents.append(event)
    source["documents"] = sorted(documents, key=lambda item: (item["available_at"], item["id"]))
    source["profiles"] = profiles
    return source


def related_events(source: dict, base: str, cutoff: str) -> list[dict]:
    """保存消息与历史披露持仓的两端证据；行业相关不推定公司实际受益或受损。"""
    sessions = source["sessions"]
    base_index = sessions.index(base)
    report = legacy.choose_report(source["reports"], cutoff)
    weights = legacy.holding_weights(report)
    chosen = []
    for event in source["documents"]:
        if moment(event["available_at"]) > moment(cutoff):
            continue
        # 已先检查可用时间；目标日早晨的新消息属于刚到达的信息，不能因基日是昨天而漏掉。
        age = max(0, base_index - (bisect.bisect_right(sessions, event["available_at"][:10]) - 1))
        if not 0 <= age < 20:
            continue
        if event["event_type"] == "ANNOUNCEMENT":
            relation = [
                {
                    "code": item["code"],
                    "weight": min(item["at_event_weight"], weights.get(item["code"], 0)),
                    "basis": "ISSUER",
                    "report_available_at": report["available_at"] if report else None,
                    "report_hash": report["raw"]["sha256"] if report else None,
                }
                for item in event.get("links", [])
                if weights.get(item["code"], 0) > 0
            ]
        else:
            relation = links.relate(event, report, source["profiles"], cutoff)
        if not relation:
            continue
        text = event["title"] + " " + " ".join(event["facts"])
        stage = "UNCONFIRMED_DISCUSSION" if event["uncertain_claim"] else event["stage"]
        chosen.append(
            {
                "id": event["id"],
                "kind": event["event_type"],
                "available_at": event["available_at"],
                "published_date": event.get("published_date"),
                "age": age,
                "title": event["title"],
                "source_url": event.get("source_url"),
                "source_hash": event.get("source_hash"),
                "facts": event["facts"],
                "links": relation,
                "stage": stage,
                "direction": event["direction"],
                "risk_mentioned": bool(RISK_TERMS.search(text)),
                "raw_stage": event["stage"],
            }
        )
    return chosen


def event_features(source: dict, base: str, cutoff: str) -> tuple[list, str, dict]:
    """内容信号、披露权重和原文字符特征并列；缺失保留None，不假装完整新闻覆盖。

    方向字段描述原文对象，不直接决定基金方向；风险词仅表示提及，未知/推演独立保存。
    原文字符词表由训练阶段建立，本函数不拟合词表、不接触目标收益。
    """
    events = related_events(source, base, cutoff)
    vector = []
    for kind in KINDS:
        for window in (5, 20):
            subset = [event for event in events if event["kind"] == kind and event["age"] < window]
            if not subset:
                vector += [0.0] + [None] * (len(EVENT_FIELDS) - 1)
                continue

            def exposure(predicate, bases=None, observed=subset):
                by_stock = {}
                for event in observed:
                    if not predicate(event):
                        continue
                    for link in event["links"]:
                        if bases is None or link["basis"] in bases:
                            by_stock[link["code"]] = max(by_stock.get(link["code"], 0), link["weight"])
                return sum(by_stock.values())

            vector += [
                float(len(subset)),
                exposure(lambda _: True, {"ISSUER"}),
                exposure(lambda _: True, {"PRODUCT"}),
                exposure(lambda _: True, {"INDUSTRY_CONTEXT"}),
                exposure(
                    lambda event: event["direction"] == "BENEFIT" and event["stage"] in ("IMPLEMENTATION", "REALIZED"),
                    {"ISSUER"},
                ),
                exposure(
                    lambda event: event["direction"] == "PRESSURE" and event["stage"] in ("IMPLEMENTATION", "REALIZED"),
                    {"ISSUER"},
                ),
                exposure(lambda event: event["stage"] in ("PROPOSAL", "FORECAST", "UNCONFIRMED_DISCUSSION")),
                exposure(lambda event: event["stage"] in ("CORRECTION", "CANCELLED")),
                exposure(lambda event: event["risk_mentioned"]),
            ]
    # 使用已有数值的原字段名profit_yoy，避免旧版本parent_profit_yoy错配；只读原件。
    report = legacy.choose_report(source["reports"], cutoff)
    weights = legacy.holding_weights(report)
    numeric_evidence = []
    for field in ("profit_yoy", "revenue_yoy"):
        latest = {}
        for event in source["company_events"]:
            stamp = event.get("version_available_at")
            if event["status"] != "QUALIFIED" or not stamp or moment(stamp) > moment(cutoff):
                continue
            age = sessions_age(source["sessions"], base, stamp)
            number = event["numeric"].get(field)
            weight = weights.get(event["code"], 0)
            if 0 <= age < 20 and weight > 0 and number and number.get("quote"):
                previous = latest.get(event["code"])
                if previous is None or previous[0] < stamp:
                    latest[event["code"]] = (stamp, number["value"], weight, event, number["quote"])
        cover = sum(item[2] for item in latest.values())
        vector += [sum(item[1] * item[2] for item in latest.values()) / cover if cover else None, cover]
        numeric_evidence += [
            {
                "field": field,
                "code": code,
                "available_at": item[0],
                "value": item[1],
                "weight": item[2],
                "id": item[3]["id"],
                "source_hash": item[3].get("source_hash"),
                "source_url": item[3].get("source_url"),
                "quote": item[4],
                "report_available_at": report["available_at"],
            }
            for code, item in latest.items()
        ]
    # 各类最多16条最近原文，防止公告数量挤掉政策/新闻；不依据收益挑文本。
    text_ids, texts = [], []
    for kind in KINDS:
        subset = sorted(
            (event for event in events if event["kind"] == kind), key=lambda event: (event["available_at"], event["id"])
        )[-16:]
        for event in subset:
            text_ids.append(event["id"])
            texts.append((event["title"] + " " + " ".join(event["facts"]))[:500])
    assert len(vector) == len(EVENT_NAMES)
    return (
        vector,
        "\n".join(texts),
        {
            "events": events,
            "numeric_evidence": numeric_evidence,
            "text_event_ids": text_ids,
            "counts": dict(Counter(event["kind"] for event in events)),
            "collection_complete": False,
            "historical_first_seen_proven": False,
        },
    )


def sessions_age(sessions: list[str], base: str, stamp: str) -> int:
    return max(0, sessions.index(base) - (bisect.bisect_right(sessions, stamp[:10]) - 1))


def build_dataset(source: dict) -> tuple[list, list, list]:
    """先建时间安全输入，再挂接成熟标签；同一目标日多个时刻始终属于同一历史区间。"""
    rows, proofs, excluded = [], [], []
    for target in source["sessions"]:
        if not "2024-01-02" <= target <= "2026-09-30":
            continue
        for phase in PHASES:
            cutoff = cutoff_for(target, phase)
            base, resolved = prediction_target(source["sessions"], cutoff)
            assert resolved == target
            nav_values, nav_proof = strict_nav(source, base, cutoff)
            if nav_values is None or target not in source["nav"]:
                excluded.append(
                    {
                        "target": target,
                        "phase": phase,
                        "cutoff": cutoff,
                        **(nav_proof if nav_values is None else {"reason": "TARGET_NAV_MISSING"}),
                    }
                )
                continue
            market, market_proof = legacy.market_features(base, cutoff, source)
            events, text, event_proof = event_features(source, base, cutoff)
            raw_input = {"A_NAV": nav_values, "B_MARKET": nav_values + market, "C_EVENTS": nav_values + market + events}
            base_nav, target_nav = (Decimal(source["nav"][day]["unit_nav"]) for day in (base, target))
            row = {
                "base": base,
                "target": target,
                "phase": phase,
                "cutoff": cutoff,
                "groups": raw_input,
                "text": text,
                "input_hash": digest([raw_input, text, cutoff]),
                "label": "UP" if target_nav > base_nav else "DOWN" if target_nav < base_nav else "FLAT",
                "return": float(target_nav / base_nav - 1),
                "mature_at": max(source["nav"][day]["available_at"] for day in (base, target)),
            }
            rows.append(row)
            proofs.append(
                {
                    "target": target,
                    "base": base,
                    "phase": phase,
                    "cutoff": cutoff,
                    "input_hash": row["input_hash"],
                    "nav": nav_proof,
                    "market": market_proof,
                    "events": event_proof,
                }
            )
    return rows, proofs, excluded
