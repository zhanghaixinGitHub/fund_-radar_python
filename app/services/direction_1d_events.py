"""002112事件依据版：资料数量不参与方向；没有可核验事件时允许暂不判断。

原综合包及原预测仍由V1恢复。本版只读取预测时已知的事件正文和披露持仓，
按公开日期衰减；接收时间只作准入检查，晚下载不能让旧消息重新变新。
"""

from __future__ import annotations

import bisect
import hashlib
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import expit

VERSION = "002112_EVENT_EVIDENCE_V2"
GROUPS = {"MORNING_0830": "CN_MIXED_002112_EVENT_AM", "EVENING_2300": "CN_MIXED_002112_EVENT_PM"}
FEATURES = [
    "holding_return_1",
    "holding_return_5",
    "hs300_return_1",
    "zz500_return_1",
    "announcement_realized",
    "announcement_forecast",
    "policy_implemented",
    "news_realized",
]
MARKET_INDEX = [7, 8, 19, 24]
REASONS = {
    "NO_DIRECTIONAL_EVENT": "没有找到内容、时间和持仓关联均可核验的明确方向事件。",
    "EVENT_CONFLICT": "相关事件同时存在支持和拖累，暂时无法形成明确方向。",
    "MARKET_EVENT_CONFLICT": "事件指向与已知持仓行情相反，暂时无法确认哪一方占主导。",
    "WEAK_SIGNAL": "综合信号差距较小，暂不足以给出涨跌判断。",
    "EVENT_MODEL_CONFLICT": "计算结果缺少同方向的具体事件支持。",
    "EVENT_LIMIT": "相关事件尚未全部完成核对，暂不判断。",
    "MARKET_INCOMPLETE": "可核对的持仓行情不够完整，暂不判断。",
    "MODEL_UNSUPPORTED": "现有历史样本尚不足以支持这类事件的方向判断。",
}
POLICY = {
    "max_age_sessions": 5,
    "half_life_sessions": 2,
    "min_score": 0.6,
    "conflict_ratio": 0.35,
    "max_events": 32,
    "min_holding_coverage": 0.5,
}
ADMIN = re.compile(r"会议|董事会工作|监事会工作|督导|法律意见|审计委员会|财务决算|权益分派|分红|股权激励|限制性股票")
BUSINESS = re.compile(r"净利润|营业收入|营业总收入|亏损|减值|订单|合同|中标|批准|注册证|补贴|出口|关税|制裁|处罚|停产")


def moment(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("EVENT_TIME_INVALID")
    return result


def build_evidence(proof: dict, base: str, cutoff: str, sessions: list[str]) -> dict:
    """每公司、事件类别、方向只取最强一条，重复文件/转载不累加。

    只接纳已核验方向标签且正文含具体经营事项的事件；行业泛相关、未知方向、
    提案、传闻、仅有风险提示均保留为未采用。业绩预告单列，不能冒充已实现利润。
    """
    rejected = Counter()
    selected = {}
    end = moment(cutoff)
    base_index = bisect.bisect_right(sessions, base) - 1
    first_publication = {}
    for event in proof.get("events", []):
        signature = (
            event.get("kind"),
            event.get("direction"),
            re.sub(r"\s+", "", " ".join(event.get("facts", []))),
            tuple(sorted(link.get("code", "") for link in event.get("links", []))),
        )
        published = event.get("published_date")
        if isinstance(published, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", published):
            first_publication[signature] = min(first_publication.get(signature, published), published)
    for event in proof.get("events", []):
        try:
            signature = (
                event.get("kind"),
                event.get("direction"),
                re.sub(r"\s+", "", " ".join(event.get("facts", []))),
                tuple(sorted(link.get("code", "") for link in event.get("links", []))),
            )
            published = first_publication.get(signature, event["published_date"])
            age = max(0, base_index - (bisect.bisect_right(sessions, published) - 1))
            if moment(event["available_at"]) > end or published > cutoff[:10] or age >= POLICY["max_age_sessions"]:
                rejected["OUTSIDE_TIME"] += 1
                continue
            quote = " ".join(event.get("facts", []))
            direction, stage, kind = event["direction"], event["stage"], event["kind"]
            if (
                not re.fullmatch(r"[a-f0-9]{64}", event.get("source_hash") or "")
                or not quote
                or not BUSINESS.search(quote)
                or ADMIN.search(event["title"])
                or direction not in ("BENEFIT", "PRESSURE")
            ):
                rejected["CONTENT_UNCONFIRMED"] += 1
                continue
            if stage in ("IMPLEMENTATION", "REALIZED"):
                field = {"ANNOUNCEMENT": 4, "POLICY": 6, "NEWS": 7}[kind]
            elif stage == "FORECAST" and kind == "ANNOUNCEMENT" and re.search(r"业绩预告|业绩快报", event["title"]):
                field = 5
            else:
                rejected["NOT_IMPLEMENTED"] += 1
                continue
            sign = 1 if direction == "BENEFIT" else -1
            decay = 2 ** (-age / POLICY["half_life_sessions"])
            linked = False
            for link in event.get("links", []):
                weight = link.get("weight")
                # 政策/新闻的整体利好不代表每家关联公司受益；产品关联还需两端引文和相同方向。
                if link.get("basis") == "PRODUCT" and (
                    link.get("direction") != direction
                    or not link.get("source_quote")
                    or not link.get("profile", {}).get("quote")
                ):
                    continue
                if (
                    link.get("basis") not in ("ISSUER", "PRODUCT")
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(weight)
                    or not 0 < weight <= 1
                    or not link.get("report_available_at")
                    or moment(link["report_available_at"]) > end
                    or not re.fullmatch(r"[a-f0-9]{64}", link.get("report_hash") or "")
                ):
                    continue
                linked = True
                key = (link["code"], field, sign)
                # 正向、负向分别保留，不能相互抵消后伪装为无冲突；数量不进入特征。
                item = {
                    "id": event["id"],
                    "kind": kind,
                    "title": event["title"][:200],
                    "quote": quote[:600],
                    "published_date": published,
                    "available_at": event["available_at"],
                    "source_url": event.get("source_url"),
                    "source_hash": event["source_hash"],
                    "code": link["code"],
                    "weight": weight,
                    "relation": link["basis"],
                    "report_hash": link["report_hash"],
                    "report_available_at": link["report_available_at"],
                    "direction": direction,
                    "stage": stage,
                    "age_sessions": age,
                    "field": FEATURES[field],
                    "value": sign * weight * decay * 100,
                }
                previous = selected.get(key)
                if previous is None or (abs(item["value"]), item["available_at"], item["id"]) > (
                    abs(previous["value"]),
                    previous["available_at"],
                    previous["id"],
                ):
                    selected[key] = item
            if not linked:
                rejected["HOLDING_LINK_UNCONFIRMED"] += 1
        except (KeyError, TypeError, ValueError):
            rejected["INVALID_EVIDENCE"] += 1
    events = sorted(selected.values(), key=lambda x: (x["code"], x["field"], x["direction"]))
    overflow = len(events) > POLICY["max_events"]
    # 超限时不截取最利好的片段继续预测；完整性不足即拒绝方向输出。
    events = events[: POLICY["max_events"]]
    values = [sum(e["value"] for e in events if e["field"] == name) for name in FEATURES[4:]]
    return {
        "version": VERSION,
        "events": events,
        "values": values,
        "overflow": overflow,
        "positive": sum(max(e["value"], 0) for e in events),
        "negative": sum(max(-e["value"], 0) for e in events),
        "excluded": dict(rejected),
    }


def validate_model(model: dict) -> None:
    """非负系数保证负面事实不会因拟合被倒解释为利好；零截距不预设偏涨。"""
    if (
        model.get("feature_version") != VERSION
        or model.get("features") != FEATURES
        or model.get("fund_code") != "002112"
        or model.get("protocol") != "DIRECTION_1D_V2"
        or model.get("group_id") not in GROUPS.values()
        or model.get("policy") != POLICY
        or model.get("intercept") != 0
        or len(model.get("coef", [])) != len(FEATURES)
        or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in model["coef"])
        or model.get("classes") != ["DOWN", "UP"]
    ):
        raise ValueError("EVENT_MODEL_INVALID")
    if model.get("recipe_hash") and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != model["recipe_hash"]:
        raise ValueError("EVENT_RECIPE_CHANGED")


def transform(information: dict) -> np.ndarray:
    """原87项仅保留作历史证据；方向只读取4项行情和4项具体事件，不读条数/缺失标记/净值。"""
    raw, evidence = information["numeric"], information["event_evidence"]
    if evidence.get("version") != VERSION or len(raw) != 87 or len(evidence.get("values", [])) != 4:
        raise ValueError("EVENT_INPUT_INVALID")
    values = [0 if raw[i] is None else raw[i] * 100 for i in MARKET_INDEX] + evidence["values"]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("EVENT_INPUT_INVALID")
    return np.asarray(values, dtype=float)


def predict(model: dict, information: dict) -> dict:
    validate_model(model)
    x = transform(information)
    effects = x * np.asarray(model["coef"])
    up = float(expit(effects.sum()))
    direction = "UP" if up >= 0.5 else "DOWN"
    ev = information["event_evidence"]
    reasons = []
    if ev["overflow"]:
        reasons.append("EVENT_LIMIT")
    if not ev["events"]:
        reasons.append("NO_DIRECTIONAL_EVENT")
    if information["holding_coverage"] < POLICY["min_holding_coverage"] or any(
        information["numeric"][i] is None for i in MARKET_INDEX
    ):
        reasons.append("MARKET_INCOMPLETE")
    strongest = max(ev["positive"], ev["negative"])
    if strongest and min(ev["positive"], ev["negative"]) / strongest >= POLICY["conflict_ratio"]:
        reasons.append("EVENT_CONFLICT")
    event_effect = float(effects[4:].sum())
    sign = 1 if direction == "UP" else -1
    if ev["events"] and not any(model["coef"][FEATURES.index(e["field"])] > 0 for e in ev["events"]):
        reasons.append("MODEL_UNSUPPORTED")
    if ev["events"] and event_effect * sign <= 0:
        reasons.append("EVENT_MODEL_CONFLICT")
    if event_effect and x[0] * event_effect < 0:
        reasons.append("MARKET_EVENT_CONFLICT")
    if max(up, 1 - up) < POLICY["min_score"]:
        reasons.append("WEAK_SIGNAL")
    return {
        "direction": None if reasons else direction,
        "runner": None if reasons else "DOWN" if direction == "UP" else "UP",
        "status": "ABSTAINED" if reasons else "AVAILABLE",
        "score": None if reasons else max(up, 1 - up),
        "class_scores": None if reasons else {"UP": up, "DOWN": 1 - up, "FLAT": 0.0},
        "decision": {"policy": VERSION, "reason_codes": reasons},
        "effects": effects.tolist(),
    }


def explain(model: dict, information: dict) -> dict:
    result = predict(model, information)
    return {
        "direction": result["direction"],
        "referenceDirection": result["runner"],
        "status": result["status"],
        "decision": result["decision"],
        "intercept": 0.0,
        "factors": [
            {"feature": name, "value": float(value), "contribution": effect}
            for name, value, effect in zip(FEATURES, transform(information), result["effects"], strict=True)
        ],
    }
