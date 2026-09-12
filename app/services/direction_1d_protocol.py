"""独立的单位净值一交易日协议；不改变任何20日研究常量。"""

import hashlib
import json
import math
from bisect import bisect_right
from datetime import date, datetime, time
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

from app.services.trading_calendar import load_calendar, load_current_calendar

ZONE = ZoneInfo("Asia/Shanghai")
PROTOCOL = "DIRECTION_1D_V1"
TARGET = "UNIT_NAV_DIRECTION_V1"
FEATURE_VERSION = "UNIT_NAV_7_T_CLOSE_V1"
FEATURES = (
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "max_drawdown_60d",
    "relative_position_60d",
    "consecutive_decline_days",
)
RECIPE = {"C": 1.0, "solver": "lbfgs", "tol": 1e-8, "max_iter": 1000, "random_state": 0}


def canonical(value) -> str:
    """唯一的跨服务原文字节格式；输出同时保留字符串，Java不重新序列化算hash。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def calendar() -> tuple[tuple[date, ...], str]:
    """仅在新协议中拼接两个已验哈希日历，支持2025/2026年界，不改旧日历。"""
    a, b = load_calendar(), load_current_calendar()
    return a.sessions + b.sessions, digest([a.content_hash, b.content_hash])


def window(now: datetime) -> dict:
    if now.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED")
    now = now.astimezone(ZONE)
    days, version = calendar()
    if not date(2021, 1, 1) <= now.date() <= date(2026, 12, 31):
        raise ValueError("CALENDAR_UNAVAILABLE")
    i = bisect_right(days, now.date()) - 1
    if days[i] == now.date() and now.time() < time(18):
        i -= 1
    if i < 60 or i + 1 >= len(days):
        raise ValueError("CALENDAR_UNAVAILABLE")
    base, target = days[i : i + 2]
    opened, deadline = datetime.combine(base, time(18), ZONE), datetime.combine(target, time(8, 30), ZONE)
    return {
        "base_nav_date": str(base),
        "target_nav_date": str(target),
        "window_open_at": opened.isoformat(),
        "deadline_at": deadline.isoformat(),
        "calendar_version": version,
        "status": "OPEN" if opened <= now < deadline else "MISSED_DEADLINE",
        "next_window_open_at": datetime.combine(target, time(18), ZONE).isoformat(),
    }


def input_days(base: date) -> tuple[date, ...]:
    days, _ = calendar()
    if base not in days:
        raise ValueError("BASE_NOT_SESSION")
    i = days.index(base)
    if i < 60:
        raise ValueError("HISTORY_TOO_SHORT")
    return days[i - 60 : i + 1]


def features(values) -> list[float]:
    """61条连续日历净值含T；全平相对位置无定义，不能补0。"""
    if len(values) != 61:
        raise ValueError("HISTORY_TOO_SHORT")
    v = [float(x) for x in values]
    if any(not math.isfinite(x) or x <= 0 for x in v):
        raise ValueError("INVALID_NAV")
    last = v[1:]
    low, high = min(last), max(last)
    if low == high:
        raise ValueError("FLAT_FEATURE_WINDOW")
    returns = [v[i] / v[i - 1] - 1 for i in range(41, 61)]
    avg = sum(returns) / 20
    peak, drawdown = last[0], 0.0
    for point in last:
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


def label(base, target) -> dict:
    """先用原始十进制比较，再舍入展示回报，持平作为NON_UP且单列。"""
    a, b = Decimal(str(base)), Decimal(str(target))
    if not a.is_finite() or not b.is_finite() or a <= 0 or b <= 0:
        raise ValueError("INVALID_NAV")
    return {
        "base_unit_nav": str(a),
        "target_unit_nav": str(b),
        "y": int(b > a),
        "actual_direction": "UP" if b > a else "DOWN" if b < a else "FLAT",
        "nav_return": str((b / a - 1).quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP)),
    }


def score(model: dict, x: list[float]) -> float:
    if (
        model.get("protocol") != PROTOCOL
        or model.get("horizon") != 1
        or model.get("target_definition") != TARGET
        or model.get("feature_version") != FEATURE_VERSION
        or model.get("features") != list(FEATURES)
    ):
        raise ValueError("MODEL_PROTOCOL_MISMATCH")
    for key in ("coef", "mean", "scale"):
        if len(model.get(key, [])) != 7 or any(not math.isfinite(v) for v in model[key]):
            raise ValueError("INVALID_MODEL")
    if len(x) != 7 or any(not math.isfinite(v) for v in x) or any(v <= 0 for v in model["scale"]):
        raise ValueError("INVALID_MODEL")
    z = model["intercept"] + sum(
        c * (v - m) / s for c, v, m, s in zip(model["coef"], x, model["mean"], model["scale"], strict=True)
    )
    if not math.isfinite(z):
        raise ValueError("INVALID_MODEL")
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
