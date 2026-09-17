"""医疗行情的基金个体反应原型：仅用目标日之前已成熟的旧方向标签，禁止当前或未来标签泄漏。"""

from datetime import date

import numpy as np

from app.services import direction_1d_sprint as b


def context(rows, cutoff):
    """最近126个可用且已成熟观察估计单基金反应；不足63个时明确不可用。

    自变量是旧目标日对应的KURE同日对数收益百分数，因变量是已成熟方向的+1/-1。
    协方差除以方差加0.25，再乘n/(n+126)，限幅±3。它是固定收缩的历史条件反应，
    不声称为持仓权重、基金真实医疗仓位或概率。筛选先于读取y，当前/未来y不能参与统计。
    """
    date.fromisoformat(cutoff)
    if len({r["code"] for r in rows}) > 1:
        raise ValueError("HEALTH_RESPONSE_MULTIPLE_FUNDS")
    prior = sorted(
        (r for r in rows if r["u"] < cutoff and r["mature"] < cutoff and r["market"]["china_health"]["available"]),
        key=lambda r: r["u"],
    )[-126:]
    if len({r["u"] for r in prior}) != len(prior):
        raise ValueError("HEALTH_RESPONSE_DUPLICATE_DATE")
    samples = []
    for row in prior:
        x = row["market"]["china_health"]["intraday_log_pct"]
        if (
            type(x) not in (float, int)
            or not np.isfinite(x)
            or abs(x) > 20
            or type(row["y"]) is not int
            or row["y"] not in (0, 1)
        ):
            raise ValueError("HEALTH_RESPONSE_INVALID_PRIOR")
        samples.append({"u": row["u"], "mature": row["mature"], "x": x, "y": row["y"]})
    n = len(samples)
    beta = 0.0
    if n >= 63:
        x = np.asarray([r["x"] for r in samples])
        y = np.asarray([2 * r["y"] - 1 for r in samples])
        centered = x - x.mean()
        beta = float(np.clip(np.mean(centered * (y - y.mean())) / (np.mean(centered**2) + 0.25) * n / (n + 126), -3, 3))
    return {
        "cutoff": cutoff,
        "available": n >= 63,
        "count": n,
        "beta": beta,
        "max_u": max((r["u"] for r in samples), default=None),
        "max_mature": max((r["mature"] for r in samples), default=None),
        "sample_hash": b.digest(samples),
    }


def feature(health, prior, target):
    """历史每题用自己的截止日；真实预测可用较早冻结的上下文，但不得使用目标日新标签。"""
    if (
        prior["cutoff"] > target
        or type(prior["count"]) is not int
        or not 0 <= prior["count"] <= 126
        or prior["available"] is not (prior["count"] >= 63)
        or not np.isfinite(prior["beta"])
        or abs(prior["beta"]) > 3
    ):
        raise ValueError("HEALTH_RESPONSE_CONTEXT_INVALID")
    for name in ("max_u", "max_mature"):
        if prior[name] is not None and prior[name] >= prior["cutoff"]:
            raise ValueError("HEALTH_RESPONSE_CONTEXT_NOT_MATURE")
    available = health["available"] and prior["available"]
    return {
        "available": available,
        "response_signal": float(prior["beta"] * health["intraday_log_pct"]) if available else 0.0,
        "context_hash": b.digest(prior),
    }
