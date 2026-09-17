"""只读过去成熟方向，估计各基金对IF日内两项已知行情的经验反应；不表示因果或持仓。"""

import math

import numpy as np

from app.services import direction_1d_sprint as base

WINDOW, MINIMUM, RIDGE, SHRINK, BOUND = 126, 63, 0.25, 126, 3.0
INPUTS = ("intraday_return_pct", "close_location")


def require(value, reason):
    if not value:
        raise ValueError("FUND_IF_RESPONSE_" + reason)


def prior(rows, cutoff):
    """先检查成熟时间再读取y，只取最近126条实际有输入的旧记录。

    每列独立计算固定收缩系数；当前行、未成熟行和未来行连标签都不读取。
    单基金重复目标日、混入其他基金、异常取值均拒绝，不能靠覆盖最后一条掩盖。
    """
    require(isinstance(cutoff, str) and len(cutoff) == 10, "CUTOFF_INVALID")
    eligible = [r for r in rows if r["u"] < cutoff and r["mature"] < cutoff]
    require(len({r["code"] for r in eligible}) <= 1, "MIXED_FUNDS")
    require(len({r["u"] for r in eligible}) == len(eligible), "DUPLICATE_TARGET")
    eligible = [r for r in eligible if r["market"]["futures_intraday"]["available"]]
    selected = sorted(eligible, key=lambda r: r["u"])[-WINDOW:]
    sample = []
    for row in selected:
        feature = row["market"]["futures_intraday"]
        x = [feature[k] for k in INPUTS]
        require(row["y"] in (0, 1) and type(row["y"]) is int, "LABEL_INVALID")
        require(feature["date"] == row["t"] < row["u"], "SOURCE_TIME_INVALID")
        require(
            all(type(v) in (int, float) and math.isfinite(v) for v in x) and abs(x[0]) <= 20 and abs(x[1]) <= 1,
            "INPUT_INVALID",
        )
        sample.append({"t": row["t"], "u": row["u"], "mature": row["mature"], "x": x, "y": row["y"]})
    count = len(sample)
    beta = [0.0, 0.0]
    if count >= MINIMUM:
        x = np.asarray([r["x"] for r in sample], dtype=float)
        y = 2 * np.asarray([r["y"] for r in sample], dtype=float) - 1
        centered = x - x.mean(axis=0)
        covariance = np.mean(centered * (y - y.mean())[:, None], axis=0)
        beta = np.clip(
            covariance / (np.mean(centered**2, axis=0) + RIDGE) * count / (count + SHRINK), -BOUND, BOUND
        ).tolist()
    return {
        "cutoff": cutoff,
        "available": count >= MINIMUM,
        "count": count,
        "beta": beta,
        "max_u": max((r["u"] for r in sample), default=None),
        "max_mature": max((r["mature"] for r in sample), default=None),
        "sample_hash": base.digest(sample),
    }


def feature(future, context, target):
    """把冻结旧统计应用于当期IF；不可把当期标签或未来上下文混入。"""
    require(context["cutoff"] <= target, "CONTEXT_FROM_FUTURE")
    count, beta = context["count"], context["beta"]
    require(
        type(count) is int and 0 <= count <= WINDOW and context["available"] is (count >= MINIMUM),
        "CONTEXT_COUNT_INVALID",
    )
    require(
        len(beta) == 2 and all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= BOUND for v in beta),
        "BETA_INVALID",
    )
    if count:
        require(
            context["max_u"] < context["cutoff"] and context["max_mature"] < context["cutoff"], "CONTEXT_NOT_MATURE"
        )
    else:
        require(context["max_u"] is None and context["max_mature"] is None, "EMPTY_CONTEXT_TIME_INVALID")
    if not context["available"]:
        require(beta == [0.0, 0.0], "MISSING_CONTEXT_NONZERO")
    require(future["date"] < target, "LIVE_SOURCE_FROM_FUTURE")
    available = future["available"] and context["available"]
    values = [0.0, 0.0]
    if available:
        x = [future[k] for k in INPUTS]
        require(
            all(type(v) in (int, float) and math.isfinite(v) for v in x) and abs(x[0]) <= 20 and abs(x[1]) <= 1,
            "LIVE_INPUT_INVALID",
        )
        values = [a * z for a, z in zip(x, beta, strict=True)]
    return {
        "available": available,
        "response_return": values[0],
        "response_location": values[1],
        "context_hash": base.digest(context),
        "context_cutoff": context["cutoff"],
        "max_prior_mature": context["max_mature"],
    }
