"""基金对资金流信号的历史下一日反应；仅用目标日前已成熟的旧标签。

与旧同期市场敏感度不同，这里的响应是严格过去记录的下一日原NAV方向。
当前/未来目标的标签不会参与计算，日期成熟过滤必须先于标签和来源访问。
"""

from datetime import date

import numpy as np

WINDOW = 126
VARIANCE_PRIOR = 0.0025
LIMIT = 3.0
KEYS = ("large_imbalance", "net_fraction")


def require(ok, reason):
    if not ok:
        raise ValueError("FUND_MONEYFLOW_CONTEXT_" + reason)


def context(rows, cutoff, points):
    """同基金最近126个已成熟且有当日资金流的旧问题，返回收缩斜率及可用历史数量。

    资金流为无量纲比例，旧y为0/1方向；方差先验0.0025、n/(n+126)收缩和±3限幅
    均在看到本轮分数前固定。未成熟、当前及未来题不读取标签，不进入缺失统计。
    """
    date.fromisoformat(cutoff)
    selected = []
    for row in rows:
        if row["u"] >= cutoff or row["mature"] >= cutoff:
            continue
        point = points.get(row["t"])
        if point is None or not point["available"]:
            continue
        selected.append((row, point))
    selected.sort(key=lambda pair: pair[0]["u"])
    selected = selected[-WINDOW:]
    require(len({row["code"] for row, _ in selected}) <= 1, "MULTIPLE_FUNDS")
    require(len({row["u"] for row, _ in selected}) == len(selected), "DUPLICATE_TARGET_DATE")
    if not selected:
        return {
            "beta": [0.0, 0.0],
            "count": 0,
            "available_fraction": 0.0,
            "max_t": None,
            "max_u": None,
            "max_mature": None,
        }
    xs, ys = [], []
    for row, point in selected:
        for key in ("t", "u", "mature"):
            require(str(date.fromisoformat(row[key])) == row[key] and row[key] < cutoff, "INVALID_PAST_DATE")
        require(point["date"] == row["t"] and row["t"] < row["u"], "SOURCE_ALIGNMENT")
        require(type(row["y"]) is int and row["y"] in (0, 1), "INVALID_MATURE_LABEL")
        x = [point[k] for k in KEYS]
        require(all(type(v) in (int, float) and np.isfinite(v) and abs(v) <= 1 for v in x), "INVALID_FEATURE")
        xs.append(x)
        ys.append(row["y"])
    x, y = np.asarray(xs), np.asarray(ys)
    centered = x - x.mean(axis=0)
    covariance = (centered * (y - y.mean())[:, None]).mean(axis=0)
    variance = (centered**2).mean(axis=0)
    beta = np.clip(covariance / (variance + VARIANCE_PRIOR) * len(y) / (len(y) + WINDOW), -LIMIT, LIMIT)
    return {
        "beta": beta.tolist(),
        "count": len(y),
        "available_fraction": len(y) / WINDOW,
        "max_t": max(row["t"] for row, _ in selected),
        "max_u": max(row["u"] for row, _ in selected),
        "max_mature": max(row["mature"] for row, _ in selected),
    }


def validate(prior, cutoff):
    date.fromisoformat(cutoff)
    count, beta = prior["count"], prior["beta"]
    require(type(count) is int and 0 <= count <= WINDOW, "COUNT_INVALID")
    require(len(beta) == 2 and np.isfinite(beta).all() and max(abs(v) for v in beta) <= LIMIT, "BETA_INVALID")
    require(prior["available_fraction"] == count / WINDOW, "FRACTION_CHANGED")
    if count == 0:
        require(beta == [0.0, 0.0] and all(prior[k] is None for k in ("max_t", "max_u", "max_mature")), "EMPTY_INVALID")
    else:
        for key in ("max_t", "max_u", "max_mature"):
            value = prior[key]
            require(isinstance(value, str) and str(date.fromisoformat(value)) == value and value < cutoff, "NOT_MATURE")


def extra_features(prior, cutoff, point):
    """五项个体输入；来源缺失返回None，让调用方执行明确回退，不能补为真实零流量。"""
    validate(prior, cutoff)
    if point is None or not point["available"]:
        return None
    require(point["date"] < cutoff, "CURRENT_FEATURE_NOT_PAST")
    flow = [point[k] for k in KEYS]
    require(all(type(v) in (int, float) and np.isfinite(v) and abs(v) <= 1 for v in flow), "INVALID_FEATURE")
    beta = prior["beta"]
    return beta + [beta[i] * flow[i] for i in range(2)] + [prior["available_fraction"]]
