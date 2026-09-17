"""个体市场敏感度的历史上下文与未来冻结快照，禁止读取方向标签。"""

import hashlib
from collections import defaultdict
from datetime import date

import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_hk as hk


def root():
    return base.ROOT / "round-36"


def market_returns(row):
    """按同一基准日期读取四个市场收益，单位百分数，港股收益由相对差还原。"""
    z = row["z"]
    if len(z) < 12 or not np.isfinite(z[:12]).all():
        raise ValueError("SENSITIVITY_MARKET_INPUT_INVALID")
    return np.asarray([z[6], z[7], z[6] + z[8], z[7] + z[9]], dtype=float)


def context(rows, cutoff):
    """只取同基金、目标和成熟日期均早于截止日的最近126条记录，不读取y。

    斜率以同期百分数收益的协方差除以市场方差加0.25，再乘n/(n+126)并限幅±3。
    未成熟记录先排除，其输入即使缺失或修订也不得影响当前上下文。
    """
    date.fromisoformat(cutoff)
    if len({r["code"] for r in rows}) > 1:
        raise ValueError("SENSITIVITY_CONTEXT_MULTIPLE_FUNDS")
    eligible = sorted((r for r in rows if r["u"] < cutoff and r["mature"] < cutoff), key=lambda r: r["u"])[-126:]
    if not eligible:
        return {"beta": [0.0] * 4, "count": 0, "available_fraction": 0.0, "max_u": None, "max_mature": None}
    if len({r["u"] for r in eligible}) != len(eligible):
        raise ValueError("SENSITIVITY_CONTEXT_DUPLICATE_DATE")
    markets = np.asarray([market_returns(r) for r in eligible])
    funds = np.asarray([r["x"][7] * 100 for r in eligible])
    if not np.isfinite(markets).all() or not np.isfinite(funds).all():
        raise ValueError("SENSITIVITY_CONTEXT_NONFINITE")
    centered = markets - markets.mean(axis=0)
    covariance = (centered * (funds - funds.mean())[:, None]).mean(axis=0)
    variance = (centered**2).mean(axis=0)
    beta = np.clip(covariance / (variance + 0.25) * len(eligible) / (len(eligible) + 126), -3, 3)
    return {
        "beta": beta.tolist(),
        "count": len(eligible),
        "available_fraction": len(eligible) / 126,
        "max_u": max(r["u"] for r in eligible),
        "max_mature": max(r["mature"] for r in eligible),
    }


def validate_context(prior, cutoff):
    """检查冻结上下文的单位、样本完整度和可见时间；无历史使用显式零状态。"""
    date.fromisoformat(cutoff)
    count = prior["count"]
    beta = prior["beta"]
    if (
        type(count) is not int
        or not 0 <= count <= 126
        or len(beta) != 4
        or not np.isfinite(beta).all()
        or max(abs(x) for x in beta) > 3
        or prior["available_fraction"] != count / 126
    ):
        raise ValueError("SENSITIVITY_CONTEXT_VALUES_INVALID")
    if count == 0:
        if beta != [0.0] * 4 or prior["max_u"] is not None or prior["max_mature"] is not None:
            raise ValueError("SENSITIVITY_EMPTY_CONTEXT_INVALID")
    else:
        for name in ("max_u", "max_mature"):
            day = prior[name]
            if not isinstance(day, str) or str(date.fromisoformat(day)) != day or day >= cutoff:
                raise ValueError("SENSITIVITY_CONTEXT_NOT_MATURE")


def extra_features(row, prior):
    validate_context(prior, row["u"])
    beta = np.asarray(prior["beta"])
    return beta.tolist() + (beta * market_returns(row)).tolist() + [prior["available_fraction"]]


def reference():
    """验证事前原型、输入摘要与快照关系；历史结果不能被另一份推导静默替换。"""
    proposal = base.read(root() / "proposal-before-training.json")
    result = base.read(root() / "input-feasibility.json")
    snapshot = base.read(root() / "current-context-feasibility.json")
    if (
        hashlib.sha256((root() / "context-feasibility.py").read_bytes()).hexdigest() != proposal["script_sha256"]
        or result["proposal_hash"] != base.digest(proposal)
        or snapshot["proposal_hash"] != base.digest(proposal)
        or proposal["source_hashes"] != hk.plan()["input_hashes"]
    ):
        raise ValueError("SENSITIVITY_REFERENCE_CHANGED")
    for name, expected in proposal["source_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected:
            raise ValueError("SENSITIVITY_HISTORY_INPUT_CHANGED")
    for prior in snapshot["contexts"].values():
        validate_context(prior, proposal["current_context_cutoff"])
    return proposal, result, snapshot


def dataset():
    """重新计算每个历史目标日的上下文，与事前完整特征摘要和当前快照精确对齐。"""
    proposal, expected, snapshot = reference()
    rows, proof = hk.dataset()
    by_code = defaultdict(list)
    for row in rows:
        by_code[row["code"]].append(row)
    output, derived = [], []
    for row in rows:
        prior = context(by_code[row["code"]], row["u"])
        extra = extra_features(row, prior)
        derived.append({"code": row["code"], "u": row["u"], "extra": extra})
        output.append(row | {"z": row["z"] + extra})
    actual_contexts = {code: context(values, proposal["current_context_cutoff"]) for code, values in by_code.items()}
    if base.digest(derived) != expected["feature_hash"] or actual_contexts != snapshot["contexts"]:
        raise ValueError("SENSITIVITY_PROTOTYPE_RESULT_CHANGED")
    return output, proof | {
        "sensitivity_feature_hash": base.digest(derived),
        "current_context_reference_hash": base.digest(snapshot),
        "context_cutoff": proposal["current_context_cutoff"],
        "context_features": 9,
    }
