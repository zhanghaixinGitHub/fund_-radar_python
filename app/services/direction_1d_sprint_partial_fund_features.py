"""共享线性模型的基金偏移特征，固定代码表、输入次序及个体收缩强度。"""

import hashlib

import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_sequence as sequence


def root():
    return base.ROOT / "round-40"


def validate_codes(codes):
    """代码表只取既有30只基金，按字符串升序固定；重排会改变模型列的业务含义。"""
    if (
        not isinstance(codes, (list, tuple))
        or len(codes) != 30
        or any(not isinstance(c, str) or len(c) != 6 or not c.isascii() or not c.isdigit() for c in codes)
        or list(codes) != sorted(set(codes))
    ):
        raise ValueError("PARTIAL_FUND_CODE_TABLE_INVALID")


def extras(z, code, codes):
    """前30列是0.25倍基金身份，后30列是身份乘SPX百分数收益（限幅±5）。"""
    validate_codes(codes)
    if code not in codes or len(z) != 12 or not np.isfinite(z).all():
        raise ValueError("PARTIAL_FUND_FEATURE_OR_CODE_INVALID")
    identity = [0.25 if selected == code else 0.0 for selected in codes]
    return identity + [value * float(np.clip(z[0], -5, 5)) for value in identity]


def normalize_training(x, weights):
    """只标准化原12项，基金身份和交互列保持原值，避免放大稀少基金的个体项。"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 72 or not np.isfinite(x).all():
        raise ValueError("PARTIAL_FUND_NORMALIZATION_INPUT_INVALID")
    shared, mean, scale = sequence.normalize_training(x[:, :12], weights)
    return np.concatenate((shared, x[:, 12:]), axis=1), list(mean) + [0.0] * 60, list(scale) + [1.0] * 60


def reference():
    """代码顺序、事前计算脚本及输入摘要都与方案绑定，不能重排身份后沿用原模型。"""
    preliminary = base.read(root() / "feature-plan.json")
    proposal = base.read(root() / "proposal-before-training.json")
    expected = base.read(root() / "input-feasibility.json")
    if (
        proposal["feature_plan_hash"] != base.digest(preliminary)
        or any(proposal[k] != v for k, v in preliminary.items() if k != "at")
        or expected["proposal_hash"] != base.digest(proposal)
        or hashlib.sha256((root() / "feature-feasibility.py").read_bytes()).hexdigest() != proposal["script_sha256"]
        or proposal["source_hashes"] != hk.plan()["input_hashes"]
    ):
        raise ValueError("PARTIAL_FUND_REFERENCE_CHANGED")
    validate_codes(proposal["codes"])
    for name, expected_hash in proposal["source_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected_hash:
            raise ValueError("PARTIAL_FUND_INPUT_CHANGED")
    return proposal, expected


def dataset():
    proposal, expected = reference()
    rows, proof = hk.dataset()
    if sorted({r["code"] for r in rows}) != proposal["codes"]:
        raise ValueError("PARTIAL_FUND_UNIVERSE_CHANGED")
    derived, output = [], []
    for row in rows:
        extra = extras(row["z"], row["code"], proposal["codes"])
        derived.append({"code": row["code"], "u": row["u"], "extra": extra})
        output.append(row | {"z": row["z"] + extra})
    if base.digest(derived) != expected["feature_hash"]:
        raise ValueError("PARTIAL_FUND_PROTOTYPE_MISMATCH")
    return output, proof | {"partial_fund_feature_hash": base.digest(derived), "funds": len(proposal["codes"])}
