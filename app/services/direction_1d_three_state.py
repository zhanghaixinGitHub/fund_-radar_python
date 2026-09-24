"""一日三分类独立协议；旧二分类协议保持不变，净值相等才定义为持平。"""

import math

from app.services.direction_1d_protocol import FEATURE_VERSION, FEATURES

PROTOCOL = "DIRECTION_1D_V2"
TARGET = "UNIT_NAV_DIRECTION_THREE_STATE_V2"
SCHEMA = "DIRECTION_1D_EXPERIMENT_V2"
POLICY = "EXACT_UNIT_NAV_CHANGE_V1"
CLASSES = ("DOWN", "FLAT", "UP")
TIE_ORDER = ("FLAT", "UP", "DOWN")


def predict(model: dict, values: list[float]) -> dict:
    """严格核对三行系数，还原分类分数；分数未校准，不作为用户上涨概率。"""
    if (
        model.get("protocol") != PROTOCOL
        or model.get("target_definition") != TARGET
        or model.get("horizon") != 1
        or model.get("feature_version") != FEATURE_VERSION
        or model.get("features") != list(FEATURES)
        or model.get("direction_policy") != POLICY
        or model.get("class_order") != list(CLASSES)
        or model.get("tie_order") != list(TIE_ORDER)
    ):
        raise ValueError("MODEL_PROTOCOL_MISMATCH")
    vectors = [model.get("mean", []), model.get("scale", []), *model.get("coef", [])]
    intercept = model.get("intercept", [])
    if (
        len(vectors) != 5
        or any(len(v) != 7 for v in vectors)
        or len(intercept) != 3
        or len(values) != 7
        or any(
            not isinstance(v, (int, float)) or not math.isfinite(v)
            for row in [*vectors, intercept, values]
            for v in row
        )
        or any(v <= 0 for v in model["scale"])
    ):
        raise ValueError("INVALID_MODEL")
    normalized = [(x - m) / s for x, m, s in zip(values, model["mean"], model["scale"], strict=True)]
    logits = [
        b + sum(w * x for w, x in zip(row, normalized, strict=True))
        for row, b in zip(model["coef"], intercept, strict=True)
    ]
    if not all(math.isfinite(v) for v in logits):
        raise ValueError("INVALID_MODEL")
    exp = [math.exp(v - max(logits)) for v in logits]
    scores = {key: value / sum(exp) for key, value in zip(CLASSES, exp, strict=True)}
    ranked = sorted(CLASSES, key=lambda key: (-scores[key], TIE_ORDER.index(key)))
    return {"direction": ranked[0], "runner": ranked[1], "class_scores": scores, "score": scores[ranked[0]]}
