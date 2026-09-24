"""三分类收益口径：持平只表示小幅总回报，不表示缺资料、低信心或推理失败。"""

from decimal import Decimal, InvalidOperation

from app.services.prediction_contract import PredictionFailure, fingerprint, prediction_policy

TARGET = "NAV_ANCHORED_CASH_REINVESTED_THREE_STATE_V3"
TRI_TARGETS = {TARGET, "NEXT_EXECUTABLE_CASH_REINVESTED_THREE_STATE_V2"}
CLASSES = ("DOWN", "FLAT", "UP")
TRI_ADAPTERS = {"NAV_MOMENTUM_THREE_STATE_V2", "LOGISTIC_MULTICLASS_V2"}
BASELINE_ADAPTERS = {"NAV_MOMENTUM_V1", "NAV_MOMENTUM_THREE_STATE_V2"}
LABELS = {"UP": "上涨", "FLAT": "持平", "DOWN": "下跌", "NON_UP": "下跌或持平（旧版二分类）"}


def three_state(policy=None):
    return (policy or prediction_policy())["target_definition_id"] in TRI_TARGETS


def validate_rule(rule):
    """规则完整且单位明确；拒绝零、负数、NaN或缺少周期，绝不默认零阈值。"""
    try:
        assert rule["version"] == "RETURN_BAND_V1" and rule["unit"] == "RATIO"
        assert rule["boundary"] == "CLOSED_FLAT" and tuple(rule["classes"]) == CLASSES
        assert sorted(rule["tieBreakOrder"]) == sorted(CLASSES)
        assert isinstance(rule["minimumClassSamples"], int) and rule["minimumClassSamples"] >= 5
        for field in ("thresholds", "momentumThresholds"):
            assert set(rule[field]) == {"T5_V1", "T20_V1", "M6_V1"}
            for value in rule[field].values():
                number = Decimal(str(value))
                assert number.is_finite() and number > 0
    except (KeyError, TypeError, ValueError, AssertionError, InvalidOperation) as error:
        raise PredictionFailure(
            "DIRECTION_POLICY_INVALID", "CONFIG", "持平范围或分类规则不合法", retryable=False
        ) from error
    return rule


def direction_fields(horizon, policy=None):
    policy = policy or prediction_policy()
    if not three_state(policy):
        return {}
    rule = validate_rule(policy["direction"])
    return {
        "directionPolicyId": rule["version"],
        "directionPolicyHash": fingerprint(rule),
        "directionPolicySnapshot": rule,
        "flatThreshold": rule["thresholds"][horizon],
    }


def classify_return(value, threshold):
    """以十进制未舍入收益分类；正负边界包含在持平区间内。"""
    try:
        value, threshold = Decimal(str(value)), Decimal(str(threshold))
        if not value.is_finite() or not threshold.is_finite() or threshold <= 0:
            raise ValueError("NON_FINITE_OR_NON_POSITIVE")
    except (InvalidOperation, TypeError, ValueError) as error:
        raise PredictionFailure(
            "DIRECTION_VALUE_INVALID", "CLASSIFY", "收益或持平范围不合法", retryable=False
        ) from error
    return "UP" if value > threshold else "DOWN" if value < -threshold else "FLAT"


def classify(value, horizon, policy=None):
    policy = policy or prediction_policy()
    if three_state(policy):
        return classify_return(value, validate_rule(policy["direction"])["thresholds"][horizon])
    if policy["target_definition_id"] != "NEXT_EXECUTABLE_CASH_REINVESTED_DIRECTION_V1":
        raise PredictionFailure("TARGET_DEFINITION_MISMATCH", "CLASSIFY", "不支持的收益目标", retryable=False)
    return "UP" if Decimal(str(value)) > 0 else "NON_UP"


def validate_identity(value, horizon, policy=None):
    """模型、发布和预测使用同一核验入口，三态包不能靠改目标字符串冒充兼容。"""
    policy = policy or prediction_policy()
    fields = direction_fields(horizon, policy)
    if (
        value.get("targetDefinitionId") != policy["target_definition_id"]
        or value.get("horizonId", horizon) != horizon
        or any(value.get(k) != v for k, v in fields.items())
    ):
        raise PredictionFailure(
            "DIRECTION_POLICY_MISMATCH", "VALIDATION", "目标、周期或持平规则不兼容", retryable=False
        )
    return fields
