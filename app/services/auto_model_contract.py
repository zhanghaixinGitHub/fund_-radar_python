"""自动选用的唯一协议、时间边界和冻结输入编码，不包含用户或交易信息。"""

import json
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal, DecimalException
from pathlib import Path
from zoneinfo import ZoneInfo

from app.services.prediction_contract import (
    PredictionFailure,
    ValuationCalendar,
    add_months,
    fingerprint,
    prediction_policy,
)

POLICY_FILE = Path(__file__).resolve().parents[1] / "data/auto_model_policy_v1.json"
ZONE = ZoneInfo("Asia/Shanghai")
IMPLEMENTATION_HASH = fingerprint(
    {
        name: (Path(__file__).parent / name).read_text(encoding="utf-8")
        for name in (
            "prediction_research.py",
            "prediction_models.py",
            "prediction_selection.py",
            "prediction_features.py",
            "prediction_replay_inputs.py",
            "auto_model_selection.py",
            "auto_model_store.py",
            "prediction_contract.py",
            "auto_model_contract.py",
            "prediction_generation.py",
        )
    }
)


def auto_policy():
    """不静默回退配置：无效配置中止研究，真实预测仍使用原有独立入口。"""
    try:
        policy = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
        assert policy["version"] == "AUTO_MODEL_SELECTION_V1"
        assert policy["timezone"] == "Asia/Shanghai"
        assert 0 <= policy["weeklyDay"] <= 6 and 0 <= policy["weeklyHour"] <= 23
        assert 1 <= policy["shardSize"] <= 50 and 1 <= policy["maximumBundles"] <= 4
        assert 1 <= policy["maximumCandidatesPerHorizon"] <= 3
        assert 5 <= policy["stride"] <= 20 and policy["maximumHistoryDays"] <= 1461
        assert 1 <= policy["maximumReplayDays"] <= 732 and policy["leaseSeconds"] >= 60
        assert 0 < policy["maximumWorkerSeconds"] <= 7200
        assert 1 <= policy["maximumSamples"] <= 200000
        assert policy["retryMinutes"] == [1, 5, 30]
        assert policy["candidateRecipes"] == ["TOTAL_RETURN_LOGISTIC_V1"]
        assert 0 <= policy["tolerance"] <= 1e-6 and 0 <= policy["maxDrawdownDeterioration"] <= 0.02
        assert all(policy[k] > 0 for k in ("trainingMonths", "validationMonths", "selectionMonths"))
        fee = policy["executionPolicy"]
        assert Decimal(fee["initialCash"]) > 0 and 0 <= Decimal(fee["buyFee"]) < 1
        assert fee["confirmationSessions"] >= 1 and fee["cashArrivalSessions"] >= 1
        end = 0
        for tier in fee["redemptionFees"]:
            assert tier["minDays"] == end and 0 <= Decimal(tier["rate"]) < 1
            end = tier["maxDays"] + 1 if tier["maxDays"] is not None else -1
        assert end == -1
    except (OSError, ValueError, KeyError, TypeError, AssertionError, DecimalException) as error:
        raise PredictionFailure("AUTO_POLICY_INVALID", "CONFIG", "自动改进配置缺失或无效", retryable=False) from error
    return policy | {"predictionPolicy": prediction_policy(), "implementationHash": IMPLEMENTATION_HASH}


def weekly_slot(now, policy):
    """恢复后只补最近一个周日02:00，不补造过去多份周任务。"""
    local = now.astimezone(ZONE)
    day = local.date() - timedelta(days=(local.weekday() - policy["weeklyDay"]) % 7)
    slot = datetime.combine(day, time(policy["weeklyHour"]), ZONE)
    return (slot if slot <= local else slot - timedelta(days=7)).isoformat()


def rolling_window(now, policy, earliest=None):
    end = now.astimezone(ZONE).date() - timedelta(days=1)
    validation = add_months(end, -policy["selectionMonths"])
    train = add_months(validation, -policy["validationMonths"])
    if (end - validation).days > policy["maximumReplayDays"]:
        raise PredictionFailure("AUTO_POLICY_INVALID", "CONFIG", "选优窗口超过冻结回放预算", retryable=False)
    start = max(add_months(train, -policy["trainingMonths"]), end - timedelta(days=policy["maximumHistoryDays"]))
    if earliest and earliest > start:
        start = earliest
    if start >= train:
        # 缩窗仍保留三段和真实成熟边界；不能将半年未成熟标签用于拟合。
        span = (end - start).days
        if span < 180:
            raise PredictionFailure(
                "AUTO_DATA_INSUFFICIENT", "WAITING_DATA", "历史不足180日，继续现有预测等待新数据", retryable=False
            )
        train, validation = start + timedelta(days=span // 2), start + timedelta(days=span * 3 // 4)
    return dict(trainStart=str(start), trainEnd=str(train), validationEnd=str(validation), selectionEnd=str(end))


def freeze_value(value):
    """带类型的JSON用于中断恢复，不能重新读取已修订数据替换原冻结输入。"""
    if isinstance(value, ValuationCalendar):
        return {"_type": "calendar", "value": freeze_value(asdict(value))}
    if isinstance(value, (datetime, date, time, Decimal)):
        return {"_type": type(value).__name__, "value": str(value)}
    if isinstance(value, dict):
        return {str(k): freeze_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [freeze_value(v) for v in value]
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


def thaw_value(value):
    if isinstance(value, list):
        return [thaw_value(v) for v in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("_type")
    if tag == "calendar":
        fields = thaw_value(value["value"])
        fields["sessions"] = tuple(fields["sessions"])
        return ValuationCalendar(**fields)
    if tag in {"datetime", "date", "time", "Decimal"}:
        return {
            "datetime": datetime.fromisoformat,
            "date": date.fromisoformat,
            "time": time.fromisoformat,
            "Decimal": Decimal,
        }[tag](value["value"])
    return {k: thaw_value(v) for k, v in value.items()}


def protocol_hash():
    return fingerprint(auto_policy())
