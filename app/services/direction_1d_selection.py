"""模型在真实预测时已训练、登记即可使用；不要求早于交易日预测窗口。"""

from datetime import datetime

ACTIVATION_POLICY = "AVAILABLE_AT_PREDICTION_V2"


def available_at_prediction(model: dict, now: datetime) -> bool:
    """只检查真实可用时间；文件、特征和模型摘要仍由原有完整性校验负责。"""
    return model["trained_at"] <= model["registered_at"] <= now < model["expires_at"]
