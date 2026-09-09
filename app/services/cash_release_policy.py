"""服务器发布规则读取；请求不能提交规则正文、审批标记或审批依据。"""

from pathlib import Path

from app.schemas.cash_release_review import CashReleasePolicy
from app.services.historical_nav_storage import HistoricalNavStorageError

POLICY_PATH = Path(__file__).resolve().parents[1] / "data" / "cash_release_policy_v2.json"


def load_release_policy() -> CashReleasePolicy:
    """只读固定服务器文件，限制大小；损坏或缺失时不回退到宽松默认值。"""
    try:
        with POLICY_PATH.open("rb") as stream:
            raw = stream.read(16385)
        if len(raw) > 16384:
            raise ValueError("oversized cash release policy")
        return CashReleasePolicy.model_validate_json(raw)
    except (OSError, ValueError) as error:
        raise HistoricalNavStorageError(
            "CASH_RELEASE_POLICY_UNAVAILABLE", "服务器发布规则无法核验，暂不提供审查结果。", 503
        ) from error
