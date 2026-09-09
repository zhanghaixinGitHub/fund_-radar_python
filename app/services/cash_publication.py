"""生成/读取共用的发布授权边界；现有研究不能签发正式凭证，旧ACTIVE不参与。"""

from datetime import datetime

from sqlalchemy.orm import Session

from app.repositories.cash_reinvestment_research import find_research
from app.schemas.cash_forecast import CashAuthorizedModel, CashForecastRequest
from app.services.cash_prediction_check import check_cash_prediction_in_session
from app.services.cash_reinvestment_research import restore_research
from app.services.cash_reinvestment_storage import cash_hash
from app.services.historical_nav_calibration import restore_calibrated_artifact
from app.services.historical_nav_storage import HistoricalNavStorageError


class CashPublicationUnavailable(Exception):
    """有证据表明不能发布，不是数据库故障；上层返回无概率状态。"""

    def __init__(self, codes: tuple[str, ...]):
        self.codes = codes
        super().__init__("cash model publication is not authorized")


def authorization_hash(grant: CashAuthorizedModel) -> str:
    return cash_hash(grant.model_dump(mode="json", exclude={"authorization_hash"}))


def validate_cash_authorization(grant: CashAuthorizedModel, request: CashForecastRequest) -> CashAuthorizedModel:
    """指纹只防错配，不是安全签名；授权必须来自服务器解析器，不能由客户端提交。"""
    grant = CashAuthorizedModel.model_validate_json(grant.model_dump_json())
    model = restore_calibrated_artifact(grant.artifact.model_dump_json())
    if (
        grant.authorization_hash != authorization_hash(grant)
        or grant.research_run_id != request.research_run_id
        or grant.report_hash != request.expected_report_hash
        or model.model_hash != request.expected_model_hash
        or request.fund_code not in model.base_model.train_counts_per_fund
        or request.fund_code not in model.calibrator.counts_per_fund
    ):
        raise HistoricalNavStorageError("CASH_AUTHORIZATION_MISMATCH", "发布凭证与基金、研究或模型不匹配。", 409)
    return grant


def resolve_cash_authorization(session: Session, request: CashForecastRequest, *, now: datetime) -> CashAuthorizedModel:
    """当前真实入口只能拒绝；正式证据驱动的凭证签发仍须实现，不能用研究分数代替。

    生成和读取只依赖这个服务器内边界。隔离测试可提供人工授权来验证其后续工程分支，
    但真实HTTP没有授权对象、强制通过或任意上传模型参数。
    """
    check = check_cash_prediction_in_session(session, request.check_request(), now=now)
    stored = restore_research(find_research(session, run_id=request.research_run_id))
    candidate = next((w.model for w in stored.report.windows if w.window.window_id == "VALIDATION_2024"), None)
    if candidate is not None and candidate.model_hash != request.expected_model_hash:
        raise HistoricalNavStorageError("MODEL_HASH_MISMATCH", "模型指纹与固定2024窗候选不一致。", 409)
    # 数据与独立测试缺口是实际门槛；即使未来比较结果全好，也不能隐式签发缺失的正式凭证。
    raise CashPublicationUnavailable(check.blocking_codes or ("FORMAL_RELEASE_NOT_ISSUED",))
