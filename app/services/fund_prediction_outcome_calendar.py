"""到期核对复用原综合预测已经核准的估值日历，不退回普通境内产品假设。"""

from datetime import datetime

from sqlalchemy import text

from app.db.session import get_engine
from app.services.prediction_contract import PredictionFailure, fingerprint, policy_for_record, target_dates


def restored_calendar(prediction):
    """从同一预测的不可变快照重建原日历，并严格复核全部目标日期与摘要。

    只用于历史结果核对；不采集、不生成、不修改原预测。普通旧方法返回None，
    缺失、错基金、原文摘要或日历发生变化时保留明确等待，不套用最新基金资料。
    """
    if prediction.get("generationMethod") != "COMPREHENSIVE":
        return None
    try:
        ref = prediction["featureSnapshot"]
        if ref["fundCode"] != prediction["fundCode"]:
            raise ValueError("SNAPSHOT_FUND")
        with get_engine().connect() as c:
            c.execute(text("SET TRANSACTION READ ONLY"))
            row = c.execute(
                text("SELECT fund_code,knowledge_cutoff,content_hash,payload FROM prediction_feature_snapshot "
                     "WHERE snapshot_id=:id"), {"id": ref["inputSnapshotId"]}
            ).mappings().one()
        return verified_calendar(prediction, row)
    except PredictionFailure:
        raise
    except (KeyError, ValueError, TypeError) as error:
        raise PredictionFailure(
            "OUTCOME_ORIGINAL_CALENDAR_UNVERIFIED", "CALENDAR", "原综合预测的日历来源未通过核对，保留原预测等待",
            details={"fundCode": prediction.get("fundCode"), "reason": str(error)[:160]},
        ) from error


def verified_calendar(prediction, row):
    """验证原快照与原周期一致；独立函数方便验证错基金、篡改、跨期和日历修订。"""
    from app.services.comprehensive_multi_prediction import review_calendar
    from app.services.fund_prediction_snapshot_compat import checked_snapshot

    if row["fund_code"] != prediction["fundCode"] or row["content_hash"] != prediction["featureHash"]:
        raise ValueError("OUTCOME_SNAPSHOT_IDENTITY")
    source = checked_snapshot(row["payload"], row["content_hash"])
    data, review = source["data"], source["review"]
    cutoff = datetime.fromisoformat(data["as_of"])
    if (
        data["fund_code"] != prediction["fundCode"]
        or cutoff != row["knowledge_cutoff"]
        or cutoff > datetime.fromisoformat(prediction["generatedAt"])
        or fingerprint(review) != prediction["reviewHash"]
        or review.get("horizon_id") != prediction["horizonId"]
    ):
        raise ValueError("OUTCOME_ORIGINAL_PERIOD_OR_REVIEW")
    calendar = review_calendar(data, prediction["fundCode"], cutoff)
    dates = target_dates(calendar, cutoff, prediction["horizonId"], policy=policy_for_record(prediction))
    fields = ("baseNavDate", "startDate", "endDate", "horizonId", "calendarId", "calendarHash", "targetDefinitionId")
    if any(dates.get(k) != prediction.get(k) for k in fields):
        raise ValueError("OUTCOME_ORIGINAL_CALENDAR_CHANGED")
    return calendar
