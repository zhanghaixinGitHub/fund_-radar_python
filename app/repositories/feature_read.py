"""M3 特征快照的受控只读查询。"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.analysis import FeatureSnapshot


def get_latest_feature_snapshot(
    session: Session, *, fund_code: str, feature_version: str
) -> FeatureSnapshot | None:
    """返回一只基金指定特征版本的最新快照；读取绝不触发重算。"""
    return session.scalar(
        select(FeatureSnapshot)
        .where(
            FeatureSnapshot.fund_code == fund_code,
            FeatureSnapshot.feature_version == feature_version,
        )
        .order_by(
            FeatureSnapshot.as_of_date.desc(),
            FeatureSnapshot.computed_at.desc(),
            FeatureSnapshot.feature_id.desc(),
        )
        .limit(1)
    )
