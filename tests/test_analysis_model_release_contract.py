"""M3 模型发布控制面静态契约测试。"""

from app.models.analysis import AnalysisModelRelease, ForecastResult
from sqlalchemy import CheckConstraint


def test_model_release_has_single_active_version_gate() -> None:
    """同一模型代码和基金类型在数据库层最多只能有一个 ACTIVE 发布。"""
    indexes = {index.name: index for index in AnalysisModelRelease.__table__.indexes}

    assert "uq_analysis_model_release_active" in indexes
    assert indexes["uq_analysis_model_release_active"].unique is True
    active_condition = indexes["uq_analysis_model_release_active"].dialect_options["postgresql"]["where"]

    assert "release_status = 'ACTIVE'" in str(active_condition)


def test_scored_forecast_requires_a_model_release_reference() -> None:
    """SCORED 结果必须关联批准发布；数据不足等非方向状态可保持为空。"""
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in ForecastResult.__table__.constraints
        if isinstance(constraint, CheckConstraint) and constraint.name
    }
    foreign_keys = {foreign_key.target_fullname for foreign_key in ForecastResult.__table__.foreign_keys}

    assert "ck_forecast_result_scored_release" in constraints
    assert "model_release_id IS NOT NULL" in constraints["ck_forecast_result_scored_release"]
    assert "analysis_model_release.model_release_id" in foreign_keys
