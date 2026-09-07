"""检查历史样本计算是否遵守“当时只能用当时已知信息”的规则。

测试直接调用 Python 函数，不请求 HTTP、不读写真实数据库，也不训练模型。
assert 表示“结果必须满足这个条件”，不满足时该测试会失败。
"""

from dataclasses import replace
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from uuid import UUID

import pytest
from app.services.historical_nav_samples import (
    HISTORICAL_NAV_SAMPLE_RULE_VERSION,
    MIN_FEATURE_NAV_OBSERVATIONS,
    HistoricalNavPoint,
    HistoricalNavSampleInput,
    build_historical_nav_samples,
)


def _input_with_points(points: tuple[HistoricalNavPoint, ...]) -> HistoricalNavSampleInput:
    """给测试净值补齐基金和来源信息；这里的同步编号是测试用的固定值。"""
    return HistoricalNavSampleInput(
        fund_code="008888",
        fund_type="STOCK",
        source_code="TUSHARE_PRO_FUND",
        source_sync_run_id=UUID("00000000-0000-0000-0000-000000000008"),
        nav_points=points,
    )


def _stage_one_points() -> tuple[HistoricalNavPoint, ...]:
    """保留阶段 1 的关键净值，其余日期和数值用于构造完整测试窗口。

    这不是一份完整的真实交易日数据：部分日期按自然日生成，包括周末。
    本函数用于核对已知公式和记录位置，不能用来检验真实交易日历。
    """
    nav_dates = tuple(
        date(2025, 5, 14) + timedelta(days=index) for index in range(MIN_FEATURE_NAV_OBSERVATIONS - 1)
    ) + (date(2025, 8, 7),) + tuple(
        date(2025, 8, 8) + timedelta(days=index) for index in range(19)
    ) + (date(2025, 9, 4),)
    values = [Decimal("1.0000") + Decimal(index) / Decimal("1000") for index in range(len(nav_dates))]
    values[0] = Decimal("1.09430000")
    values[40] = Decimal("1.05110000")
    values[55] = Decimal("1.11150000")
    values[60] = Decimal("1.12280000")
    values[80] = Decimal("1.28010000")
    return tuple(
        HistoricalNavPoint(
            nav_date=nav_date,
            ann_date=nav_date + timedelta(days=1),
            unit_nav=value,
            accumulated_nav=value,
        )
        for nav_date, value in zip(nav_dates, values, strict=True)
    )


def _sample_for(samples, as_of_date: date):
    """从逐日样本中取出指定净值日期的那一份结果。"""
    return next(sample for sample in samples if sample.as_of_date == as_of_date)


def test_historical_sample_reproduces_stage_one_label_without_putting_it_in_features() -> None:
    """008888 的阶段 1 示例仍是离线标签，不能进入 2025-08-08 的特征。"""
    samples = build_historical_nav_samples(_input_with_points(_stage_one_points()))
    sample = _sample_for(samples, date(2025, 8, 7))

    assert sample.eligibility_status == "SCORABLE"
    assert sample.available_at == date(2025, 8, 8)
    # 这里独立核对 5/20/60 日收益率及指标名称；其余四项引用计算结果本身，
    # 并没有独立验证它们的数值，因此不能把此断言当成所有指标公式的验证。
    assert sample.feature_payload["metrics"] == {
        "return_5d": "0.01016644",
        "return_20d": "0.06821425",
        "return_60d": "0.02604405",
        "volatility_20d": sample.feature_payload["metrics"]["volatility_20d"],
        "max_drawdown_60d": sample.feature_payload["metrics"]["max_drawdown_60d"],
        "relative_position_60d": sample.feature_payload["metrics"]["relative_position_60d"],
        "consecutive_decline_days": sample.feature_payload["metrics"]["consecutive_decline_days"],
    }
    assert sample.offline_label is not None
    assert sample.offline_label.label_end_date == date(2025, 9, 4)
    assert sample.offline_label.label_available_at == date(2025, 9, 5)
    assert sample.offline_label.future_return_20d.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) == Decimal(
        "0.140096"
    )
    assert sample.offline_label.label_up_20d == 1
    assert "label" not in sample.feature_payload
    assert "future_return_20d" not in sample.feature_payload


def test_historical_sample_excludes_a_past_nav_that_was_not_announced_by_the_cutoff() -> None:
    """业务日较早但公告日晚于截止日的净值，不能偷偷进入当时特征。"""
    start = date(2025, 1, 1)
    points = tuple(
        HistoricalNavPoint(
            nav_date=start + timedelta(days=index),
            ann_date=start + timedelta(days=index),
            unit_nav=Decimal(100 + index),
            accumulated_nav=Decimal(100 + index),
        )
        for index in range(MIN_FEATURE_NAV_OBSERVATIONS + 21)
    )
    anchor_index = MIN_FEATURE_NAV_OBSERVATIONS
    anchor = points[anchor_index]
    delayed_point = replace(
        points[anchor_index - 1],
        ann_date=anchor.ann_date + timedelta(days=1),
        unit_nav=Decimal("1000"),
        accumulated_nav=Decimal("1000"),
    )
    points_with_delayed_nav = points[: anchor_index - 1] + (delayed_point,) + points[anchor_index:]

    sample = _sample_for(build_historical_nav_samples(_input_with_points(points_with_delayed_nav)), anchor.nav_date)

    assert sample.eligibility_status == "SCORABLE"
    assert sample.feature_payload["input"]["usable_nav_observation_count"] == MIN_FEATURE_NAV_OBSERVATIONS
    metrics = sample.feature_payload["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["return_5d"] == "0.03870968"


def test_historical_sample_marks_missing_anchor_announcement_as_data_insufficient() -> None:
    """没有公告日时，净值不能被伪装成当日已知输入。"""
    points = list(_stage_one_points())
    points[60] = replace(points[60], ann_date=None)

    sample = _sample_for(build_historical_nav_samples(_input_with_points(tuple(points))), date(2025, 8, 7))

    assert sample.eligibility_status == "DATA_INSUFFICIENT"
    assert sample.unavailable_reason == "MISSING_NAV_ANNOUNCEMENT_DATE"
    assert sample.feature_payload["metrics"] is None
    assert sample.offline_label is None


def test_historical_sample_hash_is_deterministic_for_identical_source_data() -> None:
    """同一来源水位重跑时，特征哈希必须完全相同。"""
    input_record = _input_with_points(_stage_one_points())

    first = build_historical_nav_samples(input_record)
    second = build_historical_nav_samples(input_record)

    assert [sample.feature_hash for sample in first] == [sample.feature_hash for sample in second]


def test_future_missing_nav_cannot_change_the_past_basis_or_features() -> None:
    """未来累计净值缺失只影响答案可用性，不能使过去特征退回单位净值。"""
    # 故意让单位净值与累计净值不同，帮助观察是否错误切换了取值口径。
    points = tuple(replace(p, unit_nav=p.unit_nav / 2) for p in _stage_one_points())
    original = build_historical_nav_samples(_input_with_points(points))[60]
    changed = points[:-1] + (replace(points[-1], accumulated_nav=None),)
    result = build_historical_nav_samples(_input_with_points(changed))[60]

    assert result.nav_value_basis == original.nav_value_basis == "ACCUMULATED_NAV"
    assert result.feature_payload == original.feature_payload
    assert result.feature_hash == original.feature_hash
    assert result.eligibility_status == "DATA_INSUFFICIENT"
    assert result.offline_label is None
    assert result.unavailable_reason == "INVALID_LABEL_NAV_VALUE: nav_date=2025-09-04"


def test_adding_future_points_preserves_features_when_label_matures() -> None:
    """只有 61 条历史时也保留特征；补足未来 20 条后只新增答案。"""
    points = _stage_one_points()
    # 下标从 0 开始：[60] 是第 61 份样本，[:61] 是只保留前 61 条净值。
    pending = build_historical_nav_samples(_input_with_points(points[:61]))[60]
    mature = build_historical_nav_samples(_input_with_points(points))[60]

    assert pending.eligibility_status == "LABEL_NOT_MATURED"
    assert pending.feature_payload["quality"]["status"] == "SCORABLE"
    assert pending.offline_label is None
    assert pending.feature_payload == mature.feature_payload
    assert pending.feature_hash == mature.feature_hash
    assert mature.offline_label is not None


def test_future_outcome_can_turn_down_without_changing_past_features() -> None:
    """把未来结局从涨改成跌，只能改变标签，不能改变过去已生成的特征。"""
    points = _stage_one_points()
    before = build_historical_nav_samples(_input_with_points(points))[60]
    points = points[:-1] + (replace(points[-1], accumulated_nav=Decimal("0.9")),)
    after = build_historical_nav_samples(_input_with_points(points))[60]

    assert before.offline_label.label_up_20d == 1
    assert after.offline_label.label_up_20d == 0
    assert before.feature_hash == after.feature_hash


def test_label_waits_for_all_required_future_announcements() -> None:
    """未来窗口中间一条公告较晚，答案要等这条公告出来后才算完整可知。"""
    points = list(_stage_one_points())
    points[70] = replace(points[70], ann_date=date(2025, 9, 10))
    sample = build_historical_nav_samples(_input_with_points(tuple(points)))[60]

    assert sample.offline_label.label_available_at == date(2025, 9, 10)


def test_a_label_already_known_at_anchor_announcement_is_rejected() -> None:
    """起点公告拖得太晚、未来结局已经可知时，拒绝把它当正常预测样本。"""
    points = list(_stage_one_points())
    points[60] = replace(points[60], ann_date=date(2025, 9, 6))
    sample = build_historical_nav_samples(_input_with_points(tuple(points)))[60]

    assert sample.eligibility_status == "DATA_INSUFFICIENT"
    assert sample.unavailable_reason == "STALE_NAV_AT_CUTOFF"
    assert sample.offline_label is None


@pytest.mark.parametrize("value", [Decimal("0"), Decimal("NaN"), Decimal("Infinity")])
def test_invalid_future_nav_does_not_move_the_twentieth_observation(value: Decimal) -> None:
    """未来窗口有非法净值时报告不可用，不跳过坏记录另找一个第 20 日。"""
    # 参数化测试会分别检查 0、NaN（不是有效数字）、Infinity（无穷大）。
    points = list(_stage_one_points())
    points[70] = replace(points[70], accumulated_nav=value)
    sample = build_historical_nav_samples(_input_with_points(tuple(points)))[60]

    assert sample.eligibility_status == "DATA_INSUFFICIENT"
    assert sample.offline_label is None
    assert sample.unavailable_reason.startswith("INVALID_LABEL_NAV_VALUE")


@pytest.mark.parametrize("cutoff", [date(2025, 8, 9), date(2025, 8, 15)])
def test_partially_announced_future_rejects_stale_anchor(cutoff: date) -> None:
    """哪怕只公布了第一条后续净值，旧起点也已过时，不必等第20条答案揭晓。"""
    points = list(_stage_one_points())
    points[60] = replace(points[60], ann_date=cutoff)
    sample = build_historical_nav_samples(_input_with_points(tuple(points)))[60]

    assert points[-1].ann_date > cutoff
    assert sample.eligibility_status == "DATA_INSUFFICIENT"
    assert sample.unavailable_reason == "STALE_NAV_AT_CUTOFF"
    assert sample.as_of_date == date(2025, 8, 7)
    assert sample.available_at == cutoff
    assert sample.nav_value_basis == "UNDETERMINED"
    assert sample.feature_payload["metrics"] is None
    assert sample.feature_payload["quality"]["issues"] == ["STALE_NAV_AT_CUTOFF"]
    assert sample.offline_label is None
    assert sample.sample_rule_version == HISTORICAL_NAV_SAMPLE_RULE_VERSION


def test_stale_check_runs_before_label_history_shortage() -> None:
    """未来不足20条也不能掩盖旧起点：这里仅有1条后续净值，但它在截止日已经公布。"""
    points = list(_stage_one_points()[:62])
    points[60] = replace(points[60], ann_date=points[61].ann_date)
    sample = build_historical_nav_samples(_input_with_points(tuple(points)))[60]
    assert sample.eligibility_status == "DATA_INSUFFICIENT"
    assert sample.unavailable_reason == "STALE_NAV_AT_CUTOFF"


def test_stale_check_searches_beyond_the_twentieth_future_point() -> None:
    """前20条公告都较晚，但第21条先公布了；不能只看标签窗口而漏过旧起点。"""
    points = list(_stage_one_points())
    points[60] = replace(points[60], ann_date=date(2025, 9, 6))
    points[61:] = [replace(point, ann_date=date(2025, 9, 10)) for point in points[61:]]
    points.append(HistoricalNavPoint(date(2025, 9, 5), date(2025, 9, 6), Decimal("1.3"), Decimal("1.3")))
    sample = build_historical_nav_samples(_input_with_points(tuple(points)))[60]
    assert sample.unavailable_reason == "STALE_NAV_AT_CUTOFF"
    assert sample.offline_label is None


def test_long_announcement_gap_without_newer_known_nav_is_not_stale() -> None:
    """模拟长间隔但没有更新公告；不设置拍脑袋的自然日阈值，也不把终点改为公告日加20天。"""
    points = list(_stage_one_points())
    points[60] = replace(points[60], ann_date=date(2025, 8, 18))
    points[61:] = [replace(point, ann_date=max(point.ann_date, date(2025, 8, 20))) for point in points[61:]]
    sample = build_historical_nav_samples(_input_with_points(tuple(points)))[60]
    assert sample.eligibility_status == "SCORABLE"
    assert sample.offline_label.label_end_date == date(2025, 9, 4)


@pytest.mark.parametrize("announcement,reason", [
    (date(2025, 8, 8), "STALE_NAV_AT_CUTOFF"),
    (date(2025, 8, 9), None),
    (None, "MISSING_LABEL_ANNOUNCEMENT_DATE"),
    (date(2025, 8, 7), "INVALID_LABEL_ANNOUNCEMENT_DATE"),
])
def test_newer_announcement_boundary_and_invalid_dates(announcement, reason) -> None:
    """同日公告算已知，次日不算；缺失或早于净值日的公告不能充当已知证据。"""
    points = list(_stage_one_points())
    points[61] = replace(points[61], ann_date=announcement)
    sample = build_historical_nav_samples(_input_with_points(tuple(points)))[60]
    assert sample.unavailable_reason == reason


def test_repository_stale_fact_only_applies_to_its_own_anchor() -> None:
    """读库补充的过时日期不能误伤同一资料包里的其他辅助日期，也不能混入模型特征。"""
    input_record = _input_with_points(_stage_one_points())
    original = build_historical_nav_samples(input_record)
    marked = replace(input_record, stale_anchor_dates=frozenset({date(2025, 8, 7)}))
    changed = build_historical_nav_samples(marked)
    assert changed[60].unavailable_reason == "STALE_NAV_AT_CUTOFF"
    assert changed[:60] == original[:60]
    assert changed[61:] == original[61:]
    assert "stale_anchor_dates" not in changed[60].feature_payload
    assert "sample_rule_version" not in original[60].feature_payload


@pytest.mark.parametrize("announcement,reason", [
    (None, "MISSING_NAV_ANNOUNCEMENT_DATE"),
    (date(2025, 8, 6), "ANNOUNCEMENT_BEFORE_NAV_DATE"),
])
def test_invalid_anchor_date_has_priority_over_stale_fact(announcement, reason) -> None:
    """先说明起点自身日期错误，不用额外过时标记盖掉更基础的问题。"""
    points = list(_stage_one_points())
    points[60] = replace(points[60], ann_date=announcement)
    input_record = replace(_input_with_points(tuple(points)), stale_anchor_dates=frozenset({date(2025, 8, 7)}))
    assert build_historical_nav_samples(input_record)[60].unavailable_reason == reason
