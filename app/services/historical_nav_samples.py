"""把一只基金的历史净值整理成“已知条件 + 后来答案”，此处不训练模型。

阅读顺序：先看下面4个数据对象，再看 build_historical_nav_samples → _build_one_sample。
一条样本可以理解为一道练习题：feature_payload 是题目，offline_label 是后来揭晓的答案。
例如008888的起点净值属于2025-08-07，8月8日才公布；20条净值之后的结果只能作为答案。
本文件只计算传入的数据，不访问数据库；数据库读取在 historical_nav_preview.py 中编排。

常见语法：Decimal 是十进制数，用于避免普通浮点数的小数误差；tuple 是固定顺序的序列；
T | None 表示“有一个T类型的值，或者没有值”。下面的dataclass是内存数据对象，不是数据库表；
frozen=True 禁止直接给对象字段重新赋值，但不意味着内部字典也自动不可修改。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from uuid import UUID

# 本轮只接受股票型基金。这个标记用于判断范围，不是模型的输入特征。
STOCK_FUND_TYPE = "STOCK"
# 特征规则版本：V2修正了“未来缺值导致过去更换净值口径”的问题，不是训练出的模型版本。
HISTORICAL_NAV_FEATURE_VERSION = "M3_STOCK_HISTORICAL_NAV_FEATURE_V2"
# 标签规则版本：后第20条净值相对起点上涨记1，否则记0。
HISTORICAL_NAV_LABEL_VERSION = "FUTURE_UP_20D_V1"
# 按净值序列向后数20条，而不是把日历日期直接加20天。
HORIZON_TRADING_DAYS = 20
# 特征分别回看5、20、60个净值区间；一个区间由相邻的两个净值点构成。
RETURN_LOOKBACK_DAYS = (5, 20, 60)
# 20个日收益需要21个净值点，用来衡量这段历史中涨跌幅的波动程度。
VOLATILITY_LOOKBACK_DAYS = 20
# 当前回撤和相对位置取最近60条净值；与“60个区间收益需要61条”区分开。
MAX_DRAWDOWN_LOOKBACK_DAYS = 60
# 起点也要占一条：计算60个区间收益至少需要61条净值。这不是模型训练样本量门槛。
MIN_FEATURE_NAV_OBSERVATIONS = max(RETURN_LOOKBACK_DAYS) + 1
# 特征小数统一保留8位；0.01000000表示1%，并非0.01%。
_METRIC_QUANTUM = Decimal("0.00000001")


@dataclass(frozen=True)
class HistoricalNavPoint:
    """一条净值记录；四个字段对应nav_daily同名列，也可以由POST请求提供。"""

    # 净值属于哪一天，例如2025-08-07；不代表这天已经能够看到这个数。
    nav_date: date
    # 来源标注的公告日，例如2025-08-08。本版按公告日结束后可用解释；缺失则无法证明当时可见。
    ann_date: date | None
    # 单位净值，即来源给出的一份基金的净值；累计净值窗口不完整时作为统一回退口径。
    unit_nav: Decimal
    # 来源给出的累计净值。本模块不自行推算；None表示缺失，不能当成0使用。
    accumulated_nav: Decimal | None


@dataclass(frozen=True)
class HistoricalNavSampleInput:
    """一次计算的输入包：基金身份、来源说明，以及按日期从早到晚排列的净值。"""

    # 基金份额代码，如008888；用来标识样本归属，不能靠代码本身猜测涨跌。
    fund_code: str
    # 基金品类，本轮必须为STOCK；GET读数据库确认，POST由调用者传入。
    fund_type: str
    # 来源编码，如TUSHARE_PRO_FUND或SYNTHETIC_DEMO；是溯源说明，不是预测特征。
    source_code: str
    # 来源最近成功净值同步的运行ID（UUID唯一标识）；手工数据可为空。
    # 它说明当前来源水位，不等于每条净值都来自这次同步，也不证明恢复了历史修订版本。
    source_sync_run_id: UUID | None
    # 同一基金、同一来源的净值序列。可同时包含题目所需历史和答案所需未来，构建时会分开。
    nav_points: tuple[HistoricalNavPoint, ...]


@dataclass(frozen=True)
class OfflineDirectionLabel:
    """历史样本的标准答案，用于后续训练/回测，不能作为该起点的输入特征。"""

    # 答案的计算规则版本，方便核对不同样本是否使用同一种“上涨”定义。
    label_version: str
    # 从起点向后走多少个净值区间，本轮固定20，起点自身为第0条。
    horizon_trading_days: int
    # 后第20条净值的业务日期；示例为2025-09-04，不是答案公开的日期。
    label_end_date: date
    # 答案依赖的未来记录全部公告完成的日期，取它们最晚的公告日。
    # 后续按时间划分训练集时，还必须判断该答案在训练截止时是否已经可得。
    label_available_at: date
    # 后来实际发生的区间收益：终点净值/起点净值-1。0.140096…约为14.0096%，不是预测值。
    future_return_20d: Decimal
    # 上涨记1；持平或下跌都记0。因此0准确含义是“非上涨”，不一定是下跌。
    label_up_20d: int


@dataclass(frozen=True)
class HistoricalNavSample:
    """一个净值业务日对应的练习题及答案；GET接口直接返回此对象。"""

    # 这条样本属于哪只基金，与请求中的fundCode对应。
    fund_code: str
    # 本条样本的起点净值业务日，与GET参数asOfDate对应，例如2025-08-07。
    as_of_date: date
    # 起点公告日，也是筛选当时已知历史的截止日，例如2025-08-08；不是接口调用时间。
    available_at: date | None
    # 本条样本使用的净值口径：ACCUMULATED_NAV累计；UNIT_NAV单位；UNDETERMINED尚无法确定。
    # 特征与标签必须使用同一口径，不能因为未来缺累计净值就把过去改成单位净值。
    nav_value_basis: str
    # 特征的计算规则版本，不是机器学习模型版本。
    feature_version: str
    # 整条样本状态：SCORABLE=特征和答案均可用；DATA_INSUFFICIENT=输入或标签数据不合格；
    # LABEL_NOT_MATURED=特征已有，但未来记录/公告尚不齐。SCORABLE不等于模型已通过发布。
    eligibility_status: str
    # 不能形成完整样本的原因，如NAV_HISTORY_SHORTAGE；正常完整样本为None。
    unavailable_reason: str | None
    # 特征内容包：source来源、input日期/数量、feature_definitions指标说明、quality质量、metrics数值。
    # 真正供后续模型使用的是metrics中的指标，来源运行ID等只是追溯信息；这里不放答案。
    feature_payload: dict[str, object]
    # 上述内容包的SHA-256校验指纹；内容和来源水位相同则指纹相同，不是评分或概率。
    feature_hash: str
    # 单独存放的后来答案；无答案时为None。标签不可用时，已有的合格特征仍可能保留。
    offline_label: OfflineDirectionLabel | None


def build_historical_nav_samples(input_record: HistoricalNavSampleInput) -> tuple[HistoricalNavSample, ...]:
    """为一只股票型基金逐日生成历史样本，不读库、不写库也不训练模型。

    Args:
        input_record: 同一来源水位、按 ``nav_date`` 递增的基金净值序列。

    Returns:
        每个净值业务日对应的一条样本。只有 ``SCORABLE`` 样本带有 ``offline_label``，
        后续训练或回测只能消费这类样本。

    Raises:
        ValueError: 基金类型不是股票型，或净值业务日不是严格递增时抛出。
    """
    # 第一步：检查输入范围和顺序。错误顺序会使“向前/向后数20条”失去业务意义。
    if input_record.fund_type != STOCK_FUND_TYPE:
        raise ValueError(f"unsupported V1 fund_type={input_record.fund_type}")
    _validate_nav_dates(input_record.nav_points)
    # 第二步：把每条记录轮流当起点。anchor_index是起点在序列中的位置，从0开始。
    # 此处保留不足样本及原因，不会默默丢掉；GET调用方稍后只挑用户指定的那一天返回。
    return tuple(
        _build_one_sample(
            input_record=input_record,
            anchor_index=anchor_index,
        )
        for anchor_index in range(len(input_record.nav_points))
    )


def _validate_nav_dates(nav_points: tuple[HistoricalNavPoint, ...]) -> None:
    """拒绝无序或重复净值日，避免把日历顺序误当成有效净值日顺序。"""
    # 将第0条与第1条、第1条与第2条依次比较；后一天必须严格晚于前一天。
    # 右侧切片少一条，所以strict=False允许配对到较短序列结束。
    if any(
        current.nav_date <= previous.nav_date
        for previous, current in zip(nav_points, nav_points[1:], strict=False)
    ):
        raise ValueError("nav_points must be strictly ordered by nav_date")


def _select_nav_value_basis(nav_points: tuple[HistoricalNavPoint, ...]) -> str:
    """只检查当时可见的特征窗口，未来记录不得决定过去使用的净值口径。"""
    # 只有历史窗口内每个累计净值都有效，才整段使用累计净值；否则整段使用单位净值。
    # “一段内统一”很重要：不能今天用累计、昨天用单位，也不能检查未来数据来做此选择。
    if nav_points and all(_is_positive_nav(point.accumulated_nav) for point in nav_points):
        return "ACCUMULATED_NAV"
    return "UNIT_NAV"


def _build_one_sample(
    *,
    input_record: HistoricalNavSampleInput,
    anchor_index: int,
) -> HistoricalNavSample:
    """围绕一个起点依次检查公告、筛选历史、计算特征，最后尝试补上答案。"""
    points = input_record.nav_points
    anchor = points[anchor_index]
    nav_value_basis = "UNDETERMINED"
    # 1. 先确定“站在哪一天看历史”。没公告日不能猜成净值日，公告早于净值日也不接受。
    if anchor.ann_date is None:
        return _unavailable_sample(
            input_record=input_record,
            anchor=anchor,
            nav_value_basis=nav_value_basis,
            status="DATA_INSUFFICIENT",
            reason="MISSING_NAV_ANNOUNCEMENT_DATE",
        )
    if anchor.ann_date < anchor.nav_date:
        return _unavailable_sample(
            input_record=input_record,
            anchor=anchor,
            nav_value_basis=nav_value_basis,
            status="DATA_INSUFFICIENT",
            reason="ANNOUNCEMENT_BEFORE_NAV_DATE",
        )

    # 2. 只看业务日在起点及之前、且到起点公告日已经公布的记录。
    # 业务日虽然更早但公告更晚的记录，此时仍不可见。切片的+1是为了包含起点本身。
    known_points = tuple(
        point
        for point in points[: anchor_index + 1]
        if point.ann_date is not None and point.nav_date <= point.ann_date <= anchor.ann_date
    )
    # 取最后61条可见记录，既能算60区间收益，也限制每条题目的回看范围。
    # 此处按可见记录计数，不会自行补出日历上的缺失净值日。
    feature_points = known_points[-MIN_FEATURE_NAV_OBSERVATIONS:]
    nav_value_basis = _select_nav_value_basis(feature_points)
    if len(known_points) < MIN_FEATURE_NAV_OBSERVATIONS:
        return _unavailable_sample(
            input_record=input_record,
            anchor=anchor,
            nav_value_basis=nav_value_basis,
            status="DATA_INSUFFICIENT",
            reason=(
                "NAV_HISTORY_SHORTAGE: "
                f"observed={len(known_points)}, required={MIN_FEATURE_NAV_OBSERVATIONS}"
            ),
        )

    # 3. 日期筛选通过后，再按已固定的口径取数。找出第一条非法值，把日期写进拒绝原因。
    feature_values = tuple(_nav_value(point, nav_value_basis) for point in feature_points)
    invalid_feature_point = next(
        (point for point, value in zip(feature_points, feature_values, strict=True) if not _is_positive_nav(value)),
        None,
    )
    if invalid_feature_point is not None:
        return _unavailable_sample(
            input_record=input_record,
            anchor=anchor,
            nav_value_basis=nav_value_basis,
            status="DATA_INSUFFICIENT",
            reason=f"INVALID_NAV_VALUE: nav_date={invalid_feature_point.nav_date.isoformat()}",
        )
    # 上面已拒绝所有None/非法值；这里的筛选只提取已确认有效的数，不是用“跳过坏值”凑窗口。
    valid_feature_values = tuple(value for value in feature_values if value is not None)
    metrics = _build_metrics(valid_feature_values)
    if metrics is None:
        return _unavailable_sample(
            input_record=input_record,
            anchor=anchor,
            nav_value_basis=nav_value_basis,
            status="DATA_INSUFFICIENT",
            reason="NAV_POSITION_UNDEFINED: flat_60d_window",
        )

    # 4. 特征先定稿；未来标签缺失、修订或补齐，均不得改写这个payload和哈希。
    feature_payload = _scorable_feature_payload(
        input_record=input_record,
        anchor=anchor,
        nav_value_basis=nav_value_basis,
        metrics=metrics,
    )
    # 先暂记“标签未成熟”，下一步检查通过才升级为完整可用样本。
    sample = HistoricalNavSample(
        fund_code=input_record.fund_code,
        as_of_date=anchor.nav_date,
        available_at=anchor.ann_date,
        nav_value_basis=nav_value_basis,
        feature_version=HISTORICAL_NAV_FEATURE_VERSION,
        eligibility_status="LABEL_NOT_MATURED",
        unavailable_reason=None,
        feature_payload=feature_payload,
        feature_hash=_feature_hash(feature_payload),
        offline_label=None,
    )
    # 5. 切片包含起点+未来20条，共21条；Python切片右边界不包含在结果中，所以需要+1。
    return _attach_offline_label(sample, points[anchor_index : anchor_index + HORIZON_TRADING_DAYS + 1])


def _attach_offline_label(
    sample: HistoricalNavSample, label_points: tuple[HistoricalNavPoint, ...]
) -> HistoricalNavSample:
    """按起点已确定的口径生成答案；仅更新总体状态、拒绝原因或标签，不改特征。

    label_points[0]是起点，label_points[20]才是终点。
    replace会创建一个新对象，并保留没指定修改的字段；不会就地修改原来的冻结数据对象。
    """
    # 少于21条说明后续20条没凑齐。保留已有特征，返回“还没有完整答案”。
    if len(label_points) <= HORIZON_TRADING_DAYS:
        return replace(
            sample,
            unavailable_reason=(
                f"LABEL_HISTORY_SHORTAGE: observed={len(label_points) - 1}, required={HORIZON_TRADING_DAYS}"
            ),
        )
    # 起点公告在特征阶段已经检查，这里检查后续每条记录的公告日期与数值。
    future_points = label_points[1:]
    if any(point.ann_date is None for point in future_points):
        return replace(sample, unavailable_reason="MISSING_LABEL_ANNOUNCEMENT_DATE")
    if any(point.ann_date < point.nav_date for point in future_points if point.ann_date is not None):
        return replace(
            sample, eligibility_status="DATA_INSUFFICIENT", unavailable_reason="INVALID_LABEL_ANNOUNCEMENT_DATE"
        )
    # 中间有坏值也拒收整段，不跳过坏值后多取一天；否则“20日”的终点会被悄悄改变。
    for point in future_points:
        if not _is_positive_nav(_nav_value(point, sample.nav_value_basis)):
            return replace(
                sample,
                eligibility_status="DATA_INSUFFICIENT",
                unavailable_reason=f"INVALID_LABEL_NAV_VALUE: nav_date={point.nav_date.isoformat()}",
            )
    label_end = label_points[-1]
    # 若起点公告严重滞后，以至终点已公告，这条题目的答案在当时已经揭晓，必须拒收。
    if label_end.ann_date <= sample.available_at:
        return replace(
            sample, eligibility_status="DATA_INSUFFICIENT", unavailable_reason="LABEL_ALREADY_KNOWN_AT_CUTOFF"
        )
    anchor_value = _nav_value(label_points[0], sample.nav_value_basis)
    label_end_value = _nav_value(label_end, sample.nav_value_basis)
    # 示例：1.2801 / 1.1228 - 1 = 0.140096…；这就是后来实际发生的14.0096%涨幅。
    future_return = label_end_value / anchor_value - Decimal("1")
    return replace(
        sample,
        eligibility_status="SCORABLE",
        offline_label=OfflineDirectionLabel(
            label_version=HISTORICAL_NAV_LABEL_VERSION,
            horizon_trading_days=HORIZON_TRADING_DAYS,
            label_end_date=label_end.nav_date,
            # 中间记录可能比终点更晚公告，故取全窗口最晚公告日，而不只看终点公告日。
            label_available_at=max(point.ann_date for point in future_points if point.ann_date is not None),
            future_return_20d=future_return,
            label_up_20d=int(future_return > 0),
        ),
    )


def _is_positive_nav(value: Decimal | None) -> bool:
    """净值必须有值、有限且大于0；NaN（非数值）和Infinity（无穷大）都不能参与除法。"""
    return value is not None and value.is_finite() and value > 0


def _build_metrics(nav_values: tuple[Decimal, ...]) -> dict[str, str | int] | None:
    """计算历史可得的净值特征；平坦窗口的位置指标没有含义，明确返回缺失。"""
    # 收益用完整61条数60个区间；回撤和相对位置按当前规则只看最后60条。
    recent_60_values = nav_values[-MAX_DRAWDOWN_LOOKBACK_DAYS:]
    window_min = min(recent_60_values)
    window_max = max(recent_60_values)
    # 若整段净值相同，下面“相对位置”的分母为0；本版因此拒收，而不是伪造位置0或0.5。
    # 这是位置指标的缺失，不表示平稳行情的真实收益0或波动0就是非法数值。
    if window_min == window_max:
        return None
    try:
        return {
            # 过去5/20/60个区间的首尾收益，都是站在起点时已经能够算出的信息。
            "return_5d": _decimal_text(_period_return(nav_values, 5)),
            "return_20d": _decimal_text(_period_return(nav_values, 20)),
            "return_60d": _decimal_text(_period_return(nav_values, 60)),
            # 过去20次日收益的波动程度；输入21条净值才能得到20次相邻变化。
            "volatility_20d": _decimal_text(_volatility(nav_values[-(VOLATILITY_LOOKBACK_DAYS + 1) :])),
            # 这段历史从先前高点向下跌得最深的一次；-0.05表示回撤5%。
            "max_drawdown_60d": _decimal_text(_max_drawdown(recent_60_values)),
            # (当前值-窗口最低值)/(窗口最高值-最低值)；0为窗口最低，1为窗口最高，不是上涨概率。
            "relative_position_60d": _decimal_text((nav_values[-1] - window_min) / (window_max - window_min)),
            # 从末尾向前连续下跌了几个区间，最多只能看到当前61条里的60个区间。
            "consecutive_decline_days": _consecutive_decline_days(nav_values),
        }
    except InvalidOperation as error:
        raise ValueError("historical NAV metrics cannot be represented") from error


def _period_return(nav_values: tuple[Decimal, ...], interval_count: int) -> Decimal:
    """计算首尾相隔固定有效净值日数的收益率。"""
    # [-1]是最后一条，[-21]是往前20个区间的起点；上游已保证长度足够且分母大于0。
    return nav_values[-1] / nav_values[-(interval_count + 1)] - Decimal("1")


def _volatility(nav_values: tuple[Decimal, ...]) -> Decimal:
    """计算连续 20 个净值区间的总体标准差，不年化。"""
    # 先把净值变成每日收益：第二天/第一天-1，第三天/第二天-1，依次得到20个数。
    returns = tuple(
        current / previous - Decimal("1")
        for previous, current in zip(nav_values[:-1], nav_values[1:], strict=True)
    )
    # mean是平均日收益；variance是各日收益偏离平均值的平方平均数。
    # 再开平方得到标准差：数值越大，说明这段历史每日涨跌幅越不稳定。
    # 这里分母用N（总体标准差），不乘交易日数量换算成年化指标。
    mean = sum(returns, Decimal("0")) / Decimal(len(returns))
    variance = sum(((value - mean) ** 2 for value in returns), Decimal("0")) / Decimal(len(returns))
    return variance.sqrt()


def _max_drawdown(nav_values: tuple[Decimal, ...]) -> Decimal:
    """计算固定窗口内历史最大回撤，结果恒为零或负数。"""
    # peak始终是“走到当前这一天为止”见过的最高值，不能提前拿未来最高值作参照。
    peak = nav_values[0]
    max_drawdown = Decimal("0")
    for value in nav_values:
        if value > peak:
            peak = value
        # 例如先到1.20再跌到1.08，回撤为1.08/1.20-1=-10%；保留最负的一次。
        drawdown = value / peak - Decimal("1")
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def _consecutive_decline_days(nav_values: tuple[Decimal, ...]) -> int:
    """计算末尾连续低于前一有效净值日的区间个数；横盘会终止连续下跌。"""
    decline_days = 0
    # 反向检查相邻两天；一旦遇到持平/上涨就停止，因为末尾“连续下跌”已经被打断。
    for previous, current in zip(reversed(nav_values[:-1]), reversed(nav_values[1:]), strict=True):
        if current >= previous:
            break
        decline_days += 1
    return decline_days


def _nav_value(point: HistoricalNavPoint, nav_value_basis: str) -> Decimal | None:
    """按已固定的单一净值口径返回数值，避免累计净值与单位净值混算。"""
    if nav_value_basis == "ACCUMULATED_NAV":
        return point.accumulated_nav
    return point.unit_nav


def _scorable_feature_payload(
    *,
    input_record: HistoricalNavSampleInput,
    anchor: HistoricalNavPoint,
    nav_value_basis: str,
    metrics: dict[str, str | int],
) -> dict[str, object]:
    """组装当时可见的统计指标及追溯信息，标签禁止写入此对象。

    source是当前来源元数据；模型应使用metrics，不把后来发生的同步时间/运行ID当作特征。
    """
    return {
        # 内容结构/计算规则版本，供后续消费者确认自己能否识别这份特征。
        "schema_version": HISTORICAL_NAV_FEATURE_VERSION,
        # 来源说明：人可读的来源编码、同步运行标识，以及本条样本的统一净值口径。
        "source": {
            "source_code": input_record.source_code,
            "source_sync_run_id": str(input_record.source_sync_run_id) if input_record.source_sync_run_id else None,
            "nav_value_basis": nav_value_basis,
        },
        # 日期均输出ISO日期字符串；usable为实际取用数，minimum_required为最低要求。
        "input": {
            "as_of_date": anchor.nav_date.isoformat(),
            "available_at": anchor.ann_date.isoformat() if anchor.ann_date else None,
            "usable_nav_observation_count": MIN_FEATURE_NAV_OBSERVATIONS,
            "minimum_required_nav_observation_count": MIN_FEATURE_NAV_OBSERVATIONS,
        },
        # 指标字典供人阅读，不作为额外数值输入；键名与下面metrics一一对应。
        "feature_definitions": {
            "return_5d": "截至日净值/5个有效净值日前净值-1",
            "return_20d": "截至日净值/20个有效净值日前净值-1",
            "return_60d": "截至日净值/60个有效净值日前净值-1",
            "volatility_20d": "最近20个净值区间收益率的总体标准差，不年化",
            "max_drawdown_60d": "最近60条可用净值内的最大回撤",
            "relative_position_60d": "截至日净值在最近60条可用净值最小值和最大值之间的位置",
            "consecutive_decline_days": "截至日向前连续低于前一有效净值日的区间数",
        },
        # 这里只评价“题目”是否齐全，issues为空表示未发现阻断特征计算的问题。
        # 即使后来答案缺失，此处也保持SCORABLE；总体样本状态由外层eligibility_status表达。
        "quality": {"status": "SCORABLE", "issues": []},
        # 具体指标值：收益/波动/回撤等以8位小数字符串表示，连续下跌次数为整数。
        "metrics": metrics,
    }


def _unavailable_sample(
    *,
    input_record: HistoricalNavSampleInput,
    anchor: HistoricalNavPoint,
    nav_value_basis: str,
    status: str,
    reason: str,
) -> HistoricalNavSample:
    """特征本身无法计算时使用：保留来源和日期，把缺失原因放进quality，指标和答案置空。

    此函数不处理“特征有效、只是答案未成熟”的情况，后者由_attach_offline_label保留原有特征。
    """
    feature_payload: dict[str, object] = {
        "schema_version": HISTORICAL_NAV_FEATURE_VERSION,
        "source": {
            "source_code": input_record.source_code,
            "source_sync_run_id": str(input_record.source_sync_run_id) if input_record.source_sync_run_id else None,
            "nav_value_basis": nav_value_basis,
        },
        "input": {
            "as_of_date": anchor.nav_date.isoformat(),
            "available_at": anchor.ann_date.isoformat() if anchor.ann_date else None,
        },
        "quality": {"status": status, "issues": [reason]},
        "metrics": None,
    }
    return HistoricalNavSample(
        fund_code=input_record.fund_code,
        as_of_date=anchor.nav_date,
        available_at=anchor.ann_date,
        nav_value_basis=nav_value_basis,
        feature_version=HISTORICAL_NAV_FEATURE_VERSION,
        eligibility_status=status,
        unavailable_reason=reason,
        feature_payload=feature_payload,
        feature_hash=_feature_hash(feature_payload),
        offline_label=None,
    )


def _decimal_text(value: Decimal) -> str:
    """按固定精度序列化指标，确保相同输入具有相同哈希。"""
    # quantize按8位小数四舍五入；format(..., 'f')输出普通小数字符串而非科学计数法。
    return format(value.quantize(_METRIC_QUANTUM, rounding=ROUND_HALF_UP), "f")


def _feature_hash(value: dict[str, object]) -> str:
    """生成不包含标签的稳定特征哈希。"""
    # 将字典键排序并去除无关空格，让仅键顺序不同的内容也得到同一个校验指纹。
    # 只传入feature_payload，不传offline_label；指纹用于比较内容，不能单独证明无未来泄漏。
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
