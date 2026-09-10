"""净值可用性 V2：公告原值仅追溯，研究按下一交易日可用；缺失填充只在特征副本中。"""

from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext

from app.schemas.direction_training import DirectionAnswer, DirectionInput
from app.services.cash_reinvestment_samples import _event_available_at, _events_for_period, _number
from app.services.direction_training_artifacts import digest
from app.services.direction_training_dataset import END, FUNDS, START
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.historical_nav_samples import _build_metrics
from app.services.trading_calendar import load_calendar

POLICY = "NAV_NEXT_SESSION_AVAILABLE_V2"
GAP_POLICY = "MAX2_ISOLATED_CAUSAL_FFILL_PROTECTED_ENDPOINTS_V1"


def history_dates(cutoff):
    if not START <= cutoff <= END:
        raise ValueError("INPUT_DATE_PROTECTED")
    calendar = load_calendar()
    index = calendar.at_or_before_index(cutoff)
    # 固定使用上一交易日，不能因今天已经出现某行而改变回测输入延迟。
    return calendar.sessions[index - 61 : index] if index >= 61 else ()


def assumed_available(day):
    return load_calendar().future_sessions(day, 1)[0]


def known_events(events, cutoff):
    return tuple(e for e in events if (d := _event_available_at(e)) is not None and d <= cutoff)


def eligible_gap_indices(dates, events):
    """保护收益端点、最新六条以及已知分红日及其相邻日；只读已知事件元数据。"""
    event_days = {d for e in events for d in (e.ex_date, e.nav_ex_date) if d is not None}
    return tuple(i for i in range(1, 55) if i != 40 and not any(d in event_days for d in dates[i - 1 : i + 2]))


def stress_dates(fund, cutoff, events, count):
    if count not in (0, 1, 2):
        raise ValueError("MASK_COUNT")
    dates = history_dates(cutoff)
    if len(dates) != 61:
        return ()
    eligible = eligible_gap_indices(dates, known_events(events, cutoff))
    ranked = sorted(eligible, key=lambda i: digest(["NAV_GAP_SEED_42", fund, str(cutoff), str(dates[i])]))
    chosen = []
    for i in ranked:
        if len(chosen) == count:
            break
        if all(abs(i - j) > 1 for j in chosen):
            chosen.append(i)
    if len(chosen) != count:
        raise ValueError("NO_ELIGIBLE_STRESS_DATES")
    return tuple(dates[i] for i in sorted(chosen))


def cash_series(dates, nav, events, cutoff, *, max_missing=0, hidden=()):
    """保留完整交易日序列，不将多日收益压成单日；标签一律 max_missing=0。"""
    if max_missing not in (0, 2) or len(dates) not in (21, 61) or len(set(dates)) != len(dates):
        raise ValueError("SERIES_POLICY_OR_SHAPE")
    if any(a >= b for a, b in zip(dates[:-1], dates[1:], strict=True)):
        raise ValueError("SERIES_DATE_ORDER")
    if len(hidden) > 2 or len(set(hidden)) != len(hidden) or not set(hidden).issubset(dates):
        raise ValueError("HIDDEN_DATE_SCOPE")
    cash, problems = _events_for_period(events, dates, cutoff)
    if problems:
        return None, None, sorted({p.code for p in problems})
    missing = [i for i, d in enumerate(dates) if d not in nav or d in hidden]
    if missing:
        eligible = eligible_gap_indices(dates, events) if len(dates) == 61 else ()
        if len(missing) > max_missing:
            return None, None, ["NAV_GAP_COUNT_EXCEEDED"]
        if any(i not in eligible for i in missing):
            return None, None, ["NAV_GAP_AT_PROTECTED_DATE"]
        if any(b - a == 1 for a, b in zip(missing[:-1], missing[1:], strict=True)):
            return None, None, ["NAV_GAP_CONSECUTIVE"]
    values, availability = [], []
    for i, day in enumerate(dates):
        if i in missing:
            values.append(values[-1])  # 临时估算，绝不写回原始净值；不读取后一天的值。
            continue
        point = nav[day]
        if point.unit_nav is None or not point.unit_nav.is_finite() or point.unit_nav <= 0:
            return None, None, ["UNIT_NAV_INVALID"]
        available = assumed_available(day)
        if available > cutoff:
            return None, None, ["NAV_AFTER_ASSUMED_CUTOFF"]
        values.append(point.unit_nav)
        availability.append(available)
    availability.extend(_event_available_at(e) for e in cash.values())
    with localcontext() as context:
        context.prec = 40
        context.rounding = ROUND_HALF_UP
        index, result = Decimal(100), []
        for i, value in enumerate(values):
            if i:
                dividend = cash[dates[i]].cash_dividend if dates[i] in cash else Decimal(0)
                index *= (value + dividend) / values[i - 1]
            result.append(Decimal(_number(index)))
    audit = {
        "availability_policy": POLICY,
        "gap_policy": GAP_POLICY if max_missing else "REQUIRE_COMPLETE",
        "historical_first_publication_verified": False,
        "available_at": str(max(availability)),
        "dates": [str(d) for d in dates],
        "missing_dates": [str(dates[i]) for i in missing],
        "synthetic_missing_dates": [str(d) for d in hidden],
        "values_used": [str(v) for v in values],
        "cash_events": {str(d): {"key": e.event_key, "cash": str(e.cash_dividend)} for d, e in cash.items()},
    }
    return tuple(result), audit, []


def build_input(fund, cutoff, nav, events, *, tolerate=False, hidden=()):
    if fund not in FUNDS:
        raise ValueError("INPUT_FUND")
    dates = history_dates(cutoff)
    if len(dates) != 61:
        return None, {}, ["CALENDAR_HISTORY_SHORTAGE"]
    series, audit, issues = cash_series(
        dates, nav, known_events(events, cutoff), cutoff, max_missing=2 if tolerate else 0, hidden=hidden
    )
    if issues:
        return None, {}, issues
    with localcontext() as context:
        context.prec = 40
        metrics = _build_metrics(series)
    if metrics is None:
        return None, audit, ["FLAT_HISTORY_POSITION_UNDEFINED"]
    item = DirectionInput(
        fund=fund,
        cutoff=cutoff,
        anchor=dates[-1],
        available_at=date.fromisoformat(audit["available_at"]),
        x=tuple(float(metrics[n]) for n in FEATURE_NAMES),
        input_hash=digest(audit),
    )
    return item, audit, []


def build_answer(fund, cutoff, nav, events, *, include_value=False):
    if fund not in FUNDS or not START <= cutoff <= END:
        raise ValueError("ANSWER_DATE_PROTECTED")
    calendar = load_calendar()
    future = calendar.future_sessions(cutoff)
    if future[-1] > END:
        return None, ["LABEL_CROSSES_PROTECTED_PERIOD"]
    base = calendar.sessions[calendar.at_or_before_index(cutoff)]
    series, audit, issues = cash_series((base, *future), nav, events, END)
    if issues:
        return None, issues
    metadata = {"fund": fund, "cutoff": str(cutoff), "end": str(future[-1]), "available_at": audit["available_at"]}
    if not include_value:
        return metadata, []
    with localcontext() as context:
        context.prec = 40
        value = (series[-1] / 100 - 1).quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP)
    return DirectionAnswer(**metadata, y=int(value > 0), future_return=str(value)), []
