"""已核总回报序列的合同目标稳定性计算；不选择参照、不填日期或生成等级。"""

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, localcontext


def _dates(values: Sequence[str]) -> list[str]:
    """校验实际估值日的唯一性及顺序；不以自然月末替换真实交易日。"""
    result = list(values)
    if len(result) < 2 or result != sorted(set(result)):
        raise ValueError("QUALITY_STABILITY_DATE_ORDER")
    for value in result:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("QUALITY_STABILITY_DATE_FORMAT")
    return result


def _level(series: Mapping[str, str | Decimal], day: str) -> Decimal:
    if day not in series:
        raise ValueError("QUALITY_STABILITY_MISSING_DATE:" + day)
    value = Decimal(str(series[day]))
    if not value.is_finite() or value <= 0:
        raise ValueError("QUALITY_STABILITY_LEVEL")
    return value


def blended_benchmark(dates: Sequence[str], components: Mapping[str, Mapping[str, str | Decimal]],
                      weights: Mapping[str, str | Decimal]) -> dict[str, str]:
    """按合同固定权重合成逐日基准，起点归一为1；权重为0—1的比例。

    调用方须核准指数身份、收益口径和本产品适用合同，并用官方报告中的
    区间基准回报交叉检查日复合口径。此函数不把单个指数或期末固定配置
    自动认作正确基准，任何必要日缺失均报错。
    """
    days = _dates(dates)
    with localcontext() as context:
        context.prec = 60
        amounts = {key: Decimal(str(value)) for key, value in weights.items()}
        if set(amounts) != set(components) or not amounts or any(
            not value.is_finite() or value <= 0 for value in amounts.values()
        ) or sum(amounts.values()) != 1:
            raise ValueError("QUALITY_STABILITY_BENCHMARK_WEIGHTS")
        output = {days[0]: "1"}
        for previous, day in zip(days[:-1], days[1:], strict=True):
            growth = sum(amounts[key] * _level(components[key], day) / _level(components[key], previous)
                         for key in amounts)
            output[day] = str(Decimal(output[previous]) * growth)
        return output


def score_stability(total_return: Mapping[str, str | Decimal], benchmark: Mapping[str, str | Decimal],
                    month_ends: Sequence[str]) -> dict:
    """使用37个连续月末形成36个月和25个滚动年度，月度30%、年度70%。

    输入是已核分红/折算后的同份额总回报指数及适配参照，数值单位一致；
    仅比较同一真实端点的回报。相对回报正、零、负分别计1、0.5、0。
    滚动年度直接取各自12个月端点，不能连乘月度横截面中位数代替。
    25个重叠年度不是25次独立观察，输出不是未来胜率或完整基金评级。
    """
    days = _dates(month_ends)
    if len(days) != 37:
        raise ValueError("QUALITY_STABILITY_REQUIRES_37_MONTH_ENDS")
    month_ids = [date.fromisoformat(day).year * 12 + date.fromisoformat(day).month for day in days]
    if any(right - left != 1 for left, right in zip(month_ids[:-1], month_ids[1:], strict=True)):
        raise ValueError("QUALITY_STABILITY_MONTH_GAP")
    with localcontext() as context:
        context.prec = 60

        def windows(length: int) -> list[dict]:
            rows = []
            for index in range(length, len(days)):
                start, end = days[index - length], days[index]
                fund_return = _level(total_return, end) / _level(total_return, start) - 1
                reference_return = _level(benchmark, end) / _level(benchmark, start) - 1
                difference = fund_return - reference_return
                credit = Decimal(1) if difference > 0 else Decimal(0) if difference < 0 else Decimal(".5")
                rows.append({"start": start, "end": end, "fund_return": str(fund_return),
                             "reference_return": str(reference_return), "difference": str(difference),
                             "credit": str(credit)})
            return rows

        monthly, annual = windows(1), windows(12)
        monthly_score = sum(Decimal(row["credit"]) for row in monthly) * 100 / 36
        annual_score = sum(Decimal(row["credit"]) for row in annual) * 100 / 25
        return {"score": str(monthly_score * Decimal(".3") + annual_score * Decimal(".7")),
                "monthly_score": str(monthly_score), "annual_score": str(annual_score),
                "monthly": monthly, "rolling_annual": annual,
                "point_definition": "公开净值和已核合同基准的历史相对达标；不代表未来胜率"}
