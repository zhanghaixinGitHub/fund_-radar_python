"""公开持仓的成交承压研究：不估计策略超额收益容量，不生成评级。

旧金额加权卖出负荷含有明确的股票权重平方和因子。本研究以可流通股票
账面规模除以持仓权重加权的真实市场成交额，检查一个不含该机械因子的
替代描述。结果仍与持仓权重有关，不能宣称与集中度完全独立，或把比值
解释为可以执行的清仓天数。八维必要资料、校准及正式启用条件均未改变。
"""

from decimal import Decimal, localcontext

from app.services.fund_quality_v2_turnover import stock_turnover_load
from app.services.fund_rating_rules import decimal

VERSION = "QUALITY_STOCK_ACTIVITY_PRESSURE_20261017_2"


def stock_activity_pressure(*, positions: list[dict], stock_amount: str, restricted_rows: list[dict],
                            restricted_amount: str, days: list[str], quotes: dict, report_date: str) -> dict:
    """从完整持仓、受限原表及20交易日成交额重建非评分研究值。

    调用方先核原件、来源授权、币种单位及交易日历。本函数复用已实现的
    全股票/受限金额守恒和逐证券20日数据校验，缺任何正可流通持仓的成交额
    均不汇总。B股的成交币种在本研究尚未核定，不能仅凭SH/SZ后缀按人民币收纳。

    Q=Σq_i，w_i=q_i/Q，A_i为同窗真实日均成交金额；返回Q/Σ(w_i*A_i)。
    金额均为人民币元，比值单位是“账面金额相对加权日成交额”，没有假定本基金
    可以占用整个市场的成交量。压力情景参与率、实际可售性及冲击成本仍未验证。
    """
    b_shares = [row["stock_code"] for row in positions if (
        row["stock_code"].startswith("200") and row["stock_code"].endswith(".SZ")
        or row["stock_code"].startswith("900") and row["stock_code"].endswith(".SH")
    )]
    if b_shares:
        raise ValueError("ACTIVITY_PRESSURE_B_SHARE_AMOUNT_CURRENCY_UNVERIFIED")
    # 这里只借用完整股票、受限金额及20日成交额的输入校验。内层单位参数
    # 产生的load_days不进入本研究输出，不表示允许基金占用100%市场成交量。
    checked = stock_turnover_load(
        positions=positions, stock_amount=stock_amount, restricted_rows=restricted_rows,
        restricted_amount=restricted_amount, days=days, quotes=quotes, report_date=report_date,
        stress_fraction="1", participation_fraction="1",
    )
    result = {"version": VERSION, "report_date": report_date, "sessions": list(days),
              "state": "NOT_READY", "activity_pressure": None, "weighted_daily_turnover_cny": None,
              "free_stock_amount_cny": checked["free_stock_amount_cny"], "gaps": checked["gaps"],
              "scope": "FREE_SH_SZ_STOCK_BOOK_ACTIVITY_NOT_ALPHA_CAPACITY_OR_EXECUTABLE_LIQUIDATION_TIME",
              "quality_metric_admitted": False, "formula": "Q / sum((q_i/Q)*ADV_i)",
              "unit": "BOOK_VALUE_TO_WEIGHTED_DAILY_TURNOVER_RATIO"}
    if checked["gaps"]:
        return result
    with localcontext() as context:
        context.prec = 40
        rows = [row for row in checked["securities"] if row["state"] == "CALCULATED"]
        total = decimal(checked["free_stock_amount_cny"])
        weighted = sum((decimal(row["free_value_cny"]) * decimal(row["average_daily_amount_cny"]) / total
                        for row in rows), Decimal(0))
        if total <= 0 or weighted <= 0:
            raise ValueError("ACTIVITY_PRESSURE_POSITIVE_STOCK_BOOK_AND_TURNOVER_REQUIRED")
        result.update(state="UNSCORED_ACTIVITY_PRESSURE_DIAGNOSTIC", activity_pressure=str(total / weighted),
                      weighted_daily_turnover_cny=str(weighted),
                      free_stock_hhi=str(sum((decimal(row["free_value_cny"]) / total) ** 2 for row in rows)),
                      securities=[{key: row[key] for key in ("stock_code", "free_value_cny", "average_daily_amount_cny",
                                                            "amount_source_hashes")} for row in rows])
        # P=Q/加权ADV描述整个股票账簿相对市场成交量的大小，既可能高于，
        # 也可能低于逐证券约束。不能将其解释为并行比例减持的瓶颈或清仓时间。
        loads = {row["stock_code"]: decimal(row["free_value_cny"]) / decimal(row["average_daily_amount_cny"])
                 for row in rows}
        bottleneck = max(loads.values())
        result["security_volume_constraints"] = {
            "parallel_bottleneck": str(bottleneck), "aggregate_load": str(sum(loads.values(), Decimal(0))),
            "bottleneck_codes": sorted(code for code, value in loads.items() if value == bottleneck),
            "arithmetic_to_bottleneck_ratio": str(total / weighted / bottleneck),
            "unit": "SECURITY_BOOK_VALUE_TO_OWN_20_SESSION_DAILY_AMOUNT_RATIO",
            "definition": "max_i(q_i/ADV_i), sum_i(q_i/ADV_i); both free-stock sleeve only",
            "strategy_capacity_upper_bound": None, "actual_liquidation_days": None,
            "scope": "OBSERVED_MARKET_VOLUME_CONSTRAINTS_NOT_WHOLE_FUND_OR_ALPHA_CAPACITY",
        }
    return result
