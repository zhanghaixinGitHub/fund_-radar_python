"""用户授权的本地市场广度研究适配器；仅查询指定历史交易日的股票日线。"""

from datetime import date

from app.integrations.tushare_market_reference import TushareMarketReferenceClient

FIELDS = ("ts_code", "trade_date", "close", "pre_close", "pct_chg", "vol")


class TushareBreadthClient(TushareMarketReferenceClient):
    def list_daily_market(self, day: date) -> tuple[dict, ...]:
        """按历史日期读取全市场响应，保留退市股票当年的记录，不套用当前存续名单。

        停牌期间来源不提供日线。单次达到官方6000行上限时拒绝，不把截断当完整。
        返回原始字段供研究包保存并检查；不写正式来源权限或股票行情表。
        """
        if not date(2021, 1, 1) <= day <= date(2024, 12, 31):
            raise ValueError("BREADTH_RESEARCH_DATE_SCOPE")
        rows = self._query("daily", params={"trade_date": day.strftime("%Y%m%d")}, fields=",".join(FIELDS))
        if len(rows) >= 6000:
            raise ValueError("BREADTH_RESPONSE_TRUNCATION")
        return rows
