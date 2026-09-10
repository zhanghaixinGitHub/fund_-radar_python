"""为 Java 模拟账本提供同源、带版本的行情；刷新只覆盖已登记基金。"""

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import get_engine
from app.models.fund import FundDividend, FundShareClass, NavDaily, SourceRegistry
from app.repositories.fund_sync import TUSHARE_SOURCE_CODE
from app.schemas.simulation_market import SimulationDividend, SimulationMarket, SimulationNav
from app.services.fund_catalog_read import get_fund
from app.services.tushare_fund_sync import TushareFundSyncService

logger = get_logger(__name__)
SHANGHAI = ZoneInfo("Asia/Shanghai")


def unsupported_reason(fund) -> str | None:
    """只接纳已登记的普通场外净值基金；不以缺失资料推断可交易。"""
    if fund.status != "ACTIVE" or fund.market != "O" or fund.profile_status != "SYNCED":
        return "仅支持资料完整的存续场外基金。"
    if fund.fund_type not in {"EQUITY", "STOCK", "MIXED", "HYBRID", "BOND", "INDEX"}:
        return "这类基金的交易规则尚未纳入本次模拟范围。"
    description = " ".join(filter(None, (fund.fund_name, fund.source_fund_type, fund.invest_type))).upper()
    if re.search(r"QDII|FOF|货币|美元|港元|定开|定期开放|持有|封闭|滚动|养老|REIT", description):
        return "首版暂不支持跨境、货币、FOF、持有期或定期开放等特殊产品。"
    if fund.data_source != TUSHARE_SOURCE_CODE or not fund.unit_nav or fund.unit_nav <= 0:
        return "尚无可核验的同源单位净值。"
    return None


def read_market(fund_code: str, start: date, end: date) -> SimulationMarket | None:
    """查询来源原值及版本，不把旧缓存、演示行情或全市场同步状态作为结算凭证。"""
    fund = get_fund(fund_code)
    if fund is None:
        return None
    with Session(get_engine()) as session:
        source = session.scalar(
            select(SourceRegistry.source_id).where(
                SourceRegistry.source_code == TUSHARE_SOURCE_CODE,
            )
        )
        navs = session.scalars(
            select(NavDaily)
            .where(
                NavDaily.fund_code == fund_code,
                NavDaily.source_id == source,
                NavDaily.nav_date >= start,
                NavDaily.nav_date <= end,
                NavDaily.unit_nav > 0,
            )
            .order_by(NavDaily.nav_date)
        ).all()
        dividends = session.scalars(
            select(FundDividend)
            .where(
                FundDividend.fund_code == fund_code,
                FundDividend.source_id == source,
            )
            .order_by(FundDividend.ex_date, FundDividend.source_event_key)
            .limit(1001)
        ).all()
        state = (
            session.execute(
                text(
                    "SELECT status, dividends_verified_at, message FROM simulation_market_refresh WHERE fund_code=:code"
                ),
                {"code": fund_code},
            )
            .mappings()
            .first()
        )
        if len(dividends) > 1000:
            raise ValueError("分红记录超过核验上限，需要分页核验后才能结算。")
        return SimulationMarket(
            fund_code=fund_code,
            fund_name=fund.fund_name,
            supported=unsupported_reason(fund) is None,
            reason=unsupported_reason(fund),
            navs=tuple(
                SimulationNav(
                    nav_date=n.nav_date,
                    unit_nav=n.unit_nav,
                    accumulated_nav=n.accumulated_nav,
                    announced_on=n.ann_date,
                    revision=n.content_hash,
                )
                for n in navs
            ),
            dividends=tuple(
                SimulationDividend(
                    event_key=d.source_event_key,
                    record_date=d.record_date,
                    ex_date=d.ex_date,
                    pay_date=d.pay_date,
                    cash_per_unit=d.cash_dividend,
                    implemented=d.process_status == "实施",
                    revision=d.content_hash,
                )
                for d in dividends
            ),
            dividends_verified_at=state["dividends_verified_at"] if state else None,
            refresh_status=state["status"] if state else "NOT_SYNCED",
            refresh_message=state["message"] if state else "等待后台核验分红资料。",
        )


def validate_registered_codes(codes: list[str]) -> tuple[str, ...]:
    codes = tuple(sorted(set(codes)))
    if not codes or len(codes) > 50 or any(not re.fullmatch(r"[0-9]{6}", code) for code in codes):
        raise ValueError("基金代码格式或数量不合法。")
    with Session(get_engine()) as session:
        registered = set(
            session.scalars(
                select(FundShareClass.fund_code).where(
                    FundShareClass.fund_code.in_(codes),
                    FundShareClass.source_code == TUSHARE_SOURCE_CODE,
                )
            )
        )
    if registered != set(codes):
        raise ValueError("刷新范围仅限已登记的基金，不自动扩展目录。")
    return codes


def refresh_market(codes: tuple[str, ...]) -> None:
    """按基金跨进程加锁、限频，复用已授权净值/分红适配器，不读取任何个人数据。"""
    for code in codes:
        with get_engine().connect() as lock_connection:
            # 固定命名空间 + 基金代码，连接关闭自动释放；数据库状态同时提供跨重启水位。
            locked = lock_connection.execute(
                text("SELECT pg_try_advisory_lock(721104, :code)"), {"code": int(code)}
            ).scalar()
            if not locked:
                continue
            try:
                _refresh_one(code)
            finally:
                lock_connection.execute(text("SELECT pg_advisory_unlock(721104, :code)"), {"code": int(code)})


def _refresh_one(code: str) -> None:
    now = datetime.now(SHANGHAI)
    with Session(get_engine()) as session, session.begin():
        previous = session.execute(
            text("SELECT attempted_at FROM simulation_market_refresh WHERE fund_code=:code"), {"code": code}
        ).scalar()
        if previous and now - previous < timedelta(minutes=30):
            return
        ts_code = session.scalar(select(FundShareClass.source_fund_code).where(FundShareClass.fund_code == code))
        if not ts_code or not ts_code.endswith(".OF"):
            return
        last = session.scalar(
            select(NavDaily.nav_date)
            .join(SourceRegistry)
            .where(
                NavDaily.fund_code == code,
                SourceRegistry.source_code == TUSHARE_SOURCE_CODE,
            )
            .order_by(NavDaily.nav_date.desc())
            .limit(1)
        )
        session.execute(
            text("""
            INSERT INTO simulation_market_refresh (fund_code,status,attempted_at,message)
            VALUES (:code,'RUNNING',:now,'正在核验净值和分红。')
            ON CONFLICT (fund_code) DO UPDATE SET status='RUNNING',attempted_at=:now,message=EXCLUDED.message
        """),
            {"code": code, "now": now},
        )
    service = None
    try:
        service = TushareFundSyncService()
        # 留出重叠窗口以发现近七天来源修正；缺失基线仅回填一年，仍由提交端校验支持范围。
        start = (last - timedelta(days=7)) if last else now.date() - timedelta(days=365)
        service.sync_market_nav_history((ts_code,), start_date=start, end_date=now.date())
        service.sync_market_dividends((ts_code,))
        with Session(get_engine()) as session, session.begin():
            session.execute(
                text("""
                UPDATE simulation_market_refresh SET status='SUCCEEDED',dividends_verified_at=:now,
                    message=NULL WHERE fund_code=:code
            """),
                {"code": code, "now": datetime.now(SHANGHAI)},
            )
    except Exception:
        logger.exception("simulation_market.refresh_market >>> refresh failed, fund_code=%s", code)
        with Session(get_engine()) as session, session.begin():
            session.execute(
                text("""
                UPDATE simulation_market_refresh SET status='FAILED',message='行情更新失败，等待重试；保留已有数据。'
                WHERE fund_code=:code
            """),
                {"code": code},
            )
    finally:
        if service is not None:
            service.close()
