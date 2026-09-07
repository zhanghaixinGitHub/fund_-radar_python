"""GET读库预览的中间层：负责接好“读取数据”和“计算样本”两步。

HTTP入口在app/api/routes/features.py，SQL在app/repositories/historical_nav.py，
公式在app/services/historical_nav_samples.py；本文件将它们串起来，不重复实现公式。
"""

from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.historical_nav import read_historical_nav_sample_input
from app.services.historical_nav_samples import HistoricalNavSample, build_historical_nav_samples


def preview_stored_historical_nav_sample(*, fund_code: str, as_of_date: date) -> HistoricalNavSample:
    """读取最多81条已存净值并返回指定日样本，不同步、不落库、不训练。

    Args:
        fund_code: GET地址中的fundCode，如008888。
        as_of_date: GET地址中的asOfDate，即要查看的净值业务日，如2025-08-07。

    Returns:
        一条HistoricalNavSample，包括当时特征、后来答案或无法计算的原因。

    数据库异常和不可用原因交由HTTP入口映射；多次查询使用同一只读快照，
    但来源运行ID仅说明当前来源水位，不代表已恢复历史修订版本。
    """
    # Session管理这次数据库交互；with结束后自动释放连接，避免请求越来越多时耗尽连接池。
    with Session(get_nav_preview_engine()) as session:
        # REPEATABLE READ：本次多条SQL看到同一份数据库快照，防止同步任务恰好在查询中途改数据。
        # READ ONLY：数据库层禁止本事务写入。它不会把数据库恢复到过去，只保证本次读取一致。
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        input_record = read_historical_nav_sample_input(session, fund_code=fund_code, as_of_date=as_of_date)
    # 数据已经复制成普通对象，可以先归还数据库连接，再进行计算。
    # 构建器会返回多个起点，next挑出用户指定的一天；仓储层已保证该起点存在。
    return next(sample for sample in build_historical_nav_samples(input_record) if sample.as_of_date == as_of_date)
