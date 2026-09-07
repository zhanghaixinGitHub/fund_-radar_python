"""单日与批量 GET 读库预览的中间层：负责接好“读取数据”和“计算样本”两步。

HTTP入口在app/api/routes/features.py，SQL在app/repositories/historical_nav.py，
公式在app/services/historical_nav_samples.py；本文件将它们串起来，不重复实现公式。
"""

from datetime import date
from time import perf_counter

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.historical_nav import (
    read_historical_nav_input_page,
    read_historical_nav_sample_input,
    read_historical_nav_source,
)
from app.schemas.historical_nav import HistoricalNavBatchPreviewRequest, HistoricalNavBatchPreviewResponse
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


class HistoricalNavBatchTimeoutError(TimeoutError):
    """批量试跑超过页间检查的15秒预算；本次不返回不完整的成功结果。"""


def preview_stored_historical_nav_batch(
    request: HistoricalNavBatchPreviewRequest,
) -> HistoricalNavBatchPreviewResponse:
    """分页制作一只基金、最多31个自然日的练习题，只读、不存储、不训练。

    request 已完成范围校验。整次请求使用同一个只读数据库快照和来源运行标识，
    避免某页碰巧读取到同步任务更新后的数据。这里只在内存保留最多31份输出。
    """
    deadline = perf_counter() + 15
    samples: list[HistoricalNavSample] = []
    page_count = 0
    after_date = None
    with Session(get_nav_preview_engine()) as session:
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        source = read_historical_nav_source(session, fund_code=request.fund_code)
        while True:
            # 这是页间总耗时检查，不是硬中断。每条 SQL 另有连接池配置的5秒超时。
            if perf_counter() >= deadline:
                raise HistoricalNavBatchTimeoutError("批量预览超时，请缩短日期范围后重试。")
            page = read_historical_nav_input_page(
                session,
                fund_code=request.fund_code,
                source=source,
                start_date=request.start_date,
                end_date=request.end_date,
                after_date=after_date,
                page_size=request.page_size,
            )
            if not page:
                break
            page_count += 1
            for window in page:
                # 每份完整资料仍交给已经学过的构建器，不在这里另写一套收益公式。
                # 构建器也会生成辅助日期的样本，这里仅取当前要制作的那份题目。
                sample = next(
                    item for item in build_historical_nav_samples(window.input_record)
                    if item.as_of_date == window.as_of_date
                )
                samples.append(sample)
            # 游标总是上一页最后一个起点，下一页严格从其后开始，因此不会重复。
            after_date = page[-1].as_of_date
            if len(page) < request.page_size or after_date == request.end_date:
                break
        # 最后一页也检查预算；失败会退出 with、释放连接，且没有任何结果落库。
        if perf_counter() >= deadline:
            raise HistoricalNavBatchTimeoutError("批量预览超时，请缩短日期范围后重试。")

    reasons: dict[str, int] = {}
    for sample in samples:
        if sample.unavailable_reason is not None:
            reasons[sample.unavailable_reason] = reasons.get(sample.unavailable_reason, 0) + 1
    # 没有净值的范围正常返回0条，不偷偷改日期；有问题的样本保留原因，不默默丢掉。
    return HistoricalNavBatchPreviewResponse(
        fund_code=request.fund_code,
        start_date=request.start_date,
        end_date=request.end_date,
        page_size=request.page_size,
        page_count=page_count,
        sample_count=len(samples),
        scorable_count=sum(item.eligibility_status == "SCORABLE" for item in samples),
        data_insufficient_count=sum(item.eligibility_status == "DATA_INSUFFICIENT" for item in samples),
        label_not_matured_count=sum(item.eligibility_status == "LABEL_NOT_MATURED" for item in samples),
        unavailable_reasons=reasons,
        items=tuple(samples),
    )
