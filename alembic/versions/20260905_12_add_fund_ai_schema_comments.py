"""为基金 AI 数据库的现有表和字段补充中文注释。

Revision ID: 20260905_12
Revises: 20260902_11
Create Date: 2026-09-05 13:15:00

此迁移只写 PostgreSQL 元数据中的 COMMENT，不修改业务数据、字段类型、约束或索引。
``nav_daily`` 是分区父表，字段注释不会自动继承到子分区，因此会同步处理当前
所有月分区和默认分区。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260905_12"
down_revision: str | None = "20260902_11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TABLE_COMMENTS: dict[str, str] = {
    "alembic_version": "Alembic 数据库迁移版本记录",
    "analysis_explanation_snapshot": "基金分析说明快照",
    "analysis_model_release": "分析模型发布记录",
    "analysis_run": "分析任务运行控制记录",
    "backtest_run": "模型回测运行记录",
    "benchmark_nav_daily": "业绩比较基准日序列",
    "benchmark_series": "业绩比较基准主数据",
    "event_relation": "市场事件与业务对象关联",
    "feature_snapshot": "基金分析特征快照",
    "forecast_result": "基金分析评分结果",
    "fund_dividend": "基金分红记录",
    "fund_exchange_daily": "场内基金日行情",
    "fund_manager_assignment": "基金经理任职记录",
    "fund_master": "基金主体主数据",
    "fund_profile": "基金基础资料快照",
    "fund_share_class": "基金份额类别主数据",
    "fund_share_snapshot": "基金份额规模快照",
    "index_weight_snapshot": "指数成分权重快照",
    "market_event": "审核后的市场事件",
    "market_index_catalog": "市场指数目录",
    "market_index_classification": "市场指数分类目录",
    "nav_daily": "基金日净值（按净值日期分区）",
    "news_item": "已授权来源的新闻或公告",
    "news_source_reference": "新闻来源引用",
    "source_registry": "数据来源授权与运行治理台账",
    "source_sync_cursor": "数据源同步游标",
    "source_sync_run": "数据源同步运行记录",
}


COLUMN_COMMENTS: dict[str, str] = {
    "accumulated_dividend": "累计分红金额",
    "accumulated_nav": "累计单位净值",
    "active": "记录是否在有效期内",
    "adjusted_nav": "复权单位净值",
    "amount": "成交金额",
    "analysis_run_id": "分析任务运行唯一标识",
    "ann_date": "公告日期",
    "approval_status": "审核状态",
    "approved_at": "审核完成时间",
    "as_of_date": "数据或分析结果截至日期",
    "authorization_verified_at": "数据源授权核验时间",
    "authorized_api_names": "已核验可调用的 API 名称清单",
    "backtest_run_id": "关联的回测运行唯一标识",
    "base_date": "基准日期",
    "base_unit": "分红计算基准份额",
    "base_year": "分红所属年度",
    "baselines": "基线策略或对照指标（JSON）",
    "begin_date": "任职开始日期",
    "benchmark": "业绩比较基准说明",
    "benchmark_code": "业绩比较基准代码",
    "cash_dividend": "每份派发的现金红利",
    "category": "指数分类",
    "change_percent": "涨跌幅（百分比）",
    "change_value": "涨跌额",
    "classification_code": "指数分类代码",
    "classification_name": "指数分类名称",
    "close_price": "收盘价",
    "closing_value": "基准日收盘点位或收盘值",
    "completeness": "特征数据完整度，取值范围 0 至 1",
    "completion_tokens": "模型生成内容消耗的 Token 数",
    "computed_at": "特征计算完成时间",
    "confidence": "结果置信度，取值范围 0 至 1",
    "config_hash": "运行配置内容哈希",
    "consecutive_failure_count": "连续同步失败次数",
    "constituent_code": "指数成分证券代码",
    "content_hash": "来源内容或快照内容哈希，用于幂等去重",
    "created_at": "记录创建时间",
    "created_count": "本次同步新建记录数",
    "custodian_fee": "基金托管费率",
    "custodian_name": "基金托管人名称",
    "data_as_of_date": "本次同步数据的最新可用日期",
    "data_gap": "数据缺口或限制说明",
    "dataset_code": "来源数据集代码",
    "delist_date": "终止上市或终止日期",
    "direction": "评分方向（上涨、下跌或中性）",
    "directional_probability": "方向预测概率，取值范围 0 至 1",
    "disclaimer": "风险提示和免责声明",
    "display_name": "面向用户展示的名称",
    "distributable_earnings": "可供分配收益",
    "due_date": "基金到期日期",
    "duration_year": "基金存续期限（年）",
    "earnings_amount": "收益分配金额",
    "earnings_pay_date": "收益发放日期",
    "education": "基金经理学历信息",
    "effective_at": "模型版本生效时间",
    "eligibility_status": "是否满足评分前置条件的状态",
    "enabled": "数据源是否允许启用",
    "end_date": "任职结束日期",
    "entity_id": "关联业务对象标识",
    "entity_key": "来源数据集中业务对象键",
    "entity_type": "关联业务对象类型",
    "error_summary": "本次运行的脱敏错误摘要",
    "established_date": "基金成立日期",
    "event_hash": "事件内容哈希，用于幂等去重",
    "event_id": "市场事件唯一标识",
    "event_type": "市场事件类型",
    "evidence": "生成说明引用的证据（JSON）",
    "ex_date": "除息日",
    "expected_return": "预期收益率",
    "expiry_date": "指数终止或到期日期",
    "explanation": "结果解释或事件解读",
    "explanation_id": "分析说明快照唯一标识",
    "failure_reason": "失败原因",
    "feature_hash": "特征内容哈希，用于幂等去重",
    "feature_id": "特征快照唯一标识",
    "feature_payload": "特征明细载荷（JSON）",
    "feature_version": "特征计算规则版本",
    "fee_rate": "回测采用的费率",
    "fetched_at": "来源内容抓取时间",
    "fetched_count": "本次同步从来源获取的记录数",
    "finished_at": "运行完成时间",
    "forecast_id": "评分结果唯一标识",
    "found_date": "来源提供的基金成立日期",
    "fund_code": "基金代码（份额类别标识）",
    "fund_manager_assignment_id": "基金经理任职记录唯一标识",
    "fund_master_id": "基金主体唯一标识",
    "fund_name": "基金名称",
    "fund_profile_id": "基金基础资料快照唯一标识",
    "fund_share": "基金总份额",
    "fund_type": "标准化基金类型",
    "generated_at": "说明内容生成时间",
    "hierarchy_level": "分类层级",
    "high_price": "最高价",
    "implementation_ann_date": "分红实施公告日期",
    "index_code": "指数代码",
    "invest_type": "投资类型",
    "issue_amount": "基金发行规模",
    "issue_date": "发行日期",
    "last_error_at": "最近一次同步失败时间",
    "last_error_summary": "最近一次同步失败的脱敏摘要",
    "last_success_at": "最近一次同步成功时间",
    "last_successful_data_date": "最近一次成功同步的数据日期",
    "last_successful_published_at": "最近一次成功同步内容的发布时间",
    "last_sync_run_id": "最近一次同步运行唯一标识",
    "license_reference": "数据授权或许可凭据引用",
    "license_scope": "数据授权范围、保留和展示规则",
    "list_date": "上市日期",
    "low_price": "最低价",
    "management_company_name": "基金管理人名称",
    "management_fee": "基金管理费率",
    "manager_name": "基金经理名称",
    "market": "所属市场标识",
    "max_drawdown_estimate": "预测的最大回撤估计值",
    "metrics": "回测指标（JSON）",
    "min_purchase_amount": "最低申购金额",
    "model_code": "模型代码",
    "model_release_id": "模型发布记录唯一标识",
    "model_version": "模型版本",
    "nav_date": "基金净值日期",
    "nav_ex_date": "净值除权日",
    "net_asset": "资产净值",
    "news_id": "新闻或公告唯一标识",
    "open_price": "开盘价",
    "overview": "面向用户的分析概述",
    "par_value": "基金面值",
    "parent_classification_code": "上级指数分类代码",
    "parent_sync_run_id": "父同步运行唯一标识",
    "pay_date": "现金红利发放日期",
    "previous_close_price": "前收盘价",
    "process_status": "分红实施进度状态",
    "prompt_tokens": "模型提示词消耗的 Token 数",
    "prompt_version": "生成提示词模板版本",
    "provider": "生成服务提供方",
    "provider_model": "生成服务使用的模型名称",
    "provider_request_id": "生成服务请求标识",
    "publication_status": "回测结果的发布资格状态",
    "published_at": "来源内容或事件发布时间",
    "publisher": "指数发布机构",
    "purchase_start_date": "开放申购日期",
    "rate_limit_per_minute": "来源每分钟允许请求次数",
    "record_date": "权益登记日",
    "redemption_start_date": "开放赎回日期",
    "reference_id": "新闻来源引用唯一标识",
    "reinvestment_arrival_date": "红利再投资到账日期",
    "relation_id": "事件关联记录唯一标识",
    "relation_reason": "建立关联的依据说明",
    "release_reason": "模型发布或状态变更原因",
    "release_status": "模型发布状态",
    "relevance_score": "事件与对象的关联度，取值范围 0 至 1",
    "request_payload": "分析任务请求参数（JSON）",
    "requested_at": "任务请求创建时间",
    "requested_nav_date": "请求同步的目标净值日期",
    "requested_window_end": "请求同步窗口结束日期（含义由任务类型定义）",
    "requested_window_start": "请求同步窗口开始日期（含义由任务类型定义）",
    "result_hash": "评分结果内容哈希，用于幂等去重",
    "result_payload": "分析任务结果载荷（JSON）",
    "retention_days": "来源数据允许保留的天数",
    "retention_until": "来源内容允许保留至的时间",
    "risk_level": "风险等级",
    "risk_notice": "面向用户的风险提示",
    "row_hash": "业务行内容哈希，用于幂等识别",
    "run_id": "运行唯一标识",
    "run_type": "分析任务类型",
    "score_status": "评分结果状态",
    "scored_at": "评分完成时间",
    "share_class": "基金份额类别",
    "skipped_count": "本次同步跳过的记录数",
    "source_code": "数据来源代码",
    "source_event_key": "来源侧分红事件唯一键",
    "source_fund_code": "来源系统中的基金代码",
    "source_fund_type": "来源系统中的基金类型",
    "source_id": "数据来源唯一标识",
    "source_input_hash": "生成说明所用输入数据哈希",
    "source_kind": "数据来源类别",
    "source_name": "来源侧分类名称",
    "source_published_at": "来源数据发布时间",
    "source_record_key": "来源系统中的记录唯一键",
    "started_at": "运行开始时间",
    "status": "当前业务或处理状态（取值范围见本表约束）",
    "strategy_version": "回测策略版本",
    "summary": "摘要内容",
    "suspended_at": "模型版本暂停时间",
    "sync_run_id": "同步运行唯一标识",
    "sync_type": "同步类型",
    "task_id": "异步任务唯一标识",
    "test_end": "回测测试集结束日期",
    "test_start": "回测测试集开始日期",
    "title": "新闻或公告标题",
    "total_net_asset": "基金资产总额",
    "trace_id": "请求链路追踪标识",
    "trade_date": "交易日期",
    "train_end": "回测训练集结束日期",
    "trustee_name": "基金受托人名称",
    "unavailable_reason": "数据不可用或不满足条件的原因",
    "unit_nav": "单位净值",
    "updated_at": "记录最后更新时间",
    "updated_count": "本次同步更新的记录数",
    "url": "来源链接",
    "validation_end": "回测验证集结束日期",
    "version_num": "当前已应用的 Alembic 迁移版本号",
    "volume": "成交量",
    "weight": "指数成分权重（百分比）",
    "weight_date": "成分权重生效日期",
    "wide_created_at": "模型发布记录创建时间",
    "wide_updated_at": "模型发布记录最后业务更新时间",
    "window_end": "回测时间窗口结束日期",
    "window_start": "回测时间窗口开始日期",
}


def _quote_identifier(identifier: str) -> str:
    """返回可安全用于 PostgreSQL DDL 的双引号标识符。"""
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _comment_literal(comment: str | None) -> str:
    """返回 COMMENT 语句使用的 SQL 字面量。"""
    if comment is None:
        return "NULL"
    return "'" + comment.replace("'", "''") + "'"


def _comment_on_table(table_name: str, comment: str | None) -> None:
    """设置 public schema 中指定表的注释。"""
    target = f'{_quote_identifier("public")}.{_quote_identifier(table_name)}'
    op.execute(sa.text(f"COMMENT ON TABLE {target} IS {_comment_literal(comment)}"))


def _comment_on_column(table_name: str, column_name: str, comment: str | None) -> None:
    """设置 public schema 中指定列的注释。"""
    target = ".".join(
        (_quote_identifier("public"), _quote_identifier(table_name), _quote_identifier(column_name))
    )
    op.execute(sa.text(f"COMMENT ON COLUMN {target} IS {_comment_literal(comment)}"))


def _logical_table_columns() -> dict[str, set[str]]:
    """读取当前 schema 的非分区子表字段，用于阻止字段漂移时漏写注释。"""
    rows = op.get_bind().execute(
        sa.text(
            """
            SELECT relation.relname AS table_name, attribute.attname AS column_name
            FROM pg_class AS relation
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            JOIN pg_attribute AS attribute ON attribute.attrelid = relation.oid
            WHERE namespace.nspname = current_schema()
              AND relation.relkind IN ('r', 'p')
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
              AND NOT EXISTS (
                  SELECT 1
                  FROM pg_inherits AS inheritance
                  WHERE inheritance.inhrelid = relation.oid
              )
            ORDER BY relation.relname, attribute.attnum
            """
        )
    )
    table_columns: dict[str, set[str]] = {}
    for table_name, column_name in rows:
        table_columns.setdefault(table_name, set()).add(column_name)
    return table_columns


def _nav_daily_partitions() -> list[str]:
    """读取 nav_daily 的直接分区，确保列注释同步写入物理表。"""
    rows = op.get_bind().execute(
        sa.text(
            """
            SELECT child.relname
            FROM pg_inherits AS inheritance
            JOIN pg_class AS parent ON parent.oid = inheritance.inhparent
            JOIN pg_class AS child ON child.oid = inheritance.inhrelid
            JOIN pg_namespace AS namespace ON namespace.oid = parent.relnamespace
            WHERE namespace.nspname = current_schema()
              AND parent.relname = 'nav_daily'
            ORDER BY child.relname
            """
        )
    )
    return [row[0] for row in rows]


def _validate_comment_coverage(table_columns: dict[str, set[str]]) -> None:
    """校验迁移覆盖当前逻辑表的每个字段，防止无声漏注释。"""
    actual_tables = set(table_columns)
    expected_tables = set(TABLE_COMMENTS)
    unexpected_tables = actual_tables - expected_tables
    missing_tables = expected_tables - actual_tables
    missing_columns = {
        table_name: sorted(columns - set(COLUMN_COMMENTS))
        for table_name, columns in table_columns.items()
        if columns - set(COLUMN_COMMENTS)
    }
    if unexpected_tables or missing_tables or missing_columns:
        raise RuntimeError(
            "Schema comment coverage mismatch: "
            f"unexpected_tables={sorted(unexpected_tables)}, "
            f"missing_tables={sorted(missing_tables)}, "
            f"missing_columns={missing_columns}"
        )


def _apply_comments(*, clear_comments: bool) -> None:
    """写入或清除全部逻辑表、字段及净值分区的注释。"""
    table_columns = _logical_table_columns()
    _validate_comment_coverage(table_columns)

    for table_name in sorted(table_columns):
        _comment_on_table(table_name, None if clear_comments else TABLE_COMMENTS[table_name])
        for column_name in sorted(table_columns[table_name]):
            column_comment = None if clear_comments else COLUMN_COMMENTS[column_name]
            _comment_on_column(table_name, column_name, column_comment)

    for partition_name in _nav_daily_partitions():
        partition_comment = "基金日净值月分区，字段语义与 nav_daily 保持一致"
        _comment_on_table(partition_name, None if clear_comments else partition_comment)
        for column_name in sorted(table_columns["nav_daily"]):
            column_comment = None if clear_comments else COLUMN_COMMENTS[column_name]
            _comment_on_column(partition_name, column_name, column_comment)


def upgrade() -> None:
    """为全部现有基金 AI 表及字段补充中文元数据注释。"""
    _apply_comments(clear_comments=False)


def downgrade() -> None:
    """清除本迁移写入的表和字段注释，不改变任何业务数据。"""
    _apply_comments(clear_comments=True)
