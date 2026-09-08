"""SQLAlchemy 引擎工厂；M0 启动阶段不会主动建立数据库连接。"""

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from app.core.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """按需创建并缓存连接池引擎，供 M1 迁移和仓储层使用。"""
    return create_engine(get_settings().ai_database_url, pool_pre_ping=True)


@lru_cache
def get_nav_preview_engine() -> Engine:
    """为单日与小范围批量历史样本预览提供独立的小连接池和5秒建连/查询超时。"""
    # lru_cache让本进程的后续请求复用同一连接池；创建引擎本身不会立即读取任何净值。
    # 这里只配置连接与超时；真正的READ ONLY限制在historical_nav_preview服务的事务中设置。
    return create_engine(
        # 从项目既有环境配置读取地址与凭据，不在代码或日志中写明密码。
        get_settings().ai_database_url,
        # 借出连接前检查它是否仍可用，避免长期空闲连接断开后直接执行正式查询。
        pool_pre_ping=True,
        # 独立小池最多保留2个连接，避免教学预览占用太多数据库资源。
        pool_size=2,
        # 两个连接都忙时不再额外创建连接。
        max_overflow=0,
        # 等待空闲连接最多5秒，超过后向上层报错，而不是无限等待。
        pool_timeout=5,
        # SQLAlchemy的SQL日志/错误展示隐藏绑定参数，减少意外泄露输入内容的风险。
        hide_parameters=True,
        # connect_timeout单位是秒；PostgreSQL statement_timeout单位是毫秒（5000即5秒）。
        # 前者限制建连，后者限制每条SQL；它们不代表整个HTTP请求总共只允许5秒。
        connect_args={"connect_timeout": 5, "options": "-c statement_timeout=5000"},
    )


@lru_cache
def get_nav_sample_storage_engine() -> Engine:
    """样本保存/读回使用独立有界连接池；不复用或改变只读预览的事务。"""
    return create_engine(
        get_settings().ai_database_url,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=0,
        pool_timeout=5,
        hide_parameters=True,
        connect_args={
            "connect_timeout": 5,
            "options": "-c statement_timeout=5000 -c lock_timeout=3000 -c idle_in_transaction_session_timeout=30000",
        },
    )
