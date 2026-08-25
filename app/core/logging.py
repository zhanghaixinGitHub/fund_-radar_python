"""统一日志辅助方法，避免在日志中暴露密钥或令牌。"""

import logging


def configure_logging(log_level: str) -> None:
    """按项目日志格式配置进程日志级别；应只在应用启动时调用一次。"""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """返回指定模块的日志记录器，不额外重复注册处理器。"""
    return logging.getLogger(name)
