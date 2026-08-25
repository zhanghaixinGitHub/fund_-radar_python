"""供基金 AI 服务 M1 及后续模型继承的 SQLAlchemy 声明式基类。"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """基金 AI 服务全部 SQLAlchemy 持久化模型的公共基类。"""
