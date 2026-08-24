"""SQLAlchemy declarative base reserved for M1 fund AI models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base type for all fund-ai SQLAlchemy models."""
