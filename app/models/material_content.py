"""公告、报告和资料包的提取信息；不保存 PDF 等二进制原件。"""

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, LargeBinary, String, Text, false, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class MaterialContent(Base):
    __tablename__ = "material_extracted_content"

    # 原材料摘要仅标识来源版本；读取和计算不要求原文件存在。
    source_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    format: Mapped[str] = mapped_column(String(16), nullable=False)
    source_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(64), nullable=False)
    # zlib 压缩后的 UTF-8 提取数据，包含正文/表格文本和结构化值，不含原 PDF。
    extracted_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    complete: Mapped[bool] = mapped_column(Boolean, nullable=False)
    issue: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MaterialLocation(Base):
    __tablename__ = "material_source_location"

    # 历史路径到来源版本的映射；混合读取可复用同版缓存或定位D盘原件，不全表覆盖旧身份。
    location: Mapped[str] = mapped_column(Text, primary_key=True)
    source_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("material_extracted_content.source_hash"), nullable=False, index=True
    )
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    verification: Mapped[str | None] = mapped_column(String(64))
    # 历史保留决定继续保存；本轮已停止自动清理，不因默认值或旧凭据删除原件。
    retention_hold: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=false())


class MaterialPageRepresentation(Base):
    """已有计算器实际使用的页面排版；追加保存，不反复处理整篇PDF。"""

    __tablename__ = "material_page_representation"
    source_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("material_extracted_content.source_hash"), primary_key=True
    )
    engine: Mapped[str] = mapped_column(String(32), primary_key=True)
    page_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    extracted_text: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class MaterialDocument(Base):
    """与评分无关的文档底稿及内容验收；同一来源按提取政策追加版本。"""

    __tablename__ = "material_document_content"
    source_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("material_extracted_content.source_hash"), primary_key=True
    )
    policy: Mapped[str] = mapped_column(String(32), primary_key=True)
    base_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    document_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # 压缩JSON只保存正文、结构和内容资产的引用；不包含PDF/Word原容器二进制。
    document_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    ready: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
