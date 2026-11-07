"""通用文档底稿和独立内容验收；保留所有现有原件的暂缓标记。"""

import sqlalchemy as sa
from alembic import op

revision = "20261102_43"
down_revision = "20261102_42"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "material_source_location", sa.Column("retention_hold", sa.Boolean(), nullable=False, server_default=sa.true())
    )
    op.alter_column("material_source_location", "retention_hold", server_default=sa.false())
    op.create_table(
        "material_document_content",
        sa.Column(
            "source_hash", sa.String(64), sa.ForeignKey("material_extracted_content.source_hash"), primary_key=True
        ),
        sa.Column("policy", sa.String(32), primary_key=True),
        sa.Column("base_content_hash", sa.String(64), nullable=False),
        sa.Column("document_hash", sa.String(64), nullable=False),
        sa.Column("document_data", sa.LargeBinary(), nullable=False),
        sa.Column("ready", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_material_document_content_document_hash", "material_document_content", ["document_hash"])


def downgrade():
    if op.get_bind().scalar(sa.text("SELECT EXISTS(SELECT 1 FROM material_document_content)")):
        raise RuntimeError("通用内容底稿已有数据，禁止删除唯一提取结果")
    op.drop_table("material_document_content")
    op.drop_column("material_source_location", "retention_hold")
