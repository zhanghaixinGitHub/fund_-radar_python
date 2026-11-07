"""保存材料提取信息和原件清理凭据，不修改历史评分与预测。"""

import sqlalchemy as sa
from alembic import op

revision = "20261102_41"
down_revision = "20261028_40"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "material_extracted_content",
        sa.Column("source_hash", sa.String(64), primary_key=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("format", sa.String(16), nullable=False),
        sa.Column("source_size", sa.BigInteger(), nullable=False),
        sa.Column("extractor_version", sa.String(64), nullable=False),
        sa.Column("extracted_data", sa.LargeBinary(), nullable=False),
        sa.Column("complete", sa.Boolean(), nullable=False),
        sa.Column("issue", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_material_extracted_content_content_hash", "material_extracted_content", ["content_hash"])
    op.create_table(
        "material_source_location",
        sa.Column("location", sa.Text(), primary_key=True),
        sa.Column(
            "source_hash", sa.String(64), sa.ForeignKey("material_extracted_content.source_hash"), nullable=False
        ),
        sa.Column("verified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("verification", sa.String(64)),
    )
    op.create_index("ix_material_source_location_source_hash", "material_source_location", ["source_hash"])


def downgrade():
    # 删除原件后不可丢弃唯一的提取信息；只允许在表为空时回退结构。
    connection = op.get_bind()
    if connection.scalar(sa.text("SELECT EXISTS(SELECT 1 FROM material_extracted_content)")):
        raise RuntimeError("资料库已有内容，禁止降级删除")
    op.drop_table("material_source_location")
    op.drop_table("material_extracted_content")
