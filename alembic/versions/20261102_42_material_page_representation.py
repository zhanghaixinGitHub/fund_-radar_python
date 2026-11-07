"""按需保存旧计算器已核验的页面排版，原文件删除后仍可复算。"""

import sqlalchemy as sa
from alembic import op

revision = "20261102_42"
down_revision = "20261102_41"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "material_page_representation",
        sa.Column(
            "source_hash", sa.String(64), sa.ForeignKey("material_extracted_content.source_hash"), primary_key=True
        ),
        sa.Column("engine", sa.String(32), primary_key=True),
        sa.Column("page_index", sa.Integer(), primary_key=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("extracted_text", sa.LargeBinary(), nullable=False),
    )


def downgrade():
    if op.get_bind().scalar(sa.text("SELECT EXISTS(SELECT 1 FROM material_page_representation)")):
        raise RuntimeError("已有必要页面排版，禁止删除唯一复算资料")
    op.drop_table("material_page_representation")
