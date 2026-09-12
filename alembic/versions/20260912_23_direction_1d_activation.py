"""允许预测时已完成的模型进入当前窗口，旧映射原样保留。"""

from alembic import op

revision = "20260912_23"
down_revision = "20260911_22"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 增加选择策略维度，不删除旧空映射，也不改旧模型、预测或训练原文。
    op.execute("""
      ALTER TABLE direction_1d_window_model ADD COLUMN activation_policy varchar(40)
        NOT NULL DEFAULT 'BEFORE_WINDOW_V1';
      COMMENT ON COLUMN direction_1d_window_model.activation_policy IS
        '模型选择策略：旧BEFORE_WINDOW_V1；预测时已可用AVAILABLE_AT_PREDICTION_V2';
      ALTER TABLE direction_1d_window_model DROP CONSTRAINT direction_1d_window_model_pkey;
      ALTER TABLE direction_1d_window_model ADD PRIMARY KEY
        (cohort_id,group_id,target_nav_date,branch_id,activation_policy);
    """)


def downgrade() -> None:
    raise RuntimeError("模型选择证据保留；回退应用，不删除已留档的新策略映射。")
