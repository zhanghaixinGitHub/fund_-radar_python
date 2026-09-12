"""独立1日实验的公共证据、训练注册、窗口锁定与有界作业。"""

from alembic import op

revision = "20260911_21"
down_revision = "20260910_20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE FUNCTION direction_1d_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'direction_1d immutable evidence'; END $$;
    CREATE TABLE direction_1d_source_version (
      version_id uuid PRIMARY KEY, fund_code varchar(32) NOT NULL, source_id uuid NOT NULL,
      kind varchar(16) NOT NULL, business_date date NOT NULL, content_hash char(64) NOT NULL,
      payload jsonb NOT NULL, received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
      stored_at timestamptz NOT NULL DEFAULT clock_timestamp(), expires_at timestamptz NOT NULL,
      UNIQUE(fund_code,source_id,kind,business_date,content_hash));
    CREATE TABLE direction_1d_snapshot (
      snapshot_id uuid PRIMARY KEY, kind varchar(16) NOT NULL, task_key varchar(256) NOT NULL,
      as_of timestamptz NOT NULL, payload jsonb NOT NULL, content_hash char(64) NOT NULL,
      expires_at timestamptz NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE INDEX ix_direction_1d_snapshot_task ON direction_1d_snapshot(task_key,kind,as_of);
    CREATE TABLE direction_1d_training_run (
      run_id uuid PRIMARY KEY, cohort_id varchar(80) NOT NULL, train_as_of timestamptz NOT NULL,
      spec jsonb NOT NULL, spec_hash char(64) NOT NULL, status varchar(32) NOT NULL,
      evidence jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE TABLE direction_1d_model (
      model_id uuid PRIMARY KEY, run_id uuid NOT NULL REFERENCES direction_1d_training_run,
      cohort_id varchar(80) NOT NULL, group_id varchar(32) NOT NULL, horizon integer NOT NULL CHECK(horizon=1),
      file_name varchar(128) NOT NULL, content_hash char(64) NOT NULL, metadata jsonb NOT NULL,
      trained_at timestamptz NOT NULL, registered_at timestamptz NOT NULL DEFAULT clock_timestamp(),
      expires_at timestamptz NOT NULL, UNIQUE(cohort_id,group_id,run_id));
    CREATE TABLE direction_1d_window_model (
      cohort_id varchar(80) NOT NULL, group_id varchar(32) NOT NULL, base_nav_date date NOT NULL,
      target_nav_date date NOT NULL, branch_id varchar(16) NOT NULL CHECK(branch_id IN ('FIXED','WEEKLY')),
      model_id uuid REFERENCES direction_1d_model, locked_at timestamptz NOT NULL DEFAULT clock_timestamp(),
      PRIMARY KEY(cohort_id,group_id,target_nav_date,branch_id));
    CREATE TABLE direction_1d_job (
      job_id uuid PRIMARY KEY, task_key varchar(256) NOT NULL UNIQUE, kind varchar(16) NOT NULL,
      state varchar(32) NOT NULL, payload jsonb NOT NULL, result jsonb,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp(), finished_at timestamptz);
    CREATE TABLE direction_1d_assessment_ack (
      task_key varchar(256) NOT NULL, label_hash char(64) NOT NULL,
      assessed_at timestamptz NOT NULL, received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
      PRIMARY KEY(task_key,label_hash));
    """)
    # 所有新增物理字段均补中文注释；JSON各字段契约随主实施文档及冻结协议保存。
    names = {
        "source_version": "公开来源实际读取版本，不倒填历史首次公布时间",
        "snapshot": "可复算的一日输入和答案不可变快照",
        "training_run": "固定配方训练运行和冻结清单",
        "model": "仅供一日实验的已核验模型，不是正式发布表",
        "window_model": "按窗口不可变的FIXED和WEEKLY映射",
        "job": "有界异步作业和稳定结果，不含用户身份",
        "assessment_ack": "Java先核对后的公共学习许可回执",
    }
    comments = {
        "version_id": "来源版本唯一编号",
        "fund_code": "公共基金份额代码",
        "source_id": "来源登记编号",
        "kind": "证据或作业类别",
        "business_date": "源数据业务日期",
        "content_hash": "规范原文SHA256",
        "payload": "不含用户身份的冻结业务原文",
        "received_at": "实际读取接纳时刻，非历史公布时刻",
        "stored_at": "数据库实际写入时刻",
        "expires_at": "来源许可范围内的证据失效时间",
        "snapshot_id": "证据快照唯一编号",
        "task_key": "协议及基金目标日幂等键",
        "as_of": "一致性读取界限",
        "created_at": "数据库创建时刻",
        "run_id": "冻结训练运行编号",
        "cohort_id": "冻结训练集合版本",
        "train_as_of": "成熟样本训练读取界限",
        "spec": "固定训练配方与名单",
        "spec_hash": "冻结配方摘要",
        "status": "技术状态，不表示预测有效性",
        "evidence": "训练复现与样本清单证据",
        "model_id": "注册模型编号",
        "group_id": "经过核验的资产组",
        "horizon": "预测交易日数，恒为1",
        "file_name": "允许目录内的模型文件名",
        "metadata": "特征标签与训练证据身份",
        "trained_at": "真实拟合完成时刻",
        "registered_at": "真实注册时刻",
        "base_nav_date": "基准交易日T",
        "target_nav_date": "相邻目标交易日U",
        "branch_id": "冻结或每周更新分支",
        "locked_at": "映射实际锁定时刻",
        "job_id": "异步作业编号",
        "state": "作业状态",
        "result": "稳定结果或脱敏失败原因",
        "finished_at": "真实结束时刻",
        "label_hash": "已核对答案摘要",
        "assessed_at": "Java真实核对完成时刻",
    }
    import sqlalchemy as sa

    connection = op.get_bind()
    for suffix, description in names.items():
        table = "direction_1d_" + suffix
        op.execute(f"COMMENT ON TABLE {table} IS '{description}'")
        for column in sa.inspect(connection).get_columns(table):
            op.execute(f"COMMENT ON COLUMN {table}.{column['name']} IS '{comments[column['name']]}'")
        if suffix != "job":
            op.execute(
                f"CREATE TRIGGER guard_{table} BEFORE UPDATE ON {table} "
                "FOR EACH ROW EXECUTE FUNCTION direction_1d_immutable()"
            )


def downgrade() -> None:
    raise RuntimeError("1日原始证据不能自动删除；停用功能并保留只读历史。")
