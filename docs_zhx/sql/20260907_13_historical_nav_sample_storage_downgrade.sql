-- 历史净值学习样本存储：PostgreSQL 等价原生回退 DDL（20260907_13 -> 20260905_12）。
-- 破坏性操作：会删除三张新表的结构、索引和注释；本轮未执行。
-- 仅允许三表全部为空（数据量阈值为 0 行）；先加锁，任一表有数据或锁不可得则拒绝。
-- 执行前暂停使用这些表的代码、确认目标 fund_ai 并备份；失败后执行 ROLLBACK。
-- 不使用 CASCADE，不删除任何既有业务表。优先 Alembic downgrade 20260905_12，勿重复执行。
-- 本文件不更新 alembic_version；有数据时保留表，另行评审备份/回退方案，不绕过保护。
-- BEGIN GENERATED ALEMBIC SQL
BEGIN;

LOCK TABLE historical_nav_sample_label, historical_nav_sample, historical_nav_sample_batch IN ACCESS EXCLUSIVE MODE NOWAIT;

DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM historical_nav_sample_label)
                OR EXISTS (SELECT 1 FROM historical_nav_sample)
                OR EXISTS (SELECT 1 FROM historical_nav_sample_batch) THEN
                RAISE EXCEPTION 'Historical NAV sample tables are not empty; downgrade refused';
            END IF;
        END;
        $$;

DROP TABLE historical_nav_sample_label;

DROP TABLE historical_nav_sample;

DROP INDEX ix_historical_nav_batch_fund_created;

DROP TABLE historical_nav_sample_batch;

COMMIT;
