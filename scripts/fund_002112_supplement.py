"""本轮额外公开数据采集入口；只读供应商、写本地资料，不训练或选用新模型。"""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import get_engine  # noqa: E402
from app.integrations.cninfo_exposure import acquire_announcements, acquire_attachments  # noqa: E402
from app.integrations.dbfund_news_mirrors import recover_news_mirrors  # noqa: E402
from app.integrations.dbfund_supplement import acquire_documents, acquire_nav, recover_public_documents  # noqa: E402
from app.services.fund_exposure_field_enrichment import acquire_full_fields  # noqa: E402
from app.services.fund_exposure_quotes import safe_error  # noqa: E402
from app.services.fund_exposure_supplement import (  # noqa: E402
    acquire_financials,
    acquire_operating_data,
    acquire_stock_context,
    plan,
)
from app.services.fund_exposure_supplement_audit import audit_supplement  # noqa: E402
from sqlalchemy import text  # noqa: E402


@contextmanager
def command_lock(command):
    """同类补齐任务不能重复启动；财务任务共用一个锁，避免双进程叠加突破账号限速。"""
    groups = {
        "financials": 211201,
        "operating": 211201,
        "context": 211201,
        "documents": 211202,
        "recover-documents": 211202,
        "recover-news": 211202,
        "announcements": 211203,
        "attachments": 211204,
        "audit": 211205,
        "plan": 211206,
        "full-fields": 211207,
    }
    # 官网净值和原输入维护使用相同锁，避免两个过程同时替换当前来源索引。
    group = 2112 if command == "official-nav" else groups[command]
    with get_engine().connect() as c:
        locked = c.execute(text("SELECT pg_try_advisory_lock(20260924,:group)"), {"group": group}).scalar_one()
        if not locked:
            raise ValueError("EXPOSURE_SUPPLEMENT_TASK_ALREADY_RUNNING")
        try:
            yield
        finally:
            c.execute(text("SELECT pg_advisory_unlock(20260924,:group)"), {"group": group})


def main():
    parser = argparse.ArgumentParser(description="002112 第二批零新增采购数据补齐")
    commands = {
        "plan": plan,
        "financials": acquire_financials,
        "operating": acquire_operating_data,
        "full-fields": acquire_full_fields,
        "context": acquire_stock_context,
        "official-nav": acquire_nav,
        "documents": acquire_documents,
        "recover-documents": recover_public_documents,
        "recover-news": recover_news_mirrors,
        "announcements": acquire_announcements,
        "attachments": acquire_attachments,
        "audit": audit_supplement,
    }
    parser.add_argument("command", choices=commands)
    command = parser.parse_args().command
    with command_lock(command):
        print(json.dumps(commands[command](), ensure_ascii=False, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "reason": safe_error(exc)}))
        raise SystemExit(1) from None
