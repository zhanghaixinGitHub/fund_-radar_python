"""既定三试点的本机研究运行；显式--execute才写独立现金表，可安全重试。

不下载、不更新原净值、不复制旧样本、不读取2025数值。标准输出仅进度/编号/汇总；
完整研究报告已在数据库保存。运行前先执行Alembic迁移。
"""

import argparse
import calendar as month_calendar
import json
from datetime import date
from uuid import UUID, uuid5

from app.core.config import get_settings
from app.db.session import get_nav_sample_storage_engine
from app.schemas.cash_reinvestment_research import CashPrepareRequest, CashResearchRequest
from app.schemas.cash_reinvestment_storage import CashBatchSaveRequest
from app.services.cash_reinvestment_research import FUNDS, load_cash_dataset, save_cash_research
from app.services.cash_reinvestment_storage import get_cash_batch, save_cash_batch
from app.services.trading_calendar import load_calendar
from sqlalchemy import text
from sqlalchemy.engine import make_url

# 固定该轮操作命名空间；重启脚本不会产生重复批次。改变实验数据须明确新版本/新凭证。
NAMESPACE = UUID("a5ac7832-4b8d-4d24-ac41-c0f45f32525f")


def pilot_requests():
    calendar = load_calendar()
    last = max(d for d in calendar.sessions if d.year == 2024)
    latest_cutoff = calendar.sessions[calendar.at_or_before_index(last) - 20]
    for fund in FUNDS:
        for year in range(2022, 2025):
            for month in range(1, 13):
                start = date(year, month, 1)
                end = min(date(year, month, month_calendar.monthrange(year, month)[1]), latest_cutoff)
                if start > end:
                    continue
                key = uuid5(NAMESPACE, f"cash-v1:{fund}:{start}:{end}")
                yield CashBatchSaveRequest(fundCode=fund, startDate=start, endDate=end, pageSize=30, requestKey=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="明确执行本机独立现金批次和研究报告写入")
    args = parser.parse_args()
    requests = tuple(pilot_requests())
    print(
        json.dumps(
            {
                "mode": "EXECUTE" if args.execute else "PLAN",
                "batch_count": len(requests),
                "funds": FUNDS,
                "cutoff_end": str(requests[-1].end_date),
                "test_scored": False,
            }
        ),
        flush=True,
    )
    if not args.execute:
        return
    url = make_url(get_settings().ai_database_url)
    if url.host not in {"localhost", "127.0.0.1", "::1"} or url.database != "fund_ai":
        raise RuntimeError("pilot writes only the configured local fund_ai")
    with get_nav_sample_storage_engine().connect() as conn:
        if conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one() != "20260908_14":
            raise RuntimeError("cash schema migration required")
    batches = []
    for index, request in enumerate(requests, 1):
        stored, created = save_cash_batch(request)
        if get_cash_batch(stored.batch_id) != stored:
            raise RuntimeError("batch readback differs")
        batches.append(stored.batch_id)
        if index % 12 == 0 or index == len(requests):
            print(
                json.dumps(
                    {
                        "saved": index,
                        "total": len(requests),
                        "last_fund": request.fund_code,
                        "last_cutoff": str(request.end_date),
                        "last_created": created,
                    }
                ),
                flush=True,
            )
    data = load_cash_dataset(CashPrepareRequest(batchIds=batches))
    print(json.dumps({"preparation": data.report.model_dump(mode="json")}, ensure_ascii=False), flush=True)
    result, created = save_cash_research(
        CashResearchRequest(
            batchIds=batches,
            expectedDatasetHash=data.report.dataset_hash,
            requestKey=uuid5(NAMESPACE, "cash-research-v1:" + data.report.dataset_hash),
        )
    )
    print(
        json.dumps(
            {
                "run_id": str(result.run_id),
                "created": created,
                "report_hash": result.report.report_hash,
                "status": result.report.status,
                "model_fitted": result.report.model_fitted,
                "release_gate": result.report.release_gate,
                "release_blockers": result.report.release_blockers,
                "windows": [
                    {
                        "window": w.window.window_id,
                        "status": w.status,
                        "reason": w.reason,
                        "counts": [f.model_dump(mode="json") for f in w.funds],
                        "brier": str(w.after.validation.brier_score) if w.after else None,
                        "ece": str(w.reliability_after.ece) if w.reliability_after else None,
                    }
                    for w in result.report.windows
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
