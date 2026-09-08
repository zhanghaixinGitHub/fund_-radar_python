"""已确认的三基金2022–2025试点；默认只读预检，--execute才新增月度样本批次。

在项目根目录用python -m scripts.historical_nav_baseline_pilot --run-key UUID运行。
再次执行沿用run-key；每个月凭证由它稳定派生，中断后重跑会复用已保存批次。
本脚本不接收任意基金或日期，不升级表、不同步来源、不训练/发布、不写报告文件。
"""

import argparse
import json
from collections import Counter
from datetime import date, timedelta
from time import perf_counter
from uuid import UUID, uuid5

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.historical_nav_evaluation import EvaluationProtocol, HistoricalNavEvaluationRequest
from app.schemas.historical_nav_storage import HistoricalNavBatchSaveRequest
from app.services.historical_nav_evaluation import _sample_split, evaluate_stored_historical_nav_batches
from app.services.historical_nav_preview import preview_stored_historical_nav_batch
from app.services.historical_nav_samples import (
    HISTORICAL_NAV_FEATURE_VERSION,
    HISTORICAL_NAV_LABEL_VERSION,
    HISTORICAL_NAV_SAMPLE_RULE_VERSION,
)
from app.services.historical_nav_storage import save_historical_nav_batch
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

logger = get_logger(__name__)
PILOT_FUNDS = ("001632", "006730", "008888")
PILOT_PROTOCOL = EvaluationProtocol(
    train_start_date=date(2022, 1, 1),
    train_end_date=date(2023, 12, 31),
    validation_end_date=date(2024, 12, 31),
    test_end_date=date(2025, 12, 31),
)


def pilot_requests(run_key: UUID) -> tuple[HistoricalNavBatchSaveRequest, ...]:
    """仅生成144个已授权月度请求；凭证含规则版本，读库分页不参与身份。"""
    requests = []
    for fund_code in PILOT_FUNDS:
        for year in range(2022, 2026):
            for month in range(1, 13):
                start = date(year, month, 1)
                next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
                end = next_month - timedelta(days=1)
                identity = ":".join(
                    (
                        "NAV_BASELINE_PILOT_V1",
                        fund_code,
                        start.isoformat(),
                        end.isoformat(),
                        HISTORICAL_NAV_FEATURE_VERSION,
                        HISTORICAL_NAV_SAMPLE_RULE_VERSION,
                        HISTORICAL_NAV_LABEL_VERSION,
                    )
                )
                requests.append(
                    HistoricalNavBatchSaveRequest(
                        fund_code=fund_code,
                        start_date=start,
                        end_date=end,
                        page_size=30,
                        request_key=uuid5(run_key, identity),
                    )
                )
    return tuple(requests)


def preview_pilot(requests: tuple[HistoricalNavBatchSaveRequest, ...]) -> dict:
    """只读逐月现有净值，统计真实过滤后的各段数量，不生成或伪造存储批次编号。"""
    funds = {}
    for fund_code in PILOT_FUNDS:
        samples = 0
        counts, excluded = Counter(), Counter()
        for request in requests:
            if request.fund_code != fund_code:
                continue
            result = preview_stored_historical_nav_batch(request)
            samples += result.sample_count
            for sample in result.items:
                split, reason = _sample_split(sample, PILOT_PROTOCOL)
                if reason:
                    excluded[reason] += 1
                else:
                    counts[split] += 1
        funds[fund_code] = {
            "input_samples": samples,
            "split_counts": {s: counts[s] for s in PILOT_PROTOCOL.minimum_samples_per_fund},
            "excluded_reasons": dict(sorted(excluded.items())),
            "missing_samples": {s: max(0, n - counts[s]) for s, n in PILOT_PROTOCOL.minimum_samples_per_fund.items()},
        }
    return {
        "planned_batches": len(requests),
        "funds": funds,
        "sufficient": all(not any(f["missing_samples"].values()) for f in funds.values()),
    }


def execute_pilot(requests: tuple[HistoricalNavBatchSaveRequest, ...]) -> dict:
    """每月整批事务成功后才收集编号；不是跨144个月的大事务，失败后可沿用凭证续跑。"""
    batch_ids = []
    created = 0
    for request in requests:
        batch, is_new = save_historical_nav_batch(request)
        batch_ids.append(batch.batch_id)
        created += int(is_new)
    evaluation_request = HistoricalNavEvaluationRequest(
        batch_ids=tuple(batch_ids),
        train_start_date=PILOT_PROTOCOL.train_start_date,
        train_end_date=PILOT_PROTOCOL.train_end_date,
        validation_end_date=PILOT_PROTOCOL.validation_end_date,
        test_end_date=PILOT_PROTOCOL.test_end_date,
    )
    report = evaluate_stored_historical_nav_batches(evaluation_request)
    return {
        "created_batches": created,
        "reused_batches": len(batch_ids) - created,
        "evaluation_request": evaluation_request.model_dump(mode="json", by_alias=True),
        "report": report.model_dump(mode="json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", type=UUID, required=True, help="本次试点的UUID凭证，中断/重试必须沿用")
    parser.add_argument("--execute", action="store_true", help="预检满足样本量后实际保存；不加此参数只读")
    args = parser.parse_args()
    started = perf_counter()
    logger.info("historical_nav_baseline_pilot.main >>> started, run_key=%s, execute=%s", args.run_key, args.execute)
    try:
        # 本轮只获准在本机fund_ai执行，不允许因换了.env而误写到其他库或远程环境。
        target = make_url(get_settings().ai_database_url)
        if target.host not in {"localhost", "127.0.0.1", "::1"} or target.database != "fund_ai":
            raise ValueError("pilot is restricted to local fund_ai")
        requests = pilot_requests(args.run_key)
        preflight = preview_pilot(requests)
        result = {
            "run_key": str(args.run_key),
            "mode": "EXECUTE" if args.execute else "DRY_RUN",
            "preflight": preflight,
        }
        if args.execute and preflight["sufficient"]:
            result["execution"] = execute_pilot(requests)
        elif args.execute:
            result["status"] = "BLOCKED_BY_SAMPLE_SHORTAGE"
        print("PILOT_RESULT=" + json.dumps(result, ensure_ascii=False))
        logger.info(
            "historical_nav_baseline_pilot.main >>> completed, run_key=%s, elapsed_ms=%.2f",
            args.run_key,
            (perf_counter() - started) * 1000,
        )
        return 0 if not args.execute or preflight["sufficient"] else 2
    except (SQLAlchemyError, RuntimeError, ValueError, ArithmeticError):
        logger.exception("historical_nav_baseline_pilot.main >>> failed, run_key=%s; reuse key to resume", args.run_key)
        # 异常不输出连接信息或Token。已完成月份保留；不能为假装整次原子性而删除已存批次。
        print("PILOT_RESULT=" + json.dumps({"run_key": str(args.run_key), "status": "FAILED_RESUMABLE"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
