"""现金口径七列X/独立y -> 固定时间研究 -> 不可变报告；不触碰2025数值。"""

from collections import Counter
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from time import perf_counter
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.db.session import get_nav_sample_storage_engine
from app.models.cash_reinvestment import CashResearchRun
from app.repositories.cash_reinvestment_research import find_research, iter_cash_batches
from app.schemas.cash_reinvestment_research import (
    CashPreparation,
    CashPrepareRequest,
    CashResearchReport,
    CashResearchRequest,
    CashStoredResearch,
)
from app.schemas.historical_nav_calibration import CalibrationWindowReport, WindowFundCounts
from app.services.cash_reinvestment_storage import cash_hash, restore_cash_batch
from app.services.historical_nav_calibration import (
    WINDOWS,
    evaluate_calibration_window_rows,
    restore_calibrated_artifact,
)
from app.services.historical_nav_evaluation import FEATURE_NAMES, PreparedRow
from app.services.historical_nav_storage import HistoricalNavStorageError
from app.services.historical_nav_training import training_slot

FUNDS = ("001632", "006730", "008888")
VERSIONS = {
    "feature": "CASH_REINVESTMENT_FEATURE_V1",
    "label": "CASH_REINVESTMENT_FORWARD_20TD_V1",
    "sample": "CASH_REINVESTMENT_SAMPLE_RULE_V1",
    "research": "CASH_RESEARCH_PROTOCOL_V1",
}
BLOCKERS = (
    "HISTORICAL_FIRST_VERSIONS_UNVERIFIED",
    "DIVIDEND_COMPLETENESS_UNVERIFIED",
    "NON_CASH_ADJUSTMENTS_UNVERIFIED",
    "INDEPENDENT_TEST_NOT_EVALUATED",
)
START, TRAIN_END, END = date(2022, 1, 1), date(2023, 12, 31), date(2024, 12, 31)


@dataclass(frozen=True)
class CashDataset:
    rows: tuple[PreparedRow, ...]  # available_at在适配器内明确使用cutoff，不用旧版净值日起题。
    report: CashPreparation


def prepare_cash_batches(batches, *, deadline: float) -> CashDataset:
    """流式读批次后只保留小矩阵；冲突拒绝，不随意选最新值，未来答案不进入X。"""
    seen, rows, batch_hashes = {}, [], []
    excluded = Counter()
    input_count = duplicates = 0
    for stored in batches:
        if perf_counter() >= deadline:
            raise TimeoutError("cash preparation timeout")
        preview = stored.preview
        batch_hashes.append((str(stored.batch_id), preview.batch_hash))
        for item in preview.items:
            input_count += 1
            identity = item.fund_code, item.cutoff_date
            content_hash = cash_hash(item.model_dump(mode="json"))
            if identity in seen:
                if seen[identity] != content_hash:
                    raise HistoricalNavStorageError(
                        "CASH_SAMPLE_CONFLICT", "同一基金/截止日有不同内容，须明确选择一套批次。", 409
                    )
                duplicates += 1
                continue
            seen[identity] = content_hash
            if item.status != "RESEARCH_SAMPLE_READY":
                excluded[item.status] += 1
                continue
            feature, label = item.feature_payload, item.offline_label
            if not START <= item.cutoff_date <= END or label.label_available_at > END:
                raise HistoricalNavStorageError("TEST_PERIOD_PROTECTED", "研究资料不得进入2025数值或答案。", 409)
            x = tuple(Decimal(feature.metrics[name]) for name in FEATURE_NAMES)
            if any(not n.is_finite() for n in x):
                raise ValueError("cash matrix nonfinite")
            rows.append(
                PreparedRow(
                    batch_id=stored.batch_id,
                    fund_code=item.fund_code,
                    as_of_date=item.cutoff_date,
                    available_at=item.cutoff_date,
                    label_available_at=label.label_available_at,
                    x=x,
                    y=int(label.label_up_20d),
                    content_hash=content_hash,
                )
            )
    rows.sort(key=lambda r: (r.fund_code, r.as_of_date))
    counts = {
        fund: {
            "TRAIN": sum(
                r.fund_code == fund and r.available_at <= TRAIN_END and r.label_available_at <= TRAIN_END for r in rows
            ),
            "VALIDATION": sum(
                r.fund_code == fund and TRAIN_END < r.available_at <= END and r.label_available_at <= END for r in rows
            ),
        }
        for fund in FUNDS
    }
    missing = {
        fund: {stage: max(0, minimum - counts[fund][stage]) for stage, minimum in (("TRAIN", 252), ("VALIDATION", 120))}
        for fund in FUNDS
    }
    batch_hashes.sort()
    from uuid import UUID

    report = CashPreparation(
        status="INSUFFICIENT_DATA" if any(any(g.values()) for g in missing.values()) else "RESEARCH_READY",
        batch_ids=tuple(UUID(key) for key, _ in batch_hashes),
        dataset_hash=cash_hash({"versions": VERSIONS, "batches": batch_hashes}),
        feature_names=FEATURE_NAMES,
        versions=VERSIONS,
        input_count=input_count,
        duplicate_count=duplicates,
        unique_count=len(seen),
        usable_count=len(rows),
        excluded_reasons=dict(sorted(excluded.items())),
        fund_counts=counts,
        missing_counts=missing,
        admission_blockers=BLOCKERS,
    )
    return CashDataset(tuple(rows), report)


def load_cash_dataset(request: CashPrepareRequest) -> CashDataset:
    deadline = perf_counter() + 30
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        return load_cash_dataset_in_session(session, request, deadline=deadline)


def load_cash_dataset_in_session(session: Session, request: CashPrepareRequest, *, deadline: float) -> CashDataset:
    """供已开启只读快照的流程复用；不另开连接，不绕过日期先行和完整性检查。"""
    batches = (restore_cash_batch(b, rows) for b, rows in iter_cash_batches(session, tuple(sorted(request.batch_ids))))
    return prepare_cash_batches(batches, deadline=deadline)


def cash_window_rows(data: CashDataset, window):
    """每段按cutoff划分，标签公告越过本段末则剔除；没有随机打散或复用考试答案。"""
    periods = (
        ("FIT", date(2021, 12, 31), window.fit_end_date, 252),
        ("CALIBRATION", window.fit_end_date, window.calibration_end_date, 60),
        ("EXAM", window.calibration_end_date, window.evaluation_end_date, window.minimum_exam_per_fund),
    )
    rows, purged, minimums = {}, {}, {}
    for name, lower, upper, minimum in periods:
        selected = tuple(r for r in data.rows if lower < r.available_at <= upper)
        rows[name] = tuple(r for r in selected if r.label_available_at <= upper)
        purged[name] = Counter(r.fund_code for r in selected if r.label_available_at > upper)
        minimums[name] = minimum
    counts = []
    for fund in FUNDS:
        actual = {stage: sum(r.fund_code == fund for r in subset) for stage, subset in rows.items()}
        counts.append(
            WindowFundCounts(
                fund_code=fund,
                counts=actual,
                missing={s: max(0, minimums[s] - n) for s, n in actual.items()},
                purged={s: purged[s][fund] for s in rows},
            )
        )
    return rows, tuple(counts)


def evaluate_cash_dataset(data: CashDataset) -> CashResearchReport:
    """沿用冻结三窗纯数学算法，来源口径及切分独立；研究成功仍不授权产品概率。"""
    deadline = perf_counter() + 60
    results = []
    for window in WINDOWS:
        rows, counts = cash_window_rows(data, window)
        if data.report.status == "INSUFFICIENT_DATA":
            result = CalibrationWindowReport(
                window=window, status="INSUFFICIENT_DATA", funds=counts, reason="GLOBAL_RESEARCH_DATA_INSUFFICIENT"
            )
        else:
            history = tuple(
                r
                for r in data.rows
                if r.available_at <= window.calibration_end_date and r.label_available_at <= window.calibration_end_date
            )
            result = evaluate_calibration_window_rows(
                rows, counts, history, VERSIONS, window, preview_size=0, deadline=deadline
            )
        results.append(result)
    evaluated = sum(w.status == "EVALUATED" for w in results)
    blockers = (*BLOCKERS, *(f"{w.window.window_id}:{w.reason}" for w in results if w.status != "EVALUATED"))
    result = CashResearchReport(
        status="EVALUATED" if evaluated == 3 else "PARTIAL_EVALUATION" if evaluated else "INSUFFICIENT_DATA",
        preparation=data.report,
        windows=tuple(results),
        model_fitted=evaluated > 0,
        release_blockers=blockers,
        report_hash="",
    )
    return result.model_copy(update={"report_hash": cash_hash(result.model_dump(mode="json", exclude={"report_hash"}))})


def restore_research(row) -> CashStoredResearch:
    if row is None:
        raise HistoricalNavStorageError("CASH_RESEARCH_NOT_FOUND", "现金研究记录不存在。", 404)
    try:
        report = CashResearchReport.model_validate(row.report)
        if (
            report.report_hash != cash_hash(report.model_dump(mode="json", exclude={"report_hash"}))
            or row.dataset_hash != report.preparation.dataset_hash
            or row.publication_status != "MODEL_NOT_RELEASED"
            or report.preparation.versions != VERSIONS
            or not set(BLOCKERS).issubset(report.release_blockers)
        ):
            raise ValueError("research integrity mismatch")
        for window in report.windows:
            if window.model:
                model = restore_calibrated_artifact(window.model.model_dump_json())
                if model.base_model.versions != VERSIONS:
                    raise ValueError("wrong data basis model")
        return CashStoredResearch(
            run_id=row.run_id, request_key=row.request_key, created_at=row.created_at, report=report
        )
    except (ValueError, TypeError, KeyError) as error:
        raise HistoricalNavStorageError("CASH_RESEARCH_CORRUPTED", "研究报告或模型完整性校验失败。", 503) from error


def get_cash_research(run_id):
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        return restore_research(find_research(session, run_id=run_id))


def _retry_result(row, request):
    if row.dataset_hash != request.expected_dataset_hash or tuple(row.report["preparation"]["batch_ids"]) != tuple(
        str(i) for i in sorted(request.batch_ids)
    ):
        raise HistoricalNavStorageError("REQUEST_KEY_CONFLICT", "该研究requestKey已用于不同资料，请勿换参数重试。", 409)
    return restore_research(row), False


def save_cash_research(request: CashResearchRequest):
    """先读完快照并释放连接，再占计算槽；同key重试不重新训练，写入失败不留下半报告。"""
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        existing = find_research(session, request_key=request.request_key)
        if existing:
            return _retry_result(existing, request)
    with training_slot():
        data = load_cash_dataset(request)
        if data.report.dataset_hash != request.expected_dataset_hash:
            raise HistoricalNavStorageError("DATASET_HASH_MISMATCH", "资料指纹已变化，请重新核对准备报告。", 409)
        report = evaluate_cash_dataset(data)
    for attempt in range(2):
        try:
            with Session(get_nav_sample_storage_engine()) as session, session.begin():
                existing = find_research(session, request_key=request.request_key)
                if existing:
                    return _retry_result(existing, request)
                row = CashResearchRun(
                    run_id=uuid4(),
                    request_key=request.request_key,
                    dataset_hash=data.report.dataset_hash,
                    report=report.model_dump(mode="json"),
                    publication_status="MODEL_NOT_RELEASED",
                )
                session.add(row)
                session.flush()
                return restore_research(row), True
        except DBAPIError as error:
            if (
                attempt
                or getattr(getattr(error.orig, "diag", None), "constraint_name", None) != "uq_cash_research_request"
            ):
                raise
    raise RuntimeError("unreachable")
