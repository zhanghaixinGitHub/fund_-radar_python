"""已确认计划先行的新研究；新报告和关联一起保存，绝不接收旧报告补绑定请求。"""

from datetime import UTC
from decimal import localcontext
from time import perf_counter
from uuid import UUID, uuid4

from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.db.session import get_nav_sample_storage_engine
from app.models.cash_planned_research import CashPlannedResearchBinding
from app.models.cash_reinvestment import CashResearchRun
from app.repositories.cash_planned_research import find_planned_research
from app.repositories.cash_policy_freeze import find_policy_freeze
from app.repositories.cash_reinvestment_research import find_research
from app.schemas.cash_exam_plan import CashExamPreparation
from app.schemas.cash_planned_research import CashPlannedResearch, CashPlannedResearchRequest
from app.schemas.cash_policy_freeze import CashPlannedRuleBinding
from app.services.cash_exam_preparation import summarize_cash_exam_preparation
from app.services.cash_policy_freeze import restore_policy_freeze, validate_policy_binding
from app.services.cash_prediction_check import inspect_cash_research
from app.services.cash_reinvestment_research import (
    evaluate_cash_dataset,
    load_cash_dataset_in_session,
    restore_research,
)
from app.services.cash_reinvestment_storage import cash_hash
from app.services.cash_release_policy import load_release_policy
from app.services.historical_nav_storage import HistoricalNavStorageError
from app.services.historical_nav_training import training_slot

logger = get_logger(__name__)


def binding_content_hash(row) -> str:
    """指纹包含全部关联及服务器时刻；它防错配，不是密码学审批签名。"""
    return cash_hash(
        {
            "version": "CASH_PLANNED_RESEARCH_BINDING_V1",
            "binding_id": str(row.binding_id),
            "request_key": str(row.request_key),
            "research_run_id": str(row.research_run_id),
            "policy_freeze_id": str(row.policy_freeze_id),
            "policy_freeze_hash": row.policy_freeze_hash,
            "report_hash": row.report_hash,
            "preparation": row.preparation,
            "evaluation_started_at": row.evaluation_started_at.astimezone(UTC).isoformat(),
            "completed_at": row.completed_at.astimezone(UTC).isoformat(),
        }
    )


def restore_planned_research(row, research_row, freeze_row, *, created=False) -> CashPlannedResearch:
    if row is None:
        raise HistoricalNavStorageError("CASH_PLANNED_RESEARCH_NOT_FOUND", "计划绑定研究不存在。", 404)
    try:
        frozen = restore_policy_freeze(freeze_row)
        stored = restore_research(research_row)
        preparation = CashExamPreparation.model_validate(row.preparation)
        if (
            not isinstance(frozen.snapshot.binding, CashPlannedRuleBinding)
            or row.research_run_id != stored.run_id
            or row.binding_id != stored.request_key
            or row.policy_freeze_id != frozen.freeze_id
            or row.policy_freeze_hash != frozen.content_hash
            or row.report_hash != stored.report.report_hash
            or stored.created_at != row.completed_at
            or preparation.plan != frozen.snapshot.binding.exam_plan
            or preparation.preparation != stored.report.preparation
            or preparation.model_dump(mode="json") != row.preparation
            or row.evaluation_started_at.tzinfo is None
            or row.completed_at.tzinfo is None
            or row.content_hash != binding_content_hash(row)
        ):
            raise ValueError("planned research identity, plan or report mismatch")
        # 与报告计算、发布审查使用相同精度，避免调用方Decimal上下文造成假损坏。
        with localcontext() as context:
            context.prec = 28
            inspect_cash_research(stored)  # 共用全部固定窗口及数值/分档一致性检查，不因新绑定跳过。
        groups = {(c.window_id, c.fund_code): c for c in preparation.coverage}
        for window in stored.report.windows:
            for fund in window.funds:
                if fund.counts["EXAM"] != groups[window.window.window_id, fund.fund_code].usable_count:
                    raise ValueError("evaluated exam population differs from bound preparation")
        return CashPlannedResearch(
            binding_id=row.binding_id,
            request_key=row.request_key,
            evaluation_started_at=row.evaluation_started_at,
            completed_at=row.completed_at,
            policy_freeze=frozen,
            preparation=preparation,
            research=stored,
            content_hash=row.content_hash,
            created=created,
            database_written=created,
        )
    except (ValueError, TypeError, AttributeError, KeyError, ArithmeticError, HistoricalNavStorageError) as error:
        raise HistoricalNavStorageError(
            "CASH_PLANNED_RESEARCH_CORRUPTED", "计划、计算时刻或新研究关联无法核验，不能作为发布证据。", 503
        ) from error


def read_planned_in_session(session, row) -> CashPlannedResearch:
    return restore_planned_research(
        row,
        find_research(session, run_id=row.research_run_id) if row is not None else None,
        find_policy_freeze(session, freeze_id=row.policy_freeze_id) if row is not None else None,
    )


def get_planned_research(binding_id: UUID) -> CashPlannedResearch:
    """按编号只读历史关联，不依赖今天的审批配置，不训练或重算输入。"""
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        return read_planned_in_session(session, find_planned_research(session, binding_id=binding_id))


def _retry(session, row, request, policy):
    saved = read_planned_in_session(session, row)
    if (
        saved.policy_freeze.freeze_id != request.policy_freeze_id
        or saved.policy_freeze.content_hash != request.expected_policy_freeze_hash
        or saved.preparation.preparation.dataset_hash != request.expected_dataset_hash
        or saved.preparation.preparation.batch_ids != tuple(sorted(request.batch_ids))
    ):
        raise HistoricalNavStorageError("REQUEST_KEY_CONFLICT", "同一计划研究请求号已用于不同规则或资料。", 409)
    validate_policy_binding(saved.policy_freeze, policy)
    return saved


def save_planned_research(request: CashPlannedResearchRequest) -> CashPlannedResearch:
    """先核验已确认快照及资料，再评估；计算不用数据库连接，保存失败两条记录一起回滚。"""
    policy = load_release_policy()
    if policy.approval_state != "APPROVED":
        raise HistoricalNavStorageError("CASH_POLICY_APPROVAL_REQUIRED", "规则仍是草案，不能启动计划绑定研究。", 409)
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        previous = find_planned_research(session, request_key=request.request_key)
        if previous is not None:
            return _retry(session, previous, request, policy)
    with training_slot():
        with Session(get_nav_sample_storage_engine()) as session, session.begin():
            session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            previous = find_planned_research(session, request_key=request.request_key)
            if previous is not None:
                return _retry(session, previous, request, policy)
            frozen = restore_policy_freeze(find_policy_freeze(session, freeze_id=request.policy_freeze_id))
            if frozen.content_hash != request.expected_policy_freeze_hash:
                raise HistoricalNavStorageError("CASH_POLICY_FREEZE_HASH_MISMATCH", "规则快照完整指纹不一致。", 409)
            validate_policy_binding(frozen, policy)
            data = load_cash_dataset_in_session(session, request, deadline=perf_counter() + 30)
            if data.report.dataset_hash != request.expected_dataset_hash:
                raise HistoricalNavStorageError("DATASET_HASH_MISMATCH", "资料指纹变化，未启动计算。", 409)
            # 与随后交给研究器的可变嵌套字典彻底分离，避免计算中的意外修改回写“计算前”底稿。
            preparation = CashExamPreparation.model_validate_json(
                summarize_cash_exam_preparation(data, frozen.snapshot.binding.exam_plan).model_dump_json()
            )
            started = session.scalar(select(func.clock_timestamp()))
            if started < frozen.frozen_at:
                raise HistoricalNavStorageError(
                    "CASH_POLICY_FREEZE_TIME_INVALID", "规则冻结时刻晚于当前数据库时间。", 503
                )
        logger.info(
            "cash_planned_research.save >>> evaluation start, trace_id=%s, request_key=%s, "
            "policy_freeze_id=%s, dataset_hash=%s",
            get_trace_id(),
            request.request_key,
            frozen.freeze_id,
            data.report.dataset_hash,
        )
        report = evaluate_cash_dataset(data)  # 至此快照已持久存在且资料被选定；不能在计算后更换或补绑旧模型。
        binding_id, run_id = uuid4(), uuid4()
        for attempt in range(2):
            try:
                with Session(get_nav_sample_storage_engine()) as session, session.begin():
                    previous = find_planned_research(session, request_key=request.request_key)
                    if previous is not None:
                        return _retry(session, previous, request, policy)
                    completed = session.scalar(select(func.clock_timestamp()))
                    research = CashResearchRun(
                        run_id=run_id,
                        request_key=binding_id,  # 服务器独立生成，不能用调用者的旧研究请求号命中旧报告。
                        dataset_hash=data.report.dataset_hash,
                        report=report.model_dump(mode="json"),
                        publication_status="MODEL_NOT_RELEASED",
                        created_at=completed,
                    )
                    row = CashPlannedResearchBinding(
                        binding_id=binding_id,
                        request_key=request.request_key,
                        research_run_id=run_id,
                        policy_freeze_id=frozen.freeze_id,
                        policy_freeze_hash=frozen.content_hash,
                        report_hash=report.report_hash,
                        preparation=preparation.model_dump(mode="json"),
                        evaluation_started_at=started,
                        completed_at=completed,
                    )
                    row.content_hash = binding_content_hash(row)
                    session.add(research)
                    session.flush()
                    session.add(row)
                    session.flush()
                    # 从数据库重读关联快照，确保FK指向实际存在、完整一致的已确认记录。
                    return restore_planned_research(
                        row, research, find_policy_freeze(session, freeze_id=frozen.freeze_id), created=True
                    )
            except DBAPIError as error:
                constraint = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
                if attempt or constraint != "uq_cash_planned_request":
                    raise
    raise RuntimeError("unreachable")
