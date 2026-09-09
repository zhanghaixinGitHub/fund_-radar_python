"""冻结服务器已确认的规则，不能替人审批、解封独立测试或签发模型授权。"""

from datetime import UTC
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.db.session import get_nav_sample_storage_engine
from app.models.cash_policy_freeze import CashPolicyFreezeRecord
from app.repositories.cash_policy_freeze import find_policy_freeze
from app.schemas.cash_policy_freeze import (
    CashFrozenPolicy,
    CashPolicyDescriptor,
    CashPolicyFreezeRequest,
    CashPolicySnapshot,
    CashRuleBinding,
)
from app.schemas.cash_release_review import CashReleasePolicy
from app.services.cash_reinvestment_research import VERSIONS
from app.services.cash_reinvestment_storage import cash_hash
from app.services.cash_release_policy import load_release_policy
from app.services.historical_nav_calibration import WINDOWS
from app.services.historical_nav_storage import HistoricalNavStorageError


def current_rule_binding() -> CashRuleBinding:
    """冻结比较约定的内容版本；不是完整独立测试执行协议或运行代码签名。"""
    return CashRuleBinding(
        windows=WINDOWS, feature_versions=dict(VERSIONS), bin_edges=tuple(Decimal(i) / 5 for i in range(6))
    )


def describe_cash_policy() -> CashPolicyDescriptor:
    policy, binding = load_release_policy(), current_rule_binding()
    return CashPolicyDescriptor(
        policy=policy,
        binding=binding,
        policy_hash=cash_hash(policy.model_dump(mode="json")),
        binding_hash=cash_hash(binding.model_dump(mode="json")),
        approval_ready=policy.approval_state == "APPROVED",
    )


def _content_hash(row) -> str:
    return cash_hash(
        {
            "freeze_id": str(row.freeze_id),
            "request_key": str(row.request_key),
            "frozen_at": row.created_at.astimezone(UTC).isoformat(),
            "policy_version": row.policy_version,
            "policy_hash": row.policy_hash,
            "binding_hash": row.binding_hash,
            "snapshot": row.snapshot,
        }
    )


def restore_policy_freeze(row, *, created=False) -> CashFrozenPolicy:
    if row is None:
        raise HistoricalNavStorageError("CASH_POLICY_FREEZE_NOT_FOUND", "规则冻结记录不存在。", 404)
    try:
        snapshot = CashPolicySnapshot.model_validate(row.snapshot)
        if (
            row.created_at.tzinfo is None
            or snapshot.policy.version != row.policy_version
            or cash_hash(snapshot.policy.model_dump(mode="json")) != row.policy_hash
            or cash_hash(snapshot.binding.model_dump(mode="json")) != row.binding_hash
            or _content_hash(row) != row.content_hash
        ):
            raise ValueError("policy snapshot integrity mismatch")
        return CashFrozenPolicy(
            freeze_id=row.freeze_id,
            request_key=row.request_key,
            frozen_at=row.created_at,
            policy_hash=row.policy_hash,
            binding_hash=row.binding_hash,
            content_hash=row.content_hash,
            snapshot=snapshot,
            created=created,
            database_written=created,
        )
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise HistoricalNavStorageError(
            "CASH_POLICY_FREEZE_CORRUPTED", "规则冻结记录校验失败，不能作为发布依据。", 503
        ) from error


def validate_policy_binding(frozen: CashFrozenPolicy, policy: CashReleasePolicy) -> None:
    """内容必须等于本次服务器规则及比较约定；旧快照存在不表示适用于新配置。"""
    if frozen.policy_hash != cash_hash(policy.model_dump(mode="json")) or frozen.binding_hash != cash_hash(
        current_rule_binding().model_dump(mode="json")
    ):
        raise HistoricalNavStorageError(
            "CASH_POLICY_FREEZE_MISMATCH", "已有冻结规则与当前配置不一致，须明确新版本，不能覆盖旧记录。", 409
        )


def read_matching_policy_freeze(session, policy: CashReleasePolicy) -> CashFrozenPolicy | None:
    if policy.approval_state != "APPROVED":
        return None  # 草案不查询或占用冻结表，更不能把已有记录借来充当审批。
    row = find_policy_freeze(session, policy_version=policy.version)
    if row is None:
        return None
    frozen = restore_policy_freeze(row)
    validate_policy_binding(frozen, policy)
    return frozen


def freeze_cash_policy(request: CashPolicyFreezeRequest) -> CashFrozenPolicy:
    """当前服务器审批/两份指纹均核对后原子保存；并发仅允许相同key读回或版本冲突。"""
    current = describe_cash_policy()
    if not current.approval_ready:
        raise HistoricalNavStorageError(
            "CASH_POLICY_APPROVAL_REQUIRED", "规则仍是草案，尚未获得业务确认，不能冻结。", 409
        )
    if (request.expected_policy_hash, request.expected_binding_hash) != (current.policy_hash, current.binding_hash):
        raise HistoricalNavStorageError("CASH_POLICY_HASH_MISMATCH", "规则或比较约定已变化，请重新核对当前内容。", 409)
    snapshot = CashPolicySnapshot(policy=current.policy, binding=current.binding)
    for attempt in range(2):
        try:
            with Session(get_nav_sample_storage_engine()) as session, session.begin():
                row = find_policy_freeze(session, request_key=request.request_key)
                if row is not None:
                    frozen = restore_policy_freeze(row)
                    validate_policy_binding(frozen, current.policy)
                    return frozen
                if find_policy_freeze(session, policy_version=current.policy.version) is not None:
                    raise HistoricalNavStorageError(
                        "CASH_POLICY_VERSION_FROZEN",
                        "此规则版本已有冻结记录，请读回原记录，不要换请求号重复冻结。",
                        409,
                    )
                # 时间来自数据库，不从请求/规则文件回填。此后不做UPDATE，避免产生可改写的“冻结”记录。
                stamp = session.scalar(select(func.clock_timestamp()))
                row = CashPolicyFreezeRecord(
                    freeze_id=uuid4(),
                    request_key=request.request_key,
                    policy_version=current.policy.version,
                    policy_hash=current.policy_hash,
                    binding_hash=current.binding_hash,
                    snapshot=snapshot.model_dump(mode="json"),
                    created_at=stamp,
                )
                row.content_hash = _content_hash(row)
                session.add(row)
                session.flush()
                return restore_policy_freeze(row, created=True)
        except DBAPIError as error:
            constraint = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
            if attempt or constraint not in {"uq_cash_policy_request", "uq_cash_policy_version"}:
                raise
    raise RuntimeError("unreachable")


def get_cash_policy_freeze(freeze_id: UUID) -> CashFrozenPolicy:
    """只读当时快照；不依赖今天的规则文件，也不重新审批或调用模型。"""
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        return restore_policy_freeze(find_policy_freeze(session, freeze_id=freeze_id))
