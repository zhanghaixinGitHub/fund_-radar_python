"""一日输入顺序和去重；只收公共输入，不接收个人资金或浏览器指定模型。"""

import hashlib
import json
from contextlib import contextmanager

from sqlalchemy import text

from app.db.session import get_engine
from app.services.direction_1d_protocol import canonical, digest


@contextmanager
def scope_lock(code: str, target: str, protocol: str):
    """跨进程串行读取同一基金目标日，防止较早输入在较新输入之后登记顺序。"""
    key = f"{protocol}:{code}:{target}"
    with get_engine().connect() as connection:
        connection.execute(text("SELECT pg_advisory_lock(hashtextextended(:key,721129))"), {"key": key})
        try:
            yield
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(hashtextextended(:key,721129))"), {"key": key})


def identity(code, target, protocol, mapping, observed, locked, events):
    """只对实际采用的来源版本和模型做摘要，不把检查时间当成输入变化。"""
    return {
        "fund_code": code,
        "target_nav_date": target,
        "protocol": protocol,
        "mapping": mapping,
        "values": [{k: value[k] for k in ("nav_date", "source_id", "content_hash")} for value in observed],
        "models": [
            {"branch": row["branch_id"], "model_id": str(row["model_id"]), "hash": row["content_hash"]}
            for row in locked
        ],
        "events": sorted(events, key=lambda event: str(event.get("content_hash"))),
    }


def select_revision(connection, data, deadline):
    """调用方须持有 scope_lock；只与最近输入比较，A→B→A 仍产生第三个顺序。"""
    params = {"code": data["fund_code"], "target": data["target_nav_date"], "protocol": data["protocol"]}
    previous = (
        connection.execute(
            text("""
        SELECT * FROM direction_1d_input_revision
        WHERE fund_code=:code AND target_nav_date=:target AND protocol=:protocol
        ORDER BY revision_sequence DESC LIMIT 1
    """),
            params,
        )
        .mappings()
        .first()
    )
    value_hash = digest(data)
    if previous and previous["input_identity"] == value_hash:
        if previous["result"]:
            verify_result(previous["result"], previous["revision_sequence"], value_hash)
        return dict(previous)
    created = (
        connection.execute(
            text("""
        INSERT INTO direction_1d_input_revision(
            fund_code,target_nav_date,protocol,input_identity,identity_payload,deadline_at)
        SELECT :code,:target,:protocol,:hash,CAST(:data AS jsonb),:deadline
        WHERE clock_timestamp()<CAST(:deadline AS timestamptz) RETURNING *
    """),
            {**params, "hash": value_hash, "data": canonical(data), "deadline": deadline},
        )
        .mappings()
        .first()
    )
    if not created:
        raise ValueError("MISSED_DEADLINE")
    return dict(created)


def verify_result(result, sequence, input_identity):
    raw = result["payload_json"]
    value = json.loads(raw)
    if (
        hashlib.sha256(raw.encode()).hexdigest() != result["content_hash"]
        or value.get("revision_sequence") != sequence
        or value.get("input_identity") != input_identity
    ):
        raise ValueError("REVISION_HASH_MISMATCH")


def complete(sequence, result):
    """数据库写入这一刻重新检查截止；晚完成只保留失败作业，不发布为有效预测。"""
    with get_engine().begin() as connection:
        row = connection.execute(
            text("""
            UPDATE direction_1d_input_revision SET result=CAST(:result AS jsonb),completed_at=clock_timestamp()
            WHERE revision_sequence=:sequence AND result IS NULL AND clock_timestamp()<deadline_at
            RETURNING revision_sequence
        """),
            {"sequence": sequence, "result": canonical(result)},
        ).first()
        if not row:
            raise ValueError("MISSED_DEADLINE")
