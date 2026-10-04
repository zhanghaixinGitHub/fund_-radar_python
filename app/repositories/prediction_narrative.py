"""解释缓存与有界生成租约；数据库事务中不调用外部服务。"""

from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories.prediction_store import encode, one
from app.services.prediction_contract import fingerprint


def read(kind, source_id, style_version):
    with get_engine().connect() as c:
        return one(
            c,
            "SELECT * FROM prediction_narrative WHERE source_kind=:kind AND source_id=:id AND style_version=:version",
            kind=kind,
            id=source_id,
            version=style_version,
        )


def previous(kind, source_id, style_version):
    """仅为明确支持的历史版本提供回退；配置变化不会隐式发起内容升级。"""
    with get_engine().connect() as c:
        return one(
            c,
            """SELECT * FROM prediction_narrative WHERE source_kind=:kind AND source_id=:id
          AND state='READY' AND style_version IN ('PREDICTION_NARRATIVE_ZH_V1','PREDICTION_NARRATIVE_ZH_V2')
          AND style_version<>:version ORDER BY created_at DESC LIMIT 1""",
            kind=kind,
            id=source_id,
            version=style_version,
        )


def claim(kind, source_id, fund_code, source_hash, input_hash, style_version, model):
    """按原预测去重，全服务最多两个在途调用；过期租约可恢复，成功记录不重写。"""
    owner = uuid4()
    with get_engine().begin() as c:
        # 仅串行化很短的领取操作，防止多进程同时绕过全局并发上限。
        c.execute(text("SELECT pg_advisory_xact_lock(hashtext('prediction-narrative-capacity-v1'))"))
        mismatch = one(
            c,
            """SELECT 1 AS found FROM prediction_narrative WHERE source_kind=:kind AND source_id=:id
          AND (source_hash<>:hash OR fund_code<>:code) LIMIT 1""",
            kind=kind,
            id=source_id,
            hash=source_hash,
            code=fund_code,
        )
        if mismatch:
            return None
        old = one(
            c,
            "SELECT *,clock_timestamp() AS now FROM prediction_narrative "
            "WHERE source_kind=:kind AND source_id=:id AND style_version=:version FOR UPDATE",
            kind=kind,
            id=source_id,
            version=style_version,
        )
        if old and (
            old["source_hash"] != source_hash or old["fund_code"] != fund_code or old["input_hash"] != input_hash
        ):
            return None
        if old and (
            old["state"] == "READY"
            or old["attempts"] >= 3
            or (old["state"] == "RUNNING" and old["lease_until"] > old["now"])
            or (old["retry_after"] and old["retry_after"] > old["now"])
        ):
            return None
        active = c.execute(
            text("SELECT count(*) FROM prediction_narrative WHERE state='RUNNING' AND lease_until>clock_timestamp()")
        ).scalar()
        if active >= 2:
            return None
        params = dict(
            kind=kind,
            id=source_id,
            code=fund_code,
            source=source_hash,
            input=input_hash,
            version=style_version,
            model=model,
            owner=owner,
        )
        c.execute(
            text("""
          INSERT INTO prediction_narrative(source_kind,source_id,fund_code,source_hash,input_hash,
            style_version,state,owner_id,lease_until,attempts,provider_model)
          VALUES(:kind,:id,:code,:source,:input,:version,'RUNNING',:owner,
            clock_timestamp()+interval '90 seconds',1,:model)
          ON CONFLICT(source_kind,source_id,style_version) DO UPDATE SET state='RUNNING',owner_id=:owner,
            lease_until=clock_timestamp()+interval '90 seconds',attempts=prediction_narrative.attempts+1,
            updated_at=clock_timestamp(),retry_after=NULL,failure_code=NULL,provider_model=:model,
            style_version=:version,input_hash=:input
        """),
            params,
        )
    return owner


def finish(kind, source_id, owner, *, payload=None, failure=None):
    """仅当前租约持有者能落库，迟到响应不得覆盖新的处理或已保存正文。"""
    with get_engine().begin() as c:
        return (
            c.execute(
                text("""
          UPDATE prediction_narrative SET state=:state,payload=CAST(:payload AS jsonb),content_hash=:hash,
            failure_code=:failure,
            retry_after=CASE WHEN :failed THEN clock_timestamp()+interval '15 minutes' ELSE NULL END,
            updated_at=clock_timestamp()
          WHERE source_kind=:kind AND source_id=:id AND owner_id=:owner AND state='RUNNING'
            AND lease_until>clock_timestamp()
        """),
                dict(
                    kind=kind,
                    id=source_id,
                    owner=owner,
                    state="READY" if payload else "FAILED",
                    payload=encode(payload) if payload else None,
                    hash=fingerprint(payload) if payload else None,
                    failure=failure,
                    failed=payload is None,
                ),
            ).rowcount
            == 1
        )
