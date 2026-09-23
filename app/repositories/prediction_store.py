"""多周期证据仓储：绑定参数、短事务及数据库约束保证幂等和不可变。"""

import json
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.services.prediction_contract import fingerprint


def encode(value):
    return json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)


def rows(connection, sql, **params):
    return [dict(row) for row in connection.execute(text(sql), params).mappings()]


def one(connection, sql, **params):
    return next(iter(rows(connection, sql, **params)), None)


def snapshot(connection, fund_code, cutoff, payload):
    content_hash = fingerprint(payload)
    connection.execute(
        text("""
        INSERT INTO prediction_feature_snapshot(snapshot_id,fund_code,knowledge_cutoff,content_hash,payload)
        VALUES(:id,:code,:cutoff,:hash,CAST(:payload AS jsonb)) ON CONFLICT(content_hash) DO NOTHING
    """),
        {"id": uuid4(), "code": fund_code, "cutoff": cutoff, "hash": content_hash, "payload": encode(payload)},
    )
    return one(
        connection, "SELECT snapshot_id FROM prediction_feature_snapshot WHERE content_hash=:hash", hash=content_hash
    )["snapshot_id"]


def save_prediction(connection, payload, period_key):
    """保存超时可用 period_key 重新读取；冲突返回首份原文，不重算独立样本。"""
    inserted = connection.execute(
        text("""
      INSERT INTO fund_prediction_record(prediction_id,period_key,fund_code,horizon_id,mode,model_id,
        activation_revision,generated_at,content_hash,payload)
      VALUES(:id,:key,:fund,:horizon,:mode,:model,:revision,:generated,:hash,CAST(:payload AS jsonb))
      ON CONFLICT(period_key) DO NOTHING RETURNING prediction_id
    """),
        {
            "id": payload["predictionId"],
            "key": period_key,
            "fund": payload["fundCode"],
            "horizon": payload["horizonId"],
            "mode": payload["mode"],
            "model": payload["modelId"],
            "revision": payload["activationRevision"],
            "generated": payload["generatedAt"],
            "hash": fingerprint(payload),
            "payload": encode(payload),
        },
    ).scalar()
    stored = one(
        connection, "SELECT payload,content_hash FROM fund_prediction_record WHERE period_key=:key", key=period_key
    )
    if fingerprint(stored["payload"]) != stored["content_hash"]:
        raise ValueError("PREDICTION_EVIDENCE_HASH_MISMATCH")
    return stored["payload"], inserted is not None


def history(fund_code, *, limit=30, before=None, before_id=None):
    with get_engine().connect() as connection:
        return rows(
            connection,
            """
          SELECT p.payload, (SELECT r.payload FROM prediction_target_resolution r
            WHERE r.prediction_id=p.prediction_id ORDER BY r.created_at DESC LIMIT 1) resolution,
            (SELECT jsonb_build_object('status',s.status,'details',s.payload,'lastAttemptAt',s.last_attempt_at)
             FROM prediction_check_state s WHERE s.prediction_id=p.prediction_id) "outcomeCheck",
            (SELECT jsonb_agg(o.payload ORDER BY o.checked_at)
            FROM prediction_outcome o WHERE o.prediction_id=p.prediction_id) outcomes
          FROM fund_prediction_record p WHERE fund_code=:code AND mode='LIVE' AND payload->>'role'='PRIMARY'
          AND (CAST(:before AS timestamptz) IS NULL OR generated_at<:before
            OR (generated_at=:before AND prediction_id>CAST(:before_id AS uuid)))
          ORDER BY generated_at DESC,prediction_id LIMIT :limit
        """,
            code=fund_code,
            before=before,
            before_id=before_id,
            limit=min(100, limit),
        )
