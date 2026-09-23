"""自动周期仓储：全局锁、租约所有权、不可变输入与事务发布。"""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories.prediction_store import encode, one, rows
from app.services.auto_model_contract import freeze_value, thaw_value
from app.services.prediction_contract import PredictionFailure, fingerprint


def cycle_state(cycle_id):
    with get_engine().connect() as c:
        result = one(c, "SELECT * FROM prediction_auto_cycle WHERE cycle_id=:id", id=cycle_id)
    if not result:
        raise PredictionFailure("AUTO_CYCLE_NOT_FOUND", "READ", "自动周期不存在", retryable=False)
    return result


def event(c, cycle_id, kind, payload):
    c.execute(
        text("""INSERT INTO prediction_auto_event(event_id,cycle_id,kind,payload)
      VALUES(:id,:cycle,:kind,CAST(:payload AS jsonb))"""),
        {"id": uuid4(), "cycle": cycle_id, "kind": kind, "payload": encode(payload)},
    )


def transition(cycle_id, owner, status, *, checkpoint=None, result=None, error=None):
    """旧worker即使晚返回也不能覆盖新租约持有者；终态停止心跳。"""
    terminal = status in {"COMPLETED", "FAILED", "CANCELLED", "WAITING_DATA", "VERIFYING_ADOPTION"}
    with get_engine().begin() as c:
        changed = c.execute(
            text("""UPDATE prediction_auto_cycle SET status=:status,
          checkpoint=COALESCE(CAST(:checkpoint AS jsonb),checkpoint),
          result=COALESCE(CAST(:result AS jsonb),result),error=CAST(:error AS jsonb),
          heartbeat_at=clock_timestamp(),updated_at=clock_timestamp(),
          lease_until=CASE WHEN :terminal THEN NULL ELSE clock_timestamp()
            +make_interval(secs=>(spec->'policy'->>'leaseSeconds')::integer) END,
          finished_at=CASE WHEN :terminal THEN clock_timestamp() ELSE NULL END
          WHERE cycle_id=:id AND lease_owner=:owner
            AND (NOT cancel_requested OR :status='CANCELLED')
            AND (status NOT IN ('COMPLETED','FAILED','WAITING_DATA','VERIFYING_ADOPTION') OR status=:status)
            RETURNING cycle_id"""),
            {
                "id": cycle_id,
                "owner": owner,
                "status": status,
                "checkpoint": encode(checkpoint) if checkpoint is not None else None,
                "result": encode(result) if result is not None else None,
                "error": encode(error) if error else None,
                "terminal": terminal,
            },
        ).scalar()
        if not changed:
            raise PredictionFailure("AUTO_LEASE_LOST", "LEASE", "后台周期租约已被接管", retryable=False)
        event(c, cycle_id, status, {"checkpoint": checkpoint, "error": error})


def frozen_input(cycle_id, key, loader):
    with get_engine().connect() as c:
        existing = one(
            c, "SELECT * FROM prediction_auto_input WHERE cycle_id=:id AND input_key=:key", id=cycle_id, key=key
        )
    if not existing:
        payload = freeze_value(loader())
        with get_engine().begin() as c:
            c.execute(
                text("""INSERT INTO prediction_auto_input(cycle_id,input_key,content_hash,payload)
              VALUES(:id,:key,:hash,CAST(:payload AS jsonb)) ON CONFLICT DO NOTHING"""),
                {"id": cycle_id, "key": key, "hash": fingerprint(payload), "payload": encode(payload)},
            )
            existing = one(
                c, "SELECT * FROM prediction_auto_input WHERE cycle_id=:id AND input_key=:key", id=cycle_id, key=key
            )
    if fingerprint(existing["payload"]) != existing["content_hash"]:
        raise PredictionFailure("AUTO_INPUT_HASH_MISMATCH", "RECOVERY", "冻结输入校验失败", retryable=False)
    return thaw_value(existing["payload"])


def publish_bundle(cycle_id, bundle, frozen_routes, reason):
    """全部路由同一事务CAS；包真实加载后仍核对登记hash，不按名称发布。"""
    from app.services.prediction_contract import prediction_policy
    from app.services.prediction_direction import direction_fields, validate_identity
    from app.services.prediction_models import load_model, route_key

    refs = bundle["modelRefs"]
    if {r["horizonId"] for r in refs} != {"T5_V1", "T20_V1", "M6_V1"} or len(refs) != 3:
        raise PredictionFailure("RELEASE_INCOMPLETE", "ACTIVATE", "发布清单周期不完整", retryable=False)
    for ref in refs:
        model = load_model(ref["modelId"])
        validate_identity(model["manifest"], ref["horizonId"])
        if reason.get("predictionPolicy", prediction_policy()) != prediction_policy():
            raise PredictionFailure("DIRECTION_POLICY_MISMATCH", "ACTIVATE", "发布规则与本次执行规则不一致")
        if model["modelHash"] != ref["modelHash"] or model["manifest"]["horizonId"] != ref["horizonId"]:
            raise PredictionFailure("RELEASE_MODEL_MISMATCH", "ACTIVATE", "发布模型指纹或周期不符", retryable=False)
    with get_engine().begin() as c:
        c.execute(text("SELECT pg_advisory_xact_lock(721109,3)"))
        state = one(c, "SELECT cancel_requested FROM prediction_auto_cycle WHERE cycle_id=:id FOR UPDATE", id=cycle_id)
        if state and state["cancel_requested"]:
            raise PredictionFailure("AUTO_CANCELLED", "ACTIVATE", "周期已取消，不能发布", retryable=False)
        existing = one(c, "SELECT * FROM prediction_model_release WHERE cycle_id=:id", id=cycle_id)
        if existing:
            return existing
        routes = {r["route_key"]: r for r in rows(c, "SELECT * FROM model_route ORDER BY route_key FOR UPDATE")}
        for ref in refs:
            key = route_key(ref["horizonId"])
            if (routes[key]["model_id"], routes[key]["revision"]) != (
                frozen_routes[key]["model_id"],
                frozen_routes[key]["revision"],
            ):
                raise PredictionFailure(
                    "ACTIVATION_REVISION_CONFLICT",
                    "ACTIVATE",
                    "路由已被其他任务更新，须按新当前版本再评价",
                    retryable=False,
                )
        old = one(c, "SELECT * FROM prediction_release_pointer WHERE scope='GLOBAL' FOR UPDATE")
        release_id = uuid4()
        manifest = {
            "targetDefinitionId": prediction_policy()["target_definition_id"],
            "predictionPolicySnapshot": prediction_policy(),
            **direction_fields("T5_V1"),
            "modelRefs": [r | {"activationRevision": routes[route_key(r["horizonId"])]["revision"] + 1} for r in refs],
            "policyHash": reason["policyHash"],
            "executionPolicy": reason["executionPolicy"],
            "state": "WAITING_ACTUAL_USE",
        }
        c.execute(
            text("""INSERT INTO prediction_model_release
              (release_id,cycle_id,previous_release_id,content_hash,manifest,reason)
          VALUES(:id,:cycle,:previous,:hash,CAST(:manifest AS jsonb),CAST(:reason AS jsonb))"""),
            {
                "id": release_id,
                "cycle": cycle_id,
                "previous": old["release_id"] if old else None,
                "hash": fingerprint(manifest),
                "manifest": encode(manifest),
                "reason": encode(reason),
            },
        )
        for ref in refs:
            key = route_key(ref["horizonId"])
            previous = routes[key]
            c.execute(
                text("""UPDATE model_route SET model_id=:model,previous_model_id=:previous,
              revision=revision+1,release_id=:release,updated_at=clock_timestamp() WHERE route_key=:key"""),
                {"model": ref["modelId"], "previous": previous["model_id"], "release": release_id, "key": key},
            )
            c.execute(
                text("""INSERT INTO model_activation_event
                  (event_id,route_key,revision,previous_model_id,model_id,action,reason)
              VALUES(:id,:key,:revision,:previous,:model,'ACTIVATE',CAST(:reason AS jsonb))"""),
                {
                    "id": uuid4(),
                    "key": key,
                    "revision": previous["revision"] + 1,
                    "previous": previous["model_id"],
                    "model": ref["modelId"],
                    "reason": encode(
                        {"releaseId": str(release_id), "cycleId": str(cycle_id), "decision": reason["decision"]}
                    ),
                },
            )
        c.execute(
            text("""INSERT INTO prediction_release_pointer(scope,release_id,revision) VALUES('GLOBAL',:id,1)
          ON CONFLICT(scope) DO UPDATE SET release_id=excluded.release_id,
            revision=prediction_release_pointer.revision+1"""),
            {"id": release_id},
        )
        event(c, cycle_id, "RELEASE_PUBLISHED", {"releaseId": str(release_id), "manifest": manifest})
        return {"release_id": release_id, "manifest": manifest}


def actual_use_receipt(release_id, receipt):
    """Java只传公共预测引用/摘要，不传个人仓位；必须逐项核对Python不可变原文。"""
    from app.services.prediction_direction import validate_identity

    with get_engine().begin() as c:
        release = one(c, "SELECT * FROM prediction_model_release WHERE release_id=:id", id=release_id)
        if not release:
            raise PredictionFailure("RELEASE_NOT_FOUND", "VERIFY", "发布清单不存在", retryable=False)
        expected = {r["horizonId"]: r for r in release["manifest"]["modelRefs"]}
        found = rows(
            c,
            "SELECT payload FROM fund_prediction_record WHERE prediction_id=ANY(:ids) AND mode='LIVE'",
            ids=receipt["predictionIds"],
        )
        valid = set()
        for row in found:
            p = row["payload"]
            if p.get("role") != "PRIMARY":
                continue
            if release["manifest"].get("predictionPolicySnapshot"):
                validate_identity(p, p["horizonId"], release["manifest"]["predictionPolicySnapshot"])
            ref = expected.get(p["horizonId"])
            if (
                ref
                and p.get("releaseId") == str(release_id)
                and p["modelId"] == ref["modelId"]
                and p["modelHash"] == ref["modelHash"]
                and not p.get("fallbackReason")
            ):
                valid.add(p["horizonId"])
        if valid != set(expected) or not receipt.get("apiReadback") or not receipt.get("adviceReferenced"):
            raise PredictionFailure(
                "ADOPTION_NOT_USED", "VERIFY", "尚未取得完整周期真实推理、保存、API回读及建议引用", retryable=False
            )
        c.execute(
            text("""INSERT INTO prediction_release_receipt(release_id,receipt_hash,kind,payload)
          VALUES(:id,:hash,'ACTUAL_USE',CAST(:payload AS jsonb)) ON CONFLICT DO NOTHING"""),
            {"id": release_id, "hash": fingerprint(receipt), "payload": encode(receipt)},
        )
        c.execute(
            text("""UPDATE prediction_auto_cycle SET status='COMPLETED',finished_at=clock_timestamp(),
          updated_at=clock_timestamp() WHERE cycle_id=:cycle AND status='VERIFYING_ADOPTION'"""),
            {"cycle": release["cycle_id"]},
        )
        return {
            "releaseId": str(release_id),
            "status": "ACTUAL_USE_CONFIRMED",
            "confirmedAt": datetime.now(UTC).isoformat(),
        }
