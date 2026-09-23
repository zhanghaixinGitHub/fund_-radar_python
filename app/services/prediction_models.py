"""受控 JSON 模型包、实验路由和真实推理身份；不加载 pickle 或执行外部代码。"""

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine
from app.repositories.prediction_store import encode, one, rows
from app.services.prediction_contract import PredictionFailure, fingerprint, prediction_policy
from app.services.prediction_direction import (
    BASELINE_ADAPTERS,
    CLASSES,
    TARGET,
    TRI_ADAPTERS,
    classify_return,
    direction_fields,
    three_state,
    validate_identity,
    validate_rule,
)

FEATURES = (
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "max_drawdown_60d",
    "relative_position_60d",
    "consecutive_decline_days",
)
ADAPTERS = {"NAV_MOMENTUM_V1", "LOGISTIC_STANDARDIZED_V1", "DECISION_TREE_V1", "LEGACY_LINEAR_V1"} | TRI_ADAPTERS


def model_directory() -> Path:
    return Path(get_settings().prediction_model_directory).resolve()


def validate_package(package: dict):
    """校验字段公式、单位和训练标签，而不是把训练完成日期当作标签截止。"""
    required = {
        "adapter",
        "recipeVersion",
        "horizonId",
        "targetDefinitionId",
        "assetGroup",
        "featureSchemaVersion",
        "features",
        "featureUnits",
        "missingPolicy",
        "threshold",
        "labelEndMax",
        "trainedAt",
        "codeVersion",
        "dependencies",
        "evidenceLevel",
        "parameters",
    }
    if not required <= package.keys() or package["adapter"] not in ADAPTERS:
        raise PredictionFailure(
            "MODEL_PACKAGE_INCOMPATIBLE", "MODEL_LOAD", "模型包缺少契约或推理适配器不支持", retryable=False
        )
    if package["featureSchemaVersion"] not in {"NAV_TOTAL_RETURN_V1", "LEGACY_NAV_7_V1"}:
        raise PredictionFailure("FEATURE_SCHEMA_MISMATCH", "MODEL_LOAD", "模型特征版本未接入", retryable=False)
    is_tri = package["targetDefinitionId"] == TARGET
    if is_tri != (package["adapter"] in TRI_ADAPTERS):
        raise PredictionFailure("MODEL_PACKAGE_INCOMPATIBLE", "MODEL_LOAD", "二分类适配器不能使用三分类目标")
    if is_tri:
        rule = validate_rule(package.get("directionPolicySnapshot", {}))
        validate_identity(package, package["horizonId"], {"target_definition_id": TARGET, "direction": rule})
    if package["adapter"] in {"LOGISTIC_STANDARDIZED_V1", "DECISION_TREE_V1", "LOGISTIC_MULTICLASS_V2"}:
        if tuple(package["features"]) != FEATURES or package["featureUnits"] != ["RATIO"] * 6 + ["SESSIONS"]:
            raise PredictionFailure(
                "FEATURE_SCHEMA_MISMATCH", "MODEL_LOAD", "特征顺序或单位与训练公式不一致", retryable=False
            )
        label_end = package["labelEndMax"]
        trained = datetime.fromisoformat(package["trainedAt"])
        if not label_end or trained.tzinfo is None or datetime.fromisoformat(label_end) > trained:
            raise PredictionFailure(
                "TRAINING_LABEL_NOT_MATURE", "MODEL_LOAD", "训练标签截止晚于训练时点或未声明", retryable=False
            )
    if len(encode(package)) > 2_000_000:
        raise PredictionFailure("MODEL_PACKAGE_INCOMPATIBLE", "MODEL_LOAD", "模型包超过资源限制", retryable=False)
    # 预加载使用相同适配器，拒绝维度、非有限数字、树循环和缺失依赖。
    infer_package(package, {name: 0.0 for name in FEATURES} | {"momentum": 0.0})


def infer_package(package, features):
    adapter, parameters = package["adapter"], package["parameters"]
    score = None
    if adapter in TRI_ADAPTERS:
        rule = validate_rule(package["directionPolicySnapshot"])
        if adapter == "NAV_MOMENTUM_THREE_STATE_V2":
            # 回看阈值描述历史窗口，与未来收益的持平带分别保存；它是基础假设而非收益预测值。
            direction = classify_return(features["momentum"], parameters["momentumThreshold"])
            if str(parameters["momentumThreshold"]) != rule["momentumThresholds"][package["horizonId"]]:
                raise PredictionFailure("MODEL_PACKAGE_INCOMPATIBLE", "INFERENCE", "基础模型回看阈值不符")
            return {
                "direction": direction,
                "score": None,
                "classScores": None,
                "scoreMeaning": "DETERMINISTIC_RULE_NO_PROBABILITY",
            }
        classes = parameters.get("classes", [])
        x = [float(features[name]) for name in package["features"]]
        means, scales, coefficients, intercepts = (
            parameters[k] for k in ("mean", "scale", "coefficients", "intercepts")
        )
        if (
            tuple(classes) != CLASSES
            or len(coefficients) != 3
            or len(intercepts) != 3
            or not len(x) == len(means) == len(scales) == len(FEATURES)
            or any(len(row) != len(x) for row in coefficients)
            or any(
                not math.isfinite(float(v))
                for v in x + means + scales + intercepts + [v for row in coefficients for v in row]
            )
            or any(s <= 0 for s in scales)
        ):
            raise PredictionFailure("MODEL_PACKAGE_INCOMPATIBLE", "INFERENCE", "三分类参数、类序或标准差不合法")
        logits = [
            b + sum((v - m) / s * w for v, m, s, w in zip(x, means, scales, row, strict=True))
            for b, row in zip(intercepts, coefficients, strict=True)
        ]
        if any(not math.isfinite(v) for v in logits):
            raise PredictionFailure("INFERENCE_ERROR", "INFERENCE", "三分类模型返回非有限数值")
        weights = [math.exp(v - max(logits)) for v in logits]
        scores = dict(zip(classes, [v / sum(weights) for v in weights], strict=True))
        # 只在并列最高的类别中按固定顺序决定，不以“低信心”补造持平。
        direction = next(k for k in rule["tieBreakOrder"] if scores[k] == max(scores.values()))
        return {
            "direction": direction,
            "score": scores["UP"],
            "classScores": scores,
            "scoreMeaning": "UNCALIBRATED_THREE_CLASS_SCORES",
        }
    if adapter == "NAV_MOMENTUM_V1":
        up = float(features["momentum"]) > 0
    elif adapter == "LOGISTIC_STANDARDIZED_V1":
        x = [float(features[name]) for name in package["features"]]
        means, scales, coefficients = (parameters[k] for k in ("mean", "scale", "coefficients"))
        if not len(x) == len(means) == len(scales) == len(coefficients) or any(s <= 0 for s in scales):
            raise PredictionFailure("MODEL_PACKAGE_INCOMPATIBLE", "INFERENCE", "线性模型维度或标准差不合法")
        logit = float(parameters["intercept"]) + sum(
            (v - m) / s * c for v, m, s, c in zip(x, means, scales, coefficients, strict=True)
        )
        if not math.isfinite(logit):
            raise PredictionFailure("INFERENCE_ERROR", "INFERENCE", "模型返回非有限数值")
        score = 1 / (1 + math.exp(-max(-700, min(700, logit))))
        up = score > package["threshold"]
    elif adapter == "DECISION_TREE_V1":
        node = parameters["tree"]
        for _ in range(65):
            if "score" in node:
                score = float(node["score"])
                break
            feature = node["feature"]
            if feature not in FEATURES:
                raise PredictionFailure("FEATURE_SCHEMA_MISMATCH", "INFERENCE", "树模型引用未知特征")
            node = node["left"] if features[feature] <= node["threshold"] else node["right"]
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            raise PredictionFailure("MODEL_PACKAGE_INCOMPATIBLE", "INFERENCE", "树深度或叶节点不合法")
        up = score > package["threshold"]
    elif adapter == "LEGACY_LINEAR_V1":
        from types import SimpleNamespace

        from app.services.direction_linear_models import predict_model

        score = predict_model(parameters, [SimpleNamespace(fund="POOLED", x=tuple(features[k] for k in FEATURES))])[0]
        up = score > package["threshold"]
    else:
        raise PredictionFailure("MODEL_PACKAGE_INCOMPATIBLE", "MODEL_LOAD", "此算法没有可用适配器", retryable=False)
    return {
        "direction": "UP" if up else "NON_UP",
        "score": score,
        "scoreMeaning": "UNCALIBRATED_DIRECTION_SCORE" if score is not None else "DETERMINISTIC_RULE_NO_PROBABILITY",
    }


def register_model(package: dict) -> dict:
    validate_package(package)
    digest = fingerprint(package)
    model_id = "MP-" + digest[:32]
    directory = model_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (model_id + ".json")
    data = encode(package)
    if path.exists() and fingerprint(json.loads(path.read_text(encoding="utf-8"))) != digest:
        raise PredictionFailure("MODEL_HASH_MISMATCH", "REGISTER", "已有模型文件校验失败", retryable=False)
    if not path.exists():
        temporary = directory / (uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    with get_engine().begin() as connection:
        connection.execute(
            text("""
          INSERT INTO model_artifact(model_id,content_hash,horizon_id,target_definition_id,asset_group,adapter,manifest)
          VALUES(:id,:hash,:horizon,:target,:group,:adapter,CAST(:manifest AS jsonb)) ON CONFLICT DO NOTHING
        """),
            {
                "id": model_id,
                "hash": digest,
                "horizon": package["horizonId"],
                "target": package["targetDefinitionId"],
                "group": package["assetGroup"],
                "adapter": package["adapter"],
                "manifest": data,
            },
        )
    return {"modelId": model_id, "modelHash": digest, "manifest": package}


def load_model(model_id: str) -> dict:
    with get_engine().connect() as connection:
        row = one(connection, "SELECT * FROM model_artifact WHERE model_id=:id", id=model_id)
    if row is None:
        raise PredictionFailure("MODEL_PACKAGE_MISSING", "MODEL_LOAD", "模型尚未登记", details={"modelId": model_id})
    path = model_directory() / (row["model_id"] + ".json")
    try:
        if path.stat().st_size > 2_000_000:
            raise ValueError("PACKAGE_SIZE")
        package = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionFailure(
            "MODEL_PACKAGE_MISSING", "MODEL_LOAD", "已登记模型文件缺失或损坏", details={"modelId": model_id}
        ) from error
    if fingerprint(package) != row["content_hash"] or package != row["manifest"]:
        raise PredictionFailure(
            "MODEL_HASH_MISMATCH",
            "MODEL_LOAD",
            "模型文件与登记指纹不一致",
            details={"modelId": model_id},
            retryable=False,
        )
    validate_package(package)
    return {"modelId": model_id, "modelHash": row["content_hash"], "manifest": package}


def route_key(horizon, group="ALL"):
    policy = prediction_policy()
    version = ":" + fingerprint(validate_rule(policy["direction"])) if three_state(policy) else ""
    return policy["target_definition_id"] + version + ":" + horizon + ":" + group


def activate(model_id, *, reason, expected_revision=None, shadow_ids=()):
    model = load_model(model_id)
    package = model["manifest"]
    validate_identity(package, package["horizonId"])
    if package["targetDefinitionId"] != prediction_policy()["target_definition_id"]:
        raise PredictionFailure(
            "TARGET_DEFINITION_MISMATCH", "ACTIVATE", "旧目标模型不能冒充新目标模型", retryable=False
        )
    key = route_key(package["horizonId"], package["assetGroup"])
    with get_engine().begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key,0))"), {"key": key})
        previous = one(connection, "SELECT * FROM model_route WHERE route_key=:key FOR UPDATE", key=key)
        revision = previous["revision"] if previous else 0
        if expected_revision is not None and revision != expected_revision:
            raise PredictionFailure("ACTIVATION_REVISION_CONFLICT", "ACTIVATE", "其他任务已切换模型，请重新比较")
        connection.execute(
            text("""
          INSERT INTO model_route(route_key,model_id,previous_model_id,revision,shadow_ids)
          VALUES(:key,:model,:previous,:revision,CAST(:shadows AS jsonb))
          ON CONFLICT(route_key) DO UPDATE SET model_id=excluded.model_id,
            previous_model_id=excluded.previous_model_id,revision=excluded.revision,
            shadow_ids=excluded.shadow_ids,updated_at=clock_timestamp()
        """),
            {
                "key": key,
                "model": model_id,
                "previous": previous["model_id"] if previous else None,
                "revision": revision + 1,
                "shadows": encode(list(shadow_ids)[:3]),
            },
        )
        connection.execute(
            text("""
          INSERT INTO model_activation_event(event_id,route_key,revision,previous_model_id,model_id,action,reason)
          VALUES(:id,:key,:revision,:previous,:model,'ACTIVATE',CAST(:reason AS jsonb))
        """),
            {
                "id": uuid4(),
                "key": key,
                "revision": revision + 1,
                "previous": previous["model_id"] if previous else None,
                "model": model_id,
                "reason": encode(reason),
            },
        )
    return {"routeKey": key, "modelId": model_id, "revision": revision + 1}


def baseline_package(horizon_id):
    """训练无关的固定基础配方，真实生成与历史回放引用同一份声明。"""
    policy = prediction_policy()
    horizon = next(h for h in policy["horizons"] if h["horizon_id"] == horizon_id)
    tri = three_state(policy)
    return {
        "adapter": "NAV_MOMENTUM_THREE_STATE_V2" if tri else "NAV_MOMENTUM_V1",
        "recipeVersion": "NAV_MOMENTUM_THREE_STATE_V2" if tri else "NAV_MOMENTUM_BASELINE_V1",
        **direction_fields(horizon_id, policy),
        "horizonId": horizon["horizon_id"],
        "targetDefinitionId": policy["target_definition_id"],
        "assetGroup": "ALL",
        "featureSchemaVersion": policy["feature_schema_version"],
        "features": ["momentum"],
        "featureUnits": ["RATIO"],
        "missingPolicy": "FAIL_REQUIRED",
        "threshold": 0,
        "labelEndMax": None,
        "trainedAt": None,
        "codeVersion": policy["version"],
        "dependencies": {},
        "evidenceLevel": "DEVELOPMENT_ONLY",
        "parameters": {
            "lookbackReturns": horizon["lookback_returns"],
            "minimumReturns": 20,
            **({"momentumThreshold": policy["direction"]["momentumThresholds"][horizon_id]} if tri else {}),
        },
    }


def bootstrap_models():
    """先登记完整包，再原子初始化三条新路由；重复启动不覆盖已采用的配置。"""
    horizons = prediction_policy()["horizons"]
    registered = [(h, register_model(baseline_package(h["horizon_id"]))) for h in horizons]
    with get_engine().begin() as c:
        c.execute(text("SELECT pg_advisory_xact_lock(721109,3)"))
        for horizon, model in registered:
            c.execute(
                text("""INSERT INTO prediction_horizon(horizon_id,content_hash,payload)
              VALUES(:id,:hash,CAST(:payload AS jsonb)) ON CONFLICT DO NOTHING"""),
                {"id": horizon["horizon_id"], "hash": fingerprint(horizon), "payload": encode(horizon)},
            )
            key = route_key(horizon["horizon_id"])
            created = c.execute(
                text("""INSERT INTO model_route(route_key,model_id,revision,shadow_ids)
              VALUES(:key,:model,1,'[]'::jsonb) ON CONFLICT DO NOTHING RETURNING route_key"""),
                {"key": key, "model": model["modelId"]},
            ).scalar()
            if created:
                c.execute(
                    text("""INSERT INTO model_activation_event
                  (event_id,route_key,revision,model_id,action,reason)
                  VALUES(:id,:key,1,:model,'BOOTSTRAP',CAST(:reason AS jsonb))"""),
                    {
                        "id": uuid4(),
                        "key": key,
                        "model": model["modelId"],
                        "reason": encode(
                            {"decision": "INITIALIZE", "reason": "首批三周期完整基础方法，尚未证明长期优势"}
                        ),
                    },
                )


def freeze_routes():
    with get_engine().connect() as connection:
        # 一条语句的MVCC快照固定整个发布批次，不能逐周期读取混入不同发布。
        keys = [route_key(h["horizon_id"]) for h in prediction_policy()["horizons"]]
        return {
            r["route_key"]: r
            for r in rows(
                connection, "SELECT * FROM model_route WHERE route_key=ANY(:keys) ORDER BY route_key", keys=keys
            )
        }


def infer_route(route, features, horizon_id):
    """单基金特征失败只降阶本条；包故障追加事件，实际身份必须为真实执行的包。"""
    requested = route["model_id"]
    with get_engine().connect() as connection:
        baseline = one(
            connection,
            """SELECT model_id FROM model_artifact
            WHERE horizon_id=:h AND target_definition_id=:target AND adapter=:adapter
            AND COALESCE(manifest->>'directionPolicyHash','')=:rule ORDER BY created_at LIMIT 1""",
            h=horizon_id,
            target=prediction_policy()["target_definition_id"],
            adapter="NAV_MOMENTUM_THREE_STATE_V2" if three_state() else "NAV_MOMENTUM_V1",
            rule=direction_fields(horizon_id).get("directionPolicyHash", ""),
        )
    candidates = list(
        dict.fromkeys(
            filter(None, [requested, route.get("previous_model_id"), baseline["model_id"] if baseline else None])
        )
    )
    if route.get("strictModel"):
        candidates = [requested]
    failures = []
    for candidate in candidates:
        try:
            with get_engine().connect() as connection:
                quarantined = one(
                    connection,
                    """SELECT reason FROM prediction_model_quarantine
                  WHERE model_id=:id AND retry_after>clock_timestamp()""",
                    id=candidate,
                )
            if quarantined:
                raise PredictionFailure("MODEL_QUARANTINED", "INFERENCE", "问题包处于故障隔离期", retryable=False)
            model = load_model(candidate)
            validate_identity(model["manifest"], horizon_id)
            result = infer_package(model["manifest"], features)
            result.update(direction_fields(horizon_id))
            result.update(
                modelId=candidate,
                modelHash=model["modelHash"],
                requestedModelId=requested,
                activationRevision=route["revision"],
                fallbackReason=failures or None,
                modelManifest=model["manifest"],
                baseline=model["manifest"]["adapter"] in BASELINE_ADAPTERS,
                releaseId=str(route["release_id"]) if route.get("release_id") and not failures else None,
                requestedReleaseId=str(route["release_id"]) if route.get("release_id") else None,
            )
            if failures:
                with get_engine().begin() as connection:
                    connection.execute(
                        text("""INSERT INTO model_activation_event
                      (event_id,route_key,revision,previous_model_id,model_id,action,reason)
                      VALUES(:id,:key,:revision,:previous,:model,'FALLBACK',CAST(:reason AS jsonb))"""),
                        {
                            "id": uuid4(),
                            "key": route["route_key"],
                            "revision": route["revision"],
                            "previous": requested,
                            "model": candidate,
                            "reason": encode(failures),
                        },
                    )
            return result
        except KeyError:
            failures.append(
                {
                    "code": "FEATURE_SCHEMA_MISMATCH",
                    "stage": "FEATURE_BUILD",
                    "summary": "此模型所需特征不完整，尝试基础方法",
                    "modelId": candidate,
                }
            )
        except PredictionFailure as error:
            failures.append(error.payload | {"modelId": candidate})
            if error.payload["code"] in {
                "MODEL_HASH_MISMATCH",
                "MODEL_PACKAGE_MISSING",
                "MODEL_ADAPTER_UNSUPPORTED",
                "MODEL_PACKAGE_INCOMPATIBLE",
                "INFERENCE_ERROR",
            }:
                with get_engine().begin() as connection:
                    connection.execute(
                        text("""INSERT INTO prediction_model_quarantine(model_id,reason,retry_after)
                      VALUES(:id,CAST(:reason AS jsonb),clock_timestamp()+interval '30 minutes')
                      ON CONFLICT(model_id) DO UPDATE SET reason=excluded.reason,retry_after=excluded.retry_after,
                      failure_count=prediction_model_quarantine.failure_count+1"""),
                        {"id": candidate, "reason": encode(error.payload)},
                    )
    raise PredictionFailure(
        "INFERENCE_ERROR", "INFERENCE", "当前模型、旧模型和基线均无法运行", details={"attempts": failures}
    )


def model_status():
    with get_engine().connect() as connection:
        return {
            "activeRouteKeys": [route_key(h["horizon_id"]) for h in prediction_policy()["horizons"]],
            "activeTargetDefinitionId": prediction_policy()["target_definition_id"],
            "routes": rows(connection, "SELECT * FROM model_route ORDER BY route_key"),
            "models": rows(
                connection,
                """SELECT m.*, (SELECT count(*) FROM fund_prediction_record p
                   WHERE p.model_id=m.model_id AND mode='LIVE') live_calls,
                   (SELECT max(generated_at) FROM fund_prediction_record p WHERE p.model_id=m.model_id
                   AND mode='LIVE') last_live_call FROM model_artifact m
                   WHERE m.model_id IN (SELECT model_id FROM model_artifact ORDER BY created_at DESC LIMIT 100)
                   OR m.model_id IN (SELECT model_id FROM model_route)
                   OR m.model_id IN (SELECT previous_model_id FROM model_route)
                   OR m.model_id IN (SELECT jsonb_array_elements_text(shadow_ids) FROM model_route)
                   ORDER BY created_at DESC""",
            ),
            "events": rows(connection, "SELECT * FROM model_activation_event ORDER BY created_at DESC LIMIT 100"),
            "evaluations": rows(connection, "SELECT * FROM model_evaluation ORDER BY created_at DESC LIMIT 30"),
            "readAt": datetime.now(UTC).isoformat(),
        }
