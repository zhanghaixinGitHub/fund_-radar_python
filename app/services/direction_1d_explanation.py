"""只读还原一日原预测的指标作用；核对原文与登记包，不训练、不生成、不改历史。"""

import hashlib
import json
import math
from datetime import datetime
from uuid import UUID

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.services import direction_1d_three_state as three
from app.services.direction_1d_inference import load_model
from app.services.direction_1d_protocol import FEATURES, canonical, digest, score


def explain_original(body, model_loader, *, include_reasoning=False):
    """将每项标准化输入乘以原系数，保留支持与反对作用，不把分数解释为概率。"""
    source = body["input"]
    if source.get("fund_code") != body["fund_code"] or digest(source) != body["input_hash"]:
        raise ValueError("INPUT_HASH_MISMATCH")
    values = source.get("features")
    if not isinstance(values, list) or len(values) != len(FEATURES):
        raise ValueError("FEATURES_MISSING")
    branches = []
    for branch in body["branches"]:
        if branch.get("status") not in ("AVAILABLE", "ABSTAINED"):
            continue
        model = model_loader(branch["model_id"], branch["model_hash"])
        from app.services import direction_1d_information as information

        if source.get("feature_version") in information.SUPPORTED_VERSIONS:
            information.validate_model(model)
            full_input = source.get("information", {})
            if full_input.get("numeric", [])[:7] != values:
                raise ValueError("INFORMATION_NAV_MISMATCH")
            classification = information.predict(model, full_input)
            if model.get("feature_version") == information.event_model.VERSION:
                if (branch["status"] != classification["status"] or branch["score"] != classification["score"]
                        or branch["predicted_direction"] != classification["direction"]
                        or branch.get("class_scores") != classification["class_scores"]
                        or branch.get("decision") != classification["decision"]):
                    raise ValueError("PREDICTION_RESTORE_MISMATCH")
                branches.append({"branchId": branch["branch_id"], "modelId": branch["model_id"],
                                 "modelHash": branch["model_hash"], **information.explain(model, full_input)})
                continue
            if (classification["direction"] != branch["predicted_direction"]
                    or abs(classification["score"] - branch["score"]) > 1e-12
                    or any(abs(classification["class_scores"][key] - branch["class_scores"][key]) > 1e-12
                           for key in three.CLASSES)):
                raise ValueError("PREDICTION_RESTORE_MISMATCH")
            branches.append({"branchId": branch["branch_id"], "modelId": branch["model_id"],
                             "modelHash": branch["model_hash"], **information.explain(model, full_input)})
            continue
        ternary = body.get("protocol") == three.PROTOCOL
        classification = three.predict(model, values) if ternary else None
        restored = classification["score"] if classification else score(model, values)
        original = branch["score"]
        expected = classification["direction"] if classification else "UP" if restored > 0.5 else "NON_UP"
        if classification and (
            set(branch.get("class_scores", {})) != set(three.CLASSES)
            or any(
                abs(branch["class_scores"][key] - classification["class_scores"][key]) > 1e-12 for key in three.CLASSES
            )
        ):
            raise ValueError("PREDICTION_RESTORE_MISMATCH")
        if (
            not isinstance(original, (int, float))
            or not math.isfinite(original)
            or abs(restored - original) > 1e-12
            or expected != branch["predicted_direction"]
        ):
            raise ValueError("PREDICTION_RESTORE_MISMATCH")
        if classification:
            winner, runner = three.CLASSES.index(expected), three.CLASSES.index(classification["runner"])
            weights = [a - b for a, b in zip(model["coef"][winner], model["coef"][runner], strict=True)]
            intercept = model["intercept"][winner] - model["intercept"][runner]
        else:
            weights, intercept = model["coef"], model["intercept"]
        impacts = [
            {
                "feature": key,
                "value": value,
                # 正值推向上涨，负值推向非上涨；单位是模型内部判别量，不是收益百分点。
                "contribution": weight * (value - mean) / scale,
            }
            for key, value, weight, mean, scale in zip(
                FEATURES, values, weights, model["mean"], model["scale"], strict=True
            )
        ]
        if any(not math.isfinite(item["contribution"]) for item in impacts):
            raise ValueError("INVALID_CONTRIBUTION")
        if include_reasoning:
            # 对照仅替换一项输入为当时保存的参考值，其余六项保持原值；不读取新行情或重训。
            # 这是解释原判断的假设计算，不是新的预测，也不是可实际发生的市场情景。
            reference_direction = classification["runner"] if classification else "NON_UP" if expected == "UP" else "UP"
            for i, item in enumerate(impacts):
                reference_values = list(values)
                reference_values[i] = model["mean"][i]
                direction_at_reference = (
                    three.predict(model, reference_values)["direction"]
                    if classification
                    else "UP"
                    if score(model, reference_values) > 0.5
                    else "NON_UP"
                )
                item["reasoning"] = {
                    "referenceValue": model["mean"][i],
                    "relativeWeight": weights[i] if classification or expected == "UP" else -weights[i],
                    "comparisonDirection": reference_direction,
                    "directionAtReference": direction_at_reference,
                }
        branches.append(
            {
                "branchId": branch["branch_id"],
                "modelId": branch["model_id"],
                "modelHash": branch["model_hash"],
                "direction": expected,
                "intercept": intercept,
                **({"referenceDirection": classification["runner"]} if classification else {}),
                "factors": impacts,
            }
        )
    if not branches:
        raise ValueError("NO_AVAILABLE_BRANCH")
    return {
        "fundCode": body["fund_code"],
        "inputHash": body["input_hash"],
        "baseNavDate": body["base_nav_date"],
        "targetNavDate": body["target_nav_date"],
        "branches": branches,
    }


def read_explanation(job_id: UUID, *, include_reasoning=False):
    """仅接纳已成功留档的作业；一日数据到期、原文损坏或模型版本不符时拒绝还原。"""
    job = repo.get_job(job_id)
    if not job or job["kind"] != "FORECAST" or job["state"] != "SUCCEEDED":
        raise ValueError("FORECAST_NOT_FOUND")
    raw = job["result"]["payload_json"]
    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if content_hash != job["result"]["content_hash"]:
        raise ValueError("FORECAST_HASH_MISMATCH")
    body = json.loads(raw)
    from app.schemas.fund_information_analysis import PROTOCOL as analysis_protocol

    if body.get("protocol") == analysis_protocol:
        from app.services.fund_information_analysis import evidence
        return evidence(body, content_hash)
    if datetime.fromisoformat(body["expires_at"]) <= repo.clock():
        raise ValueError("EVIDENCE_EXPIRED")

    def original_model(model_id, expected_hash):
        with get_engine().connect() as connection:
            row = (
                connection.execute(text("SELECT * FROM direction_1d_model WHERE model_id=:id"), {"id": UUID(model_id)})
                .mappings()
                .first()
            )
        if not row or row["content_hash"] != expected_hash:
            raise ValueError("MODEL_REGISTRY_MISMATCH")
        return load_model(row)

    result = explain_original(body, original_model, include_reasoning=include_reasoning)
    return json.loads(canonical(result | {"contentHash": content_hash}))
