"""实验选优协议：同题、同分母、固定次序；不以80%或长期显著性作为实验采用门槛。"""

from collections import defaultdict
from datetime import datetime
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories.prediction_store import encode
from app.services.prediction_contract import PredictionFailure, fingerprint, prediction_policy
from app.services.prediction_models import activate, load_model


def evaluate_answers(planned: list[dict], answers: dict[str, dict]) -> dict:
    """准确率仅用成功且成熟题，覆盖率保留全部计划题；日期内先聚合家族再等权日期。"""
    if len({p["key"] for p in planned}) != len(planned):
        raise ValueError("DUPLICATE_EVALUATION_QUESTION")
    families = defaultdict(list)
    recalls = {"UP": [], "NON_UP": []}
    correct, counts, breakdown = 0, 0, defaultdict(lambda: [0, 0])
    for item in planned:
        answer = answers.get(item["key"])
        if answer is None or answer.get("direction") not in recalls or item.get("actual") not in recalls:
            continue
        success = int(answer["direction"] == item["actual"])
        families[(item["date"], item["family"])].append(success)
        recalls[item["actual"]].append(success)
        correct += success
        counts += 1
        for group in ("type:" + item.get("fundType", "UNKNOWN"), "period:" + item["date"][:7], "fund:" + item["fund"]):
            breakdown[group][0] += success
            breakdown[group][1] += 1
    by_date = defaultdict(list)
    for (day, _), values in families.items():
        by_date[day].append(sum(values) / len(values))
    date_scores = [sum(values) / len(values) for values in by_date.values()]
    recall_values = {k: sum(v) / len(v) if v else None for k, v in recalls.items()}
    available = [v for v in recall_values.values() if v is not None]
    return {
        "plannedItems": len(planned),
        "successfulMatureItems": counts,
        "failedOrUnmaturedItems": len(planned) - counts,
        "fundCount": len({p["fund"] for p in planned}),
        "dateCount": len({p["date"] for p in planned}),
        "coverage": counts / len(planned) if planned else 0,
        "primaryScore": sum(date_scores) / len(date_scores) if date_scores else None,
        "accuracy": correct / counts if counts else None,
        "classRecall": recall_values,
        "balancedAccuracy": sum(available) / len(available) if available else None,
        "breakdown": {k: {"correct": v[0], "count": v[1]} for k, v in breakdown.items()},
    }


def choose_candidate(current: dict, candidates: list[dict], *, protocol_hash: str) -> dict:
    """同题主指标严格改善才提名，平分保留当前，训练更新日期不是优势。"""
    tolerance = prediction_policy()["selection"]["tolerance"]
    winner, decisions = current, []
    for candidate in sorted(candidates, key=lambda c: c["modelId"]):
        if candidate["protocolHash"] != protocol_hash or candidate["sampleHash"] != current["sampleHash"]:
            raise PredictionFailure(
                "EVALUATION_CONDITIONS_MISMATCH", "SELECTION", "候选与当前模型的评价条件不一致", retryable=False
            )
        if not candidate.get("runnable") or not candidate.get("timeValid"):
            decisions.append(
                {"modelId": candidate["modelId"], "decision": "BLOCKED_TECHNICAL", "reason": "包或时间正确性检查未通过"}
            )
            continue
        metrics, best = candidate["metrics"], winner["metrics"]
        if metrics["primaryScore"] is None or metrics["coverage"] < current["metrics"]["coverage"]:
            decisions.append(
                {"modelId": candidate["modelId"], "decision": "RUN_AS_SHADOW", "reason": "覆盖下降或无成熟评分"}
            )
            continue
        difference = metrics["primaryScore"] - (best["primaryScore"] if best["primaryScore"] is not None else -1)
        if difference > tolerance:
            winner = candidate
        decisions.append(
            {
                "modelId": candidate["modelId"],
                "decision": "RUN_AS_SHADOW",
                "reason": "保留同题对照；最终胜者按冻结协议确定",
            }
        )
    changed = winner["modelId"] != current["modelId"]
    for item in decisions:
        if changed and item["modelId"] == winner["modelId"]:
            item.update(decision="ACTIVATE", reason="同题主指标严格改善；自动采用仍须完整组合扣费比较")
    return {
        "decision": "ACTIVATE" if changed else "KEEP_CURRENT",
        "winner": winner["modelId"],
        "protocolHash": protocol_hash,
        "sampleHash": current["sampleHash"],
        "decisions": decisions,
        "shadows": [d["modelId"] for d in decisions if d["decision"] == "RUN_AS_SHADOW"][:3],
    }


def compare_and_activate(experiment_id, current, candidates, protocol, *, publish=True):
    """研究完成调用此入口；先保存比较证据，再CAS切换，回执可独立重读。"""
    protocol_hash = fingerprint(protocol)
    result = choose_candidate(current, candidates, protocol_hash=protocol_hash)
    proof = {"protocol": protocol, "current": current, "candidates": candidates, "result": result}
    with get_engine().begin() as connection:
        connection.execute(
            text("""INSERT INTO model_evaluation(evaluation_id,experiment_id,content_hash,payload)
          VALUES(:id,:experiment,:hash,CAST(:payload AS jsonb)) ON CONFLICT(content_hash) DO NOTHING"""),
            {"id": uuid4(), "experiment": experiment_id, "hash": fingerprint(proof), "payload": encode(proof)},
        )
    if not publish:
        return result | {"publication": "CANDIDATE_ONLY"}
    if result["decision"] == "ACTIVATE":
        package = load_model(result["winner"])["manifest"]
        if package["trainedAt"] and datetime.fromisoformat(package["trainedAt"]) > datetime.now().astimezone():
            raise PredictionFailure("MODEL_NOT_AVAILABLE_AT_TIME", "ACTIVATE", "模型尚未到实际训练完成时间")
        result["activation"] = activate(
            result["winner"],
            reason=proof,
            expected_revision=current["activationRevision"],
            shadow_ids=result["shadows"],
        )
    else:
        with get_engine().begin() as connection:
            connection.execute(
                text("""UPDATE model_route SET shadow_ids=CAST(:shadows AS jsonb)
              WHERE model_id=:id AND revision=:revision"""),
                {
                    "shadows": encode(result["shadows"]),
                    "id": current["modelId"],
                    "revision": current["activationRevision"],
                },
            )
    return result
