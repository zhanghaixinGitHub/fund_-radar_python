"""有限历史实验队列：冻结配置、成熟标签、同题对照、检查点及实验采用。"""

import importlib.metadata
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories.prediction_store import encode, one
from app.services.prediction_contract import (
    PredictionFailure,
    fingerprint,
    prediction_policy,
    reinvested_series,
    target_dates,
)
from app.services.prediction_features import build_features, cash_events, read_fund_data
from app.services.prediction_models import FEATURES, freeze_routes, infer_package, load_model, register_model, route_key
from app.services.prediction_selection import compare_and_activate, evaluate_answers

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="historical-research")
ZONE = ZoneInfo("Asia/Shanghai")


def validate_spec(spec):
    dates = [date.fromisoformat(spec[k]) for k in ("trainStart", "trainEnd", "validationEnd", "selectionEnd")]
    if not dates[0] < dates[1] < dates[2] < dates[3] < date.today():
        raise PredictionFailure("RESEARCH_SPLIT_INVALID", "RESEARCH", "训练、验证、选优时间必须严格先后且已经结束")
    if len(spec["fundCodes"]) > 50 or len(set(spec["fundCodes"])) != len(spec["fundCodes"]):
        raise ValueError("RESEARCH_FUND_BUDGET")
    if (dates[3] - dates[0]).days > 1461 or spec.get("stride", 5) < 5:
        raise ValueError("RESEARCH_RESOURCE_BUDGET")
    if spec.get("evidenceLevel") != "DEVELOPMENT_ONLY":
        raise PredictionFailure("EVALUATION_ALREADY_USED", "RESEARCH", "现有历史区间已用于研究，只能标记为开发证据")
    allowed = {h["horizon_id"] for h in prediction_policy()["horizons"]}
    if not set(spec["horizonIds"]) <= allowed:
        raise ValueError("RESEARCH_HORIZON_NOT_CONFIGURED")


def mature_training_rows(samples, fit_as_of):
    """拒绝跨区间未成熟标签，而非悄悄让未来答案进入标准化或拟合。"""
    for sample in samples:
        if datetime.fromisoformat(sample["labelAvailableAt"]) > fit_as_of:
            raise PredictionFailure(
                "TRAINING_LABEL_NOT_MATURE",
                "TRAIN",
                "训练标签尚未成熟",
                details={"sampleKey": sample["key"], "labelAvailableAt": sample["labelAvailableAt"]},
            )
        if datetime.fromisoformat(sample["knowledgeCutoff"]) >= fit_as_of:
            raise PredictionFailure("TRAINING_TIME_LEAKAGE", "TRAIN", "输入时点跨入拟合截止")
    return samples


def validate_historical_model(manifest, fit_as_of):
    """模型标签必须在模拟拟合时点已成熟；今天采用的包不能自动获得过去使用资格。"""
    if manifest["adapter"] == "NAV_MOMENTUM_V1":
        return
    label_end = manifest.get("labelEndMax")
    if not label_end or datetime.fromisoformat(label_end) > fit_as_of:
        raise PredictionFailure(
            "MODEL_NOT_AVAILABLE_AT_TIME",
            "RESEARCH",
            "当前模型标签晚于历史拟合截止，需按该时点重新训练后比较",
            details={"labelEndMax": label_end, "simulatedFitKnowledgeCutoff": fit_as_of.isoformat()},
        )


def create_research(spec):
    validate_spec(spec)
    run_id = uuid4()
    frozen = spec | {
        "protocolVersion": "EXPERIMENT_SELECTION_V1",
        "seed": 42,
        "candidates": ["NAV_MOMENTUM_BASELINE_V1", "TOTAL_RETURN_LOGISTIC_V1"],
        "scopeNote": "按当前关注集合回看历史，不代表全市场；历史版本存在可得性假设",
        "policyHash": fingerprint(prediction_policy()),
        "routes": freeze_routes(),
    }
    with get_engine().begin() as c:
        c.execute(text("SELECT pg_advisory_xact_lock(721109,2)"))
        if (
            c.execute(
                text("SELECT count(*) FROM prediction_research_run WHERE status IN ('QUEUED','RUNNING')")
            ).scalar()
            >= 2
        ):
            raise PredictionFailure("RESEARCH_BUSY", "QUEUE", "已有研究任务排队，请等待完成或取消")
        c.execute(
            text("""INSERT INTO prediction_research_run(run_id,spec_hash,spec,status)
          VALUES(:id,:hash,CAST(:spec AS jsonb),'QUEUED')"""),
            {"id": run_id, "hash": fingerprint(frozen), "spec": encode(frozen)},
        )
        c.execute(
            text("""INSERT INTO prediction_evaluation_usage(usage_id,sample_hash,purpose,run_id,start_date,end_date)
          VALUES(:id,:hash,'DEVELOPMENT_ONLY',:run,:start,:end)"""),
            {
                "id": uuid4(),
                "hash": fingerprint(spec["fundCodes"]),
                "run": run_id,
                "start": spec["trainStart"],
                "end": spec["selectionEnd"],
            },
        )
    _executor.submit(run_research, run_id)
    return research_status(run_id)


def research_status(run_id):
    with get_engine().connect() as c:
        value = one(c, "SELECT * FROM prediction_research_run WHERE run_id=:id", id=run_id)
    if not value:
        raise PredictionFailure("RESEARCH_NOT_FOUND", "READ", "研究任务不存在", retryable=False)
    return value


def checkpoint(run_id, value, *, result=None, status="RUNNING"):
    with get_engine().begin() as c:
        c.execute(
            text("""UPDATE prediction_research_run SET checkpoint=CAST(:point AS jsonb),
          result=CAST(:result AS jsonb),status=CAST(:status AS varchar),updated_at=clock_timestamp(),
          lease_until=CASE WHEN CAST(:status AS varchar)='RUNNING'
            THEN clock_timestamp()+interval '5 minutes' ELSE NULL END
          WHERE run_id=:id"""),
            {"point": encode(value), "result": encode(result), "status": status, "id": run_id},
        )


def fit_package(samples, horizon, fit_as_of):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    mature_training_rows(samples, fit_as_of)
    if len(samples) < 40 or len({s["actual"] for s in samples}) < 2:
        raise PredictionFailure("TRAINING_SAMPLE_INSUFFICIENT", "TRAIN", "成熟训练样本不足或只包含一个方向")
    x = np.array([[s["features"][k] for k in FEATURES] for s in samples])
    y = np.array([int(s["actual"] == "UP") for s in samples])
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales[scales == 0] = 1
    with threadpool_limits(limits=1):
        trained = LogisticRegression(C=1.0, random_state=42, max_iter=1000).fit((x - means) / scales, y)
    package = {
        "adapter": "LOGISTIC_STANDARDIZED_V1",
        "recipeVersion": "TOTAL_RETURN_LOGISTIC_V1",
        "horizonId": horizon,
        "targetDefinitionId": prediction_policy()["target_definition_id"],
        "assetGroup": "ALL",
        "featureSchemaVersion": "NAV_TOTAL_RETURN_V1",
        "features": list(FEATURES),
        "featureUnits": ["RATIO"] * 6 + ["SESSIONS"],
        "missingPolicy": "FALLBACK_BASELINE",
        "threshold": 0.5,
        "labelEndMax": max(s["labelAvailableAt"] for s in samples),
        "trainedAt": datetime.now(UTC).isoformat(),
        "simulatedFitKnowledgeCutoff": fit_as_of.isoformat(),
        "codeVersion": "PREDICTION_POLICY_V1",
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")},
        "evidenceLevel": "DEVELOPMENT_ONLY",
        "trainingSampleHash": fingerprint(samples),
        "parameters": {
            "mean": means.tolist(),
            "scale": scales.tolist(),
            "coefficients": trained.coef_[0].tolist(),
            "intercept": float(trained.intercept_[0]),
        },
    }
    actual = trained.predict_proba((x[:10] - means) / scales)[:, 1]
    restored = [infer_package(package, s["features"])["score"] for s in samples[:10]]
    if not np.allclose(actual, restored, atol=1e-12, rtol=0):
        raise PredictionFailure("MODEL_RESTORE_MISMATCH", "TRAIN", "训练与JSON适配器推理不一致")
    package["restoreMaxDiff"] = float(np.max(np.abs(actual - restored)))
    return package


def run_research(run_id):
    with get_engine().begin() as c:
        claimed = c.execute(
            text("""UPDATE prediction_research_run SET status='RUNNING',
                   lease_until=clock_timestamp()+interval '5 minutes'
                   WHERE run_id=:id AND status IN ('QUEUED','RUNNING','INTERRUPTED')
                   AND (lease_until IS NULL OR lease_until<clock_timestamp()) RETURNING run_id"""),
            {"id": run_id},
        ).scalar()
        if not claimed:
            return
    state = research_status(run_id)
    if state["status"] == "SUCCEEDED":
        return
    spec, saved = state["spec"], state["checkpoint"]
    results = (state["result"] or {}).get("horizons", {})
    config = {h["horizon_id"]: h for h in prediction_policy()["horizons"]}
    cutoff = datetime.combine(date.fromisoformat(spec["selectionEnd"]), time(23, 59), ZONE)
    data_cache = {}
    try:
        for horizon_id in spec["horizonIds"]:
            if horizon_id in results:
                continue
            if research_status(run_id)["cancel_requested"]:
                checkpoint(run_id, saved, result={"horizons": results}, status="CANCELLED")
                return
            horizon = config[horizon_id]
            samples, failures = [], []
            planned_selection = []
            for fund in spec["fundCodes"]:
                with get_engine().begin() as c:
                    cancelled = c.execute(
                        text("""UPDATE prediction_research_run
                              SET lease_until=clock_timestamp()+interval '5 minutes' WHERE run_id=:id
                              RETURNING cancel_requested"""),
                        {"id": run_id},
                    ).scalar()
                if cancelled:
                    checkpoint(run_id, saved, result={"horizons": results}, status="CANCELLED")
                    return
                try:
                    if fund not in data_cache:
                        data_cache[fund] = read_fund_data(
                            fund, cutoff, replay=True, start=date.fromisoformat(spec["trainStart"])
                        )
                    data = data_cache[fund]
                    sessions = [
                        d for d in data["calendar"].sessions if spec["trainStart"] <= str(d) <= spec["selectionEnd"]
                    ][:: spec.get("stride", 5)]
                    nav = {r["nav_date"]: r["unit_nav"] for r in data["navs"]}
                    for day in sessions:
                        asof = datetime.combine(day, time(10), ZONE)
                        key = f"{fund}:{day}:{horizon_id}"
                        planned = {
                            "key": key,
                            "fund": fund,
                            "family": str(data["fund"]["fund_master_id"]),
                            "fundType": data["fund"]["fund_type"],
                            "date": str(day),
                            "actual": None,
                        }
                        if spec["validationEnd"] < str(day) <= spec["selectionEnd"]:
                            planned_selection.append(planned)
                        try:
                            target = target_dates(data["calendar"], asof, horizon_id)
                            if not target["endDate"] or target["endDate"] >= spec["selectionEnd"]:
                                continue
                            end = date.fromisoformat(target["endDate"])
                            future = data["calendar"].sessions
                            index = future.index(end) + 1
                            if index >= len(future):
                                continue
                            available = datetime.combine(future[index], time(), ZONE)
                            features = build_features(
                                data, asof, horizon["lookback_returns"], replay=True, persist=False
                            )
                            if not set(FEATURES) <= features["features"].keys():
                                continue
                            dates = [d for d in future if day <= d <= end]
                            series = reinvested_series(dates, nav, cash_events(data, dates, cutoff, replay=True))
                            actual = "UP" if series[-1] > series[0] else "NON_UP"
                            planned["actual"] = actual
                            samples.append(
                                planned
                                | {
                                    "features": features["features"],
                                    "knowledgeCutoff": asof.isoformat(),
                                    "labelAvailableAt": available.isoformat(),
                                    "dataAsOf": features["dataAsOf"],
                                    "inputHash": features["featureHash"],
                                    "startDate": str(day),
                                    "endDate": str(end),
                                    "totalReturn": str(series[-1] / series[0] - 1),
                                    "dataQuality": "ASSUMED_AVAILABILITY",
                                }
                            )
                        except PredictionFailure as error:
                            failures.append({"key": key, "error": error.payload})
                except PredictionFailure as error:
                    failures.append({"fund": fund, "horizon": horizon_id, "error": error.payload})
            fit_asof = datetime.combine(date.fromisoformat(spec["trainEnd"]), time(23, 59), ZONE)
            fit = [
                s
                for s in samples
                if datetime.fromisoformat(s["labelAvailableAt"]) <= fit_asof
                and datetime.fromisoformat(s["knowledgeCutoff"]) < fit_asof
            ]
            candidate = register_model(fit_package(fit, horizon_id, fit_asof))
            route = spec["routes"][route_key(horizon_id)]
            baseline = load_model(route["model_id"])
            exam = [s for s in samples if spec["validationEnd"] < s["date"] <= spec["selectionEnd"]]
            # 有数据的题之外，基金级失败按相同日期计划补回覆盖分母。
            dates = sorted({p["date"] for p in planned_selection})
            represented = {p["fund"] for p in planned_selection}
            for fund in set(spec["fundCodes"]) - represented:
                planned_selection.extend(
                    {
                        "key": f"{fund}:{d}:{horizon_id}",
                        "fund": fund,
                        "family": fund,
                        "fundType": "UNKNOWN",
                        "date": d,
                        "actual": None,
                    }
                    for d in dates
                )
            protocol = {
                "version": "EXPERIMENT_SELECTION_V1",
                "specHash": state["spec_hash"],
                "target": prediction_policy()["target_definition_id"],
                "horizon": horizon_id,
            }
            sample_hash = fingerprint(planned_selection)
            evaluated = []
            answers_by_model = {}
            for model, cost in ((baseline, 1.0), (candidate, 2.0)):
                validate_historical_model(model["manifest"], fit_asof)
                answers = {s["key"]: infer_package(model["manifest"], s["features"]) for s in exam}
                answers_by_model[model["modelId"]] = answers
                evaluated.append(
                    {
                        "modelId": model["modelId"],
                        "metrics": evaluate_answers(planned_selection, answers),
                        "protocolHash": fingerprint(protocol),
                        "sampleHash": sample_hash,
                        "runnable": True,
                        "timeValid": True,
                        "cost": cost,
                        "recipe": model["manifest"]["recipeVersion"],
                        "labelEndMax": model["manifest"]["labelEndMax"],
                        "activationRevision": route["revision"],
                    }
                )
            selection = compare_and_activate(str(run_id), evaluated[0], evaluated[1:], protocol)
            results[horizon_id] = {
                "selection": selection,
                "models": evaluated,
                "failures": failures,
                "trainingSamples": len(fit),
                "evaluationSamples": len(exam),
                "planned": planned_selection,
                "answers": answers_by_model,
                "samples": exam,
                "evidenceLevel": "DEVELOPMENT_ONLY",
                "historicalQuality": "ASSUMED_AVAILABILITY",
                "note": "当前重建历史可得性；模型实际训练发生在今天，历史模拟拟合只用当时成熟标签",
            }
            saved = {"completedHorizons": list(results), "lastHorizon": horizon_id}
            checkpoint(run_id, saved, result={"horizons": results})
        checkpoint(run_id, saved, result={"horizons": results}, status="SUCCEEDED")
    except Exception as error:
        logger.exception("prediction_research.run_research >>> runId=%s failed", run_id)
        detail = (
            error.payload
            if isinstance(error, PredictionFailure)
            else {"code": "RESEARCH_ERROR", "summary": type(error).__name__}
        )
        checkpoint(run_id, saved, result={"horizons": results, "error": detail}, status="FAILED")


def recover_research():
    """只恢复已过期租约；保留完成周期的检查点，不与仍在运行的进程抢占。"""
    with get_engine().begin() as c:
        ids = (
            c.execute(
                text("""UPDATE prediction_research_run SET status='INTERRUPTED',lease_until=NULL
                           WHERE status IN ('RUNNING','QUEUED','INTERRUPTED')
                           AND (lease_until IS NULL OR lease_until<clock_timestamp()) RETURNING run_id""")
            )
            .scalars()
            .all()
        )
        for run_id in ids:
            c.execute(
                text("""INSERT INTO prediction_task_event(event_id,task_id,kind,payload)
                          VALUES(:event,:task,'RESEARCH_RECOVERED','{"reason":"恢复已保存的研究周期检查点"}')"""),
                {"event": uuid4(), "task": run_id},
            )
    for run_id in ids:
        _executor.submit(run_research, run_id)
    return len(ids)
