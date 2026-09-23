"""按逐个估值日导出公共回放输入；成交账本仅由Java同一个决策组件推进。"""

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

from app.services.prediction_contract import PredictionFailure, fingerprint, prediction_policy
from app.services.prediction_features import ZONE, build_features, cash_events, read_fund_data
from app.services.prediction_models import infer_package
from app.services.prediction_replay_models import replay_model_bundles


def replay_inputs(code: str, start: date, end: date, *, frozen_bundles=None, auto_cycle_id=None):
    if not start < end < datetime.now(ZONE).date() or (end - start).days > 732:
        raise PredictionFailure("REPLAY_RANGE_INVALID", "REPLAY", "回放最多两年，结束日必须在今天之前")
    cutoff = datetime.combine(end + timedelta(days=1), time(23, 59), ZONE)
    if auto_cycle_id:
        from app.services.auto_model_store import frozen_input

        # 训练和回放读取同一份冻结行情；恢复时绝不换成后来修订的数据。
        data = frozen_input(
            auto_cycle_id,
            "fund:" + code,
            lambda: read_fund_data(code, cutoff, replay=True, start=start - timedelta(days=160)),
        )
    else:
        data = read_fund_data(code, cutoff, replay=True, start=start - timedelta(days=160))
    if "frozenFailure" in data:
        failure = data["frozenFailure"]
        raise PredictionFailure(failure["code"], "FROZEN_INPUT", failure["summary"], retryable=False)
    sessions = [day for day in data["calendar"].sessions if start <= day <= end]
    nav = {row["nav_date"]: row["unit_nav"] for row in data["navs"]}
    missing = [str(day) for day in sessions if day not in nav]
    if missing:
        raise PredictionFailure(
            "REPLAY_NAV_GAP", "REPLAY", "成交净值缺失，不能压缩日期跳过该日", details={"dates": missing[:20]}
        )
    dividends = cash_events(data, sessions, cutoff, replay=True)
    if frozen_bundles is None:
        bundles, excluded = replay_model_bundles(start)
    else:
        from app.services.prediction_models import load_model
        from app.services.prediction_research import validate_historical_model

        bundles, excluded = [], []
        for bundle in frozen_bundles:
            models = {}
            for ref in bundle["modelRefs"]:
                model = load_model(ref["modelId"])
                if model["modelHash"] != ref["modelHash"]:
                    raise PredictionFailure("REPLAY_MODEL_HASH_MISMATCH", "REPLAY", "冻结模型指纹变化", retryable=False)
                validate_historical_model(model["manifest"], datetime.combine(start, time.min, ZONE))
                models[ref["horizonId"]] = model | {"activationRevision": ref["activationRevision"]}
            bundles.append(bundle | {"models": models})
    frames, failures = [], []
    bundle_frames = {bundle["id"]: [] for bundle in bundles}
    failed_bundles = set()
    policy = prediction_policy()
    for day in sessions:
        asof = datetime.combine(day, time(10), ZONE)
        features, feature = {}, None
        for horizon in policy["horizons"]:
            try:
                feature = build_features(data, asof, horizon["lookback_returns"], replay=True, persist=False)
                features[horizon["horizon_id"]] = feature
            except PredictionFailure as error:
                failures.append({"date": str(day), "horizonId": horizon["horizon_id"], "error": error.payload})
        # 公告数据的历史实收证据不足，单列重建组；不把当前经理/规模套到过去。
        facts = [
            {
                "title": "分红公告",
                "content": f"公告{row['ann_date']}，每份{row['cash_dividend']}元；不预设利好利空",
                "source": data["fund"]["source_code"] + ":" + row["source_event_key"],
                "factor": 0,
            }
            for row in data["dividends"]
            if row["ann_date"] and 0 <= (day - row["ann_date"]).days <= 30
        ]
        frames.append(
            {
                "date": str(day),
                "nav": str(nav[day]),
                "cashDividend": str(dividends.get(day, Decimal(0))),
                "redemptionPaused": False,
                "input": {
                    "predictions": [],
                    "trendRisk": feature["features"]["trendRiskFactor"] if feature else None,
                    "facts": facts,
                    "missing": ["历史申赎开放状态未恢复，按正常开放假设", "新闻及经理历史版本不足未参与"],
                    "held": False,
                    "preference": "BALANCED",
                    "defaultPreference": True,
                    "holdingGainRate": None,
                    "currentDrawdown": feature["features"]["currentDrawdown"] if feature else None,
                    "personalRule": None,
                    "constraints": [],
                },
            }
        )
        # 各组合共用同一天的净值、分红、趋势和费用输入，只替换实际执行的预测模型。
        for bundle in bundles:
            if bundle["id"] in failed_bundles:
                continue
            signals = []
            try:
                for horizon in policy["horizons"]:
                    horizon_id = horizon["horizon_id"]
                    if horizon_id not in features:
                        raise PredictionFailure("REPLAY_FEATURE_MISSING", "REPLAY", "该日多周期输入不完整")
                    model, feature_input = bundle["models"][horizon_id], features[horizon_id]
                    answer = infer_package(model["manifest"], feature_input["features"])
                    signals.append(
                        {
                            "predictionId": f"REPLAY:{bundle['id']}:{code}:{day}:{horizon_id}",
                            "horizonId": horizon_id,
                            "direction": answer["direction"],
                            "modelId": model["modelId"],
                            "modelHash": model["modelHash"],
                            "activationRevision": model["activationRevision"],
                            "dataAsOf": feature_input["dataAsOf"],
                        }
                    )
            except (PredictionFailure, KeyError, ValueError) as error:
                reason = error.payload["summary"] if isinstance(error, PredictionFailure) else "模型所需特征或参数无效"
                excluded.append({"label": bundle["label"], "reason": f"{day}：{reason}"})
                failed_bundles.add(bundle["id"])
                continue
            bundle_frames[bundle["id"]].append(frames[-1] | {"input": frames[-1]["input"] | {"predictions": signals}})
    result = {
        "fundCode": code,
        "family": str(data["fund"]["fund_master_id"]),
        "startDate": str(start),
        "endDate": str(end),
        "frames": frames,
        "comparisonVersion": "MODEL_BUNDLE_REPLAY_V1",
        "modelComparisons": [
            {k: v for k, v in bundle.items() if k != "models"} | {"frames": bundle_frames[bundle["id"]]}
            for bundle in bundles
            if bundle["id"] not in failed_bundles
        ],
        "excludedModels": excluded,
        "failures": failures,
        "policyVersion": policy["version"],
        "policyHash": fingerprint(policy),
        "dataQuality": "ASSUMED_AVAILABILITY",
        "evidenceLevel": "DEVELOPMENT_ONLY",
        "modelScope": (
            "当前采用与已登记候选的完整多周期组合；训练标签早于区间。"
            "今天的模型回看历史，不代表当时已经运行，也不据本次区间自动选优"
        ),
        "eventRuleVersion": "PUBLIC_FACT_NEUTRAL_V1",
        "eventRuleDecision": "本批公告无可靠方向解释，保留事实、数值中性",
        "calendarHash": data["calendar"].source_hash,
    }
    result["inputHash"] = fingerprint(result)
    result["builtAt"] = datetime.now(UTC).isoformat()
    return result
