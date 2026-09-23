"""按逐个估值日导出公共回放输入；成交账本仅由Java同一个决策组件推进。"""

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

from app.services.prediction_contract import PredictionFailure, fingerprint, prediction_policy
from app.services.prediction_features import ZONE, build_features, cash_events, read_fund_data
from app.services.prediction_models import baseline_package, infer_package


def replay_inputs(code: str, start: date, end: date):
    if not start < end < date.today() or (end - start).days > 732:
        raise PredictionFailure("REPLAY_RANGE_INVALID", "REPLAY", "回放最多两年，结束日必须在今天之前")
    cutoff = datetime.combine(end + timedelta(days=1), time(23, 59), ZONE)
    data = read_fund_data(code, cutoff, replay=True, start=start - timedelta(days=160))
    sessions = [day for day in data["calendar"].sessions if start <= day <= end]
    nav = {row["nav_date"]: row["unit_nav"] for row in data["navs"]}
    missing = [str(day) for day in sessions if day not in nav]
    if missing:
        raise PredictionFailure(
            "REPLAY_NAV_GAP", "REPLAY", "成交净值缺失，不能压缩日期跳过该日", details={"dates": missing[:20]}
        )
    dividends = cash_events(data, sessions, cutoff, replay=True)
    frames, failures = [], []
    policy = prediction_policy()
    for day in sessions:
        asof = datetime.combine(day, time(10), ZONE)
        signals, feature = [], None
        for horizon in policy["horizons"]:
            try:
                feature = build_features(data, asof, horizon["lookback_returns"], replay=True, persist=False)
                package = baseline_package(horizon["horizon_id"])
                digest = fingerprint(package)
                answer = infer_package(package, feature["features"])
                signals.append(
                    {
                        "predictionId": f"REPLAY:{code}:{day}:{horizon['horizon_id']}",
                        "horizonId": horizon["horizon_id"],
                        "direction": answer["direction"],
                        "modelId": "MP-" + digest[:32],
                        "modelHash": digest,
                        "activationRevision": 0,
                        "dataAsOf": feature["dataAsOf"],
                    }
                )
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
                    "predictions": signals,
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
    result = {
        "fundCode": code,
        "startDate": str(start),
        "endDate": str(end),
        "frames": frames,
        "failures": failures,
        "policyVersion": policy["version"],
        "policyHash": fingerprint(policy),
        "dataQuality": "ASSUMED_AVAILABILITY",
        "evidenceLevel": "DEVELOPMENT_ONLY",
        "modelScope": "各周期事先固定的基础模型，无训练标签；不把今天训练的模型冒充过去已运行",
        "eventRuleVersion": "PUBLIC_FACT_NEUTRAL_V1",
        "eventRuleDecision": "本批公告无可靠方向解释，保留事实、数值中性",
        "calendarHash": data["calendar"].source_hash,
    }
    result["inputHash"] = fingerprint(result)
    result["builtAt"] = datetime.now(UTC).isoformat()
    return result
