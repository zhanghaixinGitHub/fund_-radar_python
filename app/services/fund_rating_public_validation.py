"""对真实公开原件进行独立复算和敏感性检验；验证范围不扩大为预测有效性。"""

import math
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal

from app.models.fund_rating import RatingMethodology
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_public_policy import (
    BASIS,
    ETF_BASIS,
    INDEX_BASES,
    MARKET_BASES,
    PUBLIC_BASES,
    calculate_public,
    policy,
)
from app.services.fund_rating_rules import DIMENSIONS


def independent_score(metrics: dict, config: dict) -> tuple[float, dict]:
    """使用独立浮点区间计算，不调用生产插值函数，检查分段、权重和总和。"""
    result = {}
    for dimension, terms in config["terms"].items():
        subtotal = 0.0
        for key, weight in terms:
            points = [(float(a), float(b)) for a, b in config["anchors"][key]]
            x = float(metrics[key])
            if x <= points[0][0]:
                value = points[0][1]
            elif x >= points[-1][0]:
                value = points[-1][1]
            else:
                a, sa, b, sb = next(
                    (a, sa, b, sb) for (a, sa), (b, sb) in zip(points, points[1:], strict=False) if a <= x < b
                )
                value = (sa * (b - x) + sb * (x - a)) / (b - a)
            subtotal += value * weight / 100.0
        result[dimension] = subtotal
    return sum(result[k] * float(config["weights"][k]) / 100 for k in DIMENSIONS), result


def independent_history(bundle: dict, metrics: dict, *, history_only: bool = False) -> dict:
    """直接从真实逐日净值/事件复算，不引用生产历史指标辅助函数。"""
    if bundle.get("basis") in INDEX_BASES:
        from app.services.fund_rating_public_index import independent

        return independent(bundle, metrics)
    if bundle.get("basis") in MARKET_BASES:
        from app.services.fund_rating_public_market import independent_metrics

        return independent_metrics(bundle, metrics)
    if bundle.get("basis") == ETF_BASIS and not history_only:
        from app.services.fund_rating_public_etf import independent_etf_history

        return independent_etf_history(bundle, metrics)
    from datetime import date

    h = bundle["history"]
    nav, dividends, splits = (list(map(float, h[k])) for k in ("unit_nav", "cash_dividend_per_share", "split_factor"))
    values = [1.0]
    for i in range(1, len(nav)):
        values.append(values[-1] * (nav[i] + dividends[i]) * splits[i] / nav[i - 1])
    monthly = [values[h["dates"].index(d)] for d in h["month_ends"]]
    returns = [b / a - 1 for a, b in zip(values, values[1:], strict=False)]
    peak, drawdown = values[0], 0.0
    for x in values:
        peak = max(peak, x)
        drawdown = max(drawdown, (peak - x) / peak)
    rolling = [monthly[i] / monthly[i - 12] - 1 for i in range(12, len(monthly))]
    actual = {
        "return_1y": monthly[-1] / monthly[-13] - 1,
        "return_3y": (values[-1] / values[0])
        ** (365.25 / (date.fromisoformat(h["dates"][-1]) - date.fromisoformat(h["dates"][0])).days)
        - 1,
        "drawdown": drawdown,
        "downside": math.sqrt(252 * sum(min(r, 0) ** 2 for r in returns) / len(returns)),
        "rolling_positive": sum(r > 0 for r in rolling) / len(rolling),
        "monthly_positive": sum(b > a for a, b in zip(monthly, monthly[1:], strict=False)) / (len(monthly) - 1),
    }
    actual["calmar"] = (
        actual["return_3y"] / drawdown
        if drawdown
        else 2
        if actual["return_3y"] > 0
        else 0
        if actual["return_3y"] == 0
        else -1
    )
    if history_only:
        if any(not math.isclose(v, float(metrics[k]), rel_tol=1e-12, abs_tol=1e-12) for k, v in actual.items()):
            raise ValueError("RATING_PUBLIC_INDEPENDENT_HISTORY_MISMATCH")
        return {
            "metrics": actual,
            "rolling_one_year": [{"end": d, "return": r} for d, r in zip(h["month_ends"][12:], rolling, strict=True)],
            "note": "仅历史收益窗口，不把当前持仓及经理资料回填成历史综合评级",
        }
    # 非历史维度也从事实重新计算，不能只复算生产程序已经给出的分数。
    holding, disclosure, manager = (bundle[k] for k in ("holdings", "disclosure", "manager"))
    total = sum(float(p["amount"]) for p in holding["positions"])
    for field in ("issuer", "industry"):
        groups = {}
        for p in holding["positions"]:
            groups[p[field]] = groups.get(p[field], 0.0) + float(p["amount"])
        actual[field + "_hhi"] = sum((v / total) ** 2 for v in groups.values())
    net = float(holding["net_assets"])
    actual.update(
        net_assets=net,
        restricted_ratio=float(disclosure["restricted_stock_amount"]) / net,
        cash_ratio=float(holding["balance_sheet"]["other_assets"]["货币资金"]) / net,
        **{k: float(disclosure[k]) for k in ("holder_risk_band", "strategy_continuity", "governance_state")},
    )
    start, end = date.fromisoformat(h["dates"][0]), date.fromisoformat(h["dates"][-1])
    teams = manager["teams"]
    actual["team_months"] = min(36, (end - date.fromisoformat(teams[-1]["effective_date"])).days / 30.4375)
    actual["team_changes"] = sum(start < date.fromisoformat(t["effective_date"]) <= end for t in teams[1:])
    actual["calmar"] = (
        actual["return_3y"] / drawdown
        if drawdown
        else 2
        if actual["return_3y"] > 0
        else 0
        if actual["return_3y"] == 0
        else -1
    )
    fees = bundle["key_information"]["fees"]
    buy, sell = float(fees["subscription"]), float(fees["redemption"])
    purchase = buy / (1 + buy) if fees["subscription_mode"] == "EXTERNAL" else buy
    actual["cost_1y"] = purchase + (1 - purchase) * sell + float(fees["operating_cost"])
    if any(not math.isclose(v, float(metrics[k]), rel_tol=1e-12, abs_tol=1e-12) for k, v in actual.items()):
        raise ValueError("RATING_PUBLIC_INDEPENDENT_HISTORY_MISMATCH")
    return {
        "metrics": actual,
        "rolling_one_year": [{"end": d, "return": r} for d, r in zip(h["month_ends"][12:], rolling, strict=True)],
        "note": "仅历史收益窗口，不把当前持仓及经理资料回填成历史综合评级",
    }


def precision_sensitivity(bundle: dict, baseline: dict) -> dict:
    """按净值公开四位小数的半单位扰动；仅是精度情景，不作为新事实或置信区间。"""
    from datetime import date

    from app.services.fund_rating_public_inputs import build_public_metrics

    rows = []
    for label, mode in (("全部增加半单位", 1), ("全部减少半单位", -1), ("交替增减半单位", 0)):
        changed = deepcopy(bundle)
        if bundle["basis"] in {*MARKET_BASES, *INDEX_BASES}:
            # 存储八位小数不等于原始净值有八位精度。按公开单位净值四位小数的半单位，
            # 用当日复权/单位净值比例传递到复权值；这只是舍入情景，不修改原始事实。
            for i, row in enumerate(changed["history"]["rows"]):
                if row["adjusted_nav"] is not None:
                    half_unit = Decimal("0.00005") * Decimal(row["adjusted_nav"]) / Decimal(row["unit_nav"])
                    row["adjusted_nav"] = str(Decimal(row["adjusted_nav"]) + half_unit * (mode or (1 if i % 2 else -1)))
            metrics, _, _ = build_public_metrics(changed, date.fromisoformat(bundle["as_of_date"]), datetime.now(UTC))
            result = calculate_public(metrics, config=policy(bundle["basis"]))
            rows.append({"scenario": label, "score": result["score"], "grade": result["grade"]})
            continue
        changed["history"]["unit_nav"] = [
            str(Decimal(v) + Decimal("0.00005") * (mode or (1 if i % 2 else -1)))
            for i, v in enumerate(bundle["history"]["unit_nav"])
        ]
        metrics, _, _ = build_public_metrics(changed, date.fromisoformat(bundle["as_of_date"]), datetime.now(UTC))
        result = calculate_public(metrics, config=policy(bundle["basis"]))
        rows.append({"scenario": label, "score": result["score"], "grade": result["grade"]})
    if bundle["basis"] in {ETF_BASIS, *INDEX_BASES}:
        # 官方基准累计收益显示到0.01%，对应归一化指数半单位0.00005。
        for mode in (1, -1):
            changed = deepcopy(bundle)
            half_unit = Decimal(policy(bundle["basis"])["benchmark_public_precision"])
            changed["history"]["benchmark_index"] = [
                str(Decimal(v) + half_unit * mode * (1 if i % 2 else -1))
                for i, v in enumerate(bundle["history"]["benchmark_index"])
            ]
            metrics, _, _ = build_public_metrics(changed, date.fromisoformat(bundle["as_of_date"]), datetime.now(UTC))
            result = calculate_public(metrics, config=policy(bundle["basis"]))
            rows.append(
                {"scenario": f"基准披露精度交替半单位{mode:+d}", "score": result["score"], "grade": result["grade"]}
            )
    return {
        "scenarios": rows,
        "changed_grades": sum(r["grade"] != baseline["grade"] for r in rows),
        "note": "只检验公开净值四位小数精度对结论的影响；不覆盖所有误差或市场变化",
    }


def sensitivity(metrics: dict, baseline: dict, *, config: dict | None = None) -> dict:
    config = config or policy()
    values = []
    for key in DIMENSIONS:
        for factor in (Decimal(".8"), Decimal("1.2")):
            changed = deepcopy(config)
            weight = Decimal(config["weights"][key])
            changed["weights"] = {
                k: weight * factor if k == key else Decimal(w) * (100 - weight * factor) / (100 - weight)
                for k, w in config["weights"].items()
            }
            adjust = next(k for k in DIMENSIONS if k != key)
            changed["weights"][adjust] += 100 - sum(changed["weights"].values())
            result = calculate_public(metrics, config=changed)
            values.append({"dimension": key, "factor": str(factor), "score": result["score"], "grade": result["grade"]})
    equal = deepcopy(config)
    equal["weights"] = dict.fromkeys(DIMENSIONS, Decimal("12.5"))
    equal_result = calculate_public(metrics, config=equal)
    return {
        "weight_20_percent": values,
        "minimum": min((v["score"] for v in values), key=Decimal),
        "maximum": max((v["score"] for v in values), key=Decimal),
        "changed_grades": sum(v["grade"] != baseline["grade"] for v in values),
        "equal_weight_alternative": equal_result,
        "note": "敏感性结果不用于倒调正式权重",
    }


def validate_category(members: list[dict]) -> dict:
    admitted = [m for m in members if m["admitted"]]
    bases = {m["bundle"].get("basis") for m in admitted}
    if not admitted or len(bases) != 1 or not bases <= PUBLIC_BASES:
        return {"basis": BASIS, "eligible": False, "validation_scope": "缺少完整八维公开原件"}
    basis = next(iter(bases))
    from app.services.fund_rating_reviewed_quality import BASES as REVIEWED_BASES

    if basis in REVIEWED_BASES:
        from app.services.fund_rating_reviewed_quality import validate_category as validate_reviewed

        return validate_reviewed(members)
    config = policy(basis)
    outcomes = {}
    for m in admitted:
        result = calculate_public(m["metrics"], config=config)
        score, dimensions = independent_score(m["metrics"], config)
        if abs(score - float(result["score"])) > 1e-10 or any(
            abs(dimensions[k] - float(result["dimensions"][k])) > 1e-10 for k in DIMENSIONS
        ):
            raise ValueError("RATING_PUBLIC_INDEPENDENT_SCORE_MISMATCH")
        history = independent_history(m["bundle"], m["metrics"])
        outcomes[m["fund_code"]] = {
            "result": result,
            "independent_score": score,
            "independent_dimensions": dimensions,
            "history_check": history,
            "sensitivity": sensitivity(m["metrics"], result, config=config),
            "precision_sensitivity": precision_sensitivity(m["bundle"], result),
        }
    from collections import Counter

    return {
        "basis": basis,
        "eligible": True,
        "methodology_id": digest(config),
        "real_cases": outcomes,
        "distribution": dict(Counter(v["result"]["grade"] for v in outcomes.values())),
        "validation_scope": "公开事实口径、独立运算、缺失边界和敏感性；不证明同类排名或未来预测有效性",
    }


def activate_public(session, report: dict, review: str, *, basis: str = BASIS) -> str:
    """真实重算一致才激活本机公开资料政策；未验证类别不自动开放。"""
    from datetime import date

    from app.services.fund_rating_admin import trial

    if not review.strip():
        raise ValueError("RATING_REVIEW_REQUIRED")
    current = trial(session, date.fromisoformat(report["as_of"]))
    if basis not in PUBLIC_BASES:
        raise ValueError("RATING_PUBLIC_BASIS_UNSUPPORTED")
    actual = {c["category"]: c for c in current["categories"] if c.get("basis") == basis and c["eligible"]}
    supplied = {c["category"]: c for c in report["categories"] if c.get("basis") == basis and c["eligible"]}
    if (
        not actual
        or set(actual) != set(supplied)
        or any(
            actual[k]["input_hash"] != supplied[k]["input_hash"] or actual[k]["real_cases"] != supplied[k]["real_cases"]
            for k in actual
        )
    ):
        raise ValueError("RATING_REAL_TRIAL_REQUIRED")
    config = policy(basis)
    key = digest(config)
    method = session.get(RatingMethodology, key)
    if not method:
        method = RatingMethodology(
            methodology_id=key, family=config["family"], config=config, created_at=datetime.now(UTC)
        )
        session.add(method)
    method.validation = {"report": current, "review": review, "validated_at": datetime.now(UTC).isoformat()}
    method.active = True
    return key
