"""境内股票指数基金：使用发行人逐日业绩基准，八维同时完整才允许发布。

不以指数涨跌推断完整合同基准，不借用其他份额净值。主题集中度只解释持仓风险。
"""

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from app.core.config import get_settings
from app.services.fund_rating_public_market import balance, disclosures, key_information, provider_history
from app.services.fund_rating_public_stock_tables import parse
from app.services.fund_rating_rules import decimal

BASIS = "PUBLIC_DISCLOSURE_DOMESTIC_INDEX_V1"
SCOPE = {"basis": BASIS, "asset": "DOMESTIC_EQUITY_INDEX", "currency": "CNY"}


def index_policy() -> dict:
    """新指数口径在首次试算前固定；合同目标按每只基金原件核验，不因结果调参。"""
    from app.services.fund_rating_public_etf import etf_policy

    value = deepcopy(etf_policy())
    value.update(basis=BASIS, revision="public-domestic-index-2026-10-15-v1", scope=SCOPE)
    for field in ("tracking_benchmark", "tracking_error_target", "daily_bias_target"):
        value.pop(field)
    value["benchmark_public_precision"] = "0.0000005"
    value["input_contract"] = {
        "history": "OWN_AUTHORIZED_ADJUSTED_NAV_COMPLETE_THREE_YEARS",
        "benchmark": "ISSUER_PUBLISHED_COMPLETE_DAILY_CONTRACT_BENCHMARK",
        "tracking_targets": "EXPLICIT_PROSPECTUS_OR_SUMMARY_VALUES",
        "holdings": "ALL_DOMESTIC_STOCKS_RECONCILED_TO_BALANCE_AND_INDUSTRY_TABLE",
        "liquidity": "DISCLOSED_RESTRICTED_ASSETS_CASH_AND_TOTAL_PRODUCT_ASSETS",
        "unknowns": "REJECT_NO_FILL_NO_WEIGHT_REDISTRIBUTION",
    }
    return value


def facts(refs: dict, root: Path, code: str, name: str, as_of: date, *, history_reconciler=None) -> dict:
    from app.services.fund_rating_public_market import matches_name, public_document, publication

    def raw(label):
        ref = refs[label]
        value = (root / ref["file"]).read_bytes()
        if hashlib.sha256(value).hexdigest() != ref["sha256"]:
            raise ValueError("RATING_INDEX_ORIGINAL_HASH")
        return value

    report, _ = public_document(raw("half"), refs["half"], code, "中期报告")
    report_text = re.sub(r"\s", "", "\n".join(report))
    if (
        code not in report_text
        or not matches_name(report_text, name)
        or not re.search(r"2026年(?:中期报告|6月30日)", report_text)
    ):
        raise ValueError("RATING_INDEX_REPORT_IDENTITY_OR_PERIOD")
    summary, _ = public_document(raw("summary"), refs["summary"], code, "产品资料概要")
    prospectus = (
        public_document(raw("prospectus"), refs["prospectus"], code, "招募说明书")[0] if "prospectus" in refs else []
    )
    provider = json.loads(raw("provider"))
    if (
        provider.get("fund_code") != code
        or provider.get("source", {}).get("enabled") is not True
        or not provider.get("source", {}).get("authorization_verified_at")
    ):
        raise ValueError("RATING_INDEX_PROVIDER_IDENTITY_OR_AUTHORIZATION")
    current = [
        r["manager_name"]
        for r in provider["manager_records"]
        if r["begin_date"] and r["begin_date"] <= str(as_of) and (not r["end_date"] or r["end_date"] > str(as_of))
    ]
    info = key_information(summary, code, name, prospectus, current, index=True)
    if date.fromisoformat(info["published_on"]) > as_of:
        raise ValueError("RATING_INDEX_FUTURE_SUMMARY")
    b = balance(report)
    tables = parse(report, b["stock_amount"], b["net_assets"])
    d = disclosures(report, tables["holdings"], include_holder=False)
    _, history = provider_history(provider, as_of)
    benchmark = json.loads(raw("benchmark"))
    if refs["benchmark"].get("format") == "ISSUER_DAILY_BENCHMARK_WJ":
        # 发行人公开曲线采用另一种字段名；身份来自官网请求参数，并用本份额净值再次交叉核验。
        # 空值仍为空值，绝不沿用网页首行把空基准改为零的展示处理。
        if (
            benchmark.get("code") != "ETS-5BP0000"
            or refs["benchmark"].get("params", {}).get("fundCode") != code
            or not isinstance(benchmark.get("results"), list)
        ):
            raise ValueError("RATING_INDEX_BENCHMARK_RESPONSE")
        benchmark = {
            "code": 0,
            "data": [
                {
                    "navDate": date.fromisoformat(r["tradeDate"]).isoformat(),
                    "productCode": code,
                    "benchmarkRate": r.get("benchmarkRatio"),
                    "syNavRate": r.get("returnRatio"),
                }
                for r in benchmark["results"]
            ],
        }
    rows = benchmark.get("data")
    if benchmark.get("code") != 0 or not isinstance(rows, list) or len({r["navDate"] for r in rows}) != len(rows):
        raise ValueError("RATING_INDEX_BENCHMARK_RESPONSE")
    if any(r["productCode"] != code for r in rows):
        raise ValueError("RATING_INDEX_BENCHMARK_PRODUCT")
    dates = {r["navDate"]: r for r in rows}
    if any(day not in dates or dates[day].get("benchmarkRate") in (None, "") for day in history["dates"]):
        raise ValueError("RATING_INDEX_BENCHMARK_HISTORY_INCOMPLETE")
    history["benchmark_index"] = [str(1 + decimal(dates[day]["benchmarkRate"]) / 100) for day in history["dates"]]
    if any(decimal(v) <= 0 for v in history["benchmark_index"]):
        raise ValueError("RATING_INDEX_BENCHMARK_NONPOSITIVE")
    publisher = [1 + decimal(dates[day]["syNavRate"]) / 100 for day in history["dates"]]
    nav = list(map(decimal, history["adjusted_nav"]))
    if any(v <= 0 for v in publisher):
        raise ValueError("RATING_INDEX_PUBLISHER_FUND_NONPOSITIVE")
    if history_reconciler is not None:
        # 独立的新来源政策才允许进入此分支；原 V1 仍严格拒绝来源冲突。
        history = history_reconciler(history, dates, provider, refs, root, code, as_of)
        nav = list(map(decimal, history["adjusted_nav"]))
    # 发行人累计曲线显示百分比四位小数，单位净值通常显示四位；仅允许末位舍入解释的差异。
    tolerance = Decimal(".0001") / min(nav) + Decimal(".000001") / min(publisher)
    discrepancy = max(abs(v / nav[0] - p / publisher[0]) for v, p in zip(nav, publisher, strict=True))
    if discrepancy > tolerance:
        raise ValueError("RATING_INDEX_PROVIDER_PUBLISHER_NAV_CONFLICT")
    history["publisher_cross_check"] = {
        "maximum_normalized_difference": str(discrepancy),
        "rounding_bound": str(tolerance),
    }
    text = re.sub(r"\s", "", "\n".join([*summary, *prospectus]))
    error = re.search(r"年(?:化)?跟踪误差(?:不超过|控制在|控制)([\d.]+)%", text)
    bias = re.search(r"日均跟踪偏离度(?:的)?绝对值(?:不超过|控制在)([\d.]+)%", text)
    if not error or not bias:
        raise ValueError("RATING_INDEX_TRACKING_TARGETS_MISSING")
    targets = {"tracking_error": str(decimal(error[1]) / 100), "tracking_bias": str(decimal(bias[1]) / 100)}
    if any(decimal(v) <= 0 for v in targets.values()):
        raise ValueError("RATING_INDEX_TRACKING_TARGETS_NONPOSITIVE")
    published = publication(report)
    if published > str(as_of):
        raise ValueError("RATING_INDEX_FUTURE_REPORT")
    return {
        "key_information": info,
        "holdings": {
            "report_date": "2026-06-30",
            "published_on": published,
            "positions": tables["holdings"],
            "industries": tables["reported_industries"],
            "balance": b,
            "net_assets": b["net_assets"],
        },
        "disclosure": d,
        "history": history,
        "tracking_targets": targets,
    }


def metrics(bundle: dict, as_of: date, frozen: datetime) -> tuple[dict, dict, datetime]:
    from app.services.fund_rating_public_inputs import public_valid_until

    if bundle["basis"] != BASIS or bundle["family"] != "INDEX" or bundle["currency"] != "CNY":
        raise ValueError("RATING_INDEX_SCOPE")
    result, h = provider_history({"history": bundle["history"]}, as_of)
    positions, industries = bundle["holdings"]["positions"], bundle["holdings"]["industries"]
    stock = sum(decimal(p["market_value_cny"]) for p in positions)
    if abs(stock - decimal(bundle["holdings"]["balance"]["stock_amount"])) > Decimal(".02") or abs(
        sum(decimal(p["value_cny"]) for p in industries) - stock
    ) > Decimal(".02"):
        raise ValueError("RATING_INDEX_STOCK_TOTAL")
    result["issuer_hhi"] = sum((decimal(p["market_value_cny"]) / stock) ** 2 for p in positions)
    result["industry_hhi"] = sum((decimal(p["value_cny"]) / stock) ** 2 for p in industries)
    nav = list(map(decimal, h["adjusted_nav"]))
    benchmark = list(map(decimal, bundle["history"]["benchmark_index"]))
    if len(benchmark) != len(nav):
        raise ValueError("RATING_INDEX_TRACKING_DATES")
    differences = [nav[i] / nav[i - 1] - benchmark[i] / benchmark[i - 1] for i in range(1, len(nav))]
    mean = sum(differences) / len(differences)
    result["tracking_error"] = (252 * sum((v - mean) ** 2 for v in differences) / (len(differences) - 1)).sqrt()
    result["tracking_bias"] = sum(abs(v) for v in differences) / len(differences)
    for metric in ("tracking_error", "tracking_bias"):
        result[metric + "_ratio"] = result[metric] / decimal(bundle["tracking_targets"][metric])
    net = decimal(bundle["holdings"]["net_assets"])
    result.update(
        net_assets=net,
        cash_ratio=decimal(bundle["holdings"]["balance"]["assets"]["货币资金"]) / net,
        restricted_ratio=decimal(bundle["disclosure"]["restricted_stock_amount"]) / net,
    )
    result.update({k: decimal(bundle["disclosure"][k]) for k in ("strategy_continuity", "governance_state")})
    result["cost_1y"] = decimal(bundle["key_information"]["cost_1y"])
    checked = datetime.fromisoformat(bundle["documents_checked_at"])
    report = date.fromisoformat(bundle["holdings"]["report_date"])
    return (
        result,
        {
            "nav": str(as_of),
            "holdings": str(report),
            "fees": checked.date().isoformat(),
            "manager": checked.date().isoformat(),
        },
        public_valid_until(as_of, report, checked, frozen),
    )


def verify_originals(bundle: dict) -> None:
    original = facts(
        {r["label"]: r for r in bundle["raw_archive"]},
        Path(get_settings().fund_insight_directory) / "ratings",
        bundle["fund_code"],
        bundle["fund_name_stem"],
        date.fromisoformat(bundle["as_of_date"]),
    )
    if any(original[k] != bundle[k] for k in original):
        raise ValueError("RATING_INDEX_ORIGINAL_FACT_CONFLICT")


def independent(bundle: dict, expected: dict) -> dict:
    # 直接复算已包含再投资的复权净值；这里不创建或填补任何分红、拆分记录。
    h = bundle["history"]
    rows = {r["nav_date"]: r for r in h["rows"]}
    nav = [float(rows[d]["adjusted_nav"]) for d in h["dates"]]
    bm = list(map(float, h["benchmark_index"]))
    differences = [nav[i] / nav[i - 1] - bm[i] / bm[i - 1] for i in range(1, len(nav))]
    average = sum(differences) / len(differences)
    actual = {
        "tracking_error": math.sqrt(252 * sum((v - average) ** 2 for v in differences) / (len(differences) - 1)),
        "tracking_bias": sum(abs(v) for v in differences) / len(differences),
    }
    for name in ("tracking_error", "tracking_bias"):
        actual[name + "_ratio"] = actual[name] / float(bundle["tracking_targets"][name])
    monthly = [nav[h["dates"].index(d)] for d in h["month_ends"]]
    returns = [b / a - 1 for a, b in zip(nav, nav[1:], strict=False)]
    rolling = [monthly[i] / monthly[i - 12] - 1 for i in range(12, len(monthly))]
    peak, drawdown = nav[0], 0.0
    for value in nav:
        peak = max(peak, value)
        drawdown = max(drawdown, 1 - value / peak)
    start, end = map(date.fromisoformat, (h["dates"][0], h["dates"][-1]))
    actual.update(
        return_1y=monthly[-1] / monthly[-13] - 1,
        return_3y=(nav[-1] / nav[0]) ** (365.25 / (end - start).days) - 1,
        drawdown=drawdown,
        downside=math.sqrt(252 * sum(min(r, 0) ** 2 for r in returns) / len(returns)),
        rolling_positive=sum(r > 0 for r in rolling) / len(rolling),
        monthly_positive=sum(b > a for a, b in zip(monthly, monthly[1:], strict=False)) / (len(monthly) - 1),
    )
    actual["calmar"] = (
        actual["return_3y"] / drawdown
        if drawdown
        else 2
        if actual["return_3y"] > 0
        else 0
        if actual["return_3y"] == 0
        else -1
    )
    stock = sum(float(p["market_value_cny"]) for p in bundle["holdings"]["positions"])
    actual.update(
        issuer_hhi=sum((float(p["market_value_cny"]) / stock) ** 2 for p in bundle["holdings"]["positions"]),
        industry_hhi=sum((float(p["value_cny"]) / stock) ** 2 for p in bundle["holdings"]["industries"]),
    )
    net = float(bundle["holdings"]["net_assets"])
    actual.update(
        net_assets=net,
        cash_ratio=float(bundle["holdings"]["balance"]["assets"]["货币资金"]) / net,
        restricted_ratio=sum(float(r["amount"]) for r in bundle["disclosure"]["restricted_rows"]) / net,
        cost_1y=float(bundle["key_information"]["cost_1y"]),
        **{k: float(bundle["disclosure"][k]) for k in ("strategy_continuity", "governance_state")},
    )
    if any(not math.isclose(v, float(expected[k]), rel_tol=1e-12, abs_tol=1e-12) for k, v in actual.items()):
        raise ValueError("RATING_INDEX_INDEPENDENT_MISMATCH")
    return {
        "metrics": actual,
        "rolling_one_year": [{"end": d, "return": r} for d, r in zip(h["month_ends"][12:], rolling, strict=True)],
        "note": "历史公开资料复算，不回填历史综合评级",
    }
