"""018125风险/稳定性独立研究重放；重大事项None绝不转成可发布确定分。

只依赖正式app及标准库。共同日历、单位净值、已实施分红、请求身份和同策略准入
逐件追到原字节；研究算术输出与正式维度/总分分开，当前版本不提供发布能力。
"""

import json
from copy import deepcopy
from datetime import date
from decimal import Decimal as D
from decimal import localcontext
from functools import lru_cache
from pathlib import Path
from statistics import median

from app.services.fund_rating_active_performance import (
    TIME_CONTRACT,
    _calendar,
    _growth,
    _product,
    _require_time_contract,
)
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_r5 import exposure_factor, peer_gap_score, window_frequency_score
from app.services.fund_rating_yy_manufacturing_liquidity import require
from app.services.fund_rating_yy_manufacturing_market_liquidity import PEERS, PERIODS, _admission
from app.services.material_content import content_digest, source_bytes

VERSION = "R5_018125_RISK_STABILITY_RESEARCH_ORIGINALS_20261030_1"


def _source(inputs, root):
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "018125"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("research_on") == "2026-10-30"
        and inputs.get("time_contract") == TIME_CONTRACT,
        "YYRS_RESEARCH_SCOPE",
    )
    require(
        inputs.get("material_event_pending") is None and inputs.get("performance_release_requested") is False,
        "YYRS_NO_RELEASE_OR_MATERIAL_DEFAULT",
    )
    require(
        not any(k in inputs for k in ["score", "dimensions", "metrics", "observations", "packet"]),
        "YYRS_ORIGINALS_NOT_PRECOMPUTED",
    )
    c = inputs["admission"]
    products = inputs["products"]
    require(
        tuple(c["peer_codes"]) == PEERS
        and set(c["exposure_refs"]) == {"018125", *PEERS}
        and set(products) == {"018125", *PEERS},
        "YYRS_FIXED_COHORT",
    )
    require(
        len(c["admission_reviews"]) == 40
        and {(r["code"], r["period"]) for r in c["admission_reviews"]}
        == {(code, period) for code in PEERS for period in PERIODS},
        "YYRS_FOUR_LEGAL_CHECKPOINTS",
    )
    financial = {r["source_hash"] for r in c["admission_reviews"]}
    exposure = {h for rows in c["exposure_refs"].values() for h in rows}
    derived = {
        inputs["calendar_ref"],
        inputs["time_contract_ref"],
        inputs["discovery_ref"],
        *(r["dividend_receipt_ref"] for r in products.values()),
    }
    originals = {
        *financial,
        *exposure,
        *(h for p in products.values() for k in ["nav_refs", "dividend_refs"] for h in p[k]),
    }
    # 日历索引只含原始trade_cal的引用；先核每个源摘要，再由正式_calendar逐行对账。
    calendar_path = (Path(root) / inputs["sources"][inputs["calendar_ref"]]["file"]).resolve()
    require(calendar_path.is_relative_to(Path(root).resolve() / "sources"), "YYRS_CALENDAR_SOURCE_SCOPE")

    calendar_raw = source_bytes(calendar_path)
    require(content_digest(calendar_raw) == inputs["calendar_ref"], "YYRS_CALENDAR_INDEX_HASH")
    calendar = json.loads(calendar_raw)
    originals.update(s["sha256"] for s in calendar["sources"])
    require(set(inputs["sources"]) == originals | derived and not originals & derived, "YYRS_SOURCE_GRAPH_COMPLETE")
    for h, ref in inputs["sources"].items():
        require(
            ref.get("sha256") == h and ref.get("archive_kind") == ("DERIVED" if h in derived else "ORIGINAL"),
            "YYRS_SOURCE_ROLE",
        )
        if h in financial:
            require(
                isinstance(ref.get("published_on"), str)
                and date.fromisoformat(ref["published_on"]) <= date(2026, 9, 30),
                "YYRS_FINANCIAL_PUBLICATION",
            )
    source = Originals(inputs, Path(root).resolve())
    for code, p in products.items():
        require(
            p.get("fund_code") == code
            and p.get("ts_code") == code + ".OF"
            and p.get("format") == "TUSHARE_NAV_AND_DIVIDEND_ORIGINALS"
            and p["nav_refs"]
            and p["dividend_refs"],
            "YYRS_PRODUCT_IDENTITY",
        )
        for key, api in [("nav_refs", "fund_nav"), ("dividend_refs", "fund_div")]:
            for h in p[key]:
                request = inputs["sources"][h]["request_binding"]
                require(
                    request["api_name"] == api
                    and request["status"] == 200
                    and request["params"]["ts_code"] == code + ".OF",
                    "YYRS_ACTUAL_REQUEST_IDENTITY",
                )
                if api == "fund_nav":
                    require(
                        request["params"]["start_date"] == "20230928" and request["params"]["end_date"] == "20260930",
                        "YYRS_ACTUAL_NAV_WINDOW",
                    )
        receipt = source.json(p["dividend_receipt_ref"])
        require(
            receipt["code"] == code and {s["sha256"] for s in receipt["sources"]} == set(p["dividend_refs"]),
            "YYRS_DIVIDEND_REQUEST_BINDING",
        )
        for s in receipt["sources"]:
            request = inputs["sources"][s["sha256"]]["request_binding"]
            require(
                all(s[k] == request[k] for k in ["api_name", "params", "status"]), "YYRS_DIVIDEND_REQUEST_PROVENANCE"
            )
    return source


def _series(inputs, root):
    """使用公开估值日与真实分红，727个共同日缺任一日即拒算，不用调整净值捷径。"""
    days = _calendar(inputs, Path(root))
    months = []
    for day in days:
        if not months or day[:6] != months[-1][:6]:
            months.append(day)
        else:
            months[-1] = day
    require(len(months) == 37, "YYRS_MONTH_END_COUNT")
    data = {}
    audits = {}
    for code, p in inputs["products"].items():
        audit = {}
        points, events = _product(inputs, Path(root), p, audit=audit)
        require(not set(days) - points.keys(), "YYRS_MISSING_COMMON_NAV:" + code)
        data[code] = (points, events)
        audits[code] = audit
    return days, months, data, audits


def _compile(inputs, source, root):
    frozen = source.json(inputs["discovery_ref"])
    require(
        frozen["risk_stability_admission"]
        == {"mean_stock_gap_pp": "5", "max_stock_gap_pp": "10", "min_domestic_fraction": "0.9"},
        "YYRS_FROZEN_CRITERIA",
    )
    time = _require_time_contract(inputs, Path(root))
    admission = _admission(source, inputs["admission"])
    days, months, data, audits = _series(inputs, root)
    with localcontext() as ctx:
        ctx.prec = 60
        # 相邻交易日序列长度本来相差一，保留原来的逐相邻区间配对方式。
        daily = {
            c: [_growth(*v, days, a, b) - 1 for a, b in zip(days, days[1:], strict=False)] for c, v in data.items()
        }
        sums = [sum(daily[c][i] for c in PEERS) for i in range(len(days) - 1)]
        metrics = {}
        for code in ["018125", *PEERS]:
            own = benchmark = peak = D(1)
            drawdown = square = D(0)
            included = code in PEERS
            count = len(PEERS) - int(included)
            for i, r in enumerate(daily[code]):
                reference = (sums[i] - (r if included else 0)) / count
                own *= 1 + r
                benchmark *= 1 + reference
                relative = own / benchmark
                peak = max(peak, relative)
                drawdown = max(drawdown, 1 - relative / peak)
                square += min(r - reference, D(0)) ** 2
            endpoint = _growth(*data[code], days, days[0], days[-1])
            require(abs(own - endpoint) < D("1e-50"), "YYRS_DIVIDEND_WEALTH_ENDPOINT")
            metrics[code] = {
                "extra_drawdown": drawdown,
                "extra_downside": (D(252) / D(len(days) - 1) * square).sqrt(),
                "noncalendar_NAV_dates_excluded": sorted(set(data[code][0]) - set(days)),
            }
        parts = {}
        for name in ["extra_drawdown", "extra_downside"]:
            x = metrics["018125"][name]
            parts[name] = peer_gap_score(x, [metrics[c][name] for c in PEERS]) * exposure_factor(name, x)
        risk = D(".7") * parts["extra_drawdown"] + D(".3") * parts["extra_downside"]
        windows = {}
        means = []
        for horizon, name, expected in [(1, "monthly_success", 36), (12, "annual_success", 25)]:
            rows = []
            unit = D(".1") / (12 if horizon == 1 else 1)
            for a, b in zip(months, months[horizon:], strict=False):
                returns = {c: _growth(*data[c], days, a, b) - 1 for c in ["018125", *PEERS]}
                target = returns["018125"]
                reference = median(returns[c] for c in PEERS)
                point = window_frequency_score(target - reference, unit)
                rows.append({"start": a, "end": b, "target": target, "reference": reference, "point": point})
            require(len(rows) == expected, "YYRS_COMPLETE_WINDOWS")
            windows[name] = rows
            means.append(sum(r["point"] for r in rows) / expected)
        stability = D(".3") * means[0] + D(".7") * means[1]

    def strings(v):
        if isinstance(v, D):
            return str(v)
        if isinstance(v, dict):
            return {k: strings(x) for k, x in v.items()}
        if isinstance(v, list):
            return [strings(x) for x in v]
        return v

    return strings(
        {
            "research_scores": {"risk": risk, "stability": stability},
            "dimensions": {},
            "complete_score": None,
            "grade": None,
            "performance_release_permitted": False,
            "material_event_pending": None,
            "metrics_by_share": metrics,
            "windows": windows,
            "audit": {
                "calendar_points": len(days),
                "daily_returns": len(days) - 1,
                "month_ends": len(months),
                "peer_admission": admission,
                "time_contract": time,
                "date_audit": audits,
                "daily_benchmark": "10只预先冻结同行每日等权；评价同行自身时剔除自身",
                "contract_benchmark_substituted": False,
                "missing_NAV_interpolated": False,
            },
        }
    )


@lru_cache(maxsize=1)
def _cached(serialized, root):
    inputs = json.loads(serialized)
    return _compile(inputs, _source(inputs, root), root)


def calculate(inputs, *, root):
    """重核字节后返回研究算术，不把未知重大事项转换为确定维度或发布许可。"""
    _source(inputs, root)
    return deepcopy(_cached(canonical(inputs), str(Path(root).resolve())))


def prepare(*args, **kwargs):
    """当前研究版本没有正式prepare出口；不得靠翻转None/False绕过实际事项审阅。"""
    raise ValueError("YYRS_RESEARCH_ONLY_MATERIAL_REVIEW_NOT_CLOSED")
