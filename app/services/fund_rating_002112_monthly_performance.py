"""002112滚动表现原件编译；完整347收益父池、实际风险池，不删除稀疏净值同行。"""

from datetime import date
from decimal import Decimal as D, localcontext
from statistics import median

from app.services.fund_rating_thematic_source import RawArchive, iso, require
from app.services.fund_rating_monthly_r5 import rolling_window

VERSION = "R5_002112_MONTHLY_PERFORMANCE_ORIGINALS_V1"


def compile_observations(inputs, *, root):
    from app.services.fund_rating_active_monthly_admission import replay_monthly_admission
    from app.services.fund_rating_active_performance import _product, _growth, TIME_CONTRACT

    require(inputs.get("version") == VERSION and inputs.get("fund_code") == "002112", "002112_PERF_SCOPE")
    source = RawArchive(inputs, root)
    graph = source.json(inputs["monthly_admission_ref"])
    require(graph["as_of_date"] == inputs["as_of_date"], "002112_PERF_ADMISSION_PERIOD")
    admission = replay_monthly_admission(graph, root=root)
    parent = sorted(row["code"] for row in admission["rows"])
    risk = admission["risk_codes"]
    require(len(parent) == 347 and set(inputs["funds"]) == set(parent) | {"002112"}, "002112_PERF_FULL_PARENT")
    days, months = rolling_window([iso(r["cal_date"]) for r in source.provider_rows(
        inputs["calendar_ref"], ["exchange", "cal_date", "is_open"]) if r["exchange"] == "SSE" and r["is_open"] == 1],
        inputs["as_of_date"])
    days8, months8 = [d.replace("-", "") for d in days], [d.replace("-", "") for d in months]
    profile = dict(inputs, time_contract=TIME_CONTRACT)
    raw = {}
    for code, spec in inputs["funds"].items():
        require(spec["fund_code"] == code and spec["ts_code"].split(".")[0] == code,
                "002112_PERF_OWN_SHARE")
        for digest in spec["nav_refs"] + spec["dividend_refs"]:
            ref = source.original(digest)
            params = ref.get("params_without_credentials") or ref.get("params") or {}
            requested = str(params.get("ts_code", ""))
            require(requested == spec["ts_code"] if digest in spec["nav_refs"] else requested.split(".")[0] == code,
                    "002112_PERF_REQUEST_IDENTITY")
        raw[code] = _product(profile, root, spec)
    missing, observations = {}, {}
    common = ["002112", *risk]

    def absent(codes, required):
        return [dict(code=c, dates=[iso(d) for d in sorted(set(required) - set(raw[c][0]))])
                for c in codes if not set(required) <= set(raw[c][0])]

    missing["return"] = absent(["002112", *parent], [days8[0], days8[-1], months8[-13]])
    missing["risk"] = absent(common, days8)
    missing["stability"] = absent(common, months8)
    ids = ["CN_PUBLIC_FUND:" + c for c in parent]
    risk_ids = ["CN_PUBLIC_FUND:" + c for c in risk]
    with localcontext() as context:
        context.prec = 70
        if not missing["return"]:
            values = {}
            years = D("365.25") / (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days
            try:
                for code in ["002112", *parent]:
                    values[code] = [_growth(*raw[code], days8, months8[-13], days8[-1]) - 1,
                                    _growth(*raw[code], days8, days8[0], days8[-1]) ** years - 1]
                for n, key in enumerate(["return_1y", "return_3y"]):
                    observations[key] = dict(kind="rank", target=str(values["002112"][n]), direction="higher",
                        peers=[dict(product_id="CN_PUBLIC_FUND:" + c, value=str(values[c][n])) for c in parent], expected_products=ids)
            except ValueError as error:
                missing["return"] = [dict(code=code, error=str(error))]
        if not missing["risk"]:
            try:
                daily = {c: [_growth(*raw[c], days8, a, b) - 1 for a, b in zip(days8, days8[1:])] for c in common}
                sums = [sum(daily[c][i] for c in risk) for i in range(len(days8) - 1)]
                metrics = {}
                for c in common:
                    included = c in risk
                    baseline = [(value - (daily[c][n] if included else 0)) / (len(risk) - int(included))
                                for n, value in enumerate(sums)]
                    wealth = benchmark = peak = D(1)
                    drawdown = square = D(0)
                    for actual, reference in zip(daily[c], baseline, strict=True):
                        wealth *= 1 + actual
                        benchmark *= 1 + reference
                        relative = wealth / benchmark
                        peak = max(peak, relative)
                        drawdown = max(drawdown, 1 - relative / peak)
                        square += min(actual - reference, D(0)) ** 2
                    metrics[c] = [drawdown, (D(252) / len(baseline) * square).sqrt()]
                for n, key in enumerate(["extra_drawdown", "extra_downside"]):
                    observations[key] = dict(kind="rank", target=str(metrics["002112"][n]), direction="lower",
                        peers=[dict(product_id="CN_PUBLIC_FUND:" + c, value=str(metrics[c][n])) for c in risk], expected_products=risk_ids)
            except ValueError as error:
                missing["risk"] = [dict(error=str(error))]
        if not missing["stability"]:
            try:
                for horizon, key, count in [(1, "monthly_success", 36), (12, "annual_success", 25)]:
                    rows = []
                    for a, b in zip(months8, months8[horizon:]):
                        returns = {c: _growth(*raw[c], days8, a, b) - 1 for c in common}
                        rows.append(dict(start=a, end=b, target=str(returns["002112"]), reference=str(median(returns[c] for c in risk))))
                    require(len(rows) == count, "002112_PERF_WINDOW_COUNT")
                    observations[key] = dict(kind="windows", expected_windows=count, rows=rows)
            except ValueError as error:
                missing["stability"] = [dict(error=str(error))]
    # 一个维度的任一必要子项未知，就不留下半个维度供下游误当完整。
    fields = {"return": ["return_1y", "return_3y"], "risk": ["extra_drawdown", "extra_downside"],
              "stability": ["monthly_success", "annual_success"]}
    for dimension, rows in missing.items():
        if rows:
            for key in fields[dimension]:
                observations.pop(key, None)
    return dict(version=VERSION, observations=observations, missing=missing,
                ready_dimensions=[key for key, value in missing.items() if not value],
                parent_count=len(parent), risk_count=len(risk), dates=days, month_ends=months,
                missing_peers_removed=False, imputed=False)
