"""有界本地队列与已登记指数只读采集；只允许2021—2024研究正文。"""

import math
import re
from collections import Counter
from dataclasses import asdict
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal, localcontext

from sqlalchemy import text

from app.schemas.direction_followup import StudyAnswer, StudyCohort, StudyInput
from app.services.direction_training_artifacts import digest
from app.services.direction_training_dataset import END, FUNDS, START, json_ready
from app.services.direction_training_inventory import local_snapshot
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.trading_calendar import load_calendar


def metadata():
    queries = {
        "sources": "select source_code,enabled,authorized_api_names,rate_limit_per_minute,"
        "authorization_verified_at is not null as authorization_recorded "
        "from source_registry order by source_code limit 30",
        "catalog": "select f.fund_code,f.fund_master_id,f.fund_type,f.status,f.source_code,m.fund_name,"
        "m.established_date,p.source_fund_type,p.market,p.benchmark from fund_share_class f "
        "join fund_master m using(fund_master_id) left join fund_profile p using(fund_code) "
        "order by f.fund_code limit 1001",
        "coverage": "select fund_code,count(*) as n,min(nav_date) as first_date,max(nav_date) as last_date "
        "from nav_daily where nav_date between DATE '2021-01-01' and DATE '2024-12-31' "
        "group by fund_code order by fund_code limit 1001",
        "market_index": "select index_code,display_name,market,publisher,category from market_index_catalog "
        "where index_code='000300.SH' limit 2",
        "benchmark_count": "select count(*) as n from benchmark_series",
    }
    with local_snapshot() as (session, statements):
        result = {
            name: json_ready([dict(r) for r in session.execute(text(q)).mappings()]) for name, q in queries.items()
        }
    if len(result["catalog"]) > 1000 or len(result["coverage"]) > 1000:
        raise ValueError("CATALOG_BUDGET")
    result["sql_statement_counts"] = dict(statements)
    return result


def normalize_product(name):
    return re.sub(r"[\s\-（）()·]", "", re.sub(r"[-－]?[ABC](?:类)?$", "", name.upper()))


def select_cohort(meta, protocol):
    rule = protocol["cohort_rule"]
    coverage = {r["fund_code"]: r for r in meta["coverage"]}
    sources = {r["source_code"]: r for r in meta["sources"]}
    rows = sorted(meta["catalog"], key=lambda r: (r["fund_code"] not in FUNDS, r["fund_code"]))
    selected, decisions, masters, products = [], [], set(), set()
    for row in rows:
        f = row["fund_code"]
        reasons = []
        for field, expected in (
            ("fund_type", "STOCK"),
            ("source_fund_type", "股票型"),
            ("status", "ACTIVE"),
            ("market", "O"),
        ):
            if row[field] != expected:
                reasons.append(f"UNSUPPORTED_{field.upper()}")
        source = sources.get(row["source_code"], {})
        if (
            not source.get("enabled")
            or not source.get("authorization_recorded")
            or not {"fund_nav", "fund_div"}.issubset(source.get("authorized_api_names", []))
        ):
            reasons.append("SOURCE_NOT_AUTHORIZED")
        if any(
            term.casefold() in (row["fund_name"] + " " + (row["benchmark"] or "")).casefold()
            for term in rule["exclude_foreign_terms"]
        ):
            reasons.append("DIFFERENT_MARKET_CALENDAR")
        if not row["established_date"] or row["established_date"] > rule["established_on_or_before"]:
            reasons.append("LATE_OR_UNKNOWN_INCEPTION")
        cov = coverage.get(f, {})
        if not cov or cov["first_date"] > rule["history_first_by"] or cov["last_date"] < rule["history_end_at"]:
            reasons.append("HISTORY_SPAN_INSUFFICIENT")
        product = normalize_product(row["fund_name"])
        if row["fund_master_id"] in masters or product in products:
            reasons.append("DUPLICATE_SHARE_OR_PRODUCT")
        if not reasons and len(selected) >= rule["maximum_funds"]:
            reasons.append("PREDECLARED_COHORT_BUDGET")
        if f in FUNDS and reasons:
            raise ValueError("ORIGINAL_FUND_METADATA_INVALID")
        if not reasons:
            selected.append(f)
            masters.add(row["fund_master_id"])
            products.add(product)
        decisions.append(
            {
                **row,
                "coverage": cov,
                "status": "METADATA_ELIGIBLE" if not reasons else "EXCLUDED",
                "reasons": reasons,
                "correlation_group": row["benchmark"],
                "historical_classification_verified": False,
            }
        )
    return StudyCohort(funds=tuple(selected)), decisions


def load_funds(cohort):
    from app.repositories.cash_reinvestment_samples import read_cash_dividends, read_cash_nav_window
    from app.repositories.historical_nav import read_historical_nav_source

    output = {}
    with local_snapshot() as (session, statements):
        for fund in cohort.funds:
            source = read_historical_nav_source(session, fund_code=fund)
            nav, events = [], {}
            cursor = START
            while cursor <= END:
                end = min(cursor + timedelta(days=89), END)
                nav.extend(
                    read_cash_nav_window(session, fund_code=fund, source_id=source.source_id, start=cursor, end=end)
                )
                for event in read_cash_dividends(
                    session, fund_code=fund, source_id=source.source_id, start=cursor, end=end
                ):
                    if event.event_key in events and events[event.event_key] != event:
                        raise ValueError("SOURCE_EVENT_CONFLICT")
                    events[event.event_key] = event
                cursor = end + timedelta(days=1)
            if len(nav) > 1461 or len(events) > 100 or len({p.nav_date for p in nav}) != len(nav):
                raise ValueError("SOURCE_BUDGET_OR_DUPLICATE")
            output[fund] = json_ready(
                {
                    "source": asdict(source),
                    "nav": [asdict(p) for p in nav],
                    "events": [asdict(events[k]) for k in sorted(events)],
                }
            )
    return output, dict(statements)


def restore_fund(raw):
    from datetime import datetime
    from uuid import UUID

    from app.repositories.cash_reinvestment_samples import CashDividend, CashNavPoint
    from app.repositories.feature_snapshot import FeatureSourceReadiness

    meta = raw["source"]
    source = FeatureSourceReadiness(
        UUID(meta["source_id"]),
        meta["source_code"],
        UUID(meta["source_sync_run_id"]),
        datetime.fromisoformat(meta["source_sync_finished_at"]),
    )
    if source.source_code != "TUSHARE_PRO_FUND" or len(raw["nav"]) > 1461 or len(raw["events"]) > 100:
        raise ValueError("SOURCE_SCOPE")
    nav = tuple(
        CashNavPoint(
            date.fromisoformat(p["nav_date"]),
            date.fromisoformat(p["ann_date"]) if p["ann_date"] else None,
            Decimal(p["unit_nav"]) if p["unit_nav"] is not None else None,
        )
        for p in raw["nav"]
    )
    if any(not START <= p.nav_date <= END for p in nav) or any(
        a.nav_date >= b.nav_date for a, b in zip(nav[:-1], nav[1:], strict=True)
    ):
        raise ValueError("SOURCE_DATE_OR_ORDER")
    events = []
    for e in raw["events"]:
        record = dict(e)
        for field in ("ann_date", "implementation_ann_date", "ex_date", "nav_ex_date"):
            record[field] = date.fromisoformat(e[field]) if e[field] else None
            if record[field] and record[field] > END:
                raise ValueError("SOURCE_EVENT_PROTECTED")
        record["cash_dividend"] = Decimal(e["cash_dividend"]) if e["cash_dividend"] is not None else None
        events.append(CashDividend(**record))
    return source, nav, tuple(events)


def historical_input(fund, cutoff, source, nav, events, cohort):
    from app.repositories.trading_nav_window import NavDatePoint
    from app.services.cash_reinvestment_samples import build_cash_history_feature
    from app.services.trading_nav_window import build_history_date_window

    if fund not in cohort.funds or not START <= cutoff <= END:
        raise ValueError("INPUT_COHORT_OR_DATE")
    calendar = load_calendar()
    history = build_history_date_window(cutoff, tuple(NavDatePoint(p.nav_date, p.ann_date) for p in nav), calendar)
    issues = ([history.anchor_issue] if history.anchor_issue else []) + [i.reason for i in history.issues]
    if issues:
        return None, sorted(set(issues))
    feature, problems = build_cash_history_feature(
        fund_code=fund,
        cutoff_date=cutoff,
        history_dates=history.dates,
        source=source,
        nav={p.nav_date: p for p in nav},
        events=events,
        calendar=calendar,
    )
    if feature is None:
        return None, sorted({p.code for p in problems})
    return StudyInput(
        fund=fund,
        cutoff=cutoff,
        anchor=feature.anchor_nav_date,
        available_at=feature.available_at,
        anchor_lag_sessions=calendar.at_or_before_index(cutoff) - calendar.at_or_before_index(feature.anchor_nav_date),
        x=tuple(float(feature.metrics[k]) for k in FEATURE_NAMES),
        input_hash=digest(feature.model_dump(mode="json")),
    ), []


def answer(fund, cutoff, nav, events, cohort, *, value):
    from app.services.cash_reinvestment_samples import build_cash_return_series

    if fund not in cohort.funds or not START <= cutoff <= END:
        raise ValueError("ANSWER_COHORT_OR_DATE")
    calendar = load_calendar()
    future = calendar.future_sessions(cutoff)
    if future[-1] > END:
        return None, ["PROTECTED_2025_LABEL"]
    if any(y.announced_on > cutoff for y in calendar.definition.years if y.year in {d.year for d in future}):
        return None, ["CALENDAR_NOT_KNOWN"]
    base = calendar.sessions[calendar.at_or_before_index(cutoff)]
    points, problems = build_cash_return_series(
        (base, *future), {p.nav_date: p for p in nav}, events, latest_available=END
    )
    if problems:
        return None, sorted({p.code for p in problems})
    meta = {
        "fund": fund,
        "cutoff": str(cutoff),
        "base": str(base),
        "end": str(future[-1]),
        "available_at": str(points[-1].available_at),
    }
    if not value:
        return meta, []
    with localcontext() as context:
        context.prec = 40
        ret = (Decimal(points[-1].growth_index) / 100 - 1).quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP)
    return StudyAnswer(**meta, y=int(ret > 0), future_return=str(ret)).model_dump(mode="json"), []


def feature_dataset(sources, cohort):
    dates = [d for d in load_calendar().sessions if START <= d <= END]
    if len(dates) * len(cohort.funds) > 10000:
        raise ValueError("DATASET_ROW_BUDGET")
    inputs, availability = {}, {}
    for f in cohort.funds:
        source, nav, events = restore_fund(sources[f])
        inputs[f], availability[f] = [], []
        for cutoff in dates:
            item, issues = historical_input(f, cutoff, source, nav, events, cohort)
            label, problems = (
                answer(f, cutoff, nav, events, cohort, value=False) if item else (None, ["INPUT_UNAVAILABLE"])
            )
            if item:
                inputs[f].append(item.model_dump(mode="json"))
            availability[f].append(
                {"fund": f, "cutoff": str(cutoff), "input_issues": issues, "label": label, "label_issues": problems}
            )
    return inputs, availability


def market_prices(meta, protocol):
    from app.core.config import get_settings
    from app.integrations.tushare_market_reference import TushareMarketReferenceClient

    config = protocol["market"]
    source = next((s for s in meta["sources"] if s["source_code"] == config["source"]), {})
    if (
        not source.get("enabled")
        or not source.get("authorization_recorded")
        or config["api"] not in source.get("authorized_api_names", [])
    ):
        return {"status": "NOT_RUN", "reason": "SOURCE_NOT_AUTHORIZED", "prices": []}
    if len(meta["market_index"]) != 1 or meta["market_index"][0]["display_name"] != config["expected_name"]:
        return {"status": "NOT_RUN", "reason": "REFERENCE_MAPPING_UNVERIFIED", "prices": []}
    settings = get_settings()
    prices, requests = [], []
    with TushareMarketReferenceClient(
        token=settings.tushare_token.get_secret_value(),
        api_url=settings.tushare_api_url,
        connect_timeout_seconds=5,
        read_timeout_seconds=20,
        max_retries=0,
        catalog_max_rows_per_query=8000,
        max_rows_per_query=367,
    ) as client:
        for year in config["years"]:
            if year not in (2021, 2022, 2023, 2024):
                raise ValueError("PROTECTED_MARKET_YEAR")
            try:
                records = client.list_index_daily(
                    config["index_code"], start_date=date(year, 1, 1), end_date=date(year, 12, 31)
                )
            except Exception as error:
                return {
                    "status": "NOT_RUN",
                    "reason": "SOURCE_REQUEST_FAILED",
                    "error_type": type(error).__name__,
                    "requests": requests,
                    "prices": [],
                }
            requests.append({"api": "index_daily", "code": config["index_code"], "year": year, "rows": len(records)})
            if len(records) > 366 or any(
                p.trade_date.year != year or p.close_price <= 0 or not p.close_price.is_finite() for p in records
            ):
                raise ValueError("MARKET_RESPONSE_SCOPE")
            prices.extend({"date": str(p.trade_date), "close": str(p.close_price)} for p in records)
            # 小固定请求数，最低按来源登记频次节流。
            if year != config["years"][-1]:
                import time

                time.sleep(max(0.5, 60 / source["rate_limit_per_minute"]))
    if len({p["date"] for p in prices}) != len(prices):
        raise ValueError("MARKET_DUPLICATE_DATE")
    return {
        "status": "DOWNLOADED_RESEARCH_SNAPSHOT",
        "prices": sorted(prices, key=lambda p: p["date"]),
        "requests": requests,
        "assumption": config["availability_assumption"],
        "historical_timestamp_verified": False,
    }


def add_market(item, prices):
    calendar = load_calendar()
    index = calendar.at_or_before_index(item.cutoff) - 1
    if index < 20:
        return None, "MARKET_PREWARM_SHORT"
    if item.anchor != calendar.sessions[index] or item.anchor_lag_sessions != 1:
        return None, "MARKET_FUND_ANCHOR_MISMATCH"
    dates = calendar.sessions[index - 20 : index + 1]
    if any(str(d) not in prices for d in dates):
        return None, "MARKET_MISSING_SESSION"
    values = [float(prices[str(d)]) for d in dates]
    if any(not math.isfinite(p) or p <= 0 for p in values):
        return None, "MARKET_INVALID_CLOSE"
    returns = [b / a - 1 for a, b in zip(values[:-1], values[1:], strict=True)]
    mean = sum(returns) / 20
    ret = values[-1] / values[0] - 1
    volatility = math.sqrt(sum((r - mean) ** 2 for r in returns) / 20)
    x = (*item.x, ret, volatility, item.x[1] - ret)
    result = StudyInput(
        **{
            **item.model_dump(),
            "x": x,
            "input_hash": digest({"input": item.input_hash, "market_dates": [str(d) for d in dates], "prices": values}),
        }
    )
    return result, None


def stage_cutoffs(availability, lower, upper, *, exam=False):
    calendar = load_calendar()
    rows = [
        r
        for r in availability
        if lower < r["cutoff"] <= upper
        and (not exam or str(calendar.future_sessions(date.fromisoformat(r["cutoff"]))[-1]) <= upper)
    ]
    good, reasons = [], Counter()
    for r in rows:
        if r["input_issues"]:
            reasons.update(r["input_issues"])
        elif not r["label"]:
            reasons.update(r["label_issues"])
        elif r["label"]["available_at"] > upper:
            reasons["ANSWER_NOT_MATURE"] += 1
        else:
            good.append(r["cutoff"])
    return {"planned": len(rows), "usable": len(good), "cutoffs": good, "issues": dict(reasons)}
