"""2021—2024现金研究快照与覆盖；复用原公式，不修改旧HTTP样本范围。"""

import json
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, localcontext
from uuid import UUID

from app.repositories.cash_reinvestment_samples import (
    CashDividend,
    CashNavPoint,
    read_cash_dividends,
    read_cash_nav_window,
)
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.repositories.historical_nav import read_historical_nav_source
from app.repositories.trading_nav_window import NavDatePoint
from app.schemas.direction_training import DirectionAnswer, DirectionInput
from app.services.cash_reinvestment_research import FUNDS
from app.services.cash_reinvestment_samples import build_cash_history_feature, build_cash_return_series
from app.services.direction_training_artifacts import digest, new_folder, read_seal, seal, write_json, write_jsonl
from app.services.direction_training_inventory import local_snapshot
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.trading_calendar import load_calendar
from app.services.trading_nav_window import build_history_date_window

START, END = date(2021, 1, 1), date(2024, 12, 31)
WINDOWS = (
    ("DEV_2023_Q3_V2", "2022-12-31", "2023-06-30", "2023-09-30"),
    ("DEV_2023_Q4_V2", "2023-03-31", "2023-09-30", "2023-12-31"),
    ("DEV_2024_Q1_V2", "2023-06-30", "2023-12-31", "2024-03-31"),
    ("DEV_2024_Q2_V2", "2023-09-30", "2024-03-31", "2024-06-30"),
    ("DEV_2024_Q3_V2", "2023-12-31", "2024-06-30", "2024-09-30"),
    ("DEV_2024_Q4_V2", "2024-03-31", "2024-09-30", "2024-12-31"),
)


def json_ready(value):
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def read_source_snapshot() -> dict:
    funds = {}
    with local_snapshot() as (session, statements):
        for fund in FUNDS:
            source = read_historical_nav_source(session, fund_code=fund)
            if source.source_code != "TUSHARE_PRO_FUND":
                raise ValueError("UNSUPPORTED_SOURCE")
            nav, events = [], {}
            cursor = START
            while cursor <= END:
                upper = min(cursor + timedelta(days=89), END)
                nav.extend(
                    read_cash_nav_window(session, fund_code=fund, source_id=source.source_id, start=cursor, end=upper)
                )
                for record in read_cash_dividends(
                    session, fund_code=fund, source_id=source.source_id, start=cursor, end=upper
                ):
                    if record.event_key in events and events[record.event_key] != record:
                        raise ValueError("EVENT_SNAPSHOT_CONFLICT")
                    events[record.event_key] = record
                cursor = upper + timedelta(days=1)
            if len(nav) > 1461 or len(events) > 100 or len({p.nav_date for p in nav}) != len(nav):
                raise ValueError("SOURCE_BUDGET_OR_DUPLICATE")
            funds[fund] = json_ready(
                {
                    "source": asdict(source),
                    "nav": [asdict(p) for p in nav],
                    "events": [asdict(events[k]) for k in sorted(events)],
                }
            )
    return {
        "version": "DIRECTION_SOURCE_V1",
        "start": str(START),
        "end": str(END),
        "funds": funds,
        "calendar_hash": load_calendar().content_hash,
        "sql_statement_counts": dict(statements),
    }


def restore_source(snapshot: dict, fund: str):
    if fund not in FUNDS or snapshot["start"] != str(START) or snapshot["end"] != str(END):
        raise ValueError("SOURCE_RANGE_INVALID")
    if snapshot["calendar_hash"] != load_calendar().content_hash:
        raise ValueError("CALENDAR_CHANGED")
    if set(snapshot["funds"]) != set(FUNDS) or snapshot["version"] != "DIRECTION_SOURCE_V1":
        raise ValueError("SOURCE_COHORT_INVALID")
    data = snapshot["funds"][fund]
    meta = data["source"]
    if meta["source_code"] != "TUSHARE_PRO_FUND":
        raise ValueError("UNSUPPORTED_SOURCE")
    source = FeatureSourceReadiness(
        UUID(meta["source_id"]),
        meta["source_code"],
        UUID(meta["source_sync_run_id"]),
        datetime.fromisoformat(meta["source_sync_finished_at"]),
    )
    nav = tuple(
        CashNavPoint(
            date.fromisoformat(p["nav_date"]),
            date.fromisoformat(p["ann_date"]) if p["ann_date"] else None,
            Decimal(p["unit_nav"]) if p["unit_nav"] is not None else None,
        )
        for p in data["nav"]
    )
    if len(nav) > 1461 or any(not START <= p.nav_date <= END for p in nav):
        raise ValueError("PROTECTED_SOURCE_DATE")
    if any(a.nav_date >= b.nav_date for a, b in zip(nav[:-1], nav[1:], strict=True)):
        raise ValueError("SOURCE_ORDER_OR_DUPLICATE")
    events = []
    for raw in data["events"]:
        row = dict(raw)
        for key in ("ann_date", "implementation_ann_date", "ex_date", "nav_ex_date"):
            row[key] = date.fromisoformat(row[key]) if row[key] else None
        row["cash_dividend"] = Decimal(row["cash_dividend"]) if row["cash_dividend"] is not None else None
        if any(row[k] and row[k] > END for k in ("ann_date", "implementation_ann_date", "ex_date", "nav_ex_date")):
            raise ValueError("PROTECTED_EVENT_DATE")
        events.append(CashDividend(**row))
    if len(events) > 100 or len({e.event_key for e in events}) != len(events):
        raise ValueError("SOURCE_EVENT_BUDGET")
    return source, nav, tuple(events)


def build_input(fund, cutoff, source, nav, events) -> tuple[DirectionInput | None, list[str]]:
    if fund not in FUNDS or not START <= cutoff <= END:
        raise ValueError("INPUT_DATE_PROTECTED")
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
    return DirectionInput(
        fund=fund,
        cutoff=cutoff,
        anchor=feature.anchor_nav_date,
        available_at=feature.available_at,
        x=tuple(float(feature.metrics[n]) for n in FEATURE_NAMES),
        input_hash=digest(feature.model_dump(mode="json")),
    ), []


def build_answer(fund, cutoff, nav, events, *, include_value: bool):
    """只允许完整窗口留在2024内；覆盖阶段仅返回可得元数据，不计算/返回方向。"""
    if fund not in FUNDS or not START <= cutoff <= END:
        raise ValueError("ANSWER_DATE_PROTECTED")
    calendar = load_calendar()
    future = calendar.future_sessions(cutoff)
    if future[-1] > END:
        return None, ["LABEL_CROSSES_PROTECTED_PERIOD"]
    if any(n.announced_on > cutoff for n in calendar.definition.years if n.year in {d.year for d in future}):
        return None, ["FUTURE_CALENDAR_NOT_KNOWN_AT_CUTOFF"]
    base = calendar.sessions[calendar.at_or_before_index(cutoff)]
    points, problems = build_cash_return_series(
        (base, *future), {p.nav_date: p for p in nav}, events, latest_available=END
    )
    if problems:
        return None, sorted({p.code for p in problems})
    metadata = {
        "fund": fund,
        "cutoff": str(cutoff),
        "end": str(future[-1]),
        "available_at": str(points[-1].available_at),
    }
    if not include_value:
        return metadata, []
    with localcontext() as context:
        context.prec = 40
        context.rounding = ROUND_HALF_UP
        value = (Decimal(points[-1].growth_index) / 100 - 1).quantize(Decimal("0.000000000001"))
    return DirectionAnswer(**metadata, y=int(value > 0), future_return=str(value)), []


def exam_dates(lower: date, upper: date):
    calendar = load_calendar()
    if not START <= lower < upper <= END:
        raise ValueError("EXAM_DATES_PROTECTED")
    dates = [d for d in calendar.sessions if lower < d <= upper]
    return dates, [d for d in dates if calendar.future_sessions(d)[-1] <= upper]


def build_coverage(snapshot):
    rows, inputs = [], []
    dates = [d for d in load_calendar().sessions if START <= d <= END]
    for fund in FUNDS:
        source, nav, events = restore_source(snapshot, fund)
        for cutoff in dates:
            item, issues = build_input(fund, cutoff, source, nav, events)
            if item:
                inputs.append(item.model_dump(mode="json"))
                label, label_issues = build_answer(fund, cutoff, nav, events, include_value=False)
            else:
                label, label_issues = None, ["INPUT_UNAVAILABLE"]
            rows.append(
                {
                    "fund": fund,
                    "cutoff": str(cutoff),
                    "input_available": item is not None,
                    "input_issues": issues,
                    "label": label,
                    "label_issues": label_issues,
                }
            )
    windows = {}
    for name, fit, cal, exam in WINDOWS:
        fit, cal, exam = map(date.fromisoformat, (fit, cal, exam))
        candidates, planned = exam_dates(cal, exam)
        groups = {}
        for fund in FUNDS:
            stages = {}
            for stage, lower, upper, minimum in (
                ("FIT", START - timedelta(days=1), fit, 252),
                ("CAL", fit, cal, 60),
                ("EXAM", cal, exam, 40),
            ):
                selected = [r for r in rows if r["fund"] == fund and lower < date.fromisoformat(r["cutoff"]) <= upper]
                if stage == "EXAM":
                    selected = [r for r in selected if date.fromisoformat(r["cutoff"]) in planned]
                good, reasons = [], Counter()
                for row in selected:
                    if not row["input_available"]:
                        reasons.update(row["input_issues"])
                    elif row["label"] is None:
                        reasons.update(row["label_issues"])
                    elif date.fromisoformat(row["label"]["available_at"]) > upper:
                        reasons["ANSWER_NOT_MATURE_BY_STAGE_END"] += 1
                    else:
                        good.append(row["cutoff"])
                stages[stage] = {
                    "planned": len(selected),
                    "usable": len(good),
                    "missing_minimum": max(0, minimum - len(good)),
                    "cutoffs": good,
                    "issue_counts_nonexclusive": dict(reasons),
                }
            groups[fund] = stages
        windows[name] = {
            "calendar_cutoff_count": len(candidates),
            "planned_exam_count": len(planned),
            "calendar_excluded": len(candidates) - len(planned),
            "funds": groups,
            "status": "READY"
            if all(not s["missing_minimum"] for g in groups.values() for s in g.values())
            else "INSUFFICIENT_DATA",
        }
    report = {
        "version": "DIRECTION_COVERAGE_V1",
        "source_hash": digest(snapshot),
        "windows": windows,
        "planned_rows": len(rows),
        "input_rows": len(inputs),
        "label_ready_rows": sum(r["label"] is not None for r in rows),
        "model_fitted": False,
        "test_scored": False,
        "coverage_is_not_accuracy": True,
    }
    return report, inputs, rows


def export_coverage(inventory_folder):
    inventory_manifest = read_seal(inventory_folder, "complete.json")
    folder = new_folder()
    snapshot = read_source_snapshot()
    coverage, inputs, rows = build_coverage(snapshot)
    files = {
        "source.json": write_json(folder / "source.json", snapshot),
        "coverage.json": write_json(folder / "coverage.json", coverage),
        "inputs.jsonl": write_jsonl(folder / "inputs.jsonl", inputs),
        "availability.jsonl": write_jsonl(folder / "availability.jsonl", rows),
    }
    seal(
        folder,
        "coverage-manifest.json",
        files,
        status="COVERED",
        inventory_hash=inventory_manifest["manifest_hash"],
        inventory_folder=inventory_folder.name,
        model_fitted=False,
        test_scored=False,
    )
    return folder, coverage
