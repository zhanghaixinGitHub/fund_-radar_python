"""在新目录修订逐来源D08覆盖；显式datetime边界，不把旧main或aux当作08点准入。"""

from __future__ import annotations

import bisect
import json
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from scripts import fund_002112_update_frequency_v1 as previous

io = previous.io
ROOT = io.RESEARCH / "event-time-combination-review/20261001-v1"
ZONE = timezone(timedelta(hours=8))


def constraint(value, name):
    """只有日期时以次日00点作保守上界；中国来源无时区的明确时刻按北京时间解释并记录。"""
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 10:
        point = datetime.combine(date.fromisoformat(text) + timedelta(days=1), time(), ZONE)
        basis = "DATE_ONLY_DELAY_TO_NEXT_CALENDAR_DAY_0000"
    else:
        point = datetime.fromisoformat(text)
        basis = "EXPLICIT_TIMESTAMP"
        if point.tzinfo is None:
            point = point.replace(tzinfo=ZONE)
            basis = "EXPLICIT_CHINA_TIMESTAMP_ASSUMED_ASIA_SHANGHAI"
        point = point.astimezone(ZONE)
    return {"name": name, "raw": value, "not_before": point.isoformat(), "basis": basis}


def availability(event, extra_revisions=()):
    """公布时刻和已知修订取最晚值；日期级不准同日08点，版本历史是否完整另行保留未知。"""
    evidence = []
    published = event.get("published_at") or event.get("published_date")
    if not published:
        return None, [], "NO_PUBLICATION_BOUND"
    try:
        evidence.append(constraint(published, "publication"))
        if event.get("published_at") and event.get("published_date"):
            declared = date.fromisoformat(event["published_date"])
            parsed = datetime.fromisoformat(evidence[0]["not_before"]).date()
            if declared != parsed:
                evidence.append(constraint(event["published_date"], "conflicting_declared_day_conservative"))
        for name, value in [("event_revision", event.get("revision_at")), *extra_revisions]:
            if value:
                evidence.append(constraint(value, name))
        if event.get("force_next"):
            evidence.append(constraint(event.get("published_date") or str(published)[:10], "preserve_post_close_delay"))
    except (ValueError, TypeError):
        return None, evidence, "MALFORMED_TIME_CONSTRAINT_NOT_ADMITTED"
    points = [datetime.fromisoformat(v["not_before"]) for v in evidence]
    return max(points), evidence, None


def first_morning(point, sessions):
    """返回第一个08:00不早于所有已知约束的交易日；08点之后发布必须等到后续交易日。"""
    if point is None:
        return None
    boundaries = [datetime.combine(date.fromisoformat(d), time(8), ZONE) for d in sessions]
    index = bisect.bisect_left(boundaries, point)
    return sessions[index] if index < len(sessions) else None


def revise():
    """逐事件保存时间证据，逐来源逐日与已冻结旧覆盖对照，不修改旧字段。"""
    if (ROOT / "coverage-revision.json").exists():
        return io.read(ROOT / "coverage-revision.json")
    old = previous.prior.source.timing.event.OLD
    frozen = io.Frozen(old)
    sources = {}
    prior_sources = io.read(previous.ROOT / "event-audit-source-manifest.json")["files"]

    def bind(path):
        path = Path(path)
        if str(path) in prior_sources:
            assert io.sha(path) == prior_sources[str(path)]
        sources[str(path)] = io.sha(path)
        return path

    events = io.lines(bind(old / "events.jsonl"))
    rows = io.lines(bind(previous.ROOT / "inputs.jsonl"))
    earlier = io.lines(bind(previous.ROOT / "event-source-variation.jsonl"))
    old_by_key = {(v["source"], v["target"]): v for v in earlier}
    summary = io.read(bind(previous.ROOT / "event-source-admission.json"))
    groups = {r["source"]: set() for r in summary["sources"]}
    columns = frozen.entry("catalogs")["columns"]
    calendars = [
        frozen.entry(name) for name in ("cn_a_share_2015_2020_research_v1.json", "cn_a_share_2021_2025_v1.json")
    ]
    for name in ("catalogs", "cn_a_share_2015_2020_research_v1.json", "cn_a_share_2021_2025_v1.json"):
        record = frozen.records[frozen.inventory["entries"][name]]
        bind(frozen.root / record["snapshot_path"])
    closed = [(a, b) for c in calendars for y in c["years"] for a, b in y["closed_ranges"]]
    sessions = []
    day = date(2015, 1, 1)
    while day <= date(2025, 12, 31):
        value = day.isoformat()
        if day.weekday() < 5 and not any(a <= value <= b for a, b in closed):
            sessions.append(value)
        day += timedelta(days=1)
    morning_values = [datetime.combine(date.fromisoformat(d), time(8), ZONE) for d in sessions]
    revised = []
    semantic_cache = {}
    for index, e in enumerate(events):
        revisions = []
        for ref in e["source_refs"]:
            name = ref["entry"]
            if name == "catalogs":
                name = f"catalog_{columns[ref['column']]['column']}"
            elif name == "news":
                name = "news_current_14"
            if name in groups:
                groups[name].add(index)
            semantic = ref.get("raw_refs", {}).get("semantic_source")
            if semantic:
                path = semantic["path"]
                if path not in semantic_cache:
                    semantic_cache[path] = frozen.get(path)
                    bind(frozen.root / frozen.records[path]["snapshot_path"])
                value = semantic_cache[path].get("body_version_updated_at")
                if value:
                    revisions.append((path + "#body_version_updated_at", value))
        point, evidence, error = availability(e, revisions)
        position = bisect.bisect_left(morning_values, point) if point else len(sessions)
        effective = sessions[position] if position < len(sessions) else None
        revised.append(
            {
                "event_index": index,
                "event_id": e["event_id"],
                "title": e["title"],
                "published_date": e["published_date"],
                "published_at": e["published_at"],
                "revision_at": e["revision_at"],
                "source_refs": e["source_refs"],
                "known_time_constraints": evidence,
                "not_before": point.isoformat() if point else None,
                "effective_D0800": effective,
                "old_main": e["effective_session_main"],
                "old_aux": e["effective_session_aux"],
                "error": error,
                "count_coverage_candidate": bool(e["usable_for_count"] and effective and not error),
                "historical_same_version_complete": False,
                "strict_training_admitted": False,
            }
        )
    differences, group_summary, examples = [], [], []
    for name, indices in groups.items():
        active = [i for i in sorted(indices) if revised[i]["count_coverage_candidate"]]
        local = []
        for row in rows:
            origin = row["base"]
            as_of = datetime.fromisoformat(row["as_of"])
            offset = bisect.bisect_left(sessions, origin)
            values = []
            for window in (1, 5, 20):
                first = sessions[max(0, offset - window + 1)]
                members = [i for i in active if first <= revised[i]["effective_D0800"] <= origin]
                assert all(datetime.fromisoformat(revised[i]["not_before"]) <= as_of for i in members)
                values.extend(
                    [
                        len(members),
                        sum(events[i]["usable_for_text"] and events[i]["text_level"] != "TITLE_ONLY" for i in members),
                    ]
                )
            old_values = old_by_key[name, row["target"]]["counts"]
            value = {
                "source": name,
                "target": row["target"],
                "as_of": row["as_of"],
                "old": old_values,
                "revised": values,
                "delta": [b - a for a, b in zip(old_values, values, strict=True)],
            }
            local.append(value)
        differences.extend(local)
        by_year = {}
        for year in ("2024", "2025"):
            subset = [v for v in local if v["target"].startswith(year)]
            by_year[year] = {
                "days": len(subset),
                "any_vector_changed_days": sum(v["old"] != v["revised"] for v in subset),
            }
            for field in ("old", "revised"):
                by_year[year][field] = {
                    "nonzero_count_days": sum(any(v[field][::2]) for v in subset),
                    "nonzero_body_days": sum(any(v[field][1::2]) for v in subset),
                    "distinct_count_vectors": len({tuple(v[field][::2]) for v in subset}),
                    "distinct_body_vectors": len({tuple(v[field][1::2]) for v in subset}),
                }
        group_summary.append({"source": name, "years": by_year})
        for i in sorted(indices):
            e, v = events[i], revised[i]
            if (
                e["published_date"] == "2025-01-02"
                and e["published_at"] is None
                and v["effective_D0800"] != e["effective_session_main"]
            ):
                examples.append(
                    {
                        "source": name,
                        "event_id": e["event_id"],
                        "title": e["title"],
                        "origin_as_of": "2025-01-02T08:00:00+08:00",
                        "old_main": v["old_main"],
                        "not_before": v["not_before"],
                        "revised_effective_D0800": v["effective_D0800"],
                    }
                )
                break
    io.save_lines(ROOT / "event-time-boundaries.jsonl", revised)
    io.save_lines(ROOT / "source-daily-differences.jsonl", differences)
    io.save(ROOT / "coverage-source-manifest.json", {"at": io.now(), "files": sources})
    outcome = {
        "at": io.now(),
        "status": "D0800_REFERENCE_COVERAGE_CORRECTED_NOT_TRAINING_ADMISSION",
        "sources": group_summary,
        "counterexamples": examples,
        "event_count": len(events),
        "event_errors": dict(Counter(v["error"] for v in revised if v["error"])),
        "old_files_overwritten": False,
        "supervised_fits": 0,
        "new_event_training": False,
        "source_version_gap_closed": False,
        "old_130_model_affected": False,
        "coverage_only": True,
        "script_sha256": io.sha(Path(__file__)),
    }
    io.save(ROOT / "coverage-revision.json", outcome)
    return {"sources": len(groups), "events": len(events), "daily_comparisons": len(differences), "supervised_fits": 0}


if __name__ == "__main__":
    print(json.dumps(revise(), ensure_ascii=False, indent=2))
