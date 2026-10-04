"""独立核验修订：直接以时间区间计数，不调用覆盖生成器的交易日映射。"""

from __future__ import annotations

import ast
import bisect
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from scripts import fund_002112_fixed_combinations_v1 as batch

io, ROOT = batch.io, batch.ROOT
ZONE = timezone(timedelta(hours=8))


def point(value):
    """独立解析已知时刻；日期最早允许次日，不能假定当天08点可见。"""
    if len(value) == 10:
        return datetime.combine(date.fromisoformat(value) + timedelta(days=1), time(), ZONE)
    result = datetime.fromisoformat(value)
    return result.replace(tzinfo=ZONE) if result.tzinfo is None else result.astimezone(ZONE)


def verify():
    batch.check_freeze()
    old = batch.old.prior.source.timing.event.OLD
    frozen = io.Frozen(old)
    events = io.lines(old / "events.jsonl")
    saved = io.lines(ROOT / "event-time-boundaries.jsonl")
    groups = {s["source"]: set() for s in io.read(ROOT / "coverage-revision.json")["sources"]}
    columns = frozen.entry("catalogs")["columns"]
    raw_cache = {}
    known = []
    for index, (e, out) in enumerate(zip(events, saved, strict=True)):
        assert e["event_id"] == out["event_id"]
        values = [point(e["published_at"] or e["published_date"])]
        if e["published_at"] and date.fromisoformat(e["published_date"]) != values[0].date():
            values.append(point(e["published_date"]))
        if e["revision_at"]:
            values.append(point(e["revision_at"]))
        if e["force_next"]:
            values.append(point(e["published_date"]))
        for ref in e["source_refs"]:
            key = ref["entry"]
            if key == "catalogs":
                key = f"catalog_{columns[ref['column']]['column']}"
            elif key == "news":
                key = "news_current_14"
            if key in groups:
                groups[key].add(index)
            source = ref.get("raw_refs", {}).get("semantic_source")
            if source:
                path = source["path"]
                if path not in raw_cache:
                    raw_cache[path] = frozen.get(path)
                value = raw_cache[path].get("body_version_updated_at")
                if value:
                    values.append(point(value))
        maximum = max(values)
        assert maximum == datetime.fromisoformat(out["not_before"])
        assert not out["strict_training_admitted"] and not out["historical_same_version_complete"]
        known.append(maximum)
    calendars = [frozen.entry(k) for k in ("cn_a_share_2015_2020_research_v1.json", "cn_a_share_2021_2025_v1.json")]
    closed = [(a, b) for c in calendars for y in c["years"] for a, b in y["closed_ranges"]]
    sessions = []
    day = date(2015, 1, 1)
    while day <= date(2025, 12, 31):
        if day.weekday() < 5 and not any(a <= str(day) <= b for a, b in closed):
            sessions.append(str(day))
        day += timedelta(days=1)
    times = [datetime.combine(date.fromisoformat(day), time(8), ZONE) for day in sessions]
    for e, bound, out in zip(events, known, saved, strict=True):
        index = bisect.bisect_left(times, bound)
        assert out["effective_D0800"] == (sessions[index] if index < len(sessions) else None)
        assert out["count_coverage_candidate"] == (e["usable_for_count"] and index < len(sessions))
    changes = io.lines(ROOT / "source-daily-differences.jsonl")
    old_vectors = {
        (v["source"], v["target"]): v["counts"] for v in io.lines(batch.old.ROOT / "event-source-variation.jsonl")
    }
    sorted_times = {}
    for name, indices in groups.items():
        allowed = [i for i in indices if events[i]["usable_for_count"]]
        sorted_times[name] = [
            sorted(known[i] for i in allowed),
            sorted(
                known[i] for i in allowed if events[i]["usable_for_text"] and events[i]["text_level"] != "TITLE_ONLY"
            ),
        ]
    for row in changes:
        end = datetime.fromisoformat(row["as_of"])
        index = sessions.index(str(end.date()))
        expected = []
        for window in (1, 5, 20):
            # 首个窗口交易日之前一次08点是开区间，下界之后、当日08点之前为可观察事件。
            begin = times[index - window]
            expected.extend(
                bisect.bisect_right(values, end) - bisect.bisect_right(values, begin)
                for values in sorted_times[row["source"]]
            )
        assert expected == row["revised"]
        assert row["old"] == old_vectors[row["source"], row["target"]]
        assert row["delta"] == [a - b for a, b in zip(expected, row["old"], strict=True)]
    for s in io.read(ROOT / "coverage-revision.json")["sources"]:
        for year, stats in s["years"].items():
            subset = [r for r in changes if r["source"] == s["source"] and r["target"].startswith(year)]
            assert stats["days"] == len(subset)
            assert stats["any_vector_changed_days"] == sum(r["old"] != r["revised"] for r in subset)
            for field in ("old", "revised"):
                assert stats[field]["nonzero_count_days"] == sum(any(r[field][::2]) for r in subset)
                assert stats[field]["nonzero_body_days"] == sum(any(r[field][1::2]) for r in subset)
                assert stats[field]["distinct_count_vectors"] == len({tuple(r[field][::2]) for r in subset})
                assert stats[field]["distinct_body_vectors"] == len({tuple(r[field][1::2]) for r in subset})
    for path, digest in io.read(ROOT / "coverage-source-manifest.json")["files"].items():
        assert io.sha(Path(path)) == digest
    inventory = io.read(ROOT / "remaining-information-inventory.json")
    assert all(io.sha(Path(p)) == h for p, h in inventory["source_files"].items())
    executed = ROOT / "inventory-code-snapshot/fund_002112_frozen_information_inventory_v1.py"
    current = Path("scripts/fund_002112_frozen_information_inventory_v1.py")
    assert io.sha(executed) == inventory["script_sha256"]
    assert ast.dump(ast.parse(executed.read_text("utf-8")), include_attributes=False) == ast.dump(
        ast.parse(current.read_text("utf-8")), include_attributes=False
    )
    prior_proof = io.read(batch.old.ROOT / "final-verification.json")
    assert all(io.sha(batch.old.ROOT / p) == h for p, h in prior_proof["artifacts"].items())
    result = {
        "at": io.now(),
        "event_datetime_bounds_independently_checked": len(events),
        "source_day_vectors_independently_recomputed": len(changes),
        "source_summaries_checked": len(groups),
        "all_known_revision_constraints_checked": True,
        "old_main_and_aux_not_used_for_new_admission": True,
        "old_stage_artifacts_preserved": len(prior_proof["artifacts"]),
        "supervised_fits": 0,
        "inventory_generator_format_only_ast_identical": True,
        "strict_event_admission_passed": False,
        "script_sha256": io.sha(Path(__file__)),
    }
    io.save(ROOT / "coverage-independent-verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
