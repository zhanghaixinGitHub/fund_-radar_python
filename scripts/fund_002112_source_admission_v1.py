"""按冻结来源分别审查事件范围、时间、去重及开发期变化；不联网、不生成训练候选。"""

from __future__ import annotations

import bisect
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from app.services.fund_public_catalog_v3 import append_catalog

from scripts import fund_002112_price_sources_v1 as prior
from scripts.fund_002112_public_history_v1 import records

io = prior.io
ROOT = io.RESEARCH / "update-frequency-development/20261001-v1"


def audit():
    """独立栏目逐页读回；官网域名和显示日期本身不证明历史同版本已公开。"""
    if (ROOT / "event-source-admission.json").exists():
        return io.read(ROOT / "event-source-admission.json")
    frozen = io.Frozen(prior.timing.event.OLD)
    bound = {}

    def read_entry(key):
        name = frozen.inventory["entries"][key]
        return read_frozen(name)

    def read_frozen(name):
        record = frozen.records[str(Path(name).resolve())]
        path = frozen.root / record["snapshot_path"]
        bound[str(path)] = record["sha256"]
        return frozen.get(name)

    columns = read_entry("catalogs")["columns"]
    news = read_entry("news")
    public = read_entry("public")["rows"]
    historical = read_entry("historical")["rows"]
    materials = io.payload(read_entry("materials"))
    source_rows, catalog_checks = {}, []
    for name, column in [(f"catalog_{c['column']}", c) for c in columns] + [("news_current_14", news)]:
        source_rows[name] = column["all_entries"]
        replay, receipts = [], []
        for index, receipt in enumerate(column["receipts"]):
            path = Path(receipt["path"])
            expected = receipt["sha256"]
            assert io.sha(path) == expected
            bound[str(path)] = expected  # 摘要来自已冻结栏目记录，不能用当前任意同名文件替代。
            rejected_group = any(v.get("group") == index for v in column["issues"] if isinstance(v, dict))
            if not rejected_group:
                append_catalog(replay, records(path.read_text(encoding="utf-8")), index, column["declared_total"])
            receipts.append({k: receipt.get(k) for k in ("path", "sha256", "url", "received_at", "params")})
        assert replay == column["all_entries"]
        urls = [r["url"] for r in replay]
        catalog_checks.append(
            {
                "source": name,
                "column": column["column"],
                "records": len(replay),
                "unique_urls": len(set(urls)),
                "url_alias_records": len(urls) - len(set(urls)),
                "declared_total": column["declared_total"],
                "current_snapshot_complete": column["complete"],
                "issues": column["issues"],
                "raw_pages_replayed": len(receipts),
                "receipts": receipts,
                "request_scope": "REGISTERED_OFFICIAL_COLUMN_PAGINATION_NO_STOCK_LIST_OR_KEYWORD_FILTER",
                "historical_deleted_items_complete": False,
                "historical_same_version_publication_proven": False,
            }
        )
    # 正文验收行由三个独立栏目2016-12-24至2023-12-31的条目网址去重而来。
    independent = [columns[0], columns[2], news]
    entry_urls = {r["url"] for c in independent for r in c["entries_in_range"]}
    public_details = []
    for r in public:
        doc = read_frozen(r["source"]["path"])
        semantic = read_frozen(r["semantic_source"]["path"])
        aliases = doc.get("aliases", [])
        alias_urls = {a.get("url") for a in aliases if isinstance(a, dict)}
        linked = r["url"] in entry_urls or bool(alias_urls & entry_urls)
        public_details.append(
            {
                "id": r["id"],
                "url": r["url"],
                "origin_catalog_url_verified": linked,
                "aliases": aliases,
                "published_date": r.get("published_date"),
                "available_at": r.get("available_at"),
                "body_present": bool(doc.get("text")),
                "body_verified": doc.get("body_verified", False),
                "title_identity_verified": doc.get("title_identity_verified", False),
                "publication_day_verified": doc.get("publication_day_verified", False),
                "whole_version_verified": doc.get("version_complete", False),
                "first_seen_archive_proven": r["first_seen_archive_proven"],
                "body_version_updated_at": semantic.get("body_version_updated_at"),
                "gaps": r.get("gaps"),
                "field_gaps": semantic.get("field_gaps"),
                "source": r["source"],
                "semantic_source": r["semantic_source"],
            }
        )
    source_rows.update(public=public, historical=historical, materials=materials["documents"])
    events_path = prior.timing.event.OLD / "events.jsonl"
    bound[str(events_path)] = io.sha(events_path)
    events = io.lines(events_path)
    # r7已经去重；逐来源引用可能重叠，不把分组之和当总事件数。
    membership = {name: set() for name in source_rows}
    for index, event in enumerate(events):
        for ref in event["source_refs"]:
            entry = ref["entry"]
            key = f"catalog_{columns[ref['column']]['column']}" if entry == "catalogs" else entry
            if key == "news":
                key = "news_current_14"
            if key in membership:
                membership[key].add(index)
    calendar = read_entry("cn_a_share_2021_2025_v1.json")
    closed = [(a, b) for y in calendar["years"] for a, b in y["closed_ranges"]]
    sessions = []
    day = date(2021, 1, 1)
    while day <= date(2025, 12, 31):
        value = day.isoformat()
        if day.weekday() < 5 and not any(a <= value <= b for a, b in closed):
            sessions.append(value)
        day += timedelta(days=1)
    rows = io.lines(prior.ROOT / "inputs.jsonl")
    variations, table = [], []
    for name, raw in source_rows.items():
        members = [events[i] for i in sorted(membership[name])]
        dates = [r.get("display_date") or r.get("published_date") or r.get("publishedDate") for r in raw]
        dates = [str(d)[:10] for d in dates if d]
        active = [r for r in members if r["usable_for_count"] and r["effective_session_main"]]
        per_row = []
        for row in rows:
            position = bisect.bisect_left(sessions, row["base"])
            counts = []
            for window in (1, 5, 20):
                start = sessions[max(0, position - window + 1)]
                allowed = [r for r in active if start <= r["effective_session_main"] <= row["base"]]
                counts.extend(
                    [len(allowed), sum(r["usable_for_text"] and r["text_level"] != "TITLE_ONLY" for r in allowed)]
                )
            per_row.append({"source": name, "target": row["target"], "as_of": row["as_of"], "counts": counts})
        variations.extend(per_row)

        def coverage(year, per_row=per_row):
            values = [r["counts"] for r in per_row if r["target"].startswith(year)]
            return {
                "days": len(values),
                "nonzero_count_days": sum(any(v[::2]) for v in values),
                "nonzero_body_days": sum(any(v[1::2]) for v in values),
                "distinct_count_vectors": len({tuple(v[::2]) for v in values}),
                "distinct_body_vectors": len({tuple(v[1::2]) for v in values}),
            }

        independent_source = name.startswith("catalog_") or name in ("public", "news_current_14")
        table.append(
            {
                "source": name,
                "raw_rows": len(raw),
                "deduplicated_event_references": len(members),
                "raw_date_min": min(dates) if dates else None,
                "raw_date_max": max(dates) if dates else None,
                "raw_year_counts": dict(sorted(Counter(d[:4] for d in dates).items())),
                "domains": dict(Counter(urlsplit(r.get("url") or "").netloc for r in raw)),
                "usable_count_references": len(active),
                "usable_body_references": sum(
                    r["usable_for_text"] and r["text_level"] != "TITLE_ONLY" for r in members
                ),
                "title_only_references": sum(r["text_level"] == "TITLE_ONLY" for r in members),
                "excluded_reasons": dict(Counter(r["excluded_reason"] for r in members if r["excluded_reason"])),
                "revision_marked_references": sum(bool(r.get("revision_at") or r.get("version_note")) for r in members),
                "collection_class": (
                    "INDEPENDENT_COLUMN_WITH_TIME_COVERAGE_LIMITS"
                    if independent_source
                    else "HOLDINGS_DEPENDENT_COMPANY_SCOPE"
                    if name == "materials"
                    else "FUTURE_HOLDINGS_SELECTION_NOT_EXCLUDED_FOR_FIXED_COMPANY_SCOPE"
                ),
                "initial_topic_choice_independent_of_later_holdings_proven": False,
                "within_column_company_or_keyword_filter_found": False if independent_source else None,
                "strict_historical_admission": False,
                "development_2024": coverage("2024"),
                "development_2025": coverage("2025"),
                "variation_basis": "R7_CONSERVATIVE_DATE_RECONSTRUCTION_COUNTS_FOR_COVERAGE_ONLY_NOT_TRAINING",
            }
        )
    code_paths = [
        Path(__file__),
        Path("scripts/fund_002112_closure_public_v1.py"),
        Path("scripts/fund_002112_closure_news_snapshot_v1.py"),
        Path("scripts/fund_002112_public_history_v1.py"),
        Path("scripts/fund_002112_data_completion_v1.py"),
        Path("app/services/fund_public_catalog_v3.py"),
    ]
    code = {}
    for path in code_paths:
        dest = ROOT / "source-audit-code" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        code[str(path.resolve())] = {"sha256": io.sha(path), "copy": str(dest)}
    io.save_lines(ROOT / "public-row-admission.jsonl", public_details)
    io.save_lines(ROOT / "event-source-variation.jsonl", variations)
    io.save(ROOT / "catalog-source-replay.json", catalog_checks)
    io.save(ROOT / "event-audit-source-manifest.json", {"at": io.now(), "files": bound, "code": code})
    outcome = {
        "at": io.now(),
        "sources": table,
        "public_body_rows": len(public),
        "public_origin_url_unmatched": sum(not r["origin_catalog_url_verified"] for r in public_details),
        "public_first_seen_proven": sum(r["first_seen_archive_proven"] for r in public_details),
        "public_body_present": sum(r["body_present"] for r in public_details),
        "public_whole_version_verified": sum(r["whole_version_verified"] for r in public_details),
        "materials_notice": materials.get("notice"),
        "db_research_cards": "HOLDINGS_OR_COMPANY_SELECTED_SCOPE_NOT_AN_INDEPENDENT_NEWS_UNIVERSE",
        "db_news_items": "NO_ROWS_IN_FROZEN_R7_ADAPTED_SOURCE_REFERENCES",
        "no_permanent_blanket_event_ban": True,
        "new_fits": 0,
        "new_external_requests": 0,
        "decision": "SOURCE_SPECIFIC_RESEARCH_REVIEW_ONLY_NO_EVENT_TRAINING_THIS_STAGE",
        "next_stage_candidate": "Independent column title/count branch after time and topic-selection review",
        "next_stage_budget_proposal": "At most2 variants x3 folds =6fits plus1replay; not authorized or run here",
    }
    io.save(ROOT / "event-source-admission.json", outcome)
    return {
        "sources": len(table),
        "public_origin_url_unmatched": outcome["public_origin_url_unmatched"],
        "new_fits": 0,
        "manifest_files": len(bound),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
