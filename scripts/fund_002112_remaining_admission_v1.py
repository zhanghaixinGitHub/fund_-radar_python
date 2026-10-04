"""有25分钟上限的剩余资料准入审查：仅冻结输入，只算可构建性，不拟合。"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from scripts import fund_002112_fixed_combinations_v1 as prior

io = prior.io
ROOT = io.RESEARCH / "remaining-admission-review/20261001-v1"
inputs = prior.old.prior.source.timing.old
extra = prior.old.prior.source


def run():
    """严格准入与按元数据推算的可构建性分开；未知来源时刻绝不变成已通过。"""
    if (ROOT / "admission-results.json").exists():
        return io.read(ROOT / "admission-results.json")
    plan = io.read(ROOT / "protocol.json")
    freeze = io.read(ROOT / "source-freeze.json")
    assert io.sha(ROOT / "protocol.json") == freeze["protocol_sha256"]
    assert all(io.sha(Path(p)) == h for p, h in freeze["files"].items())
    assert (ROOT / "fit-ledger.jsonl").stat().st_size == 0
    accessed = []

    def check_time():
        if datetime.fromisoformat(io.now()) >= datetime.fromisoformat(plan["deadline_at"]):
            raise TimeoutError("REGISTERED25MIN_LIMIT_NO_SCOPE_EXPANSION")

    def read(path, mode="JSON"):
        check_time()
        path = Path(path)
        digest = freeze["files"][str(path)]
        assert io.sha(path) == digest
        accessed.append({"at": io.now(), "path": str(path), "sha256": digest, "mode": mode})
        if mode == "JSON":
            return io.read(path)
        if mode == "JSONL":
            return io.lines(path)
        return PdfReader(path)

    inv = read(prior.ROOT / "remaining-information-inventory.json")
    facts_ref = inv["domains"]["peer_funds"]["source"]
    facts = io.payload(read(facts_ref["snapshot"]))
    calendars = []
    for path in freeze["files"]:
        if Path(path).suffix == ".bin" and path != facts_ref["snapshot"]:
            value = read(path)
            if "years" in value:
                calendars.append(value)
    assert len(calendars) == 2
    closed = [(a, b) for c in calendars for y in c["years"] for a, b in y["closed_ranges"]]
    sessions = []
    day = date(2015, 1, 1)
    while day <= date(2025, 12, 31):
        if day.weekday() < 5 and not any(a <= str(day) <= b for a, b in closed):
            sessions.append(str(day))
        day += timedelta(days=1)
    session_position = {d: i for i, d in enumerate(sessions)}
    peers, sample_rows, report_rows = [], [], []
    for code in plan["fund_codes"]:
        check_time()
        fund = facts["funds"][code]
        raw = fund["nav"]["rows"]
        assert fund["nav"]["fund_code"] == code and max(r["date"] for r in raw) <= "2025-12-31"
        dates = [r["date"] for r in raw]
        assert len(set(dates)) == len(dates)
        nav = {}
        invalid = []
        for value in raw:
            if not value.get("ann_date") or not np.isfinite(float(value["nav"])) or float(value["nav"]) <= 0:
                invalid.append(value["date"])
                continue
            nav[value["date"]] = {"unit_nav": value["nav"], "available_at": inputs.available_at(value["ann_date"])}
        local, reasons = [], Counter()
        for origin in sorted(nav):
            if origin not in session_position:
                reasons["NOT_TRADING_SESSION"] += 1
                continue
            j = session_position[origin]
            if j + 1 >= len(sessions) or sessions[j + 1] not in nav:
                reasons["NEXT_SESSION_NAV_ABSENT"] += 1
                continue
            target = sessions[j + 1]
            end, lag, n = inputs.choose_nav_window(sessions, nav, origin)
            if end is None:
                reasons["NO_VALID61_CONTIGUOUS_KNOWN_NAV_WINDOW_WITHIN20SESSION_LAG"] += 1
                continue
            end_index = session_position[end]
            window = sessions[end_index - 60 : end_index + 1]
            ne = extra.nav_extra([float(nav[d]["unit_nav"]) for d in window])
            assert len(n + ne) == 15 and np.isfinite(n + ne).all()
            latest_input = max(nav[d]["available_at"] for d in window)
            as_of = inputs.at0800(origin).isoformat()
            assert latest_input <= as_of and max(window) < origin < target
            maturity = max(nav[origin]["available_at"], nav[target]["available_at"])
            row = {
                "fund_code": code,
                "origin": origin,
                "target": target,
                "as_of": as_of,
                "max_input_metadata_available_at": latest_input,
                "label_mature_metadata_at": maturity,
                "latest_feature_nav_date": end,
                "lag_sessions": lag,
                "nav_window_dates": window,
                "feature_values_15": n + ne,
                "field_names": inputs.N + extra.NE,
                "label_definition": "UNIT_NAV_TARGET_VS_ORIGIN_UNCHANGED_NO_DIVIDEND_ADJUSTMENT",
                "time_basis": "ANN_DATE_NEXT_CALENDAR_DAY0800_METADATA_ONLY_NOT_HISTORICAL_VERSION_PROOF",
                "strict_training_admitted": False,
            }
            local.append(row)
        sample_rows.extend(local)
        reports = fund["reports"]
        for report in reports:
            report_rows.append(
                {
                    "fund_code": code,
                    "master_code": report["fund_master_code"],
                    "title": report["title"],
                    "report_end": report["report_end"],
                    "published_date": report["published_date"],
                    "available_at": report["available_at"],
                    "availability_basis": report["availability_basis"],
                    "raw": report["raw"],
                    "embedded_excerpt_present": bool(report.get("product_description_excerpt")),
                    "raw_external_file_read_this_stage": False,
                }
            )
        lag_counts = Counter((date.fromisoformat(r["ann_date"]) - date.fromisoformat(r["date"])).days for r in raw)
        peers.append(
            {
                "fund_code": code,
                "source": facts_ref,
                "json_pointer": f"payload.funds.{code}",
                "share_identity": {
                    "nav_code_matches": True,
                    "report_codes_match": all(r["fund_code"] == code for r in reports),
                    "master_codes": sorted({r["fund_master_code"] for r in reports}),
                    "package_family": fund.get("family"),
                    "nav_product_family_id": fund["nav"].get("product_family_id"),
                    "letter_class_independently_verified": code == "002112",
                    "same_family_nav_substitution_used": False,
                },
                "nav": {
                    "rows": len(raw),
                    "date_min": min(dates),
                    "date_max": max(dates),
                    "invalid_rows": invalid,
                    "duplicate_dates": len(dates) - len(set(dates)),
                    "source_hash_count": len({r["source_hash"] for r in raw}),
                    "ann_date_lag_days": dict(sorted(lag_counts.items())),
                    "price_basis": "UNIT_NAV_AS_PACKAGED",
                    "cash_dividend_and_split_ledger_present": False,
                    "total_return_nav_present": False,
                    "per_row_direct_raw_receipt_path_present": any("path" in r or "received_at" in r for r in raw),
                    "ann_date_not_first_seen_or_version_proof": True,
                },
                "N_NE": {
                    "fields": inputs.N + extra.NE,
                    "metadata_constructible_rows": len(local),
                    "first_target": local[0]["target"] if local else None,
                    "last_target": local[-1]["target"] if local else None,
                    "unconstructible_reasons": dict(reasons),
                    "feature_missing_values_in_constructed_rows": 0,
                    "all_label_maturity_requires_both_origin_and_target": True,
                    "mature_before_first_2025_origin": sum(
                        r["label_mature_metadata_at"] < "2024-12-31T08:00:00+08:00" for r in local
                    ),
                    "strictly_admitted_rows": 0,
                },
                "selection": {
                    "role": fund["cohort"].get("role"),
                    "group": fund["nav"].get("group_id"),
                    "compatibility": fund["cohort"].get("compatibility"),
                    "evidence_report_periods": sorted({e["report_end"] for e in fund["cohort"].get("evidence", [])}),
                    "first_membership_selection_time_proven": False,
                    "selection_by_future_performance_excluded": False,
                    "group_is_historical_style_or_holdings": False,
                },
                "report_version": {
                    "count": len(reports),
                    "raw_first_seen_verified": sum(
                        r["raw"].get("historical_first_seen_verified") is True for r in reports
                    ),
                    "availability_bases": dict(Counter(r["availability_basis"] for r in reports)),
                    "no_external_report_file_followed": True,
                },
                "admission": "BLOCKED_FOR_STRICT_TRANSFER_TRAINING",
                "blockers": [
                    "NAV_HISTORICAL_VERSION_AND_FIRST_PUBLICATION_UNPROVEN",
                    "COHORT_SELECTION_TIME_UNPROVEN",
                    "DIVIDEND_SHARE_CLASS_COMPARABILITY_NOT_FULLY_VERIFIED",
                ],
                "not_a_blocker_by_itself": "NO2025_PEER_NAV_DOES_NOT_MAKE_PRE2025_TRANSFER_USELESS",
            }
        )
    io.save_lines(ROOT / "peer-metadata-feature-audit.jsonl", sample_rows)
    io.save_lines(ROOT / "embedded-report-version-audit.jsonl", report_rows)
    if (ROOT / "peer-admission.json").exists():
        assert io.read(ROOT / "peer-admission.json")["funds"] == json.loads(json.dumps(peers))
    else:
        io.save(ROOT / "peer-admission.json", {"at": io.now(), "funds": peers, "supervised_fits": 0})
    increment_ref = inv["domains"]["fund_attributes"]["source_increment"]
    increment = read(increment_ref["path"])
    extracts = []
    patterns = {
        "contract_date": r"2015\s*年\s*6\s*月\s*19\s*日",
        "C_date_16": r"2015\s*年\s*11\s*月\s*16\s*日",
        "C_subscription_date_18": r"2015\s*年\s*11\s*月\s*18\s*日",
        "C_share_code": r"002112",
    }
    for reference in plan["additional_target_report_pdfs"]:
        reader = read(reference["raw_path"], "PDF_FIRST16_PAGES_LITERAL_REVIEW")
        quotes = []
        for number, page in enumerate(reader.pages[:16], 1):
            text = page.extract_text() or ""
            for name, pattern in patterns.items():
                match = re.search(pattern, text)
                if match:
                    quotes.append(
                        {"field": name, "page": number, "quote": text[max(0, match.start() - 110) : match.end() + 160]}
                    )
        extracts.append(
            {
                "source": reference,
                "pages_total": len(reader.pages),
                "pages_read": min(16, len(reader.pages)),
                "quotes": quotes,
                "independent_early_publication_proven": False,
                "known_version_bound": reference["conservative_version_bound"],
                "document_content_verified_not_historical_availability": True,
            }
        )
    manager_evidence = []
    by_sha = {v["raw_sha256"]: v for v in increment["manager_notices"]}
    for reference in plan["manager_notice_pdfs"]:
        reader = read(reference["path"], "PDF_FIRST6_PAGES_MANAGER_IDENTITY_REVIEW")
        text = "\n".join(page.extract_text() or "" for page in reader.pages[:6])
        state = by_sha[reference["sha256"]]
        manager_evidence.append(
            {
                "source": reference,
                "pages_read": min(6, len(reader.pages)),
                "fund_code_literal_found": "002112" in text or "001412" in text,
                "fund_name_literal_found": "鑫星" in text,
                "declared_publication": state["declared_publication_date"],
                "cms_time_constraints": state["cms_time_constraints"],
                "available_at": state["available_at"],
                "events": state["events"],
                "version_risk": state["version_risk"],
                "eligible_for2025": state["available_at"] <= "2025-12-31T23:59:59+08:00",
            }
        )
    states = [increment["report_states"][v["raw_sha256"]] for v in plan["additional_target_report_pdfs"]]
    dates = [r["base"] for r in read(prior.old.ROOT / "inputs.jsonl", "JSONL") if r["target"].startswith("2025")]
    ages = [(date.fromisoformat(d) - date(2015, 6, 19)).days for d in dates]
    attributes = {
        "at": io.now(),
        "static": [
            {
                "field": "fund_contract_date",
                "value": "2015-06-19",
                "type": "SINGLE_FUND_CONSTANT",
                "content_quotes_in_reports": sum(
                    any(q["field"] == "contract_date" for q in e["quotes"]) for e in extracts
                ),
                "independent_early_source_available_at": None,
                "admitted": False,
                "incremental_information": "IDENTITY_CONSTANT",
            },
            {
                "field": "C_share_code",
                "value": "002112",
                "type": "SINGLE_FUND_CONSTANT_IDENTITY",
                "content_quotes_in_reports": sum(
                    any(q["field"] == "C_share_code" for q in e["quotes"]) for e in extracts
                ),
                "independent_early_source_available_at": None,
                "admitted": False,
                "incremental_information": "PREVENT_WRONG_SHARE_SUBSTITUTION_NOT_DAILY_MARKET_STATE",
            },
            {
                "field": "fund_age_days",
                "type": "CALENDAR_DATE_MINUS_CONSTANT",
                "development_min": min(ages),
                "development_max": max(ages),
                "distinct_values": len(set(ages)),
                "new_information_beyond_date": False,
                "admitted": False,
                "standalone_predictive_gain_evaluated": False,
            },
        ],
        "C_share_dates": {
            "package_C_added_dates": sorted({s["c_share_added_date"] for s in states}, key=lambda value: value or ""),
            "missing_C_added_date_records": sum(s["c_share_added_date"] is None for s in states),
            "report_16_date_quotes": sum(any(q["field"] == "C_date_16" for q in e["quotes"]) for e in extracts),
            "report_18_subscription_quotes": sum(
                any(q["field"] == "C_subscription_date_18" for q in e["quotes"]) for e in extracts
            ),
            "do_not_conflate": "SHARE_CREATION_METADATA_FIRST_NAV_AND_SUBSCRIPTION_START_ARE_DISTINCT",
        },
        "dynamic": {
            "report_states_reviewed": len(states),
            "C_net_assets_distinct": len({s["financial"]["c_share_net_assets_cny"] for s in states}),
            "manager_rosters_distinct": len({io.digest(s["managers"]) for s in states}),
            "units": increment["semantics"],
            "report_version_times_before2025_proven": 0,
            "manager_notices_eligible2025": sum(e["eligible_for2025"] for e in manager_evidence),
            "type": "REAL_REPORTED_STATE_CHANGES_BUT_HISTORICAL_VERSION_BOUNDARY_BLOCKED",
            "admitted": False,
        },
        "all_static_facts_permanently_nonexistent": False,
    }
    io.save(ROOT / "static-original-evidence.json", extracts)
    io.save(ROOT / "manager-notice-admission.json", manager_evidence)
    io.save(ROOT / "attribute-admission.json", attributes)
    io.save(ROOT / "actual-read-ledger.json", accessed)
    check_time()
    code_copy = ROOT / "code-snapshot" / Path(__file__).name
    code_copy.parent.mkdir(parents=True, exist_ok=True)
    code_copy.write_bytes(Path(__file__).read_bytes())
    result = {
        "at": io.now(),
        "status": "AUDIT_COMPLETE_NO_STRICTLY_ELIGIBLE_NEW_TRAINING_ROUTE",
        "funds_reviewed": len(peers),
        "metadata_constructible_rows": len(sample_rows),
        "strictly_admitted_new_peer_rows": 0,
        "target_report_pdfs_read": len(extracts),
        "manager_notice_pdfs_read": len(manager_evidence),
        "supervised_fits": 0,
        "new_candidates_proposed": 0,
        "proposal_reason": (
            "Known source-version/publication and cohort-selection gaps prevent strict admission; "
            "no relaxed assumptions"
        ),
        "unreviewed": [
            "External peer raw NAV receipts and full dividend logs not bound as standalone files here",
            "Other early independent fund contracts outside specified15PDF scope",
            "All external out-of-registry financial references",
        ],
        "target_answer_definition_changed": False,
        "new2026_answers_read": False,
        "script_sha256": io.sha(Path(__file__)),
        "read_entry_count": len(accessed),
    }
    io.save(ROOT / "admission-results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
