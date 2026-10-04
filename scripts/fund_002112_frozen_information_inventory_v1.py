"""只盘点冻结清单中的剩余信息，区分已存在未审查和已有证据阻断，不新增数据。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts import fund_002112_morning_coverage_revision_v1 as revision

io, ROOT = revision.io, revision.ROOT


def inventory():
    if (ROOT / "remaining-information-inventory.json").exists():
        return io.read(ROOT / "remaining-information-inventory.json")
    timing = revision.previous.prior.source.timing
    frozen = io.Frozen(timing.event.OLD)
    used = {}

    def entry(key):
        ref = frozen.records[frozen.inventory["entries"][key]]
        path = frozen.root / ref["snapshot_path"]
        assert io.sha(path) == ref["sha256"]
        used[str(path)] = ref["sha256"]
        return frozen.entry(key), {"original_path": ref["source_path"], "snapshot": str(path), "sha256": ref["sha256"]}

    facts, fact_ref = entry("historical_facts")
    facts = io.payload(facts)
    peers = []
    for code, fund in facts["funds"].items():
        nav = fund["nav"]["rows"]
        dates = [v["date"] for v in nav]
        # 这里仅盘点日期/公告日期及行数，不输出净值，不生成其他基金涨跌标签。
        assert max(dates) <= "2025-12-31"
        reports = fund["reports"]
        peers.append(
            {
                "fund_code": code,
                "source": fact_ref,
                "json_pointer": f"payload.funds.{code}",
                "nav_rows": len(nav),
                "nav_date_min": min(dates),
                "nav_date_max": max(dates),
                "nav_2025_rows": sum(d.startswith("2025") for d in dates),
                "nav_rows_missing_ann_date": sum(not r.get("ann_date") for r in nav),
                "nav_basis": "FUND_OWN_SHARE_UNIT_NAV_WITH_ANN_DATE_METADATA",
                "report_count": len(reports),
                "report_end_min": min(r["report_end"] for r in reports),
                "report_end_max": max(r["report_end"] for r in reports),
                "report_available_min": min(r["available_at"] for r in reports),
                "report_available_max": max(r["available_at"] for r in reports),
                "recorded_report_eligibility": dict(
                    Counter(str(r.get("training_eligible", "NOT_RECORDED")) for r in reports)
                ),
                "family": fund.get("family"),
                "cohort_compatibility": fund.get("cohort", {}).get("compatibility"),
                "cohort_evidence_count": len(fund.get("cohort", {}).get("evidence", [])),
                "evaluated_in_current_five_stages": "TARGET_NAV_ONLY"
                if code == "002112"
                else "NO_PEER_TRANSFER_EVALUATION",
                "status": "PRESENT_NOT_FULLY_REVIEWED_FOR_CROSS_FUND_USE",
                "missing_evidence": [
                    "POINT_IN_TIME_COHORT_SELECTION",
                    "SHARE_AND_DIVIDEND_TARGET_COMPATIBILITY",
                    "COMMON_FEATURE_CONTRACT",
                    "SAME_VERSION_HISTORICAL_PUBLICATION",
                ],
                "immediate_2025_peer_input_block": "NO_2025_NAV_IN_THIS_FROZEN_PACKAGE",
                "transfer_training_before2025": "POTENTIALLY_REVIEWABLE_NOT_YET_ADMITTED_OR_TESTED_HERE",
            }
        )
    history, history_ref = entry("historical")
    financial = []
    for row in history["rows"]:
        refs = {}
        for key in ("source_record", "literal_evidence", "forecast_evidence", "forecast_literal_evidence"):
            value = row.get(key)
            if value:
                bound = frozen.records.get(str(Path(value["path"]).resolve()))
                refs[key] = {
                    **value,
                    "in_frozen_registry": bound is not None,
                    "snapshot": str(frozen.root / bound["snapshot_path"]) if bound else None,
                    "this_stage_check": "FROZEN_REGISTRY_REFERENCE_ONLY_NOT_NEW_SEMANTIC_REVIEW",
                }
        financial.append(
            {
                "id": row["id"],
                "company": row["company"],
                "published_date": row["published_date"],
                "kind": row.get("kind"),
                "available_at": row.get("available_at"),
                "references": refs,
                "unresolved": row.get("unresolved"),
                "old_version_issues": row.get("old_version_issues"),
                "first_seen_proven": row.get("historical_first_seen_archive_proven"),
                "whole_version_chain_verified": row.get("whole_event_version_chain_verified"),
                "recorded_training_eligible": row.get("training_eligible"),
            }
        )
    io.save_lines(ROOT / "financial-frozen-entries.jsonl", financial)
    supplement_path = timing.ROOT / "fund-attribute-time-supplement.json"
    supplement = io.read(supplement_path)
    increment_path = timing.old.DEFAULT_ROOT / "handoff/data/revisions/v2/source-increment.json"
    assert io.sha(increment_path) == supplement["source_increment_sha256"]
    increment = io.read(increment_path)
    for p in (
        supplement_path,
        increment_path,
        timing.ROOT / "report-time-evidence.json",
        timing.ROOT / "feature-use-inventory.json",
    ):
        used[str(p)] = io.sha(p)
    evidence = io.read(timing.ROOT / "report-time-evidence.json")
    for item in evidence:
        assert io.sha(Path(item["raw_path"])) == item["raw_sha256"]
        used[item["raw_path"]] = item["raw_sha256"]
    columns = io.read(timing.ROOT / "feature-use-inventory.json")["F"]["columns"]
    attributes = {
        "source_increment": {"path": str(increment_path), "sha256": io.sha(increment_path)},
        "source_increment_keys": list(increment),
        "registered_fields": columns,
        "semantics": increment.get("semantics"),
        "report_version_evidence": [
            {
                k: e[k]
                for k in ("raw_path", "raw_sha256", "report_end", "declared_publication", "conservative_version_bound")
            }
            for e in evidence
        ],
        "manager_notice_evidence": supplement["manager_notices"],
        "manager_notices_eligible_by_2025": supplement["manager_notices_eligible_by_2025"],
        "evaluated": "F10_INCLUDED_IN_EARLY_SEARCH_THEN_QUARANTINED_AFTER_SOURCE_VERSION_REVIEW",
        "status": "REPORT_STATE_AND_MANAGER_ROSTER_BLOCKED_BY_EXISTING_TIME_EVIDENCE",
        "fund_age_separate_status": (
            "CONTRACT_DATE2015_06_19_RECORDED; "
            "INDEPENDENT_EARLY_SOURCE_PROOF_AND_INCREMENTAL_VALUE_NOT_SEPARATELY_EVALUATED"
        ),
        "do_not_infer": "Group quarantine does not prove every static fund fact permanently unusable",
    }
    domains = {
        "financial_company_events": {
            "source": history_ref,
            "rows": len(financial),
            "date_min": min(r["published_date"] for r in financial),
            "date_max": max(r["published_date"] for r in financial),
            "kinds": dict(Counter(str(r["kind"]) for r in financial)),
            "recorded_training_eligible": dict(Counter(str(r["recorded_training_eligible"]) for r in financial)),
            "basis": "ANNOUNCEMENT_AND_LITERAL_FACTS; PERIOD_CURRENCY_UNIT_COMPARATIVES_REQUIRE_SOURCE_REVIEW",
            "evaluated": "EVENT_BRANCH_TESTED_EARLIER; NO_VALID_2025_BODY_VARIATION; NO_PROVEN_CAUSAL_PRICE_EFFECT",
            "status": "EXISTING_EVIDENCE_BLOCKS_UNREVIEWED_NUMERIC_AND_VERSION_ADMISSION",
            "remaining_review": (
                "Individual period/currency/unit/correction fields and collection scope not re-admitted"
            ),
        },
        "peer_funds": {
            "source": fact_ref,
            "funds_including_target": len(peers),
            "peer_funds": len(peers) - 1,
            "status": "PRESENT_NOT_FULLY_REVIEWED",
            "details": peers,
            "no_claim_about_unbound_earlier_peer_experiments": True,
        },
        "fund_attributes": attributes,
        "independent_events": {
            "entry": str(ROOT / "coverage-revision.json"),
            "status": "D0800_REFERENCE_COVERAGE_CORRECTED_BUT_TIME_VERSION_ADMISSION_NOT_PASSED",
            "next7fits": "NOT_AUTHORIZED_NOT_RUN_NO_RELAXED_GATE",
        },
        "full_market_breadth": {
            "entry": str(timing.io.RESEARCH / "price-volume-development/20261001-v1/breadth-admission.json"),
            "status": "COMPLETE_CONTEMPORANEOUS_UNIVERSE_UNPROVEN_NO_NEW_EVIDENCE",
        },
    }
    outcome = {
        "at": io.now(),
        "domains": domains,
        "source_files": used,
        "source_registry": {
            "path": str(frozen.root / "source-inventory.json"),
            "sha256": io.sha(frozen.root / "source-inventory.json"),
        },
        "later_files_absorbed": False,
        "new_collection": False,
        "supervised_fits": 0,
        "new_llm_requests": 0,
        "script_sha256": io.sha(Path(__file__)),
    }
    io.save(ROOT / "remaining-information-inventory.json", outcome)
    return {
        "financial_rows": len(financial),
        "peer_funds": len(peers) - 1,
        "attributes": len(columns),
        "supervised_fits": 0,
    }


if __name__ == "__main__":
    print(json.dumps(inventory(), ensure_ascii=False, indent=2))
