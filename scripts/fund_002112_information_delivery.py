"""汇总本轮停止决定、同日配对变化和事实关联，不拟合、不写历史预测。"""

import json
from collections import Counter
from datetime import date, timedelta

from app.services.fund_002112_information_admission import BASE, PREVIOUS, RUN, save_once
from app.services.fund_002112_information_experiment import HYPOTHESES, QUARTERS, status
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, quarter, read_json


def main():
    paths = [RUN / h / "historical-decision.json" for h in HYPOTHESES]
    results = [read_json(p) for p in paths]
    before, after = [], []
    for h, output in zip(HYPOTHESES, (before, after), strict=True):
        for q in QUARTERS:
            output.extend(read_json(RUN / h / "predictions" / (q + "-CANDIDATE-main.json"))["exam"])
    rows2 = {(r["fund_code"], r["target"]): r for r in read_json(RUN / "strategy-admission-v1/inputs.json")["train"]}
    daily = []
    for a, b in zip(before, after, strict=True):
        if (a["fund_code"], a["target"], a["actual_direction"]) != (b["fund_code"], b["target"], b["actual_direction"]):
            raise ValueError("PAIR_IDENTITY_CHANGED")
        row = rows2[b["fund_code"], b["target"]]
        if row["base_input_sha256"] != a["input_hash"] or digest(row) != b["input_hash"]:
            raise ValueError("PAIR_BASE_NUMERIC_INPUT_CHANGED")
        daily.append(
            {
                "target": a["target"],
                "actual": a["actual_direction"],
                "H1": a["direction"],
                "H2": b["direction"],
                "delta": int(b["direction"] == b["actual_direction"]) - int(a["direction"] == a["actual_direction"]),
            }
        )

    def counts(rows):
        values = [r["delta"] for r in rows]
        return {"gained": values.count(1), "lost": values.count(-1), "net": sum(values)}

    pair = {
        **counts(daily),
        "daily": daily,
        "classes": {c: counts([r for r in daily if r["actual"] == c]) for c in ("DOWN", "FLAT", "UP")},
        "quarters": {q: counts([r for r in daily if quarter(r["target"]) == q]) for q in QUARTERS},
        "same_base_pool_and_twenty_features": True,
        "evidence": "DEVELOPMENT_VALIDATION",
    }
    save_once(RUN / "h2-versus-h1.json", pair)
    manifest = read_json(RUN / "event-admission/bundle-manifest.json")
    bundle = read_json(manifest["path"])
    reports = {r["raw"]["sha256"]: r for r in read_json(ROOT / "style-training-coverage/20260928-v1/reports.json")}
    tags = {
        r["report_sha256"]: r["tag"] for r in read_json(ROOT / "style-training-coverage/20260928-v1/reviewed-tags.json")
    }
    q2 = next(f for f in read_json(BASE / "folds.json") if f["name"] == "2023Q2")
    timeline = []
    for row in q2["exam"]:
        start = (date.fromisoformat(row["target"]) - timedelta(days=30)).isoformat()
        report = reports[row["report_sha256"]]
        stocks = {h["stock_code"].split(".")[0]: h["nav_weight_pct"] for h in report["holdings"]}
        associations = []
        for fact in bundle["facts"]:
            if not start <= fact["published_date"] < row["target"]:
                continue
            if fact["entity_type"] == "COMPANY" and fact["entity_id"] in stocks:
                reason = "PRIOR_PUBLIC_REPORTED_HOLDING_NOT_DAILY_POSITION"
                weight = stocks[fact["entity_id"]]
            elif fact["category"] in ("NEWS", "INDUSTRY_POLICY") and tags[row["report_sha256"]] == "MEDICINE_FOCUS":
                reason, weight = "PRIOR_PUBLIC_STRATEGY_TOPIC_MEDICINE", None
            else:
                continue
            associations.append(
                {
                    "fact_identity": fact["identity"],
                    "category": fact["category"],
                    "title": fact["title"],
                    "published_date": fact["published_date"],
                    "relation_basis": reason,
                    "reported_nav_weight_pct": weight,
                    "revision_unresolved": fact["revision_unresolved"],
                }
            )
        timeline.append(
            {
                "target": row["target"],
                "report_sha256": row["report_sha256"],
                "report_publication": row["report_publication"],
                "facts": associations,
                "meaning": "Known facts for review; not a historical prediction or verified trading impact",
            }
        )
    save_once(RUN / "q2-known-information.json", timeline)
    protected = read_json(PREVIOUS / "all-protected-sources.json")
    protected = protected.get("files", protected)
    for path, sha in protected.items():
        if file_hash(path) != sha:
            raise ValueError("PREVIOUS_SOURCE_CHANGED")
    for path, sha in bundle["sources"].items():
        if file_hash(path) != sha:
            raise ValueError("NEW_FACT_SOURCE_CHANGED")
    save_once(
        RUN / "protection-final.json",
        {
            "passed": True,
            "previous_protected_files": len(protected),
            "fact_source_files": len(bundle["sources"]),
            "scope": "Exact frozen manifests only",
        },
    )
    final = {
        "status": "STOP_BOTH_REGISTERED_HYPOTHESES_FAILED",
        **status(),
        "adopted": False,
        "production_used": False,
        "full_executed": False,
        "failed_fit_calls": 0,
        "completed_fit_calls": 12,
        "H1": {
            "correct": results[0]["models"]["CANDIDATE"]["correct"],
            "vs_L20": {k: results[0]["pairs"]["L20_ORIGINAL"][k] for k in ("gained", "lost", "net")},
            "failed_checks": results[0]["failed_checks"],
        },
        "H2_strategy_only": {
            "correct": results[1]["models"]["CANDIDATE"]["correct"],
            "vs_L20": {k: results[1]["pairs"]["L20_ORIGINAL"][k] for k in ("gained", "lost", "net")},
            "vs_H1": counts(daily),
            "failed_checks": results[1]["failed_checks"],
        },
        "facts_persisted": len(bundle["facts"]),
        "fact_types": dict(Counter(r["category"] for r in bundle["facts"])),
        "quarter2_days_with_associated_facts": sum(bool(r["facts"]) for r in timeline),
        "unused_budget_is_not_authorization_for_unregistered_hypotheses": True,
        "news_policy_company_predictive_gain_tested": False,
        "strategy_predictive_gain_tested": True,
        "evidence_type": "2023_AND_2024_OBSERVED_DEVELOPMENT_DATA; FULL_NOT_RUN",
    }
    save_once(RUN / "final-decision.json", final)
    deliverables = [
        RUN / "protocol.json",
        RUN / "budget-authorized.json",
        RUN / "final-decision.json",
        RUN / "h2-versus-h1.json",
        RUN / "q2-known-information.json",
        RUN / "protection-final.json",
        RUN / "early-admission/decision.json",
        RUN / "early-admission/independent-audit-v2.json",
        RUN / "event-admission/decision.json",
        RUN / "event-admission/bundle-manifest.json",
        RUN / "event-admission/company-gap-worklist.json",
        RUN / "strategy-admission-v1/feature-contract.json",
        RUN / "strategy-admission-v1/independent-audit-v2.json",
    ]
    deliverables += paths
    deliverables += list((RUN / "fit-ledger").glob("*.json"))
    save_once(RUN / "delivery-manifest.json", {"files": {str(p): file_hash(p) for p in deliverables}})
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
