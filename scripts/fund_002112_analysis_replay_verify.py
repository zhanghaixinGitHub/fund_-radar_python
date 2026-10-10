"""对最终回放独立复算：不调用分析、打分函数、数据库或模型服务。"""

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".local-runs/direction-1d-information/20261010-history-replay-v5"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify():
    protocol, summary = read(RUN / "protocol.json"), read(RUN / "summary.json")
    seal = read(RUN / "prediction-seal.json")
    for fields in (protocol["protected_hashes"], protocol["old_artifact_hashes"], protocol["code_hashes"], seal):
        assert all(sha(Path(p)) == expected for p, expected in fields.items())
    source_path = next(Path(p) for p in protocol["protected_hashes"] if p.endswith("sources.json"))
    source = read(source_path)
    rows, audits = [], []
    saved_rows = {r["target"]: r for r in read(RUN / "daily-results.json")}
    public_date_warnings = {}
    for target in protocol["targets"]:
        path = RUN / "inputs" / f"{target}.json"
        assert sha(path) == protocol["input_hashes"][target]
        value = read(path)
        prediction = read(RUN / "predictions" / f"{target}.json")
        cutoff = datetime.fromisoformat(value["as_of"])
        assert cutoff.hour == 8 and cutoff.minute == 30 and cutoff.date().isoformat() == target
        assert cutoff.utcoffset().total_seconds() == 8 * 3600
        base = source["sessions"][source["sessions"].index(target) - 1]
        assert value["window"]["base_nav_date"] == base and value["window"]["target_nav_date"] == target
        assert value["latest_nav_date"] == max(r["nav_date"] for r in value["nav"]) == base
        assert all(r["nav_date"] <= base for r in value["nav"])
        assert datetime.fromisoformat(value["report"]["available_at"]) <= cutoff
        assert datetime.fromisoformat(value["time_proof"]["nav_max_available_at"]) <= cutoff
        assert datetime.fromisoformat(value["time_proof"]["quote_available_at"]) <= cutoff
        assert all(q["date"] <= base for q in value["market"].values())
        assert all(c["quote"] is None or c["quote"]["date"] <= base for c in value["companies"])
        documents = {d["id"]: d for d in value["documents"]}
        for doc in documents.values():
            assert datetime.fromisoformat(doc["available_at"]) <= cutoff
            assert doc["date"] <= target
            url_year = re.search(r"/art/(20\d{2})/", doc.get("url") or "")
            if doc["kind"] == "POLICY" and url_year and url_year[1] < doc["date"][:4]:
                public_date_warnings[doc["id"]] = {
                    "title": doc["title"],
                    "url": doc["url"],
                    "archived_date": doc["date"],
                    "url_year": url_year[1],
                    "finding": "保存标注日期晚于URL年份，不把它当成已核实的近期新政策",
                }
        initial = Decimal(str(source["nav"][base]["unit_nav"]))
        final = Decimal(str(source["nav"][target]["unit_nav"]))
        actual = "UP" if final > initial else "DOWN" if final < initial else "FLAT"
        assert saved_rows[target]["actual"] == actual
        row = {**saved_rows[target], "independent_actual": actual}
        quote_count = 0
        if prediction["state"] == "READY":
            analysis, facts = prediction["analysis"], prediction["evidence"]
            assert row["new"] == analysis["direction"]
            refs = {ref for reason in analysis["reasons"] + analysis["counterpoints"] for ref in reason["refs"]}
            assert refs <= set(facts) and sorted(refs) == analysis["evidence_units"]
            doc_coverage, events, grouped_events = set(), set(), []
            for fact in facts.values():
                for receipt in fact.get("quote_receipts", []):
                    doc = documents[receipt["document_id"]]
                    assert doc["body"][receipt["start"] : receipt["end"]] == receipt["quote"]
                    assert doc["source_hash"] == receipt["source_hash"]
                    assert set(doc["codes"]) == set(fact["company_codes"])
                    assert receipt["quote"] in fact["text"]
                    doc_coverage.add(doc["id"])
                    events.add(receipt["event_id"])
                    quote_count += 1
                for matter in fact.get("matters", []):
                    grouped_events.extend(matter["event_ids"])
                    assert set(matter["company_codes"]) == set(fact["company_codes"])
                    assert {documents[e.rsplit(":", 1)[0]]["kind"] for e in matter["event_ids"]} == {matter["kind"]}
            assert doc_coverage == set(documents)
            assert len(grouped_events) == len(set(grouped_events)) and set(grouped_events) == events
        else:
            assert row["new"] is None
        audits.append(
            {
                "target": target,
                "baseline": base,
                "state": prediction["state"],
                "cutoff_valid": True,
                "exact_quotes_checked": quote_count,
            }
        )
        rows.append(row)

    comparisons = {}
    for cohort in ("regression", "expanded", "all"):
        scope = [r for r in rows if cohort == "all" or r["cohort"] == cohort]
        common = [r for r in scope if r["state"] == "READY"]
        assert len(scope) == summary["cohorts"][cohort]["planned"]
        assert len(common) == summary["cohorts"][cohort]["ready"]
        comparisons[cohort] = {}
        for method in ("new", "A_NAV", "B_MARKET", "C_EVENTS", "always_up", "always_down"):
            right = len([r for r in common if r[method] == r["independent_actual"]])
            metric = summary["cohorts"][cohort]["common_dates"][method]
            assert metric["correct"] == right and metric["judged"] == len(common)
            assert metric["accuracy"] == (right / len(common) if common else None)
            for direction in ("UP", "DOWN", "FLAT"):
                actual_rows = [r for r in common if r["independent_actual"] == direction]
                assert metric["by_actual"][direction] == {
                    "total": len(actual_rows),
                    "correct": sum(r[method] == direction for r in actual_rows),
                }
            comparisons[cohort][method] = {"correct": right, "common": len(common)}

    forbidden = {"actual", "actual_direction", "label", "target_unit_nav", "nav_return", "mature_at"}

    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    calls = list((RUN / "calls").glob("*/request.json"))
    for path in calls:
        request = read(path)
        visit(json.loads(request["payload"]["messages"][1]["content"]))
    total_calls = protocol["diagnostic_calls"] + len(calls)
    assert total_calls <= protocol["total_authorized_budget_including_diagnostic"] == 420
    reviews = [read(p) for p in (RUN / "semantic-review").glob("*.json")]
    result = {
        "verifier_hash": sha(Path(__file__)),
        "planned": len(rows),
        "date_and_quote_audit": audits,
        "independent_decimal_labels_and_metrics": comparisons,
        "input_and_code_hashes_verified": True,
        "old_artifacts_unchanged": len(protocol["old_artifact_hashes"]),
        "prediction_seals_unchanged": True,
        "request_answer_keys_absent": True,
        "calls_final": len(calls),
        "calls_diagnostic": protocol["diagnostic_calls"],
        "calls_total": total_calls,
        "policy_date_warnings": list(public_date_warnings.values()),
        "separate_review_counts": dict(Counter(r["verdict"] for r in reviews)),
        "semantic_limit": "逐字和程序检查不等于语义完全真实；单独模型复核存在相关错误，需对照人工抽查",
    }
    destination = RUN / "independent-verification.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in {"date_and_quote_audit"}}, ensure_ascii=False))


if __name__ == "__main__":
    verify()
