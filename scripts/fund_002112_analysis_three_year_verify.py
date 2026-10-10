"""三年回放独立复核：仅标准库，重算原始净值标签、分母和引用，不调用分析服务。"""

import hashlib
import json
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".local-runs/direction-1d-information/20261010-three-year-replay-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify():
    protocol, summary = read(OUT / "protocol.json"), read(OUT / "summary.json")
    seal = read(OUT / "prediction-seal.json")
    for mapping in [protocol["protected_hashes"], protocol["old_artifact_hashes"], protocol["code_hashes"], seal]:
        assert all(sha(Path(p)) == expected for p, expected in mapping.items())
    source = read(next(Path(p) for p in protocol["protected_hashes"] if p.endswith("sources.json")))
    rows = read(OUT / "daily-results.json")
    assert [r["target"] for r in rows] == [t for t in source["sessions"] if protocol["start"] <= t <= protocol["end"]]
    counts = Counter()
    quote_count = 0
    for row in rows:
        target = row["target"]
        p = read(OUT / "predictions" / f"{target}.json")
        base = source["sessions"][source["sessions"].index(target) - 1]
        nav = source["nav"]
        if base in nav and target in nav:
            change = Decimal(str(nav[target]["unit_nav"])) - Decimal(str(nav[base]["unit_nav"]))
            assert row["actual"] == ("UP" if change > 0 else "DOWN" if change < 0 else "FLAT")
        else:
            assert row["actual"] is None
        assert row["state"] == p["state"]
        counts[p["state"]] += 1
        if target in protocol["input_hashes"]:
            path = OUT / "inputs" / f"{target}.json"
            assert sha(path) == protocol["input_hashes"][target]
            value = read(path)
            cutoff = datetime.fromisoformat(value["as_of"])
            assert cutoff.hour == 8 and cutoff.minute == 30 and cutoff.utcoffset().total_seconds() == 28800
            assert value["window"]["base_nav_date"] == base and value["window"]["target_nav_date"] == target
            assert datetime.fromisoformat(value["report"]["available_at"]) <= cutoff
            assert datetime.fromisoformat(value["time_proof"]["nav_max_available_at"]) <= cutoff
            assert all(n["nav_date"] <= base for n in value["nav"])
            assert all(c["quote"] is None or c["quote"]["date"] <= base for c in value["companies"])
            docs = {d["id"]: d for d in value["documents"]}
            assert all(
                datetime.fromisoformat(d["available_at"]) <= cutoff and d["date"] <= target for d in docs.values()
            )
        if p["state"] == "READY":
            assert row["new"] == p["analysis"]["direction"] in {"UP", "DOWN"}
            refs = {k for reason in p["analysis"]["reasons"] + p["analysis"]["counterpoints"] for k in reason["refs"]}
            assert sorted(refs) == p["analysis"]["evidence_units"] and refs <= p["evidence"].keys()
            doc_ids, event_ids, partition = set(), set(), []
            for fact in p["evidence"].values():
                for q in fact.get("quote_receipts", []):
                    d = docs[q["document_id"]]
                    assert d["body"][q["start"] : q["end"]] == q["quote"] and d["source_hash"] == q["source_hash"]
                    assert set(d["codes"]) == set(fact["company_codes"])
                    doc_ids.add(d["id"])
                    event_ids.add(q["event_id"])
                    quote_count += 1
                for matter in fact.get("matters", []):
                    partition.extend(matter["event_ids"])
                    assert set(matter["company_codes"]) == set(fact["company_codes"])
            assert doc_ids == set(docs) and len(partition) == len(set(partition)) and set(partition) == event_ids
        else:
            assert row["new"] is None
        if row["reused"]:
            old = read(Path(p["reused_from"]) / "predictions" / f"{target}.json")
            assert (
                p["state"] == old["state"]
                and p.get("analysis") == old.get("analysis")
                and p.get("error") == old.get("error")
            )

    for name, saved in summary["cohorts"].items():
        subset = [
            r
            for r in rows
            if name == "all"
            or r["year"] == name
            or (name == "new_dates" and r["target"] in protocol["new_targets"])
            or (name == "reused_dates" and r["reused"])
        ]
        assert saved["planned"] == len(subset) and saved["states"] == dict(Counter(r["state"] for r in subset))
        successful = [r for r in subset if r["new"] is not None]
        common = [r for r in successful if all(r[k] is not None for k in ("A_NAV", "B_MARKET", "C_EVENTS"))]
        for field, scope in [("all_dates", subset), ("successful_dates", successful), ("common_old_dates", common)]:
            for method, metric in saved[field].items():
                judged = [r for r in scope if r[method] is not None and r["actual"] is not None]
                correct = sum(r[method] == r["actual"] for r in judged)
                assert metric["correct"] == correct and metric["judged"] == len(judged)
                assert metric["accuracy"] == (correct / len(judged) if judged else None)
                for direction in ("UP", "DOWN", "FLAT"):
                    matching = [r for r in judged if r["actual"] == direction]
                    assert metric["by_actual"][direction] == {
                        "total": len(matching),
                        "correct": sum(r[method] == direction for r in matching),
                    }
    forbidden = {"actual", "actual_direction", "label", "target_unit_nav", "nav_return", "mature_at"}

    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    calls = list((OUT / "calls").glob("*/*/request.json"))
    by_year = Counter()
    for path in calls:
        request = read(path)
        by_year[request["year"]] += 1
        visit(json.loads(request["payload"]["messages"][1]["content"]))
    assert len(calls) == summary["calls"] <= protocol["max_calls"] == 1800
    assert all(n <= protocol["year_limits"][y] for y, n in by_year.items())
    result = {
        "verifier_hash": sha(Path(__file__)),
        "planned": len(rows),
        "states": dict(counts),
        "calls": len(calls),
        "calls_by_year": dict(by_year),
        "old_files_unchanged": len(protocol["old_artifact_hashes"]),
        "quote_count": quote_count,
        "all_predictions_sealed": True,
        "independent_scores_checked": True,
        "request_answer_keys_absent": True,
        "input_time_checked": True,
        "semantics": "引文和时间通过不证明解释全部真实，另见逐日复核和人工抽查",
    }
    with (OUT / "independent-verification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    verify()
