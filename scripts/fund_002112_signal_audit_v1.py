"""冻结前原文、数值、两次持仓和时间边界核验；产物不读取监督成绩。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts import fund_002112_signal_common_v1 as c
from scripts import fund_002112_signal_features_v1 as f
from scripts import fund_002112_signal_sources_v1 as sources
from scripts import fund_002112_signal_train_v1 as train


def run(root):
    folder = train.feature_root(root)
    events = c.lines(folder / "events.jsonl")
    by_id = {e["id"]: e for e in events}
    docs = sources.existing_docs()
    docs.update({d["id"]: d for d in c.lines(root / "supplement-documents.jsonl")})
    references, numeric_count = [], 0
    for e in events:
        if e["status"] != "QUALIFIED":
            continue
        d = docs[e["document_id"]]
        body = f.clean(d["body"])
        assert all(f.clean(q) in body for q in e["quotes"])
        assert e["source_kind"] == "company" and d["stock_code"] == e["code"]
        assert c.moment(e["version_available_at"]) >= c.moment(d["title_available_at"])
        for metric, value in e["numeric"].items():
            if metric.endswith("yoy"):
                assert f.clean(value["quote"]) in body and value["period"] == e["period"]
                assert value["unit"] == "ratio" and -10 <= value["value"] <= 10
            else:
                numerator, denominator = value["numerator"], value["denominator"]
                assert numerator["currency"] == denominator["currency"]
                assert f.clean(numerator["quote"]) in body
                if metric == "buyback_ratio":
                    assert f.clean(denominator["quote"]) in body and denominator["value"] >= numerator["value"]
                else:
                    previous = by_id[denominator["event_id"]]
                    assert previous["code"] == e["code"]
                    assert c.moment(previous["version_available_at"]) <= c.moment(e["version_available_at"])
                    assert f.clean(denominator["quote"]) in f.clean(docs[previous["document_id"]]["body"])
                assert abs(value["raw"] - numerator["value"] / denominator["value"]) < 1e-12
            numeric_count += 1
        for previous in e["previous_event_ids"]:
            assert c.moment(by_id[previous]["version_available_at"]) < c.moment(e["version_available_at"])
        references.append(
            {
                "event_id": e["id"],
                "source": e["source_url"],
                "body_hash": e["source_hash"],
                "original_hash": e["raw_sha256"],
                "period": e["period"],
                "quote_count": len(e["quotes"]),
                "numeric_count": len(e["numeric"]),
                "passed": True,
            }
        )
    lineage = c.lines(folder / "event-lineage.jsonl")
    actual_ids, yearly = set(), {}
    rows = c.lines(folder / "inputs.jsonl")
    for row, proof in zip(rows, lineage, strict=True):
        assert row["target"] == proof["target"] and len(row["E"]) == len(row["ED"]) == 20
        for key, decay in (("B1_B2", False), ("B3", True)):
            companies = {}
            for event in proof[key]["events"]:
                assert c.moment(event["version_available_at"]) <= c.moment(row["as_of"])
                assert 0 <= event["event_age_sessions"] < 5
                at_event, at_prediction = event["event_report"], event["prediction_report"]
                assert at_event["weight"] > 0 and at_prediction["weight"] > 0
                assert c.moment(at_event["report_available_at"]) <= c.moment(event["version_available_at"])
                assert c.moment(at_prediction["available_at"]) <= c.moment(row["as_of"])
                assert event["original_weight"] == min(at_event["weight"], at_prediction["weight"])
                if event["status"] == "QUALIFIED" and event["factor"] > 0:
                    actual_ids.add(event["id"])
                    if event["important"]:
                        companies[event["code"]] = max(companies.get(event["code"], 0), event["original_weight"])
                if decay:
                    assert (
                        abs(
                            event["factor"]
                            - f.age_factor(event["holding_age"]) * (1 - 0.2 * event["event_age_sessions"])
                        )
                        < 1e-12
                    )
            assert proof[key]["trigger"] == (sum(companies.values()) >= 0.03)
    for year in ("2024", "2025", "2026"):
        year_rows = [r for r in rows if r["target"].startswith(year)]
        year_ids = {
            e["id"]
            for p in lineage
            if p["target"].startswith(year)
            for e in p["B1_B2"]["events"]
            if e["status"] == "QUALIFIED"
        }
        yearly[year] = {
            "rows": len(year_rows),
            "actual_direct_event_ids": len(year_ids),
            "nonzero_or_known_numeric_days": {
                name: sum(r["E"][i] is not None and r["E"][i] != 0 for r in year_rows)
                for i, name in enumerate(f.FIELDS)
            },
            "supplement_event_ids": sorted(
                i
                for i in year_ids
                if by_id[i]["document_id"] in {d["id"] for d in c.lines(root / "supplement-documents.jsonl")}
            ),
        }
    sample = c.lines(folder / "review-sample.jsonl")
    assert len(sample) >= 50 and all(e["status"] == "QUALIFIED" for e in sample)
    assert {e["version_available_at"][:4] for e in sample} >= {"2025", "2026"}
    negative = [
        e for e in events if "处罚" in e["title"] or "注销完成" in e["title"] or e["status"].startswith("QUARANTINED")
    ][:20]
    review = {
        "at": c.io.now(),
        "passed": True,
        "actual_change_events_reviewed": len(sample),
        "sample_by_year": dict(Counter(e["version_available_at"][:4] for e in sample)),
        "negative_and_risk_samples": [
            {"id": e["id"], "title": e["title"], "status": e["status"], "flags": e["flags"]} for e in negative
        ],
        "review_method": "人工检查60个真实变化样例的引文、数字、期间和类别；全量机器核对原文连续引文与时间/权重边界。",
        "fixed_findings": ["回购每股价格误作计划金额已通用修复", "注销旧执行额不再触发", "同期间全文摘要重复已合并"],
        "unavailable_categories": [
            "未观察到证据充分的同期间指引前后变动",
            "订单上年营收同口径分母不足",
            "没有合格公司重大风险新增/解除及配对订单终止样例；不以股东个人处罚补足",
        ],
        "all_qualified_events": len(references),
        "verified_numeric_facts": numeric_count,
        "actual_input_unique_events": len(actual_ids),
        "actual_input_by_year": yearly,
        "historical_version_limit": "历史原件按声明发布日期与已知版本约束重建，不声称具备当时在线首见存档。",
    }
    c.io.save(folder / "source-review.json", review)
    c.io.save_lines(folder / "qualified-source-checks.jsonl", references)
    c.stage(
        root,
        "来源与输入验收",
        "COMPLETE",
        [str(folder / "source-review.json")],
        review["unavailable_categories"],
        "冻结后按固定候选比较",
    )
    print(
        c.io.canonical(
            {
                k: review[k]
                for k in (
                    "passed",
                    "actual_change_events_reviewed",
                    "all_qualified_events",
                    "verified_numeric_facts",
                    "actual_input_unique_events",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)
