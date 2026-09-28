"""独立计数审计：仅用标准库从逐日预测重算，不调用模型或报告模块。"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def read(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = value["payload"]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(raw.encode()).hexdigest() == value["hash"], path
    return payload


def count(rows):
    classes = ("DOWN", "FLAT", "UP")
    table = {a: dict.fromkeys(classes, 0) for a in classes}
    for r in rows:
        table[r["actual_direction"]][r["direction"]] += 1
    return {
        "days": len(rows),
        "correct": sum(table[k][k] for k in classes),
        "class_correct": {k: table[k][k] for k in classes},
        "actual": {k: sum(table[k].values()) for k in classes},
        "confusion": table,
    }


def audit(p):
    """核对总数、类别、季度、逐日新增/丢失；主/复现文件各自检查方向与分数。"""
    comparisons = read(p / "comparison.json")
    dates = {f["name"]: f["expected_dates"] for f in read(p / "folds.json")}
    evidence, daily = {}, {}
    for phase, stages in (("historical", ("2023Q2", "2023Q3", "2023Q4")), ("development", ("FULL",))):
        c = comparisons[phase]
        if c is None:
            assert not list((p / "attempts").glob("FULL-*.json"))
            evidence[phase] = "NOT_RUN_NO_SCORE"
            continue
        predictions = {}
        for variant in ("N7", "L20", "T20"):
            predictions[variant] = []
            for stage in stages:
                main = read(p / "predictions" / f"{stage}-{variant}-main.json")["exam"]
                replay = read(p / "predictions" / f"{stage}-{variant}-replay.json")["exam"]
                assert [r["target"] for r in main] == dates[stage]
                assert len(main) == len(replay)
                for a, b in zip(main, replay, strict=True):
                    assert {k: v for k, v in a.items() if k != "scores"} == {
                        k: v for k, v in b.items() if k != "scores"
                    }
                    assert all(math.isfinite(v) for v in a["scores"] + b["scores"])
                    assert max(abs(x - y) for x, y in zip(a["scores"], b["scores"], strict=True)) <= 1e-12
                    scores = dict(zip(("DOWN", "FLAT", "UP"), a["scores"], strict=True))
                    assert a["direction"] == max(("FLAT", "UP", "DOWN"), key=lambda k: scores[k])
                predictions[variant].extend(main)
            assert count(predictions[variant]) == c["models"][variant]
        for quarter, summaries in c["quarters"].items():
            for variant, rows in predictions.items():
                selected = [r for r in rows if r["target"][:4] + "Q" + str((int(r["target"][5:7]) + 2) // 3) == quarter]
                assert count(selected) == summaries[variant]
        for direction, value in c["constants"].items():
            assert count([{**r, "direction": direction} for r in predictions["L20"]]) == value
        for baseline in ("N7", "L20"):
            gained = lost = 0
            expected_daily = []
            for t, b in zip(predictions["T20"], predictions[baseline], strict=True):
                assert (t["target"], t["actual_direction"], t["input_hash"]) == (
                    b["target"],
                    b["actual_direction"],
                    b["input_hash"],
                )
                tc, bc = t["direction"] == t["actual_direction"], b["direction"] == b["actual_direction"]
                gained += int(tc and not bc)
                lost += int(bc and not tc)
                expected_daily.append(
                    {
                        "target": t["target"],
                        "actual": t["actual_direction"],
                        "T20": t["direction"],
                        baseline: b["direction"],
                        "outcome": "both_correct" if tc and bc else "gained" if tc else "lost" if bc else "both_wrong",
                    }
                )
            pair = c["pairs"][baseline]
            assert (gained, lost, gained - lost) == (pair["gained"], pair["lost"], pair["net"])
            assert expected_daily == pair["daily"]
        daily[phase] = [
            {
                "target": rows[0]["target"],
                "actual": rows[0]["actual_direction"],
                **{v: row["direction"] for v, row in zip(("N7", "L20", "T20"), rows, strict=True)},
            }
            for rows in zip(*(predictions[v] for v in ("N7", "L20", "T20")), strict=True)
        ]
        evidence[phase] = {
            "passed": True,
            "days": len(daily[phase]),
            "models": c["models"],
            "pairs": {k: {name: value for name, value in v.items() if name != "daily"} for k, v in c["pairs"].items()},
        }
    return {"independent_standard_library_count": True, "phases": evidence, "new_fits": 0}, daily


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"002112-r3-[0-9a-f]{24}", args.run_id):
        parser.error("运行身份不合法")
    p = Path(__file__).resolve().parents[1] / ".local-runs/fund-exposure-002112/round3-runs" / args.run_id
    evidence, daily = audit(p)
    for name, value in (("independent-audit.json", evidence), ("daily-comparison.json", daily)):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        path = p / name
        if path.exists():
            assert path.read_text(encoding="utf-8") == raw
        else:
            path.write_text(raw, encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
