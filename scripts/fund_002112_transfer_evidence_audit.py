"""独立复核参考基金关联统计；不调用主分析统计函数，不拟合或访问网络。"""

import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / ".local-runs/fund-exposure-002112"
OUT = ROOT / "peer-transfer-evidence/20260928-v1"
NEW = ROOT / "peer-recovered-experiment/20260928-v1"


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return value["payload"] if isinstance(value, dict) and set(value) == {"hash", "payload"} else value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = read(OUT / "protocol.json")
    for path, expected in protocol["source_files"].items():
        assert sha(path) == expected, path
    inputs, folds = read(NEW / "frozen/inputs.json"), read(NEW / "frozen/folds.json")
    own = [r for r in inputs["train"] if r["fund_code"] == "002112"]
    historical = [r for f in folds[:3] for r in f["exam"]]
    development = inputs["development"]
    groups = {
        "own_train_before_2023": [r for r in own if r["target"] < "2023-01-01"],
        "own_train_2023_complete": [r for r in own if r["target"][:4] == "2023"],
        "historical_161": historical,
        "development_2024_230": development,
        **{f["name"]: f["exam"] for f in folds[:3]},
        **{
            f"2024Q{q}": [r for r in development if int(r["target"][5:7]) in range(q * 3 - 2, q * 3 + 1)]
            for q in range(1, 5)
        },
    }
    expected = read(OUT / "feature-evidence.json")
    preds = {r["target"]: r for r in read(ROOT / "peer-mechanism-review/20260928-v1/daily.json")}
    maximum_error, rank_cells, sign_cells = 0.0, 0, 0
    for name, rows in groups.items():
        assert len(rows) == expected[name]["rows"]
        assert dict(Counter(r["actual_direction"] for r in rows)) == expected[name]["actual"]
        assert len({r["target"] for r in rows}) == len(rows)
        assert all(r["target"] < "2025-01-01" and r["report_publication"] < r["target"] for r in rows)
        for i, feature in enumerate(protocol["features"]):
            result = expected[name]["features"][feature]
            up = [r["x"][i] for r in rows if r["actual_direction"] == "UP"]
            down = [r["x"][i] for r in rows if r["actual_direction"] == "DOWN"]
            # 显式逐对比较，与主分析的排序查找算法独立。
            if up and down:
                differences = np.asarray(up)[:, None] - np.asarray(down)[None, :]
                score = float(((differences > 0).sum() + (differences == 0).sum() * 0.5) / differences.size)
                error = abs(score - result["up_down_rank_score"])
                assert error < 1e-14
                maximum_error = max(error, maximum_error)
            else:
                assert result["up_down_rank_score"] is None
            rank_cells += 1
            for c in ("DOWN", "FLAT", "UP"):
                values = [r["x"][i] for r in rows if r["actual_direction"] == c]
                wanted = result["by_actual_class"][c]
                assert wanted["n"] == len(values)
                assert wanted["median"] == (median(values) if values else None)
            if feature in protocol["statistics"]["sign_features"]:
                for bucket in ("negative", "zero", "positive"):
                    selected = [
                        r
                        for r in rows
                        if (
                            r["x"][i] < 0
                            if bucket == "negative"
                            else r["x"][i] > 0
                            if bucket == "positive"
                            else r["x"][i] == 0
                        )
                    ]
                    wanted = result["fixed_zero_groups"][bucket]
                    assert wanted["rows"] == len(selected)
                    assert wanted["actual"] == dict(Counter(r["actual_direction"] for r in selected))
                    if "saved_L20_recovered" in wanted:
                        assert name == "historical_161" or name.startswith("2023Q")
                        assert wanted["saved_L20_recovered"]["actual_UP_predicted_DOWN"] == sum(
                            r["actual_direction"] == "UP" and preds[r["target"]]["new_direction"] == "DOWN"
                            for r in selected
                        )
                        assert wanted["saved_L20_recovered"]["correct"] == sum(
                            preds[r["target"]]["new_direction"] == r["actual_direction"] for r in selected
                        )
                    sign_cells += 1
    own_index = {(r["target"], r["base"]): r for r in own}
    pairs = read(OUT / "paired-returns.json")
    pair_count = 0
    for code, wanted in pairs.items():
        selected = [r for r in inputs["train"] if r["fund_code"] == code]
        assert len(selected) == wanted["peer_rows"]
        matched, missing = [], []
        for peer in selected:
            target = own_index.get((peer["target"], peer["base"]))
            if target is None:
                missing.append({"target": peer["target"], "base": peer["base"]})
                continue
            matched.append(
                {
                    "target": peer["target"],
                    "base": peer["base"],
                    "own_return": float(
                        (Decimal(target["target_unit_nav"]) - Decimal(target["base_unit_nav"]))
                        / Decimal(target["base_unit_nav"])
                    ),
                    "peer_return": float(
                        (Decimal(peer["target_unit_nav"]) - Decimal(peer["base_unit_nav"]))
                        / Decimal(peer["base_unit_nav"])
                    ),
                    "own_actual": target["actual_direction"],
                    "peer_actual": peer["actual_direction"],
                }
            )
        assert missing == wanted["missing_target_dates"]
        assert matched == wanted["daily"]
        for name, summary in wanted["groups"].items():
            sample = matched if name == "all" else [r for r in matched if r["target"][:4] == name]
            assert len(sample) == summary["matched"]
            if sample:
                x, y = [r["own_return"] for r in sample], [r["peer_return"] for r in sample]
                ax, ay = sum(x) / len(x), sum(y) / len(y)
                numerator = sum((a - ax) * (b - ay) for a, b in zip(x, y, strict=True))
                denominator = math.sqrt(sum((a - ax) ** 2 for a in x) * sum((b - ay) ** 2 for b in y))
                score = numerator / denominator if denominator else None
                assert score is None or abs(score - summary["pearson_same_day_return"]) < 1e-12
            else:
                assert summary["pearson_same_day_return"] is None
        pair_count += len(matched)
    context = read(OUT / "investment-context.json")
    source_context = read(OUT / "report-context-protocol.json")
    for spec in source_context["reports"]:
        assert sha(spec["path"]) == spec["sha256"]
    all_rows = {r["target"]: r for r in historical + development}
    assert len(context["daily"]) == len(all_rows) == 391
    by_report = {r["source"]["sha256"]: r for r in context["reports"]}
    for r in context["daily"]:
        original = all_rows[r["target"]]
        assert r["report_sha256"] == original["report_sha256"]
        assert r["report_publication"] == original["report_publication"] < r["target"]
        assert r["context"] == by_report[r["report_sha256"]]["descriptive_context"]
        if r["context"] == "AI_COMPUTE":
            assert r["target"] >= "2024-04-01"
        assert "direction" not in r and "prediction" not in r
    for name in ("historical_161", "development_2024_230"):
        assert context["groups"][name] == dict(Counter(r["context"] for r in context["daily"] if r["group"] == name))
    result = {
        "passed": True,
        "protocol_sources_verified": len(protocol["source_files"]),
        "feature_groups": len(groups),
        "rank_cells_verified": rank_cells,
        "sign_groups_verified": sign_cells,
        "max_rank_difference": maximum_error,
        "exact_paired_days_verified": pair_count,
        "report_time_mappings_verified": 391,
        "report_context_source_files_verified": len(source_context["reports"]),
        "new_fits": 0,
        "cumulative_fits": 58,
        "script_sha256": sha(__file__),
        "evidence_sha256": {
            p.name: sha(p)
            for p in OUT.glob("*.json")
            if p.name
            in (
                "summary.json",
                "feature-evidence.json",
                "paired-returns.json",
                "investment-context.json",
                "report-allocations.json",
            )
        },
    }
    with (OUT / "independent-audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
