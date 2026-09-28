"""002112 参考资料可迁移性检查：固定输入的描述统计，不拟合或生成预测。

同日基金收益的相关性只能说明共同波动，不能当作预测前已知信息。
单项输入的排序分数也不等于模型正确率；全部结果保留，不按表现挑指标。
"""

from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

from app.services.fund_002112_mechanism_review import no_fits
from app.services.fund_002112_round3_data import FEATURES
from app.services.fund_002112_zero_fit_review import file_hash, read_json, save_once, validate_row

ROOT = Path(__file__).resolve().parents[2] / ".local-runs/fund-exposure-002112"
OUT = ROOT / "peer-transfer-evidence/20260928-v1"
NEW = ROOT / "peer-recovered-experiment/20260928-v1"
PREVIOUS = ROOT / "peer-mechanism-review/20260928-v1"
SNAPSHOT = ROOT / "training-ready/sources/98fb1c51f2582687a09ac0c404d514c2ef74520e4cb63a9a9f89a300ef150611.json"
CLASSES = ("DOWN", "FLAT", "UP")


def descriptive(values):
    """空组保留为空；不把缺失、非有限值或零方差解释成有效信号。"""
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("NONFINITE_VALUES")
    return {"n": len(a), "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def rank_score(up, down):
    """随机取一个上涨日和下跌日，上涨日输入更大计 1，相等计 0.5。

    不训练分类器，不决定采用正向或反向，不拟合切分点。
    任一类别为空时返回 None，不将不可计算写成 0.5。
    """
    if not up or not down:
        return None
    positive, negative = np.asarray(up, dtype=float), np.sort(np.asarray(down, dtype=float))
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("NONFINITE_RANK_VALUES")
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    return float(np.sum(lower + (upper - lower) / 2) / (len(up) * len(down)))


def sign(value):
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def feature_evidence(rows, sign_features, predictions=None):
    """保留三类总数；仅二类排序分母排除持平，固定零分组仍包含持平。"""
    output = {"rows": len(rows), "actual": dict(Counter(r["actual_direction"] for r in rows)), "features": {}}
    for i, name in enumerate(FEATURES):
        by_class = {c: [r["x"][i] for r in rows if r["actual_direction"] == c] for c in CLASSES}
        item = {
            "up_down_rank_score": rank_score(by_class["UP"], by_class["DOWN"]),
            "by_actual_class": {c: descriptive(v) for c, v in by_class.items()},
        }
        if name in sign_features:
            groups = {}
            for bucket in ("negative", "zero", "positive"):
                selected = [r for r in rows if sign(r["x"][i]) == bucket]
                groups[bucket] = {
                    "rows": len(selected),
                    "actual": dict(Counter(r["actual_direction"] for r in selected)),
                }
                if predictions is not None:
                    groups[bucket]["saved_L20_recovered"] = {
                        "correct": sum(
                            predictions[r["target"]]["new_direction"] == r["actual_direction"] for r in selected
                        ),
                        "actual_UP_predicted_DOWN": sum(
                            r["actual_direction"] == "UP" and predictions[r["target"]]["new_direction"] == "DOWN"
                            for r in selected
                        ),
                    }
            item["fixed_zero_groups"] = groups
        output["features"][name] = item
    return output


def correlation(left, right):
    if len(left) != len(right):
        raise ValueError("UNPAIRED_RETURNS")
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("NONFINITE_RETURNS")
    return float(np.corrcoef(left, right)[0, 1])


def daily_return(row):
    return float(Decimal(row["target_unit_nav"]) / Decimal(row["base_unit_nav"]) - 1)


def paired_returns(own, peers):
    """只按目标日和基准日精确匹配；不向前填充，不把同日答案变成输入。"""
    index = {(r["target"], r["base"]): r for r in own}
    if len(index) != len(own):
        raise ValueError("DUPLICATE_OWN_DAY")
    records, missing = [], []
    for peer in peers:
        target = index.get((peer["target"], peer["base"]))
        if target is None:
            missing.append({"target": peer["target"], "base": peer["base"]})
            continue
        records.append(
            {
                "target": peer["target"],
                "base": peer["base"],
                "own_return": daily_return(target),
                "peer_return": daily_return(peer),
                "own_actual": target["actual_direction"],
                "peer_actual": peer["actual_direction"],
            }
        )
    groups = {"all": records}
    groups.update({year: [r for r in records if r["target"].startswith(year)] for year in ("2021", "2022", "2023")})
    summary = {}
    for name, group in groups.items():
        summary[name] = {
            "matched": len(group),
            "same_direction": sum(r["own_actual"] == r["peer_actual"] for r in group),
            "pearson_same_day_return": correlation([r["own_return"] for r in group], [r["peer_return"] for r in group]),
            "own_return": descriptive([r["own_return"] for r in group]),
            "peer_return": descriptive([r["peer_return"] for r in group]),
        }
    return {"peer_rows": len(peers), "missing_target_dates": missing, "groups": summary, "daily": records}


def allocation_evidence(protocol):
    snapshot = read_json(SNAPSHOT)
    reports = {r["raw"]["sha256"]: r for f in snapshot["funds"].values() for r in f["reports"]}
    for path in protocol["parsed_report_paths"]:
        r = read_json(path)
        reports[r["raw"]["sha256"]] = r
    overlap = read_json(PREVIOUS / "peer-overlap.json")["daily"]
    used = Counter(h for r in overlap for h in (r["target_report_sha256"], r["peer_report_sha256"]))
    records = []
    for h, count in sorted(used.items()):
        r = reports[h]
        records.append(
            {
                "fund_code": r["fund_code"],
                "report_end": r["report_end"],
                "report_type": r["report_type"],
                "published_date": r["published_date"],
                "report_sha256": h,
                "matched_rows_using_report": count,
                "stock_nav_pct": float(Decimal(r["stock_nav_pct"])),
                "full_disclosure": r["full_stock_disclosure"],
                "holding_count": len(r["holdings"]),
                "assets_total_asset_denominator": r["assets"],
            }
        )
    summary = {}
    for code in protocol["funds"]:
        rows = [r for r in records if r["fund_code"] == code]
        summary[code] = {
            "distinct_reports": len(rows),
            "stock_nav_pct_report_level": descriptive([r["stock_nav_pct"] for r in rows]),
        }
    return {
        "reports": records,
        "summary": summary,
        "denominator_note": (
            "Stock share is net assets; asset table percentages use total assets. "
            "Distinct reports are not independent trading-day observations."
        ),
    }


def run():
    protocol = read_json(OUT / "protocol.json")
    if (
        protocol["fit_budget"] != 0
        or protocol["network_requests_allowed"] != 0
        or protocol["features"] != list(FEATURES)
    ):
        raise ValueError("FROZEN_PROTOCOL_CHANGED")
    for path, expected in protocol["source_files"].items():
        if file_hash(path) != expected:
            raise ValueError("SOURCE_CHANGED:" + path)
    save_once(
        OUT / ("code-freeze-" + file_hash(__file__)[:12] + ".json"),
        {
            "file": str(Path(__file__)),
            "sha256": file_hash(__file__),
            "protocol_sha256": file_hash(OUT / "protocol.json"),
        },
    )
    inputs, folds = read_json(NEW / "frozen/inputs.json"), read_json(NEW / "frozen/folds.json")
    own = [r for r in inputs["train"] if r["fund_code"] == "002112"]
    exam = [r for f in folds[:3] for r in f["exam"]]
    development = inputs["development"]
    for row in inputs["train"] + development + exam:
        validate_row(row)
    groups = {
        "own_train_before_2023": [r for r in own if r["target"] < "2023-01-01"],
        "own_train_2023_complete": [r for r in own if r["target"].startswith("2023")],
        "historical_161": exam,
        "development_2024_230": development,
    }
    groups.update({f["name"]: f["exam"] for f in folds[:3]})
    groups.update(
        {f"2024Q{q}": [r for r in development if (int(r["target"][5:7]) - 1) // 3 + 1 == q] for q in range(1, 5)}
    )
    predictions = {r["target"]: r for r in read_json(PREVIOUS / "daily.json")}
    with no_fits():
        scores = {
            name: feature_evidence(
                rows,
                protocol["statistics"]["sign_features"],
                predictions if name == "historical_161" or name.startswith("2023Q") else None,
            )
            for name, rows in groups.items()
        }
        paired = {
            code: paired_returns(own, [r for r in inputs["train"] if r["fund_code"] == code])
            for code in ("017493", "160323")
        }
        allocations = allocation_evidence(protocol)
    result = {
        "actual_new_fits": 0,
        "cumulative_actual_fits": 58,
        "groups": {n: {"rows": v["rows"], "actual": v["actual"]} for n, v in scores.items()},
        "paired": {
            k: {
                "peer_rows": v["peer_rows"],
                "missing_target_count": len(v["missing_target_dates"]),
                "groups": v["groups"],
            }
            for k, v in paired.items()
        },
        "allocations": allocations["summary"],
        "no_predictions_generated": True,
        "limitations": [
            "Posthoc univariate descriptions, not proof of out-of-sample model improvement",
            "Named sample groups overlap; never sum them as independent cases",
            "No p-value screening or threshold/model selection",
        ],
    }
    for name, value in (
        ("feature-evidence", scores),
        ("paired-returns", paired),
        ("report-allocations", allocations),
        ("summary", result),
    ):
        save_once(OUT / (name + ".json"), value)
    return result
