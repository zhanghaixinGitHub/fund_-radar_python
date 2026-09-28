"""002112 已保存结果的零拟合复盘；只读冻结输入，不加载模型、数据库或外部来源。

分组和时间块参数来自事先保存的 protocol.json。所有统计均属已经观察过的开发
资料的事后诊断，不能改变旧实验门槛，不能作为新增模型或真实未来预测。
"""

import hashlib
import json
import math
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / ".local-runs/fund-exposure-002112"
REVIEW = ROOT / "zero-fit-review/20260926-v1"
CLASSES = ("DOWN", "FLAT", "UP")
MODEL_NAMES = {
    "old_2024": ("A7", "B16", "C20", "W20", "I24"),
    "old_2023": ("A7", "C20", "W20", "I24"),
    "prior_2023": ("N7", "L20", "T20"),
}
GROUPS = {
    "nav_lag": ("0", "1", "2+", "UNKNOWN_OLD_CONTRACT"),
    "report_age": ("0-90", "91-180", "181+"),
    "stock_weight": ("<50%", "50%-<80%", ">=80%"),
    "disclosure_fraction": ("<50%", "50%-<80%", ">=80%", "NOT_APPLICABLE"),
    "full_disclosure": ("FULL", "PARTIAL"),
    "absolute_return": ("EXACT_FLAT", "0-0.1%", ">0.1%-0.5%", ">0.5%"),
    "market_agreement": ("SAME_SIGN", "DIFFERENT_SIGN", "ZERO_INVOLVED"),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    """兼容既有 hash/payload 封装，校验内容；不接受 JSON 非有限数值。"""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and set(value) == {"hash", "payload"}:
        if digest(value["payload"]) != value["hash"]:
            raise ValueError("CONTENT_HASH_CHANGED")
        return value["payload"]
    canonical(value)  # 即使没有封装，也拒绝 NaN/Infinity。
    return value


def save_once(path, value):
    """复盘证据排他创建，相同内容可复用，不覆盖另一组统计结果。"""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("REVIEW_ARTIFACT_ALREADY_EXISTS:" + path.name)
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def source(protocol, name):
    """只能读取协议清单内且摘要仍一致的本地来源。"""
    item = protocol["sources"][name]
    path = Path(item["path"]).resolve()
    if not path.is_relative_to(ROOT.resolve()) or file_hash(path) != item["sha256"]:
        raise ValueError("REVIEW_SOURCE_CHANGED:" + name)
    return read_json(path)


def quarter(day):
    parsed = date.fromisoformat(day)
    return f"{parsed.year}Q{(parsed.month - 1) // 3 + 1}"


def validate_row(row):
    """核验允许日期、原始净值标签和完整 20 维输入，不填补缺数。"""
    for field in ("base", "target"):
        day = row[field]
        if date.fromisoformat(day).isoformat() != day or day > "2024-12-31":
            raise ValueError("DATE_OUTSIDE_REVIEW_SCOPE")
    if row["base"] >= row["target"]:
        raise ValueError("INVALID_TARGET_ORDER")
    a, b = Decimal(row["base_unit_nav"]), Decimal(row["target_unit_nav"])
    if not a.is_finite() or not b.is_finite() or min(a, b) <= 0:
        raise ValueError("INVALID_NAV")
    actual = "UP" if b > a else "DOWN" if b < a else "FLAT"
    if actual != row["actual_direction"]:
        raise ValueError("RAW_LABEL_MISMATCH")
    x = row["x"]
    if len(x) != 20 or any(type(v) not in (int, float) or not math.isfinite(v) for v in x):
        raise ValueError("INVALID_FEATURE_VECTOR")
    if x[12] <= 0 or x[13] < 0 or x[14] < 0 or x[15] not in (0, 1):
        raise ValueError("INVALID_EXPOSURE")


def index_rows(rows):
    indexed = {}
    for row in rows:
        validate_row(row)
        key = row["fund_code"], row["target"]
        if key in indexed:
            raise ValueError("DUPLICATE_FUND_TARGET")
        indexed[key] = row
    return indexed


def groups(row, protocol):
    """单位为比例值；涨跌幅分组仅供事后诊断，不改变三类标签。"""
    x, bins = row["x"], protocol["bins"]
    lag = row.get("nav", {}).get("lag_sessions")
    if lag is not None and (type(lag) is not int or lag < 0):
        raise ValueError("INVALID_LAG")
    stock_cut, coverage_cut = bins["stock_nav_weight"], bins["disclosed_stock_fraction"]
    fraction = x[12] / x[13] if x[13] > 0 else None
    a, b = Decimal(row["base_unit_nav"]), Decimal(row["target_unit_nav"])
    magnitude = abs(b / a - 1)
    small, medium = (Decimal(str(v)) for v in bins["absolute_return_fraction"][1:])
    signs = {int(v > 0) - int(v < 0) for v in (x[7], x[16], x[18])}

    def weight_group(value, cut):
        if value is None:
            return "NOT_APPLICABLE"
        return "<50%" if value < cut[0] else "50%-<80%" if value < cut[1] else ">=80%"

    return {
        "nav_lag": "UNKNOWN_OLD_CONTRACT" if lag is None else str(lag) if lag <= 1 else "2+",
        "report_age": "0-90"
        if x[14] <= bins["report_age_days"][0]
        else "91-180"
        if x[14] <= bins["report_age_days"][1]
        else "181+",
        "stock_weight": weight_group(x[13], stock_cut),
        "disclosure_fraction": weight_group(fraction, coverage_cut),
        "full_disclosure": "FULL" if x[15] == 1 else "PARTIAL",
        "absolute_return": "EXACT_FLAT"
        if magnitude == 0
        else "0-0.1%"
        if magnitude <= small
        else ">0.1%-0.5%"
        if magnitude <= medium
        else ">0.5%",
        "market_agreement": "ZERO_INVOLVED" if 0 in signs else "SAME_SIGN" if len(signs) == 1 else "DIFFERENT_SIGN",
    }


def join_daily(raw, inputs, names, protocol):
    """以基金和日期连接，不依赖文件行序；标签或预测缺失时整次拒绝。"""
    daily, seen = [], set()
    for entry in raw:
        target = entry["target"]
        if target in seen:
            raise ValueError("DUPLICATE_EXAM_DATE")
        seen.add(target)
        row = inputs[("002112", target)]
        actual = entry.get("actual_direction", entry.get("actual"))
        directions = entry.get("directions", entry)
        if actual != row["actual_direction"] or any(directions[n] not in CLASSES for n in names):
            raise ValueError("EXAM_LABEL_OR_DIRECTION_MISMATCH")
        daily.append(
            {
                "target": target,
                "base": row["base"],
                "actual": actual,
                "directions": {n: directions[n] for n in names},
                "groups": groups(row, protocol),
                "input_hash": digest(row),
                "nav_lag_sessions": row.get("nav", {}).get("lag_sessions"),
                "report_age_days": row["x"][14],
                "stock_weight_pct": 100 * row["x"][13],
                "disclosed_weight_pct": 100 * row["x"][12],
                "absolute_return_pct": float(
                    abs(Decimal(row["target_unit_nav"]) / Decimal(row["base_unit_nav"]) - 1) * 100
                ),
            }
        )
    return sorted(daily, key=lambda r: r["target"])


def correct(row, name):
    return (name if name in CLASSES else row["directions"][name]) == row["actual"]


def summary(rows, names):
    """分别给出真实类别、预测类别与各类命中，空组准确率保持未知。"""
    actual = {k: sum(r["actual"] == k for r in rows) for k in CLASSES}
    models = {}
    for name in names:
        by_class = {k: sum(r["actual"] == k and correct(r, name) for r in rows) for k in CLASSES}
        hits = sum(by_class.values())
        models[name] = {
            "correct": hits,
            "accuracy": hits / len(rows) if rows else None,
            "correct_by_class": by_class,
            "predicted": dict(Counter(name if name in CLASSES else r["directions"][name] for r in rows)),
        }
    return {"n": len(rows), "actual": actual, "models": models}


def block_indices(days, settings):
    """季度内循环时间块，所有配对共享抽样日期；保留原合格样本间的日期缺口。

    10 指按时间排序的 10 个合格检查日，不冒充 10 个无缺口交易日。此方法保留
    块内依赖和季度样本量，不保证处理所有非平稳性，也不校正事后模型选择。
    """
    if days != sorted(set(days)):
        raise ValueError("BOOTSTRAP_DATES_NOT_UNIQUE_SORTED")
    rng = np.random.default_rng(settings["seed"])
    reps, size = settings["replications"], settings["block_length"]
    if reps < 1 or size < 1 or not days:
        raise ValueError("INVALID_BOOTSTRAP_SETTINGS")
    parts = []
    for q in sorted({quarter(d) for d in days}):
        positions = np.array([i for i, d in enumerate(days) if quarter(d) == q])
        starts = rng.integers(0, len(positions), size=(reps, math.ceil(len(positions) / size)))
        blocks = ((starts[..., None] + np.arange(size)) % len(positions)).reshape(reps, -1)[:, : len(positions)]
        parts.append(positions[blocks])
    return np.concatenate(parts, axis=1)


def pair_result(rows, candidate, control, indices):
    """用同日正确性差值估计诊断区间，不把主训练/复现算作独立样本。"""
    delta = np.array([int(correct(r, candidate)) - int(correct(r, control)) for r in rows])
    boot = delta[indices].mean(axis=1)
    interval = np.quantile(boot, [0.025, 0.975], method="linear")
    return {
        "candidate": candidate,
        "control": control,
        "n": len(rows),
        "gained": int(sum(delta == 1)),
        "lost": int(sum(delta == -1)),
        "net_correct": int(delta.sum()),
        "difference_pp": float(delta.mean() * 100),
        "diagnostic_interval_pp": (interval * 100).tolist(),
        "includes_zero": bool(interval[0] <= 0 <= interval[1]),
        "quarter_net": {
            q: int(sum(v for r, v in zip(rows, delta, strict=True) if quarter(r["target"]) == q))
            for q in sorted({quarter(r["target"]) for r in rows})
        },
        "class_net": {k: int(sum(v for r, v in zip(rows, delta, strict=True) if r["actual"] == k)) for k in CLASSES},
        "claim": "POSTHOC_DIAGNOSTIC_NOT_NEW_PASS_DECISION",
    }


def distribution(rows):
    """只描述该基金输入分布；披露股票占比与总股票仓位分别统计。"""
    result = {"n": len(rows)}
    for name, i in (("stock_weight_pct", 13), ("disclosed_weight_pct", 12), ("report_age_days", 14)):
        values = [r["x"][i] * (1 if i == 14 else 100) for r in rows]
        result[name] = dict(
            zip(
                ("min", "q25", "median", "q75", "max"),
                np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                strict=True,
            )
        )
    return result


def proxy_diagnostics(rows):
    """比较同一市场日的已披露股票篮子和宽基；这不是预测次日的相关性。"""
    basket = np.array([r["x"][7] / r["x"][12] for r in rows])
    result = {"n": len(rows), "meaning": "SAME_T_DAY_DESCRIPTIVE_NOT_NEXT_DAY_PREDICTION", "indices": {}}
    for name, i in (("CSI300", 16), ("CSI500", 18)):
        values = np.array([r["x"][i] for r in rows])
        result["indices"][name] = {
            "pearson": float(np.corrcoef(basket, values)[0, 1]) if np.std(basket) and np.std(values) else None,
            "opposite_nonzero_sign_days": int(sum(basket * values < 0)),
            "zero_involved_days": int(sum((basket == 0) | (values == 0))),
        }
    return result


def analyze(protocol):
    """读取固定清单并生成全部预定统计；没有训练、推理重跑和动态选样入口。"""
    old, new = source(protocol, "old_inputs"), source(protocol, "third_inputs")
    if (len(old["train"]), len(old["development"]), len(new["train"]), len(new["development"])) != (
        4422,
        230,
        4419,
        230,
    ):
        raise ValueError("FROZEN_ROW_COUNTS_CHANGED")
    old_index = index_rows(old["train"] + old["development"])
    new_index = index_rows(new["train"] + new["development"])
    for key, row in new_index.items():
        if (
            row["old_row_hash"] != digest(old_index[key])
            or row["actual_direction"] != old_index[key]["actual_direction"]
        ):
            raise ValueError("PRIOR_INPUT_IDENTITY_CHANGED")
    second = source(protocol, "second_comparison")
    dev = second["development"]["daily"]
    predictions = {n: source(protocol, key) for n, key in (("A7", "old_a7"), ("B16", "old_b16"), ("C20", "old_c20"))}
    for n, values in predictions.items():
        by_date = {r["target"]: r for r in values}
        if len(by_date) != 230 or set(by_date) != {r["target"] for r in dev}:
            raise ValueError("FIRST_ROUND_PREDICTION_DATES_CHANGED")
        for entry in dev:
            p = by_date[entry["target"]]
            original = old_index[("002112", entry["target"])]
            if p["input_hash"] != digest(original) or p["actual_direction"] != original["actual_direction"]:
                raise ValueError("FIRST_ROUND_PREDICTION_IDENTITY_CHANGED")
            if n != "B16" and entry["directions"][n] != p["direction"]:
                raise ValueError("OLD_CONTROL_REUSE_MISMATCH")
            entry["directions"][n] = p["direction"]
    raw_prior = source(protocol, "third_daily")["historical"]
    raw_old = second["historical"]["daily"]
    folds = source(protocol, "third_folds")
    expected = sorted(d for f in folds if f["name"] != "FULL" for d in f["expected_dates"])
    if sorted(r["target"] for r in raw_prior) != expected or sorted(r["target"] for r in raw_old) != expected:
        raise ValueError("HISTORICAL_DATE_SETS_CHANGED")
    if len(expected) != 161 or {quarter(d) for d in expected} != {"2023Q2", "2023Q3", "2023Q4"}:
        raise ValueError("HISTORICAL_SCOPE_CHANGED")
    daily = {
        "old_2024": join_daily(dev, old_index, MODEL_NAMES["old_2024"], protocol),
        "old_2023": join_daily(raw_old, old_index, MODEL_NAMES["old_2023"], protocol),
        "prior_2023": join_daily(raw_prior, new_index, MODEL_NAMES["prior_2023"], protocol),
    }
    cohorts = {}
    for cohort, rows in daily.items():
        names = MODEL_NAMES[cohort]
        indices = block_indices([r["target"] for r in rows], protocol["bootstrap"])
        grouped = {}
        for dimension, labels in GROUPS.items():
            grouped[dimension] = {}
            for label in labels:
                subset = [r for r in rows if r["groups"][dimension] == label]
                item = summary(subset, names)
                item["small_group"] = len(subset) < protocol["small_group_minimum"]
                item["quarters"] = {
                    q: summary([r for r in subset if quarter(r["target"]) == q], names)
                    for q in sorted({quarter(r["target"]) for r in rows})
                }
                grouped[dimension][label] = item
            if sum(g["n"] for g in grouped[dimension].values()) != len(rows):
                raise ValueError("GROUP_NOT_EXHAUSTIVE")
        cohorts[cohort] = {
            "overall": summary(rows, names + CLASSES),
            "groups": grouped,
            "quarters": {
                q: summary([r for r in rows if quarter(r["target"]) == q], names + CLASSES)
                for q in sorted({quarter(r["target"]) for r in rows})
            },
            "pairs": [pair_result(rows, a, b, indices) for a, b in protocol["pair_comparisons"][cohort]],
            "eligible_date_gaps": sum(
                left["target"] != right["base"] for left, right in zip(rows, rows[1:], strict=False)
            ),
        }
    prior_rows = [new_index[("002112", r["target"])] for r in daily["prior_2023"]]
    news = source(protocol, "news_analysis")["records"]
    selection = source(protocol, "news_selection")["selected"]
    cards = source(protocol, "news_cards")["cards"]
    if len(news) != 12 or {r["id"] for r in news} != {r["sample_id"] for r in selection} or len(cards) != 12:
        raise ValueError("NEWS_SAMPLE_IDENTITY_CHANGED")
    news_audit = {
        "records": len(news),
        "target_dates": sorted({r["target"] for r in news}),
        "companies": sorted({r["stock_code"] for r in news}),
        "anchors": sum(len(r["anchors"]) for r in news),
        "prediction_eligible_true": sum(r["prediction_eligible"] is True for r in news),
        "training_eligible_true": sum(r["training_eligible"] is True for r in news),
        "field_presence": {
            k: sum(k in r and r[k] is not None for r in news)
            for k in ("facts", "event_stage", "published_date", "anchors", "unknowns", "priced_in_status")
        },
        "validation_scope": "PRESENCE_AND_EXISTING_SAMPLE_IDENTITY_ONLY_NOT_SEMANTIC_ACCURACY_OR_PREDICTIVE_GAIN",
    }
    result = {
        "protocol_hash": digest(protocol),
        "new_model_fits": 0,
        "cumulative_model_fits": 52,
        "cohorts": cohorts,
        "news": news_audit,
        "input_distribution": {
            name: distribution(rows)
            for name, rows in {
                "old_target_training": [r for r in old["train"] if r["fund_code"] == "002112"],
                "prior_target_training": [r for r in new["train"] if r["fund_code"] == "002112"],
                "prior_2023_exam": prior_rows,
                "prior_2024_inputs_only": new["development"],
            }.items()
        },
        "market_proxy": {
            name: proxy_diagnostics(rows)
            for name, rows in {
                "prior_2023_exam": prior_rows,
                "prior_2024_inputs_only": new["development"],
            }.items()
        },
        "unchanged_historical_decision": source(protocol, "third_decision"),
        "limitations": [
            "POSTHOC",
            "UNADJUSTED_MULTIPLE_COMPARISONS",
            "CORRELATION_NOT_CAUSATION",
            "SMALL_QUARTERS",
            "FIXED_BLOCK_LENGTH_NOT_UNIVERSAL",
            "NO_NEW_2024_MODEL_RESULTS",
        ],
    }
    return result, daily


def render(result):
    lines = [
        "# 002112 零拟合复盘：完整统计",
        "",
        "新增真实训练 0 次，累计仍为 52 次。旧实验决定不变。",
        "",
        "统计规则在计算前固定，但数据已参与研究；所有区间都是未校正多重比较的事后诊断区间，不能作为独立确认。",
        "季度内循环块固定为 10 个相邻合格检查日、重抽样 10,000 次；保留原日期缺口，不跨季度，未搜索块长。",
        "",
    ]
    for cohort, data in result["cohorts"].items():
        lines += [
            f"## {cohort}",
            "",
            "| 做法 | 正确/总天数 | 上涨正确 | 持平正确 | 下跌正确 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, model in data["overall"]["models"].items():
            c = model["correct_by_class"]
            lines.append(
                f"| {name} | {model['correct']}/{data['overall']['n']} | {c['UP']} | {c['FLAT']} | {c['DOWN']} |"
            )
        lines += [
            "",
            "| 比较 | 新对 | 丢失 | 净多对 | 差值/百分点 | 95% 诊断区间/百分点 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for pair in data["pairs"]:
            low, high = pair["diagnostic_interval_pp"]
            lines.append(
                f"| {pair['candidate']} - {pair['control']} | {pair['gained']} | {pair['lost']} "
                f"| {pair['net_correct']} | {pair['difference_pp']:.2f} | [{low:.2f}, {high:.2f}] |"
            )
        names = MODEL_NAMES[cohort]
        for dimension, bins in data["groups"].items():
            lines += [
                "",
                f"### {dimension}",
                "",
                "| 分组 | 天数 | 实际涨/平/跌 | " + " | ".join(names) + " |",
                "| --- | ---: | --- | " + " | ".join("---:" for _ in names) + " |",
            ]
            for label, group in bins.items():
                a = group["actual"]
                counts = " | ".join(str(group["models"][n]["correct"]) for n in names)
                lines.append(
                    f"| {label}{'（不足20天）' if group['small_group'] else ''} | {group['n']} "
                    f"| {a['UP']}/{a['FLAT']}/{a['DOWN']} | {counts} |"
                )
    lines += [
        "",
        "逐季度分组、每日输入身份和所有模型方向见 summary.json、daily.json。空组不是准确率 0，JSON 中为 null。",
        "",
        "时间块方法参考：https://bashtage.github.io/arch/doc/bootstrap/timeseries-bootstraps.html",
        "",
    ]
    return "\n".join(lines)


def run(directory=REVIEW):
    """只写自己的复盘目录；已有结果可重算核对，规则或来源变化即拒绝。"""
    directory = Path(directory)
    if directory.resolve() != REVIEW.resolve():
        raise ValueError("REVIEW_OUTPUT_SCOPE")
    protocol = read_json(directory / "protocol.json")
    if protocol["maximum_new_model_fits"] != 0 or protocol["bootstrap"]["block_length"] != 10:
        raise ValueError("REVIEW_PROTOCOL_CHANGED")
    started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    result, daily = analyze(protocol)
    save_once(directory / "summary.json", result)
    save_once(directory / "daily.json", daily)
    report_path, report = directory / "report.md", render(result)
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") != report:
            raise ValueError("REVIEW_REPORT_CHANGED")
    else:
        with report_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report)
    evidence = {
        "protocol_file_sha256": file_hash(directory / "protocol.json"),
        "implementation_sha256": file_hash(Path(__file__)),
        "summary_sha256": file_hash(directory / "summary.json"),
        "daily_sha256": file_hash(directory / "daily.json"),
        "new_model_fits": 0,
    }
    save_once(directory / "execution-evidence.json", evidence)
    return {
        "started": started,
        "completed": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "output": str(directory),
        "new_model_fits": 0,
        "summary_sha256": evidence["summary_sha256"],
    }
