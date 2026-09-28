"""核对旧持仓输入与已保存的 2024 结果；只复算证据，不训练、不生成预测。"""

import argparse
import json
import math
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

from app.services.fund_002112_holdings_compatibility import json_subtree
from app.services.fund_002112_zero_fit_review import digest, file_hash, read_json, save_once

ROOT = Path(__file__).resolve().parents[1] / ".local-runs/fund-exposure-002112"
OUT = ROOT / "holdings-input-lag-review/20260928-v1"
CLASSES = ("DOWN", "FLAT", "UP")
MODELS = ("A7", "B16", "C20")
PAIRS = (("B16", "A7"), ("C20", "A7"), ("C20", "B16"))


def source_path(protocol, name):
    """只允许协议列出的文件，先核对完整字节摘要，再读取需要的内容。"""
    entry = protocol["sources"][name]
    path = Path(entry["path"])
    if file_hash(path) != entry["sha256"]:
        raise ValueError("SOURCE_CHANGED:" + name)
    return path


def branch(protocol, name, path):
    """按路径读取冻结 JSON；快照只读报告、目录及指数，不解码基金净值分支。"""
    text = source_path(protocol, name).read_text(encoding="utf-8")
    try:
        return json_subtree(text, ["payload", *path])
    except KeyError:
        return json_subtree(text, path)


def publication(report):
    """用真实公开和修订日期的较晚者；未知日期停止，不向前推定。"""
    first = report.get("published_date")
    if not first:
        # 冻结目录的 available_at 是公开日加一天；该旧契约已有来源核对。
        from datetime import timedelta

        if not report.get("available_at"):
            raise ValueError("PUBLICATION_UNKNOWN")
        first = (date.fromisoformat(report["available_at"][:10]) - timedelta(days=1)).isoformat()
    return max(first, *(report.get(k, first)[:10] for k in
                        ("source_publication_date", "revised_at", "version_publication_date")))


def report_key(report):
    return (report["report_end"], publication(report),
            report.get("full_stock_disclosure", report["report_type"] in ("ANNUAL", "HALF")))


def legal_report(reports, catalog, target):
    """重核原公开日规则；最新已公开目录项缺正文时，不退回旧报告。"""
    available = [r for r in reports if publication(r) < target]
    if not available:
        raise ValueError("NO_AVAILABLE_REPORT")
    key = max(map(report_key, available))
    choices = [r for r in available if report_key(r) == key]
    if len({r["raw"]["sha256"] for r in choices}) != 1:
        raise ValueError("REPORT_VERSION_CONFLICT")
    if max((report_key(r) for r in catalog if publication(r) < target), default=key) > key:
        raise ValueError("LATEST_REPORT_MISSING")
    return choices[0]


def recompute_exposure(report, days, quotes, indices):
    """用十进制独立复算原 9+4 项，权重单位为基金净资产比例。

    复算的是已披露股票篮子在前一交易日的行情，不估计实际仓位，也不算新方向。
    每只正权重股票必须具有完整 21 日窗口；缺数、非有限值和零成交基数均报错。
    """
    if len(days) != 21 or days != sorted(set(days)):
        raise ValueError("COMPLETE_21_SESSION_WINDOW_REQUIRED")
    age = (date.fromisoformat(days[-1]) - date.fromisoformat(report["report_end"])).days
    if not 0 <= age <= 210:
        raise ValueError("REPORT_AGE_OUTSIDE_ORIGINAL_RULE")
    weights, values = {}, [Decimal(0)] * 5
    for h in report["holdings"]:
        code, weight = h["stock_code"], Decimal(str(h["nav_weight_pct"])) / 100
        if code in weights or not weight.is_finite() or weight < 0:
            raise ValueError("INVALID_HOLDING")
        weights[code] = weight
        if not weight:  # 原表明确写 0.00%，保持它与数据缺失的区别。
            continue
        try:
            rows = [[Decimal(str(quotes[d][code][k])) for k in ("pct_chg", "amount")] for d in days]
        except (KeyError, TypeError, ArithmeticError) as exc:
            raise ValueError("POSITIVE_HOLDING_QUOTE_MISSING") from exc
        if not all(v.is_finite() for row in rows for v in row):
            raise ValueError("QUOTE_NONFINITE")
        mean = sum(r[1] for r in rows[:-1]) / 20
        if mean <= 0:
            raise ValueError("AMOUNT_BASE_NOT_POSITIVE")
        values[0] += weight * rows[-1][0] / 100
        values[1] += weight * (math.prod(1 + r[0] / 100 for r in rows[-5:]) - 1)
        values[2] += weight if rows[-1][0] > 0 else 0
        values[3] += weight * (rows[-1][1] / mean - 1)
        values[4] += weight * weight
    disclosed = Decimal(str(report["disclosed_nav_pct"])) / 100
    if disclosed <= 0 or abs(sum(weights.values()) - disclosed) > Decimal("1e-12"):
        raise ValueError("DISCLOSED_WEIGHT_MISMATCH")
    values.extend([disclosed, Decimal(str(report["stock_nav_pct"])) / 100,
                   Decimal(age), Decimal(int(report["full_stock_disclosure"]))])
    for code in ("000300.SH", "000905.SH"):
        window = [Decimal(str(indices[code]["rows"][d]["pct_chg"])) / 100 for d in days[-5:]]
        values.extend([window[-1], math.prod(1 + x for x in window) - 1])
    if not all(x.is_finite() for x in values):
        raise ValueError("EXPOSURE_NONFINITE")
    return [float(x) for x in values]


def prediction_index(predictions, original):
    """验证旧预测与原输入逐行同源，禁止缺日、重复日或冒用修复后的输入。"""
    by_date = {}
    for p in predictions:
        day = p["target"]
        if day in by_date or day not in original:
            raise ValueError("PREDICTION_DATE_MISMATCH")
        row = original[day]
        if (p["fund_code"] != "002112" or p["input_hash"] != digest(row)
                or p["actual_direction"] != row["actual_direction"] or p["base"] != row["base"]
                or p["direction"] not in CLASSES):
            raise ValueError("PREDICTION_INPUT_IDENTITY_MISMATCH")
        by_date[day] = p
    if set(by_date) != set(original):
        raise ValueError("PREDICTION_DATE_MISMATCH")
    return by_date


def summarize(rows):
    """完整保留三类及常量基线；成对差值只来自同一日期保存的原预测。"""
    actual = {c: sum(r["actual_direction"] == c for r in rows) for c in CLASSES}
    models = {}
    for name in MODELS:
        correct = {c: sum(r["actual_direction"] == r["directions"][name] == c for r in rows) for c in CLASSES}
        models[name] = {"correct": sum(correct.values()), "class_correct": correct}
    pairs = {}
    for left, right in PAIRS:
        gains = [r["target"] for r in rows if r["directions"][left] == r["actual_direction"]
                 and r["directions"][right] != r["actual_direction"]]
        losses = [r["target"] for r in rows if r["directions"][left] != r["actual_direction"]
                  and r["directions"][right] == r["actual_direction"]]
        pairs[left + "_vs_" + right] = {
            "gained": len(gains), "lost": len(losses), "net": len(gains) - len(losses),
            "gained_dates": gains, "lost_dates": losses,
        }
    return {"days": len(rows), "actual": actual, "constant_correct": actual,
            "models": models, "pairs": pairs}


def run(directory=OUT):
    """在冻结范围内复核并排他保存；复跑只复用相同结果，没有拟合或预测接口。"""
    protocol = read_json(directory / "protocol.json")
    if protocol["fit_budget"] != 0 or protocol["scope"]["expected_dates"] != 230:
        raise ValueError("REVIEW_SCOPE_CHANGED")
    for name in protocol["sources"]:
        source_path(protocol, name)
    plan_features = branch(protocol, "plan", ["features"])
    if plan_features != branch(protocol, "old_protocol", ["features"]) or len(plan_features) != 20:
        raise ValueError("FEATURE_DEFINITION_CHANGED")
    current = branch(protocol, "inputs", ["development"])
    if current != branch(protocol, "original_inputs", ["development"]):
        raise ValueError("RECOVERY_CHANGED_DEVELOPMENT_INPUTS")
    legacy = branch(protocol, "old_inputs", ["development"])
    old = {r["target"]: r for r in legacy}
    if len(current) != 230 or len(legacy) != 230 or len(old) != 230 or set(old) != {r["target"] for r in current}:
        raise ValueError("ORIGINAL_230_DATES_REQUIRED")
    predictions = {n: prediction_index(read_json(source_path(protocol, "old_" + n.lower())), old) for n in MODELS}
    reports = branch(protocol, "snapshot", ["funds", "002112", "reports"])
    catalog = branch(protocol, "snapshot", ["funds", "002112", "catalog"])
    indices = branch(protocol, "snapshot", ["indices"])
    gap = set(protocol["scope"]["gap_dates"])
    event = read_json(source_path(protocol, "event"))
    if gap != set(event["original_sample_dates"]) or len(gap) != 10:
        raise ValueError("PREDECLARED_EVENT_DATES_CHANGED")
    quotes = {}
    for name in protocol["sources"]:
        if not name.startswith("stock_day_"):
            continue
        day = name.removeprefix("stock_day_")
        wrapper = read_json(source_path(protocol, name))
        raw_path = source_path(protocol, "raw_quote_" + day)
        if file_hash(raw_path) != wrapper["receipt"]["sha256"] or wrapper["date"] != day:
            raise ValueError("QUOTE_RECEIPT_MISMATCH")
        raw = read_json(raw_path)["data"]
        mapped = {}
        for vals in raw["items"]:
            r = dict(zip(raw["fields"], vals, strict=True))
            if r.pop("trade_date") != day.replace("-", ""):
                raise ValueError("QUOTE_DATE_CHANGED")
            code = r.pop("ts_code")
            if code in mapped:
                raise ValueError("DUPLICATE_QUOTE")
            mapped[code] = r
        if any(mapped.get(c) != r for c, r in wrapper["rows"].items()):
            raise ValueError("AGGREGATE_QUOTE_MISMATCH")
        quotes[day] = mapped
    daily, trace, differences = [], [], []
    for row in current:
        day, before = row["target"], old[row["target"]]
        if (row["fund_code"] != "002112" or not "2024-01-01" <= day <= "2024-12-31"
                or not row["base"] < day or len(row["x"]) != 20
                or any(not math.isfinite(x) for x in row["x"])):
            raise ValueError("INPUT_SCOPE_INVALID")
        a, b = Decimal(before["base_unit_nav"]), Decimal(before["target_unit_nav"])
        actual = "UP" if b > a else "DOWN" if b < a else "FLAT"
        if row["old_row_hash"] != digest(before) or row["actual_direction"] != before["actual_direction"]:
            raise ValueError("OLD_ROW_BINDING_CHANGED")
        if row["actual_direction"] != actual:
            raise ValueError("EXACT_LABEL_MISMATCH")
        report = legal_report(reports, catalog, day)
        if (report["raw"]["sha256"] != row["report_sha256"] or publication(report) != row["report_publication"]
                or report["report_end"] != row["report_end"]):
            raise ValueError("REPORT_SELECTION_MISMATCH")
        report_id = row["report_end"] + "_" + report["report_type"]
        changed = [i for i, (x, y) in enumerate(zip(before["x"], row["x"], strict=True)) if x != y]
        differences.append({"target": day, "feature_indices_changed": changed,
                            "report_changed": before["exposure"]["report_raw_sha256"] != row["report_sha256"]})
        regime = "STATEMENT_WITH_OLD_SNAPSHOT" if day in gap else (
            "BEFORE_STATEMENT" if day <= protocol["scope"]["before_end"] else "AFTER_NEW_SNAPSHOT")
        daily.append({"target": day, "actual_direction": actual, "regime": regime,
                      "quarter": "2024Q" + str((int(day[5:7]) - 1) // 3 + 1), "report_id": report_id,
                      "legacy_input_hash": digest(before), "repaired_input_hash": digest(row),
                      "directions": {n: predictions[n][day]["direction"] for n in MODELS}})
        if day in gap:
            days = row["quote_dates"]
            # 指数日历为原冻结交易日历；不得删掉停牌日或缩短股票成交额窗口。
            calendar = sorted(indices["000300.SH"]["rows"])
            end = calendar.index(row["base"])
            if days != calendar[end - 20:end + 1] or max(days) >= day:
                raise ValueError("QUOTE_WINDOW_CHANGED")
            recomputed = recompute_exposure(report, days, quotes, indices)
            delta = max(abs(x - y) for x, y in zip(recomputed, row["x"][7:], strict=True))
            if delta > 1e-12:
                raise ValueError("INDEPENDENT_EXPOSURE_MISMATCH")
            trace.append({"target": day, "base": row["base"], "report_sha256": row["report_sha256"],
                          "publication": publication(report), "holdings_as_of": report["report_end"],
                          "holding_count": len(report["holdings"]), "quote_dates": days,
                          "original_13_values": dict(zip(plan_features[7:], row["x"][7:], strict=True)),
                          "decimal_recomputed_13": recomputed, "max_abs_difference": delta,
                          "old_13_equal": before["x"][7:] == row["x"][7:]})
    grouped = {"all": summarize(daily)}
    for key in ("regime", "quarter", "report_id"):
        grouped[key] = {value: summarize([r for r in daily if r[key] == value])
                        for value in sorted({r[key] for r in daily})}
    fit_inventory_unchanged = True
    for run_path, expected in protocol["existing_fit_file_inventory"].items():
        path = Path(run_path)
        actual_files = [str(p.relative_to(path)) for p in sorted(path.rglob("*")) if p.is_file()
                        and p.parent.name in ["predictions", "attempts", "started", "classifier-started",
                                             "classifier-completed", "fit-complete"]]
        if actual_files != expected or any("FULL" in p for p in actual_files):
            raise ValueError("FIT_INVENTORY_CHANGED_OR_FULL_FOUND")
    training_counts = {name: len(branch(protocol, name, ["train"]))
                       for name in ("old_inputs", "original_inputs", "inputs")}
    if training_counts != {"old_inputs": 4422, "original_inputs": 4419, "inputs": 5137}:
        raise ValueError("FROZEN_TRAINING_POOL_CHANGED")
    availability = {
        "L20_N7_repaired_2024": "UNAVAILABLE_HISTORICAL_GATE_STOPPED",
        "L20_recovered_2024": "UNAVAILABLE_HISTORICAL_GATE_STOPPED",
        "saved_2024_original_controls": {n: len(v) for n, v in predictions.items()},
        "legacy_training_records": 4422, "original_repaired_training_records": 4419,
        "recovered_training_records": 5137,
        "changed_days_by_feature": {f: sum(i in r["feature_indices_changed"] for r in differences)
                                    for i, f in enumerate(plan_features)},
        "report_changed_days": sum(r["report_changed"] for r in differences),
        "any_vector_changed_days": sum(bool(r["feature_indices_changed"]) for r in differences),
        "fit_inventory_unchanged": fit_inventory_unchanged,
        "saved_prediction_hashes_verified": 690,
        "L20_gain_established": False,
    }
    if branch(protocol, "prior_decision", ["passed"]) or branch(protocol, "recovered_decision", ["full_executed"]):
        raise ValueError("FROZEN_DECISION_CHANGED")
    for name, value in [("input-trace.json", trace), ("input-differences.json", differences),
                        ("daily-saved-results.json", daily), ("comparison.json", grouped),
                        ("result-availability.json", availability)]:
        save_once(directory / name, value)
    receipt = {"verified_sources": len(protocol["sources"]), "development_rows": len(daily),
               "independently_recomputed_rows": len(trace), "new_fits": 0, "generated_predictions": 0,
               "cumulative_fits": 58, "counts_by_regime": dict(Counter(r["regime"] for r in daily))}
    save_once(directory / "calculation-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))
