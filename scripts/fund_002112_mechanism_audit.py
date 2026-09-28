"""独立重算机制复盘：不调用主复盘统计函数，不训练，不改历史数据。"""

import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1] / ".local-runs/fund-exposure-002112"
OUT = ROOT / "peer-mechanism-review/20260928-v1"
NEW = ROOT / "peer-recovered-experiment/20260928-v1"
SNAPSHOT = ROOT / "training-ready/sources/98fb1c51f2582687a09ac0c404d514c2ef74520e4cb63a9a9f89a300ef150611.json"


def read(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return data["payload"] if isinstance(data, dict) and set(data) == {"hash", "payload"} else data


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject(*args, **kwargs):
    raise RuntimeError("INDEPENDENT_AUDIT_FORBIDS_FIT")


def run():
    # 审计进程内直接拒绝拟合；只读取由协议哈希约束的旧模型状态。
    for estimator in (LogisticRegression, StandardScaler, HistGradientBoostingClassifier):
        estimator.fit = reject
    protocol = read(OUT / "protocol.json")
    assert protocol["new_fit_budget"] == 0
    for path, expected in protocol["files"].items():
        assert sha(path) == expected, path
    summary, daily = read(OUT / "summary.json"), read(OUT / "daily.json")
    folds, old_folds = read(NEW / "frozen/folds.json"), read(NEW / "frozen/original-folds.json")
    inputs, original = read(NEW / "frozen/inputs.json"), read(NEW / "frozen/original-inputs.json")
    index = {r["target"]: r for fold in folds[:3] for r in fold["exam"]}
    assert len(index) == len(daily) == 161
    records = {r["target"]: r for r in daily}
    outcomes = Counter()
    max_numeric_error = 0.0
    all_controls = {"N7": [], "L20": [], "L20_RECOVERED": []}
    for new_fold, old_fold in zip(folds[:3], old_folds[:3], strict=True):
        q, exam = new_fold["name"], new_fold["exam"]
        x = np.array([r["x"] for r in exam])
        models, predictions = {}, {}
        for name, fold, pool in (("old", old_fold, original), ("new", new_fold, inputs)):
            spec = protocol["models"][q + "-" + name]
            manifest_path = Path(spec["directory"]) / (spec["slot"] + ".manifest.json")
            model_path = Path(spec["directory"]) / (spec["slot"] + ".joblib")
            manifest = read(manifest_path)
            assert sha(manifest_path) == spec["manifest_sha256"]
            assert sha(model_path) == manifest["file_sha256"]
            model = joblib.load(model_path)
            models[name] = model
            pred = read(Path(spec["directory"]).parent / "predictions" / (spec["slot"] + ".json"))
            predictions[name] = pred
            # 直接矩阵运算及 softmax，独立于旧预测函数和本次分解函数。
            scaler, classifier = model["scaler"], model["classifier"]
            logits = ((x - scaler.mean_) / scaler.scale_) @ classifier.coef_.T + classifier.intercept_
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            scores = exp / exp.sum(axis=1, keepdims=True)
            error = float(np.max(np.abs(scores - np.array([p["scores"] for p in pred["exam"]]))))
            assert error < 1e-12
            max_numeric_error = max(max_numeric_error, error)
            classes = list(classifier.classes_)
            for row, p, probs in zip(exam, pred["exam"], scores, strict=True):
                assert row["target"] == p["target"] and row["actual_direction"] == p["actual_direction"]
                assert classes[int(np.argmax(probs))] == p["direction"]
            pool_index = {(r["fund_code"], r["target"]): r for r in pool["train"]}
            train = [pool_index[tuple(k)] for k in fold["train_ids"]]
            train_x = np.array([r["x"] for r in train])
            train_logits = ((train_x - scaler.mean_) / scaler.scale_) @ classifier.coef_.T + classifier.intercept_
            e = np.exp(train_logits - train_logits.max(axis=1, keepdims=True))
            train_scores = e / e.sum(axis=1, keepdims=True)
            assert np.max(np.abs(train_scores - np.array([p["scores"] for p in pred["train"]]))) < 1e-12
            actual_flat = [i for i, r in enumerate(train) if r["actual_direction"] == "FLAT"]
            learning = summary["folds"][q]["flat_learning"][name]["all"]
            assert len(actual_flat) == learning["actual_flat_rows"]
            assert len({train[i]["target"] for i in actual_flat}) == learning["actual_flat_dates"]
            assert (
                int(np.sum(np.argmax(train_scores, axis=1) == classes.index("FLAT"))) == learning["predicted_flat"] == 0
            )
            share = sum(fold["weights"][i] for i in actual_flat) / sum(fold["weights"])
            assert abs(share - learning["flat_share_of_total_saved_weight"]) < 1e-14
        anchor = models["old"]["scaler"].mean_
        parts, offsets = {}, {}
        for name, model in models.items():
            s, c = model["scaler"], model["classifier"]
            k = list(c.classes_)
            coefficients = (c.coef_[k.index("UP")] - c.coef_[k.index("DOWN")]) / s.scale_
            parts[name] = (x - anchor) * coefficients
            offsets[name] = (
                c.intercept_[k.index("UP")] - c.intercept_[k.index("DOWN")] + (anchor - s.mean_) @ coefficients
            )
        for i, (row, old, new) in enumerate(
            zip(exam, predictions["old"]["exam"], predictions["new"]["exam"], strict=True)
        ):
            r = records[row["target"]]
            assert r["old_direction"] == old["direction"] and r["new_direction"] == new["direction"]
            assert r["x"] == row["x"] and r["actual"] == row["actual_direction"]
            a, b = old["direction"] == r["actual"], new["direction"] == r["actual"]
            outcome = {
                (True, True): "both_correct",
                (False, False): "both_wrong",
                (False, True): "gained",
                (True, False): "lost",
            }[a, b]
            assert outcome == r["outcome"]
            outcomes[outcome] += 1
            delta = parts["new"][i] - parts["old"][i]
            assert np.max(np.abs(delta - np.array([r["delta_features"][f] for f in models["new"]["features"]]))) < 1e-12
            assert abs(delta.sum() + offsets["new"] - offsets["old"] - r["delta_logit"]) < 1e-12
        for name in ("N7", "L20"):
            all_controls[name].extend(read(NEW / f"frozen/controls/{q}-{name}-main.json")["exam"])
        all_controls["L20_RECOVERED"].extend(predictions["new"]["exam"])
        assert (
            abs(
                sum(old_fold["weights"]) / sum(new_fold["weights"])
                - summary["folds"][q]["regularization"]["relative_strength_new_to_old"]
            )
            < 1e-14
        )
    assert dict(outcomes) == summary["outcomes"]
    assert sum(r["old_direction"] != r["new_direction"] for r in daily) == summary["direction_changes"] == 8
    confusion = {}
    for name, preds in all_controls.items():
        confusion[name] = {
            "predicted": dict(Counter(p["direction"] for p in preds)),
            "correct_by_class": {
                c: sum(p["direction"] == p["actual_direction"] == c for p in preds) for c in ("DOWN", "FLAT", "UP")
            },
            "matrix_actual_to_prediction": {
                c: dict(Counter(p["direction"] for p in preds if p["actual_direction"] == c))
                for c in ("DOWN", "FLAT", "UP")
            },
        }
    required = {c: max(confusion[n]["correct_by_class"][c] for n in ("N7", "L20")) for c in ("DOWN", "FLAT", "UP")}
    assert required == summary["gates"]["required_class_correct"] and sum(required.values()) == 112
    correct_sets = [
        {p["target"] for p in all_controls[n] if p["direction"] == p["actual_direction"]} for n in ("N7", "L20")
    ]
    assert len(set.union(*correct_sets)) == summary["gates"]["same_day_oracle_union_correct"] == 116
    snapshot = read(SNAPSHOT)
    nav = {r["date"]: r for r in snapshot["funds"]["002112"]["nav"]["rows"]}
    flat = read(OUT / "flat-daily.json")
    pools = {
        "historical": list(index.values()),
        "own_train": [r for r in inputs["train"] if r["fund_code"] == "002112"],
        "development_input_only": inputs["development"],
    }
    for name, rows in pools.items():
        derived = {r["target"]: r for r in flat[name]["daily"]}
        assert len(derived) == len(rows)
        for row in rows:
            target, base = row["target"], row["base"]
            assert target < "2025-01-01"
            difference = Decimal(nav[target]["nav"]) - Decimal(nav[base]["nav"])
            assert row["actual_direction"] == ("UP" if difference > 0 else "DOWN" if difference < 0 else "FLAT")
            count, dates = 0, row["nav"]["nav_dates"]
            assert all(nav[d]["ann_date"] < target for d in dates)
            for a, b in zip(dates[:0:-1], dates[-2::-1], strict=True):
                if Decimal(nav[a]["nav"]) != Decimal(nav[b]["nav"]):
                    break
                count += 1
            assert derived[target]["prior_equal_run"] == count
    reports = {r["raw"]["sha256"]: r for f in snapshot["funds"].values() for r in f["reports"]}
    manifests = read(ROOT / "peer-fold-impact/20260927-v1/report-admission-worklist.json")["reports"] + read(
        ROOT / "peer-gap-evidence/20260927-v1/report-supplements-v2.json"
    )
    for path in {r["parsed_file"] for r in manifests if r.get("parsed_file")}:
        report = read(path)
        reports[report["raw"]["sha256"]] = report
    peer = read(OUT / "peer-overlap.json")
    assert len(peer["daily"]) == 830
    for row in peer["daily"]:
        assert row["status"] == "KNOWN"
        left, right = [reports[row[k]] for k in ("target_report_sha256", "peer_report_sha256")]
        maps = [
            {h["stock_code"]: Decimal(h["nav_weight_pct"]) for h in r["holdings"] if Decimal(h["nav_weight_pct"]) > 0}
            for r in (left, right)
        ]
        common = maps[0].keys() & maps[1].keys()
        assert sorted(common) == row["common_stocks"]
        assert float(sum(min(maps[0][s], maps[1][s]) for s in common)) == row["shared_nav_pct_lower_bound"]
    cross = read(OUT / "source-crosscheck.json")
    unique_dates = set()
    for number in range(1, 9):
        receipt = read(OUT / f"public-sources/{number:02d}-receipt.json")
        raw_path = OUT / f"public-sources/{number:02d}-response.json"
        assert receipt["valid"] and sha(raw_path) == receipt["response_sha256"]
        data = read(raw_path)
        raw_rows = data["dataList"] if receipt["provider"] == "dbfund" else data["Data"]["LSJZList"]
        independent_rows = [
            {"date": r["date"], "nav": str(r["netvalue"])}
            if receipt["provider"] == "dbfund"
            else {"date": r["FSRQ"], "nav": str(r["DWJZ"])}
            for r in raw_rows
        ]
        assert sorted(independent_rows, key=lambda r: r["date"]) == receipt["rows"]
        for row in independent_rows:
            assert receipt["window"][0] <= row["date"] <= receipt["window"][1] < "2025-01-01"
            assert Decimal(row["nav"]) == Decimal(nav[row["date"]]["nav"])
            unique_dates.add(row["date"])
    assert cross["requests"] == cross["valid_requests"] == 8 and not cross["disagreements"]
    assert len(unique_dates) == 24 and len(cross["comparisons"]) == 48
    result = {
        "passed": True,
        "protocol_sources_verified": len(protocol["files"]),
        "models_verified": 6,
        "daily_records_verified": 161,
        "flat_source_and_prior_window_records_verified": sum(map(len, pools.values())),
        "peer_rows_verified": 830,
        "public_source_requests": 8,
        "unique_nav_dates_crosschecked": 24,
        "public_values_disagreed": 0,
        "max_exam_score_error": max_numeric_error,
        "confusion": confusion,
        "actual_new_fits": 0,
        "cumulative_actual_fits": 58,
        "script_sha256": sha(__file__),
        "evidence_hashes": {
            name: sha(OUT / name)
            for name in ("summary.json", "daily.json", "flat-daily.json", "peer-overlap.json", "source-crosscheck.json")
        },
    }
    with (OUT / ("independent-audit-" + sha(__file__)[:12] + ".json")).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    run()
