"""转债资产的固定离线对照；外部证据清单限定范围，不改线上基金分组或预测。"""

import warnings
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_comparison as comparison
from app.services import direction_1d_market_study as market
from app.services import direction_1d_training as original
from app.services.direction_1d_protocol import FEATURES, RECIPE, ZONE, calendar, digest
from app.services.direction_training_artifacts import file_hash

CANDIDATES = ("CB_NAV7", "CB_NAV9")
INDEX_FEATURES = ("convertible_index_return_1d", "convertible_index_return_5d")
QUARTERS = comparison.QUARTERS
read, write_new = original.read, original.write_new


def fingerprint() -> dict:
    """绑定本轮独立实现与入口；复用的旧算法文件保持不变。"""
    result = market.fingerprint()
    result["convertible_code"] = file_hash(Path(__file__))
    result["convertible_cli"] = file_hash(
        Path(__file__).resolve().parents[2] / "scripts/direction_1d_convertible_study.py"
    )
    return result


def feature_names(candidate: str) -> list[str]:
    if candidate not in CANDIDATES:
        raise ValueError("CONVERTIBLE_CANDIDATE_INVALID")
    return list(FEATURES) + (list(INDEX_FEATURES) if candidate == "CB_NAV9" else [])


def vector(row: dict, candidate: str) -> list[float]:
    """两条分支共享转债样本；新增输入是指数历史涨跌小数，不是基金完整基准。"""
    feature_names(candidate)
    if row["group"] != "CN_BOND" or row["asset_class"] != "CN_CONVERTIBLE":
        raise ValueError("CONVERTIBLE_ASSET_INVALID")
    nav, index = [float(v) for v in row["x"]], row["index_input"]["x"]
    if len(nav) != 7 or len(index) != 2 or not np.isfinite([*nav, *index]).all():
        raise ValueError("CONVERTIBLE_FEATURE_INVALID")
    return nav + (list(index) if candidate == "CB_NAV9" else [])


def attach_index(rows: list[dict], mapping: dict, prices: dict) -> tuple[list[dict], dict]:
    """按外部预先核验清单选择基金，缺日失败；不硬编码某只基金或混入普通债基。"""
    if mapping["asset_class"] != "CN_CONVERTIBLE" or not mapping["funds"]:
        raise ValueError("CONVERTIBLE_MAPPING_INVALID")
    result, excluded = [], Counter()
    for row in rows:
        entry = mapping["funds"].get(row["fund_code"])
        if entry is None:
            excluded["OUTSIDE_FROZEN_CONVERTIBLE_SCOPE"] += 1
            continue
        if row["group"] != "CN_BOND" or row["kind"] != "HISTORICAL_RECONSTRUCTION" or row["u"] > "2024-12-31":
            raise ValueError("CONVERTIBLE_ASSET_OR_PERIOD_INVALID")
        if row["t"] < entry["mapping_available_from"]:
            excluded["BEFORE_VERIFIED_MAPPING_DATE"] += 1
            continue
        index = {**market.market_input(row["t"], prices), "index_code": mapping["index_code"]}
        if datetime.fromisoformat(index["available_at_assumed"]) > datetime.combine(
            date.fromisoformat(row["u"]), time(8), ZONE
        ):
            raise ValueError("CONVERTIBLE_EXAM_INPUT_NOT_AVAILABLE")
        result.append({**row, "asset_class": "CN_CONVERTIBLE", "index_input": index})
    return result, dict(excluded)


def selected_fit(root: Path, quarter: str) -> list[dict]:
    """先验证原分组504日FIT，再取转债子集；两个候选使用同一份子集。"""
    rows, history = read(root / "baseline/dataset.json"), read(root / "baseline/history.json")
    models = read(root / "baseline/models.json")
    selected = comparison.exact_fit(rows, models, quarter, comparison.availability(rows, history))
    selected, _ = attach_index(selected, read(root / "mapping.json"), read(root / "prices.json"))
    cutoff = datetime.fromisoformat(models["CN_BOND-" + quarter]["train_as_of"])
    if any(datetime.fromisoformat(r["index_input"]["available_at_assumed"]) > cutoff for r in selected):
        raise ValueError("CONVERTIBLE_FIT_INPUT_NOT_AVAILABLE")
    return selected


def validate_data(folder: Path, mapping: dict) -> tuple[dict, dict, list[str]]:
    """核验原始年线、身份、披露与授权封条；复制研究包不会重置365日留存期。"""
    seal = read(folder / "data-seal.json")
    if digest(seal) != read(folder / "data-seal-receipt.json")["hash"]:
        raise ValueError("CONVERTIBLE_DATA_SEAL_CHANGED")

    def checked(name: str) -> dict:
        path = folder / name
        if Path(name).name != name or path.is_symlink() or seal["files"].get(name) != file_hash(path):
            raise ValueError("CONVERTIBLE_RAW_EVIDENCE_CHANGED")
        names.add(name)
        return read(path)

    names = {"data-seal.json", "data-seal-receipt.json"}
    inventory = checked("source-inventory.json")
    sources = [s for s in inventory["sources"] if s["source_code"] == "TUSHARE_PRO_FUND" and s["enabled"]]
    if (
        len(sources) != 1
        or not sources[0]["authorization_verified_at"]
        or not {"index_basic", "index_daily"}.issubset(sources[0]["authorized_api_names"])
    ):
        raise ValueError("CONVERTIBLE_SOURCE_NOT_AUTHORIZED")
    now = datetime.now(ZONE)
    if not timedelta(0) <= now - datetime.fromisoformat(inventory["checked_at"]) <= timedelta(days=1):
        raise ValueError("CONVERTIBLE_AUTHORIZATION_AUDIT_STALE")
    code = mapping["index_code"]
    identity = checked("identity-" + code + ".json")
    if (
        identity["status"] != "DOWNLOADED"
        or identity["record"]["index_code"] != code
        or identity["record"]["list_date"] > "2021-01-04"
    ):
        raise ValueError("CONVERTIBLE_INDEX_IDENTITY_INVALID")
    prices, acquired = {}, []
    for year in (2021, 2022, 2023, 2024):
        raw = checked(f"daily-{code}-{year}.json")
        if raw["status"] != "DOWNLOADED" or raw["api"] != "index_daily" or raw["code"] != code or raw["year"] != year:
            raise ValueError("CONVERTIBLE_DAILY_IDENTITY_INVALID")
        started, retrieved = datetime.fromisoformat(raw["started_at"]), datetime.fromisoformat(raw["retrieved_at"])
        if not started <= retrieved <= now:
            raise ValueError("CONVERTIBLE_ACQUISITION_INVALID")
        acquired.append(started)
        for item in raw["prices"]:
            day = item["date"]
            if day in prices or date.fromisoformat(day).year != year:
                raise ValueError("CONVERTIBLE_DAILY_DUPLICATE_OR_PERIOD_INVALID")
            prices[day] = item["close"]
    expected = {str(d) for d in calendar()[0] if 2021 <= d.year <= 2024}
    if set(prices) != expected or any(not np.isfinite(float(v)) or float(v) <= 0 for v in prices.values()):
        raise ValueError("CONVERTIBLE_INDEX_DAILY_INCOMPLETE")
    if prices != checked("new-prices.json")["prices"][code]:
        raise ValueError("CONVERTIBLE_DAILY_ASSEMBLY_CHANGED")
    for fund, entry in mapping["funds"].items():
        metadata = checked(entry["evidence_metadata"])
        if (
            metadata["id"] != fund
            or metadata["status"] != "SAVED"
            or metadata["published"] >= entry["mapping_available_from"]
        ):
            raise ValueError("CONVERTIBLE_MAPPING_DISCLOSURE_INVALID")
        raw_name = metadata["file"]
        if (
            Path(raw_name).name != raw_name
            or seal["files"].get(raw_name) != metadata["sha256"]
            or file_hash(folder / raw_name) != metadata["sha256"]
        ):
            raise ValueError("CONVERTIBLE_DISCLOSURE_CHANGED")
        names.add(raw_name)
        # 披露语义由训练前人工核验，程序绑定那份证据和目录身份，防止复制时换成别的基金。
        registered = [r for r in inventory["funds"] if r["fund_code"] == fund]
        if (
            len(registered) != 1
            or entry["fund_name_in_document"] not in metadata["heading"]
            or entry["benchmark_name_in_document"] not in registered[0]["benchmark"]
        ):
            raise ValueError("CONVERTIBLE_FUND_IDENTITY_INVALID")
        if not any(entry["benchmark_name_in_document"] in hit["text"] for hit in metadata["evidence_hits"]):
            raise ValueError("CONVERTIBLE_BENCHMARK_EVIDENCE_MISSING")
    if sources[0]["retention_days"] <= 0:
        raise ValueError("CONVERTIBLE_RETENTION_INVALID")
    expires = min(acquired) + timedelta(days=sources[0]["retention_days"])
    if now >= expires:
        raise ValueError("CONVERTIBLE_SOURCE_RETENTION_EXPIRED")
    source = {
        "data_seal_hash": digest(seal),
        "index_identity": identity,
        "acquired_at": min(acquired).isoformat(),
        "source_expires_at": expires.isoformat(),
        "new_api_calls": 0,
        "reused_annual_requests": 4,
        "database_writes": 0,
        "historical_first_versions_verified": False,
    }
    return prices, source, sorted(names)


def freeze(base: Path, data: Path, mapping_path: Path, root: Path) -> dict:
    """拟合前冻结转债范围、共同考题和十次预算；原普通债基分组只作为次要参考。"""
    comparison.verify(base)
    mapping = read(mapping_path)
    own_codes = {r["fund_code"] for r in read(base / "baseline/dataset.json")}
    if not set(mapping["funds"]).issubset(own_codes):
        raise ValueError("CONVERTIBLE_OUTSIDE_OWNER_BASELINE")
    prices, source, evidence_names = validate_data(data, mapping)
    exam, exclusions = attach_index(read(base / "common-exam.json"), mapping, prices)
    if not exam or {r["fund_code"] for r in exam} != set(mapping["funds"]):
        raise ValueError("CONVERTIBLE_EXAM_COVERAGE_MISSING")
    root.mkdir(parents=True, exist_ok=False)
    (root / "baseline").mkdir()
    files = ["baseline/" + name for name in comparison.BASE_FILES]
    for name in comparison.BASE_FILES:
        write_new(root / "baseline" / name, read(base / "baseline" / name))
    for name, content in (
        ("mapping.json", mapping),
        ("prices.json", prices),
        ("common-exam.json", exam),
        ("source.json", source),
    ):
        write_new(root / name, content)
        files.append(name)
    (root / "evidence").mkdir()
    for name in evidence_names:
        with (root / "evidence" / name).open("xb") as stream:
            stream.write((data / name).read_bytes())
        files.append("evidence/" + name)
    fits = {q: selected_fit(root, q) for q in (*QUARTERS, "FINAL")}
    for rows in fits.values():
        if (
            {r["fund_code"] for r in rows} != set(mapping["funds"])
            or len({r["t"] for r in rows}) < 252
            or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30
        ):
            raise ValueError("CONVERTIBLE_INSUFFICIENT_SAMPLES")
    spec = {
        "kind": "HISTORICAL_CONVERTIBLE_COMPARISON_ONLY",
        "created_at": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "recipe": RECIPE,
        "threshold": 0.5,
        "max_main_fits": 10,
        "max_replay_fits": 10,
        "fund_codes": sorted(mapping["funds"]),
        "source": source,
        "input_files": {name: file_hash(root / name) for name in files},
        "fit_hashes": {q: digest(rows) for q, rows in fits.items()},
        "fit_counts": {q: len(rows) for q, rows in fits.items()},
        "exam_count": len(exam),
        "exam_exclusions": exclusions,
        "primary_reference": CANDIDATES[0],
        "primary_metric": "FAMILY_DATE_WEIGHTED_ACCURACY",
        "bootstrap_seed": 20260912,
        "bootstrap_repetitions": 2000,
        "bootstrap_block_days": 5,
        "protected_years_excluded": [2025, 2026],
        "model_released": False,
    }
    spec["cohort_id"] = (
        "D1-CB-" + digest({"inputs": spec["input_files"], "recipe": RECIPE, "candidates": CANDIDATES})[:24]
    )
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"hash": digest(spec)})
    return {
        k: spec[k] for k in ("cohort_id", "created_at", "fund_codes", "fit_counts", "exam_count", "exam_exclusions")
    }


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if digest(spec) != read(root / "study-receipt.json")["hash"]:
        raise ValueError("CONVERTIBLE_STUDY_CHANGED")
    if spec["fingerprint"] != fingerprint() or spec["candidates"] != list(CANDIDATES) or spec["recipe"] != RECIPE:
        raise ValueError("CONVERTIBLE_CODE_OR_RECIPE_CHANGED")
    for name, expected in spec["input_files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or file_hash(path) != expected:
            raise ValueError("CONVERTIBLE_FROZEN_INPUT_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source"]["source_expires_at"]):
        raise ValueError("CONVERTIBLE_SOURCE_RETENTION_EXPIRED")
    return spec


def predict(model: dict, rows: list[dict]) -> np.ndarray:
    x = np.asarray([vector(r, model["candidate"]) for r in rows])
    z = ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def fit(rows: list[dict], candidate: str, spec: dict, cutoff: str) -> dict:
    if len({r["t"] for r in rows}) < 252 or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30:
        raise ValueError("CONVERTIBLE_INSUFFICIENT_SAMPLES")
    x, y, weights = (
        np.asarray([vector(r, candidate) for r in rows]),
        np.asarray([r["y"] for r in rows]),
        original.weights(rows),
    )
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaler = StandardScaler().fit(x, sample_weight=weights)
        estimator = LogisticRegression(**RECIPE).fit(scaler.transform(x), y, sample_weight=weights)
    model = {
        "candidate": candidate,
        "kind": "RESEARCH_ONLY",
        "target_definition": "UNIT_NAV_DIRECTION_V1",
        "cohort_id": spec["cohort_id"],
        "model_released": False,
        "train_as_of": cutoff,
        "features": feature_names(candidate),
        "recipe": RECIPE,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "fit_hash": digest(rows),
        "fit_count": len(rows),
        "distinct_dates": len({r["t"] for r in rows}),
        "weight_hash": digest(weights.tolist()),
        "weight_total": float(weights.sum()),
        "classes": {str(y): sum(r["y"] == y for r in rows) for y in (0, 1)},
        "majority": int(float(np.dot(y, weights)) > float(weights.sum()) / 2),
    }
    difference = float(np.max(np.abs(predict(model, rows) - estimator.predict_proba(scaler.transform(x))[:, 1])))
    if difference > 1e-12:
        raise ValueError("CONVERTIBLE_MODEL_RESTORE_MISMATCH")
    model["restore_max_score_diff"] = difference
    return model


def run(root: Path, *, replay: bool = False) -> dict:
    spec = verify_inputs(root)
    if replay:
        verify(root)
    out = root / ("replay" if replay else "main")
    out.mkdir(exist_ok=False)
    write_new(out / "budget-reserved.json", {"at": datetime.now(ZONE).isoformat(), "max_fits": 10})
    models = {}
    baseline_models = read(root / "baseline/models.json")
    for quarter in (*QUARTERS, "FINAL"):
        selected = selected_fit(root, quarter)
        if digest(selected) != spec["fit_hashes"][quarter]:
            raise ValueError("CONVERTIBLE_FIT_CHANGED")
        cutoff = baseline_models["CN_BOND-" + quarter]["train_as_of"]
        for candidate in CANDIDATES:
            key = candidate + "-" + quarter
            write_new(
                out / (key + "-reserved.json"), {"at": datetime.now(ZONE).isoformat(), "fit_hash": digest(selected)}
            )
            model = fit(selected, candidate, spec, cutoff)
            write_new(out / (key + ".json"), model)
            models[key] = model
    exam, predictions = read(root / "common-exam.json"), []
    initial_majority = models[CANDIDATES[0] + "-2024Q1"]["majority"]
    for r in exam:
        scores = {c: float(predict(models[c + "-" + r["quarter"]], [r])[0]) for c in CANDIDATES}
        scores[comparison.BASELINE] = r["baseline_score"]
        predictions.append(
            {
                **{k: r[k] for k in ("fund_code", "family", "group", "t", "u", "y", "actual_direction", "quarter")},
                "kind": "HISTORICAL_RECONSTRUCTION",
                "scores": scores,
                "directions": {
                    **{c: int(s > 0.5) for c, s in scores.items()},
                    "ALWAYS_UP": 1,
                    "ALWAYS_NON_UP": 0,
                    "INITIAL_MAJORITY": initial_majority,
                    "MOMENTUM": r["momentum"],
                },
            }
        )
    write_new(out / "models.json", models)
    write_new(out / "predictions.json", predictions)
    result = {
        "finished_at": datetime.now(ZONE).isoformat(),
        "successful_fits": len(models),
        "models_hash": digest(models),
        "predictions_hash": digest(predictions),
        "model_released": False,
    }
    write_new(out / "completion.json", result)
    if replay:
        prior = read(root / "main/predictions.json")
        difference = max(
            abs(a["scores"][c] - b["scores"][c]) for a, b in zip(prior, predictions, strict=True) for c in CANDIDATES
        )
        if models != read(root / "main/models.json") or prior != predictions:
            raise ValueError("CONVERTIBLE_REPLAY_MISMATCH")
        write_new(
            root / "replay-proof.json",
            {
                "models_equal": True,
                "max_score_diff": difference,
                "prediction_count": len(predictions),
                "successful_fits": len(models),
            },
        )
    return result


def verify(root: Path) -> dict:
    spec = verify_inputs(root)
    expected_keys = {c + "-" + q for c in CANDIDATES for q in (*QUARTERS, "FINAL")}
    for mode in ("main", "replay"):
        folder = root / mode
        if not (folder / "completion.json").exists():
            if mode == "main":
                raise ValueError("CONVERTIBLE_TRAINING_INCOMPLETE")
            continue
        complete = read(folder / "completion.json")
        models, predictions = read(folder / "models.json"), read(folder / "predictions.json")
        if digest(models) != complete["models_hash"] or digest(predictions) != complete["predictions_hash"]:
            raise ValueError("CONVERTIBLE_OUTPUT_CHANGED")
        if set(models) != expected_keys or complete["successful_fits"] != 10:
            raise ValueError("CONVERTIBLE_FIT_BUDGET_MISMATCH")
        if {p.name for p in folder.glob("*-reserved.json")} != {
            "budget-reserved.json",
            *(k + "-reserved.json" for k in expected_keys),
        }:
            raise ValueError("CONVERTIBLE_FIT_RESERVATION_MISMATCH")
        for key, model in models.items():
            quarter = key.rsplit("-", 1)[1]
            if model != read(folder / (key + ".json")) or model["fit_hash"] != spec["fit_hashes"][quarter]:
                raise ValueError("CONVERTIBLE_MODEL_CHANGED")
        if len(predictions) != spec["exam_count"]:
            raise ValueError("CONVERTIBLE_QUESTION_COUNT_CHANGED")
    return read(root / "main/completion.json")


def paired(rows: list[dict], candidate: str, reference: str, spec: dict) -> dict:
    # 复用已验证的目标日块重采样算法，仅在副本中设置对照键，不改变保存的旧分组分数。
    copied = [{**r, "directions": {**r["directions"], comparison.BASELINE: r["directions"][reference]}} for r in rows]
    return {"reference": reference, **comparison.paired(copied, candidate, spec)}


def evaluate(root: Path) -> dict:
    verify(root)
    spec, proof = read(root / "study.json"), read(root / "replay-proof.json")
    if not proof["models_equal"] or proof["max_score_diff"] != 0:
        raise ValueError("CONVERTIBLE_REPLAY_REQUIRED")
    for name in ("models.json", "predictions.json"):
        if read(root / "main" / name) != read(root / "replay" / name):
            raise ValueError("CONVERTIBLE_REPLAY_OUTPUT_CHANGED")
    rows = read(root / "main/predictions.json")
    branches = (*CANDIDATES, comparison.BASELINE, "ALWAYS_UP", "ALWAYS_NON_UP", "INITIAL_MAJORITY", "MOMENTUM")
    result = {
        "kind": "DEVELOPMENT_COMPARISON_NOT_INDEPENDENT_TEST",
        "overall": {b: comparison.metrics(rows, b) for b in branches},
        "strata": {
            field: {
                value: {b: comparison.metrics([r for r in rows if r[field] == value], b) for b in branches}
                for value in sorted({r[field] for r in rows})
            }
            for field in ("quarter", "group", "fund_code")
        },
        "paired_primary": paired(rows, "CB_NAV9", "CB_NAV7", spec),
        "paired_old_group": paired(rows, "CB_NAV9", comparison.BASELINE, spec),
        "runtime_model_changed": False,
        "model_released": False,
        "limitations": [
            "2021—2024已使用开发数据",
            "历史收盘/公告可用时间采用假设",
            "仅本人已核验转债基金子集，当前只有一只，不能外推其他转债基金",
            "历史映射从核实披露后向后沿用，未穷尽所有变更公告",
            "转债指数仅为历史基准组成部分，不是完整基准或实际持仓",
            "未证明未来有效，未进行参数搜索",
        ],
    }
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"hash": digest(result), "at": datetime.now(ZONE).isoformat()})
    return result
