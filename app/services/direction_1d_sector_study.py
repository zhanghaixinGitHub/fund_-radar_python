"""基金对应指数的固定离线对照；外部清单决定范围，旧模型和真实预测只读。"""

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
from app.services.direction_training_artifacts import file_hash, read_seal

GROUPS = market.GROUPS
CANDIDATES = ("COMMON9", "SPECIFIC11")
MARKET_FEATURES = market.MARKET_FEATURES
SPECIFIC_FEATURES = ("assigned_index_minus_hs300_return_1d", "assigned_index_minus_hs300_return_5d")
QUARTERS = comparison.QUARTERS
read, write_new = original.read, original.write_new


def fingerprint() -> dict:
    """沿用已冻结算法的版本指纹，另绑定本轮代码及入口，后续研究不覆盖本文件。"""
    result = market.fingerprint()
    result["sector_code"] = file_hash(Path(__file__))
    project = Path(__file__).resolve().parents[2]
    for name in ("direction_1d_sector_study.py", "direction_1d_sector_acquire.py"):
        result[name] = file_hash(project / "scripts" / name)
    return result


def feature_names(candidate: str) -> list[str]:
    if candidate not in CANDIDATES:
        raise ValueError("SECTOR_CANDIDATE_INVALID")
    return list(FEATURES) + list(MARKET_FEATURES) + (list(SPECIFIC_FEATURES) if candidate == "SPECIFIC11" else [])


def vector(row: dict, candidate: str) -> list[float]:
    """两条分支同一人口；仅增强分支多出对应指数相对沪深300的两个涨跌输入。"""
    feature_names(candidate)
    if row["group"] not in GROUPS:
        raise ValueError("SECTOR_GROUP_INVALID")
    nav = [float(v) for v in row["x"]]
    shared, specific = row["market_input"]["x"], row["specific_input"]["x"]
    if (len(nav), len(shared), len(specific)) != (7, 2, 2) or not np.isfinite([*nav, *shared, *specific]).all():
        raise ValueError("SECTOR_FEATURE_INVALID")
    return nav + list(shared) + (list(specific) if candidate == "SPECIFIC11" else [])


def attach_specific(rows: list[dict], mappings: dict, prices: dict) -> tuple[list[dict], dict]:
    """按照事先核验的映射日期筛选；不向过去延展映射，不插补缺日，不读取U日收盘。"""
    result, excluded, cache = [], Counter(), {}
    for row in rows:
        mapping = mappings.get(row["fund_code"])
        if mapping is None:
            raise ValueError("SECTOR_FUND_COVERAGE_MISSING")
        if row["group"] not in GROUPS or mapping["status"] != "RESEARCH_MAPPING_SUPPORTED":
            excluded[mapping["status"]] += 1
            continue
        if row["group"] != mapping["group"]:
            raise ValueError("SECTOR_MAPPING_GROUP_CHANGED")
        if row["t"] < mapping["mapping_available_from"]:
            excluded["BEFORE_VERIFIED_MAPPING_DATE"] += 1
            continue
        code, anchor = mapping["index_code"], row["t"]
        for index in ("000300.SH", code):
            if (index, anchor) not in cache:
                raw = market.market_input(anchor, prices.get(index, {}))
                cache[index, anchor] = {**raw, "index_code": index}
        common, own = cache["000300.SH", anchor], cache[code, anchor]
        own = {
            **own,
            "index_returns": own["x"],
            "x": [a - b for a, b in zip(own["x"], common["x"], strict=True)],
            "mapping_available_from": mapping["mapping_available_from"],
            "meaning": "PRICE_RETURN_DIFFERENCE_NOT_FULL_BENCHMARK_OR_HOLDINGS",
        }
        result.append({**row, "market_input": common, "specific_input": own})
    return result, dict(excluded)


def selected_fit(root: Path, quarter: str) -> list[dict]:
    """先校验旧504日FIT及净值可用时间，再统一应用映射日期和指数连续性条件。"""
    rows = read(root / "baseline/dataset.json")
    available = comparison.availability(rows, read(root / "baseline/history.json"))
    models = read(root / "baseline/models.json")
    selected = comparison.exact_fit(rows, models, quarter, available)
    selected, _ = attach_specific(selected, read(root / "mapping.json")["funds"], read(root / "prices.json")["prices"])
    cutoff = datetime.fromisoformat(models["CN_EQUITY-" + quarter]["train_as_of"])
    if any(datetime.fromisoformat(r["specific_input"]["available_at_assumed"]) > cutoff for r in selected):
        raise ValueError("SECTOR_FIT_INPUT_NOT_AVAILABLE")
    return selected


def validate_data(folder: Path, previous: dict) -> dict:
    """验证采集封条、来源权限及原始留存起点；冻结研究不会重置数据留存期。"""
    seal = read(folder / "data-seal.json")
    if digest(seal) != read(folder / "data-seal-receipt.json")["hash"]:
        raise ValueError("SECTOR_DATA_SEAL_CHANGED")
    for name, expected in seal["files"].items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink() or file_hash(path) != expected:
            raise ValueError("SECTOR_RAW_EVIDENCE_CHANGED")
    new = read(folder / "new-prices.json")
    if digest(new) != read(folder / "new-prices-receipt.json")["hash"]:
        raise ValueError("SECTOR_NEW_PRICES_CHANGED")
    sources = [
        s
        for s in read(folder / "source-inventory.json")["sources"]
        if s["source_code"] == "TUSHARE_PRO_FUND" and s["enabled"]
    ]
    if (
        len(sources) != 1
        or not sources[0]["authorization_verified_at"]
        or not {"index_basic", "index_daily"}.issubset(sources[0]["authorized_api_names"])
    ):
        raise ValueError("SECTOR_SOURCE_NOT_AUTHORIZED")
    checked = datetime.fromisoformat(read(folder / "source-inventory.json")["checked_at"])
    now = datetime.now(ZONE)
    if not timedelta(0) <= now - checked <= timedelta(days=1):
        raise ValueError("SECTOR_AUTHORIZATION_AUDIT_STALE")
    if len(new["requests"]) > 52 or seal["api_calls"]["index_daily"] != len(new["requests"]):
        raise ValueError("SECTOR_DAILY_BUDGET_INVALID")
    for r in new["requests"]:
        if file_hash(folder / r["file"]) != r["hash"] or r["year"] not in (2021, 2022, 2023, 2024):
            raise ValueError("SECTOR_DAILY_REQUEST_CHANGED")
    combined = read(folder / "prices.json")
    old_dir = Path(combined["old_market_dir"])
    old_seal = read_seal(old_dir, "market-acquired.json")
    if file_hash(old_dir / "market-data.json") != combined["old_market_hash"] or combined[
        "new_prices_hash"
    ] != file_hash(folder / "new-prices.json"):
        raise ValueError("SECTOR_OLD_SOURCE_CHANGED")
    old_prices = read(old_dir / "market-data.json")["prices"]
    if old_prices.keys() & new["prices"].keys() or combined["prices"] != {**old_prices, **new["prices"]}:
        raise ValueError("SECTOR_COMBINED_PRICES_CHANGED")
    acquired = min(datetime.fromisoformat(r["started_at"]) for r in new["requests"])
    if acquired > now or sources[0]["retention_days"] <= 0:
        raise ValueError("SECTOR_ACQUISITION_OR_RETENTION_INVALID")
    expires = min(
        datetime.fromisoformat(previous["source"]["source_expires_at"]),
        acquired + timedelta(days=sources[0]["retention_days"]),
    )
    if now >= expires:
        raise ValueError("SECTOR_SOURCE_RETENTION_EXPIRED")
    expected = {str(d) for d in calendar()[0] if 2021 <= d.year <= 2024}
    mapping = read(folder / "mapping-reviewed.json")
    for row in mapping["funds"].values():
        if row["status"] == "RESEARCH_MAPPING_SUPPORTED":
            values = combined["prices"].get(row["index_code"], {})
            if set(values) != expected or any(not np.isfinite(float(v)) or float(v) <= 0 for v in values.values()):
                raise ValueError("SECTOR_QUALIFIED_INDEX_DAILY_INCOMPLETE")
            if not row["evidence_files"] or any(name not in seal["files"] for name in row["evidence_files"]):
                raise ValueError("SECTOR_MAPPING_EVIDENCE_MISSING")
    return {
        "data_seal_hash": digest(seal),
        "old_market_manifest": old_seal["manifest_hash"],
        "old_source": previous["source"],
        "new_acquired_at": acquired.isoformat(),
        "source_expires_at": expires.isoformat(),
        "api_calls": seal["api_calls"],
        "database_writes": 0,
        "historical_first_versions_verified": False,
    }


def freeze(base: Path, data: Path, root: Path) -> dict:
    """训练前封存两个候选共享的人口、FIT与考题；两候选各5次主拟合及一次复现。"""
    market.verify(base)
    previous = read(base / "study.json")
    source = validate_data(data, previous)
    mapping, prices = read(data / "mapping-reviewed.json"), read(data / "prices.json")
    original_codes = {r["fund_code"] for r in read(base / "baseline/dataset.json")}
    if set(mapping["funds"]) != original_codes:
        raise ValueError("SECTOR_ALL_FUND_COVERAGE_MISMATCH")
    exam, exclusions = attach_specific(read(base / "common-exam.json"), mapping["funds"], prices["prices"])
    if not exam or any(
        datetime.fromisoformat(r["specific_input"]["available_at_assumed"])
        > datetime.combine(date.fromisoformat(r["u"]), time(8), ZONE)
        for r in exam
    ):
        raise ValueError("SECTOR_EXAM_INPUT_NOT_AVAILABLE")
    root.mkdir(parents=True, exist_ok=False)
    (root / "baseline").mkdir()
    for name in comparison.BASE_FILES:
        write_new(root / "baseline" / name, read(base / "baseline" / name))
    for name, content in (
        ("mapping.json", mapping),
        ("prices.json", prices),
        ("common-exam.json", exam),
        ("source.json", source),
    ):
        write_new(root / name, content)
    # 只复制已经取得且校验过的证据，不再次联网；冻结文件在后续训练中逐个校验。
    evidence = root / "evidence"
    evidence.mkdir()
    files = [
        *("baseline/" + n for n in comparison.BASE_FILES),
        "mapping.json",
        "prices.json",
        "common-exam.json",
        "source.json",
    ]
    names = list(read(data / "data-seal.json")["files"]) + [
        "data-seal.json",
        "data-seal-receipt.json",
        "mapping-reviewed.json",
    ]
    for name in names:
        with (evidence / name).open("xb") as stream:
            stream.write((data / name).read_bytes())
        files.append("evidence/" + name)
    fits = {q: selected_fit(root, q) for q in (*QUARTERS, "FINAL")}
    spec = {
        "kind": "HISTORICAL_SECTOR_COMPARISON_ONLY",
        "created_at": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "recipe": RECIPE,
        "threshold": 0.5,
        "max_main_fits": 10,
        "max_replay_fits": 10,
        "groups": GROUPS,
        "fund_codes": sorted({r["fund_code"] for r in exam}),
        "source": source,
        "input_files": {name: file_hash(root / name) for name in files},
        "fit_hashes": {q: digest(r) for q, r in fits.items()},
        "fit_counts": {q: len(r) for q, r in fits.items()},
        "fit_fund_counts": {q: dict(sorted(Counter(r["fund_code"] for r in rows).items())) for q, rows in fits.items()},
        "exam_count": len(exam),
        "base_exam_count": len(read(base / "common-exam.json")),
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
        "D1-S-" + digest({"inputs": spec["input_files"], "recipe": RECIPE, "candidates": CANDIDATES})[:24]
    )
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"hash": digest(spec)})
    return {
        k: spec[k] for k in ("cohort_id", "created_at", "fund_codes", "fit_counts", "exam_count", "exam_exclusions")
    }


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if digest(spec) != read(root / "study-receipt.json")["hash"]:
        raise ValueError("SECTOR_STUDY_CHANGED")
    if spec["fingerprint"] != fingerprint() or spec["candidates"] != list(CANDIDATES) or spec["recipe"] != RECIPE:
        raise ValueError("SECTOR_CODE_OR_RECIPE_CHANGED")
    for name, expected in spec["input_files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or file_hash(path) != expected:
            raise ValueError("SECTOR_FROZEN_INPUT_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source"]["source_expires_at"]):
        raise ValueError("SECTOR_SOURCE_RETENTION_EXPIRED")
    return spec


def predict(model: dict, rows: list[dict]) -> np.ndarray:
    x = np.asarray([vector(r, model["candidate"]) for r in rows])
    z = ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def fit(rows: list[dict], candidate: str, spec: dict, cutoff: str) -> dict:
    if len({r["t"] for r in rows}) < 252 or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30:
        raise ValueError("SECTOR_INSUFFICIENT_SAMPLES")
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
        raise ValueError("SECTOR_MODEL_RESTORE_MISMATCH")
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
            raise ValueError("SECTOR_FIT_CHANGED")
        cutoff = baseline_models["CN_EQUITY-" + quarter]["train_as_of"]
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
            raise ValueError("SECTOR_REPLAY_MISMATCH")
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
                raise ValueError("SECTOR_TRAINING_INCOMPLETE")
            continue
        complete = read(folder / "completion.json")
        models, predictions = read(folder / "models.json"), read(folder / "predictions.json")
        if digest(models) != complete["models_hash"] or digest(predictions) != complete["predictions_hash"]:
            raise ValueError("SECTOR_OUTPUT_CHANGED")
        if set(models) != expected_keys or complete["successful_fits"] != 10:
            raise ValueError("SECTOR_FIT_BUDGET_MISMATCH")
        if {p.name for p in folder.glob("*-reserved.json")} != {
            "budget-reserved.json",
            *(k + "-reserved.json" for k in expected_keys),
        }:
            raise ValueError("SECTOR_FIT_RESERVATION_MISMATCH")
        for key, model in models.items():
            quarter = key.rsplit("-", 1)[1]
            if model != read(folder / (key + ".json")) or model["fit_hash"] != spec["fit_hashes"][quarter]:
                raise ValueError("SECTOR_MODEL_CHANGED")
        if len(predictions) != spec["exam_count"]:
            raise ValueError("SECTOR_QUESTION_COUNT_CHANGED")
    return read(root / "main/completion.json")


def paired(rows: list[dict], candidate: str, reference: str, spec: dict) -> dict:
    # 复用已验证的目标日块重采样算法，仅在副本中设置对照键，不改变保存的旧分组分数。
    copied = [{**r, "directions": {**r["directions"], comparison.BASELINE: r["directions"][reference]}} for r in rows]
    return {"reference": reference, **comparison.paired(copied, candidate, spec)}


def evaluate(root: Path) -> dict:
    verify(root)
    spec, proof = read(root / "study.json"), read(root / "replay-proof.json")
    if not proof["models_equal"] or proof["max_score_diff"] != 0:
        raise ValueError("SECTOR_REPLAY_REQUIRED")
    for name in ("models.json", "predictions.json"):
        if read(root / "main" / name) != read(root / "replay" / name):
            raise ValueError("SECTOR_REPLAY_OUTPUT_CHANGED")
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
        "paired_primary": paired(rows, "SPECIFIC11", "COMMON9", spec),
        "paired_old_group": paired(rows, "SPECIFIC11", comparison.BASELINE, spec),
        "runtime_model_changed": False,
        "model_released": False,
        "limitations": [
            "2021—2024已使用开发数据",
            "历史收盘/公告可用时间采用假设",
            "当前基金类型不能证明历史实际投资暴露",
            "历史映射从核实披露后向后沿用，未穷尽所有变更公告",
            "对应股票指数不是基金完整基准或实际持仓",
            "未证明未来有效，未进行参数搜索",
        ],
    }
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"hash": digest(result), "at": datetime.now(ZONE).isoformat()})
    return result
