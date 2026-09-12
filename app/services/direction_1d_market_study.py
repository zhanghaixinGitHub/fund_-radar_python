"""固定预算的1日市场信息对照；全程离线，不注册模型或修改既有研究包。"""

import warnings
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_comparison as comparison
from app.services import direction_1d_training as original
from app.services.direction_1d_protocol import FEATURES, RECIPE, ZONE, calendar, digest
from app.services.direction_training_artifacts import file_hash, read_seal

GROUPS = ("CN_EQUITY", "CN_MIXED")
CANDIDATES = ("NAV7_27", "MARKET9_27")
MARKET_FEATURES = ("hs300_return_1d", "hs300_return_5d")
QUARTERS = comparison.QUARTERS
read, write_new = original.read, original.write_new


def fingerprint() -> dict:
    """冻结实际执行的训练、评价、日历及本轮脚本；恢复须使用匹配的代码归档。"""
    result = comparison.fingerprint()
    result["market_code"] = file_hash(Path(__file__))
    result["artifact_code"] = file_hash(Path(__file__).with_name("direction_training_artifacts.py"))
    return result


def market_input(t: str, prices: dict) -> dict:
    """取T及之前连续6个交易日收盘；收益为小数，缺日与非法价格不插补。"""
    days, _ = calendar()
    anchor = date.fromisoformat(t)
    if anchor not in days or anchor.year > 2024 or days.index(anchor) < 5:
        raise ValueError("MARKET_ANCHOR_INVALID")
    i = days.index(anchor)
    wanted = [str(d) for d in days[i - 5 : i + 1]]
    if any(d not in prices for d in wanted):
        raise ValueError("MARKET_HISTORY_GAP")
    values = [float(prices[d]) for d in wanted]
    if not np.isfinite(values).all() or any(v <= 0 for v in values):
        raise ValueError("MARKET_PRICE_INVALID")
    return {
        "index_code": "000300.SH",
        "dates": wanted,
        "closes": [str(prices[d]) for d in wanted],
        "x": [values[-1] / values[-2] - 1, values[-1] / values[0] - 1],
        "available_at_assumed": datetime.combine(anchor, time(18), ZONE).isoformat(),
        "availability_evidence": "T_CLOSE_AT_18_ASSUMED_NOT_FIRST_RECEIPT_PROOF",
    }


def vector(row: dict, candidate: str) -> list[float]:
    if candidate not in CANDIDATES or row["group"] not in GROUPS:
        raise ValueError("MARKET_CANDIDATE_OR_GROUP_INVALID")
    x = [float(v) for v in row["x"]]
    market_x = row["market_input"]["x"]
    if len(x) != 7 or len(market_x) != 2 or not np.isfinite([*x, *market_x]).all():
        raise ValueError("MARKET_FEATURE_INVALID")
    return x + (list(market_x) if candidate == "MARKET9_27" else [])


def shared_source_lineage(origin: Path, expected_hash: str) -> list[dict]:
    """沿原包复用链追溯首次采集区间；没有逐请求时刻时不编造精确下载时间。"""
    parent, current, seen, lineage = origin.resolve().parent, origin.resolve(), set(), []
    expected_prepared = None
    for _ in range(8):
        if current.parent != parent or current in seen or current.is_symlink():
            raise ValueError("SHARED_SOURCE_LINEAGE_INVALID")
        seen.add(current)
        frozen = read_seal(current, "study-frozen.json")
        prepared = read_seal(current, "study-prepared.json")
        if (
            prepared["frozen_hash"] != frozen["manifest_hash"]
            or (expected_prepared is not None and prepared["manifest_hash"] != expected_prepared)
            or file_hash(current / "market.json") != expected_hash
        ):
            raise ValueError("SHARED_SOURCE_LINEAGE_CHANGED")
        lineage.append({"folder": current.name, "frozen": frozen, "prepared": prepared})
        reused = prepared.get("reused_source_folder")
        if not reused:
            return lineage
        expected_prepared = prepared["reused_prepared_hash"]
        current = (parent / reused).resolve()
    raise ValueError("SHARED_SOURCE_LINEAGE_TOO_DEEP")


def validate_source(market_dir: Path, audit: dict, shared_origin: Path) -> dict:
    """复核原采集封条和共享沪深300源文件；当前来源授权只使用传入的只读审计。"""
    seal = read_seal(market_dir, "market-acquired.json")
    market = read(market_dir / "market-data.json")
    shared = read(market_dir / "shared-source.json")
    raw_hash = file_hash(market_dir / "market-data.json")
    if raw_hash != audit["existing_daily_archive"]["sha256"]:
        raise ValueError("MARKET_AUDIT_SOURCE_CHANGED")
    if file_hash(market_dir / "shared-source.json") != market["catalog_receipt"]["shared_index_source_file_hash"]:
        raise ValueError("SHARED_INDEX_SOURCE_CHANGED")
    shared_prices = {r["date"]: r["close"] for r in shared["prices"]}
    prices = market["prices"]["000300.SH"]
    if len(shared_prices) != len(shared["prices"]) or {k: float(v) for k, v in shared_prices.items()} != {
        k: float(v) for k, v in prices.items()
    }:
        raise ValueError("SHARED_INDEX_PRICE_MISMATCH")
    sources = [s for s in audit["sources"] if s["source_code"] == "TUSHARE_PRO_FUND" and s["enabled"]]
    if len(sources) != 1 or not sources[0]["authorization_verified_at"]:
        raise ValueError("MARKET_SOURCE_NOT_AUTHORIZED")
    source = sources[0]
    if not {"index_basic", "index_daily"}.issubset(source["authorized_api_names"]):
        raise ValueError("MARKET_SOURCE_NOT_AUTHORIZED")
    now = datetime.now(ZONE)
    checked = datetime.fromisoformat(audit["checked_at"])
    if not timedelta(0) <= now - checked <= timedelta(days=1):
        raise ValueError("MARKET_AUTHORIZATION_AUDIT_STALE")
    # 留存从源数据实际取回起算，复制到新研究包不会重新开始留存期限。
    lineage = shared_source_lineage(shared_origin, file_hash(market_dir / "shared-source.json"))
    acquired_after = datetime.fromisoformat(lineage[-1]["frozen"]["created_at"])
    acquired_before = datetime.fromisoformat(lineage[-1]["prepared"]["created_at"])
    if not acquired_after <= acquired_before <= now:
        raise ValueError("SHARED_SOURCE_ACQUISITION_TIME_INVALID")
    expires = acquired_after + timedelta(days=source["retention_days"])
    if not source["retention_days"] or expires <= now:
        raise ValueError("MARKET_SOURCE_RETENTION_EXPIRED")
    expected = {str(d) for d in calendar()[0] if d.year in (2021, 2022, 2023, 2024)}
    if set(prices) != expected or any(not np.isfinite(float(v)) or float(v) <= 0 for v in prices.values()):
        raise ValueError("MARKET_DAILY_COVERAGE_INVALID")
    return {
        "market_acquired_manifest": seal["manifest_hash"],
        "market_data_hash": raw_hash,
        "source_code": source["source_code"],
        "authorization_checked_at": audit["checked_at"],
        "source_acquisition_interval": [acquired_after.isoformat(), acquired_before.isoformat()],
        "source_retention_basis": "ORIGINAL_ACQUISITION_INTERVAL_START_CONSERVATIVE",
        "shared_source_lineage": lineage,
        "source_expires_at": expires.isoformat(),
        "index": "000300.SH",
        "index_role": "COMMON_MARKET_CONTEXT_NOT_EVERY_FUND_BENCHMARK",
        "prices_count": len(prices),
        "historical_first_versions_verified": False,
        "provider_api_calls": 0,
    }


def attach_market(rows: list[dict], prices: dict) -> list[dict]:
    cache = {t: market_input(t, prices) for t in sorted({r["t"] for r in rows if r["group"] in GROUPS})}
    return [{**r, "market_input": cache[r["t"]]} for r in rows if r["group"] in GROUPS]


def selected_fit(root: Path, quarter: str) -> list[dict]:
    """先验证旧FIT哈希和公告时间，再筛出固定的两组；两个候选共用这一份。"""
    rows = read(root / "baseline/dataset.json")
    available = comparison.availability(rows, read(root / "baseline/history.json"))
    models = read(root / "baseline/models.json")
    selected = comparison.exact_fit(rows, models, quarter, available)
    cutoff = datetime.fromisoformat(models["CN_EQUITY-" + quarter]["train_as_of"])
    selected = attach_market(selected, read(root / "market-data.json")["prices"]["000300.SH"])
    if any(datetime.fromisoformat(r["market_input"]["available_at_assumed"]) > cutoff for r in selected):
        raise ValueError("MARKET_FIT_INPUT_NOT_AVAILABLE")
    return selected


def freeze(base: Path, market_dir: Path, audit_path: Path, shared_origin: Path, root: Path) -> dict:
    comparison.verify(base)
    audit = read(audit_path)
    source = validate_source(market_dir, audit, shared_origin)
    market = read(market_dir / "market-data.json")
    exam = attach_market(read(base / "common-exam.json"), market["prices"]["000300.SH"])
    if any(
        datetime.fromisoformat(r["market_input"]["available_at_assumed"])
        > datetime.combine(date.fromisoformat(r["u"]), time(8), ZONE)
        for r in exam
    ):
        raise ValueError("MARKET_EXAM_INPUT_NOT_AVAILABLE")
    root.mkdir(parents=True, exist_ok=False)
    (root / "baseline").mkdir()
    for name in comparison.BASE_FILES:
        write_new(root / "baseline" / name, read(base / "baseline" / name))
    for name in ("market-data.json", "market-acquired.json", "shared-source.json"):
        with (root / name).open("xb") as f:
            f.write((market_dir / name).read_bytes())
    write_new(root / "source-audit.json", audit)
    write_new(root / "common-exam.json", exam)
    fits = {q: selected_fit(root, q) for q in (*QUARTERS, "FINAL")}
    files = [
        *("baseline/" + n for n in comparison.BASE_FILES),
        "market-data.json",
        "market-acquired.json",
        "shared-source.json",
        "source-audit.json",
        "common-exam.json",
    ]
    spec = {
        "kind": "HISTORICAL_MARKET_COMPARISON_ONLY",
        "created_at": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "recipe": RECIPE,
        "threshold": 0.5,
        "max_main_fits": 10,
        "max_replay_fits": 10,
        "groups": GROUPS,
        "fund_codes": sorted({r["fund_code"] for r in exam}),
        "excluded_groups": {"CN_BOND": "REQUIRES_BOND_AND_CONVERTIBLE_MARKET_CONTEXT"},
        "source": source,
        "input_files": {name: file_hash(root / name) for name in files},
        "fit_hashes": {q: digest(r) for q, r in fits.items()},
        "fit_counts": {q: len(r) for q, r in fits.items()},
        "exam_count": len(exam),
        "base_exam_count": len(read(base / "common-exam.json")),
        "primary_reference": CANDIDATES[0],
        "primary_metric": "FAMILY_DATE_WEIGHTED_ACCURACY",
        "bootstrap_seed": 20260912,
        "bootstrap_repetitions": 2000,
        "bootstrap_block_days": 5,
        "protected_years_excluded": [2025, 2026],
        "model_released": False,
    }
    spec["cohort_id"] = (
        "D1-M-" + digest({"inputs": spec["input_files"], "recipe": RECIPE, "candidates": CANDIDATES})[:24]
    )
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"hash": digest(spec)})
    return spec


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if digest(spec) != read(root / "study-receipt.json")["hash"]:
        raise ValueError("MARKET_STUDY_CHANGED")
    if spec["fingerprint"] != fingerprint() or spec["candidates"] != list(CANDIDATES) or spec["recipe"] != RECIPE:
        raise ValueError("MARKET_CODE_OR_RECIPE_CHANGED")
    for name, expected in spec["input_files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or file_hash(path) != expected:
            raise ValueError("MARKET_FROZEN_INPUT_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source"]["source_expires_at"]):
        raise ValueError("MARKET_SOURCE_RETENTION_EXPIRED")
    return spec


def predict(model: dict, rows: list[dict]) -> np.ndarray:
    x = np.asarray([vector(r, model["candidate"]) for r in rows])
    z = ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def fit(rows: list[dict], candidate: str, spec: dict, cutoff: str) -> dict:
    if len({r["t"] for r in rows}) < 252 or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30:
        raise ValueError("MARKET_INSUFFICIENT_SAMPLES")
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
        "features": list(FEATURES) + (list(MARKET_FEATURES) if candidate == "MARKET9_27" else []),
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
        raise ValueError("MARKET_MODEL_RESTORE_MISMATCH")
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
            raise ValueError("MARKET_FIT_CHANGED")
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
            raise ValueError("MARKET_REPLAY_MISMATCH")
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
                raise ValueError("MARKET_TRAINING_INCOMPLETE")
            continue
        complete = read(folder / "completion.json")
        models, predictions = read(folder / "models.json"), read(folder / "predictions.json")
        if digest(models) != complete["models_hash"] or digest(predictions) != complete["predictions_hash"]:
            raise ValueError("MARKET_OUTPUT_CHANGED")
        if set(models) != expected_keys or complete["successful_fits"] != 10:
            raise ValueError("MARKET_FIT_BUDGET_MISMATCH")
        if {p.name for p in folder.glob("*-reserved.json")} != {
            "budget-reserved.json",
            *(k + "-reserved.json" for k in expected_keys),
        }:
            raise ValueError("MARKET_FIT_RESERVATION_MISMATCH")
        for key, model in models.items():
            quarter = key.rsplit("-", 1)[1]
            if model != read(folder / (key + ".json")) or model["fit_hash"] != spec["fit_hashes"][quarter]:
                raise ValueError("MARKET_MODEL_CHANGED")
        if len(predictions) != spec["exam_count"]:
            raise ValueError("MARKET_QUESTION_COUNT_CHANGED")
    return read(root / "main/completion.json")


def paired(rows: list[dict], candidate: str, reference: str, spec: dict) -> dict:
    # 复用已验证的目标日块重采样算法，仅在副本中设置对照键，不改变保存的旧分组分数。
    copied = [{**r, "directions": {**r["directions"], comparison.BASELINE: r["directions"][reference]}} for r in rows]
    return {"reference": reference, **comparison.paired(copied, candidate, spec)}


def evaluate(root: Path) -> dict:
    verify(root)
    spec, proof = read(root / "study.json"), read(root / "replay-proof.json")
    if not proof["models_equal"] or proof["max_score_diff"] != 0:
        raise ValueError("MARKET_REPLAY_REQUIRED")
    for name in ("models.json", "predictions.json"):
        if read(root / "main" / name) != read(root / "replay" / name):
            raise ValueError("MARKET_REPLAY_OUTPUT_CHANGED")
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
        "paired_primary": paired(rows, "MARKET9_27", "NAV7_27", spec),
        "paired_old_group": paired(rows, "MARKET9_27", comparison.BASELINE, spec),
        "runtime_model_changed": False,
        "model_released": False,
        "limitations": [
            "2021—2024已使用开发数据",
            "历史收盘/公告可用时间采用假设",
            "当前基金类型不能证明历史实际投资暴露",
            "未证明未来有效，未进行参数搜索",
        ],
    }
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"hash": digest(result), "at": datetime.now(ZONE).isoformat()})
    return result
