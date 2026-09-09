"""两个独立研究适配器；Chronos 在受限子进程按需加载，服务无需其依赖。"""

import math
import multiprocessing
import warnings
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter

from app.schemas.model_comparison import ComparisonInput
from app.services.historical_nav_calibration import fit_calibrator, predict_calibrated_scores
from app.services.historical_nav_training import fit_logistic_artifact, predict_artifact_scores, sigmoid
from app.services.model_comparison_artifacts import file_hash, fingerprint, read_json
from app.services.model_comparison_protocol import REVISION, WEIGHT_HASH


def chronos_score(item: ComparisonInput, quantiles: list[list[float]]) -> dict:
    horizon = 20 + item.anchor_lag_sessions
    if len(quantiles) != horizon or any(len(row) != 3 for row in quantiles):
        raise ValueError("QUANTILE_SHAPE_MISMATCH")
    if any(not math.isfinite(float(v)) for row in quantiles for v in row):
        raise ValueError("NONFINITE_QUANTILE")
    if any(not row[0] <= row[1] <= row[2] for row in quantiles):
        raise ValueError("QUANTILE_CROSSING")
    last = float(item.history_values[-1])
    base = last if item.anchor_lag_sessions == 0 else quantiles[0][1]
    end = quantiles[-1][1]
    if base <= 0 or end <= 0:
        raise ValueError("INVALID_FORECAST_BASE_OR_END")
    scale = max(quantiles[-1][2] - quantiles[-1][0], 1e-8 * max(abs(last), 1))
    score = (end - base) / scale
    if not math.isfinite(score):
        raise ValueError("NONFINITE_RAW_SCORE")
    return {
        "raw_score": score,
        "raw_direction": int(score > 0),
        "base_reference": base,
        "horizon": horizon,
        "quantiles": quantiles,
    }


def fit_self_trained(fit_rows, cal_rows, window, versions):
    base = fit_logistic_artifact(fit_rows, start_date=date(2022, 1, 1), end_date=window.fit_end_date, versions=versions)
    calibrated = fit_calibrator(base, cal_rows, end_date=window.calibration_end_date)
    return base, calibrated


def predict_self_trained(base, calibrated, items):
    x = tuple(i.x for i in items)
    raw = predict_artifact_scores(base, x)
    scores = predict_calibrated_scores(calibrated, x) if calibrated.calibrator.slope > 0 else [None] * len(items)
    return [
        {
            "raw_score": float(r),
            "raw_direction": int(r > 0.5),
            "probability": float(p) if p is not None else None,
            "reason": None if p is not None else "CALIBRATOR_NONPOSITIVE_SLOPE",
        }
        for r, p in zip(raw, scores, strict=True)
    ]


def fit_chronos_calibrator(rows, raw_scores, *, lower: date, upper: date, base_hash: str) -> dict:
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    if not rows or len(rows) != len(raw_scores) or len(rows) > 5580:
        raise ValueError("CALIBRATION_SIZE_INVALID")
    if any(not lower < r.available_at <= upper or r.label_available_at > upper for r in rows):
        raise ValueError("CALIBRATION_TIME_BOUNDARY")
    counts = Counter(r.fund_code for r in rows)
    if set(counts) != {"001632", "006730", "008888"} or min(counts.values()) < 60:
        raise ValueError("CALIBRATION_INSUFFICIENT_DATA")
    if {r.y for r in rows} != {0, 1}:
        raise ValueError("CALIBRATION_SINGLE_CLASS")
    if any(not math.isfinite(z) for z in raw_scores):
        raise ValueError("CALIBRATION_NONFINITE")
    weights = {f: len(rows) / (len(counts) * n) for f, n in counts.items()}
    values = np.asarray(raw_scores, dtype=np.float64).reshape(-1, 1)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        estimator = LogisticRegression(
            C=1.0,
            l1_ratio=0.0,
            solver="lbfgs",
            tol=1e-8,
            max_iter=1000,
            fit_intercept=True,
            class_weight=None,
            random_state=0,
        )
        estimator.fit(values, [r.y for r in rows], sample_weight=[weights[r.fund_code] for r in rows])
        reference = estimator.predict_proba(values)[:, 1]
    if estimator.classes_.tolist() != [0, 1] or int(estimator.n_iter_[0]) >= 1000:
        raise ValueError("CALIBRATION_NOT_CONVERGED")
    result = {
        "version": "CHRONOS2_LOCAL_SIGMOID_V1",
        "base_model_hash": base_hash,
        "slope": float(estimator.coef_[0, 0]),
        "intercept": float(estimator.intercept_[0]),
        "iterations": int(estimator.n_iter_[0]),
        "counts_per_fund": dict(counts),
        "weights_per_fund": weights,
        "class_counts": dict(Counter(str(r.y) for r in rows)),
        "start_exclusive": str(lower),
        "end_inclusive": str(upper),
        "calibration_hash": fingerprint(
            [{"sample": r.content_hash, "raw": z} for r, z in zip(rows, raw_scores, strict=True)]
        ),
        "purpose": "HISTORICAL_EXPLORATION_ONLY",
        "publication_status": "MODEL_NOT_RELEASED",
    }
    result["model_hash"] = fingerprint(result)
    # A nonpositive map remains an inspectable artifact, but cannot supply main probabilities.
    replay = [sigmoid(result["slope"] * z + result["intercept"]) for z in raw_scores]
    if any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(replay, reference, strict=True)):
        raise ValueError("CALIBRATION_REPLAY_MISMATCH")
    return result


def calibrated_chronos_score(model: dict, value: float, expected_base_hash: str) -> float:
    if (
        fingerprint({k: v for k, v in model.items() if k != "model_hash"}) != model["model_hash"]
        or model["base_model_hash"] != expected_base_hash
        or model["version"] != "CHRONOS2_LOCAL_SIGMOID_V1"
    ):
        raise ValueError("CALIBRATION_ARTIFACT_MISMATCH")
    if not math.isfinite(model["slope"]) or not math.isfinite(model["intercept"]) or model["slope"] <= 0:
        raise ValueError("CALIBRATOR_NONPOSITIVE_SLOPE")
    if not math.isfinite(value):
        raise ValueError("NONFINITE_RAW_SCORE")
    return sigmoid(model["slope"] * value + model["intercept"])


def _chronos_worker(connection, checkpoint: str, threads: int):
    """Own process permits a hard timeout without affecting the API or the user's programs."""
    try:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import psutil
        import torch
        from chronos import Chronos2Pipeline

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        started = perf_counter()
        pipeline = Chronos2Pipeline.from_pretrained(
            checkpoint,
            device_map="cpu",
            torch_dtype=torch.float32,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        pipeline.model.eval()
        process = psutil.Process()
        connection.send({"loaded_seconds": perf_counter() - started, "rss_bytes": process.memory_info().rss})
        while request := connection.recv():
            histories, horizon, batch_size = request
            started = perf_counter()
            with torch.inference_mode():
                q, _ = pipeline.predict_quantiles(
                    [torch.tensor(h, dtype=torch.float32) for h in histories],
                    prediction_length=horizon,
                    quantile_levels=[0.1, 0.5, 0.9],
                    batch_size=batch_size,
                    context_length=61,
                    cross_learning=False,
                )
            connection.send(
                {
                    "quantiles": [v[0].cpu().tolist() for v in q],
                    "seconds": perf_counter() - started,
                    "rss_bytes": process.memory_info().rss,
                    "peak_rss_bytes": getattr(process.memory_info(), "peak_wset", process.memory_info().rss),
                }
            )
    except Exception as error:
        # No source data, local settings or stack traces containing connection strings cross the boundary.
        connection.send({"error": type(error).__name__, "detail": str(error)[:300]})
    finally:
        connection.close()


class ChronosAdapter:
    def __init__(self, checkpoint: Path, budget: dict):
        if checkpoint.name != REVISION or file_hash(checkpoint / "model.safetensors") != WEIGHT_HASH:
            raise ValueError("CHECKPOINT_HASH_MISMATCH")
        manifest = read_json(checkpoint / "download.json")
        if file_hash(checkpoint / "config.json") != manifest["files"]["config.json"]["sha256"]:
            raise ValueError("CHECKPOINT_CONFIG_MISMATCH")
        self.base_hash = fingerprint(
            {
                "revision": REVISION,
                "files": manifest["files"],
                "dtype": "float32",
                "context": 61,
                "cross_learning": False,
            }
        )
        self.budget = budget
        self.telemetry = []
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_chronos_worker, args=(child, str(checkpoint.resolve()), budget["threads"])
        )
        self.process.start()
        child.close()
        try:
            self.load = self._receive(180)
        except Exception:
            self.close()
            raise

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            self.close()
            raise TimeoutError("CHRONOS_TIMEOUT")
        message = self.connection.recv()
        if "error" in message:
            raise RuntimeError(f"CHRONOS_WORKER_{message['error']}: {message['detail']}")
        if message.get("rss_bytes", 0) > self.budget["max_process_rss_bytes"]:
            self.close()
            raise MemoryError("CHRONOS_MEMORY_BUDGET")
        return message

    def predict(self, items: list[ComparisonInput]) -> list[dict]:
        if not items or len(items) > self.budget["batch_size"] or len({i.anchor_lag_sessions for i in items}) != 1:
            raise ValueError("CHRONOS_BATCH_SHAPE")
        self.connection.send(
            (
                [[float(v) for v in i.history_values] for i in items],
                20 + items[0].anchor_lag_sessions,
                self.budget["batch_size"],
            )
        )
        result = self._receive(self.budget["single_batch_seconds"])
        self.telemetry.append({k: v for k, v in result.items() if k != "quantiles"})
        if len(result["quantiles"]) != len(items):
            raise ValueError("CHRONOS_BATCH_OUTPUT_MISMATCH")
        outputs = []
        for item, quantiles in zip(items, result["quantiles"], strict=True):
            try:
                outputs.append({**chronos_score(item, quantiles), "reason": None})
            except ValueError as error:
                outputs.append({"reason": str(error), "raw_score": None})
        return outputs

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        self.connection.close()
