"""序列化修复验收：所有拟合仅用此文件生成的随机合成行，共 6 次/完整运行。"""

import hashlib
import io
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pytest
from app.services import fund_002112_model_state_v2 as repaired
from app.services import fund_002112_round3_model as fixed


def _fit_evidence(event):
    """只在执行人指定证据文件时追加事件；不访问真实资料或真实拟合账本。"""
    destination = os.environ.get("FUND_002112_SYNTHETIC_EVIDENCE")
    if destination:
        with Path(destination).open("a", encoding="utf-8") as output:
            output.write(json.dumps({"synthetic_only": True, **event}) + "\n")
            output.flush()
            os.fsync(output.fileno())


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(20260926)
    rows = [
        {
            "fund_code": "SYNTHETIC",
            "target": f"synthetic-{index:04d}",
            "actual_direction": fixed.CLASSES[index % 3],
            "x": rng.normal(size=20).tolist(),
        }
        for index in range(390)
    ]
    training, exams = rows[:360], rows[360:]
    result = {}
    for variant in fixed.VARIANTS:
        result[variant] = {}
        for role in ("main", "replay"):
            _fit_evidence({"event": "before_classifier_fit", "variant": variant, "role": role})
            model, predictions = fixed.fit(training, exams, [1.0] * len(training), variant)
            _fit_evidence({"event": "completed_classifier_fit", "variant": variant, "role": role})
            result[variant][role] = model, predictions
    return result, training, exams


def test_original_empty_array_defect_is_fixed_without_fitting():
    before = {"bits": np.empty((0, 8), dtype=np.uint32)}
    stream = io.BytesIO()
    joblib.dump(before, stream)
    stream.seek(0)
    after = joblib.load(stream)
    assert before["bits"].strides != after["bits"].strides
    assert joblib.hash(before) != joblib.hash(after)
    assert repaired.numeric_digest(before) == repaired.numeric_digest(after)


def test_layout_endian_and_negative_stride_are_not_numeric_changes():
    value = np.arange(24, dtype=np.float64).reshape(4, 6)
    expected = repaired.numeric_digest(value)
    assert repaired.numeric_digest(np.asfortranarray(value)) == expected
    assert repaired.numeric_digest(value.astype(">f8")) == expected
    view = np.ascontiguousarray(value[:, ::-1])[:, ::-1]
    assert view.strides[-1] < 0
    assert repaired.numeric_digest(view) == expected


def test_structured_array_padding_and_offsets_are_not_numeric_changes():
    packed = np.zeros(3, dtype=[("flag", "u1"), ("value", "f8")])
    aligned = np.zeros(3, dtype=np.dtype([("flag", "u1"), ("value", "f8")], align=True))
    assert packed.dtype.itemsize != aligned.dtype.itemsize
    assert repaired.numeric_digest(packed) == repaired.numeric_digest(aligned)


@pytest.mark.parametrize(
    "altered",
    [
        np.zeros((0, 7), dtype=np.uint32),
        np.zeros((0, 8), dtype=np.int32),
        np.zeros((1, 8), dtype=np.uint32),
        np.zeros((0, 8), dtype=np.uint64),
    ],
)
def test_empty_array_type_and_shape_still_matter(altered):
    assert repaired.numeric_digest(np.empty((0, 8), dtype=np.uint32)) != repaired.numeric_digest(altered)


@pytest.mark.parametrize("value", [np.array([object()]), np.array([np.nan]), np.array([np.inf]), object()])
def test_unsupported_and_nonfinite_values_fail_closed(value):
    with pytest.raises(ValueError):
        repaired.numeric_digest(value)


@pytest.mark.parametrize("variant", fixed.VARIANTS)
def test_independent_synthetic_refit_and_in_memory_roundtrip(fitted, variant):
    models, training, exams = fitted
    model, predictions = models[variant]["main"]
    replay, repeated = models[variant]["replay"]
    expected = repaired.model_state_digest(model)
    assert expected == repaired.model_state_digest(replay)
    assert repaired.verify_all_predictions(replay, training, exams, predictions) == {
        "train": {"rows": 360, "max_score_delta": 0.0},
        "exam": {"rows": 30, "max_score_delta": 0.0},
    }
    assert predictions == repeated
    stream = io.BytesIO()
    joblib.dump(model, stream)
    stream.seek(0)
    restored = joblib.load(stream)
    assert repaired.model_state_digest(restored) == expected


# 独立解释器重新加载，不共享父进程估计器；patch fit 确保此步骤没有暗中重训。
WORKER = """
import json, os, sys
from pathlib import Path
from app.services import fund_002112_model_state_v2 as r
def forbidden(*args, **kwargs):
    raise AssertionError('FIT_FORBIDDEN_IN_RESTORE')
r.StandardScaler.fit = r.LogisticRegression.fit = r.HistGradientBoostingClassifier.fit = forbidden
root, slot, manifest, reference = sys.argv[1:]
raw = (Path(root) / 'reference.json').read_bytes()
assert r.hashlib.sha256(raw).hexdigest() == reference
data = json.loads(raw)
model = r.restore_model(root, slot, manifest)
print(json.dumps({'pid': os.getpid(), 'state': r.model_state_digest(model),
                 'check': r.verify_all_predictions(model, data['train'], data['exam'], data['predictions'])}))
"""


@pytest.mark.parametrize("variant", fixed.VARIANTS)
def test_save_and_fresh_process_restore_all_rows(tmp_path, fitted, variant):
    models, training, exams = fitted
    model, predictions = models[variant]["main"]
    manifest = repaired.save_model(model, tmp_path, variant)
    raw = json.dumps({"train": training, "exam": exams, "predictions": predictions}).encode()
    (tmp_path / "reference.json").write_bytes(raw)
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-c",
            WORKER,
            str(tmp_path),
            variant,
            manifest,
            hashlib.sha256(raw).hexdigest(),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    evidence = json.loads(completed.stdout)
    assert evidence["pid"] != os.getpid()
    assert evidence["state"] == repaired.model_state_digest(model)
    assert evidence["check"] == {
        "train": {"rows": 360, "max_score_delta": 0.0},
        "exam": {"rows": 30, "max_score_delta": 0.0},
    }
    _fit_evidence({"event": "fresh_process_restore_passed", "variant": variant, **evidence})


@pytest.mark.parametrize(
    "variant,part",
    [
        ("L20", "coefficient"),
        ("L20", "intercept"),
        ("L20", "iterations"),
        ("T20", "threshold"),
        ("T20", "leaf"),
        ("T20", "bin"),
        ("T20", "baseline"),
        ("T20", "bitset_shape"),
        ("T20", "rng"),
        ("T20", "tree_order"),
        ("N7", "mean"),
        ("N7", "scale"),
        ("N7", "variance"),
        ("N7", "weight_hash"),
        ("N7", "training_hash"),
    ],
)
def test_true_numeric_and_provenance_changes_are_detected(fitted, variant, part):
    models, _, _ = fitted
    original = models[variant]["main"][0]
    model = deepcopy(original)
    classifier = model["classifier"]
    if part == "coefficient":
        classifier.coef_[0, 0] += 0.01
    elif part == "intercept":
        classifier.intercept_[0] += 0.01
    elif part == "iterations":
        classifier.n_iter_[0] += 1
    elif part in ("threshold", "leaf"):
        field = "num_threshold" if part == "threshold" else "value"
        nodes = classifier._predictors[0][0].nodes
        index = 0 if part == "threshold" else np.flatnonzero(nodes["is_leaf"])[0]
        nodes[field][index] += 0.01
    elif part == "bin":
        classifier._bin_mapper.bin_thresholds_[0][0] += 0.01
    elif part == "baseline":
        classifier._baseline_prediction[0, 0] += 0.01
    elif part == "bitset_shape":
        classifier._predictors[0][0].raw_left_cat_bitsets = np.empty((0, 7), dtype=np.uint32)
    elif part == "rng":
        classifier._feature_subsample_rng.random()
    elif part == "tree_order":
        classifier._predictors[0] = classifier._predictors[0][::-1]
    elif part in ("mean", "scale", "variance"):
        getattr(model["scaler"], {"mean": "mean_", "scale": "scale_", "variance": "var_"}[part])[0] += 0.01
    else:
        model["weights_hash" if part == "weight_hash" else "training_hash"] = "f" * 64
    assert repaired.model_state_digest(original) != repaired.model_state_digest(model)


@pytest.mark.parametrize("part", ["features", "classes", "tie_order", "recipe", "extra", "missing"])
def test_contract_drift_fails_closed(fitted, part):
    model = deepcopy(fitted[0]["T20"]["main"][0])
    if part in ("features", "classes", "tie_order"):
        model[part] = model[part][::-1]
    elif part == "recipe":
        model["classifier"].max_iter += 1
    elif part == "extra":
        model["classifier"].unreviewed_prediction_state = 1
    else:
        del model["classifier"]._baseline_prediction
    with pytest.raises(ValueError):
        repaired.model_state_digest(model)


@pytest.mark.parametrize("target", ["model", "manifest"])
def test_file_or_manifest_tamper_rejected_before_unpickle(tmp_path, fitted, monkeypatch, target):
    model = fitted[0]["N7"]["main"][0]
    manifest = repaired.save_model(model, tmp_path, "N7")
    path = tmp_path / ("N7.joblib" if target == "model" else "N7.manifest.json")
    path.write_bytes(path.read_bytes() + b"changed")

    def forbidden(*args, **kwargs):
        raise AssertionError("TAMPERED_BYTES_REACHED_UNPICKLE")

    monkeypatch.setattr(joblib, "load", forbidden)
    with pytest.raises(ValueError, match="BYTES_CHANGED"):
        repaired.restore_model(tmp_path, "N7", manifest)


def test_save_never_overwrites_and_incomplete_package_is_rejected(tmp_path, fitted):
    model = fitted[0]["N7"]["main"][0]
    repaired.save_model(model, tmp_path, "N7")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        repaired.save_model(model, tmp_path, "N7")
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    (tmp_path / "partial.joblib").write_bytes(b"incomplete")
    with pytest.raises(FileNotFoundError):
        repaired.restore_model(tmp_path, "partial", "0" * 64)
    with pytest.raises(FileExistsError):
        repaired.save_model(model, tmp_path, "partial")


def test_valid_file_checksum_cannot_hide_changed_numeric_state(tmp_path, fitted):
    """模拟文件完整但模型数值与先前清单状态不一致，不能只靠文件哈希放行。"""
    model = deepcopy(fitted[0]["L20"]["main"][0])
    repaired.save_model(model, tmp_path, "L20")
    model["classifier"].coef_[0, 0] += 0.01
    stream = io.BytesIO()
    joblib.dump(model, stream)
    raw = stream.getvalue()
    (tmp_path / "L20.joblib").write_bytes(raw)
    manifest_path = tmp_path / "L20.manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["file_sha256"] = hashlib.sha256(raw).hexdigest()
    encoded = json.dumps(manifest).encode()
    manifest_path.write_bytes(encoded)
    with pytest.raises(ValueError, match="NUMERIC_STATE_CHANGED"):
        repaired.restore_model(tmp_path, "L20", hashlib.sha256(encoded).hexdigest())


def test_environment_drift_rejected(monkeypatch):
    monkeypatch.setattr(repaired, "runtime_environment", lambda: {**repaired.ENVIRONMENT, "numpy": "unknown"})
    with pytest.raises(ValueError, match="ENVIRONMENT_CHANGED"):
        repaired.check_environment()


@pytest.mark.parametrize("part", ["nan", "score", "identity", "direction", "missing_row"])
def test_all_row_check_rejects_invalid_reference(fitted, part):
    models, training, exams = fitted
    model, predictions = models["T20"]["main"]
    reference = deepcopy(predictions)
    row = reference["exam"][-1]
    if part == "nan":
        row["scores"][0] = float("nan")
    elif part == "score":
        row["scores"][0] += 1e-8
    elif part == "identity":
        row["input_hash"] = "f" * 64
    elif part == "direction":
        row["direction"] = next(value for value in fixed.CLASSES if value != row["direction"])
    else:
        reference["exam"].pop()
    with pytest.raises(ValueError):
        repaired.verify_all_predictions(model, training, exams, reference)


def test_no_path_traversal():
    with pytest.raises(ValueError, match="INVALID_SLOT"):
        repaired.restore_model("unused", "../old-run/model", "0" * 64)
