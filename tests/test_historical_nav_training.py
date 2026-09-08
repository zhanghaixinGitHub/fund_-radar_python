"""候选训练契约、时间隔离与JSON保存测试；人工样本不能当真实基金成绩。"""

import hashlib
import json
import warnings
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from app.api.routes import historical_nav_training as api
from app.schemas.historical_nav_training import HistoricalNavTrainingRequest
from app.services import historical_nav_training as training
from app.services.historical_nav_evaluation import prepare_historical_nav_dataset
from pydantic import ValidationError
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sqlalchemy.exc import SQLAlchemyError
from tests.test_historical_nav_evaluation import batches as batches
from tests.test_historical_nav_evaluation import change_label, request_for
from tests.test_historical_nav_http import HEADERS  # noqa: F401
from tests.test_historical_nav_http import client as client

PATH = "/internal/v1/features/historical-nav-samples/candidate-training"


@pytest.fixture(scope="module")
def prepared(batches):
    return prepare_historical_nav_dataset(request_for(batches), batches)


def training_request(batches, prepared, **updates):
    return HistoricalNavTrainingRequest(
        **{
            **request_for(batches).model_dump(),
            "expected_dataset_hash": prepared.report.dataset_hash,
            **updates,
        }
    )


def train(batches, prepared):
    return training.train_prepared_candidate(training_request(batches, prepared), prepared)


def test_candidate_fits_only_train_and_replays_json(batches, prepared):
    first, second = train(batches, prepared), train(batches, prepared)
    assert first == second and first.status == "CANDIDATE_EVALUATED"
    assert first.preparation == prepared.report and first.candidate.validation.sample_count == 120
    assert first.model.train_count == 252 and first.model.train_counts_per_fund == {"008888": 252}
    assert first.model.sample_weight_per_fund == {"008888": 1.0}
    assert first.model.train_hash == prepared.report.train_hash
    assert first.model.mean == pytest.approx(np.mean([r.x for r in prepared.train], axis=0, dtype=np.float64))
    restored = training.restore_logistic_artifact(first.model.model_dump_json())
    assert restored == first.model and len(first.baseline_deltas) == 4
    assert first.training_eligible is False and first.artifact_persisted is False and first.database_written is False
    assert first.publication_status == "MODEL_NOT_RELEASED" and first.model.protocol.calibration == "NONE"
    assert "test_metrics" not in first.model_dump_json() and "test_up_rate" not in first.model_dump_json()
    scores = training.predict_artifact_scores(restored, tuple(r.x for r in prepared.validation))
    assert all(0 <= s <= 1 for s in scores) and len(first.prediction_preview) == 5
    assert first.prediction_preview[0].up_score == Decimal(str(scores[0]))


@pytest.mark.parametrize("split", ["VALIDATION", "TEST"])
def test_later_answers_cannot_change_model(batches, prepared, split):
    protocol = prepared.report.protocol
    lower = protocol.train_end_date if split == "VALIDATION" else protocol.validation_end_date
    upper = protocol.validation_end_date if split == "VALIDATION" else protocol.test_end_date
    changed = tuple(
        b.model_copy(
            update={
                "items": tuple(
                    change_label(s) if s.available_at and lower < s.available_at <= upper else s for s in b.items
                ),
            }
        )
        for b in batches
    )
    second_dataset = prepare_historical_nav_dataset(request_for(changed), changed)
    first, second = train(batches, prepared), train(changed, second_dataset)
    assert first.model == second.model and first.preparation.dataset_hash != second.preparation.dataset_hash
    if split == "TEST":
        assert first.candidate == second.candidate and first.prediction_preview == second.prediction_preview
        assert first.baseline_deltas == second.baseline_deltas
    else:
        assert first.candidate != second.candidate


def test_validation_features_not_used_for_scaler_or_fit_and_test_never_accessed(batches, prepared):
    class ForbiddenTest:
        def __iter__(self):
            pytest.fail("training touched test rows")

        def __len__(self):
            pytest.fail("training counted test rows")

    changed = replace(
        prepared,
        validation=tuple(replace(row, x=tuple(x * 2 for x in row.x)) for row in prepared.validation),
        test=ForbiddenTest(),
    )
    first, second = train(batches, prepared), train(batches, changed)
    assert first.model == second.model
    assert first.prediction_preview != second.prediction_preview


def test_training_features_change_fitted_model(batches, prepared):
    changed = replace(prepared, train=tuple(replace(row, x=tuple(x * 2 for x in row.x)) for row in prepared.train))
    first, second = train(batches, prepared), train(batches, changed)
    assert first.model.mean != second.model.mean and first.model.model_hash != second.model.model_hash


def test_order_and_preview_changes_do_not_change_model_or_scores(batches, prepared):
    reversed_batches = tuple(reversed(batches))
    request = request_for(reversed_batches, preview_size=0)
    second_dataset = prepare_historical_nav_dataset(request, reversed_batches)
    first = train(batches, prepared)
    second = training.train_prepared_candidate(
        training_request(reversed_batches, second_dataset, preview_size=0), second_dataset
    )
    assert first.model == second.model and first.candidate == second.candidate
    assert second.prediction_preview == () and second.preparation.sample_preview == ()


def test_fund_balancing_also_applies_to_scaler_and_constant_column(prepared):
    first = tuple(replace(r, x=(Decimal(0), *r.x[1:6], Decimal(9))) for r in prepared.train)
    second = tuple(replace(r, fund_code="001632", x=(Decimal(10), *r.x[1:6], Decimal(9))) for r in prepared.train[:126])
    scaler, estimator, counts, weights = training._fit_training_rows(first + second)
    assert counts == {"001632": 126, "008888": 252}
    assert weights == {"001632": 1.5, "008888": 0.75}
    assert scaler.mean_[0] == pytest.approx(5) and scaler.scale_[-1] == 1
    assert estimator.classes_.tolist() == [0, 1]


def test_hash_mismatch_and_insufficient_data_do_not_fit(batches, prepared, monkeypatch):
    monkeypatch.setattr(training, "_fit_training_rows", lambda *args: pytest.fail("must not fit"))
    with pytest.raises(training.HistoricalNavTrainingError, match="数据指纹"):
        training.train_prepared_candidate(training_request(batches, prepared, expected_dataset_hash="0" * 64), prepared)
    insufficient = replace(prepared, report=prepared.report.model_copy(update={"status": "INSUFFICIENT_DATA"}))
    result = train(batches, insufficient)
    assert result.model is None and result.candidate is None and result.baseline_deltas == ()


def test_single_class_rejected(prepared):
    with pytest.raises(training.HistoricalNavTrainingError) as error:
        training._fit_training_rows(tuple(replace(r, y=0) for r in prepared.train))
    assert error.value.code == "TRAIN_SINGLE_CLASS"


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), None])
def test_non_finite_features_rejected(prepared, bad):
    rows = (replace(prepared.train[0], x=(bad, *prepared.train[0].x[1:])), *prepared.train[1:])
    with pytest.raises((ValueError, training.HistoricalNavTrainingError)):
        training._fit_training_rows(rows)


def test_non_convergence_rejected(prepared, monkeypatch):
    def fail(*args, **kwargs):
        warnings.warn("synthetic non convergence", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(LogisticRegression, "fit", fail)
    with pytest.raises(training.HistoricalNavTrainingError) as error:
        training._fit_training_rows(prepared.train)
    assert error.value.code == "TRAINING_NOT_CONVERGED"


def test_time_budget_rejects_result(batches, prepared, monkeypatch):
    monkeypatch.setattr(training, "TRAINING_SECONDS", -1)
    with pytest.raises(training.HistoricalNavTrainingError) as error:
        train(batches, prepared)
    assert error.value.code == "TRAINING_TIMEOUT"


@pytest.mark.parametrize("mutation", ["coefficient", "feature_order", "scale", "protocol", "unknown"])
def test_corrupt_artifact_rejected(batches, prepared, mutation):
    payload = train(batches, prepared).model.model_dump(mode="json")
    if mutation == "coefficient":
        payload["coefficients"][0] += 1
    elif mutation == "feature_order":
        payload["feature_names"].reverse()
    elif mutation == "scale":
        payload["scale"][0] = 0
    elif mutation == "protocol":
        payload["protocol"]["c"] = 100
    else:
        payload["code"] = "do not execute"
    with pytest.raises(ValueError):
        training.restore_logistic_artifact(json.dumps(payload))


def test_busy_does_not_read_and_lock_released_after_failure(batches, prepared, monkeypatch):
    request = training_request(batches, prepared)

    def fail(*args):
        raise ValueError("synthetic read failure")

    monkeypatch.setattr(training, "load_historical_nav_dataset", fail)
    with training._TRAINING_LOCK, pytest.raises(training.HistoricalNavTrainingError) as error:
        training.train_stored_historical_nav_candidate(request)
    assert error.value.status_code == 429
    with pytest.raises(ValueError):
        training.train_stored_historical_nav_candidate(request)
    assert training._TRAINING_LOCK.acquire(blocking=False)
    training._TRAINING_LOCK.release()


def test_http_success(batches, prepared, client, monkeypatch):
    monkeypatch.setattr(training, "load_historical_nav_dataset", lambda request: prepared)
    response = client.post(
        PATH, json=training_request(batches, prepared).model_dump(mode="json", by_alias=True), headers=HEADERS
    )
    assert response.status_code == 200 and response.json()["status"] == "CANDIDATE_EVALUATED"


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost"}])
def test_http_auth_before_work(batches, prepared, client, monkeypatch, headers):
    monkeypatch.setattr(
        api, "train_stored_historical_nav_candidate", lambda request: pytest.fail("unauthorized training")
    )
    response = client.post(
        PATH, json=training_request(batches, prepared).model_dump(mode="json", by_alias=True), headers=headers
    )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "updates",
    [
        {"expectedDatasetHash": None},
        {"expectedDatasetHash": "bad"},
        {"includeTest": True},
        {"C": 2},
        {"outputPath": "elsewhere"},
        {"x": [[1, 2]]},
        {"previewSize": True},
        {"previewSize": 21},
    ],
)
def test_http_rejects_unapproved_parameters(batches, prepared, client, monkeypatch, updates):
    monkeypatch.setattr(api, "train_stored_historical_nav_candidate", lambda request: pytest.fail("invalid training"))
    body = training_request(batches, prepared).model_dump(mode="json", by_alias=True)
    assert client.post(PATH, json=body | updates, headers=HEADERS).status_code == 422


@pytest.mark.parametrize(
    "error,expected",
    [
        (training.HistoricalNavTrainingError("DATASET_HASH_MISMATCH", "conflict"), 409),
        (training.HistoricalNavTrainingError("TRAINING_BUSY", "busy", 429), 429),
        (SQLAlchemyError("private database details"), 503),
        (ImportError("missing local numerical runtime"), 503),
    ],
)
def test_http_errors_do_not_leak(batches, prepared, client, monkeypatch, error, expected):
    def fail(request):
        raise error

    monkeypatch.setattr(api, "train_stored_historical_nav_candidate", fail)
    result = client.post(
        PATH, json=training_request(batches, prepared).model_dump(mode="json", by_alias=True), headers=HEADERS
    )
    assert result.status_code == expected and "private database details" not in result.text


def test_cli_saves_replayable_bundle_and_refuses_overwrite(tmp_path, batches, prepared, monkeypatch):
    from scripts import historical_nav_candidate as cli

    monkeypatch.setattr(cli, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(ai_database_url="postgresql://localhost/fund_ai"))
    result = train(batches, prepared)
    monkeypatch.setattr(cli, "train_stored_historical_nav_candidate", lambda request: result)
    request, key = training_request(batches, prepared), uuid4()
    directory, manifest = cli.run_candidate(request, key)
    assert manifest["artifact_persisted"] and manifest["database_written"] is False
    assert training.restore_logistic_artifact((directory / "model.json").read_bytes()) == result.model
    assert cli.read_training_request(directory / "request.json") == request
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        cli.run_candidate(request, key)


def test_cli_write_failure_never_marks_complete(tmp_path, batches, prepared, monkeypatch):
    from scripts import historical_nav_candidate as cli

    monkeypatch.setattr(cli, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(ai_database_url="postgresql://localhost/fund_ai"))
    monkeypatch.setattr(cli, "train_stored_historical_nav_candidate", lambda request: train(batches, prepared))
    write = cli._write_json

    def fail(path, payload):
        if path.name == "model.json":
            raise OSError("synthetic disk failure")
        return write(path, payload)

    monkeypatch.setattr(cli, "_write_json", fail)
    key = uuid4()
    with pytest.raises(OSError):
        cli.run_candidate(training_request(batches, prepared), key)
    assert not (tmp_path / f"nav-candidate-{key}" / "complete.json").exists()


def test_cli_rejects_other_database_before_writing(tmp_path, batches, prepared, monkeypatch):
    from scripts import historical_nav_candidate as cli

    monkeypatch.setattr(cli, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(ai_database_url="postgresql://remote/fund_ai"))
    with pytest.raises(ValueError):
        cli.run_candidate(training_request(batches, prepared), uuid4())
    assert list(tmp_path.iterdir()) == []


def test_request_rejects_duplicate_ids_and_missing_hash(batches, prepared):
    values = training_request(batches, prepared).model_dump()
    with pytest.raises(ValidationError):
        HistoricalNavTrainingRequest(**(values | {"batch_ids": (values["batch_ids"][0],) * 2}))
    del values["expected_dataset_hash"]
    with pytest.raises(ValidationError):
        HistoricalNavTrainingRequest(**values)
