"""人工日历样本验证时间隔离、校准分箱、HTTP和本机保存；不作为真实基金成绩。"""

import hashlib
import json
import math
import warnings
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from app.api.routes import historical_nav_calibration as api
from app.schemas.historical_nav_calibration import HistoricalNavCalibrationRequest
from app.services import historical_nav_calibration as calibration
from app.services import historical_nav_training as training
from app.services.historical_nav_evaluation import prepare_historical_nav_dataset
from app.services.historical_nav_samples import (
    HistoricalNavPoint,
    HistoricalNavSampleInput,
    build_historical_nav_samples,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sqlalchemy.exc import SQLAlchemyError
from tests.test_historical_nav_evaluation import change_label, make_batches
from tests.test_historical_nav_http import HEADERS  # noqa: F401
from tests.test_historical_nav_http import client as client

PATH = "/internal/v1/features/historical-nav-samples/calibration-evaluation"


@pytest.fixture(scope="module")
def calibration_batches():
    start = date(2021, 11, 1)
    values = tuple(Decimal(2) + Decimal(str(math.sin(i / 17))) / 7 + Decimal(i) / 10000 for i in range(1570))
    points = tuple(
        HistoricalNavPoint(start + timedelta(days=i), start + timedelta(days=i + 1), v, v) for i, v in enumerate(values)
    )
    samples = build_historical_nav_samples(
        HistoricalNavSampleInput(
            fund_code="008888",
            fund_type="STOCK",
            source_code="EVALUATION_TEST",
            source_sync_run_id=UUID(int=2),
            nav_points=points,
        )
    )
    return make_batches(samples)


def request_for(batches, expected_hash="0" * 64, **updates):
    return HistoricalNavCalibrationRequest(
        **{
            "batch_ids": tuple(b.batch_id for b in batches),
            "expected_dataset_hash": expected_hash,
            "train_start_date": date(2022, 1, 1),
            "train_end_date": date(2023, 12, 31),
            "validation_end_date": date(2024, 12, 31),
            "test_end_date": date(2025, 12, 31),
            **updates,
        }
    )


@pytest.fixture(scope="module")
def prepared(calibration_batches):
    return prepare_historical_nav_dataset(request_for(calibration_batches).evaluation_request(), calibration_batches)


def evaluate(batches, data, **updates):
    return calibration.evaluate_prepared_calibration(request_for(batches, data.report.dataset_hash, **updates), data)


@pytest.fixture(scope="module")
def result(calibration_batches, prepared):
    return evaluate(calibration_batches, prepared)


def test_three_frozen_windows_and_json_replay(calibration_batches, prepared, result):
    assert result.status == "CALIBRATION_EVALUATED" and result.evaluated_window_count == 3
    assert result == evaluate(calibration_batches, prepared)
    assert not result.database_written and not result.artifact_persisted and not result.training_eligible
    assert result.publication_status == "MODEL_NOT_RELEASED"
    for w in result.windows:
        assert w.status == "EVALUATED" and len(w.baselines) == 4 and w.model.base_model.train_count >= 252
        assert len(w.prediction_preview) == 5 and w.before.validation.sample_count == w.after.validation.sample_count
        assert len(w.reliability_before.bins) == len(w.reliability_after.bins) == 5
        assert w.model.base_model.train_end_date < w.model.calibrator.start_date <= w.model.calibrator.end_date
        assert w.model.calibrator.end_date < w.window.evaluation_end_date
        assert calibration.restore_calibrated_artifact(w.model.model_dump_json()) == w.model
        assert w.brier_delta == w.after.validation.brier_score - w.before.validation.brier_score
        assert w.ece_delta == w.reliability_after.ece - w.reliability_before.ece
    assert "test_metrics" not in result.model_dump_json() and "test_up_rate" not in result.model_dump_json()


def test_each_stage_has_real_label_cutoff_and_disjoint_inputs(prepared):
    for window in calibration.WINDOWS:
        rows, counts = calibration._window_rows(prepared, window)
        assert not any(any(f.missing.values()) for f in counts)
        assert max(r.label_available_at for r in rows["FIT"]) <= window.fit_end_date
        assert max(r.label_available_at for r in rows["CALIBRATION"]) <= window.calibration_end_date
        assert max(r.label_available_at for r in rows["EXAM"]) <= window.evaluation_end_date
        assert max(r.available_at for r in rows["FIT"]) < min(r.available_at for r in rows["CALIBRATION"])
        assert max(r.available_at for r in rows["CALIBRATION"]) < min(r.available_at for r in rows["EXAM"])
        assert counts[0].purged["FIT"] == 20
        # 全局TRAIN已经剔除跨2023年末的答案，最后一窗不重复计为新增剔除。
        assert counts[0].purged["CALIBRATION"] == (0 if window.role == "FIXED_VALIDATION" else 20)


@pytest.mark.parametrize("year", [2024, 2025])
def test_2024_and_2025_labels_never_fit_any_model(calibration_batches, prepared, result, year):
    changed = tuple(
        b.model_copy(
            update={
                "items": tuple(
                    change_label(s) if s.available_at and s.available_at.year == year else s for s in b.items
                )
            }
        )
        for b in calibration_batches
    )
    data = prepare_historical_nav_dataset(request_for(changed).evaluation_request(), changed)
    second = evaluate(changed, data)
    assert result.preparation.dataset_hash != second.preparation.dataset_hash
    assert [w.model for w in result.windows] == [w.model for w in second.windows]
    if year == 2025:
        assert result.windows == second.windows
    else:
        assert result.windows[:2] == second.windows[:2]
        assert result.windows[2].after != second.windows[2].after


def test_test_object_not_read_and_validation_x_not_fit(calibration_batches, prepared, result):
    class Forbidden:
        def __iter__(self):
            pytest.fail("read TEST")

        def __len__(self):
            pytest.fail("count TEST")

    data = replace(
        prepared,
        test=Forbidden(),
        validation=tuple(replace(r, x=tuple(v * 2 for v in r.x)) for r in prepared.validation),
    )
    second = evaluate(calibration_batches, data)
    assert [w.model for w in result.windows] == [w.model for w in second.windows]
    assert result.windows[2].prediction_preview != second.windows[2].prediction_preview


def test_calibration_labels_change_mapping_not_base(calibration_batches, prepared, result):
    window = calibration.WINDOWS[0]
    data = replace(
        prepared,
        train=tuple(
            replace(r, y=1 - r.y, content_hash="f" * 64)
            if window.fit_end_date < r.available_at <= window.calibration_end_date
            else r
            for r in prepared.train
        ),
    )
    second = evaluate(calibration_batches, data)
    assert result.windows[0].model.base_model == second.windows[0].model.base_model
    assert result.windows[0].model.calibrator != second.windows[0].model.calibrator
    assert result.windows[0].model.model_hash != second.windows[0].model.model_hash


def test_internal_exam_changes_do_not_leak_back_to_same_window(calibration_batches, prepared, result):
    window = calibration.WINDOWS[0]
    data = replace(
        prepared,
        train=tuple(
            replace(r, y=1 - r.y, content_hash="f" * 64)
            if window.calibration_end_date < r.available_at <= window.evaluation_end_date
            else r
            for r in prepared.train
        ),
    )
    second = evaluate(calibration_batches, data)
    assert result.windows[0].model == second.windows[0].model
    assert result.windows[0].after != second.windows[0].after
    # 前一轮已成熟的考试历史可以参与后一轮校准，不要求后续模型也不变。


def test_batch_order_and_preview_do_not_change_scores(calibration_batches, prepared, result):
    batches = tuple(reversed(calibration_batches))
    data = prepare_historical_nav_dataset(request_for(batches, preview_size=0).evaluation_request(), batches)
    second = evaluate(batches, data, preview_size=0)
    assert result.preparation.dataset_hash == second.preparation.dataset_hash
    for a, b in zip(result.windows, second.windows, strict=True):
        assert a.model == b.model and a.after == b.after and a.reliability_after == b.reliability_after
        assert b.prediction_preview == ()


def test_reliability_hand_calculation_boundaries_and_empty_bins():
    report = calibration.reliability_report((0, 1, 0, 1), tuple(map(Decimal, ("0", "0.2", "0.2", "1"))))
    assert [b.count for b in report.bins] == [1, 2, 0, 0, 1]
    assert report.bins[1].mean_score == Decimal("0.2") and report.bins[1].observed_up_rate == Decimal("0.5")
    assert report.ece == Decimal("0.15")
    assert report.bins[2].mean_score is None and report.bins[2].observed_up_rate is None
    assert not any(b.enough_samples for b in report.bins)


@pytest.mark.parametrize("labels,scores", [((), ()), ((1,), (Decimal("NaN"),)), ((2,), (Decimal("0.5"),))])
def test_invalid_reliability_inputs_rejected(labels, scores):
    with pytest.raises(ValueError):
        calibration.reliability_report(labels, scores)


def test_global_shortage_and_bad_hash_do_not_fit(calibration_batches, prepared, monkeypatch):
    monkeypatch.setattr(calibration, "fit_logistic_artifact", lambda *a, **kw: pytest.fail("should not fit"))
    with pytest.raises(training.HistoricalNavTrainingError) as error:
        calibration.evaluate_prepared_calibration(request_for(calibration_batches), prepared)
    assert error.value.code == "DATASET_HASH_MISMATCH"
    data = replace(prepared, report=prepared.report.model_copy(update={"status": "INSUFFICIENT_DATA"}))
    response = evaluate(calibration_batches, data)
    assert response.status == "INSUFFICIENT_DATA" and response.evaluated_window_count == 0
    assert all(w.model is None and w.reason == "GLOBAL_DATA_INSUFFICIENT" for w in response.windows)


def test_window_shortage_retained_not_dropped(calibration_batches, prepared):
    # 只移除第一窗考试行；不能自动换考试日期补足，也不删掉窗口记录。
    window = calibration.WINDOWS[0]
    data = replace(
        prepared,
        train=tuple(
            r for r in prepared.train if not window.calibration_end_date < r.available_at <= window.evaluation_end_date
        ),
    )
    response = evaluate(calibration_batches, data)
    assert len(response.windows) == 3 and response.windows[0].status == "INSUFFICIENT_DATA"
    assert response.windows[0].funds[0].missing["EXAM"] == 40
    assert response.windows[0].model is None


def test_single_class_calibration_rejected_without_fake_model(prepared, result):
    rows, _ = calibration._window_rows(prepared, calibration.WINDOWS[0])
    with pytest.raises(training.HistoricalNavTrainingError) as error:
        calibration.fit_calibrator(
            result.windows[0].model.base_model,
            tuple(replace(r, y=0) for r in rows["CALIBRATION"]),
            end_date=calibration.WINDOWS[0].calibration_end_date,
        )
    assert error.value.code == "CALIBRATION_SINGLE_CLASS"


@pytest.mark.parametrize("all_train", [False, True])
def test_rejected_windows_are_reported_not_selected_away(calibration_batches, prepared, all_train):
    first = calibration.WINDOWS[0]
    data = replace(
        prepared,
        train=tuple(
            replace(r, y=0, content_hash="f" * 64)
            if all_train or first.fit_end_date < r.available_at <= first.calibration_end_date
            else r
            for r in prepared.train
        ),
    )
    report = evaluate(calibration_batches, data)
    assert len(report.windows) == 3 and report.windows[0].model is None
    assert report.windows[0].status == "REJECTED"
    assert report.status == ("NO_VALID_WINDOWS" if all_train else "PARTIAL_EVALUATION")
    assert report.evaluated_window_count == (0 if all_train else 2)
    assert report.publication_status == "MODEL_NOT_RELEASED"


def test_calibrator_never_refits_base_and_uses_fund_weights(prepared, result, monkeypatch):
    rows, _ = calibration._window_rows(prepared, calibration.WINDOWS[0])
    selected = rows["CALIBRATION"]
    extra = tuple(replace(r, fund_code="001632") for r in selected[:60])
    original_fit = LogisticRegression.fit
    seen = []

    def spy(self, x, y, **kwargs):
        seen.append((x.shape, kwargs["sample_weight"]))
        return original_fit(self, x, y, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", spy)
    fitted = calibration.fit_calibrator(
        result.windows[0].model.base_model, selected + extra, end_date=calibration.WINDOWS[0].calibration_end_date
    )
    assert len(seen) == 1 and seen[0][0][1] == 1  # 只拟合一个一维映射，不再训练七指标模型。
    weights = fitted.calibrator.weights_per_fund
    assert weights["001632"] * 60 == pytest.approx(weights["008888"] * len(selected))


def test_calibrator_cutoff_convergence_and_deadline(calibration_batches, prepared, result, monkeypatch):
    rows, _ = calibration._window_rows(prepared, calibration.WINDOWS[0])
    with pytest.raises(training.HistoricalNavTrainingError, match="时间边界"):
        calibration.fit_calibrator(
            result.windows[0].model.base_model, rows["FIT"], end_date=calibration.WINDOWS[0].calibration_end_date
        )

    def fail(*a, **kw):
        warnings.warn("synthetic convergence failure", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(LogisticRegression, "fit", fail)
    with pytest.raises(training.HistoricalNavTrainingError) as error:
        calibration.fit_calibrator(
            result.windows[0].model.base_model,
            rows["CALIBRATION"],
            end_date=calibration.WINDOWS[0].calibration_end_date,
        )
    assert error.value.code == "CALIBRATION_NOT_CONVERGED"
    monkeypatch.setattr(calibration, "COMPUTE_SECONDS", -1)
    with pytest.raises(training.HistoricalNavTrainingError) as error:
        evaluate(calibration_batches, prepared)
    assert error.value.code == "CALIBRATION_TIMEOUT"


@pytest.mark.parametrize("mutation", ["slope", "base", "date", "unknown"])
def test_corrupt_composite_json_rejected(result, mutation):
    payload = result.windows[0].model.model_dump(mode="json")
    if mutation == "slope":
        payload["calibrator"]["slope"] += 1
    elif mutation == "base":
        payload["calibrator"]["base_model_hash"] = "0" * 64
    elif mutation == "date":
        payload["calibrator"]["start_date"] = "2022-01-01"
    else:
        payload["command"] = "do not execute"
    with pytest.raises(ValueError):
        calibration.restore_calibrated_artifact(json.dumps(payload))


def test_http_success_and_shared_busy_lock(calibration_batches, prepared, result, client, monkeypatch):
    monkeypatch.setattr(calibration, "load_historical_nav_dataset", lambda request: prepared)
    body = request_for(calibration_batches, prepared.report.dataset_hash).model_dump(mode="json", by_alias=True)
    response = client.post(PATH, json=body, headers=HEADERS)
    assert response.status_code == 200 and response.json() == result.model_dump(mode="json")
    with training._TRAINING_LOCK:
        assert client.post(PATH, json=body, headers=HEADERS).status_code == 429


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost"}])
def test_http_authorization_before_work(calibration_batches, prepared, client, monkeypatch, headers):
    monkeypatch.setattr(api, "evaluate_stored_calibration", lambda request: pytest.fail("unauthorized work"))
    body = request_for(calibration_batches, prepared.report.dataset_hash).model_dump(mode="json", by_alias=True)
    assert client.post(PATH, json=body, headers=headers).status_code == 403


@pytest.mark.parametrize(
    "update",
    [
        {"includeTest": True},
        {"windows": []},
        {"method": "isotonic"},
        {"minFit": 10},
        {"expectedDatasetHash": "bad"},
        {"trainEndDate": "2024-12-31"},
        {"testEndDate": "2026-12-31"},
        {"previewSize": True},
    ],
)
def test_http_rejects_mutable_protocol(calibration_batches, prepared, client, monkeypatch, update):
    monkeypatch.setattr(api, "evaluate_stored_calibration", lambda request: pytest.fail("invalid work"))
    body = request_for(calibration_batches, prepared.report.dataset_hash).model_dump(mode="json", by_alias=True)
    assert client.post(PATH, json=body | update, headers=HEADERS).status_code == 422


def test_http_errors_do_not_leak(calibration_batches, prepared, client, monkeypatch):
    def fail(request):
        raise SQLAlchemyError("private database connection")

    monkeypatch.setattr(api, "evaluate_stored_calibration", fail)
    body = request_for(calibration_batches, prepared.report.dataset_hash).model_dump(mode="json", by_alias=True)
    response = client.post(PATH, json=body, headers=HEADERS)
    assert response.status_code == 503 and "private database connection" not in response.text


def test_cli_saves_models_and_refuses_overwrite(tmp_path, calibration_batches, prepared, result, monkeypatch):
    from scripts import historical_nav_calibration as cli

    monkeypatch.setattr(cli, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(ai_database_url="postgresql://localhost/fund_ai"))
    monkeypatch.setattr(cli, "evaluate_stored_calibration", lambda request: result)
    key = uuid4()
    directory, manifest = cli.run_calibration(request_for(calibration_batches, prepared.report.dataset_hash), key)
    assert manifest["artifact_persisted"] and manifest["evaluated_window_count"] == 3
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    models = json.loads((directory / "models.json").read_text(encoding="utf-8"))
    for w in result.windows:
        assert calibration.restore_calibrated_artifact(json.dumps(models[w.window.window_id])) == w.model
    with pytest.raises(FileExistsError):
        cli.run_calibration(request_for(calibration_batches, prepared.report.dataset_hash), key)


def test_cli_disk_failure_no_completion(tmp_path, calibration_batches, prepared, result, monkeypatch):
    from scripts import historical_nav_calibration as cli

    monkeypatch.setattr(cli, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(ai_database_url="postgresql://localhost/fund_ai"))
    monkeypatch.setattr(cli, "evaluate_stored_calibration", lambda request: result)
    original = cli._write_json

    def fail(path, payload):
        if path.name == "models.json":
            raise OSError("synthetic disk failure")
        return original(path, payload)

    monkeypatch.setattr(cli, "_write_json", fail)
    key = uuid4()
    with pytest.raises(OSError):
        cli.run_calibration(request_for(calibration_batches, prepared.report.dataset_hash), key)
    assert not (tmp_path / f"nav-calibration-{key}" / "complete.json").exists()
