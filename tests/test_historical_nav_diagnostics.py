"""只读诊断的手算、隔离、旧模型复现、源数据敏感性和本机产物保护测试。"""

import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from app.services import historical_nav_diagnostics as diagnostics
from app.services.historical_nav_calibration import calibrated_model_hash
from scripts import historical_nav_calibration as calibration_cli
from scripts import historical_nav_diagnostics as cli
from tests.test_historical_nav_calibration import calibration_batches as calibration_batches
from tests.test_historical_nav_calibration import prepared as prepared
from tests.test_historical_nav_calibration import request_for
from tests.test_historical_nav_calibration import result as result


@pytest.fixture(scope="module")
def raw_rows():
    start = date(2021, 11, 1)
    return [
        {
            "fund_code": "008888",
            "source_id": UUID(int=2),
            "nav_date": start + timedelta(days=i),
            "ann_date": start + timedelta(days=i + 1),
            "unit_nav": v,
            "accumulated_nav": v,
            "adjusted_nav": v,
            "source_published_at": None,
            "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        }
        for i in range(1157)
        if (v := Decimal(2) + Decimal(str(math.sin(i / 17))) / 7 + Decimal(i) / 10000)
    ]


def test_no_training_and_exact_old_exam_replay(prepared, result, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    monkeypatch.setattr(LogisticRegression, "fit", lambda *a, **kw: pytest.fail("diagnostics must never fit"))
    payload = diagnostics.diagnose_prepared(prepared, result)
    assert payload["model_fitted"] is False and payload["test_scored"] is False
    assert payload["publication_status"] == "MODEL_NOT_RELEASED"
    for window, old in zip(payload["windows"], result.windows, strict=True):
        assert window["stages"]["EXAM"]["overall"]["after"] == old.after.validation.model_dump(mode="json")
        assert len(window["stages"]["EXAM"]["features"]) == 7
        assert window["model_hash"] == old.model.model_hash
    final = payload["windows"][-1]
    assert (
        sum(q["overall"]["sample_count"] for q in final["exam_quarters"].values())
        == final["stages"]["EXAM"]["overall"]["sample_count"]
    )


def test_test_object_never_read(prepared, result, raw_rows):
    class Forbidden:
        def __iter__(self):
            pytest.fail("TEST must not be read by diagnostics")

    poisoned = replace(prepared, test=Forbidden())
    assert diagnostics.diagnose_prepared(poisoned, result) == diagnostics.diagnose_prepared(prepared, result)
    assert diagnostics.audit_nav_snapshot(poisoned, raw_rows) == diagnostics.audit_nav_snapshot(prepared, raw_rows)


def test_modified_training_or_calibration_inputs_rejected(prepared, result):
    row = prepared.train[0]
    # 正式准备层会从完整源样本重算content_hash；这里同时模拟载荷和其新指纹。
    changed = replace(
        prepared, train=(replace(row, x=(Decimal(99), *row.x[1:]), content_hash="f" * 64), *prepared.train[1:])
    )
    with pytest.raises(ValueError, match="input"):
        diagnostics.diagnose_prepared(changed, result)


@pytest.mark.parametrize("mutation", ["dataset", "protocol", "incomplete", "metrics"])
def test_changed_experiment_rejected(prepared, result, mutation):
    saved = result.model_copy(deep=True)
    if mutation == "dataset":
        saved.preparation.dataset_hash = "0" * 64
    elif mutation == "protocol":
        saved.protocol = saved.protocol.model_copy(update={"minimum_fit_per_fund": 1})
    elif mutation == "incomplete":
        saved.status = "PARTIAL_EVALUATION"
    else:
        saved.windows[-1].before.validation.correct_count += 1
    with pytest.raises(ValueError):
        diagnostics.diagnose_prepared(prepared, saved)


def test_statistics_empty_single_class_and_hand_calculated(prepared, result):
    model = result.windows[0].model
    assert diagnostics._statistics((), model)["status"] == "EMPTY_GROUP"
    rows = prepared.train[:4]
    rows = tuple(
        replace(r, x=tuple(Decimal(str(x)) for x in model.base_model.mean), y=i % 2) for i, r in enumerate(rows)
    )
    stats = diagnostics._statistics(rows, model)
    assert stats["actual_up_rate"] == 0.5 and stats["up_minus_down_mean_logit"] == 0
    assert stats["before"]["correct_count"] == 2
    singles = tuple(replace(r, y=0) for r in rows)
    assert diagnostics._statistics(singles, model)["up_minus_down_mean_logit"] is None


def test_negative_slope_is_not_silently_clamped(prepared, result):
    model = result.windows[0].model
    changed = model.model_copy(
        update={"calibrator": model.calibrator.model_copy(update={"slope": -1.0, "intercept": 0.0})}
    )
    changed = changed.model_copy(update={"model_hash": calibrated_model_hash(changed)})
    stats = diagnostics._statistics(prepared.train, changed)
    assert stats["mean_score_after"] == pytest.approx(1 - stats["mean_score_before"], abs=1e-9)


def test_raw_replay_and_same_basis_match(prepared, raw_rows):
    audit = diagnostics.audit_nav_snapshot(prepared, raw_rows)[0]
    assert audit["saved_sample_count_checked"] == len(prepared.train) + len(prepared.validation)
    assert audit["current_replay_mismatches"] == {}
    assert audit["basis_comparable_count"] == audit["saved_sample_count_checked"]
    assert audit["accumulated_vs_adjusted_direction_difference_count"] == 0
    assert audit["max_absolute_return_gap_bps"] == 0
    assert audit["weekend_nav_dates"] and audit["source_published_at_missing"] == len(raw_rows)


def test_adjusted_sensitivity_does_not_change_saved_labels(prepared, raw_rows):
    alternate = [dict(r, adjusted_nav=Decimal(1) / r["accumulated_nav"]) for r in raw_rows]
    before = prepared.train
    audit = diagnostics.audit_nav_snapshot(prepared, alternate)[0]
    assert audit["current_replay_mismatches"] == {}
    assert audit["accumulated_vs_adjusted_direction_difference_count"] == audit["basis_comparable_count"]
    assert audit["max_absolute_return_gap_bps"] > 0 and prepared.train == before


def test_missing_and_changed_current_nav_reported_not_overwritten(prepared, raw_rows):
    changed = [dict(r) for r in raw_rows]
    changed[100]["accumulated_nav"] += Decimal(1)
    audit = diagnostics.audit_nav_snapshot(prepared, changed)[0]
    assert sum(audit["current_replay_mismatches"].values()) > 0
    assert len(audit["mismatch_examples"]) <= 5
    empty = diagnostics.audit_nav_snapshot(prepared, [])[0]
    assert sum(empty["current_replay_mismatches"].values()) == empty["saved_sample_count_checked"]
    assert empty["max_absolute_return_gap_bps"] is None


@pytest.mark.parametrize("change", ["test_date", "unexpected_fund", "duplicate", "mixed_source", "oversized"])
def test_raw_scope_rejected(prepared, raw_rows, change):
    rows = [dict(r) for r in raw_rows]
    if change == "test_date":
        rows[-1]["nav_date"] = date(2025, 1, 1)
    elif change == "unexpected_fund":
        rows[0]["fund_code"] = "000001"
    elif change == "duplicate":
        rows.insert(1, rows[0])
    elif change == "mixed_source":
        rows[0]["source_id"] = uuid4()
    else:
        rows *= 5
    with pytest.raises(ValueError):
        diagnostics.audit_nav_snapshot(prepared, rows)


@pytest.fixture
def source_run(tmp_path, calibration_batches, prepared, result, raw_rows, monkeypatch):
    monkeypatch.setattr(calibration_cli, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(cli, "RUN_ROOT", tmp_path)

    def settings():
        return SimpleNamespace(ai_database_url="postgresql://localhost/fund_ai")

    monkeypatch.setattr(calibration_cli, "get_settings", settings)
    monkeypatch.setattr(cli, "get_settings", settings)
    monkeypatch.setattr(calibration_cli, "evaluate_stored_calibration", lambda request: result)
    monkeypatch.setattr(cli, "load_historical_nav_dataset", lambda request: prepared)
    monkeypatch.setattr(
        cli,
        "read_nav_diagnostic_snapshot",
        lambda *args: {"nav_rows": raw_rows, "source_registry": {}, "fund_profiles": [], "public_base_tables": []},
    )
    key = uuid4()
    calibration_cli.run_calibration(request_for(calibration_batches, prepared.report.dataset_hash), key)
    return key


def test_cli_source_unchanged_and_manifest_replay(source_run):
    before = cli.read_source_bundle(source_run)[2]
    key = uuid4()
    directory, complete = cli.run_diagnostics(source_run, key)
    assert cli.read_source_bundle(source_run)[2] == before
    assert (
        hashlib.sha256((directory / "report.json").read_bytes()).hexdigest() == complete["files_sha256"]["report.json"]
    )
    assert complete["model_fitted"] is False and complete["admission_status"] == "NOT_APPROVED"
    assert complete["source_files_sha256"] == before
    with pytest.raises(FileExistsError):
        cli.run_diagnostics(source_run, key)


def test_cli_bad_source_hash_and_bounded_read(source_run, tmp_path):
    path = tmp_path / f"nav-calibration-{source_run}" / "report.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        cli.read_source_bundle(source_run)
    path.write_bytes(b"x" * 2097153)
    with pytest.raises(ValueError, match="2MiB"):
        cli._read_bounded(path)


def test_cli_disk_failure_no_complete(source_run, tmp_path, monkeypatch):
    original = cli._write_json

    def fail(path, payload):
        if path.name == "report.json":
            raise OSError("simulated disk failure")
        return original(path, payload)

    monkeypatch.setattr(cli, "_write_json", fail)
    key = uuid4()
    with pytest.raises(OSError):
        cli.run_diagnostics(source_run, key)
    assert not (tmp_path / f"nav-diagnostics-{key}" / "complete.json").exists()


def test_cli_remote_and_invalid_key_rejected(monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(ai_database_url="postgresql://remote/fund_ai"))
    with pytest.raises(ValueError, match="local"):
        cli.run_diagnostics(uuid4(), uuid4())
    with pytest.raises(ValueError, match="UUID"):
        cli._run_directory("diagnostics", "../escape")


@pytest.mark.parametrize("change", ["file_names", "dataset", "model_link", "status"])
def test_source_completion_linkage_rejected(source_run, tmp_path, change):
    path = tmp_path / f"nav-calibration-{source_run}" / "complete.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if change == "file_names":
        manifest["files_sha256"]["../outside.json"] = "0" * 64
    elif change == "dataset":
        manifest["dataset_hash"] = "0" * 64
    elif change == "model_link":
        manifest["model_hashes"]["VALIDATION_2024"] = "0" * 64
    else:
        manifest["status"] = "PARTIAL_EVALUATION"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        cli.read_source_bundle(source_run)


@pytest.mark.parametrize("funds", [(), ("000001",), ("008888", "008888")])
def test_repository_scope_rejected_before_database(funds, monkeypatch):
    from app.repositories import historical_nav_diagnostics as repository

    monkeypatch.setattr(repository, "get_nav_sample_storage_engine", lambda: pytest.fail("invalid DB access"))
    with pytest.raises(ValueError):
        repository.read_nav_diagnostic_snapshot(funds, "TEST")


def test_repository_read_only_source_bounds_and_connection_release(raw_rows, monkeypatch):
    from app.repositories import historical_nav_diagnostics as repository

    source = {"source_id": UUID(int=2), "source_code": "TEST"}
    statements = []

    class Result:
        def __init__(self, value):
            self.value = value

        def mappings(self):
            return self

        def scalars(self):
            return self

        def one(self):
            return self.value

        def all(self):
            return self.value

    class Connection:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def begin(self):
            return self

        def execute(self, statement):
            statements.append(statement)
            return Result(
                [None, source, [{"fund_code": "008888", "source_code": "TEST"}], raw_rows, []][len(statements) - 1]
            )

    conn = Connection()
    monkeypatch.setattr(repository, "get_nav_sample_storage_engine", lambda: SimpleNamespace(connect=lambda: conn))
    snapshot = repository.read_nav_diagnostic_snapshot(("008888",), "TEST")
    assert conn.closed and snapshot["nav_rows"] == raw_rows
    assert str(statements[0]) == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert len(statements) == 5 and all(str(s).startswith("SELECT") for s in statements[1:])
    nav = statements[3]
    params = nav.compile().params.values()
    assert date(2024, 12, 31) in params and date(2021, 1, 1) in params and 4501 in params
    assert UUID(int=2) in params and "nav_daily.source_id =" in str(nav)
