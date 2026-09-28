"""离线训练验收：只用合成数据测试异常，不把测试拟合冒充本轮真实训练。"""

import ast
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pytest
from app.services import fund_002112_training as run
from app.services import fund_002112_training_model as model
from app.services import fund_training_package as package
from app.services.direction_1d_protocol import digest
from app.services.direction_1d_training import weights
from app.services.fund_exposure_common import now, read, save
from sklearn.exceptions import ConvergenceWarning


@pytest.fixture
def dataset():
    days = [str(d) for d in package.own.sessions()[0]]
    chosen = days[100:400]
    dev = [d for d in days if d.startswith("2024")][:12]

    def row(target, i):
        k = model.CLASSES[i % 3]
        vector = [float((i + j) % 7) / 100 for j in range(20)]
        return {
            "fund_code": "002112",
            "family": "FUND_001412",
            "base": days[days.index(target) - 1],
            "t": days[days.index(target) - 1],
            "target": target,
            "actual_direction": k,
            "base_unit_nav": "1.00000000",
            "target_unit_nav": {"UP": "1.00000001", "FLAT": "1.00000000", "DOWN": "0.99999999"}[k],
            "mature_at": target + "T20:00:00+08:00",
            "as_of": target + "T08:00:00+08:00",
            "x": vector,
            "nav_features": vector[:7],
            "exposure": {
                "holdings_features": vector[7:16],
                "market_features": vector[16:],
                "report_available_at": "2015-12-01T08:00:00+08:00",
            },
        }

    rows = [row(d, i) for i, d in enumerate(chosen)]
    return {
        "train": rows,
        "development": [row(d, i) for i, d in enumerate(dev)],
        "weights": weights(rows).tolist(),
        "excluded": [],
    }


def test_e03_e04_e06_validated_three_classes_weights_and_exact_decimal(dataset):
    model.validate_data(dataset)
    assert {r["actual_direction"] for r in dataset["train"]} == set(model.CLASSES)
    for mutation, reason in [
        (lambda d: d["train"][0].update(target="2025-01-02"), "TRADING_DATE|FUTURE_LABEL"),
        (lambda d: d["train"][0].update(mature_at="2024-01-02T08:00:00+08:00"), "IMMATURE"),
        (lambda d: d["train"][0]["exposure"].update(report_available_at="2026-01-01T08:00:00+08:00"), "FUTURE_REPORT"),
        (lambda d: d["train"][0].update(actual_direction="FLAT"), "EXACT_LABEL"),
        (lambda d: d["train"].append(deepcopy(d["train"][0])), "DUPLICATE_FAMILY"),
        (lambda d: d["weights"].__setitem__(0, 0), "WEIGHTS_CHANGED"),
        (lambda d: d["development"][0].update(fund_code="006038"), "DEVELOPMENT_SCOPE"),
    ]:
        bad = deepcopy(dataset)
        mutation(bad)
        with pytest.raises(ValueError, match=reason):
            model.validate_data(bad)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "1", True])
def test_e05_nonfinite_missing_and_wrong_type_rejected(dataset, value):
    dataset["train"][0]["x"][0] = value
    with pytest.raises(ValueError, match="VALUE_OR_SHAPE"):
        model.validate_data(dataset)


def test_e05_column_order_rejected(dataset):
    dataset["train"][0]["x"].reverse()
    with pytest.raises(ValueError, match="INPUT_ORDER"):
        model.validate_data(dataset)


def test_e04_threshold_does_not_count_many_families_same_day(dataset):
    rows = dataset["train"]
    dataset["train"] = [r for r in rows if r["actual_direction"] != "FLAT"]
    dataset["weights"] = weights(dataset["train"]).tolist()
    with pytest.raises(ValueError, match="SAMPLE_GATE"):
        model.validate_data(dataset)


@pytest.mark.parametrize("variant", model.VARIANTS)
def test_e05_e07_independent_fit_restore_train_only_scaling(dataset, variant):
    # 合成数值单测，不读取真实 002112 包，也不计入真实运行六次预算。
    for r in dataset["development"]:
        r["x"] = [1000 + x for x in r["x"]]
    first, second = model.fit(dataset, variant), model.fit(dataset, variant)
    assert first == second
    size = model.VARIANTS[variant]
    expected_mean = np.average([r["x"][:size] for r in dataset["train"]], axis=0, weights=dataset["weights"])
    assert np.allclose(first["model"]["mean"], expected_mean, atol=1e-15)
    assert first["restore"]["development"]["max_score_difference"] <= 1e-12
    with pytest.raises(ValueError, match="VALUE_OR_SHAPE"):
        model.predict(first["model"], [0.0] * (size + 1))
    changed = deepcopy(first["model"])
    changed["features"].reverse()
    with pytest.raises(ValueError, match="PROTOCOL"):
        model.predict(changed, [0.0] * size)


def fake_fit(data, variant):
    size = model.VARIANTS[variant]
    return {
        "model": {
            "protocol": model.PROTOCOL,
            "variant": variant,
            "fund_code": "002112",
            "horizon": 1,
            "features": package.PLAN["features"][:size],
            "recipe": model.RECIPE,
            "class_order": list(model.CLASSES),
            "tie_order": list(model.TIE_ORDER),
            "mean": [0.0] * size,
            "scale": [1.0] * size,
            "coef": [[0.0] * size for _ in model.CLASSES],
            "intercept": [0.0] * 3,
            "dataset_hash": digest(data),
            "weight_hash": digest(data["weights"]),
        },
        "restore": {"test_double": True},
    }


def prepare_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RUN_ROOT", tmp_path)
    directory = tmp_path / ("002112-" + "a" * 24)
    save(
        directory / "state.json", {"status": "FROZEN", "events": [], "fit_attempts": 0, "completed": [], "failures": {}}
    )
    save(directory / "protection-before.json", {"unchanged": True})
    monkeypatch.setattr(run, "protection", lambda: {"unchanged": True})
    return directory


def test_e08_repeat_no_fit_and_e10_report_recomputable(dataset, tmp_path, monkeypatch):
    directory = prepare_run(tmp_path, monkeypatch)
    calls = []

    def fit(data, variant):
        calls.append(variant)
        return fake_fit(data, variant)

    monkeypatch.setattr(model, "fit", fit)
    first = run.run_stages(directory, dataset)
    second = run.run_stages(directory, dataset)
    assert first == second and first["models"] == 3 and first["fit_attempts"] == 6
    assert len(calls) == 6
    monkeypatch.setattr(run, "load_frozen", lambda d: dataset)
    assert run.export_report(directory.name)["recomputed_equal"]
    assert len(calls) == 6


def test_e09_hard_exit_after_checkpoint_resumes_without_extra_fit(dataset, tmp_path, monkeypatch):
    directory = prepare_run(tmp_path, monkeypatch)
    calls = []

    def fit(data, variant):
        calls.append(variant)
        return fake_fit(data, variant)

    monkeypatch.setattr(model, "fit", fit)
    monkeypatch.setattr(run.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit):
        run.run_stages(directory, dataset, interrupt_after_checkpoint="NAV7:main")
    assert len(calls) == 1 and read(directory / "state.json")["status"] == "RUNNING"
    assert run.status(directory.name)["status"] == "INTERRUPTED"
    result = run.run_stages(directory, dataset)
    assert result["status"] == "COMPLETED" and result["fit_attempts"] == len(calls) == 6


def test_e09_fit_interruption_not_retried_other_variants_finish(dataset, tmp_path, monkeypatch):
    directory = prepare_run(tmp_path, monkeypatch)
    save(directory / "attempts/NAV7-main.json", {"stage": "NAV7:main"})
    calls = []

    def fit(data, variant):
        calls.append(variant)
        return fake_fit(data, variant)

    monkeypatch.setattr(model, "fit", fit)
    result = run.run_stages(directory, dataset)
    assert result["status"] == "PARTIAL" and result["models"] == 2 and result["fit_attempts"] == 5
    assert "NAV7" not in calls and "INTERRUPTED" in result["failures"]["NAV7"]
    assert run.run_stages(directory, dataset) == result and len(calls) == 4


def test_e08_convergence_failure_consumes_budget(dataset, tmp_path, monkeypatch):
    directory = prepare_run(tmp_path, monkeypatch)

    def failure(*args):
        raise ConvergenceWarning("synthetic convergence failure")

    monkeypatch.setattr(model, "fit", failure)
    result = run.run_stages(directory, dataset)
    assert result["status"] == "FAILED" and result["fit_attempts"] == 3
    assert all(v == "ConvergenceWarning" for v in result["failures"].values())
    assert run.run_stages(directory, dataset) == result


def test_e08_budget_exhausted_before_numerical_call(dataset, tmp_path, monkeypatch):
    directory = prepare_run(tmp_path, monkeypatch)
    for i in range(6):
        save(directory / f"attempts/reserved-{i}.json", {})
    monkeypatch.setattr(model, "fit", lambda *args: pytest.fail("budget must reject"))
    result = run.run_stages(directory, dataset)
    assert result["fit_attempts"] == 6 and result["status"] == "FAILED"


def test_e08_process_lock_and_path_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RUN_ROOT", tmp_path)
    with run.run_lock(), pytest.raises(ValueError, match="BUSY"), run.run_lock():
        pass
    with run.run_lock():
        pass
    for value in ("../escape", "C:/escape", "002112-123", ""):
        with pytest.raises(ValueError, match="RUN_ID_INVALID"):
            run.checked_directory(value)


def test_e02_expired_and_tampered_sources(tmp_path):
    path = tmp_path / "source.json"
    path.write_bytes(b"public test source")
    receipt = {"path": str(path), "sha256": run.file_hash(path), "expires_at": (now() + timedelta(days=1)).isoformat()}
    run.verify_sources([receipt])
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        run.verify_sources([receipt])
    receipt["expires_at"] = (now() - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="SOURCE_EXPIRED"):
        run.verify_sources([receipt])


def test_e01_frozen_pointer_ignored_e02_hash_and_environment(dataset, tmp_path, monkeypatch):
    directory = prepare_run(tmp_path, monkeypatch)
    store = tmp_path / "package"
    monkeypatch.setattr(package, "STORE", store)
    source = {"fixed_source": True}
    dataset.update(source_file="sources/test.json", calendar_hash=package.own.sessions()[1])
    save(store / "sources/test.json", source)
    save(store / "datasets/test.json", dataset)
    save(store / "plan.json", package.PLAN)
    save(store / "ready.json", {"file": "datasets/does-not-exist.json", "sha256": "new-background-version"})
    save(directory / "dataset.json", dataset)
    save(
        directory / "input-manifest.json",
        {
            "dataset_hash": digest(dataset),
            "rows": run.frozen_identity(dataset),
            "row_hash": digest(run.frozen_identity(dataset)),
            "sources": [],
        },
    )
    spec = {
        "code": {"test": "1"},
        "environment": {"test": "1"},
        "recipe": model.RECIPE,
        "selection": model.SELECTION,
        "reference": {"sha256": digest(dataset), "file": "datasets/test.json"},
        "source_hash": digest(source),
    }
    save(directory / "protocol.json", spec)
    monkeypatch.setattr(run, "code_manifest", lambda: {"test": "1"})
    monkeypatch.setattr(run, "environment", lambda: {"test": "1"})
    assert run.load_frozen(directory) == dataset
    monkeypatch.setattr(run, "environment", lambda: {"test": "2"})
    with pytest.raises(ValueError, match="CODE_OR_ENV_CHANGED"):
        run.load_frozen(directory)
    monkeypatch.setattr(run, "environment", lambda: {"test": "1"})
    changed = deepcopy(dataset)
    changed["weights"][0] += 1
    save(directory / "dataset.json", changed, replace=True)
    with pytest.raises(ValueError, match="FROZEN_DATA_CHANGED"):
        run.load_frozen(directory)


def test_e10_e12_same_dates_null_quarters_ties_and_constants(dataset):
    m = fake_fit(dataset, "NAV7")["model"]
    assert model.predict(m, [0.0] * 7)["direction"] == "FLAT"
    rows = {v: model.predictions(fake_fit(dataset, v)["model"], dataset["development"]) for v in model.VARIANTS}
    report = model.compare(dataset, rows)
    assert report["models"]["NAV7"]["correct"] == 4
    assert set(report["constants"]) == set(model.CLASSES)
    assert report["quarters"]["2"]["NAV7"]["recall"]["FLAT"] is None
    assert report["conclusion"] == "NO_DEMONSTRATED_HELP" and report["adopted"] is False
    assert "该季度没有这类样本" in model.render_report(report)
    rows["NAV7_HOLDINGS"].reverse()
    with pytest.raises(ValueError, match="COMPARISON_ROWS_CHANGED"):
        model.compare(dataset, rows)


def test_e11_no_registry_training_or_sync_trigger_import():
    for source in (run, model):
        tree = ast.parse(__import__("pathlib").Path(source.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(n, ast.Attribute) and n.attr in {"train_registered", "_train_registered"} for n in ast.walk(tree)
        )
    service_root = __import__("pathlib").Path(run.__file__).parent
    for name in ("fund_exposure_runtime.py", "fund_materials_sync.py"):
        path = service_root / name
        if path.exists():
            assert "fund_002112_training" not in path.read_text(encoding="utf-8")


def test_e02_explicit_reference_cannot_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "STORE", tmp_path)
    save(tmp_path / "plan.json", package.PLAN)
    with pytest.raises(ValueError, match="REFERENCE_READ_ONLY"):
        package.verify(reference={"file": "anything"})


def test_e12_fixed_selection_requires_all_classes_and_prefers_fewer_features(dataset):
    rows = {v: model.predictions(fake_fit(dataset, v)["model"], dataset["development"]) for v in model.VARIANTS}
    for row in rows["NAV7"]:
        row["direction"] = "UP"
    for variant in ("NAV7_HOLDINGS", "NAV7_HOLDINGS_MARKET"):
        for row in rows[variant]:
            row["direction"] = row["actual_direction"]
    assert model.compare(dataset, rows)["selected_for_further_experiment"] == "NAV7_HOLDINGS"
    # 总数更好却少判对一天上涨，仍必须拒绝，不能用多数类优势改写规则。
    next(r for r in rows["NAV7_HOLDINGS"] if r["actual_direction"] == "UP")["direction"] = "DOWN"
    for row in rows["NAV7_HOLDINGS_MARKET"]:
        row["direction"] = "UP"
    assert model.compare(dataset, rows)["selected_for_further_experiment"] is None


def test_e11_protection_change_cannot_report_complete(dataset, tmp_path, monkeypatch):
    directory = prepare_run(tmp_path, monkeypatch)
    monkeypatch.setattr(model, "fit", fake_fit)
    monkeypatch.setattr(run, "protection", lambda: {"unchanged": False})
    result = run.run_stages(directory, dataset)
    assert result["status"] == "PARTIAL" and result["protected_unchanged"] is False
    assert result["models"] == 3 and result["fit_attempts"] == 6
