"""单日广度的时间隔离、原对照保护、13维数值恢复及有限训练预算验收。"""

from copy import deepcopy
from datetime import datetime, time, timedelta

import numpy as np
import pytest
from app.services import direction_1d_breadth_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest
from sklearn.preprocessing import StandardScaler


def daily(day="2024-09-30", up=1200):
    return {
        "date": day,
        "status": "READY",
        "up_fraction": up / 4000,
        "counts": {"valid": 4000, "up": up, "flat": 200, "down": 3800 - up, "SH": 2000, "SZ": 2000},
    }


def row():
    return {"fund_code": "synthetic", "t": "2024-09-30", "u": "2024-10-08", "y": 0, "input_hash": "unchanged"}


def test_single_t_day_fraction_includes_flats_and_preserves_original():
    original = row()
    result = study.attach([original], {"2024-09-30": daily(), "2024-10-08": daily("2024-10-08", 3000)})[0]
    assert result["breadth_input"]["x"] == [0.3]
    assert result["breadth_input"]["available_at_assumed"] == "2024-09-30T18:00:00+08:00"
    assert "breadth_input" not in original
    assert study.controls([result]) == [original]


@pytest.mark.parametrize(
    "mutation", ["missing", "date", "status", "sum", "exchange", "float_count", "negative", "fraction", "nan"]
)
def test_missing_or_invalid_day_never_silently_removes_question(mutation):
    d = daily()
    if mutation == "date":
        d["date"] = "2024-10-08"
    if mutation == "status":
        d["status"] = "INSUFFICIENT_SOURCE_COVERAGE"
    if mutation == "sum":
        d["counts"]["flat"] += 1
    if mutation == "exchange":
        d["counts"]["SH"] = 999
    if mutation == "float_count":
        d["counts"]["up"] = 1200.0
    if mutation == "negative":
        d["counts"]["up"] = -1
    if mutation == "fraction":
        d["up_fraction"] += 0.01
    if mutation == "nan":
        d["up_fraction"] = float("nan")
    with pytest.raises(ValueError, match="BREADTH1D"):
        study.attach([row()], {} if mutation == "missing" else {"2024-09-30": d})


@pytest.mark.parametrize(
    "t,u",
    [
        ("2024-09-30", "2024-10-09"),
        ("2024-09-30", "2024-09-30"),
        ("2024-12-31", "2025-01-02"),
        ("2026-09-11", "2026-09-14"),
    ],
)
def test_wrong_next_session_or_protected_year_rejected(t, u):
    with pytest.raises(ValueError, match="DATE_IDENTITY"):
        study.attach([{**row(), "t": t, "u": u}], {t: daily(t)})


def test_duplicate_or_already_attached_rows_rejected():
    with pytest.raises(ValueError, match="DUPLICATE"):
        study.attach([row(), row()], {"2024-09-30": daily()})
    attached = study.attach([row()], {"2024-09-30": daily()})
    with pytest.raises(ValueError, match="DUPLICATE"):
        study.attach(attached, {"2024-09-30": daily()})


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", False),
        ("source_code", "other"),
        ("authorization_verified_at", None),
        ("authorized_api_names", ["index_daily"]),
        ("retention_days", 0),
    ],
)
def test_source_revocation_or_missing_daily_capability_blocks_freeze(field, value):
    source = {
        "source_code": "TUSHARE_PRO_FUND",
        "enabled": True,
        "authorization_verified_at": "verified",
        "authorized_api_names": ["daily"],
        "retention_days": 365,
    }
    study.validate_source(source)
    with pytest.raises(ValueError, match="SOURCE_UNAVAILABLE"):
        study.validate_source({**source, field: value})


def test_breadth_cannot_arrive_after_prediction_deadline(monkeypatch):
    checked = study.attach([row()], {"2024-09-30": daily()})
    monkeypatch.setattr(study.full, "check_available", lambda *a, **kw: None)
    deadline = datetime(2024, 10, 8, 8, tzinfo=ZONE)
    study.available(checked, deadline, fit=False)
    checked[0]["breadth_input"]["available_at_assumed"] = (deadline + timedelta(microseconds=1)).isoformat()
    with pytest.raises(ValueError, match="INPUT_TOO_LATE"):
        study.available(checked, deadline, fit=False)


@pytest.fixture(scope="module")
def fitted():
    # 合成标签只由第13维决定，确保测试实际走新增维度分裂；无真实数据或上游请求。
    rng = np.random.default_rng(20260912)
    days = calendar()[0]
    values = rng.normal(size=(540, 12))
    up_counts = rng.integers(300, 3700, size=540)
    rows = []
    for j, i in enumerate([*range(100, 600), *range(610, 650)]):
        t, u, mature = days[i : i + 3]
        x = values[j]
        r = {
            "fund_code": "synthetic",
            "family": "synthetic",
            "group": "CN_EQUITY",
            "t": str(t),
            "u": str(u),
            "y": int(up_counts[j] > 2000),
            "x": x[:7].tolist(),
            "mature_at": datetime.combine(mature, time(8), ZONE).isoformat(),
            "nav_available_at_assumed": datetime.combine(u, time(8), ZONE).isoformat(),
        }
        for key, extra in (("market_input", x[7:9]), ("specific_input", x[9:11]), ("activity_input", x[11:12])):
            r[key] = {"x": extra.tolist(), "available_at_assumed": datetime.combine(t, time(18), ZONE).isoformat()}
        rows.extend(study.attach([r], {str(t): daily(str(t), int(up_counts[j]))}))
    training, exam = rows[:500], rows[500:]
    w = study.weights(training)
    scaler = StandardScaler().fit(values[:500], sample_weight=w)
    control = {
        "candidate": study.activity.CANDIDATE,
        "features": study.activity.FEATURES,
        "feature_recipe": study.activity.FEATURE_RECIPE,
        "recipe": study.tree.TREE_RECIPE,
        "kind": "RESEARCH_ONLY",
        "model_released": False,
        "target_definition": "UNIT_NAV_DIRECTION_V1",
        "train_as_of": datetime.combine(days[609], time(), ZONE).isoformat(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "baseline": 0.0,
        "trees": [
            [
                {
                    "leaf": True,
                    "value": 0.0,
                    "feature": 0,
                    "threshold": 0.0,
                    "left": 0,
                    "right": 0,
                    "depth": 0,
                    "count": 500,
                }
            ]
            for _ in range(100)
        ],
        "fit_hash": digest(study.controls(training)),
        "fit_count": 500,
        "distinct_dates": 500,
        "weight_hash": digest(w.tolist()),
        "classes": {str(y): sum(r["y"] == y for r in training) for y in (0, 1)},
    }
    spec = {"cohort_id": "SYNTHETIC_ONLY", "fit_hashes": {"2024Q1": digest(training)}}
    model = study.fit(training, control, spec, "2024Q1", exam)
    return training, exam, control, spec, model


def test_thirteenth_dimension_restores_and_original_twelve_scales_are_identical(fitted, tmp_path):
    training, exam, control, _, model = fitted
    assert any(not n["leaf"] and n["feature"] == 12 for nodes in model["trees"] for n in nodes)
    assert model["mean"][:12] == control["mean"] and model["scale"][:12] == control["scale"]
    assert model["weight_hash"] == control["weight_hash"]
    assert max(model["restore_max_score_diff"].values()) <= 1e-12
    study.write_new(tmp_path / "model.json", model)
    assert np.array_equal(study.predict(model, training), study.predict(study.read(tmp_path / "model.json"), training))
    without_answers = [{k: v for k, v in r.items() if k != "y"} for r in exam]
    assert np.array_equal(study.predict(model, exam), study.predict(model, without_answers))
    assert len(study.predict(model, [])) == 0


@pytest.mark.parametrize("mutation", ["dimension", "release", "target", "recipe", "nan", "rounds", "cycle", "feature"])
def test_corrupt_or_wrong_protocol_model_never_scores(fitted, mutation):
    training, _, _, _, model = fitted
    m = deepcopy(model)
    if mutation == "dimension":
        m["mean"].pop()
    if mutation == "release":
        m["model_released"] = True
    if mutation == "target":
        m["target_definition"] = "20_DAY"
    if mutation == "recipe":
        m["recipe"]["max_depth"] = 3
    if mutation == "nan":
        m["baseline"] = float("nan")
    if mutation == "rounds":
        m["trees"].pop()
    if mutation == "cycle":
        m["trees"][0][0]["left"] = 0
    if mutation == "feature":
        m["trees"][0][0]["feature"] = 13
    with pytest.raises(ValueError, match="BREADTH1D_MODEL"):
        study.predict(m, training[:1])


@pytest.mark.parametrize("mutation", ["member", "weight", "future", "overlap"])
def test_fit_rejects_changed_control_or_time_overlap_before_training(fitted, mutation):
    training, exam, control, spec, _ = fitted
    c, checked = deepcopy(control), exam
    if mutation == "member":
        c["fit_hash"] = "changed"
    if mutation == "weight":
        c["weight_hash"] = "changed"
    if mutation == "future":
        c["train_as_of"] = training[0]["mature_at"]
    if mutation == "overlap":
        checked = training[:1]
    with pytest.raises(ValueError):
        study.fit(training, c, spec, "2024Q1", checked)


def test_file_seal_rejects_change_and_budget_reentry(tmp_path, monkeypatch):
    study.write_new(tmp_path / "input.json", {"original": True})
    spec = {
        "fingerprint": study.fingerprint(),
        "candidate": study.CANDIDATE,
        "reference": study.REFERENCE,
        "features": study.FEATURES,
        "feature_recipe": study.FEATURE_RECIPE,
        "tree_recipe": study.tree.TREE_RECIPE,
        "observation_rule": study.OBSERVATION_RULE,
        "threshold": 0.5,
        "max_main_fits": 4,
        "max_replay_fits": 4,
        "protected_years_excluded": [2025, 2026],
        "new_api_calls": 0,
        "database_writes": 0,
        "model_released": False,
        "source_expires_at": (datetime.now(ZONE) + timedelta(days=1)).isoformat(),
        "input_files": {"input.json": study.file_hash(tmp_path / "input.json")},
    }
    study.write_new(tmp_path / "study.json", spec)
    study.write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    study.verify_inputs(tmp_path)
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="INPUT_CHANGED"):
        study.verify_inputs(tmp_path)
    with pytest.raises(FileExistsError):
        study.write_new(tmp_path / "study.json", spec)
    monkeypatch.setattr(study, "verify_inputs", lambda _: spec)
    for n in ("universe.json", "schedule.json", "exam.json"):
        study.write_new(tmp_path / n, [])
    (tmp_path / "control/main").mkdir(parents=True)
    study.write_new(tmp_path / "control/main/models.json", {})
    (tmp_path / "main").mkdir()
    monkeypatch.setattr(study, "fit", lambda *a, **kw: pytest.fail("重复预算不得调用fit"))
    with pytest.raises(FileExistsError):
        study.run(tmp_path)


def test_cli_uses_one_numeric_thread_for_verification_and_restores_caller(monkeypatch, capsys):
    """回归本轮现场发现的16线程求和差异；只读verify同样必须遵守冻结的单线程配方。"""
    import sys

    from scripts import direction_1d_breadth_study as cli
    from threadpoolctl import threadpool_info, threadpool_limits

    def check(_):
        assert all(p["num_threads"] == 1 for p in threadpool_info())
        return {"verified": True, "new_fits": 0}

    monkeypatch.setattr(sys, "argv", ["breadth", "verify", "--run-dir", "synthetic"])
    monkeypatch.setattr(cli.study, "verify", check)
    with threadpool_limits(limits=2):
        before = [p["num_threads"] for p in threadpool_info()]
        cli.main()
        assert [p["num_threads"] for p in threadpool_info()] == before
    assert '"verified":true' in capsys.readouterr().out
