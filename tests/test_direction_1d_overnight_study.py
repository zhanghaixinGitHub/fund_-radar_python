"""隔夜假设性研究：原输入保护、可用假设不可冒充验证、数值恢复和训练预算。"""

from copy import deepcopy
from datetime import datetime, time, timedelta

import numpy as np
import pytest
from app.services import direction_1d_overnight_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def history():
    result = {}
    previous = 100.0
    for i, d in enumerate(study.audit.sessions()):
        close = 100 + i / 10
        result[d] = {
            "ts_code": "SPX",
            "trade_date": d.replace("-", ""),
            "close": close,
            "pre_close": previous,
            "pct_chg": (close / previous - 1) * 100,
        }
        previous = close
    return result


def sample():
    return {
        "fund_code": "synthetic",
        "t": "2024-09-30",
        "u": "2024-10-08",
        "y": 0,
        "input_hash": "original",
        "label_hash": "label",
    }


def test_append_one_overnight_feature_preserves_old_source(history):
    original = sample()
    attached = study.attach([original], history)[0]
    item = attached["overnight_input"]
    assert study.controls([attached]) == [original]
    assert len(item["us_sessions"]) == 6
    assert item["x"] == [history["2024-10-07"]["close"] / history["2024-09-27"]["close"] - 1]
    assert item["available_at_verified"] is None
    assert item["historical_first_version_verified"] is False
    assert item["availability_version"] == "U08_ASSUMED_EXPLORATION_V1"
    assert item["available_at_assumed"] == "2024-10-08T08:00:00+08:00"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "protected", "next_day"])
def test_missing_or_wrong_date_never_drops_questions(history, mutation):
    rows = [sample()]
    if mutation == "missing":
        del history["2024-10-01"]
    if mutation == "duplicate":
        rows.append(sample())
    if mutation == "protected":
        rows[0].update(t="2026-09-11", u="2026-09-14")
    if mutation == "next_day":
        rows[0]["u"] = "2024-10-09"
    with pytest.raises(ValueError):
        study.attach(rows, history)


@pytest.mark.parametrize(
    "field,value",
    [
        ("available_at_verified", "2024-10-08T07:50:00+08:00"),
        ("historical_first_version_verified", True),
        ("available_at_assumed", "2024-10-07T08:00:00+08:00"),
        ("availability_version", "VERIFIED"),
    ],
)
def test_assumption_cannot_be_upgraded_or_backdated(history, monkeypatch, field, value):
    rows = study.attach([sample()], history)
    monkeypatch.setattr(study.full, "check_available", lambda *a, **kw: None)
    rows[0]["overnight_input"][field] = value
    with pytest.raises(ValueError, match="AVAILABILITY_HYPOTHESIS_CHANGED"):
        study.available(rows, datetime(2024, 10, 8, 8, tzinfo=ZONE), fit=False)


def test_u08_feature_not_available_at_previous_midnight(history, monkeypatch):
    rows = study.attach([sample()], history)
    monkeypatch.setattr(study.full, "check_available", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="INPUT_TOO_LATE"):
        study.available(rows, datetime(2024, 10, 8, 0, tzinfo=ZONE), fit=False)


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
        r["overnight_input"] = {
            "x": [float((up_counts[j] - 2000) / 10000)],
            "available_at_assumed": datetime.combine(u, time(8), ZONE).isoformat(),
            "available_at_verified": None,
            "historical_first_version_verified": False,
            "availability_version": study.FEATURE_RECIPE["availability"],
        }
        rows.append(r)
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
    with pytest.raises(ValueError, match="OVERNIGHT1D_MODEL"):
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
        "kind": "HISTORICAL_ONE_DAY_OVERNIGHT_ASSUMED_ONLY",
        "availability_hypothesis": study.FEATURE_RECIPE["availability"],
        "historical_first_versions_verified": False,
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

    from scripts import direction_1d_overnight_study as cli
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
