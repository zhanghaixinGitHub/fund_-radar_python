"""第二轮的时序、预算及数据保护测试；合成样本不计入真实研究拟合账本。"""

from copy import deepcopy

import numpy as np
import pytest
from app.services import target_fund_diagnostics as diag
from app.services import target_fund_experiments as exp
from app.services import target_fund_optimization as run
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read, save


@pytest.fixture
def data():
    calendar = [str(d) for d in exp.package.own.sessions()[0]]
    training_days = [d for d in calendar if "2021-01-01" <= d <= "2022-12-31"][:330]
    historical = [d for d in calendar if d.startswith("2023")]
    development = [d for d in calendar if d.startswith("2024")][:12]

    def row(day, index, peer=False):
        direction = exp.CLASSES[index % 3]
        x = [(index + j) % 11 / 100 for j in range(20)]
        x[13] = 0.5
        return {
            "fund_code": "006038" if peer else "002112",
            "family": "PEER" if peer else "TARGET",
            "target": day,
            "base": calendar[calendar.index(day) - 1],
            "t": calendar[calendar.index(day) - 1],
            "mature_at": day + "T20:00:00+08:00",
            "x": x,
            "actual_direction": direction,
        }

    train = [row(d, i) for i, d in enumerate(training_days + historical)]
    train += [row(d, i, True) for i, d in enumerate(training_days + historical)]
    train.sort(key=lambda r: (r["target"], r["fund_code"]))
    return {"train": train, "development": [row(d, i) for i, d in enumerate(development)]}


def test_folds_share_calendar_cut_and_exclude_immature_not_shuffle(data):
    rows = exp.build_folds(data)
    for fold in rows:
        assert fold["eligible"]
        assert all(
            r["target"] < fold["start"] and r["mature_at"] <= fold["start"] + "T08:00:00+08:00" for r in fold["train"]
        )
        assert all(fold["start"] <= r["target"] <= fold["end"] and r["fund_code"] == "002112" for r in fold["exam"])
    data["train"][0]["mature_at"] = "2023-04-02T08:00:00+08:00"
    assert data["train"][0] not in exp.build_folds(data)[1]["train"]


def test_insufficient_flat_fold_skipped_before_fits(data):
    data["train"] = [r for r in data["train"] if r["actual_direction"] != "FLAT" or r["target"].startswith("2023")]
    assert not exp.build_folds(data)[0]["eligible"]
    fold = exp.build_folds(data)[0]
    with pytest.raises(ValueError, match="TRAINING_GATE"):
        exp.fit(fold["train"], fold["exam"], "C20", fold["name"])


def test_weights_target_half_preserve_total_and_class_weights(data):
    rows = exp.build_folds(data)[1]["train"]
    # 参照再增加一个不同家族，目标总权重仍为一半，另两个家族各四分之一。
    rows += [{**r, "fund_code": "002170", "family": "PEER2"} for r in rows if r["family"] == "PEER"]
    w = exp.sample_weights(rows, "W20")
    totals = {
        family: sum(v for r, v in zip(rows, w, strict=True) if r["family"] == family)
        for family in {r["family"] for r in rows}
    }
    assert np.isclose(sum(w), len(rows))
    assert np.isclose(totals["TARGET"], len(rows) / 2)
    assert np.isclose(totals["PEER"], len(rows) / 4)
    assert exp.sample_weights(rows, "A7").tolist() == exp.sample_weights(rows, "C20").tolist()
    assert exp.sample_weights(rows, "I24").tolist() == exp.sample_weights(rows, "C20").tolist()
    assert len({v for r, v in zip(rows, w, strict=True) if r["family"] == "TARGET"}) == 1


def test_interaction_units_shape_and_finite(data):
    row = data["train"][0]
    before = deepcopy(row)
    assert exp.vector(row, "I24") == row["x"] + [row["x"][13] * v for v in row["x"][16:20]]
    assert row == before and len(exp.names("I24")) == 24
    for value in (None, float("nan"), float("inf"), "0", True):
        changed = deepcopy(row)
        changed["x"][0] = value
        with pytest.raises(ValueError, match="NONFINITE_OR_SHAPE"):
            exp.vector(changed, "I24")


def test_numerical_replay_training_only_scaling_and_restore(data):
    fold = exp.build_folds(data)[1]
    for r in fold["exam"]:
        r["x"][0] += 500
    first = exp.fit(fold["train"], fold["exam"], "I24", fold["name"])
    second = exp.fit(fold["train"], fold["exam"], "I24", fold["name"])
    assert first == second and first["restore"]["exam"]["max_score_difference"] <= 1e-12
    mean = np.average(
        [exp.vector(r, "I24") for r in fold["train"]], axis=0, weights=exp.sample_weights(fold["train"], "I24")
    )
    assert np.allclose(first["model"]["mean"], mean)
    bad = deepcopy(first["model"])
    bad["features"].reverse()
    with pytest.raises(ValueError, match="MODEL_CONTRACT"):
        exp.predict(bad, fold["exam"][0])
    bad = deepcopy(first["model"])
    bad["scale"][0] = 0
    with pytest.raises(ValueError, match="MODEL_VALUE"):
        exp.predict(bad, fold["exam"][0])


def test_future_rows_cannot_enter_fit(data):
    fold = exp.build_folds(data)[1]
    fold["train"][0]["mature_at"] = "2025-01-01T08:00:00+08:00"
    with pytest.raises(ValueError, match="FUTURE_TRAINING_ROW"):
        exp.fit(fold["train"], fold["exam"], "C20", fold["name"])


def fake_fit(train, exam, config, name):
    size = exp.CONFIGS[config]
    return {
        "model": {
            "protocol": exp.PROTOCOL,
            "fund_code": "002112",
            "config": config,
            "stage": name,
            "features": exp.names(config),
            "recipe": exp.RECIPE,
            "class_order": list(exp.CLASSES),
            "tie_order": list(exp.TIE_ORDER),
            "mean": [0.0] * size,
            "scale": [1.0] * size,
            "coef": [[0.0] * size for _ in range(3)],
            "intercept": [0.0] * 3,
            "train_hash": digest(train),
            "weight_hash": digest(exp.sample_weights(train, config).tolist()),
        },
        "restore": {},
        "iterations": [1],
    }


def stage_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "STORE", tmp_path)
    save(tmp_path / "protocol.json", {"test": True})
    state = {"events": [], "fits": 0, "completed_stages": [], "failures": {}}
    monkeypatch.setattr(exp, "fit", fake_fit)
    return state


def test_stage_replay_reuse_and_hard_interruption(data, tmp_path, monkeypatch):
    state = stage_setup(tmp_path, monkeypatch)
    fold = exp.build_folds(data)[1]
    calls = []

    def tracked(*args):
        calls.append(args[2:])
        return fake_fit(*args)

    monkeypatch.setattr(exp, "fit", tracked)
    monkeypatch.setattr("os._exit", lambda c: (_ for _ in ()).throw(SystemExit(c)))
    with pytest.raises(SystemExit):
        run.stage(state, fold["train"], fold["exam"], fold["name"], "A7", "2023Q2-A7-main")
    assert len(calls) == 1
    a = run.stage(state, fold["train"], fold["exam"], fold["name"], "A7")
    b = run.stage(state, fold["train"], fold["exam"], fold["name"], "A7")
    assert a == b and len(calls) == 2 and len(list((tmp_path / "attempts").glob("*.json"))) == 2


def test_budget_and_failed_attempt_not_retried(data, tmp_path, monkeypatch):
    state = stage_setup(tmp_path, monkeypatch)
    fold = exp.build_folds(data)[1]

    def fail(*args):
        raise ValueError("OPT_SYNTHETIC_FAILURE")

    monkeypatch.setattr(exp, "fit", fail)
    for _ in range(2):
        assert run.attempt_stage(state, fold["train"], fold["exam"], fold["name"], "C20") is None
    assert state["fits"] == 1 and state["failures"]["2023Q2-C20"] == "OPT_SYNTHETIC_FAILURE"
    for i in range(27):
        save(tmp_path / f"attempts/used-{i}.json", {})
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        run.stage(state, fold["train"], fold["exam"], fold["name"], "I24")


def score(correct, up=10, flat=0, down=10):
    return {"correct": correct, "class_correct": {"UP": up, "FLAT": flat, "DOWN": down}}


def test_selection_before_2024_no_fallback_and_class_regression():
    historical = {
        "models": {"A7": score(20), "C20": score(20), "W20": score(24, 12, 0, 12), "I24": score(25, 13, 0, 12)},
        "constants": {"DOWN": score(22)},
    }
    decision = exp.select_historical(historical, [historical] * 3)
    assert decision["selected"] == "I24" and not decision["2024_used_for_selection"]
    development = deepcopy(historical)
    development["models"]["I24"] = score(19)
    development["models"]["W20"] = score(30, 15, 0, 15)
    assert exp.final_gate(decision, development)["selected"] is None
    historical["models"]["I24"] = score(25, 20, 0, 5)
    assert exp.select_historical(historical, [historical] * 3)["selected"] == "W20"
    historical["models"].pop("A7")
    assert exp.select_historical(historical, [historical] * 3)["selected"] is None


def test_comparison_rejects_different_dates_and_null_class(data):
    model = fake_fit(data["train"], data["development"], "C20", "FULL")["model"]
    rows = exp.predict_rows(model, data["development"])
    comparison = exp.compare({"C20": rows, "W20": rows})
    assert comparison["models"]["C20"]["correct"] == 4
    with pytest.raises(ValueError, match="DIFFERENT_ROWS"):
        exp.compare({"C20": rows, "W20": rows[::-1]})
    subset = [r for r in rows if r["actual_direction"] != "FLAT"]
    assert exp.compare({"C20": subset})["models"]["C20"]["recall"]["FLAT"] is None
    assert comparison["differences"]["W20_vs_C20"]["counts"]["candidate_only"] == 0
    assert len(comparison["daily"]) == len(rows)
    assert sum(m["C20"]["correct"] for m in comparison["quarters"].values()) == 4


def test_complete_run_freezes_selection_reuses_all_fits_and_checks_source(data, tmp_path, monkeypatch):
    from contextlib import nullcontext

    # 让 Q1 持平只覆盖 29 日，Q2 起恢复合格；验证真实三段预算流程，数值拟合使用替身。
    flat_dates = sorted(
        {r["target"] for r in data["train"] if r["target"] < "2023" and r["actual_direction"] == "FLAT"}
    )
    for row in data["train"]:
        if row["target"] in flat_dates[29:] and row["actual_direction"] == "FLAT":
            row["actual_direction"] = "UP"
    monkeypatch.setattr(run, "STORE", tmp_path / "round")
    monkeypatch.setattr(run, "BASE", tmp_path / "baseline")
    monkeypatch.setattr(run.original, "load_frozen", lambda _: data)
    monkeypatch.setattr(run.original, "run_lock", nullcontext)
    monkeypatch.setattr(run, "code_manifest", lambda: {"source": "original"})
    monkeypatch.setattr(run, "baseline_files", lambda: {"model": "original"})
    monkeypatch.setattr(run, "invariant_snapshot", lambda: {"baseline_files": run.baseline_files(), "system": {}})
    monkeypatch.setattr(run.diagnostics, "diagnose", lambda *_: {"synthetic": True})
    monkeypatch.setattr(run.diagnostics, "report", lambda _: "synthetic diagnostics")
    monkeypatch.setattr(run, "observation_readiness", lambda *_: {"synthetic": True})
    save(run.BASE / "protocol.json", {"synthetic": True})
    for config, legacy in (("A7", "NAV7"), ("C20", "NAV7_HOLDINGS_MARKET")):
        model = fake_fit(data["train"], data["development"], config, "FULL")["model"]
        save(run.BASE / f"predictions/{legacy}.json", exp.predict_rows(model, data["development"]))
    calls = []

    def tracked(*args):
        if args[3] == "FULL":
            assert (run.STORE / "historical-decision.json").exists()
        calls.append(args[2:])
        return fake_fit(*args)

    monkeypatch.setattr(exp, "fit", tracked)
    result = run.run()
    assert result["status"] == "COMPLETED" and result["fits"] == len(calls) == 28
    assert result == run.run() and len(calls) == 28
    assert run.verify_report()["verified"] and len(calls) == 28
    monkeypatch.setattr(run, "baseline_files", lambda: {"model": "changed"})
    with pytest.raises(ValueError, match="BASELINE_ARTIFACT_CHANGED"):
        run.load()
    monkeypatch.setattr(run, "baseline_files", lambda: {"model": "original"})
    monkeypatch.setattr(run, "code_manifest", lambda: {"source": "changed"})
    with pytest.raises(ValueError, match="SOURCE_CODE_ENV_CHANGED"):
        run.load()


def test_diagnostic_margin_is_exact_and_not_causal(data):
    model = fake_fit(data["train"], data["development"], "C20", "FULL")["model"]
    model["coef"][2] = [0.1] * 20
    model["intercept"][2] = 0.2
    row = data["development"][0]
    result = diag.margin_parts(model, row["x"])
    p = exp.predict(model, row)["class_scores"]
    assert np.isclose(result["margin"], np.log(p["UP"] / p["DOWN"]))


def test_observation_does_not_open_protected_labels_or_rewrite_input(data, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    original_read = run.read
    save(tmp_path / "forward-inputs/2025-01-01.json", {"must_not_read": True})
    input_value = {
        "kind": "LIVE_INPUT_ONLY_NOT_A_PREDICTION",
        "window": {"target_nav_date": "2026-09-28"},
        "generated_at": "2026-09-24T20:00:00+08:00",
        "expires_at": "2027-09-01T20:00:00+08:00",
        "candidate_direction": None,
        "training_eligible": False,
    }
    save(tmp_path / "forward-inputs/2026-09-28.json", input_value)

    def protected(path):
        assert not path.name.startswith("2025")
        return original_read(path)

    monkeypatch.setattr(run, "read", protected)
    result = run.observation_readiness(data, {"selected": None})
    assert result["status"] == "NO_QUALIFIED_CANDIDATE" and result["new_forward_predictions"] == 0
    assert not result["existing_input_only"][0]["has_candidate_prediction"]
    assert read(tmp_path / "forward-inputs/2026-09-28.json") == input_value
