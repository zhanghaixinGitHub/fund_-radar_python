"""四组算法边界、树节点重放、原线性对照和训练/考试隔离。"""

import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
from app.services import direction_algorithm_models as models
from app.services import direction_algorithm_runner as runner
from app.services import direction_linear_models as linear
from app.services.direction_linear_protocol import (
    ALGORITHM_BRANCHES,
    ALGORITHM_VERSION,
    COVERAGE_VERSION,
    VOLUME_VERSION,
    planned_dates,
    specification_for,
    study_windows,
)
from app.services.direction_training_artifacts import digest, write_json
from app.services.direction_training_process import run_process

from test_direction_market import market_job  # noqa: F401


@pytest.fixture
def job(market_job):  # noqa: F811
    market_job["version"] = ALGORITHM_VERSION
    return market_job


def amount(job):
    for row in job["fit"]:
        row["input"]["x"].append(1 + row["input"]["x"][0] / 10)
    for row in job["exam"]:
        row["x"].append(1.1)
    return job


@pytest.mark.parametrize("branch", ALGORITHM_BRANCHES)
def test_fixed_models_and_data_only_inference(job, branch):
    job["branch"] = branch
    if branch in ("AMOUNT_ACTIVITY", "TREE_AMOUNT"):
        amount(job)
    output = models.execute_job(job)
    _, exam = linear.validate_job(job)
    assert output["status"] == "PREDICTED"
    assert models.predict_model(output["models"]["POOLED"], exam) == output["scores"]
    assert output["model_fit_count"] == 1
    if branch.startswith("TREE"):
        tree = output["models"]["POOLED"]
        assert len(tree["trees"]) == 100
        with pytest.raises(ValueError, match="ALGORITHM_BRANCH"):
            linear.execute_job(job)
    else:
        previous = deepcopy(job)
        previous["version"] = VOLUME_VERSION
        old = linear.execute_job(previous)
        assert old["scores"] == output["scores"]
        for field in runner.NUMERIC_FIELDS:
            assert old["models"]["POOLED"][field] == output["models"]["POOLED"][field]


def synthetic_tree():
    model = {
        "format": "NUMERIC_HIST_TREE_JSON_V1",
        "version": ALGORITHM_VERSION,
        "branch": "TREE_NAV",
        "fund": "POOLED",
        "parameters": dict(models.TREE),
        "dimensions": 7,
        "baseline": 0.0,
        "trees": [[{"feature": 0, "threshold": 0.0, "left": 1, "right": 2}, {"value": -1.0}, {"value": 1.0}]]
        + [[{"value": 0.0}] for _ in range(99)],
        "train_hash": "a" * 64,
        "train_counts": {f: 252 for f in models.FUNDS},
        "fit_end": "2022-06-30",
    }
    model["hash"] = digest(model)
    return model


def test_numeric_threshold_equality_goes_left():
    items = [SimpleNamespace(fund="001632", x=(x, 0, 0, 0, 0, 0, 0)) for x in (-1.0, 0.0, 1e-15)]
    scores = models.predict_model(synthetic_tree(), items)
    assert scores[0] == scores[1] < 0.5 < scores[2]


@pytest.mark.parametrize(
    "mutation", ["cycle", "child", "feature", "extra", "unreachable", "depth", "parameter", "hash", "nonfinite"]
)
def test_corrupt_tree_rejected_even_with_recomputed_digest(mutation):
    model = synthetic_tree()
    node = model["trees"][0][0]
    if mutation == "cycle":
        node["left"] = 0
    elif mutation == "child":
        node["right"] = 50
    elif mutation == "feature":
        node["feature"] = 7
    elif mutation == "extra":
        node["executable"] = "not allowed"
    elif mutation == "unreachable":
        model["trees"][0].append({"value": 0.0})
    elif mutation == "depth":
        model["trees"][0] = [{"feature": 0, "threshold": 0.0, "left": i + 1, "right": 5 + i} for i in range(4)] + [
            {"value": 0.0} for _ in range(5)
        ]
    elif mutation == "parameter":
        model["parameters"]["max_depth"] = 4
    elif mutation == "nonfinite":
        model["baseline"] = float("nan")
    if mutation not in ("hash", "nonfinite"):
        model["hash"] = digest({k: v for k, v in model.items() if k != "hash"})
    elif mutation == "hash":
        model["hash"] = "0" * 64
    with pytest.raises(ValueError):
        models.restore_tree(model)


@pytest.mark.parametrize("mutation", ["exam_answer", "mature_label", "dimension", "anchor"])
def test_worker_rejects_leakage_and_input_mismatch(job, mutation):
    job["branch"] = "TREE_NAV"
    if mutation == "exam_answer":
        job["exam"][0]["y"] = 1
    elif mutation == "mature_label":
        job["fit"][0]["answer"]["available_at"] = "2023-01-03"
    elif mutation == "dimension":
        job["exam"][0]["x"].append(1.0)
    else:
        job["exam"][0]["anchor"] = "2022-12-29"
    with pytest.raises(ValueError):
        models.execute_job(job)


def test_same_windows_and_no_generic_entrypoint():
    assert study_windows(ALGORITHM_VERSION) == study_windows(COVERAGE_VERSION)
    assert sum(len(planned_dates(ALGORITHM_VERSION, w)) for w in study_windows(ALGORITHM_VERSION)) == 463
    with pytest.raises(ValueError, match="DEDICATED_FREEZE"):
        specification_for(ALGORITHM_VERSION)


def test_worker_under_real_process_limits(job):
    job["branch"] = "TREE_NAV"
    result = run_process(
        job, seconds=30, memory_bytes=4 * 1024**3, command=[sys.executable, "-m", "scripts.direction_algorithm_worker"]
    )
    assert result["status"] == "PREDICTED"


def test_replay_cannot_repeat_or_recurse(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "verify", lambda _: {})
    monkeypatch.setattr(runner, "load_plan", lambda _: ({}, {}))
    write_json(tmp_path / "algorithm-replay-origin.json", {})
    with pytest.raises(ValueError, match="REPLAY_OF_REPLAY"):
        runner.replay(tmp_path)


def test_four_arm_specification_preserves_gates():
    # 使用随仓库提供的旧规范构造协议，不依赖本机研究包。
    from app.services.direction_linear_protocol import specification_v1

    old = specification_v1()
    old.update(mapping_hash="a" * 64, activity_formula="fixed", volume_information_rule="previous_session")
    plan = runner.specification(old)
    for key in ("minimum", "selection", "bootstrap", "logistic", "independent", "threshold"):
        assert plan[key] == old[key]
    assert plan["fit_budget"] == {"main_maximum": 32, "replay_maximum": 32}
    assert plan["network_calls"] == plan["database_writes"] == 0
    assert plan["pairs"] == {"B_vs_A": (1, 0), "C_vs_A": (2, 0), "D_vs_B": (3, 1), "D_vs_C": (3, 2)}
    assert [len(v) for v in plan["feature_indices"].values()] == [7, 8, 7, 8]


def test_algorithm_and_activity_comparisons_are_separate(monkeypatch, tmp_path):
    from app.services.direction_linear_protocol import specification_v1

    protocol = specification_v1()
    protocol["version"] = ALGORITHM_VERSION
    protocol["windows"] = study_windows(ALGORITHM_VERSION)[:1]
    window = protocol["windows"][0]
    predictions, answers = [], []
    for index, day in enumerate(planned_dates(ALGORITHM_VERSION, window)[:20]):
        for fund in models.FUNDS:
            key, y = f"{fund}:{day}", index % 2
            answers.append(
                {"window": window["name"], "sample_key": key, "answer": {"y": y, "available_at": window["label_asof"]}}
            )
            for branch in ALGORITHM_BRANCHES:
                score = float(y) if branch.startswith("TREE") else 0.0
                predictions.append(
                    {
                        "window": window["name"],
                        "sample_key": key,
                        "branch": branch,
                        "fund": fund,
                        "score": score,
                        "predicted_up": int(score > 0.5),
                    }
                )
    monkeypatch.setattr(runner, "read_json", lambda _: {"jobs": {}} if "prepared" in _.name else {})
    result = runner.diagnostics(tmp_path, protocol, predictions, answers)
    assert result["same_questions"] == 60
    for name, delta in (("B_vs_A", 0.0), ("C_vs_A", 0.5), ("D_vs_B", 0.5), ("D_vs_C", 0.0)):
        assert result["comparisons"][name]["full_accuracy_delta"] == delta
        assert result["comparisons"][name]["time_blocks"]["interval"] == [delta, delta]
