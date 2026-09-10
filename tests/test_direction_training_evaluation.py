"""可手算指标、缺题覆盖、同步时间块和预测封存的执行顺序。"""

import math
from copy import deepcopy
from datetime import date

import pytest
from app.services import direction_training_runner as runner
from app.services.direction_training_artifacts import read_jsonl, read_seal, seal, write_json, write_jsonl
from app.services.direction_training_dataset import exam_dates
from app.services.direction_training_evaluation import (
    complete_blocks,
    evaluate,
    grouped_metrics,
    metrics,
    paired_interval,
)
from app.services.direction_training_protocol import BASELINES, CANDIDATES, specification


def test_hand_calculated_confusion_threshold_and_single_class_null():
    result = metrics([(0, 0.5), (0, 0.8), (1, 0.6), (1, 0.4)])
    assert result["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["accuracy"] == result["balanced_accuracy"] == 0.5
    assert result["brier_score"] == pytest.approx((0.25 + 0.64 + 0.16 + 0.36) / 4)
    assert metrics([(1, 1), (1, 0.9)])["balanced_accuracy"] is None
    assert metrics([]) is None
    with pytest.raises(ValueError):
        metrics([(0, float("nan"))])


def test_fund_macro_is_not_sample_weighted():
    rows = [{"fund": "a", "y": 1, "score": 1}] + [{"fund": "b", "y": 1, "score": 0}] * 9
    result = grouped_metrics(rows, ["a", "b"])
    assert result["equal_fund_macro"]["accuracy"] == 0.5
    assert result["sample_weighted"]["accuracy"] == 0.1


def fixture_scores():
    protocol = specification()
    predictions, answers, execution = [], [], {}
    for w in protocol["windows"]:
        name = w["name"]
        execution[name] = {"A": {"status": "PREDICTED"}}
        _, dates = exam_dates(date.fromisoformat(w["cal_end"]), date.fromisoformat(w["exam_end"]))
        for f in protocol["funds"]:
            for i, d in enumerate(dates):
                key = f"{f}:{d}"
                y = i % 2
                answers.append({"window": name, "sample_key": key, "answer": {"y": y}})
                for b in (*CANDIDATES, *BASELINES):
                    score = float(y if b == "A" else 1 - y)
                    predictions.append(
                        {
                            "window": name,
                            "sample_key": key,
                            "candidate": b,
                            "fund": f,
                            "cutoff": str(d),
                            "score": score,
                            "predicted_up": int(score > 0.5),
                            "status": "PREDICTED",
                        }
                    )
    return protocol, predictions, answers, execution


def test_paired_delta_sign_and_all_comparisons_and_failure_prevents_winner():
    protocol, predictions, answers, execution = fixture_scores()
    result = evaluate(protocol, predictions, answers, execution)
    assert result["candidate_status"]["A"] == "RESEARCH_CANDIDATE"
    assert result["comparisons"]["A"]["ALWAYS_UP"]["time_block_delta"]["interval"] == [1, 1]
    assert result["comparisons"]["B"]["A"]["accuracy_delta"] == -1
    assert len(result["comparisons"]["C"]) == 5
    missing = deepcopy(predictions)
    missing[0].update(score=None, predicted_up=None, status="FAILED")
    changed = evaluate(protocol, missing, answers, execution)
    assert changed["coverage"]["001632"]["planned"] == result["coverage"]["001632"]["planned"]
    assert changed["coverage"]["001632"]["common"] == result["coverage"]["001632"]["common"] - 1
    assert changed["research_status"] == "INSUFFICIENT_DATA"
    with pytest.raises(ValueError, match="PLANNED"):
        evaluate(protocol, predictions[1:], answers, execution)


def test_missing_session_does_not_compress_blocks_or_join_window_tails():
    protocol, predictions, answers, execution = fixture_scores()
    keys = {(r["window"], r["sample_key"]) for r in predictions}
    blocks, excluded = complete_blocks(protocol, keys)
    assert sum(len(b["dates"]) for b in blocks) == len(blocks) * 20
    assert excluded["DEV_2024_Q1_V2"]["tail_cutoffs_per_fund"] == 18
    first = blocks[0]
    keys.remove(first["keys"][0])
    remaining, after = complete_blocks(protocol, keys)
    assert len(remaining) == len(blocks) - 1 and first not in remaining
    assert after[first["window"]]["incomplete_blocks"] == 1
    left = {
        k: {
            "fund": k[1].split(":")[0],
            "correct": int(k[1].startswith("001632")),
            "y": 1,
            "score": float(k[1].startswith("001632")),
        }
        for b in blocks
        for k in b["keys"]
    }
    right = {
        k: {**v, "correct": int(k[1].startswith("006730")), "score": float(k[1].startswith("006730"))}
        for k, v in left.items()
    }
    interval = paired_interval(protocol, blocks, left, right)
    assert interval["interval"] == [0, 0]
    assert interval["per_fund"]["001632"]["interval"] == [1, 1]
    assert interval["per_fund"]["006730"]["interval"] == [-1, -1]


def test_log_loss_clipping_and_low_sample_bins_do_not_change_scores():
    pairs = [(1, 0.0), (0, 1.0)]
    result = metrics(pairs)
    assert result["log_loss"] == pytest.approx(-math.log(1e-8))
    assert result["log_loss_clip_count"] == 2
    assert all(b["low_sample"] for b in result["reliability"])
    assert result["brier_score"] == 1 and pairs == [(1, 0.0), (0, 1.0)]


def test_block_resampling_preserves_each_window_weight_and_brier_sign():
    protocol = {"funds": ["f"], "bootstrap": {"seed": 42, "repeats": 2000}}
    blocks = [{"window": w, "keys": [(w, "f:d")], "dates": ["d"]} for w in ("a", "b")]
    left = {(w, "f:d"): {"fund": "f", "correct": int(w == "a"), "y": 1, "score": float(w == "a")} for w in ("a", "b")}
    right = {k: {"fund": "f", "correct": 0, "y": 1, "score": 0.0} for k in left}
    result = paired_interval(protocol, blocks, left, right)
    assert result["interval"] == [0.5, 0.5]
    assert result["brier_interval"] == [-0.5, -0.5]


def test_score_cannot_read_answer_before_prediction_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_protocol", lambda folder: ({}, {}))

    def stage(folder, name, *args):
        if name == "predictions-manifest.json":
            raise FileNotFoundError(name)
        return {}

    def forbidden(*args, **kwargs):
        pytest.fail("answers accessed before predictions sealed")

    monkeypatch.setattr(runner, "load_stage", stage)
    monkeypatch.setattr(runner, "build_answer", forbidden)
    with pytest.raises(FileNotFoundError):
        runner.score(tmp_path)
    assert not (tmp_path / "answers.exam.jsonl").exists()


def test_verify_recomputes_without_source_queries_or_fitting_and_detects_wrong_metrics(tmp_path, monkeypatch):
    protocol, predictions, answers, execution = fixture_scores()
    # 使用缺失答案构造可验证的完整封存，确保verify不需重建来源标签。
    answers = [{**a, "answer": None} for a in answers]
    monkeypatch.setattr(runner, "load_protocol", lambda folder: (protocol, {}))
    monkeypatch.setattr(runner, "load_stage", lambda *args: read_seal(tmp_path, "scored-manifest.json"))

    def forbidden(*args, **kwargs):
        pytest.fail("verify must not query source or fit")

    monkeypatch.setattr(runner, "build_answer", forbidden)
    monkeypatch.setattr(runner, "run_process", forbidden)
    for filename, value in (("predictions.jsonl", predictions), ("answers.exam.jsonl", answers)):
        write_jsonl(tmp_path / filename, value)
    write_json(tmp_path / "execution.json", execution)
    result = evaluate(protocol, predictions, answers, execution)
    files = {"metrics.json": write_json(tmp_path / "metrics.json", result)}
    seal(tmp_path, "scored-manifest.json", files, status="SCORED")
    completed = runner.verify(tmp_path)
    assert completed["status"] == "VERIFIED" and not completed["model_refitted"]
    assert runner.verify(tmp_path) == completed
    assert read_jsonl(tmp_path / "predictions.jsonl") == predictions
