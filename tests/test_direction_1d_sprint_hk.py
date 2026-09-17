"""验证港股输入时点、学习器对照和实际提前预测的原始响应证据。"""

import json
from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as s

from test_direction_1d_sprint_sequence import (
    ancestor_ready,  # noqa: F401
    sample_rows,
)
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401


def rows():
    return [r | {"z": s.correction.vector(r["x"]) + [0.0] * 4} for r in sample_rows()]


class ConstantModel:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, x):
        return np.asarray([[1 - self.score, self.score]] * len(x))


def trained(name, score):
    size = 6 if name == s.CANDIDATES[0] else 12
    return {"model": ConstantModel(score), "mean": [0.0] * size, "scale": [1.0] * size}


def markets():
    return {
        code: {"2026-09-11": {"close": 100, "pre_close": 99}, "2026-09-14": {"close": 101, "pre_close": 100}}
        for code in s.hk_data.INDICES
    }


def test_hk_target_is_next_day_and_relative_returns_use_matching_cn_interval():
    x = [0.0] * 32
    x[18], x[23] = 0.01, 0.02
    assert s.hk_features(x, "2026-09-14", markets()) == pytest.approx([0, -1, 0, 0])
    with pytest.raises(ValueError, match="HK_MODEL_TARGET_NOT_ADJACENT"):
        s.vector(x, "2026-09-14", "2026-09-16", markets())


def test_future_hk_closes_never_enter_base_features():
    points = markets()
    x = [0.0] * 32
    before = s.hk_features(x, "2026-09-14", points)
    for rows in points.values():
        rows["2026-09-15"] = {"close": 1000000, "pre_close": 101}
    assert s.hk_features(x, "2026-09-14", points) == before
    for rows in points.values():
        del rows["2026-09-14"]
    assert s.hk_features(x, "2026-09-14", points) == pytest.approx([0, 0, 3 / 14, 3 / 14])
    with pytest.raises(ValueError, match="MISSING_OR_STALE"):
        s.hk_features(x, "2026-10-13", points)


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_unmatured_future_labels_and_inputs_do_not_change_fitted_model(name, monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    samples = rows()
    before = s.fit(samples, name, "2023-06-10")
    for r in samples:
        if r["mature"] >= "2023-06-10":
            r["z"], r["y"] = [float("nan")] * 12, 1 - r["y"]
    after = s.fit(samples, name, "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    assert before["mean"] == after["mean"] and before["scale"] == after["scale"]
    assert before["max_mature_date"] < "2023-06-10"
    probe = [np.zeros(len(before["mean"]))]
    np.testing.assert_array_equal(before["model"].predict_proba(probe), after["model"].predict_proba(probe))


def test_logistic_without_hk_variation_matches_original_recipe(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    samples = rows()
    current = s.fit(samples, s.CANDIDATES[0], "2023-06-10")
    control = previous.fit_lr(previous.selected(samples, "2023-06-10"))
    probe = samples[40]
    score = control["model"].predict_proba([probe["x"][30:32]])[0, 1]
    assert s.answer(probe["z"], s.CANDIDATES[0], current)["research_score"] == pytest.approx(score, abs=1e-10)


def test_correction_preserves_natural_error_prior(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    samples = rows()
    for i, r in enumerate(samples):
        baseline = int(r["z"][0] >= 0)
        r["y"] = 1 - baseline if i % 5 == 0 else baseline
    result = s.fit(samples, s.CANDIDATES[1], "2023-07-01")
    assert result["weighted_target_rate"] == pytest.approx(0.2)


@pytest.mark.parametrize("move", [-0.01, 0.0, 0.01])
def test_strict_direction_and_correction_thresholds_are_unchanged(move):
    z = [move, abs(move), 1, 0, 0, 0, 0, 0] + [0.0] * 4
    lr, extra = s.CANDIDATES
    assert s.answer(z, lr, trained(lr, 0.5))["prediction"] == 0
    assert s.answer(z, lr, trained(lr, 0.501))["prediction"] == 1
    baseline = int(move >= 0)
    assert s.answer(z, extra, trained(extra, 0.55))["prediction"] == baseline
    assert s.answer(z, extra, trained(extra, 0.551))["prediction"] == 1 - baseline


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_batch_and_single_predictions_match_and_bad_scores_fail(name):
    values = [r["z"] for r in rows()[:3]]
    model = trained(name, 0.6)
    assert s.batch_answers(values, name, model) == [s.answer(z, name, model) for z in values]
    with pytest.raises(ValueError, match="HK_MODEL_SCORE_INVALID"):
        s.answer(values[0], name, trained(name, float("nan")))


@pytest.fixture
def ready(sequence_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": trained(n, 0.8 if n == s.CANDIDATES[0] else 0.3)} for n in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    input_value = {"at": "2026-09-15T07:15:00+08:00", "base": "2026-09-14", "target": "2026-09-15", "rows": markets()}
    monkeypatch.setattr(s.hk_live, "capture", lambda at: input_value)
    monkeypatch.setattr(s.hk_live, "load", lambda target: input_value)
    return sequence_ready


def test_new_version_does_not_backfill_old_target(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("old target must not load calendar model"))
    assert s.tick()["verified_forecasts"] == 0


def test_true_predictions_are_immutable_and_outcomes_match(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    s.tick()
    assert path.read_bytes() == before
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert s.report()["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"] == 1


def test_late_answer_readback_is_invalid(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    save = b.save

    def crossing(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    result = s.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1


def test_changed_hk_feature_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][-1] = 1
    b.save(path, value, replace=True)
    receipt_path = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt_path, b.read(receipt_path) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_17_VECTOR_CHANGED"):
        s.report()


def test_calendar_revision_invalidates_frozen_model(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    b.save(s.root() / "result.json", {"fingerprint": {}, "calendar_hash": "previous-calendar"})
    monkeypatch.setattr(s, "fingerprint", lambda: {})
    with pytest.raises(ValueError, match="ROUND_17_MODEL_OR_CODE_CHANGED"):
        s.models()


def test_prior_runner_failure_still_attempts_hk_branch(monkeypatch):
    from scripts import direction_1d_sprint_hk as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.hk_model, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]


@pytest.fixture
def live_source(sequence_ready, monkeypatch):  # noqa: F811
    calls = []

    def fetch(code, start, end):
        calls.append((code, start, end))
        return json.dumps(
            {
                "code": 0,
                "data": {
                    "fields": s.hk_data.FIELDS,
                    "items": [[code, "20260911", 100, 99], [code, "20260914", 101, 100]],
                },
            }
        ).encode()

    monkeypatch.setattr(s.hk_live, "fetch_hk", fetch)
    monkeypatch.setattr(s.hk_live.sleep_time, "sleep", lambda seconds: None)
    monkeypatch.setattr(s.hk_live.overnight, "source", lambda: {"source_id": "source", "rate_limit_per_minute": 200})
    return calls


def test_actual_input_reuses_valid_raw_capture_and_rejects_modified_parsed_rows(live_source):
    first = s.hk_live.capture(b.now())
    assert len(live_source) == 2
    assert all(str(end) == "2026-09-14" for _, _, end in live_source)
    assert s.hk_live.capture(b.now()) == first and len(live_source) == 2
    first["rows"]["HSI"]["2026-09-14"]["close"] = 500
    b.save(s.hk_live.root() / "2026-09-15/input.json", first, replace=True)
    with pytest.raises(ValueError, match="PARSED_ROWS_CHANGED"):
        s.hk_live.load("2026-09-15")


def test_failed_provider_requests_stop_at_six_and_never_repeat_same_slot(live_source, monkeypatch):
    calls = []

    def fail(*args):
        calls.append(1)
        raise ValueError("HK_TEST_PROVIDER_FAILED")

    monkeypatch.setattr(s.hk_live, "fetch_hk", fail)
    for hour, minute in ((7, 15), (7, 45), (8, 15)):
        now = datetime(2026, 9, 15, hour, minute, tzinfo=b.ZONE)
        monkeypatch.setattr(b, "now", lambda at=now: at)
        for _ in range(2):
            with pytest.raises(ValueError, match="HK_LIVE_INPUT_INCOMPLETE"):
                s.hk_live.capture(b.now())
    assert len(calls) == 6
    now = datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE)
    monkeypatch.setattr(b, "now", lambda: now)
    with pytest.raises(ValueError, match="HK_LIVE_WINDOW_CLOSED"):
        s.hk_live.capture(b.now())
    assert len(calls) == 6


def test_response_crossing_deadline_cannot_create_complete_input(live_source, monkeypatch):
    fetch = s.hk_live.fetch_hk

    def late(*args):
        body = fetch(*args)
        monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))
        return body

    monkeypatch.setattr(s.hk_live, "fetch_hk", late)
    with pytest.raises(ValueError, match="HK_LIVE_DEADLINE_REACHED"):
        s.hk_live.capture(b.now())
    assert not (s.hk_live.root() / "2026-09-15/input.json").exists()
