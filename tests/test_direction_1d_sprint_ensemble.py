"""固定三模型纠错组合的规则、成员完整性与实际提前预测证据。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_ensemble as s

from test_direction_1d_sprint_hk import ConstantModel, markets
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401
from test_direction_1d_sprint_vix_data import body


def test_vix_appends_only_two_known_inputs_and_target_stays_one_day():
    x = [0.0] * 32
    points = s.vix_data.parse(body())
    z = s.vector(x, "2026-09-14", "2026-09-15", markets(), points)
    assert len(z) == 14 and z[:12] == s.hk.vector(x, "2026-09-14", "2026-09-15", markets())
    assert z[-2:] == s.vix_data.features("2026-09-14", "2026-09-15", points)
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        s.vector(x, "2026-09-14", "2026-09-16", markets(), points)


def trained(score):
    return {
        name: {"model": ConstantModel(score), "mean": [0.0] * size, "scale": [1.0] * size}
        for name, size, _ in s.MEMBERS
    }


def test_equal_mean_is_fixed_and_is_not_a_majority_vote():
    assert s.blend([0.1, 0.6, 0.6], 1)["flipped"] is False
    value = s.blend([0.2, 0.8, 0.8], 0)
    assert value["research_score"] == pytest.approx(0.6) and value["prediction"] == 1
    assert list(value["member_scores"]) == [name for name, _, _ in s.MEMBERS]


@pytest.mark.parametrize("scores", [[0.5, 0.6], [0.5] * 4, [float("nan"), 0.5, 0.6], [-0.1, 0.5, 0.6], [1.1, 0.5, 0.6]])
def test_missing_or_invalid_member_scores_fail(scores):
    with pytest.raises(ValueError, match="MEMBER_SCORES_INVALID"):
        s.blend(scores, 1)


def test_member_inputs_keep_their_original_width_and_batch_matches_single():
    values = [[0.01] + [0.0] * 13, [-0.01] + [0.0] * 13]
    models = trained(0.6)
    for name, size, _ in s.MEMBERS:

        class ShapeModel:
            def __init__(self, wanted):
                self.wanted = wanted

            def predict_proba(self, x):
                assert np.asarray(x).shape[1] == self.wanted
                return np.asarray([[0.4, 0.6]] * len(x))

        models[name]["model"] = ShapeModel(size)
    assert s.batch_answers(values, s.CANDIDATES[0], models) == [s.answer(z, s.CANDIDATES[0], models) for z in values]
    del models[s.MEMBERS[0][0]]
    with pytest.raises(KeyError):
        s.answer(values[0], s.CANDIDATES[0], models)


@pytest.mark.parametrize("changed", ["manifest", "member"])
def test_manifest_or_member_replacement_is_rejected(tmp_path, monkeypatch, changed):
    import hashlib
    from types import SimpleNamespace

    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: ([], "calendar"))
    monkeypatch.setattr(s, "fingerprint", lambda: {"code": "fixture"})
    members, specs, receipts = [], {}, []
    for name, size, _ in s.MEMBERS:
        receipt = {"model_sha256": name + "-hash"}
        bundle = {name: {"CN_EQUITY": trained(0.6)[name]}}
        receipts.append(receipt)
        members.append((name, size, SimpleNamespace(models=lambda r=receipt, v=bundle: (r, v))))
        specs[name] = {"result_hash": b.digest(receipt), "model_sha256": receipt["model_sha256"]}
    monkeypatch.setattr(s, "MEMBERS", tuple(members))
    path = s.root() / "ensemble-manifest.json"
    b.save(path, {"members": specs, "plan_hash": "plan"})
    b.save(
        s.root() / "result.json",
        {
            "fingerprint": s.fingerprint(),
            "calendar_hash": "calendar",
            "plan_hash": "plan",
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    assert "CN_EQUITY" in s.models()[1][s.CANDIDATES[0]]
    if changed == "manifest":
        path.write_bytes(path.read_bytes() + b" ")
        expected = "MANIFEST_OR_CODE_CHANGED"
    else:
        receipts[0]["model_sha256"] = "different-model"
        expected = "SOURCE_MODEL_CHANGED"
    with pytest.raises(ValueError, match=expected):
        s.models()


@pytest.mark.parametrize("move", [-0.01, 0, 0.01])
def test_correction_threshold_is_unchanged(move):
    z = [move] + [0.0] * 13
    baseline = int(move >= 0)
    assert s.answer(z, s.CANDIDATES[0], trained(0.55))["prediction"] == baseline
    assert s.answer(z, s.CANDIDATES[0], trained(0.551))["prediction"] == 1 - baseline


@pytest.fixture
def ready(hk_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {s.CANDIDATES[0]: {"CN_EQUITY": trained(0.3)}}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    input_value = {
        "at": "2026-09-15T07:15:00+08:00",
        "base": "2026-09-14",
        "target": "2026-09-15",
        "rows": s.vix_data.parse(body()),
        "source": s.vix_data.SOURCE,
    }
    monkeypatch.setattr(s.vix_data, "capture", lambda at: input_value)
    monkeypatch.setattr(s.vix_data, "load", lambda target: input_value)
    return hk_ready


def test_old_target_is_not_backfilled(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("old target model loading"))
    assert s.tick()["verified_forecasts"] == 0


def test_live_input_order_and_announcements_are_checked(ready):
    _, original, source = ready
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original | {"inputs": original["inputs"][::-1]}, markets(), s.vix_data.parse(body()))
    original["inputs"][0]["ann_date"] = "2026-09-16"
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original, markets(), s.vix_data.parse(body()))


def test_early_answers_are_immutable_and_outcomes_match(ready, monkeypatch):
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


def test_late_readback_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    save = b.save

    def late(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", late)
    result = s.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1


def test_changed_saved_vix_value_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][-1] += 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_23_VECTOR_CHANGED"):
        s.report()


def test_prior_failure_still_attempts_ensemble_branch(monkeypatch):
    from scripts import direction_1d_sprint_ensemble as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_vix"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.ensemble, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
