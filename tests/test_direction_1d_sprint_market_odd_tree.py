"""检验固定反对称变换、原头复用、成熟截止校验和八个未来分支。"""

import hashlib

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_odd_tree as s

from test_direction_1d_sprint_market_date_tree import models as original_models
from test_direction_1d_sprint_market_only import market
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


class BiasedTree:
    n_features_in_ = 3

    @property
    def estimators_(self):
        return [self]

    def predict(self, x):
        return 0.8 + 0.03 * np.tanh(np.asarray(x)[:, 0])


def test_fixed_rule_removes_constant_prior_and_has_odd_symmetry():
    model = BiasedTree()
    x = np.asarray([[2.0, -1, 3], [-0.2, 4, 1], [0, 0, 0]])
    scores = s.odd_scores(model, x)
    np.testing.assert_allclose(scores, 0.5 + 0.03 * np.tanh(x[:, 0]), atol=1e-15)
    np.testing.assert_allclose(scores + s.odd_scores(model, -x), 1, atol=1e-15)
    np.testing.assert_array_equal(scores, s.odd_scores(model, x))
    assert scores[2] == 0.5 and ((scores >= 0) & (scores <= 1)).all()


@pytest.mark.parametrize("bad", [np.ones((1, 2)), np.array([[float("nan"), 0, 0]]), np.ones(3)])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError, match="INPUT_INVALID"):
        s.odd_scores(BiasedTree(), bad)


def test_invalid_parent_scores_are_not_clipped_into_success(monkeypatch):
    monkeypatch.setattr(s.original, "tree_scores", lambda model, x: np.ones(len(x)) * 1.01)
    with pytest.raises(ValueError, match="PARENT_SCORE_INVALID"):
        s.odd_scores(BiasedTree(), [[0, 0, 0]])


def test_declaration_keeps_zero_new_fits_and_fixed_comparison_set():
    p = {
        "candidates": list(s.CANDIDATES),
        "controls": list(s.CONTROLS),
        "branches": 8,
        "preflight_checks": 240,
        "development_fits": 0,
        "current_fits": 0,
        "reproductions": 1,
        "current_cutoff": "2026-09-15",
        "first_target": "2026-09-16",
    }
    s.validate_spec(p)
    for key, value in (
        ("development_fits", 12),
        ("current_fits", 3),
        ("branches", 7),
        ("controls", list(s.CONTROLS[:-1])),
        ("first_target", "2026-09-15"),
    ):
        with pytest.raises(ValueError, match="DECLARED_SPEC_CHANGED"):
            s.validate_spec(p | {key: value})


@pytest.mark.parametrize("name", s.BRANCHES)
def test_same_missing_market_fallback_without_accessing_any_model(name):
    assert s.batch_answers([market((-1, 0, 0), False)], name)[0]["prediction"] == (1 if name == "ALWAYS_UP" else 0)


def test_control_mapping_and_candidate_native_heads_are_explicit():
    assert [s.original_name(n) for n in s.CANDIDATES] == list(s.original.CANDIDATES)
    assert [s.original_name(n) for n in s.CONTROLS] == list(s.original.CANDIDATES) + list(s.original.CONTROLS[1:])
    with pytest.raises(ValueError, match="BRANCH_INVALID"):
        s.original_name("UNKNOWN")


def test_wrong_checkpoint_cutoff_rejected_before_deserialization(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    native = s.original.CANDIDATES[0]
    path = s.original.root() / "checkpoints" / f"1-CN_EQUITY-{native}.joblib"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"NEVER_LOAD")
    b.save(
        path.with_suffix(".json"),
        {"name": native, "cutoff": "2025-04-01", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    monkeypatch.setattr(s.joblib, "load", lambda *a: pytest.fail("future-trained head must not load"))
    with pytest.raises(ValueError, match="CHECKPOINT_CHANGED"):
        s.historical_head(s.CANDIDATES[0], "CN_EQUITY", 1, "2025-01-01")


def test_rehashed_manifest_cannot_point_to_a_different_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    p = {"model_format": "ODD", "formula": "FIXED", "threshold": 0.5}
    source = {"model_sha256": "PARENT"}
    monkeypatch.setattr(s, "plan", lambda: p)
    monkeypatch.setattr(s, "fingerprint", lambda: {"code": "TEST"})
    monkeypatch.setattr(s.original, "models", lambda: (source, {}))
    bad = s.manifest_value(p, source) | {"source_model_sha256": "OTHER"}
    path = s.root() / "models-manifest.json"
    b.save(path, bad)
    b.save(
        s.root() / "result.json",
        {
            "plan_hash": b.digest(p),
            "fingerprint": s.fingerprint(),
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    with pytest.raises(ValueError, match="MANIFEST_CHANGED"):
        s.models()


def test_eight_future_branches_keep_original_market_parent(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T03:00:00+08:00", "plan_hash": "ODD", "model_sha256": "MANIFEST"}
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, original_models()))
    monkeypatch.setattr(s, "live_market", lambda source: market())
    assert runtime.tick(s)["verified_forecasts"] == 1
    value = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert set(value["answers"]) == set(s.BRANCHES)
    assert value["answers"][s.CANDIDATES[0]]["research_score"] == 0.5
    assert not (b.ROOT / "forward").exists()


def test_entry_exposes_prepare_and_keeps_frozen_previous_version():
    from scripts import direction_1d_sprint_market_odd_tree as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_date_tree"
    assert not hasattr(s, "fit") and not hasattr(s, "train")
