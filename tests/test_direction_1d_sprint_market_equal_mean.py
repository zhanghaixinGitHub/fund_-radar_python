"""固定均值边界、成员身份、输入隔离和提前预测回读；无额外研究拟合。"""

from copy import deepcopy

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_equal_mean as s

from test_direction_1d_sprint_market_fxi_interval import head, paired
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def score(p):
    return {"prediction": int(p >= 0.5), "research_score": p, "kind": "UNCALIBRATED_UP_SCORE", "route": "MARKET_LR3"}


@pytest.mark.parametrize("a,z,expected", [(0.0, 1.0, 0.5), (0.2, 0.4, 0.3), (0.9, 0.7, 0.8), (0.5, 0.5, 0.5)])
def test_equal_weights_and_threshold_are_fixed(a, z, expected):
    answer = s.blend(score(a), score(z))
    assert answer["research_score"] == pytest.approx(expected)
    assert answer["prediction"] == int(expected >= 0.5)
    assert answer["kind"] == "UNCALIBRATED_UP_SCORE"


@pytest.mark.parametrize(
    "fault", ["nan", "inf", "negative", "above_one", "bool", "wrong_kind", "wrong_route", "wrong_prediction", "missing"]
)
def test_invalid_member_cannot_be_silently_dropped(fault):
    a = score(0.3)
    if fault in ("nan", "inf", "negative", "above_one", "bool"):
        a["research_score"] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "negative": -0.1,
            "above_one": 1.1,
            "bool": True,
        }[fault]
    elif fault == "wrong_kind":
        a["kind"] = "ERROR_PROBABILITY"
    elif fault == "wrong_route":
        a["route"] = "SOME_OTHER_MODEL"
    elif fault == "wrong_prediction":
        a["prediction"] = 1
    else:
        del a["research_score"]
    with pytest.raises(ValueError):
        s.blend(a, score(0.8))


def test_only_matching_complete_fallback_is_allowed():
    fallback = {"prediction": 0, "kind": "FIXED_DIRECTION", "route": "SPX_SIGN_FALLBACK"}
    assert s.blend(fallback, fallback) == fallback
    with pytest.raises(ValueError):
        s.blend(fallback, fallback | {"prediction": 1})
    with pytest.raises(ValueError):
        s.blend(fallback, score(0.3))
    with pytest.raises(ValueError):
        s.blend(fallback | {"route": "ALWAYS_UP"}, fallback | {"route": "ALWAYS_UP"})


def test_members_use_interval_and_fixed_rules_keep_original_input():
    v = paired()
    out = s.batch([v], [head(False), head(True)])
    assert set(out) == set(s.BRANCHES)
    assert out["MARKET_MAJORITY3"][0]["prediction"] == 0
    assert out[s.CANDIDATES[0]][0]["prediction"] == 1
    for i, name in enumerate(s.CONTROLS[:2]):
        assert out[name] == s.original.batch_answers([v], s.original.CANDIDATES[i], head(bool(i)))
    assert s.blend(out[s.CONTROLS[0]][0], out[s.CONTROLS[1]][0]) == out[s.CANDIDATES[0]][0]
    missing = s.batch([paired(False)], [None, None])
    assert all(v[0]["prediction"] == (1 if n == "ALWAYS_UP" else 0) for n, v in missing.items())


def test_frozen_proposal_rejects_weight_or_threshold_search():
    p = {
        "candidates": list(s.CANDIDATES),
        "controls": list(s.CONTROLS),
        "weights": [0.5, 0.5],
        "threshold": 0.5,
        "max_development_fits": 0,
        "max_current_fits": 0,
        "reproductions": 1,
        "branch_count": 6,
        "preflight_branch_checks": 180,
        "current_cutoff": "2026-09-16",
        "years": [2024, 2025],
        "expected_questions": {"2024": 7260, "2025": 7290},
        "expected_dates": {"2024": 242, "2025": 243},
    }
    s.validate_spec(p)
    for change in ({"weights": [0.3, 0.7]}, {"threshold": 0.55}, {"max_development_fits": 1}):
        with pytest.raises(ValueError):
            s.validate_spec(p | change)


@pytest.fixture
def model_manifest(parent_ready, monkeypatch):  # noqa: F811
    p = {"current_fit_cutoff": "2026-09-16", "model_format": "TEST_FORMAT"}
    source = {"model_sha256": "R82_MEMBERS_HASH"}
    bundle = {n: {"CN_EQUITY": head(bool(i)) | {"cutoff": "2026-09-16"}} for i, n in enumerate(s.original.CANDIDATES)}
    monkeypatch.setattr(s, "plan", lambda: p)
    monkeypatch.setattr(s, "fingerprint", lambda: {"test": "fingerprint"})
    monkeypatch.setattr(s.original, "models", lambda: (source, bundle))
    monkeypatch.setattr(s.data, "scope", lambda: [{"group": "CN_EQUITY"}])
    b.save(s.root() / "models-manifest.json", s.manifest_value(p, source))
    import hashlib

    b.save(
        s.root() / "result.json",
        {
            "plan_hash": b.digest(p),
            "fingerprint": s.fingerprint(),
            "model_sha256": hashlib.sha256((s.root() / "models-manifest.json").read_bytes()).hexdigest(),
        },
    )
    return p, source, bundle


@pytest.mark.parametrize("fault", ["manifest_weights", "source_hash", "current_cutoff", "group_missing"])
def test_current_manifest_or_member_replacement_is_rejected(model_manifest, fault):
    _, source, bundle = model_manifest
    s.models()
    if fault == "manifest_weights":
        path = s.root() / "models-manifest.json"
        b.save(path, b.read(path) | {"weights": [1, 0]}, replace=True)
    elif fault == "source_hash":
        source["model_sha256"] = "CHANGED"
    elif fault == "current_cutoff":
        bundle[s.original.CANDIDATES[0]]["CN_EQUITY"]["cutoff"] = "2026-09-15"
    else:
        bundle[s.original.CANDIDATES[0]].clear()
    with pytest.raises(ValueError):
        s.models()


def test_future_six_branches_do_not_require_volatility_context(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T06:50:00+08:00", "plan_hash": "MEAN2", "model_sha256": "ENSEMBLE_MANIFEST"}
    bundle = {n: {"CN_EQUITY": head(bool(i))} for i, n in enumerate(s.original.CANDIDATES)}
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle))
    monkeypatch.setattr(s, "live_market", lambda source: paired())
    assert runtime.tick(s)["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-16/001000.json"
    saved = b.read(path)
    assert "volatility_context" not in saved["market"] and len(saved["answers"]) == 6
    original = path.read_bytes()
    runtime.tick(s)
    assert original == path.read_bytes()
    receipt = s.root() / "receipts/2026-09-16/001000.json"
    b.save(receipt, b.read(receipt) | {"readback_at": "2026-09-16T08:30:00+08:00"}, replace=True)
    assert runtime.report(s)["verified_forecasts"] == 0
    from scripts import direction_1d_sprint_market_equal_mean as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_volatility_regime"


def test_inputs_are_not_mutated_when_members_are_combined():
    a, z = score(0.2), score(0.8)
    previous = deepcopy((a, z))
    s.blend(a, z)
    assert (a, z) == previous
