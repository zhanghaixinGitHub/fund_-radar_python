"""固定三输入规则、训练成熟边界和旧分支隔离；合成数据不会写研究模型。"""

from datetime import date, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint_market_intercept as s


def market(raw=(1.0, -0.5, 0.2), available=True):
    return {
        "features": list(raw),
        "available": available,
        "etf_available": available,
        "cnya_available": available,
        "us_sessions": 1.0,
    }


def history():
    rows = []
    for i in range(220):
        day = date(2024, 1, 1) + timedelta(days=i)
        rows.append(
            {
                "code": "001000",
                "family": "F",
                "group": "CN_EQUITY",
                "u": str(day),
                "mature": str(day + timedelta(days=1)),
                "y": int(i % 3 == 0),
                "market": market(((i % 7 - 3) / 10, (i % 5 - 2) / 10, (i % 9 - 4) / 10)),
            }
        )
    return rows


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_immature_and_unavailable_rows_cannot_change_fitted_model(monkeypatch, name):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = history()
    cutoff = rows[180]["u"]
    before = s.fit(rows, name, cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] ^= 1
            r["market"] = market((float("nan"), 0, 0))
    rows.append(rows[0] | {"market": market((999, 999, 999), False)})
    after = s.fit(rows, name, cutoff)
    assert before["fit_hash"] == after["fit_hash"] and before["max_mature_date"] < cutoff
    np.testing.assert_allclose(before["model"].coef_, after["model"].coef_)
    assert before["model"].fit_intercept is True and before["model"].n_features_in_ == 3


def test_delayed_t_or_u_announcement_is_label_maturity_not_input_requirement():
    rows = history()
    cutoff = rows[180]["u"]
    a, z = rows[10], rows[11]
    a["mature"] = z["mature"] = cutoff
    selected = s.training_rows(rows, cutoff)
    assert a not in selected and z not in selected and rows[12] in selected


@pytest.mark.parametrize("name", s.BRANCHES)
def test_all_models_share_spx_fallback_without_a_nav_or_hk_head(name):
    answers = s.batch_answers([market((-1, 0, 0), False), market((0, 0, 0), False)], name)
    assert [a["prediction"] for a in answers] == ([1, 1] if name == "ALWAYS_UP" else [0, 1])


def test_majority_sign_zero_and_scale_rules():
    assert s.batch_answers([market((-10, 0, 0)), market((10, -0.1, -0.1))], "MARKET_MAJORITY3")[0]["prediction"] == 1
    assert s.batch_answers([market((10, -0.1, -0.1))], "MARKET_MAJORITY3")[0]["prediction"] == 0
    np.testing.assert_array_equal(s.transformed(np.array([[0, -5, 4]]), [1, 2, 3], True), [[1, -1, 1]])
    with pytest.raises(ValueError, match="SCALE_INVALID"):
        s.transformed(np.zeros((1, 3)), [1, 0, 1], False)


@pytest.mark.parametrize("raw", [(float("nan"), 0, 0), (1, 2), (float("inf"), 0, 0)])
def test_invalid_market_numbers_fail_closed(raw):
    with pytest.raises(ValueError):
        s.batch_answers([market(raw)], "SPX_SIGN")


def test_new_entry_runs_independent_branch_even_when_old_branch_fails(monkeypatch):
    from scripts import direction_1d_sprint_market_intercept as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_only"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(s, "tick", lambda: calls.append(1))
    monkeypatch.setattr(s.base, "save", lambda *a, **k: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]


def test_interrupted_fit_is_not_silently_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(s.base, "ROOT", tmp_path)
    calls = []

    def broken(*args):
        calls.append(1)
        raise ValueError("FIT_FAILED")

    monkeypatch.setattr(s, "fit", broken)
    with pytest.raises(ValueError, match="FIT_FAILED"):
        s.fit_checkpoint(history(), s.CANDIDATES[0], "2025-01-01", "1-CN_EQUITY", "PLAN")
    with pytest.raises(ValueError, match="ALREADY_ATTEMPTED"):
        s.fit_checkpoint(history(), s.CANDIDATES[0], "2025-01-01", "1-CN_EQUITY", "PLAN")
    assert calls == [1]


def test_changed_model_bytes_rejected_before_deserialization(tmp_path, monkeypatch):
    monkeypatch.setattr(s.base, "ROOT", tmp_path)
    p = {"synthetic": True}
    monkeypatch.setattr(s, "plan", lambda: p)
    monkeypatch.setattr(s, "fingerprint", lambda: {"code": "TEST"})
    s.base.save(
        s.root() / "result.json",
        {"plan_hash": s.base.digest(p), "fingerprint": s.fingerprint(), "model_sha256": "ORIGINAL"},
    )
    (s.root() / "models.joblib").write_bytes(b"changed")
    monkeypatch.setattr(s.joblib, "load", lambda *args: pytest.fail("tampered model must not deserialize"))
    with pytest.raises(ValueError, match="MODEL_OR_PLAN_CHANGED"):
        s.models()


def test_intercept_is_learned_from_training_prior_without_moving_threshold(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = history()
    for i, r in enumerate(rows):
        r["market"] = market((0, 0, 0))
        r["y"] = int(i % 4 != 0)
    head = s.fit(rows, s.CANDIDATES[0], "2025-01-01")
    assert head["model"].intercept_[0] > 0.5
    score = s.batch_answers([market((0, 0, 0))], s.CANDIDATES[0], head)[0]
    assert 0.70 < score["research_score"] < 0.8 and score["prediction"] == 1
