"""验证连续目标不泄露未来、回归分数不冒充概率，以及真实预测的时点和同题边界。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_return_target as r

from test_direction_1d_sprint_dual_us import parent_ready  # noqa: F401
from test_direction_1d_sprint_dual_us import ready as previous_ready  # noqa: F401
from test_direction_1d_sprint_fund_response import sample_rows


def test_continuous_target_preserves_sign_and_uses_known_volatility_floor():
    x = [0.0] * 32
    assert r.normalized_target("1", "1.00001", x) == pytest.approx(0.1)
    assert r.normalized_target("1", "1", x) == 0
    assert r.normalized_target("1", "2", x) == 5
    x[3] = 0.01
    assert r.normalized_target("1", "0.98", x) == pytest.approx(-2)
    x[3] = -1
    with pytest.raises(ValueError, match="TARGET_INPUT_INVALID"):
        r.normalized_target("1", "0.98", x)


def test_future_continuous_targets_cannot_change_fit():
    rows = [row | {"return_target": 1.0 if row["y"] else -1.0} for row in sample_rows()]
    first = r.fit(rows, "RIDGE2_RET504", "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["return_target"] = float("nan")
            row["x"] = [float("nan")] * 32
    second = r.fit(rows, "RIDGE2_RET504", "2023-06-10")
    assert first["fit_hash"] == second["fit_hash"]
    np.testing.assert_array_equal(first["model"][-1].coef_, second["model"][-1].coef_)


def test_regression_weights_preserve_product_family_influence():
    rows = sample_rows()
    duplicated = rows + [dict(row) for row in rows if row["code"] == "a"]
    w1, w2 = r.weights(rows), r.weights(duplicated)
    assert w1.sum() == pytest.approx(w2.sum())
    assert sum(w for row, w in zip(rows, w1, strict=True) if row["code"] == "a") == pytest.approx(
        sum(w for row, w in zip(duplicated, w2, strict=True) if row["code"] == "a")
    )


class ConstantRegression:
    def predict(self, x):
        return np.full(len(x), -0.25)


@pytest.fixture
def ready(previous_ready, monkeypatch):  # noqa: F811 - pytest按参数名注入父证据fixture。
    b.save(r.root() / "result.json", {"winner": "RIDGE2_RET504", "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": {"model": ConstantRegression()}} for n in r.CANDIDATES}
    monkeypatch.setattr(r, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return previous_ready


def test_saved_answers_are_immutable_and_outcomes_paired(ready, monkeypatch):
    directory, original, _ = ready
    assert r.tick()["verified_forecasts"] == 1
    path = r.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    choice = b.read(path)["answers"]["RIDGE2_RET504"]
    assert choice["prediction"] == 0 and choice["research_score"] == -0.25
    assert choice["kind"] == "NORMALIZED_RETURN_RESEARCH_SCORE"
    r.tick()
    assert before == path.read_bytes()
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 0, "actual_direction": "DOWN", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    metrics = r.report()["matched_forward_metrics"]
    assert metrics["RIDGE2_RET504"]["accuracy"] == 1
    assert metrics["RIDGE2_RET504"]["count"] == metrics["ORIGINAL7"]["count"]


def test_save_crossing_cutoff_is_counted_as_invalid(ready, monkeypatch):
    save = b.save

    def crossing(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == r.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    report = r.tick()
    assert report["verified_forecasts"] == 0 and report["invalid_or_late"] == 1


def test_post_cutoff_does_not_fit_or_generate_answers(ready, monkeypatch):
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))
    monkeypatch.setattr(r, "models", lambda: pytest.fail("unexpected model loading"))
    assert r.tick()["verified_forecasts"] == 0


def test_parent_change_after_receipt_is_rejected(ready):
    from app.services import direction_1d_sprint_fund_response as p

    path = p.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["source_hash"] = "tampered"
    b.save(path, value, replace=True)
    with pytest.raises(ValueError, match="ROUND_05_PARENT_NOT_VERIFIED"):
        r.tick()


def test_independent_branch_failure_still_runs_other_branch(monkeypatch):
    from scripts import direction_1d_sprint_return_target as entry

    for module in (entry.base, entry.market, entry.overnight, entry.sparse, entry.response):
        monkeypatch.setattr(module, "tick", lambda: {})
    monkeypatch.setattr(entry.base, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(entry.regression, "tick", lambda: (_ for _ in ()).throw(ValueError("ROUND_07_FAILURE")))
    calls = []
    monkeypatch.setattr(entry.dual, "tick", lambda: calls.append(1))
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
