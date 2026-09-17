"""时间适应训练：未成熟标签隔离、份额权重、刷新日期与新版本预测截止边界。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_adaptive as a

from test_direction_1d_sprint_dual_us import parent_ready  # noqa: F401
from test_direction_1d_sprint_dual_us import ready as previous_ready  # noqa: F401
from test_direction_1d_sprint_fund_response import sample_rows


def test_period_start_is_fixed_before_the_exam_and_respects_year_boundary():
    assert a.anchor("2025-01-01", "WEEKLY_NAT252") == "2024-12-30"
    assert a.anchor("2025-01-31", "MONTHLY_BAL504") == "2025-01-01"
    assert a.anchor("2025-02-03", "WEEKLY_NAT252") == "2025-02-03"


@pytest.mark.parametrize("name", a.CANDIDATES)
def test_future_features_and_labels_cannot_change_training(name):
    rows = sample_rows()
    cutoff = "2023-06-10"
    first = a.fit(rows, name, cutoff)
    for row in rows:
        if row["mature"] >= cutoff:
            row["x"] = [float("nan")] * 32
            row["y"] = 1 - row["y"]
    second = a.fit(rows, name, cutoff)
    assert first["fit_hash"] == second["fit_hash"]
    assert first["max_mature_date"] < cutoff
    np.testing.assert_array_equal(first["model"][-1].coef_, second["model"][-1].coef_)
    assert first["fit_dates"] <= a.RECIPES[name][1]


def test_decay_half_life_and_duplicate_shares_preserve_family_influence():
    rows = sample_rows()
    weights = a.weights(rows, "MONTHLY_NAT252")
    assert weights[-2] / weights[-128] == pytest.approx(2)
    duplicated = rows + [dict(r) for r in rows if r["code"] == "a"]
    extended = a.weights(duplicated, "MONTHLY_NAT252")
    assert weights.sum() == pytest.approx(extended.sum())
    assert sum(w for r, w in zip(rows, weights, strict=True) if r["code"] == "a") == pytest.approx(
        sum(w for r, w in zip(duplicated, extended, strict=True) if r["code"] == "a")
    )


def test_regime_features_have_known_units_and_fail_on_invalid_input():
    x = [0.0] * 32
    x[30], x[31], x[3], x[7], x[11] = 0.01, 1, 0.02, 0.04, 0.01
    assert a.vector(x, "WEEKLY_REGIME252") == pytest.approx([0.01, 1, 2, 0.5, 0.02, 0.005])
    x[3] = -1
    with pytest.raises(ValueError, match="ADAPTIVE_INPUT_INVALID"):
        a.vector(x, "WEEKLY_REGIME252")


class ConstantModel:
    def predict_proba(self, x):
        return np.asarray([[0.7, 0.3]] * len(x))


@pytest.fixture
def ready(previous_ready, monkeypatch):  # noqa: F811
    b.save(a.root() / "result.json", {"winner": "MONTHLY_BAL504", "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": {"model": ConstantModel()}} for n in a.CANDIDATES}
    monkeypatch.setattr(a, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return previous_ready


def test_new_version_cannot_backfill_september15_even_if_clock_is_before_deadline(ready, monkeypatch):
    monkeypatch.setattr(a, "models", lambda: pytest.fail("must not load new version for old target"))
    result = a.tick()
    assert result["verified_forecasts"] == 0
    assert result["missing_due_predictions"] == 0


def test_first_eligible_target_saves_immutable_answers_and_pairs_mature_outcomes(ready, monkeypatch):
    # 父链fixture使用9月15日；只在测试中移动版本生效日以复用完整、已验证的父证据。
    monkeypatch.setattr(a, "FIRST_TARGET", "2026-09-15")
    directory, original, _ = ready
    assert a.tick()["verified_forecasts"] == 1
    path = a.root() / "forward/2026-09-15/001000.json"
    content = path.read_bytes()
    assert len(b.read(path)["answers"]) == len(a.CANDIDATES)
    a.tick()
    assert path.read_bytes() == content
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 0, "actual_direction": "DOWN", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    result = a.report()
    assert result["matched_forward_metrics"]["MONTHLY_BAL504"]["accuracy"] == 1
    assert result["matched_forward_metrics"]["MONTHLY_BAL504"]["count"] == 1


def test_cutoff_crossing_is_invalid_and_post_cutoff_never_loads_models(ready, monkeypatch):
    monkeypatch.setattr(a, "FIRST_TARGET", "2026-09-15")
    save = b.save

    def crossing(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == a.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    result = a.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1
    monkeypatch.setattr(a, "models", lambda: pytest.fail("unexpected post-cutoff model loading"))
    assert a.tick()["verified_forecasts"] == 0


def test_parent_change_after_receipt_is_rejected(ready, monkeypatch):
    from app.services import direction_1d_sprint_fund_response as p

    monkeypatch.setattr(a, "FIRST_TARGET", "2026-09-15")
    path = p.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["source_hash"] = "tampered"
    b.save(path, value, replace=True)
    with pytest.raises(ValueError, match="ROUND_05_PARENT_NOT_VERIFIED"):
        a.tick()


def test_old_runner_failure_does_not_skip_new_branch(monkeypatch):
    from scripts import direction_1d_sprint_adaptive as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.adaptive, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
