"""漏列对照修复：八分支完整、规则和五输入头独立、原答案复用且不重训。"""

from datetime import date, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_gap_ablation_v2 as s

from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


class Constant:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, x):
        return np.tile([1 - self.score, self.score], (len(x), 1))

    def fit(self, *args, **kwargs):
        pytest.fail("repair must never retrain")


def bundle():
    return {
        n: {
            "CN_EQUITY": {
                "model": Constant(0.8 if n == s.CONTROLS[2] else 0.4),
                "mean": [0.0] * (5 if n == s.CONTROLS[2] else 4 if n in s.CANDIDATES else 3),
                "scale": [1.0] * (5 if n == s.CONTROLS[2] else 4 if n in s.CANDIDATES else 3),
                "signed": n == s.CONTROLS[1],
            }
        }
        for n in s.CANDIDATES + s.CONTROLS[:3]
    }


def market():
    return {
        "features": [-1, -1, 1, 100, 100],
        "available": True,
        "etf_available": True,
        "cnya_available": True,
        "us_sessions": 1.0,
    }


def test_eight_distinct_branches_include_five_input_head_and_real_fixed_majority():
    assert len(s.BRANCHES) == len(set(s.BRANCHES)) == 8
    result = s.answers(market(), "CN_EQUITY", bundle())
    assert set(result) == set(s.BRANCHES)
    assert result["FROZEN_R73_MARKET_LR5"]["prediction"] == 1
    assert result["MARKET_MAJORITY3"]["prediction"] == 0
    assert result["FROZEN_R73_MARKET_LR5"]["route"] == "MARKET_LR5"


def test_composition_uses_original_heads_and_omits_unused_misnamed_entry(monkeypatch):
    native = bundle()
    original = {n: head for n, head in native.items() if n != s.CONTROLS[2]} | {"MARKET_MAJORITY3": "UNUSED_WRONG_SLOT"}
    monkeypatch.setattr(s.original, "models", lambda: ({"model_sha256": "OLD"}, original))
    monkeypatch.setattr(
        s.five, "models", lambda: ({"model_sha256": "FIVE"}, {s.five.CANDIDATES[0]: native[s.CONTROLS[2]]})
    )
    _, _, composed = s.composed()
    assert set(composed) == set(s.CANDIDATES + s.CONTROLS[:3])
    assert composed[s.CONTROLS[2]] is native[s.CONTROLS[2]]
    assert all(composed[n] is original[n] for n in s.CANDIDATES + s.CONTROLS[:2])


def test_all_eight_future_answers_use_independent_parent_and_real_receipt(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T03:00:00+08:00", "plan_hash": "REPAIR", "model_sha256": "COMPOSITION"}
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle()))
    monkeypatch.setattr(s, "live_market", lambda source: market())
    result = runtime.tick(s)
    assert result["verified_forecasts"] == 1
    value = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert len(value["answers"]) == 8
    assert value["answers"][s.CONTROLS[2]]["prediction"] == 1
    assert not (b.ROOT / "forward").exists()


def test_preparation_copies_every_original_answer_and_only_adds_missing_control(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(s, "plan", lambda: {"synthetic": "PLAN"})
    monkeypatch.setattr(s, "fingerprint", lambda: {"code": "SYNTHETIC"})
    monkeypatch.setattr(s.original, "active", lambda: None)
    monkeypatch.setattr(s.original, "fit", lambda *args: pytest.fail("no new fit"))
    rows = [
        {
            "code": f"{fund:06d}",
            "family": str(fund),
            "group": "CN_EQUITY",
            "u": str(date(2025, 1, 1) + timedelta(days=i)),
            "y": 0,
            "prediction": 0,
            "actual_direction": "DOWN",
            "old_question": i < 189,
        }
        for i in range(243)
        for fund in range(30)
    ]
    five_rows = [r | {"prediction": 1} for r in rows]

    def metrics(values):
        return {
            subset: b.metrics([r for r in values if subset == "all" or r["old_question"] == (subset == "old")])
            for subset in ("all", "old", "added")
        }

    old_metrics, five_metrics = metrics(rows), metrics(five_rows)
    old_result = {
        "model_sha256": "ORIGINAL",
        "metrics": {subset: {n: metric for n in s.original.BRANCHES} for subset, metric in old_metrics.items()},
    }
    five_result = {
        "model_sha256": "FIVE",
        "metrics": {subset: {s.five.CANDIDATES[0]: metric} for subset, metric in five_metrics.items()},
    }
    monkeypatch.setattr(s, "composed", lambda: (old_result, five_result, {}))
    for name in s.original.BRANCHES:
        b.save(s.original.root() / "folds" / f"1-CN_EQUITY-{name}.json", rows)
    b.save(s.five.root() / "folds" / f"1-CN_EQUITY-{s.five.CANDIDATES[0]}.json", five_rows)
    result = s.train()
    assert result["new_development_fits"] == result["new_current_fits"] == 0
    assert len(result["metrics"]["all"]) == 8
    assert result["metrics"]["all"][s.CONTROLS[2]]["accuracy"] == 0
    assert result["metrics"]["all"]["MARKET_MAJORITY3"]["accuracy"] == 1
    for name in s.original.BRANCHES:
        assert b.read(s.root() / "folds" / f"1-CN_EQUITY-{name}.json") == rows


def test_repair_entry_continues_last_verified_round_not_incomplete_original_entry():
    from scripts import direction_1d_sprint_market_gap_ablation_v2 as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_gap_delta"
