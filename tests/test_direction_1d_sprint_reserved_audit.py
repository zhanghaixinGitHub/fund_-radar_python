"""验证保留期复核拒绝跨季度泄露和重复题目，并保留原有方向阈值的边界。"""

import numpy as np
import pytest
from app.services import direction_1d_sprint_reserved_audit as audit


def test_future_target_is_rejected_even_if_maturity_metadata_is_backdated(monkeypatch):
    rows = [{"u": "2026-04-01", "mature": "2026-03-30"}]
    monkeypatch.setattr(audit.response, "selected", lambda rows, cutoff: rows)
    with pytest.raises(ValueError, match="RESERVED_AUDIT_FUTURE_LABEL"):
        audit.choose_training(rows, "2026-04-01")


def test_exam_rejects_duplicate_and_same_day_inputs():
    row = {"code": "1", "u": "2026-01-06", "t": "2026-01-05"}
    with pytest.raises(ValueError, match="DUPLICATE_QUESTION"):
        audit.validate_exam([row, row], "2026-01-01", "2026-04-01")
    with pytest.raises(ValueError, match="EXAM_RANGE"):
        audit.validate_exam([row | {"t": row["u"]}], "2026-01-01", "2026-04-01")


def test_frozen_thresholds_do_not_reinterpret_half_score_or_zero_spx():
    class HalfModel:
        def predict_proba(self, x):
            return np.full((len(x), 2), 0.5)

    row = {
        "code": "1",
        "family": "1",
        "group": "CN_EQUITY",
        "u": "2026-01-06",
        "y": 0,
        "actual_direction": "FLAT",
        "x": [0.0] * 32,
    }
    result = audit.score_models({"model": HalfModel()}, {"model": HalfModel()}, [row])
    assert [result[name][0]["prediction"] for name in audit.NAMES] == [0, 1, 0, 1]


def test_family_shares_do_not_increase_daily_weight():
    rows = [
        {"u": "2026-01-06", "family": "a", "prediction": 1, "y": 1},
        {"u": "2026-01-06", "family": "b", "prediction": 0, "y": 1},
    ]
    assert audit.daily(rows + [dict(rows[0])] * 9) == {"2026-01-06": 0.5}
