"""验证时间留出基础预测、逐题重算和重写摘要后的篡改拒绝。"""

from datetime import date, timedelta

import joblib
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_stacking_oof as s

from test_direction_1d_sprint_sector_return import vector


class BaseProbe:
    n_features_in_ = 155

    def predict(self, x):
        return x[:, 0]


def head(cutoff="2025-01-01"):
    return {
        "model": BaseProbe(),
        "mean": [0.0] * 5,
        "scale": [1.0] * 5,
        "code_order": s.ridge.CODE_ORDER,
        "shrinkage": 0.25,
        "cutoff": cutoff,
        "fit_end": "2024-12-20",
        "max_mature_date": "2024-12-21",
    }


def row():
    return {"code": "001021", "u": "2025-01-03", "mature": "2025-01-04", "group": "CN_BOND", "z": vector(), "y": 1}


def test_margin_is_fixed_vote_and_second_input_is_model_score():
    z = vector(values=(2, -3, 4))
    assert s.meta_vectors([z], [2.0])[0] == pytest.approx([1 / 3, 2.0])
    z[17] = -4
    assert s.meta_vectors([z], [-0.25])[0] == pytest.approx([-1 / 3, -0.25])
    assert s.quarter("2024-12-31") == "2024-10-01"
    assert s.quarter("2025-01-01") == "2025-01-01"


def prepared(monkeypatch, tmp_path, wrong=False, future=False):
    monkeypatch.setattr(b, "PROJECT", tmp_path)
    h = head("2025-01-02" if future else "2025-01-01")
    path = tmp_path / "base.joblib"
    joblib.dump(h, path)
    r = row()
    z2 = s.meta_vectors([r["z"]], s.base_scores([r["z"]], h))[0]
    if wrong:
        z2[1] += 1
    records = {(r["code"], r["u"]): {"row_hash": b.digest(r), "base_cutoff": "2025-01-01", "z2": z2}}
    manifest = {
        "references": {
            "part": {"receipt": {"group": "CN_BOND", "cutoff": "2025-01-01", "path": "base.joblib", "reused": False}}
        }
    }
    monkeypatch.setattr(s, "load", lambda: (records, manifest))
    return r


def test_training_feature_recomputes_same_earlier_base(monkeypatch, tmp_path):
    r = prepared(monkeypatch, tmp_path)
    x, _ = s.training_features([r])
    assert x.shape == (1, 2)


def test_rehashed_oof_score_does_not_bypass_base_recomputation(monkeypatch, tmp_path):
    r = prepared(monkeypatch, tmp_path, wrong=True)
    with pytest.raises(ValueError, match="RECOMPUTATION_CHANGED"):
        s.training_features([r])


def test_base_after_quarter_boundary_rejected(monkeypatch, tmp_path):
    r = prepared(monkeypatch, tmp_path, future=True)
    with pytest.raises(ValueError, match="BASE_TIME_CHANGED"):
        s.training_features([r])


def test_checkpoint_only_fits_earlier_mature_rows_and_does_not_refit(monkeypatch, tmp_path):
    monkeypatch.setattr(b, "ROOT", tmp_path / "run")
    monkeypatch.setattr(b, "PROJECT", tmp_path)
    earlier = [
        row()
        | {
            "y": i % 2,
            "u": str(date(2024, 1, 1) + timedelta(days=i)),
            "mature": str(date(2024, 1, 2) + timedelta(days=i)),
        }
        for i in range(200)
    ]
    future = row() | {"u": "2024-12-31", "mature": "2025-01-02"}
    rows = earlier + [future]
    selected = s.chosen(rows, "2025-01-01")
    assert selected == earlier
    calls = []

    def fit(actual, name, cutoff):
        chosen = s.chosen(actual, cutoff)
        calls.append(chosen)
        return head() | {"fit_hash": b.digest(chosen)}

    monkeypatch.setattr(s.ridge, "fit", fit)
    spec = {
        "group": "CN_BOND",
        "cutoff": "2025-01-01",
        "training_hash": b.digest(selected),
        "reuse_existing_R57_2025": False,
    }
    s.checkpoint(rows, spec)
    s.checkpoint(rows, spec)
    assert calls == [earlier]
    path = s.root() / "checkpoints/CN_BOND-2025-01-01.joblib"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="CHECKPOINT_CHANGED"):
        s.checkpoint(rows, spec)
