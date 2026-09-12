"""对应指数时间边界、同题分支与不可追加拟合检查；不进行真实研究拟合。"""

from copy import deepcopy
from datetime import date, datetime, timedelta

import pytest
from app.services import direction_1d_sector_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest


def fixtures():
    days = [str(d) for d in calendar()[0] if date(2024, 1, 1) <= d <= date(2024, 1, 15)]
    prices = {
        "000300.SH": {d: str(100 + i) for i, d in enumerate(days)},
        "X": {d: str(200 + 4 * i) for i, d in enumerate(days)},
    }
    row = {
        "fund_code": "f",
        "family": "f",
        "group": "CN_EQUITY",
        "t": days[5],
        "u": days[6],
        "x": [0.1] * 7,
        "y": 1,
        "actual_direction": "UP",
        "quarter": "2024Q1",
        "kind": "HISTORICAL_RECONSTRUCTION",
    }
    mapping = {
        "f": {
            "group": "CN_EQUITY",
            "status": "RESEARCH_MAPPING_SUPPORTED",
            "index_code": "X",
            "mapping_available_from": "2023-12-01",
        }
    }
    return row, mapping, prices, days


def test_relative_features_use_known_index_identity_and_no_future_price():
    row, mapping, prices, days = fixtures()
    rows, excluded = study.attach_specific([row], mapping, prices)
    assert not excluded and rows[0]["specific_input"]["index_code"] == "X"
    assert rows[0]["specific_input"]["x"] == pytest.approx([220 / 216 - 105 / 104, 220 / 200 - 105 / 100])
    prices["X"][days[6]] = "999999"
    assert study.attach_specific([row], mapping, prices)[0] == rows
    assert study.vector(rows[0], "COMMON9") == row["x"] + rows[0]["market_input"]["x"]
    assert study.vector(rows[0], "SPECIFIC11")[:9] == study.vector(rows[0], "COMMON9")
    assert len(study.feature_names("SPECIFIC11")) == 11


def test_mapping_is_not_backdated_and_both_candidates_share_exclusions():
    row, mapping, prices, _ = fixtures()
    mapping["f"]["mapping_available_from"] = row["u"]
    before = deepcopy(row)
    assert study.attach_specific([row], mapping, prices) == ([], {"BEFORE_VERIFIED_MAPPING_DATE": 1})
    assert row == before


@pytest.mark.parametrize("bad", [None, "0", "NaN"])
def test_specific_history_gap_or_invalid_price_never_uses_common_proxy(bad):
    row, mapping, prices, days = fixtures()
    if bad is None:
        del prices["X"][days[2]]
    else:
        prices["X"][days[2]] = bad
    with pytest.raises(ValueError, match="MARKET_(HISTORY_GAP|PRICE_INVALID)"):
        study.attach_specific([row], mapping, prices)


def test_unmapped_fund_rejected_and_bond_excluded():
    row, mapping, prices, _ = fixtures()
    with pytest.raises(ValueError, match="COVERAGE_MISSING"):
        study.attach_specific([row], {}, prices)
    row["group"] = "CN_BOND"
    mapping["f"].update(group="CN_BOND", status="SEPARATE_RECIPE_PENDING")
    assert study.attach_specific([row], mapping, prices) == ([], {"SEPARATE_RECIPE_PENDING": 1})


def test_hs300_assignment_adds_zero_relative_return():
    row, mapping, prices, _ = fixtures()
    mapping["f"]["index_code"] = "000300.SH"
    assert study.attach_specific([row], mapping, prices)[0][0]["specific_input"]["x"] == [0, 0]


def test_protected_year_inputs_fail():
    row, mapping, prices, _ = fixtures()
    row.update(t="2025-01-06", u="2025-01-07")
    with pytest.raises(ValueError, match="ANCHOR_INVALID"):
        study.attach_specific([row], mapping, prices)


def test_fit_uses_original_membership_then_rejects_late_index_input(tmp_path, monkeypatch):
    row, mapping, prices, _ = fixtures()
    (tmp_path / "baseline").mkdir()
    for name, content in (
        ("dataset.json", [row]),
        ("history.json", {}),
        ("models.json", {"CN_EQUITY-2024Q1": {"train_as_of": row["t"] + "T17:00:00+08:00"}}),
    ):
        study.write_new(tmp_path / "baseline" / name, content)
    study.write_new(tmp_path / "mapping.json", {"funds": mapping})
    study.write_new(tmp_path / "prices.json", {"prices": prices})
    monkeypatch.setattr(study.comparison, "availability", lambda *a: {})
    monkeypatch.setattr(study.comparison, "exact_fit", lambda rows, *a: rows)
    with pytest.raises(ValueError, match="FIT_INPUT_NOT_AVAILABLE"):
        study.selected_fit(tmp_path, "2024Q1")


def test_frozen_source_and_mapping_changes_fail(tmp_path):
    study.write_new(tmp_path / "mapping.json", {"f": "X"})
    spec = {
        "fingerprint": study.fingerprint(),
        "candidates": study.CANDIDATES,
        "recipe": study.RECIPE,
        "input_files": {"mapping.json": study.file_hash(tmp_path / "mapping.json")},
        "source": {"source_expires_at": (datetime.now(ZONE) + timedelta(days=1)).isoformat()},
    }
    study.write_new(tmp_path / "study.json", spec)
    study.write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    study.verify_inputs(tmp_path)
    (tmp_path / "mapping.json").write_text('{"f":"Y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN_INPUT_CHANGED"):
        study.verify_inputs(tmp_path)


def test_training_cannot_start_twice_in_same_directory(tmp_path, monkeypatch):
    (tmp_path / "main").mkdir()
    monkeypatch.setattr(study, "verify_inputs", lambda root: {})
    monkeypatch.setattr(study, "fit", lambda *a: pytest.fail("不应重复拟合"))
    with pytest.raises(FileExistsError):
        study.run(tmp_path)
