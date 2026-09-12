"""市场输入时间、同人口对照、封存及配对评价；不产生真实研究拟合。"""

from copy import deepcopy
from datetime import date, datetime, timedelta

import pytest
from app.services import direction_1d_market_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest


def prices_and_day():
    days = [str(d) for d in calendar()[0] if date(2024, 1, 1) <= d <= date(2024, 1, 15)]
    return {d: str(100 + i) for i, d in enumerate(days)}, days[5], days


def row(code="000001", group="CN_EQUITY"):
    prices, day, days = prices_and_day()
    return {
        "fund_code": code,
        "family": code,
        "group": group,
        "t": day,
        "u": days[6],
        "x": [0.1] * 7,
        "market_input": study.market_input(day, prices),
        "y": 1,
        "actual_direction": "UP",
        "quarter": "2024Q1",
        "kind": "HISTORICAL_RECONSTRUCTION",
    }


def test_market_features_only_use_t_and_past_sessions():
    prices, day, days = prices_and_day()
    first = study.market_input(day, prices)
    assert first["dates"] == days[:6]
    assert first["x"] == pytest.approx([105 / 104 - 1, 105 / 100 - 1])
    prices[days[6]] = "99999"
    assert study.market_input(day, prices) == first
    assert first["available_at_assumed"].endswith("18:00:00+08:00")


@pytest.mark.parametrize("bad", [None, "0", "-1", "NaN", "Infinity"])
def test_missing_or_invalid_market_input_fails_without_fill(bad):
    prices, day, days = prices_and_day()
    if bad is None:
        del prices[days[2]]
    else:
        prices[days[2]] = bad
    with pytest.raises(ValueError, match="MARKET_(HISTORY_GAP|PRICE_INVALID)"):
        study.market_input(day, prices)


def test_baseline_and_candidate_share_population_but_only_candidate_gets_market():
    r = row()
    assert study.vector(r, "NAV7_27") == r["x"]
    assert study.vector(r, "MARKET9_27") == r["x"] + r["market_input"]["x"]
    with pytest.raises(ValueError, match="GROUP_INVALID"):
        study.vector(row(group="CN_BOND"), "MARKET9_27")


def test_fit_preserves_old_membership_and_rejects_unavailable_market(tmp_path, monkeypatch):
    prices, _, _ = prices_and_day()
    rows = [row(), row("000002", "CN_MIXED"), row("000003", "CN_BOND")]
    (tmp_path / "baseline").mkdir()
    for name, data in [
        ("dataset.json", rows),
        ("history.json", {}),
        ("models.json", {"CN_EQUITY-2024Q1": {"train_as_of": "2024-04-01T00:00:00+08:00"}}),
    ]:
        study.write_new(tmp_path / "baseline" / name, data)
    study.write_new(tmp_path / "market-data.json", {"prices": {"000300.SH": prices}})
    monkeypatch.setattr(study.comparison, "availability", lambda *a: {})
    monkeypatch.setattr(study.comparison, "exact_fit", lambda rows, *a: rows)
    selected = study.selected_fit(tmp_path, "2024Q1")
    assert [r["fund_code"] for r in selected] == ["000001", "000002"]
    assert [r["x"] for r in selected] == [r["x"] for r in rows[:2]]
    (tmp_path / "baseline/models.json").write_text(
        '{"CN_EQUITY-2024Q1":{"train_as_of":"' + rows[0]["t"] + 'T17:00:00+08:00"}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="FIT_INPUT_NOT_AVAILABLE"):
        study.selected_fit(tmp_path, "2024Q1")


def test_paired_comparison_uses_named_reference_without_changing_original_group():
    rows = []
    for i in range(3):
        r = row(str(i))
        r["directions"] = {"NAV7_27": 0, "MARKET9_27": 1, study.comparison.BASELINE: 1}
        rows.append(r)
    before = deepcopy(rows)
    spec = {"bootstrap_seed": 1, "bootstrap_block_days": 5, "bootstrap_repetitions": 20}
    primary = study.paired(rows, "MARKET9_27", "NAV7_27", spec)
    old = study.paired(rows, "MARKET9_27", study.comparison.BASELINE, spec)
    assert primary["extra_correct"] == 3 and primary["weighted_accuracy_difference"] == 1
    assert old["extra_correct"] == 0 and rows == before


def test_frozen_input_and_spec_hashes_are_checked(tmp_path):
    study.write_new(tmp_path / "input.json", {"x": 1})
    spec = {
        "fingerprint": study.fingerprint(),
        "candidates": study.CANDIDATES,
        "recipe": study.RECIPE,
        "input_files": {"input.json": study.file_hash(tmp_path / "input.json")},
        "source": {"source_expires_at": (datetime.now(ZONE) + timedelta(days=1)).isoformat()},
    }
    study.write_new(tmp_path / "study.json", spec)
    study.write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    study.verify_inputs(tmp_path)
    (tmp_path / "input.json").write_text('{"x":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN_INPUT_CHANGED"):
        study.verify_inputs(tmp_path)
    spec["max_main_fits"] = 100
    (tmp_path / "study.json").write_text(study.original.canonical(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="STUDY_CHANGED"):
        study.verify_inputs(tmp_path)


def test_source_retention_is_not_reset_by_a_new_study(tmp_path):
    spec = {
        "fingerprint": study.fingerprint(),
        "candidates": study.CANDIDATES,
        "recipe": study.RECIPE,
        "input_files": {},
        "source": {"source_expires_at": "2020-01-01T00:00:00+08:00"},
    }
    study.write_new(tmp_path / "study.json", spec)
    study.write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    with pytest.raises(ValueError, match="RETENTION_EXPIRED"):
        study.verify_inputs(tmp_path)


def test_shared_source_lineage_rejects_changed_reused_receipt(tmp_path, monkeypatch):
    for name in ("copy", "origin"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "market.json").write_text("{}", encoding="utf-8")
    expected = study.file_hash(tmp_path / "copy/market.json")

    def seal(folder, name):
        if name == "study-frozen.json":
            return {"manifest_hash": "frozen"}
        return {
            "frozen_hash": "frozen",
            "manifest_hash": "copy" if folder.name == "copy" else "changed",
            "reused_source_folder": "origin" if folder.name == "copy" else None,
            "reused_prepared_hash": "original",
        }

    monkeypatch.setattr(study, "read_seal", seal)
    with pytest.raises(ValueError, match="LINEAGE_CHANGED"):
        study.shared_source_lineage(tmp_path / "copy", expected)
