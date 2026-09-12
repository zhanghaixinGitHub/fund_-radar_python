"""验证转债研究的数据身份、时间、范围与预算边界；不消耗真实研究拟合预算。"""

from copy import deepcopy
from datetime import date, datetime, timedelta

import pytest
from app.services import direction_1d_convertible_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest


def row_fixture():
    days = [str(d) for d in calendar()[0] if date(2024, 1, 1) <= d <= date(2024, 1, 15)]
    row = {
        "fund_code": "f",
        "family": "f",
        "group": "CN_BOND",
        "kind": "HISTORICAL_RECONSTRUCTION",
        "t": days[5],
        "u": days[6],
        "x": [0.1] * 7,
        "y": 1,
    }
    mapping = {
        "asset_class": "CN_CONVERTIBLE",
        "index_code": "X",
        "funds": {"f": {"mapping_available_from": "2023-01-01"}},
    }
    return row, mapping, {d: str(100 + i) for i, d in enumerate(days)}, days


def test_shared_scope_and_no_target_day_price():
    row, mapping, prices, days = row_fixture()
    before = deepcopy(row)
    selected, excluded = study.attach_index([row, {**row, "fund_code": "pure_bond"}], mapping, prices)
    assert excluded == {"OUTSIDE_FROZEN_CONVERTIBLE_SCOPE": 1}
    assert study.vector(selected[0], "CB_NAV7") == row["x"]
    assert study.vector(selected[0], "CB_NAV9") == pytest.approx(row["x"] + [105 / 104 - 1, 105 / 100 - 1])
    prices[days[6]] = "999999"
    assert study.attach_index([row], mapping, prices)[0] == selected
    assert row == before


@pytest.mark.parametrize("bad", [None, "0", "NaN"])
def test_missing_or_bad_price_fails_both_candidates(bad):
    row, mapping, prices, days = row_fixture()
    if bad is None:
        del prices[days[2]]
    else:
        prices[days[2]] = bad
    with pytest.raises(ValueError, match="MARKET_(HISTORY_GAP|PRICE_INVALID)"):
        study.attach_index([row], mapping, prices)


@pytest.mark.parametrize("change", [{"group": "CN_EQUITY"}, {"u": "2025-01-02"}, {"kind": "FORWARD"}])
def test_asset_and_protected_period_fail(change):
    row, mapping, prices, _ = row_fixture()
    with pytest.raises(ValueError, match="ASSET_OR_PERIOD_INVALID"):
        study.attach_index([{**row, **change}], mapping, prices)


def test_disclosure_not_backdated_and_same_day_close_cannot_predict_morning():
    row, mapping, prices, _ = row_fixture()
    mapping["funds"]["f"]["mapping_available_from"] = row["u"]
    assert study.attach_index([row], mapping, prices) == ([], {"BEFORE_VERIFIED_MAPPING_DATE": 1})
    mapping["funds"]["f"]["mapping_available_from"] = "2023-01-01"
    row["u"] = row["t"]
    with pytest.raises(ValueError, match="EXAM_INPUT_NOT_AVAILABLE"):
        study.attach_index([row], mapping, prices)


def test_fit_membership_then_late_index_rejected(tmp_path, monkeypatch):
    row, mapping, prices, _ = row_fixture()
    (tmp_path / "baseline").mkdir()
    for name, content in (
        ("dataset.json", [row]),
        ("history.json", {}),
        ("models.json", {"CN_BOND-2024Q1": {"train_as_of": row["t"] + "T17:00:00+08:00"}}),
    ):
        study.write_new(tmp_path / "baseline" / name, content)
    study.write_new(tmp_path / "mapping.json", mapping)
    study.write_new(tmp_path / "prices.json", prices)
    monkeypatch.setattr(study.comparison, "availability", lambda *args: {})
    monkeypatch.setattr(study.comparison, "exact_fit", lambda rows, *args: rows)
    with pytest.raises(ValueError, match="FIT_INPUT_NOT_AVAILABLE"):
        study.selected_fit(tmp_path, "2024Q1")


def source_fixture(folder):
    now = datetime.now(ZONE)
    acquired = now - timedelta(days=10)
    files = {
        "source-inventory.json": {
            "checked_at": now.isoformat(),
            "sources": [
                {
                    "source_code": "TUSHARE_PRO_FUND",
                    "enabled": True,
                    "authorization_verified_at": now.isoformat(),
                    "authorized_api_names": ["index_basic", "index_daily"],
                    "retention_days": 365,
                }
            ],
            "funds": [{"fund_code": "f", "benchmark": "转债指数"}],
        },
        "identity-X.json": {"status": "DOWNLOADED", "record": {"index_code": "X", "list_date": "2012-01-01"}},
        "evidence-f.json": {
            "id": "f",
            "status": "SAVED",
            "published": "2020-01-01",
            "file": "evidence-f.pdf",
            "heading": "测试转债基金",
            "evidence_hits": [{"text": "转债指数"}],
        },
    }
    (folder / "evidence-f.pdf").write_bytes(b"fixture-only-no-real-disclosure")
    files["evidence-f.json"]["sha256"] = study.file_hash(folder / "evidence-f.pdf")
    prices = {}
    for year in (2021, 2022, 2023, 2024):
        rows = [{"date": str(d), "close": str(100 + i)} for i, d in enumerate(calendar()[0]) if d.year == year]
        prices.update({r["date"]: r["close"] for r in rows})
        files[f"daily-X-{year}.json"] = {
            "status": "DOWNLOADED",
            "api": "index_daily",
            "code": "X",
            "year": year,
            "started_at": acquired.isoformat(),
            "retrieved_at": (acquired + timedelta(seconds=1)).isoformat(),
            "prices": rows,
        }
    files["new-prices.json"] = {"prices": {"X": prices}}
    for name, value in files.items():
        study.write_new(folder / name, value)
    mapping = {
        "asset_class": "CN_CONVERTIBLE",
        "index_code": "X",
        "funds": {
            "f": {
                "mapping_available_from": "2020-01-02",
                "evidence_metadata": "evidence-f.json",
                "fund_name_in_document": "测试转债基金",
                "benchmark_name_in_document": "转债指数",
            }
        },
    }
    reseal(folder)
    return mapping, acquired


def reseal(folder):
    seal = {
        "files": {
            p.name: study.file_hash(p)
            for p in folder.iterdir()
            if p.name not in {"data-seal.json", "data-seal-receipt.json"}
        }
    }
    # 测试夹具重新封存代表错误在采集阶段产生；生产写入一律使用排他创建。
    (folder / "data-seal.json").write_text(study.original.canonical(seal), encoding="utf-8")
    (folder / "data-seal-receipt.json").write_text(study.original.canonical({"hash": digest(seal)}), encoding="utf-8")


def test_source_reuse_preserves_first_acquisition_and_retention(tmp_path):
    mapping, acquired = source_fixture(tmp_path)
    prices, source, names = study.validate_data(tmp_path, mapping)
    assert len(prices) == 969
    assert datetime.fromisoformat(source["source_expires_at"]) == acquired + timedelta(days=365)
    assert source["new_api_calls"] == 0 and "evidence-f.pdf" in names


def test_raw_price_tampering_fails(tmp_path):
    mapping, _ = source_fixture(tmp_path)
    (tmp_path / "daily-X-2021.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="RAW_EVIDENCE_CHANGED"):
        study.validate_data(tmp_path, mapping)


@pytest.mark.parametrize("case", ["duplicate", "identity", "permission", "stale", "disclosure"])
def test_invalid_source_even_with_consistent_seal_fails(tmp_path, case):
    mapping, _ = source_fixture(tmp_path)
    name = {
        "duplicate": "daily-X-2021.json",
        "identity": "identity-X.json",
        "permission": "source-inventory.json",
        "stale": "source-inventory.json",
        "disclosure": "evidence-f.json",
    }[case]
    value = study.read(tmp_path / name)
    if case == "duplicate":
        value["prices"].append(value["prices"][0])
    elif case == "identity":
        value["record"]["index_code"] = "Y"
    elif case == "permission":
        value["sources"][0]["authorized_api_names"] = []
    elif case == "stale":
        value["checked_at"] = (datetime.now(ZONE) - timedelta(days=2)).isoformat()
    else:
        value["published"] = "2021-01-01"
    (tmp_path / name).write_text(study.original.canonical(value), encoding="utf-8")
    reseal(tmp_path)
    with pytest.raises(ValueError, match="CONVERTIBLE_"):
        study.validate_data(tmp_path, mapping)


def test_scope_outside_owner_is_rejected_before_source_read(tmp_path, monkeypatch):
    (tmp_path / "baseline").mkdir()
    study.write_new(tmp_path / "baseline/dataset.json", [{"fund_code": "owner_fund"}])
    study.write_new(tmp_path / "mapping.json", {"funds": {"other_fund": {}}})
    monkeypatch.setattr(study.comparison, "verify", lambda *args: {})
    monkeypatch.setattr(study, "validate_data", lambda *args: pytest.fail("不应读取或混入范围外基金"))
    with pytest.raises(ValueError, match="OUTSIDE_OWNER_BASELINE"):
        study.freeze(tmp_path, tmp_path, tmp_path / "mapping.json", tmp_path / "out")


def test_frozen_input_changes_fail(tmp_path):
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


@pytest.mark.parametrize("replay", [False, True])
def test_same_training_budget_cannot_be_spent_twice(tmp_path, monkeypatch, replay):
    (tmp_path / ("replay" if replay else "main")).mkdir()
    monkeypatch.setattr(study, "verify_inputs", lambda root: {})
    monkeypatch.setattr(study, "verify", lambda root: {})
    monkeypatch.setattr(study, "fit", lambda *args: pytest.fail("不应重复拟合"))
    with pytest.raises(FileExistsError):
        study.run(tmp_path, replay=replay)
