"""输入适配的时间/缺失/身份反例测试；标签测试全部使用人工十进制数。"""

from __future__ import annotations

import copy
from datetime import date, timedelta

import pytest

from scripts import fund_002112_existing_data_inputs_v1 as data


def sessions(count=110):
    start = date(2024, 1, 1)
    days = []
    while len(days) < count:
        if start.weekday() < 5:
            days.append(start.isoformat())
        start += timedelta(days=1)
    return days


def report(end="2023-12-31", available="2024-01-02T08:00:00+08:00", weight="10"):
    return {
        "fund_code": "002112",
        "fund_master_code": "001412",
        "report_end": end,
        "available_at": available,
        "published_date": "2024-01-01",
        "raw": {"sha256": end},
        "holdings": [{"stock_code": "000001.SZ", "nav_weight_pct": weight, "reported_rank": 1}],
        "disclosed_nav_pct": weight,
        "stock_nav_pct": "90",
        "full_stock_disclosure": False,
        "reported_industries": [{"nav_weight_pct": "60"}, {"nav_weight_pct": "30"}],
    }


def test_date_only_is_next_natural_day_and_latest_constraint():
    assert data.available_at("2024-09-30") == "2024-10-01T08:00:00+08:00"
    assert data.available_at("2024-09-30", "2024-10-24") == "2024-10-25T08:00:00+08:00"
    assert data.available_at("2024-09-30", "2024-10-02T10:30:00+08:00") == "2024-10-02T10:30:00+08:00"


def test_unknown_and_naive_timestamp_rejected():
    with pytest.raises(ValueError, match="UNKNOWN"):
        data.available_at(None)
    with pytest.raises(ValueError, match="TIMEZONE"):
        data.available_at("2024-01-01T12:00:00")


def test_nav_lag_finds_61_real_contiguous_days_and_preserves_u_t():
    days = sessions()
    nav = {d: {"unit_nav": str(1 + i / 100), "available_at": data.at0800(d).isoformat()} for i, d in enumerate(days)}
    nav[days[-2]]["available_at"] = "2026-01-01T08:00:00+08:00"
    end, lag, features = data.choose_nav_window(days, nav, days[-1])
    assert end == days[-3] and lag == 1 and len(features) == 8
    assert features[-1] == 1
    assert features[:-1] == data.nav_features([nav[d]["unit_nav"] for d in days[-63:-2]])


def test_nav_lag_max20_does_not_delete_date_or_fill():
    days = sessions()
    nav = {d: {"unit_nav": str(1 + i / 100), "available_at": "2023-12-01T08:00:00+08:00"} for i, d in enumerate(days)}
    for d in days[-22:-1]:
        nav[d]["available_at"] = "2026-01-01T08:00:00+08:00"
    assert data.choose_nav_window(days, nav, days[-1]) == (None, None, None)


def test_nav_features_match_original_pure_formula():
    from app.services.direction_1d_protocol import features

    values = [1 + i * 0.005 + (i % 7) * 0.013 for i in range(61)]
    assert data.nav_features(values) == features(values)


@pytest.mark.parametrize("values", [[1] * 61, [float("nan")] * 61, [0] * 61, [1] * 60])
def test_invalid_or_flat_nav_is_missing(values):
    with pytest.raises(ValueError):
        data.nav_features(values)


def test_report_latest_period_before_latest_publication_and_identity():
    latest = report("2024-03-31", "2024-04-23T08:00:00+08:00")
    old_late = report("2023-12-31", "2024-04-25T08:00:00+08:00")
    future = report("2024-06-30", "2024-07-25T08:00:00+08:00")
    wrong = report("2024-04-30", "2024-05-01T08:00:00+08:00")
    wrong["fund_code"] = "001412"
    assert data.select_report([latest, old_late, future, wrong], data.at0800("2024-05-02")) == latest


def test_holdings_nav_weight_not_renormalized_and_amount_not_volume():
    days = sessions(24)
    stocks = {d: {"000001.SZ": {"pct_chg": 1.0, "amount": 100.0, "vol": 9000.0}} for d in days}
    stocks[days[-2]]["000001.SZ"]["amount"] = 200.0
    values, errors = data.holdings_features(report(), stocks, days[-22:-1], days, data.at0800(days[-1]))
    assert not errors
    assert values[0] == pytest.approx(0.001)
    assert values[1] == pytest.approx(0.1 * (1.01**5 - 1))
    assert values[2:6] == pytest.approx([0.1, 0.1, 0.01, 0.1])
    assert values[-1] == 0


def test_any_suspension_makes_h_missing_without_mutating_quote():
    days = sessions(24)
    stocks = {d: {"000001.SZ": {"pct_chg": 1.0, "amount": 100.0}} for d in days}
    del stocks[days[-10]]["000001.SZ"]
    before = copy.deepcopy(stocks)
    values, errors = data.holdings_features(report(), stocks, days[-22:-1], days, data.at0800(days[-1]))
    assert values is None and errors == ["MISSING_QUOTE:000001.SZ"]
    assert stocks == before


def test_late_quote_revision_not_available_early():
    days = sessions(24)
    stocks = {d: {"000001.SZ": {"pct_chg": 1.0, "amount": 100.0}} for d in days}
    stocks[days[-2]]["000001.SZ"]["revised_at"] = "2026-01-01T08:00:00+08:00"
    values, errors = data.holdings_features(report(), stocks, days[-22:-1], days, data.at0800(days[-1]))
    assert values is None and "QUOTE_VERSION_NOT_AVAILABLE" in errors[0]


def test_stale_report_rejected():
    days = sessions(24)
    values, errors = data.holdings_features(report("2022-12-31"), {}, days[-22:-1], days, data.at0800(days[-1]))
    assert values is None and errors == ["REPORT_STALE"]


def test_report_extensions_different_period_top10_and_units():
    old = report("2023-09-30", weight="8")
    new = report(weight="10")
    same_period = report(weight="99")
    values = data.report_extensions(new, [old, same_period, new], data.at0800("2024-01-11"))
    assert values[data.F[0]] == 0
    assert values[data.F[1]] == pytest.approx(0.02)
    assert values[data.F[4]] == pytest.approx(0.45)
    assert values[data.F[5]] == 10
    assert all(values[c] is None for c in data.F[6:])


def test_optional_uses_only_2024_x_and_unknown_is_not_zero():
    rows = [
        {
            "target_date": f"2024-{i:03}",
            "groups": {g: [1] for g in ["N", "H", "M", "I"]},
            "raw_F": {"x": float(i) if i < 80 else None, "unknown": None, "constant": 1},
        }
        for i in range(100)
    ]
    rows += [
        {
            "target_date": "2025-01-02",
            "groups": {g: [1] for g in ["N", "H", "M", "I"]},
            "raw_F": {"x": None, "unknown": 5, "constant": 2},
        }
    ]
    result = data.choose_optional(rows, ["x", "unknown", "constant"], "raw_F")
    assert result["columns"] == ["x"] and result["joint_2024_rows"] == 80 and result["enabled"]
    assert result["column_decisions"]["unknown"]["covered"] == 0
    assert result["column_decisions"]["constant"]["reason"] == "CONSTANT_2024_INPUT"


def test_snapshot_rejects_changed_original_and_is_non_overwriting(tmp_path):
    source = tmp_path / "original.json"
    source.write_text('{"v":1}', encoding="utf-8")
    expected = data.sha(source)
    snapshot = data.Snapshot(tmp_path / "out")
    out = snapshot.copy(source, expected)
    source.write_text('{"v":2}', encoding="utf-8")
    assert data.read(out) == {"v": 1}
    with pytest.raises(ValueError, match="EXPECTED_HASH"):
        data.Snapshot(tmp_path / "out").copy(source, expected)
    with pytest.raises(ValueError, match="CONFLICT"):
        data.save_new(out, {"v": 3})


def artificial_label_root(tmp_path, target="2024-12-31", mature="2025-01-01T08:00:00+08:00"):
    """仅人工值：U与T向上，U与S向下，用于发现错误标签基准。"""
    base, end = "2024-12-30", "2024-12-27"
    nav = {
        base: {"unit_nav": "1.0000000", "available_at": "2024-12-31T08:00:00+08:00"},
        end: {"unit_nav": "2", "available_at": "2024-12-28T08:00:00+08:00"},
        target: {"unit_nav": "1.0000001", "available_at": mature},
    }
    rows = [{"target_date": target, "base_date": base, "nav_end_date": end, "label_mature_at": mature}]
    data.save_new(tmp_path / "snapshot/nav-by-date.json", nav)
    data.save_new(
        tmp_path / "handoff/data/inputs.json",
        {"version": data.VERSION, "rows": rows, "input_digest": data.digest(rows)},
    )
    data.save_new(tmp_path / "protocol.json", {"test": "ARTIFICIAL_ONLY"})
    data.save_new(tmp_path / "handoff/data/freeze-metadata.json", {"frozen_at": "2026-09-30T18:00:00+08:00"})
    return data.sha(tmp_path / "protocol.json")


def gate(tmp_path, protocol_hash, dates, stage="V1", purpose="train", **extra):
    path = tmp_path / "gate.json"
    value = {
        "protocol_sha256": protocol_hash,
        "stage": stage,
        "purpose": purpose,
        "allowed_target_dates": dates,
        "fit_cutoff": data.FIT_CUTOFF.get(stage),
        **extra,
    }
    data.save_new(path, value)
    return path


def test_label_uses_u_t_exact_decimal_not_s_or_rounded_return(tmp_path):
    protocol = artificial_label_root(tmp_path)
    dates = ["2024-12-31"]
    result = data.load_labels(
        tmp_path,
        dates,
        stage="V1",
        purpose="train",
        protocol_sha256=protocol,
        gate_path=gate(tmp_path, protocol, dates),
    )
    assert result["labels"] == {"2024-12-31": "UP"}
    assert data.classify_label("1.0", "1.000") == "FLAT"


def test_label_maturity_excludes_late_nav_even_target_in_training(tmp_path):
    protocol = artificial_label_root(tmp_path, mature="2025-01-14T08:00:00+08:00")
    dates = ["2024-12-31"]
    result = data.load_labels(
        tmp_path,
        dates,
        stage="V1",
        purpose="train",
        protocol_sha256=protocol,
        gate_path=gate(tmp_path, protocol, dates),
    )
    assert result["labels"] == {}
    assert result["excluded"] == {"2024-12-31": "LABEL_NOT_MATURE_AT_CUTOFF"}


def test_2026_label_requires_prediction_freeze_before_reading_nav(tmp_path, monkeypatch):
    protocol = artificial_label_root(tmp_path, target="2026-01-05")
    dates = ["2026-01-05"]
    gp = gate(tmp_path, protocol, dates, stage="T", purpose="evaluate")
    original = data.read

    def deny_nav(path):
        assert not str(path).endswith("nav-by-date.json")
        return original(path)

    monkeypatch.setattr(data, "read", deny_nav)
    with pytest.raises(KeyError, match="prediction_freeze_path"):
        data.load_labels(tmp_path, dates, stage="T", purpose="evaluate", protocol_sha256=protocol, gate_path=gp)


def test_training_stage_cannot_open_future_targets(tmp_path):
    protocol = artificial_label_root(tmp_path, target="2026-01-05")
    dates = ["2026-01-05"]
    with pytest.raises(ValueError, match="DATE_BOUNDARY"):
        data.load_labels(
            tmp_path,
            dates,
            stage="V1",
            purpose="train",
            protocol_sha256=protocol,
            gate_path=gate(tmp_path, protocol, dates),
        )


def test_training_cutoff_cannot_be_extended(tmp_path):
    protocol = artificial_label_root(tmp_path)
    dates = ["2024-12-31"]
    with pytest.raises(ValueError, match="CUTOFF_BOUNDARY"):
        data.load_labels(
            tmp_path,
            dates,
            stage="V1",
            purpose="train",
            protocol_sha256=protocol,
            gate_path=gate(tmp_path, protocol, dates, fit_cutoff="2026-01-01T08:00:00+08:00"),
        )


def test_final_requires_finished_test_before_labels(tmp_path):
    protocol = artificial_label_root(tmp_path)
    dates = ["2024-12-31"]
    with pytest.raises(KeyError, match="test_evaluation_path"):
        data.load_labels(
            tmp_path,
            dates,
            stage="FINAL",
            purpose="train",
            protocol_sha256=protocol,
            gate_path=gate(tmp_path, protocol, dates, stage="FINAL", fit_cutoff="2026-09-30T18:00:00+08:00"),
        )


def test_ready_digest_rejects_tampered_inputs(tmp_path):
    artificial_label_root(tmp_path)
    data.save_new(tmp_path / "handoff/data/ready.json", {"artifacts": {"handoff/data/inputs.json": "0" * 64}})
    with pytest.raises(ValueError, match="READY_ARTIFACT_CHANGED"):
        data.load_inputs(tmp_path)


@pytest.fixture(scope="module")
def real_inputs():
    """只读真实 X；此 fixture 不调用 load_labels，也不比较净值U/T。"""
    path = data.DEFAULT_ROOT / "handoff/data/inputs.json"
    if not path.exists():
        pytest.skip("输入快照尚未生成")
    # R1回归永远读取其不可变原件，不能随active指针切换而失去保留性断言。
    return data.read(path)


def test_real_calendar_665_and_no_label_fields(real_inputs):
    rows = real_inputs["rows"]
    assert len(rows) == 665 and len({r["target_date"] for r in rows}) == 665
    assert rows[0]["target_date"] == "2024-01-02" and rows[-1]["target_date"] == "2026-09-29"
    assert {year: sum(r["target_date"].startswith(year) for r in rows) for year in ["2024", "2025", "2026"]} == {
        "2024": 242,
        "2025": 243,
        "2026": 180,
    }
    assert all(not {"actual_direction", "label", "y", "target_unit_nav", "hit"} & set(r) for r in rows)
    mapping = {r["target_date"]: r["base_date"] for r in rows}
    assert mapping["2025-01-02"] == "2024-12-31"
    assert mapping["2026-01-05"] == "2025-12-31"


def test_real_feature_coverage_and_order(real_inputs):
    rows = real_inputs["rows"]
    assert {g: sum(r["groups"][g] is not None for r in rows) for g in ["N", "M", "H", "I"]} == {
        "N": 665,
        "M": 665,
        "H": 621,
        "I": 665,
    }
    assert max(r["lag_sessions"] for r in rows) == 14
    assert len(real_inputs["candidate_columns"]["E"]) == 36
    assert len(real_inputs["candidate_columns"]["F"]) == 42
    assert real_inputs["candidate_columns"]["C"] == data.N + data.H + data.M
    for r in rows:
        for g, values in r["groups"].items():
            if values is not None:
                assert len(values) == len(real_inputs["group_columns"][g])


def test_real_h_m_formula_matches_original_without_initializer(real_inputs, monkeypatch):
    """只调用旧计算函数并替换唯一初始化入口；无数据库、旧文件或标签操作。"""
    from datetime import datetime

    from app.services import fund_exposure_features as original

    bundle = data.read(data.DEFAULT_ROOT / "snapshot/normalized-inputs.json")
    monkeypatch.setattr(original, "initialize", lambda: {"max_report_age_days": 210})
    legacy = {
        "reports": bundle["reports"],
        "days": {d: {"rows": values, "receipt": {"sha256": "FROZEN_TEST"}} for d, values in bundle["stocks"].items()},
        "indices": {c: {"rows": values} for c, values in bundle["markets"].items()},
    }
    calendar = tuple(date.fromisoformat(d) for d in bundle["sessions"])
    for row in real_inputs["rows"]:
        expected = original.calculate(
            legacy,
            date.fromisoformat(row["base_date"]),
            datetime.fromisoformat(row["as_of"]),
            research_sessions=calendar,
        )
        assert row["groups"]["H"] == expected["holdings_features"]
        assert row["groups"]["M"] == expected["market_features"]


def test_real_industry_matches_verified_package(real_inputs):
    original = {r["target_date"]: r for r in data.read(data.PACKAGE / "sector-source-inputs.json")["rows"]}
    for row in real_inputs["rows"]:
        expected = [
            float(original[row["target_date"]]["sector_background"][code][f"return_{n}d"])
            for code in data.SECTORS
            for n in [1, 5, 20]
        ]
        assert row["groups"]["I"] == pytest.approx(expected, abs=1e-14)


def test_real_all_nav_windows_obey_information_cutoff(real_inputs):
    bundle = data.read(data.DEFAULT_ROOT / "snapshot/normalized-inputs.json")
    nav, calendar = bundle["nav"], bundle["sessions"]
    for row in real_inputs["rows"]:
        ix = calendar.index(row["nav_end_date"])
        window = calendar[ix - 60 : ix + 1]
        assert len(window) == 61
        assert row["nav_end_date"] <= row["base_date"] < row["target_date"]
        assert all(nav[d]["available_at"] <= row["as_of"] for d in window)
    expected = {
        "2024-09-30": "2024-10-24",
        "2024-12-31": "2025-01-13",
        "2025-03-31": "2025-04-18",
        "2025-06-30": "2025-07-18",
        "2025-09-30": "2025-10-28",
        "2025-12-31": "2026-01-22",
        "2026-03-31": "2026-04-21",
    }
    for d, ann in expected.items():
        assert nav[d]["ann_date"] == ann and nav[d]["available_at"] == data.available_at(ann)


def test_real_suspension_19_pairs_and_no_future_resume(real_inputs):
    pairs = set()
    impacted = 0
    for row in real_inputs["rows"]:
        states = row["stock_no_trade_states"]
        if states:
            impacted += 1
            assert row["groups"]["H"] is None
        for state in states:
            pairs.add((state["code"], state["date"]))
            assert state["price"] is None and state["return"] is None
            if row["as_of"] < "2025-04-23T08:00:00+08:00" and state["code"] == "301486.SZ":
                assert not state["resume_known"]
    assert len(pairs) == 19 and impacted == 44


def test_real_snapshot_originals_still_match_without_loading_labels(real_inputs):
    manifest = data.read(data.DEFAULT_ROOT / "source-manifest.json")
    for source in manifest["sources"]:
        assert data.sha(source["source_path"]) == source["sha256"]
        assert data.sha(data.DEFAULT_ROOT / source["snapshot_path"]) == source["sha256"]


def synthetic_report_state():
    return {
        "raw_sha256": "SYNTHETIC",
        "report_end": "2023-12-31",
        "available_at": "2024-01-17T08:00:00+08:00",
        "financial": {"c_share_net_assets_cny": "100"},
        "fund_inception_date": "2015-06-19",
        "c_share_added_date": "2015-11-16",
        "managers": [{"name": "甲", "begin_date": "2023-03-24", "end_date": None}],
    }


def test_r2_fund_age_uses_contract_not_c_share_date():
    values, evidence = data.state_extensions(synthetic_report_state(), [], "2024-01-18")
    assert values["fund_age_days"] == (date(2024, 1, 18) - date(2015, 6, 19)).days
    assert evidence["c_share_added_date"] == "2015-11-16"
    assert values["manager_min_tenure_days"] == (date(2024, 1, 18) - date(2023, 3, 24)).days


def test_r2_later_known_end_date_never_backfills():
    state = synthetic_report_state()
    notices = [
        {
            "raw_sha256": "ARTIFICIAL",
            "available_at": "2024-02-02T08:00:00+08:00",
            "events": [{"kind": "DEPART", "name": "甲", "effective_date": "2024-01-20"}],
        }
    ]
    early, evidence = data.state_extensions(state, notices, "2024-01-25")
    assert early["manager_count"] == 1 and evidence["applied_notices"] == []
    late, evidence = data.state_extensions(state, notices, "2024-02-02")
    assert late["manager_count"] is None and evidence["manager_roster"] == {}


def test_r2_pause_duties_does_not_end_appointment_or_reset_tenure():
    state = synthetic_report_state()
    notices = [
        {
            "raw_sha256": "ARTIFICIAL",
            "available_at": "2024-01-19T08:00:00+08:00",
            "events": [{"kind": "PAUSE_DUTIES", "name": "甲", "effective_date": "2024-01-18"}],
        }
    ]
    values, _ = data.state_extensions(state, notices, "2024-01-20")
    assert values["manager_count"] == 1
    assert values["manager_min_tenure_days"] == (date(2024, 1, 20) - date(2023, 3, 24)).days


def test_r2_late_old_notice_cannot_override_newer_report_period():
    state = synthetic_report_state()
    notices = [
        {
            "raw_sha256": "ARTIFICIAL",
            "available_at": "2024-01-19T08:00:00+08:00",
            "events": [{"kind": "DEPART", "name": "甲", "effective_date": "2023-10-20"}],
        }
    ]
    values, evidence = data.state_extensions(state, notices, "2024-01-20")
    assert values["manager_count"] == 1 and evidence["applied_notices"] == []


def test_r2_notice_latest_cms_constraint_beats_earlier_paper_date():
    document = {
        "catalog": {"publishDate": str(int(data.at0800("2026-01-01").timestamp() * 1000))},
        "receipt": {"sha256": "ARTIFICIAL"},
    }
    pages = ["德邦鑫星价值公告送出日期：2024年12月19日离任基金经理姓名揭诗琪离任日期2024年12月19日"]
    result = data.parse_manager_notice(document, pages)
    assert result["available_at"] == "2026-01-01T08:00:00+08:00"
    assert result["events"][0]["effective_date"] == "2024-12-19"


def test_r2_report_c_asset_parser_annual_current_year_second_column():
    pages = [
        "002112德邦鑫星价值人民币元2024年德邦鑫星价值A德邦鑫星价值C"
        "期末基金资产净值1,000.002,000.003,000.004,000.00"
        "基金合同生效日2015年6月19日自2015年11月16日起本基金增加C类基金份额"
        "雷涛本基金的基金经理2024年1月30日-"
    ]
    rep = report(end="2024-12-31")
    rep["raw"]["sha256"] = "SYNTHETIC"
    result = data.parse_report_state(rep, pages)
    assert result["financial"]["c_share_net_assets_cny"] == "2000.00"
    assert result["financial"]["a_share_net_assets_cny"] == "1000.00"
    with pytest.raises(ValueError, match="SHARE_COLUMN_ORDER"):
        data.parse_report_state(rep, [pages[0].replace("德邦鑫星价值A德邦鑫星价值C", "德邦鑫星价值C德邦鑫星价值A")])
    with pytest.raises(ValueError, match="UNIT"):
        data.parse_report_state(rep, [pages[0].replace("人民币元", "人民币万元")])


@pytest.mark.parametrize("omit_feature_binding", [False, True])
def test_r2_active_pointer_binds_ready_input_and_feature_hashes(tmp_path, omit_feature_binding):
    protocol = artificial_label_root(tmp_path)
    assert protocol
    rev = tmp_path / "handoff/data/revisions/v2"
    payload = data.read(tmp_path / "handoff/data/inputs.json")
    data.save_new(rev / "inputs.json", payload)
    data.save_new(rev / "feature-spec.json", {"revision": "v2"})
    relative = "handoff/data/revisions/v2/"
    artifacts = {relative + n: data.sha(rev / n) for n in ["inputs.json", "feature-spec.json"]}
    if omit_feature_binding:
        artifacts.pop(relative + "feature-spec.json")
    data.save_new(rev / "ready.json", {"status": "READY", "revision": "v2", "artifacts": artifacts})
    pointer = {
        "revision": "v2",
        "ready_path": relative + "ready.json",
        "ready_sha256": data.sha(rev / "ready.json"),
        "inputs_path": relative + "inputs.json",
        "inputs_sha256": data.sha(rev / "inputs.json"),
        "feature_spec_path": relative + "feature-spec.json",
        "feature_spec_sha256": data.sha(rev / "feature-spec.json"),
    }
    data.save_new(tmp_path / "handoff/data/active-revision.json", pointer)
    if omit_feature_binding:
        with pytest.raises(ValueError, match="FEATURE_SPEC_NOT_BOUND_TO_READY"):
            data.load_inputs(tmp_path)
        return
    loaded = data.load_inputs(tmp_path)
    assert loaded["revision"] == "v2" and loaded["data_ready_path"] == str((rev / "ready.json").resolve())
    (rev / "inputs.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="EVIDENCE_INVALID"):
        data.load_inputs(tmp_path)


@pytest.fixture(scope="module")
def real_v2():
    rev = data.DEFAULT_ROOT / "handoff/data/revisions/v2"
    if not (rev / "inputs.json").exists():
        pytest.skip("v2尚未生成")
    return data.read(rev / "inputs.json"), data.read(rev / "source-increment.json")


def test_real_r2_core_rows_identical_and_f_46_dimensions(real_v2, real_inputs):
    payload, _ = real_v2
    assert len(payload["rows"]) == 665
    assert payload["group_columns"]["F"] == data.F
    assert len(payload["candidate_columns"]["F"]) == 46
    for new, old in zip(payload["rows"], real_inputs["rows"], strict=True):
        for field in ["target_date", "base_date", "nav_end_date", "as_of", "lag_sessions", "label_mature_at", "report"]:
            assert new[field] == old[field]
        for group in ["N", "M", "H", "I", "G"]:
            assert new["groups"][group] == old["groups"][group]
        assert new["groups"]["F"][:6] == old["groups"]["F"]
    assert all(
        x["coverage"] == 1 and x["retained"] for x in payload["optional_decisions"]["F"]["column_decisions"].values()
    )


def test_real_r2_report_c_assets_units_period_and_known_db_mismatch(real_v2):
    _, sources = real_v2
    states = list(sources["report_states"].values())
    assert len(states) == 20
    for state in states:
        assert state["financial"]["unit"] == "CNY"
        assert state["financial"]["page"] in [3, 6]
        assert state["fund_inception_date"] == "2015-06-19"
    recent = [s for s in states if s["report_end"] == "2026-06-30"]
    assert len(recent) == 2
    assert all(s["financial"]["c_share_net_assets_cny"] == "11644906140.05" for s in recent)
    # 15010837507.60是A+C合计，数据库同名字段不能成为C份额规模。
    assert all(s["financial"]["c_share_net_assets_cny"] != "15010837507.60" for s in recent)


def test_real_r2_report_and_notice_times_no_future_departure(real_v2):
    _, sources = real_v2
    rows = data.read(data.DEFAULT_ROOT / "handoff/data/revisions/v2/row-state-evidence.json")
    for row in rows:
        cutoff = data.at0800(row["target_date"]).isoformat()
        assert row["report_available_at"] <= cutoff
        assert all(x["available_at"] <= cutoff for x in row["applied_notices"])
    assert any(
        n["available_at"][:4] == "2026" and n["declared_publication_date"][:4] == "2024"
        for n in sources["manager_notices"]
    )
    mapping = {r["target_date"]: r for r in rows}
    assert "雷涛" in mapping["2026-05-29"]["manager_roster"]
    assert "雷涛" not in mapping["2026-06-01"]["manager_roster"]


def test_real_r2_preserves_r1_artifacts_and_source_code_copy(real_v2):
    ready = data.read(data.DEFAULT_ROOT / "handoff/data/ready.json")
    for relative, expected in ready["artifacts"].items():
        assert data.sha(data.DEFAULT_ROOT / relative) == expected
    for path, expected in ready["code_sha256"].items():
        from pathlib import Path

        assert data.sha(data.DEFAULT_ROOT / "handoff/data/revisions/v2/baseline-code" / Path(path).name) == expected
