from datetime import date

import pytest

from app.services.fund_rating_monthly_r5 import annual_disclosure_range, rolling_window, validate_window, three_year_disclosure_start


def test_rolling_window_uses_supplied_actual_month_ends():
    # 故意包含周末作为已核来源的特殊交易日，函数不能用星期几擅自过滤。
    days = [str(date(year, month, 15)) for year in range(2023, 2027) for month in range(1, 13)]
    selected, months = rolling_window(days, "2026-10-15")
    assert selected[0] == "2023-10-15"
    assert selected[-1] == "2026-10-15"
    assert len(months) == 37
    assert months[-13] == "2025-10-15"
    assert validate_window(selected, months, days, "2026-10-15") == (selected, months)


def test_missing_or_replaced_calendar_day_is_rejected():
    days = [str(date(year, month, 15)) for year in range(2023, 2027) for month in range(1, 13)]
    selected, months = rolling_window(days, "2026-10-15")
    with pytest.raises(ValueError, match="CALENDAR_WINDOW_CONFLICT"):
        validate_window(selected[:-1], months, days, "2026-10-15")
    with pytest.raises(ValueError, match="MONTH_END_REQUIRED"):
        rolling_window(days, "2026-10-16")


def test_disclosure_ranges_roll_across_year_and_keep_month_end():
    assert annual_disclosure_range("2026-10-30") == ("2025-11-01", "2026-10-30")
    assert annual_disclosure_range("2026-12-31") == ("2026-01-01", "2026-12-31")
    assert three_year_disclosure_start("2026-10-30") == "2023-10-31"
    assert three_year_disclosure_start("2026-09-30") == "2023-09-30"


def test_public_js_serialization_tail_does_not_allow_economic_rounding():
    from app.services.fund_rating_thematic_performance import cumulative_crosscheck

    assert cumulative_crosscheck("0.9501", "0.95009999999999994")["economic_values_rounded"] is False
    with pytest.raises(ValueError, match="NAV_SOURCE_CONFLICT"):
        cumulative_crosscheck("0.9501", "0.95009")


def test_monthly_r5_rejects_changed_observation_without_original_replay(monkeypatch, tmp_path):
    from app.services import fund_rating_monthly_r5, fund_rating_r5

    monkeypatch.setattr(fund_rating_monthly_r5, "compile_same_index_monthly",
                        lambda *args, **kwargs: {"observations": {"return_1y": {"target": "0.1"}}})
    packet = dict(fund_code="008888", as_of_date="2026-10-30", verified_at="2026-11-17T05:00:00+00:00",
                  unit_contract="RATIO_RANKS_RATIO_WINDOWS", quality_metrics={"observations": {"return_1y": {"target": "0.2"}}})
    packet["monthly_r5_compilation"] = {key: packet[key] for key in ["fund_code", "as_of_date", "verified_at"]}
    with pytest.raises(ValueError, match="MONTHLY_ORIGINAL_OBSERVATIONS_CONFLICT"):
        fund_rating_r5.calculate(packet, {}, root=tmp_path)


def test_monthly_r5_rejects_mismatched_cutoff_before_replay(tmp_path):
    from app.services import fund_rating_r5

    packet = dict(fund_code="008888", as_of_date="2026-10-30", verified_at="2026-11-17T05:00:00+00:00")
    packet["monthly_r5_compilation"] = {**packet, "as_of_date": "2026-09-30"}
    with pytest.raises(ValueError, match="MONTHLY_PACKET_SCOPE_CONFLICT"):
        fund_rating_r5.calculate(packet, {}, root=tmp_path)


def test_weekly_peer_missing_three_year_point_does_not_destroy_available_one_year():
    from app.services.fund_rating_monthly_r5 import licensed_share_returns

    days = [str(date(year, month, 15)) for year in range(2023, 2027) for month in range(1, 13)
            if "2023-10" <= f"{year}-{month:02}" <= "2026-10"]
    class Source:
        asof, research = "2026-10-15", "2026-11-17"
        def original(self, sha):
            return {"params": {"ts_code": "001728.OF"}}
        def table(self, sha, **kwargs):
            return [] if sha == "div" else [dict(nav_date=day, ann_date=day, unit_nav=str(1 + n / 100))
                                             for n, day in enumerate(days) if day != days[0]]
    with pytest.raises(ValueError, match="EXACT_RETURN_ENDPOINT_MISSING"):
        licensed_share_returns(Source(), "nav", "div", "001728.OF", days, days)
    available = licensed_share_returns(Source(), "nav", "div", "001728.OF", days, days,
                                      selected_horizons=("return_1y",))
    assert available["return_1y"] is not None
    assert available["return_3y"] is None
    assert days[0] not in available["endpoint_nav"]


def test_thematic_fee_refs_survive_canonical_json_key_reordering(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from app.services import fund_rating_thematic_complete as module
    from app.services import fund_rating_payment_domain

    empty = lambda *args, **kwargs: ({}, {})
    components = {
        "performance": SimpleNamespace(compile_performance=empty),
        "structure": SimpleNamespace(compile_structure=empty),
        "execution": SimpleNamespace(compile_execution=empty),
        "admission": SimpleNamespace(raw_admission=lambda *args: {}),
        "material": SimpleNamespace(compile_material_review=lambda *args: {
            "events": [], "anchors": {"material_events": {"018590": []}}, "material_event_pending": False}),
    }
    monkeypatch.setattr(module, "component", components.__getitem__)
    monkeypatch.setattr(module, "management_from_originals", empty)
    monkeypatch.setattr(module, "holders_from_primary", empty)
    monkeypatch.setattr(module, "fee_profile", lambda *args, **kwargs: {"fees": {}})
    monkeypatch.setattr(module, "_section", lambda *args: "净资产合计 100.00")
    monkeypatch.setattr(fund_rating_payment_domain, "compile_payment_from_primary", lambda *args, **kwargs: {"node": {}})
    class Source:
        used = set()
        sources = {"z-summary": {"published_on": "2026-02-06"}, "a-prospectus": {}}
        def original(self, sha):
            self.used.add(sha)
            return self.sources[sha]
        def body(self, sha):
            self.used.add(sha)
            return "原件正文供独立费用解析器处理"
    inputs = dict(version=module.VERSION, time_contract=module.TIME_CONTRACT, fund_code="018590", share_class="C",
                  family="ACTIVE_EQUITY", basis="PUBLIC_REVIEWED_ACTIVE_QUALITY_V2", holding_date="2026-06-30",
                  as_of_date="2026-10-30", research_on="2026-11-17", primary_ref="primary", legal_name="基金法定名称",
                  fee={"summary_ref": "z-summary", "prospectus_ref": "a-prospectus"})
    first = module._compile_observations_uncached(inputs, source=Source(), root=tmp_path, diagnostics=True)
    roundtrip = json.loads(json.dumps(inputs, sort_keys=True))
    second = module._compile_observations_uncached(roundtrip, source=Source(), root=tmp_path, diagnostics=True)
    assert first["observations"]["cost_365"] == second["observations"]["cost_365"]


def test_old_thematic_snapshot_accepts_only_fee_ref_order_difference(monkeypatch):
    from copy import deepcopy
    from app.services import fund_rating_thematic_complete as module

    expected = dict(unit_contract="RATIO_RANKS_RATIO_WINDOWS", classification={"product_id": "own"},
                    quality_public={"dataDates": {"nav": "2026-09-30"}},
                    quality_metrics={"observations": {"cost_365": {
                        "kind": "standard_cost", "fee_inputs": {"rate": ".96"}, "source_refs": ["a", "b"]}}})
    packet = deepcopy(expected)
    packet["thematic_complete_compilation"] = {}
    packet["quality_metrics"]["observations"]["cost_365"]["source_refs"] = ["b", "a"]
    monkeypatch.setattr(module, "build_packet", lambda *args: (expected, {}))
    assert module.check_compiled_binding(packet, {}, {}) == {}
    packet["quality_metrics"]["observations"]["cost_365"]["fee_inputs"]["rate"] = ".97"
    with pytest.raises(ValueError, match="ORIGINAL_OBSERVATIONS_CONFLICT"):
        module.check_compiled_binding(packet, {}, {})


def test_explicit_c_subscription_numeric_zero_is_not_nonzero_or_a_share():
    from app.services.fund_rating_market_review import fee_profile

    text = ("隔离基金C123451送出日期2026年6月26日三、投资本基金涉及的费用赎回费N≥7天0%"
            "(二)基金运作综合费率(年化)0.96%")
    support = "C类基金份额的申购费为0。"
    assert fee_profile(text, code="123451", share_class="C", legal_name="隔离基金", supporting_text=support)["fees"]["subscription"] == "0"
    for bad in [support.replace("C类", "A类"), support.replace("为0", "为0.5%")]:
        with pytest.raises(ValueError, match="免申购费"):
            fee_profile(text, code="123451", share_class="C", legal_name="隔离基金", supporting_text=bad)
    terminal_cell = text.replace("赎回费N≥7天0%", "赎回费N≥30日-0-")
    assert fee_profile(terminal_cell, code="123451", share_class="C", legal_name="隔离基金", supporting_text=support)["fees"]["redemption"] == "0"
    with pytest.raises(ValueError, match="365天"):
        fee_profile(terminal_cell.replace("30日-0-", "30日-0.5%-"), code="123451", share_class="C",
                    legal_name="隔离基金", supporting_text=support)


def test_active_r5_partial_requires_own_original_binding(tmp_path):
    from app.services.fund_rating_r5 import calculate

    with pytest.raises(ValueError, match="ACTIVE_PARTIAL_ORIGINAL_SCOPE_REQUIRED"):
        calculate({}, {}, root=tmp_path, selected_dimensions={"management"})


def test_active_partial_replay_rejects_invented_missing_dimension(monkeypatch, tmp_path):
    from app.services import fund_rating_active_r5_partial as module
    from app.services.fund_rating_reviewed_snapshot import KEYS

    monkeypatch.setattr(module, "validate_bundle", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "calculate_inputs", lambda *args, **kwargs: {
        "dimensions": {"management": "55", "cost": "76"}, "score": None, "raw_score": None})
    bundle = {"active_r5_partial_calculation": {"inputs": {}}}
    dimensions = {key: {"lower": "0", "upper": "100", "candidate": None} for key in KEYS}
    for key, point in [("management", "55"), ("fee", "76")]:
        dimensions[key] = {"lower": point, "upper": point, "candidate": point}
    module.replay_snapshot(bundle, dimensions, root=tmp_path)
    dimensions["return"] = {"lower": "80", "upper": "80", "candidate": "80"}
    with pytest.raises(ValueError, match="MISSING_DIMENSION_NOT_NULL"):
        module.replay_snapshot(bundle, dimensions, root=tmp_path)


def test_active_partial_snapshot_never_rescales_two_dimensions_into_total(monkeypatch, tmp_path):
    from app.services import fund_rating_active_r5_partial as module
    from app.services.fund_rating_reviewed_snapshot import calculate, KEYS

    monkeypatch.setattr(module, "replay_snapshot", lambda *args, **kwargs: None)
    weights = dict(zip(KEYS, [15, 20, 10, 10, 15, 10, 10, 10]))
    document = dict(code="002112", as_of="2026-10-30", outer_weights_percent=weights,
                    dimensions={key: None for key in KEYS})
    document["dimensions"].update(management="55", fee="76")
    value = calculate(dict(fund_code="002112", as_of_date="2026-10-30", reviewed_document=document,
                           active_r5_partial_calculation={"inputs": {}}), root=tmp_path)
    assert value["status"] == "PARTIAL" and value["point_dimensions"] == 2
    assert value["score"] is None and value["partial_score"] is None
    assert value["missing_weight"] == "75"
