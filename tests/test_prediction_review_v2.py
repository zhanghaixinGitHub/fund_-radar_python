"""第二版试验的错误边界；只用D盘隔离小样本，不访问真实服务或业务库。"""

import json
from copy import deepcopy
from datetime import datetime

import pytest
from app.services.direction_1d_protocol import digest
from app.services.prediction_review_v2 import (
    Classification,
    catalog_v2,
    policy_v2,
    required_checks,
    validate_review_v2,
)
from pydantic import ValidationError

AT = datetime.fromisoformat("2026-11-12T11:30:00+08:00")
NOTE = "本例已逐项检查实际来源和适用范围，未知内容不会当作已知事实。"


def sample(horizon="T1"):
    body = "测试基金为股票型基金，采用行业指数跟踪策略。主要投资于已披露的相关行业证券。"
    data = {
        "fund_code": "001234",
        "facts": {
            "up": {"category": "公司经营", "text": "公开订单开始交付，但后续兑现仍须检查。"},
            "down": {"category": "市场行情", "text": "相关行业价格走弱，可能抵消经营改善。"},
        },
        "documents": [
            {
                "id": "own",
                "kind": "REPORT",
                "codes": ["001234"],
                "body": body,
                "published_date": "2026-11-10",
                "available_at": "2026-11-11T10:00:00+08:00",
            }
        ],
    }
    packet = {
        "fund_code": "001234",
        "profile": {"fund_type": "STOCK"},
        "captured_at": AT.isoformat(),
        "source_cutoff": "2026-11-12T01:00:00+08:00",
        "source_ready": True,
        "periods": {h: {"status": "RESOLVED"} for h in policy_v2()["horizons"]},
    }
    classification = {
        "base_type": "SECTOR_INDEX",
        "overlays": [],
        "proofs": [{"kind": "DOCUMENT", "ref": "own", "text": body}],
        "explanation": NOTE,
    }
    factor = {
        "id": "delivery",
        "event_group": "delivery",
        "role": "SUPPORT_UP",
        "importance": "MAIN",
        "quotes": [{"kind": "FACT", "ref": "up", "text": data["facts"]["up"]["text"]}],
        "fund_link": NOTE,
        "coverage": "MAJOR",
        "coverage_note": NOTE,
        "stage": "IN_PROGRESS",
        "novelty": "CHANGED",
        "novelty_note": NOTE,
        "priced": "UNKNOWN",
        "pricing_note": NOTE,
        "time_bucket": "FUTURE",
        "timing": "MATCHED",
        "mechanism": NOTE,
        "window_path": NOTE,
        "invalidation": NOTE,
        "driver_kind": "OPERATING" if horizon == "M6_V1" else "EVENT",
    }
    counter = {
        **factor,
        "id": "pricing",
        "event_group": "industry",
        "role": "SUPPORT_DOWN",
        "importance": "AUXILIARY",
        "quotes": [{"kind": "FACT", "ref": "down", "text": data["facts"]["down"]["text"]}],
    }
    r = {
        "version": policy_v2()["version"],
        "policy_hash": digest(policy_v2()),
        "packet_hash": digest(packet),
        "fund_code": "001234",
        "horizon": horizon,
        "classification": classification,
        "reading": [
            {"kind": kind, "ref": ref, "start": 0, "end": len(text)} for (kind, ref), text in catalog_v2(data).items()
        ],
        "checks": {
            key: {"status": "SUPPORTED", "refs": ["up"], "note": NOTE, "material_to_direction": False}
            for key in required_checks(Classification.model_validate(classification), horizon)
        },
        "factors": [factor, counter],
        "decision": "UP",
        "summary": NOTE,
        "dominant_factors": ["delivery"],
        "strongest_counter": ["pricing"],
        "counter_search_result": "FOUND",
        "counter_search_scope": NOTE,
        "tradeoff": NOTE,
        "sensitivity": NOTE,
        "sensitive_to_unknowns": False,
        "sensitivity_cases": [
            {
                "condition": NOTE,
                "plausibility": "PLAUSIBLE",
                "direction_effect": "WEAKER",
                "refs": ["down"],
                "explanation": NOTE,
            }
        ],
        "change_conditions": [NOTE],
        "abstention_reasons": [],
        "recovery_conditions": [],
        "scenarios": [],
        "cross_horizon": NOTE,
        "limitations": [NOTE],
        "period_assessment": {
            "rule_id": policy_v2()["horizons"][horizon]["id"],
            "primary_driver": NOTE,
            "window_reason": policy_v2()["horizons"][horizon]["label"]
            + "使用本例明确的兑现时间分析，不能换成其他周期的判断。",
            "counter_tradeoff": NOTE,
            "missing_effect": NOTE,
        },
    }
    if horizon == "M6_V1":
        r["scenarios"] = [
            {"kind": kind, "refs": ["up"], "assumptions": NOTE, "outcome_path": NOTE, "verification": NOTE}
            for kind in ("BASE", "FAVORABLE", "ADVERSE")
        ]
    return packet, data, r


@pytest.mark.parametrize("horizon", ["T1", "T5_V1", "T20_V1", "M6_V1"])
def test_valid_horizon_and_type_specific_review(horizon):
    packet, data, r = sample(horizon)
    assert validate_review_v2(packet, data, r, at=AT).horizon == horizon


@pytest.mark.parametrize("field", ["version", "policy_hash", "packet_hash", "fund_code"])
def test_review_identity_cannot_be_relabelled(field):
    packet, data, r = sample()
    r[field] = "009999" if field == "fund_code" else "different"
    with pytest.raises(ValueError, match="V2_REVIEW_IDENTITY_MISMATCH"):
        validate_review_v2(packet, data, r)


def mutate(r, packet, data, change):
    if change == "missing_horizon_check":
        del r["checks"]["next_session_driver"]
    elif change == "wrong_horizon":
        r["horizon"] = "T20_V1"
    elif change == "unread_quote":
        r["reading"] = [row for row in r["reading"] if row["ref"] != "up"]
    elif change == "altered_quote":
        r["factors"][0]["quotes"][0]["text"] = "这是不存在于真实来源之中的改写句子。"
    elif change == "other_fund_proof":
        data["documents"][0]["codes"] = ["009999"]
    elif change == "unproved_asset":
        r["classification"]["base_type"] = "GOLD"
    elif change == "future_source":
        data["documents"][0]["available_at"] = "2026-11-12T12:00:00+08:00"
    elif change == "future_packet":
        packet["captured_at"] = "2026-11-12T12:00:00+08:00"
        r["packet_hash"] = digest(packet)
    elif change == "realized":
        r["factors"][0]["time_bucket"] = "REALIZED_NOT_IN_NAV"
    elif change == "unknown_exposure":
        r["factors"][0]["coverage"] = "UNKNOWN"
    elif change == "fully_priced":
        r["factors"][0]["priced"] = "FULLY_REFLECTED"
    elif change == "material_gap":
        r["checks"]["pricing"].update(
            status="LIMITED", material_to_direction=True, gap_effect="BLOCK", impact_reason=NOTE, follow_up=NOTE
        )
    elif change == "sensitive":
        r["sensitive_to_unknowns"] = True
        r["sensitivity_cases"][0]["direction_effect"] = "REVERSES"
    elif change == "no_counter":
        r["strongest_counter"] = []
    elif change == "same_counter":
        r["strongest_counter"] = ["delivery"]
    elif change == "flat":
        r["decision"] = "FLAT"
    elif change == "unverified_target":
        packet["periods"]["T1"]["status"] = "UNVERIFIED"
        r["packet_hash"] = digest(packet)
    elif change == "nav_only":
        data["facts"]["up"]["category"] = "基金净值"


@pytest.mark.parametrize(
    "change,error",
    [
        ("missing_horizon_check", "V2_REQUIRED_CHECKS_MISMATCH"),
        ("wrong_horizon", "V2_PERIOD_ASSESSMENT_RULE_MISMATCH"),
        ("unread_quote", "V2_UNREAD_CHECK_REFERENCE"),
        ("altered_quote", "V2_QUOTE_OUTSIDE_READING"),
        ("other_fund_proof", "V2_CLASSIFICATION_OWN_DOCUMENT_REQUIRED"),
        ("unproved_asset", "V2_CLASSIFICATION_EVIDENCE_REQUIRED"),
        ("future_source", "V2_SOURCE_AFTER_CUTOFF"),
        ("future_packet", "V2_FUTURE_PACKET"),
        ("realized", "V2_FUTURE_MAIN_DRIVER_REQUIRED"),
        ("unknown_exposure", "V2_FUTURE_MAIN_DRIVER_REQUIRED"),
        ("fully_priced", "V2_FUTURE_MAIN_DRIVER_REQUIRED"),
        ("material_gap", "V2_MATERIAL_GAP_REQUIRES_ABSTENTION"),
        ("sensitive", "V2_UNSTABLE_DIRECTION"),
        ("no_counter", "V2_INDEPENDENT_COUNTER_REQUIRED"),
        ("same_counter", "V2_COUNTER_NOT_INDEPENDENT"),
        ("flat", "V2_FLAT_BAND_UNREGISTERED"),
        ("unverified_target", "V2_TARGET_OR_SOURCE_UNVERIFIED"),
        ("nav_only", "V2_NAV_ONLY_DIRECTION_DISABLED"),
    ],
)
def test_invalid_review_fails_before_save(change, error):
    packet, data, r = sample()
    mutate(r, packet, data, change)
    with pytest.raises(ValueError, match=error):
        validate_review_v2(packet, data, r, at=AT)


def test_abstention_retains_real_analysis_and_recovery():
    packet, data, r = sample()
    r.update(
        decision="ABSTAIN", dominant_factors=[], abstention_reasons=["PRICING_UNCLEAR"], recovery_conditions=[NOTE]
    )
    r["checks"]["pricing"].update(
        status="LIMITED", material_to_direction=True, gap_effect="BLOCK", impact_reason=NOTE, follow_up=NOTE
    )
    assert validate_review_v2(packet, data, r).decision == "ABSTAIN"
    r["recovery_conditions"] = []
    with pytest.raises(ValueError, match="V2_ABSTENTION_RECOVERY_REQUIRED"):
        validate_review_v2(packet, data, r)


def test_m6_requires_three_distinct_grounded_scenarios_even_when_abstaining():
    packet, data, r = sample("M6_V1")
    r.update(
        decision="ABSTAIN",
        dominant_factors=[],
        abstention_reasons=["DATA_MISSING"],
        recovery_conditions=[NOTE],
        scenarios=[],
    )
    with pytest.raises(ValueError, match="V2_M6_SCENARIOS_REQUIRED"):
        validate_review_v2(packet, data, r)


def test_extra_fields_and_unfinished_short_template_are_rejected():
    packet, data, r = sample()
    r["guessed_probability"] = 0.8
    with pytest.raises(ValidationError):
        validate_review_v2(packet, data, r)
    del r["guessed_probability"]
    r["tradeoff"] = "综合判断"
    with pytest.raises(ValidationError):
        validate_review_v2(packet, data, r)


def test_v1_policy_is_unchanged_and_cannot_accept_v2_trial():
    from app.services.comprehensive_prediction_policy import policy, validate_review

    assert policy()["version"] == "FUND_COMPREHENSIVE_REVIEW_V1"
    _, _, review = sample()
    review["analysis_policy_version"] = policy()["version"]
    with pytest.raises(ValueError, match="EXPERIMENTAL_V2_NOT_PRODUCTION_REVIEW"):
        validate_review({}, review, {})


@pytest.mark.parametrize(
    "change,error",
    [
        ("waiver", "V2_SELECTED_CHECK_CANNOT_BE_WAIVED"),
        ("ineligible", "V2_INELIGIBLE_DIRECTION_FACT"),
        ("same_event", "V2_REPEATED_EVENT_AS_MAIN_DRIVER"),
        ("copied_counter", "V2_COUNTER_COPIED_FROM_SUPPORT"),
        ("empty_search", "V2_COUNTER_SEARCH_RECEIPT_REQUIRED"),
    ],
)
def test_evidence_cannot_be_waived_or_repeated(change, error):
    packet, data, r = sample()
    if change == "waiver":
        r["checks"]["exposure"]["status"] = "NOT_APPLICABLE"
    elif change == "ineligible":
        data["facts"]["up"]["direction_eligible"] = False
    elif change == "same_event":
        repeated = deepcopy(r["factors"][0])
        repeated["id"] = "same_delivery"
        r["factors"].append(repeated)
        r["dominant_factors"].append(repeated["id"])
    elif change == "copied_counter":
        r["factors"][1]["quotes"] = deepcopy(r["factors"][0]["quotes"])
    else:
        r["counter_search_result"] = "NO_VERIFIABLE_COUNTER"
        r["strongest_counter"] = []
    with pytest.raises(ValueError, match=error):
        validate_review_v2(packet, data, r)


def test_no_counter_found_needs_actual_read_search_receipt():
    packet, data, r = sample()
    doc = {
        "id": "search",
        "kind": "DISCLOSURE_SEARCH",
        "codes": ["001234"],
        "body": "本例已检查列明日期及目录范围，没有新增可以核实的反向事项。",
        "published_date": "2026-11-11",
        "available_at": "2026-11-11T21:00:00+08:00",
    }
    data["documents"].append(doc)
    r["reading"].append({"kind": "DOCUMENT", "ref": "search", "start": 0, "end": len(doc["body"])})
    r["counter_search_result"] = "NO_VERIFIABLE_COUNTER"
    r["strongest_counter"] = []
    r["checks"]["counter_search"]["refs"] = ["search"]
    assert validate_review_v2(packet, data, r).decision == "UP"


def test_classification_cannot_change_to_skip_horizon_requirements():
    from app.services.prediction_review_v2 import validate_cross_horizons

    a, b, first = sample()
    one = validate_review_v2(a, b, first)
    a, b, second = sample("T5_V1")
    second["classification"]["overlays"] = ["PASSIVE"]
    second["checks"]["tracking_risk"] = {
        "status": "SUPPORTED",
        "refs": ["own"],
        "note": NOTE,
        "material_to_direction": False,
    }
    five = validate_review_v2(a, b, second)
    with pytest.raises(ValueError, match="V2_CLASSIFICATION_CHANGED_BETWEEN_HORIZONS"):
        validate_cross_horizons([one, five])


@pytest.mark.parametrize(
    "problem,error",
    [
        ("wrong_profile", "V2_CLASSIFICATION_PROFILE_CONFLICT"),
        ("missing_feeder", "V2_REQUIRED_OVERLAY_OMITTED"),
        ("missing_overseas", "V2_REQUIRED_OVERLAY_OMITTED"),
        ("missing_enhanced", "V2_REQUIRED_OVERLAY_OMITTED"),
    ],
)
def test_official_profile_prevents_skipping_type_rules(problem, error):
    packet, data, r = sample()
    if problem == "wrong_profile":
        packet["profile"]["fund_type"] = "BOND"
    elif problem == "missing_feeder":
        packet["profile"]["fund_name"] = "本例行业ETF联接基金"
    elif problem == "missing_overseas":
        packet["profile"]["fund_name"] = "本例境外股票(QDII)"
    else:
        packet["profile"]["invest_type"] = "增强指数型"
    r["packet_hash"] = digest(packet)
    with pytest.raises(ValueError, match=error):
        validate_review_v2(packet, data, r)


@pytest.mark.parametrize("with_outlook", [False, True])
def test_trial_current_scope_immutable_readback_and_repeat_time(tmp_path, monkeypatch, with_outlook):
    from app.services import prediction_trial_v2 as trial

    monkeypatch.setattr(trial, "PROJECT", tmp_path)
    packet, data, _ = sample()
    data.update(
        as_of=packet["source_cutoff"],
        window={"base_nav_date": "2026-11-11", "target_nav_date": "2026-11-12"},
        time_context={"baseline_complete": True},
    )
    source = {"format": "CODEX_FUND_INPUT_V1", "input": data, "input_hash": digest(data)}
    source_path = tmp_path / ".local-runs/source/T1/input.json"
    memory = {}

    def fake_save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test", encoding="utf-8")
        memory[str(path)] = deepcopy(value)

    monkeypatch.setattr(trial, "save", fake_save)
    monkeypatch.setattr(trial, "read", lambda path: deepcopy(memory[str(path)]))
    fake_save(source_path, source)
    half_year_dates = {
        "horizonId": "M6_V1",
        "baseNavDate": "2026-11-11",
        "startDate": "2026-11-12",
        "endDate": None,
        "nominalEndDate": "2027-05-11",
        "endDateStatus": "PENDING_OFFICIAL_CALENDAR",
        "calendarId": "verified-2026-only",
        "calendarHash": "verified-calendar-hash",
    }
    half_year_input = {"data": data, "dates": half_year_dates}
    fake_save(
        source_path.parent.parent / "M6_V1/input.json",
        {
            "format": "CODEX_COMPREHENSIVE_MULTI_V1",
            "input": half_year_input,
            "input_hash": digest(half_year_input),
        },
    )
    inventory = {
        "owner_hash": "owner",
        "scope": ["001234"],
        "profiles": [{"fund_code": "001234", "fund_type": "STOCK"}],
    }
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            [
                {
                    "fund_code": "001234",
                    "source_kind": "FROZEN",
                    "source_path": str(source_path),
                    "sample_label": "行业指数",
                    "reason": NOTE,
                }
            ]
        ),
        encoding="utf-8",
    )
    run = tmp_path / ".local-runs/trial"
    manifest = trial.prepare_trial(run, selection, holdings=inventory, clock=AT)
    assert manifest["policy_snapshot"] == policy_v2()
    frozen = memory[str(run / "inputs/001234.json")]
    assert frozen["periods"]["M6_V1"]["nominal_end_date"] == "2027-05-11"
    assert frozen["periods"]["M6_V1"]["end_date"] is None
    assert frozen["periods"]["M6_V1"]["status"] == "PENDING_OFFICIAL_CALENDAR"
    reviews = []
    for h in trial.HORIZONS:
        _, _, r = sample(h)
        r["packet_hash"] = digest(frozen)
        if h == "M6_V1" and with_outlook:
            r["output_kind"] = "CONDITIONAL_OUTLOOK"
        else:
            r.update(
                decision="ABSTAIN", dominant_factors=[], abstention_reasons=["DATA_MISSING"], recovery_conditions=[NOTE]
            )
        reviews.append(r)
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps(reviews), encoding="utf-8")
    result = trial.evaluate_trial(run, path, holdings_provider=lambda: inventory, clock=AT)
    assert result["counts"]["ABSTAIN"] == (3 if with_outlook else 4)
    assert result["dated_direction_count"] == 0
    assert result["conditional_direction_count"] == (1 if with_outlook else 0)
    repeated = trial.evaluate_trial(
        run, path, holdings_provider=lambda: inventory, clock=datetime.fromisoformat("2026-11-12T12:00:00+08:00")
    )
    assert result["results"] == repeated["results"]
    first = memory[result["results"][0]["result_path"]]
    assert first["evaluated_at"] == AT.isoformat()
    assert first["production_publication"] is False
    with pytest.raises(ValueError, match="V2_CURRENT_HOLDING_OR_OWNER_CHANGED"):
        trial.evaluate_trial(run, path, holdings_provider=lambda: {**inventory, "scope": []})
    memory[str(source_path)]["input"]["fund_code"] = "009999"
    with pytest.raises(ValueError, match="V2_ORIGINAL_PACKET_INVALID"):
        trial.checked_input(run, manifest["selection"][0])


def test_noncritical_gap_keeps_direction_and_followup():
    packet, data, r = sample()
    r["checks"]["exposure"].update(status="LIMITED", gap_effect="QUALIFY", impact_reason=NOTE, follow_up=NOTE)
    checked = validate_review_v2(packet, data, r)
    assert checked.decision == "UP"
    assert checked.checks["exposure"].follow_up == NOTE
    del r["checks"]["exposure"]["impact_reason"]
    with pytest.raises(ValueError, match="V2_GAP_IMPACT_AND_FOLLOWUP_REQUIRED"):
        validate_review_v2(packet, data, r)


def test_representative_partial_portfolio_requires_explicit_basis():
    packet, data, r = sample()
    f = r["factors"][0]
    f.update(coverage="PARTIAL", coverage_basis="REPORTED_PORTFOLIO", coverage_refs=["own"])
    assert validate_review_v2(packet, data, r).decision == "UP"
    f["coverage_basis"] = "UNKNOWN"
    with pytest.raises(ValueError, match="V2_FUTURE_MAIN_DRIVER_REQUIRED"):
        validate_review_v2(packet, data, r)


@pytest.mark.parametrize("category,allowed", [("公司经营", True), ("市场行情", False), ("基金净值", False)])
def test_ongoing_operating_effect_allowed_but_yesterday_price_not_enough(category, allowed):
    packet, data, r = sample("T20_V1")
    data["facts"]["up"]["category"] = category
    r["factors"][0].update(
        time_bucket="ONGOING_EFFECT",
        novelty="BACKGROUND",
        continuation_explanation=NOTE,
        continuation_quotes=deepcopy(r["factors"][0]["quotes"]),
    )
    if allowed:
        assert validate_review_v2(packet, data, r).decision == "UP"
    else:
        with pytest.raises(ValueError, match="V2_ONGOING_EFFECT_EVIDENCE_REQUIRED"):
            validate_review_v2(packet, data, r)


def test_future_adverse_risk_does_not_equal_missing_current_evidence():
    packet, data, r = sample()
    risk = deepcopy(r["sensitivity_cases"][0])
    risk.update(kind="FUTURE_RISK", direction_effect="REVERSES")
    r["sensitivity_cases"].append(risk)
    assert validate_review_v2(packet, data, r).decision == "UP"
    risk["kind"] = "EVIDENCE_GAP"
    with pytest.raises(ValueError, match="V2_SENSITIVITY_FLAG_CONFLICT"):
        validate_review_v2(packet, data, r)
    r["sensitive_to_unknowns"] = True
    with pytest.raises(ValueError, match="V2_UNSTABLE_DIRECTION"):
        validate_review_v2(packet, data, r)


def test_future_risk_cannot_replace_missing_evidence_review():
    packet, data, r = sample()
    r["sensitivity_cases"][0]["kind"] = "FUTURE_RISK"
    with pytest.raises(ValueError, match="V2_CONCRETE_SENSITIVITY_CASE_REQUIRED"):
        validate_review_v2(packet, data, r)


def test_unresolved_half_year_can_save_separately_labelled_outlook():
    packet, data, r = sample("M6_V1")
    packet["periods"]["M6_V1"]["status"] = "UNVERIFIED"
    r["packet_hash"] = digest(packet)
    with pytest.raises(ValueError, match="V2_TARGET_OR_SOURCE_UNVERIFIED"):
        validate_review_v2(packet, data, r)
    r["output_kind"] = "CONDITIONAL_OUTLOOK"
    assert validate_review_v2(packet, data, r).output_kind == "CONDITIONAL_OUTLOOK"
    packet["source_ready"] = False
    r["packet_hash"] = digest(packet)
    with pytest.raises(ValueError, match="V2_TARGET_OR_SOURCE_UNVERIFIED"):
        validate_review_v2(packet, data, r)


def test_event_can_be_inapplicable_only_with_read_product_evidence():
    packet, data, r = sample()
    r["checks"]["next_session_events"].update(status="NOT_APPLICABLE", refs=["own"], impact_reason=NOTE)
    assert validate_review_v2(packet, data, r).decision == "UP"
    r["checks"]["next_session_events"]["refs"] = ["up"]
    with pytest.raises(ValueError, match="V2_NOT_APPLICABLE_EVIDENCE_REQUIRED"):
        validate_review_v2(packet, data, r)


@pytest.mark.parametrize("base_type", ["BROAD_INDEX", "ACTIVE_EQUITY", "SECTOR_INDEX", "MIXED"])
def test_short_period_does_not_inherit_long_term_earnings_gate(base_type):
    classification = Classification.model_validate({**sample()[2]["classification"], "base_type": base_type})
    for horizon in ["T1", "T5_V1"]:
        checks = required_checks(classification, horizon)
        assert "earnings_valuation" not in checks
        assert "sustainable_income" not in checks
        assert "valuation_assumptions" not in checks
    assert (
        policy_v2()["horizons"]["T1"]["asset_checks"][base_type]
        != policy_v2()["horizons"]["T20_V1"]["asset_checks"][base_type]
    )


def test_long_term_driver_cannot_be_relabelled_as_next_day_driver():
    packet, data, r = sample()
    r["factors"][0]["driver_kind"] = "OPERATING"
    with pytest.raises(ValueError, match="V2_DRIVER_NOT_SUITABLE_FOR_PERIOD"):
        validate_review_v2(packet, data, r)


def price_sample(horizon="T5_V1"):
    packet, data, r = sample(horizon)
    text = "2026-11-10相关行业价格为100；2026-11-11相关行业价格为102，多日变化仍需核实原因。"
    data["facts"]["up"].update(category="市场行情", text=text)
    r["reading"] = [{"kind": k, "ref": ref, "start": 0, "end": len(t)} for (k, ref), t in catalog_v2(data).items()]
    q = {"kind": "FACT", "ref": "up", "text": text}
    r["factors"][0].update(
        driver_kind="PRICE_CONTINUATION",
        time_bucket="ONGOING_EFFECT",
        quotes=[q],
        continuation_quotes=[q],
        continuation_explanation=NOTE,
        price_observations=[{"observed_on": day, "quote": q} for day in ["2026-11-10", "2026-11-11"]],
    )
    packet["periods"][horizon]["base_date"] = "2026-11-11"
    r["packet_hash"] = digest(packet)
    return packet, data, r


@pytest.mark.parametrize("horizon", ["T1", "T5_V1"])
def test_short_period_allows_explained_multiday_market_driver(horizon):
    packet, data, r = price_sample(horizon)
    assert validate_review_v2(packet, data, r).decision == "UP"


@pytest.mark.parametrize("problem", ["one_day", "fake_date", "financial_as_price", "no_base", "duplicate_day"])
def test_price_continuation_still_requires_real_multiday_evidence(problem):
    packet, data, r = price_sample()
    observations = r["factors"][0]["price_observations"]
    if problem == "one_day":
        observations.pop(0)
    elif problem == "fake_date":
        observations[0]["observed_on"] = "2026-11-09"
    elif problem == "financial_as_price":
        data["facts"]["up"]["category"] = "财务"
    elif problem == "no_base":
        del packet["periods"]["T5_V1"]["base_date"]
    else:
        observations[0]["observed_on"] = "2026-11-11"
    r["packet_hash"] = digest(packet)
    with pytest.raises(
        ValueError, match="V2_(MULTIDAY_PRICE_EVIDENCE_REQUIRED|PRICE_OBSERVATION_SOURCE_OR_DATE_INVALID)"
    ):
        validate_review_v2(packet, data, r)


def test_market_trend_cannot_replace_medium_term_basis():
    packet, data, r = price_sample("T20_V1")
    with pytest.raises(ValueError, match="V2_DRIVER_NOT_SUITABLE_FOR_PERIOD"):
        validate_review_v2(packet, data, r)


def test_four_different_rule_ids_do_not_allow_copying_the_same_reason():
    from app.services.prediction_review_v2 import validate_cross_horizons

    packet, data, first = sample()
    one = validate_review_v2(packet, data, first)
    packet, data, second = sample("T5_V1")
    second["period_assessment"] = {**first["period_assessment"], "rule_id": policy_v2()["horizons"]["T5_V1"]["id"]}
    five = validate_review_v2(packet, data, second)
    with pytest.raises(ValueError, match="V2_COPIED_PERIOD_ASSESSMENT"):
        validate_cross_horizons([one, five])


@pytest.mark.parametrize("kind,allowed", [("COMPANY_FINANCIAL", True), ("REPORT", False)])
def test_ongoing_effect_can_use_direct_financial_original_without_generated_fact(kind, allowed):
    packet, data, r = sample("T20_V1")
    text = "公司披露本季度真实收入与交付增加，但后续持续性仍需检验。"
    data["documents"].append({"id": "financial", "kind": kind, "body": text})
    r["reading"].append({"kind": "DOCUMENT", "ref": "financial", "start": 0, "end": len(text)})
    q = {"kind": "DOCUMENT", "ref": "financial", "text": text}
    r["factors"][0].update(
        driver_kind="OPERATING",
        time_bucket="ONGOING_EFFECT",
        quotes=[q],
        continuation_quotes=[q],
        continuation_explanation=NOTE,
    )
    if allowed:
        assert validate_review_v2(packet, data, r).decision == "UP"
    else:
        with pytest.raises(ValueError, match="V2_ONGOING_EFFECT_EVIDENCE_REQUIRED"):
            validate_review_v2(packet, data, r)
