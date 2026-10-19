"""可观察原件事实不替代其他含义的必要评分输入；合成算例不用于产品校准。"""

from app.services.fund_quality_v2_materials import apply_extraction
from app.services.fund_quality_v2_observable_liquidity import observable_liquidity_facts


def test_identified_basket_shortfall_and_reported_no_alert_cannot_fill_target_or_capacity():
    packet = {"as_of_date": "2026-09-30", "metrics": {}, "kind": "ACTIVE_EQUITY",
              "inventory": {"cash_target_deviation": {"state": "METHOD_REFERENCE_REQUIRED"},
                            "capacity_pressure": {"state": "METHOD_REFERENCE_REQUIRED"}}}
    extracted = {"metrics": {}, "findings": {}, "document_errors": {}, "facts": {
        "cash_contract_basket_observation": {"identified_shortfall_ratio": "0"},
        "operating_report_observation": {"state": "SUMMARY_NO_APPLICABLE_ALERT_DISCLOSED",
                                         "criteria_triggered": {"holders": None, "assets": None}},
    }}
    apply_extraction(packet, extracted)
    assert not packet["metrics"]
    assert all(item["state"] == "METHOD_REFERENCE_REQUIRED" for item in packet["inventory"].values())


def test_one_parser_failure_preserves_the_other_original_fact_and_its_unknowns(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("TEST_ORIGINAL_FORMAT_UNRESOLVED")

    monkeypatch.setattr("app.services.fund_quality_v2_cash_constraints.rebuild_cash_constraint", fail)
    monkeypatch.setattr("app.services.fund_quality_v2_operating_disclosure.operating_disclosure",
                        lambda *args, **kwargs: {"state": "SUMMARY_NO_APPLICABLE_ALERT_DISCLOSED",
                                                 "criteria_triggered": {"holders": None, "assets": None}})
    value = observable_liquidity_facts(["TEST_ONLY"], observed_on="2026-06-30", restricted_metric=None,
                                      source_ref={"sha256": "a" * 64, "url": "https://example.test/original"})
    assert value["cash_contract_basket_observation_gap"]["reason"] == "TEST_ORIGINAL_FORMAT_UNRESOLVED"
    assert value["operating_report_observation"]["criteria_triggered"] == {"holders": None, "assets": None}
    assert not value["operating_report_observation"]["raw4_metric_replaced"]
