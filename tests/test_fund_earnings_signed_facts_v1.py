"""验证亏损原文字段的正负号、单位与相邻字段隔离。"""

import copy

import pytest
from app.services.fund_earnings_signed_facts_v1 import review_loss_claim


def example():
    return {
        "issuer": "002714",
        "period_start": "2022-01-01",
        "period_end": "2022-06-30",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_ATTRIBUTABLE_TO_PARENT",
        "currency": "CNY",
        "unit": "亿元",
        "values": ["63.00", "69.00"],
        "kind": "FORECAST",
        "published_date": "2022-07-15",
        "document_id": "test-only",
        "page": 1,
        "quote": "归母净利润 亏损：63.00亿元—69.00亿元 盈利：95.26亿元",
        "loss_anchor": "亏损：63.00亿元—69.00亿元",
    }


def test_loss_preserves_source_and_orders_signed_interval():
    spec = example()
    original = copy.deepcopy(spec)
    result = review_loss_claim(spec, [spec["quote"]])
    assert result["signed_interval_cny"] == ["-6900000000.00", "-6300000000.00"]
    assert [m["value"] for m in result["money"]] == ["-69.00", "-63.00"]
    assert [m["cny"] for m in result["reported_magnitudes"]] == ["6300000000.00", "6900000000.00"]
    assert spec == original
    assert result["available_at"] == "2022-07-16T08:00:00+08:00"


@pytest.mark.parametrize(
    "anchor",
    [
        "同比下降：63.00亿元—69.00亿元",
        "盈利：63.00亿元—69.00亿元",
        "亏损：-63.00亿元—69.00亿元",
        "亏损：63.00万元—69.00亿元",
        "亏损：63.00亿元，收入69.00亿元",
    ],
)
def test_no_loss_inference_or_mixed_metric(anchor):
    spec = {**example(), "loss_anchor": anchor, "quote": anchor}
    with pytest.raises(ValueError, match="EXPLICIT_LOSS_MAGNITUDE_REQUIRED"):
        review_loss_claim(spec, [anchor])


def test_different_metric_values_cannot_match_another_loss_anchor():
    spec = example()
    spec["values"] = ["72.00", "78.00"]
    spec["quote"] += " 扣非净利润 亏损：72.00亿元—78.00亿元"
    with pytest.raises(ValueError, match="LOSS_MAGNITUDES_MISMATCH"):
        review_loss_claim(spec, [spec["quote"]])


def test_duplicate_loss_anchor_rejected():
    spec = example()
    spec["quote"] += spec["loss_anchor"]
    with pytest.raises(ValueError, match="LOSS_ANCHOR_NOT_UNIQUE_IN_CLAIM"):
        review_loss_claim(spec, [spec["quote"]])


def test_loss_fragment_must_still_exist_in_real_page():
    spec = example()
    with pytest.raises(ValueError, match="CLAIM_ANCHOR_NOT_UNIQUE"):
        review_loss_claim(spec, ["盈利：63.00亿元—69.00亿元"])
