"""IH/IC来源关键边界：严格区分品种、连续主力与具体月份，不补缺日。"""

import json

import pytest
from app.integrations import tushare_sprint_futures_style as s


def body(api, rows):
    return json.dumps({"code": 0, "data": {"fields": s.FIELDS[api], "items": rows}}).encode()


def daily():
    return ["IC.CFX", "20260915", 7573.6, 7564.6, 7641.8, 7531.0, 7567.8, 7565.0, 77427, 83764]


def test_both_source_types():
    assert s.parse(body("fut_daily", [daily()]), "fut_daily", "IC", ["2026-09-15"])["2026-09-15"]["open"] == 7564.6
    assert (
        s.parse(body("fut_mapping", [["IH.CFX", "20260915", "IH2609.CFX"]]), "fut_mapping", "IH", ["2026-09-15"])[
            "2026-09-15"
        ]["mapping_ts_code"]
        == "IH2609.CFX"
    )


@pytest.mark.parametrize(
    "i,v", [(0, "IF.CFX"), (1, "20260916"), (3, float("nan")), (6, 0), (8, True), (9, None), (5, 7600)]
)
def test_wrong_identity_or_prices_fail(i, v):
    row = daily()
    row[i] = v
    with pytest.raises(ValueError):
        s.parse(body("fut_daily", [row]), "fut_daily", "IC", ["2026-09-15"])


@pytest.mark.parametrize("mapped", ["IC.CFX", "ICL1.CFX", "IF2609.CFX", "IC2609.SH"])
def test_mapping_must_be_same_family_real_month(mapped):
    with pytest.raises(ValueError, match="MAPPING_INVALID"):
        s.parse(body("fut_mapping", [["IC.CFX", "20260915", mapped]]), "fut_mapping", "IC", ["2026-09-15"])


def test_missing_and_duplicate_dates_fail():
    with pytest.raises(ValueError, match="MISSING_DATES"):
        s.parse(body("fut_daily", [daily()]), "fut_daily", "IC", ["2026-09-14", "2026-09-15"])
    with pytest.raises(ValueError, match="DATE_SCOPE_OR_DUPLICATE"):
        s.parse(body("fut_daily", [daily(), daily()]), "fut_daily", "IC", ["2026-09-15"])
