"""真实公布精度、北京时间和完整日历边界，测试原件为隔离合成数据。"""

import json
from copy import deepcopy
from decimal import Decimal

import pytest
from app.services.fund_rating_gt_benchmark import compile_history


@pytest.fixture
def case():
    refs = {
        key: key for key in ("benchmark", "benchmark-page", "benchmark-script", "provider", "summary", "prospectus")
    }
    raw = {
        "benchmark": [["2026-09-28T16:00:00.000Z", 0, 0], ["2026-09-29T16:00:00.000Z", 1, 0.9]],
        "provider": {"dividend_records": []},
        "protocol": {
            "fund_code": "999999",
            "as_of": "2026-09-30",
            "frozen_at": "2026-10-01",
            "first": "2026-09-29",
            "last": "2026-09-30",
            "issuer_percent_decimal_places": 2,
            "method": "COMPLETE_ISSUER_PUBLISHED_CURVE_NO_RECONSTRUCTION",
            "published_nonvaluation_dates_retained": [],
        },
    }
    page = """fundcode: '999999', hideBenchmark: null
    <script src='/Public/static/etrade/1/dist/js/jijin_view.js?v=x'></script>
    业绩比较基准收益率</span>：<span class="J_chart_yeji1">
    业绩比较基准隔离指数收益率×95%+银行活期存款利率（税后）×5%"""
    script = """cochin("info.fund.growth_plot", {
    start:start,end:end,fundCode:yejichart.fundcode,order:"asc"}, callback);
    chartDataYeji.Y.fund = res.map(function(o){return o[1];});
    chartDataYeji.Y.company = res.map(function(o){return o[2];});
    updateLegend(benchmark,'.J_chart_yeji1');"""

    class Source:
        def __init__(self):
            self.raw = {k: json.dumps(v).encode() for k, v in raw.items()}
            self.raw.update({"benchmark-page": page.encode(), "benchmark-script": script.encode()})
            self.sources = {
                "benchmark": {
                    "url": "https://e.gtfund.com/Etrade/Public/cochin/info.fund.growth_plot",
                    "request": {
                        "api": "info.fund.growth_plot",
                        "params": {
                            "fundCode": "999999",
                            "order": "asc",
                            "start": "2026-09-28T16:00:00.000Z",
                            "end": "2026-09-29T16:00:00.000Z",
                        },
                    },
                },
                "benchmark-page": {"url": "https://e.gtfund.com/etrade/Jijin/view/id/999999"},
                "benchmark-script": {"url": "https://e.gtfund.com/Public/static/etrade/1/dist/js/jijin_view.js?v=x"},
            }

        def json(self, digest):
            return json.loads(self.raw[digest])

        def pages(self, _digest):
            return ["业绩比较基准隔离指数收益率×95%+银行活期存款利率(税后)×5%"]

    inputs = {
        "fund_code": "999999",
        "as_of_date": "2026-09-30",
        "verified_at": "2026-10-02",
        "target_refs": refs,
        "gt_protocol": "protocol",
    }
    history = {
        "dates": ["2026-09-29", "2026-09-30"],
        "adjusted_nav": ["1", "1.01"],
        "rows": [
            {"nav_date": day, "ann_date": day, "unit_nav": value, "accumulated_nav": value, "adjusted_nav": value}
            for day, value in [("2026-09-29", "1"), ("2026-09-30", "1.01")]
        ],
    }
    return Source(), inputs, history


def test_public_curve_keeps_real_published_values_and_beijing_dates(case):
    source, inputs, history = case
    value = compile_history(source, inputs, history)
    assert list(map(Decimal, value["benchmark_index"])) == [Decimal(1), Decimal("1.009")]
    assert value["publisher_cross_check"]["missing_values_filled"] == 0
    assert value["publisher_cross_check"]["unpublished_precision_reconstructed"] is False


@pytest.mark.parametrize(
    "change,error",
    [
        ("wrong_share", "REQUEST_SCOPE"),
        ("wrong_column", "COLUMN_SEMANTICS"),
        ("missing_day", "DAILY_COVERAGE"),
        ("duplicate", "DUPLICATE"),
        ("wrong_nav", "NAV_CONFLICT"),
        ("precision", "PRECISION"),
        ("utc_midnight", "TIMESTAMP"),
        ("dividend", "DIVIDEND"),
    ],
)
def test_wrong_identity_calendar_precision_or_nav_cannot_become_a_score(case, change, error):
    source, inputs, history = deepcopy(case)
    rows = source.json("benchmark")
    if change == "wrong_share":
        source.sources["benchmark"]["request"]["params"]["fundCode"] = "999998"
    elif change == "wrong_column":
        source.raw["benchmark-script"] = source.raw["benchmark-script"].replace(b"return o[2]", b"return o[3]")
    elif change == "missing_day":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[-1])
    elif change == "wrong_nav":
        rows[-1][1] = 1.01
    elif change == "precision":
        rows[-1][2] = 0.901
    elif change == "utc_midnight":
        rows[-1][0] = "2026-09-30T00:00:00.000Z"
    elif change == "dividend":
        source.raw["provider"] = b'{"dividend_records":[{"isolated":true}]}'
    source.raw["benchmark"] = json.dumps(rows).encode()
    with pytest.raises(ValueError, match=error):
        compile_history(source, inputs, history)
