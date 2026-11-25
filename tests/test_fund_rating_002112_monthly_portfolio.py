from decimal import Decimal as D

import pytest
from app.services.fund_rating_002112_monthly_portfolio import concentration, suspension_days


def test_reported_zero_residual_is_retained_without_inventing_classification():
    stocks = [dict(security="400237", market_value="0"), dict(security="600001", market_value="30"),
              dict(security="600002", market_value="70")]
    values = concentration(stocks, {"600001": "甲", "600002": "乙"}, {"600001": "C39", "600002": "C39"})
    assert values == [D(".58"), D(1)]
    assert len(stocks) == 3


def test_positive_unclassified_position_cannot_be_removed_or_renormalized():
    with pytest.raises(ValueError, match="CLASSIFICATION_MISSING"):
        concentration([dict(security="400237", market_value=".01"), dict(security="600001", market_value="30")],
                      {"600001": "甲"}, {"600001": "C39"})


class Source:
    def __init__(self, text):
        self.text = text

    def original(self, digest):
        return {"published_on": "2026-09-01"}

    def claim(self, claim):
        assert claim["quote"] in self.text

    def body(self, digest):
        return self.text


def interval(resume_quote="公司股票自2026年9月4日开市起复牌。"):
    start_quote = "证券代码600001，公司股票自2026年9月1日开市起停牌。"
    return Source(start_quote + resume_quote), dict(start_date="2026-09-01", resume_date="2026-09-04", claims=[
        dict(source_hash="a", quote=start_quote, role="SUSPENSION_START", date="2026-09-01"),
        dict(source_hash="a", quote=resume_quote, role="RESUMPTION", date="2026-09-04")])


def test_actual_full_day_interval_excludes_resumption_day():
    source, proof = interval()
    assert suspension_days(source, "600001", [proof], ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]) == {
        "2026-09-01", "2026-09-02", "2026-09-03"}


def test_expected_resumption_does_not_close_actual_interval():
    source, proof = interval("公司预计复牌日期为2026年9月4日。")
    with pytest.raises(ValueError, match="ACTUAL_RESUMPTION"):
        suspension_days(source, "600001", [proof], ["2026-09-02"])


def test_infinite_order_does_not_invent_an_unquoted_through_date():
    source, proof = interval()
    proof.pop("resume_date")
    proof["through_date"] = "2026-09-30"
    proof["claims"] = proof["claims"][:1]
    with pytest.raises(ValueError, match="ACTUAL_THROUGH_DATE"):
        suspension_days(source, "600001", [proof], ["2026-09-02"])


@pytest.mark.parametrize(
    "query_day, row_code, halt_type, total, accepted",
    [
        ("20260902", "600001", "LXTP", 1, True),
        ("20260901", "600001", "LXTP", 1, False),
        ("20260902", "600002", "LXTP", 1, False),
        ("20260902", "600001", "LSTP", 1, False),
        ("20260902", "600001", "LXTP", 2, False),
    ],
)
def test_official_exchange_status_confirms_only_matching_continuous_halt(
    query_day, row_code, halt_type, total, accepted
):
    source, proof = interval()
    proof.pop("resume_date")
    proof.update(through_date="2026-09-02", sse_status_ref="exchange")
    proof["claims"] = proof["claims"][:1]
    source.asof = "2026-09-02"
    source.original = lambda digest: {
        "published_on": "2026-09-01",
        "url": ("https://query.sse.com.cn/commonSoaQuery.do?sqlId=GW_PL_JYTS_TFPXX&productCode=600001"
                f"&startStopDate={query_day}&endStopDate={query_day}"),
    }
    source.json = lambda digest: {
        "pageHelp": {"total": total},
        "result": [{"productCode": row_code, "controlType": "TR", "type": halt_type, "startStopDate": "20260901",
                    "endStopDate": "", "stopTime": ""}],
    }
    if accepted:
        assert suspension_days(source, "600001", [proof], ["2026-09-01", "2026-09-02", "2026-09-03"]) == {
            "2026-09-01", "2026-09-02",
        }
    else:
        with pytest.raises(ValueError, match="SUSPENSION_EXCHANGE"):
            suspension_days(source, "600001", [proof], ["2026-09-02"])


def test_intraday_halt_is_not_full_day_zero_turnover():
    source, proof = interval()
    text = "证券代码600001，公司股票2026年9月1日盘中临时停牌。"
    source.text = text + proof["claims"][1]["quote"]
    proof["claims"][0]["quote"] = text
    with pytest.raises(ValueError, match="FULL_DAY_START"):
        suspension_days(source, "600001", [proof], ["2026-09-01"])


@pytest.mark.parametrize("row,accepted", [
    ("603759 海天股份 A 股 复牌 2026/10/19 2026/10/20", True),
    ("603759 海天股份 A 股 复牌 2026/10/19 2026/10/20\n113700 海天转债 可转债债券复牌", True),
    ("113700 海天转债 可转债债券复牌 2026/10/19 2026/10/20", False),
    ("600001 别的股份 A 股 复牌 2026/10/19 2026/10/20", False),
    ("603759 海天股份 A 股 复牌 2026/10/18 2026/10/20", False),
])
def test_official_full_day_schedule_requires_own_share_and_exact_dates(row, accepted):
    text = ("证券代码603759。证券代码 证券简称 停复牌类型 停牌起始日 停牌 期间 停牌终止日 复牌日\n"
            + row + "。公司股票于2026年10月19日停牌，自2026年10月20日起复牌。")
    proof = dict(start_date="2026-10-19", resume_date="2026-10-20", claims=[
        dict(source_hash="a", quote=text, role="SUSPENSION_START", date="2026-10-19"),
        dict(source_hash="a", quote=text, role="RESUMPTION", date="2026-10-20"),
    ])
    if accepted:
        assert suspension_days(Source(text), "603759", [proof], ["2026-10-19", "2026-10-20"]) == {"2026-10-19"}
    else:
        with pytest.raises(ValueError, match="FULL_DAY_START"):
            suspension_days(Source(text), "603759", [proof], ["2026-10-19"])


def test_portfolio_decimal_precision_is_independent_of_snapshot_caller(monkeypatch):
    from decimal import localcontext
    from app.services import fund_rating_002112_monthly_portfolio as module

    monkeypatch.setattr(module, "_compile_observations", lambda *args, **kwargs: D(1) / D(7))
    with localcontext() as context:
        context.prec = 28
        prepared = module.compile_observations({}, root=None)
        assert context.prec == 28
    with localcontext() as context:
        context.prec = 80
        replayed = module.compile_observations({}, root=None)
    assert prepared == replayed
    assert len(prepared.as_tuple().digits) == 80


def test_partial_calculation_keeps_fixed_precision_and_restores_caller(monkeypatch):
    from decimal import localcontext
    from app.services import fund_rating_active_r5_partial as module
    from app.services import fund_rating_r5

    monkeypatch.setattr(module, "packet_for", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(fund_rating_r5, "calculate", lambda *args, **kwargs: D(1) / D(7))
    inputs = {"selected_dimensions": ["holdings"]}
    with localcontext() as context:
        context.prec = 28
        prepared = module.calculate_inputs(inputs, root=".")
        assert context.prec == 28
    with localcontext() as context:
        context.prec = 80
        replayed = module.calculate_inputs(inputs, root=".")
    assert prepared == replayed
