"""原文折行及合同基准身份不能凭临近页码或可填写别名补造。"""

import pytest
from app.services import fund_rating_bond_disclosure as bond

STOCK = "  2    600674    川投能源    11,064,50  156,783,965.00                    6.95"
ZERO = " " * 40 + "0"


def test_observed_stock_continuation_requires_adjacent_aligned_zero():
    assert bond._adjacent_zero([STOCK, ZERO, ""], "11,064,50") == "0"


@pytest.mark.parametrize(
    "tail",
    [
        [" " * 40 + "3"],
        ["第 3 页 共 59 页", ZERO],
        ["", ZERO],
        ["普通注释", ZERO],
        [ZERO, "", ZERO],
        [" " * 40 + "00"],
        ["0"],
        ["工银瑞信可转债债券型证券投资基金2026年中期报告", ZERO],
    ],
)
def test_folio_or_ambiguous_number_cannot_fill_a_financial_cell(tail):
    assert bond._adjacent_zero([STOCK, *tail], "11,064,50") is None


def test_correct_market_data_cannot_be_presented_as_another_contract_index(monkeypatch):
    from decimal import Decimal

    days = ["2025-09-30", "2026-09-30"]
    monkeypatch.setattr(bond, "_nav", lambda *_: (days, days, [Decimal(1), Decimal(2)]))

    class Source:
        def json(self, _):
            return {
                "basic": {"index_code": "000300.SH", "display_name": "沪深300"},
                "daily": [{"trade_date": day, "close_price": str(i)} for i, day in enumerate(days, 1)],
            }

        def claim(self, _):
            return None

    item = {"kind": "CSI", "code": "000300.SH", "name": "沪深300", "weight": "1", "source_hash": "isolated"}
    good = {
        "benchmark_components": [{**item, "contract_name": "沪深300指数"}],
        "benchmark_contract": {"quote": "沪深300指数收益率×100%"},
    }
    assert bond._history(good, Source())["benchmark_index"] == ["1", "2"]
    wrong = {
        "benchmark_components": [{**item, "contract_name": "中证可转换债券指数"}],
        "benchmark_contract": {"quote": "中证可转换债券指数收益率×100%"},
    }
    with pytest.raises(ValueError, match="CONTRACT_REFERENCE_CODE_NAME_BINDING"):
        bond._history(wrong, Source())
