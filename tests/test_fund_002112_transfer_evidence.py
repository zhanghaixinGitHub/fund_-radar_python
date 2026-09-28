"""验证纯统计含义、日期匹配和缺失处理；不训练模型或访问网络。"""

import pytest
from app.services.fund_002112_transfer_evidence import correlation, paired_returns, rank_score, sign


@pytest.mark.parametrize(
    "up,down,expected",
    [([2], [1], 1), ([1], [2], 0), ([1], [1], 0.5), ([1, 2], [1, 2], 0.5), ([], [1], None), ([1], [], None)],
)
def test_rank_ordering_and_ties(up, down, expected):
    assert rank_score(up, down) == expected


def test_rank_rejects_nonfinite():
    with pytest.raises(ValueError):
        rank_score([float("nan")], [1])


def test_correlation_empty_constant_and_exact():
    assert correlation([], []) is None
    assert correlation([1, 1], [1, 2]) is None
    assert correlation([1, 2], [2, 4]) == pytest.approx(1)
    with pytest.raises(ValueError):
        correlation([1], [1, 2])


@pytest.mark.parametrize("value,expected", [(-0.1, "negative"), (0, "zero"), (0.1, "positive")])
def test_fixed_financial_zero(value, expected):
    assert sign(value) == expected


def test_pair_requires_both_dates_and_never_fills():
    row = {
        "target": "2023-06-01",
        "base": "2023-05-31",
        "base_unit_nav": "1",
        "target_unit_nav": "1.01",
        "actual_direction": "UP",
    }
    wrong_base = {**row, "base": "2023-05-30"}
    result = paired_returns([row], [row, wrong_base])
    assert result["groups"]["all"]["matched"] == 1
    assert result["missing_target_dates"] == [{"target": "2023-06-01", "base": "2023-05-30"}]
    assert result["daily"][0]["peer_return"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="DUPLICATE"):
        paired_returns([row, row], [])
