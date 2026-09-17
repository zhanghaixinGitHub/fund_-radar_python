"""基金身份列的顺序、交互单位和固定收缩强度必须在训练与未来输入之间一致。"""

import numpy as np
import pytest
from app.services import direction_1d_sprint_partial_fund_features as s

CODES = [f"{1000 + i:06d}" for i in range(30)]


def test_identity_positions_and_spx_units_are_fixed():
    z = [2.0] + [0.0] * 11
    result = s.extras(z, CODES[4], CODES)
    assert result[4] == 0.25 and result[34] == 0.5
    assert sum(value != 0 for value in result) == 2
    assert s.extras([9.0] + [0.0] * 11, CODES[4], CODES)[34] == 1.25
    assert s.extras([-9.0] + [0.0] * 11, CODES[4], CODES)[34] == -1.25


@pytest.mark.parametrize("codes", [CODES[::-1], CODES[:-1], CODES[:-1] + [CODES[0]], ["abcdef"] * 30])
def test_bad_or_reordered_code_tables_are_rejected(codes):
    with pytest.raises(ValueError, match="CODE_TABLE_INVALID"):
        s.extras([0.0] * 12, CODES[0], codes)


def test_unknown_fund_and_invalid_input_do_not_fall_back_to_another_fund():
    for z, code in (([0.0] * 12, "999999"), ([0.0] * 11, CODES[0]), ([float("nan")] * 12, CODES[0])):
        with pytest.raises(ValueError, match="FEATURE_OR_CODE_INVALID"):
            s.extras(z, code, CODES)


def test_only_global12_are_normalized_not_rare_identity_columns():
    rows = []
    for i in range(4):
        z = [float(i + 1)] + [0.0] * 11
        rows.append(z + s.extras(z, CODES[0] if i < 3 else CODES[1], CODES))
    x = np.asarray(rows)
    weights = np.asarray([1, 2, 3, 4], dtype=float)
    normalized, mean, scale = s.normalize_training(x, weights)
    assert mean[12:] == [0.0] * 60 and scale[12:] == [1.0] * 60
    np.testing.assert_array_equal(normalized[:, 12:], x[:, 12:])
    assert np.average(normalized[:, 0], weights=weights) == pytest.approx(0, abs=1e-12)
    assert np.average(normalized[:, 0] ** 2, weights=weights) == pytest.approx(1)
    assert normalized[-1, 13] == 0.25
    assert scale[1:12] == [1.0] * 11


def test_normalization_rejects_wrong_dimension_or_nonfinite():
    for value in (np.zeros((3, 71)), np.full((3, 72), np.nan)):
        with pytest.raises(ValueError, match="NORMALIZATION_INPUT_INVALID"):
            s.normalize_training(value, np.ones(3))
