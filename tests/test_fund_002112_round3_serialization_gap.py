"""冻结运行后的零拟合回归：明确保留未修复的语义状态摘要缺口，不改旧账本。

预期行为是同一空数组保存/加载后模型语义指纹不变。当前实现直接采用 joblib
对象哈希，含数组 strides，故此用例预期失败。修复须使用新方案，不能恢复旧槽位。
"""

import io

import joblib
import numpy as np
import pytest
from app.services.fund_002112_round3_model import state_hash


@pytest.mark.xfail(strict=True, reason="已确认缺陷：空类别分支数组的 strides 改变造成语义状态指纹误报")
def test_semantic_model_state_must_survive_empty_array_serialization():
    before = {"categorical_bitsets": np.empty((0, 8), dtype=np.uint32)}
    stream = io.BytesIO()
    joblib.dump(before, stream)
    stream.seek(0)
    after = joblib.load(stream)
    np.testing.assert_array_equal(before["categorical_bitsets"], after["categorical_bitsets"])
    assert state_hash(before) == state_hash(after)
