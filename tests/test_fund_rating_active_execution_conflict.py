"""已发现的同公司股债冲突不得经旧V1再次返回正常执行分。"""

import pytest

from app.services.fund_rating_active_execution import VERSION, calculate


def test_old_candidate_cannot_bypass_known_combined_issuer_conflict(tmp_path):
    with pytest.raises(ValueError, match="KNOWN_PUBLIC_ISSUER_CONFLICT_REQUIRES_NEW_ADAPTER"):
        calculate({"version": VERSION, "fund_code": "004237"}, root=tmp_path)
