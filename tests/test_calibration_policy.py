"""自训练校准参数校验和新旧发布策略的隔离。"""

import pytest
from app.services.calibration_policy import calibration_diagnostic
from app.services.cash_policy_freeze import restore_policy_freeze, validate_policy_binding
from app.services.cash_release_policy import load_release_policy
from app.services.historical_nav_storage import HistoricalNavStorageError
from tests.test_cash_policy_freeze import row_for


@pytest.mark.parametrize("slope,intercept", [(float("nan"), 0), (1, float("inf")), (float("-inf"), 0)])
def test_nonfinite_parameters_are_errors_not_direction_warnings(slope, intercept):
    with pytest.raises(ValueError, match="NONFINITE"):
        calibration_diagnostic(slope, intercept)


def test_old_policy_snapshot_restores_but_cannot_be_used_as_v2_approval():
    current = load_release_policy()
    assert current.version == "CASH_RELEASE_POLICY_V2" and current.approval_state == "DRAFT"
    old = current.model_copy(
        update={
            "version": "CASH_RELEASE_POLICY_V1",
            "approval_state": "APPROVED",
            "approval_reference": "synthetic-old-only",
        }
    )
    snapshot = restore_policy_freeze(row_for(old))
    assert snapshot.snapshot.policy == old and not snapshot.publication_allowed
    with pytest.raises(HistoricalNavStorageError) as error:
        validate_policy_binding(snapshot, current)
    assert error.value.code == "CASH_POLICY_FREEZE_MISMATCH"
