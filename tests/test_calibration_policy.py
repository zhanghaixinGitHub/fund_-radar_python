"""新旧规则的隔离、反向及常数输出、真实评分和不可伪造的发布边界。"""

from pathlib import Path

import pytest
from app.services.calibration_policy import calibration_diagnostic
from app.services.cash_policy_freeze import restore_policy_freeze, validate_policy_binding
from app.services.cash_release_policy import load_release_policy
from app.services.historical_nav_calibration import calibrated_model_hash
from app.services.historical_nav_storage import HistoricalNavStorageError
from app.services.model_comparison_adapters import predict_self_trained
from app.services.model_comparison_artifacts import file_hash, fingerprint, write_json
from app.services.model_comparison_evaluation import metrics
from app.services.model_comparison_protocol import settings, validate_protocol
from app.services.model_comparison_runner import _with_probability
from tests.test_cash_policy_freeze import row_for
from tests.test_cash_prediction_inference import model as model
from tests.test_cash_reinvestment_research import data as data
from tests.test_model_comparison_boundaries import make_input


@pytest.mark.parametrize(
    "slope,status", [(1, "FORWARD_MAPPING"), (-1, "REVERSED_PENDING_VALIDATION"), (0, "CONSTANT_PENDING_VALIDATION")]
)
def test_both_models_return_probabilities_and_diagnostics_for_finite_maps(model, slope, status):
    changed = model.model_copy(
        update={"calibrator": model.calibrator.model_copy(update={"slope": slope, "intercept": 0.0})}
    )
    changed = changed.model_copy(update={"model_hash": calibrated_model_hash(changed)})
    a = predict_self_trained(changed.base_model, changed, [make_input()])[0]
    b_model = {"version": "CHRONOS2_LOCAL_SIGMOID_V1", "base_model_hash": "a" * 64, "slope": slope, "intercept": 0.0}
    b_model["model_hash"] = fingerprint(b_model)
    b = _with_probability({"raw_score": 1.0, "raw_direction": 1, "reason": None}, b_model, "a" * 64)
    for output in (a, b):
        assert output["probability"] is not None and output["reason"] is None
        assert output["calibration"]["status"] == status
        assert metrics([(1, output["probability"])])["sample_count"] == 1
        if slope == 0:
            assert output["probability"] == 0.5
    if slope < 0:
        assert b["probability"] < 0.5 and b["raw_direction"] == 1
    for legacy in (
        predict_self_trained(changed.base_model, changed, [make_input()], reject_nonpositive_slope=True)[0],
        _with_probability({"raw_score": 1.0, "reason": None}, b_model, "a" * 64, reject_nonpositive_slope=True),
    ):
        assert "calibration" not in legacy
        assert (legacy["probability"] is None) == (slope <= 0)


@pytest.mark.parametrize("slope,intercept", [(float("nan"), 0), (1, float("inf")), (float("-inf"), 0)])
def test_nonfinite_parameters_are_errors_not_direction_warnings(slope, intercept):
    with pytest.raises(ValueError, match="NONFINITE"):
        calibration_diagnostic(slope, intercept)


@pytest.mark.parametrize("version", ["CHRONOS2_CASH_COMPARISON_V1", "CHRONOS2_CASH_COMPARISON_V2"])
def test_protocol_version_prevents_silent_reinterpretation(tmp_path, version):
    root = Path(__file__).resolve().parents[1]
    protocol = {
        "settings": settings(version),
        "requirements_sha256": file_hash(root / "requirements-chronos2.txt"),
        "base_requirements_sha256": file_hash(root / "requirements.txt"),
    }
    protocol["protocol_hash"] = fingerprint(protocol)
    write_json(tmp_path / "protocol.json", protocol)
    assert validate_protocol(tmp_path) == protocol
    protocol["settings"]["calibration"]["reject_nonpositive_slope"] = not protocol["settings"]["calibration"][
        "reject_nonpositive_slope"
    ]
    protocol["protocol_hash"] = fingerprint({k: v for k, v in protocol.items() if k != "protocol_hash"})
    other = tmp_path / "wrong"
    other.mkdir()
    write_json(other / "protocol.json", protocol)
    with pytest.raises(ValueError, match="incompatible"):
        validate_protocol(other)


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
