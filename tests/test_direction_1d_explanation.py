"""原预测解释的纯计算与只读边界；不训练或写入真实预测。"""

import copy

import pytest
from app.services.direction_1d_explanation import explain_original
from app.services.direction_1d_protocol import FEATURE_VERSION, FEATURES, PROTOCOL, TARGET, digest, score


def example():
    model = {
        "protocol": PROTOCOL,
        "horizon": 1,
        "target_definition": TARGET,
        "feature_version": FEATURE_VERSION,
        "features": list(FEATURES),
        "coef": [2, -1, 0, 0, 0, 0, 0],
        "mean": [0] * 7,
        "scale": [1] * 7,
        "intercept": -0.2,
    }
    source = {"fund_code": "008888", "features": [1, 1, 0, 0, 0, 0, 0]}
    body = {
        "fund_code": "008888",
        "base_nav_date": "2026-09-23",
        "target_nav_date": "2026-09-24",
        "input": source,
        "input_hash": digest(source),
        "branches": [
            {
                "branch_id": name,
                "model_id": "original",
                "model_hash": "hash",
                "status": "AVAILABLE",
                "score": score(model, source["features"]),
                "predicted_direction": "UP",
            }
            for name in ("FIXED", "WEEKLY")
        ],
    }
    return model, body


def test_restore_retains_opposing_factors_and_does_not_modify_original():
    model, body = example()
    before = copy.deepcopy(body)
    calls = []

    def load(model_id, expected_hash):
        calls.append((model_id, expected_hash))
        return model

    result = explain_original(body, load)
    assert body == before
    assert calls == [("original", "hash"), ("original", "hash")]
    assert [f["contribution"] for f in result["branches"][0]["factors"]] == [2, -1, 0, 0, 0, 0, 0]
    assert result["branches"][0]["intercept"] == -0.2
    assert "score" not in result["branches"][0]


@pytest.mark.parametrize("change", ["input_hash", "fund", "score", "direction", "model"])
def test_corrupted_evidence_fails_closed(change):
    model, body = example()
    if change == "input_hash":
        body["input_hash"] = "bad"
    elif change == "fund":
        body["fund_code"] = "000001"
    elif change == "score":
        body["branches"][0]["score"] = 0.9
    elif change == "direction":
        body["branches"][0]["predicted_direction"] = "NON_UP"
    else:
        model["coef"][0] = 3
    with pytest.raises(ValueError):
        explain_original(body, lambda *_: model)


def test_failed_branch_not_restored_or_converted_to_flat():
    model, body = example()
    body["branches"][1]["status"] = "MODEL_UNAVAILABLE"
    result = explain_original(body, lambda *_: model)
    assert len(result["branches"]) == 1
    body["branches"][0]["status"] = "MODEL_UNAVAILABLE"
    with pytest.raises(ValueError, match="NO_AVAILABLE_BRANCH"):
        explain_original(body, lambda *_: model)
