"""验证原预测的关系与单项对照，不向真实服务发请求或保存预测。"""

import copy

import pytest
from app.integrations.prediction_narrative import validate_response
from app.services import direction_1d_three_state as three
from app.services.direction_1d_explanation import explain_original
from app.services.direction_1d_protocol import FEATURE_VERSION, FEATURES, digest
from app.services.prediction_narrative_facts import daily_facts


def original(intercept=0.1, extra=True):
    values = [-0.1, -0.04, -0.11, 0.032, -0.36, 0.3, 2]
    model = {
        "protocol": three.PROTOCOL,
        "target_definition": three.TARGET,
        "horizon": 1,
        "feature_version": FEATURE_VERSION,
        "features": list(FEATURES),
        "direction_policy": three.POLICY,
        "class_order": list(three.CLASSES),
        "tie_order": list(three.TIE_ORDER),
        "mean": [0.002, 0, 0, 0.012, -0.1, 0.5, 1.5],
        "scale": [0.04, 1, 1, 0.01, 0.1, 1, 1],
        "coef": [[0] * 7, [0] * 7, [-1, 0, 0, 0.5 if extra else 0, -0.1 if extra else 0, 0, 0]],
        "intercept": [0, -5, intercept],
    }
    result = three.predict(model, values)
    source = {"fund_code": "002112", "features": values}
    body = {
        "fund_code": "002112",
        "protocol": three.PROTOCOL,
        "base_nav_date": "2026-09-29",
        "target_nav_date": "2026-09-30",
        "input": source,
        "input_hash": digest(source),
        "branches": [
            {
                "branch_id": "FIXED",
                "status": "AVAILABLE",
                "model_id": "test",
                "model_hash": "hash",
                "score": result["score"],
                "class_scores": result["class_scores"],
                "predicted_direction": result["direction"],
            }
        ],
    }
    return model, body


def restore(model, body):
    return explain_original(body, lambda *_: model, include_reasoning=True)


def test_original_parameters_remain_auditable_but_never_become_market_reasons():
    model, body = original()
    before = copy.deepcopy((model, body))
    detailed = restore(model, body)
    factor = detailed["branches"][0]["factors"][0]
    assert factor["reasoning"] == {
        "referenceValue": 0.002,
        "relativeWeight": -1,
        "comparisonDirection": "DOWN",
        "directionAtReference": "UP",
    }
    facts = daily_facts(detailed)
    text = str(facts)
    for phrase in ("0.20%", "1.20%", "-10.00%", "越深", "换回", "参考水平", "relativeWeight", "referenceValue"):
        assert phrase not in text
    assert "标准差，未年化，并非平均每天涨跌幅" in text
    assert "本基金历史波动分布" in text
    assert "解释依据仍有限" in text
    assert (model, body) == before
    assert "reasoning" not in explain_original(body, lambda *_: model)["branches"][0]["factors"][0]


def test_counterfactual_remains_internal_even_when_it_changes_class():
    model, body = original(intercept=-1, extra=False)
    restored = restore(model, body)
    assert restored["branches"][0]["factors"][0]["reasoning"]["directionAtReference"] == "DOWN"
    assert "换回" not in str(daily_facts(restored))


def test_opposing_observation_is_kept_without_claiming_market_causality():
    model, body = original()
    model["coef"][2][1] = 0.1
    outcome = three.predict(model, body["input"]["features"])
    body["branches"][0].update(score=outcome["score"], class_scores=outcome["class_scores"])
    facts = daily_facts(restore(model, body))
    assert "近 5 和 20 个交易日累计分别下跌 10.00% 和 4.00%" in str(facts)
    assert "与本次偏向上涨的判断并不一致" in facts["facts"]["Q1"]
    assert "削弱" not in str(facts)


def test_incompatible_branch_references_are_not_merged_into_a_single_reason():
    model, body = original()
    restored = restore(model, body)
    branch = copy.deepcopy(restored["branches"][0])
    branch["modelHash"] = "other"
    branch["factors"][0]["reasoning"]["referenceValue"] = 0.01
    restored["branches"].append(branch)
    with pytest.raises(ValueError, match="REASONING_DISAGREEMENT"):
        daily_facts(restored)


def test_mismatched_relationship_sign_is_rejected():
    model, body = original()
    restored = restore(model, body)
    restored["branches"][0]["factors"][0]["reasoning"]["relativeWeight"] = 1
    with pytest.raises(ValueError, match="SIGN_MISMATCH"):
        daily_facts(restored)


def test_editor_can_merge_and_reorder_observations_but_cannot_omit_or_duplicate_them():
    model, body = original()
    facts = daily_facts(restore(model, body))
    refs = list(reversed(facts["requiredRefs"]))
    raw = {"analysis": "，".join("{{" + key + "}}" for key in refs) + "。"}
    result = validate_response(raw, facts)
    assert result["supporting"] == result["opposing"] == ""
    assert result["evidenceRefs"]["context"] == refs
    for invalid in (raw["analysis"].replace("{{B1}}", ""), raw["analysis"] + "{{F1}}"):
        with pytest.raises(ValueError, match="REFERENCE"):
            validate_response({"analysis": invalid}, facts)


@pytest.mark.parametrize("direction", ["UP", "DOWN", "FLAT"])
def test_summary_and_conflict_follow_original_conclusion(direction):
    model, body = original()
    winning_weights = model["coef"][2]
    model["coef"] = [[0] * 7 for _ in range(3)]
    model["coef"][three.CLASSES.index(direction)] = winning_weights
    model["intercept"] = [0, 0, 0]
    outcome = three.predict(model, body["input"]["features"])
    body["branches"][0].update(
        score=outcome["score"], class_scores=outcome["class_scores"], predicted_direction=direction
    )
    facts = daily_facts(restore(model, body))
    label = {"UP": "上涨", "DOWN": "下跌", "FLAT": "持平"}[direction]
    assert f"偏向“{label}”" in facts["summary"]
    assert "2026-09-29" in facts["summary"]
    assert ("Q1" in facts["facts"]) == (direction != "DOWN")


@pytest.mark.parametrize(
    "extra",
    [
        "因为这些数据，所以接下来会上涨。",
        "回撤越深越容易涨。",
        "波动偏高。",
        "处于历史高位。",
        "同类排名靠前。",
        "上涨概率为八成。",
        "胜率很高。",
        "资金回流。",
        "历史均值是０．１８％。",
        "收益率为三点二一。",
        "不是{{B1}}。",
        "{{Q1}}不成立。",
        "预计持平。",
    ],
)
def test_unsupported_qualifications_causes_numbers_and_negated_facts_are_rejected(extra):
    model, body = original()
    facts = daily_facts(restore(model, body))
    raw = "。".join("{{" + key + "}}" for key in facts["requiredRefs"]) + "。"
    if "{{B1}}" in extra:
        raw = raw.replace("{{B1}}", "不是{{B1}}")
    elif "{{Q1}}" in extra:
        raw = raw.replace("{{Q1}}", "{{Q1}}不成立")
    else:
        raw += extra
    with pytest.raises(ValueError):
        validate_response({"analysis": raw}, facts)


@pytest.mark.parametrize(
    "clause",
    [
        "因此对后续的判断仍需与这段历史观察区分开来",
        "因此该方向判断的适用范围限于已有观察",
        "所以仍需区分历史观察与未来判断",
        "这说明现有依据仍有限",
        "这并不代表未来会延续",
        "因此不能仅凭这些观察确认后续方向",
        "因为缺少可靠的后续表现证据，所以本次判断的解释依据仍有限",
    ],
)
def test_logical_connections_expressing_evidence_limits_are_allowed(clause):
    from app.services.prediction_narrative_facts import multi_facts

    from test_prediction_narrative import baseline

    facts = multi_facts(baseline())
    raw = {"analysis": "{{F1}}。{{B1}}。" + clause + "。"}
    assert clause in validate_response(raw, facts)["context"]


@pytest.mark.parametrize(
    "clause",
    [
        "因此支持上涨",
        "所以回撤必反弹",
        "因为回撤深所以更有利于上涨",
        "这说明现有依据仍有限但上涨占优",
        "不是这说明现有依据仍有限",
        "因此对后续的判断仍需与这段历史观察区分开来但未来会上涨",
    ],
)
def test_limited_logic_exemptions_cannot_be_extended_into_market_assertions(clause):
    from app.services.prediction_narrative_facts import multi_facts

    from test_prediction_narrative import baseline

    facts = multi_facts(baseline())
    with pytest.raises(ValueError, match="UNSUPPORTED_ASSERTION"):
        validate_response({"analysis": "{{F1}}。{{B1}}。" + clause + "。"}, facts)


def test_saved_provider_draft_rejection_is_for_free_numbers_not_safe_logical_connection():
    from app.integrations.prediction_narrative import response_issues
    from app.services.prediction_narrative_facts import multi_facts

    from test_prediction_narrative import baseline

    facts = multi_facts(baseline())
    raw = {
        "analysis": (
            "{{F1}}，这一观察仅描述近 20 个交易日的累计涨跌方向。{{B1}}，因此对后续的判断仍需与这段历史观察区分开来。"
        )
    }
    assert response_issues(raw, facts) == ["NARRATIVE_FREE_NUMBER_OR_DATE"]
    # 只去掉重复的区间数字，其余正常局限文字逐字保留，离线验证应通过。
    raw["analysis"] = raw["analysis"].replace("近 20 个交易日", "当时区间")
    assert validate_response(raw, facts)["context"]


def test_optional_observations_do_not_force_a_metric_list_and_density_is_bounded():
    model, body = original()
    facts = daily_facts(restore(model, body))
    short = {"analysis": "{{F1}}。{{Q1}}。\n另需留意，{{F2}}。{{B1}}。"}
    assert "最大回撤" not in validate_response(short, facts)["context"]
    with_drawdown = {"analysis": short["analysis"].replace("{{F2}}", "{{F2}}，{{F3}}")}
    assert "最大回撤" in validate_response(with_drawdown, facts)["context"]
    with pytest.raises(ValueError, match="DENSITY"):
        validate_response({"analysis": with_drawdown["analysis"].replace("\n", "")}, facts)
    with pytest.raises(ValueError, match="DENSITY"):
        validate_response({"analysis": with_drawdown["analysis"] + "{{F4}}"}, facts)
