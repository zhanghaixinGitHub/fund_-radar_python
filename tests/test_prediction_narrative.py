"""原预测解释的事实、文风、失败隔离与缓存契约；不访问付费服务。"""

import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from app.core.config import Settings
from app.integrations.prediction_narrative import generate, validate_response
from app.services import prediction_narrative as service
from app.services.prediction_contract import fingerprint
from app.services.prediction_narrative_facts import STYLE_VERSION, daily_facts, multi_facts

KEYS = [
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "max_drawdown_60d",
    "relative_position_60d",
    "consecutive_decline_days",
]


def daily(direction="UP", opposing=True):
    values = [-0.1039, -0.0422, -0.1116, 0.0321, -0.3658, 0.3, 2]
    return {
        "baseNavDate": "2026-09-29",
        "branches": [
            {
                "modelHash": "same",
                "direction": direction,
                "factors": [
                    {
                        "feature": key,
                        "value": value,
                        "contribution": 3 - i if i < 3 else -1 if opposing else 0,
                        "reasoning": {
                            "referenceValue": 0,
                            "relativeWeight": (3 - i if i < 3 else -1 if opposing else 0) / value,
                            "comparisonDirection": "DOWN" if direction == "UP" else "UP",
                            "directionAtReference": direction,
                        },
                    }
                    for i, (key, value) in enumerate(zip(KEYS, values, strict=True))
                ],
            }
        ],
    }


def baseline(direction="DOWN", momentum=-0.1):
    return {
        "fundCode": "002112",
        "predictionId": str(uuid4()),
        "horizonId": "T20_V1",
        "direction": direction,
        "dataAsOf": "2026-09-29",
        "featureSnapshot": {
            "fundCode": "002112",
            "dataAsOf": "2026-09-29",
            "features": {"momentum": momentum, "actualLookbackReturns": 20},
        },
        "modelManifest": {"adapter": "NAV_MOMENTUM_THREE_STATE_V2", "parameters": {"momentumThreshold": 0.01}},
    }


def response(facts):
    return {"analysis": "。".join("{{" + key + "}}" for key in facts["requiredRefs"]) + "。"}


def test_daily_uses_original_signs_retains_opposition_and_deduplicates():
    original = daily()
    original["branches"].append(copy.deepcopy(original["branches"][0]))
    facts = daily_facts(original)
    assert "下跌 10.39%" in facts["facts"]["F1"]
    assert len([k for k in facts["facts"] if k.startswith("F")]) >= 5
    assert "Q1" in facts["facts"]
    assert "上涨" in facts["summary"]
    assert "contribution" not in str(facts)


def test_disagreement_and_invalid_factors_cannot_generate_a_unified_reason():
    original = daily()
    original["branches"].append({**copy.deepcopy(original["branches"][0]), "direction": "DOWN", "modelHash": "other"})
    with pytest.raises(ValueError, match="DIRECTION_DISAGREEMENT"):
        daily_facts(original)
    original = daily()
    original["branches"][0]["factors"][0]["value"] = float("nan")
    with pytest.raises(ValueError):
        daily_facts(original)


@pytest.mark.parametrize("direction,momentum", [("UP", 0.1), ("DOWN", -0.1), ("FLAT", 0.01)])
def test_baseline_restores_actual_window_without_inventing_market_information(direction, momentum):
    facts = multi_facts(baseline(direction, momentum))
    assert "Q1" not in facts["facts"]
    assert "20 个交易日" in facts["facts"]["F1"]
    assert "仅" in facts["facts"]["B1"]


@pytest.mark.parametrize("change", ["direction", "date", "fund", "method", "threshold"])
def test_baseline_mismatches_fail_closed(change):
    value = baseline()
    if change == "direction":
        value["direction"] = "UP"
    elif change in {"date", "fund"}:
        value["featureSnapshot"]["dataAsOf" if change == "date" else "fundCode"] = "wrong"
    elif change == "method":
        value["modelManifest"]["adapter"] = "UNSUPPORTED"
    else:
        value["modelManifest"]["parameters"]["momentumThreshold"] = "nan"
    with pytest.raises(ValueError):
        multi_facts(value)


def test_render_inserts_exact_facts_and_keeps_fixed_summary_and_limits():
    facts = daily_facts(daily())
    rendered = validate_response(response(facts), facts)
    assert "10.39%" in rendered["context"]
    assert "{{" not in str(rendered)
    assert rendered["summary"] == facts["summary"]
    assert rendered["limitations"] == facts["limitations"]


@pytest.mark.parametrize(
    "bad",
    [
        "预计上涨 80%。",
        "{{H1}}保证明天上涨。",
        "{{H1}}资金回流。",
        "{{H1}}预计下跌。",
        "{{H1}}持续下跌。",
        "{{H1}}<script>。",
        "{{H1}}概率较高。",
        "{{H1}}抄底。",
    ],
)
def test_unsupported_numbers_forecast_causality_html_and_advice_are_rejected(bad):
    facts = daily_facts(daily())
    raw = response(facts)
    raw["analysis"] += bad.replace("{{H1}}", "")
    with pytest.raises(ValueError):
        validate_response(raw, facts)


def test_missing_opposition_and_unknown_references_are_rejected():
    facts = daily_facts(daily())
    raw = response(facts)
    raw["analysis"] = raw["analysis"].replace("{{F2}}", "")
    with pytest.raises(ValueError, match="REFERENCE"):
        validate_response(raw, facts)
    raw = response(facts)
    raw["analysis"] += "{{R1}}"
    with pytest.raises(ValueError):
        validate_response(raw, facts)


def test_empty_opposition_must_not_become_no_risk():
    facts = daily_facts(daily(opposing=False))
    raw = response(facts)
    raw["analysis"] += "没有风险。"
    with pytest.raises(ValueError):
        validate_response(raw, facts)


def test_history_direction_and_evidence_stance_cannot_be_reversed():
    facts = daily_facts(daily())
    for extra in ("两个区间均为上涨。", "这些因素支持本次判断。", "这些因素削弱本次判断。"):
        raw = response(facts)
        raw["analysis"] += extra
        with pytest.raises(ValueError, match="UNSUPPORTED_ASSERTION"):
            validate_response(raw, facts)


def test_legacy_daily_keeps_combined_down_and_flat_boundary():
    facts = daily_facts(daily(), ternary=False)
    assert any("下跌与持平合并" in item for item in facts["limitations"])
    assert not any("相等算持平" in item for item in facts["limitations"])


def test_linear_multi_restores_winner_and_opposing_contributions():
    p = baseline("UP")
    p["featureSnapshot"]["features"] = dict.fromkeys(KEYS, 0)
    p["featureSnapshot"]["features"].update(return_5d=0.2, return_20d=0.1)
    p["modelManifest"] = {
        "adapter": "LOGISTIC_MULTICLASS_V2",
        "features": KEYS,
        "directionPolicySnapshot": {"tieBreakOrder": ["FLAT", "UP", "DOWN"]},
        "parameters": {
            "classes": ["DOWN", "FLAT", "UP"],
            "mean": [0] * 7,
            "scale": [1] * 7,
            "coefficients": [[0] * 7, [0] * 7, [2, -1, 0, 0, 0, 0, 0]],
            "intercepts": [0, 0, 0],
        },
    }
    facts = multi_facts(p)
    assert "累计分别上涨 20.00% 和 10.00%" in facts["facts"]["F1"]
    p["direction"] = "DOWN"
    with pytest.raises(ValueError, match="RESTORE_MISMATCH"):
        multi_facts(p)


@pytest.mark.parametrize(
    "status,choice",
    [
        (429, None),
        (200, {"finish_reason": "length", "message": {"content": "{}"}}),
        (200, {"finish_reason": "stop", "message": {"content": ""}}),
    ],
)
def test_provider_errors_empty_and_truncated_outputs_are_not_retried(monkeypatch, status, choice):
    real_client = httpx.Client
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"choices": [choice]})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    settings = Settings(_env_file=None, deepseek_api_key="test-key", deepseek_model="test-model")
    with pytest.raises(ValueError):
        generate(daily_facts(daily()), settings)
    assert len(calls) == 1


def test_http_contract_is_bounded_json_and_excludes_model_parameters(monkeypatch):
    facts = daily_facts(daily())
    calls = []
    real_client = httpx.Client

    def handler(request):
        import json

        body = json.loads(request.content)
        calls.append(body)
        assert body["thinking"] == {"type": "disabled"}
        assert body["temperature"] == 0.2
        assert body["response_format"] == {"type": "json_object"}
        assert "contribution" not in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(response(facts), ensure_ascii=False)}}
                ]
            },
        )

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    settings = Settings(_env_file=None, deepseek_api_key="test-key", deepseek_model="test-model")
    assert generate(facts, settings)["summary"] == facts["summary"]
    assert len(calls) == 1


def prepare_service(monkeypatch, *, row=None):
    body = baseline()
    source_id = uuid4()
    source_hash = fingerprint(body)
    monkeypatch.setattr(service, "source", lambda *_: (body, source_hash))
    monkeypatch.setattr(service.store, "read", lambda *_: row)
    monkeypatch.setattr(service.store, "previous", lambda *_: None)
    monkeypatch.setattr(
        service,
        "get_settings",
        lambda: Settings(_env_file=None, deepseek_api_key="test-key", deepseek_model="test-model"),
    )
    return source_id, source_hash


def test_success_cache_does_not_call_deepseek_or_claim_again(monkeypatch):
    source_id, original_hash = prepare_service(monkeypatch)
    narrative = {"summary": "已保存"}
    row = {
        "state": "READY",
        "source_hash": original_hash,
        "fund_code": "002112",
        "payload": narrative,
        "content_hash": fingerprint(narrative),
    }
    monkeypatch.setattr(service.store, "read", lambda *_: row)
    monkeypatch.setattr(service, "generate", lambda *_: pytest.fail("缓存不能再次调用第三方"))
    monkeypatch.setattr(service.store, "claim", lambda *_: pytest.fail("缓存不能重复领取"))
    assert service.ensure("multi", source_id, "002112")["narrative"] == narrative


def test_provider_failure_keeps_original_and_records_cooldown(monkeypatch):
    source_id, _ = prepare_service(monkeypatch)
    finished = []
    monkeypatch.setattr(service.store, "claim", lambda *_: uuid4())
    monkeypatch.setattr(service.store, "finish", lambda *args, **kwargs: finished.append(kwargs))
    monkeypatch.setattr(service, "generate", lambda *_: (_ for _ in ()).throw(ValueError("NARRATIVE_CONTENT_INVALID")))
    assert service.ensure("multi", source_id, "002112")["state"] == "FALLBACK"
    assert finished == [{"failure": "NARRATIVE_CONTENT_INVALID"}]


def test_cooldown_and_missing_config_do_not_call_provider(monkeypatch):
    row = {
        "state": "FAILED",
        "attempts": 1,
        "retry_after": datetime.now(UTC) + timedelta(minutes=10),
        "style_version": STYLE_VERSION,
    }
    source_id, _ = prepare_service(monkeypatch, row=row)
    monkeypatch.setattr(service, "generate", lambda *_: pytest.fail("禁止外部调用"))
    assert service.ensure("multi", source_id, "002112")["state"] == "FALLBACK"
    monkeypatch.setattr(
        service,
        "get_settings",
        lambda: SimpleNamespace(deepseek_api_key=SimpleNamespace(get_secret_value=lambda: ""), deepseek_model=""),
    )
    assert service.ensure("multi", source_id, "002112")["state"] == "FALLBACK"


def test_corrupt_saved_explanation_never_displays_or_regenerates(monkeypatch):
    row = {"state": "READY", "source_hash": "wrong", "fund_code": "002112"}
    source_id, _ = prepare_service(monkeypatch, row=row)
    monkeypatch.setattr(service, "generate", lambda *_: pytest.fail("原文错配不能生成"))
    assert service.ensure("multi", source_id, "002112")["state"] == "FALLBACK"


@pytest.mark.parametrize("outcome", ["pending", "failed", "ready"])
def test_content_upgrade_preserves_old_row_but_never_displays_old_causal_text(monkeypatch, outcome):
    source_id, original_hash = prepare_service(monkeypatch)
    old = {"summary": "旧版已保存", "styleVersion": "PREDICTION_NARRATIVE_ZH_V2"}
    prior = {
        "state": "READY",
        "source_hash": original_hash,
        "fund_code": "002112",
        "payload": old,
        "content_hash": fingerprint(old),
    }
    monkeypatch.setattr(service.store, "previous", lambda *_: prior)
    monkeypatch.setattr(service.store, "claim", lambda *_: None if outcome == "pending" else uuid4())
    monkeypatch.setattr(service.store, "finish", lambda *args, **kwargs: True)
    new = {"summary": "新版带原因", "styleVersion": STYLE_VERSION}

    def generate(*_):
        if outcome == "failed":
            raise ValueError("NARRATIVE_CONTENT_INVALID")
        return new

    monkeypatch.setattr(service, "generate", generate)
    result = service.ensure("multi", source_id, "002112")
    assert result["narrative"] == new if outcome == "ready" else result["narrative"] != old
    assert result["narrative"]["styleVersion"] == STYLE_VERSION
    assert result["state"] == {"pending": "PENDING", "failed": "FALLBACK", "ready": "READY"}[outcome]
    assert prior["payload"] == old
