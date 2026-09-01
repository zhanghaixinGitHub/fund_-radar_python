"""DeepSeek 解释客户端的离线结构化输出验证；不发起真实外部请求。"""

import pytest
from app.core.config import Settings
from app.services.deepseek_explanation_client import (
    DeepSeekExplanationClient,
    DeepSeekExplanationError,
    _parse_response,
)
from pydantic import SecretStr


def _response(content: str, *, finish_reason: str = "stop") -> dict[str, object]:
    """构造 DeepSeek Chat Completions 的最小成功响应。"""
    return {
        "id": "chatcmpl-safe-001",
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 240},
    }


def test_parse_response_keeps_only_expected_explanation_schema() -> None:
    """合法 JSON 只映射五个预定字段与有限证据，不保留模型附加结构。"""
    content = """{
      "overview":"评分仅反映已发布的结构化结果。",
      "evidence":[{"label":"数据截至日","detail":"2026-08-31"}],
      "risk_notice":"历史表现不代表未来。",
      "data_gap":"当前未包含个人持仓或外部资讯。",
      "disclaimer":"仅供信息参考，不构成交易建议。"
    }"""

    parsed = _parse_response(_response(content))

    assert parsed.provider_request_id == "chatcmpl-safe-001"
    assert parsed.evidence == ({"label": "数据截至日", "detail": "2026-08-31"},)
    assert parsed.prompt_tokens == 120
    assert parsed.completion_tokens == 240


def test_parse_response_rejects_extra_fields_and_truncated_output() -> None:
    """模型额外字段或被截断的内容不得进入解释快照。"""
    with pytest.raises(DeepSeekExplanationError, match="DEEPSEEK_RESPONSE_SCHEMA_INVALID"):
        _parse_response(
            _response(
                """{
                  "overview":"x", "evidence":[{"label":"a","detail":"b"}],
                  "risk_notice":"x", "data_gap":"x", "disclaimer":"x", "buy_signal":"x"
                }"""
            )
        )
    with pytest.raises(DeepSeekExplanationError, match="DEEPSEEK_RESPONSE_TRUNCATED"):
        _parse_response(_response("{}", finish_reason="length"))


def test_generate_uses_fixed_model_without_reasoning_expansion_or_user_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """外部请求只携带受控模型参数，不能引入用户标识或展开式思考。"""
    client = DeepSeekExplanationClient(
        Settings(deepseek_enabled=True, deepseek_api_key=SecretStr("test-key"), deepseek_model="deepseek-v4-pro")
    )
    captured: dict[str, object] = {}

    def fake_post(payload: dict[str, object]) -> dict[str, object]:
        captured.update(payload)
        return _response(
            """{
              "overview":"评分仅反映已发布的结构化结果。",
              "evidence":[{"label":"数据截至日","detail":"2026-08-31"}],
              "risk_notice":"历史表现不代表未来。",
              "data_gap":"当前未包含个人持仓或外部资讯。",
              "disclaimer":"仅供信息参考，不构成交易建议。"
            }"""
        )

    monkeypatch.setattr(client, "_post_with_limited_retry", fake_post)

    client.generate({"fund_code": "000001"})

    assert captured["model"] == "deepseek-v4-pro"
    assert captured["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in captured
    assert "user_id" not in captured
