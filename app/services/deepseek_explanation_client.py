"""DeepSeek V4-Pro 的受控解释客户端；只接受服务端构造的已发布评分事实。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

DEEPSEEK_PROVIDER = "DEEPSEEK"
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_EXPLANATION_PROMPT_VERSION = "M3_DEEPSEEK_EXPLANATION_V1"


class DeepSeekExplanationError(RuntimeError):
    """DeepSeek 配置、可用性或结构化输出不满足解释任务要求时抛出稳定错误码。"""


@dataclass(frozen=True)
class DeepSeekExplanationContent:
    """经过本地字段与长度校验后的模型输出，不保留自由格式外部响应。"""

    provider_request_id: str | None
    overview: str
    evidence: tuple[dict[str, str], ...]
    risk_notice: str
    data_gap: str
    disclaimer: str
    prompt_tokens: int | None
    completion_tokens: int | None


class DeepSeekExplanationClient:
    """调用固定 V4-Pro 模型生成 JSON 解释；调用方负责输入事实与持久化。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def validate_configuration(self) -> None:
        """验证本地运行配置；不输出或返回密钥内容。"""
        if not self._settings.deepseek_enabled:
            raise DeepSeekExplanationError("DEEPSEEK_DISABLED")
        if not self._settings.deepseek_api_key.get_secret_value().strip():
            raise DeepSeekExplanationError("DEEPSEEK_API_KEY_MISSING")
        if self._settings.deepseek_model != DEEPSEEK_MODEL:
            raise DeepSeekExplanationError("DEEPSEEK_MODEL_NOT_ALLOWED")
        if not self._settings.deepseek_base_url.startswith("https://"):
            raise DeepSeekExplanationError("DEEPSEEK_BASE_URL_INVALID")

    @property
    def model(self) -> str:
        """返回当前受控模型名，始终是 DeepSeek V4-Pro。"""
        return self._settings.deepseek_model

    def generate(self, facts: dict[str, object]) -> DeepSeekExplanationContent:
        """调用 Chat Completions JSON 模式并将外部失败归一为安全错误码。"""
        self.validate_configuration()
        request_payload = {
            "model": self._settings.deepseek_model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": "以下是系统已验证的基金评分事实 JSON。只能依据其中事实解释，不能补充外部信息。\n"
                    + json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                },
            ],
            # 解释任务只是格式受限的已验证事实转述；关闭展开式思考可避免
            # 其占用输出上限、导致 JSON 被截断，同时不改变固定 V4-Pro 模型。
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": self._settings.deepseek_max_tokens,
            "stream": False,
        }
        response = self._post_with_limited_retry(request_payload)
        return _parse_response(response)

    def _post_with_limited_retry(self, request_payload: dict[str, object]) -> dict[str, Any]:
        """仅对暂时性故障做有限退避重试，避免在下游拥塞时放大流量。"""
        url = f"{self._settings.deepseek_base_url.rstrip('/')}/chat/completions"
        timeout = httpx.Timeout(
            connect=self._settings.deepseek_connect_timeout_seconds,
            read=self._settings.deepseek_read_timeout_seconds,
            write=self._settings.deepseek_read_timeout_seconds,
            pool=self._settings.deepseek_connect_timeout_seconds,
        )
        for attempt in range(self._settings.deepseek_max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        url,
                        headers={
                            "Authorization": f"Bearer {self._settings.deepseek_api_key.get_secret_value()}",
                            "Content-Type": "application/json",
                        },
                        json=request_payload,
                    )
            except httpx.HTTPError as error:
                if attempt < self._settings.deepseek_max_retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                logger.warning(
                    "deepseek_explanation_client._post_with_limited_retry >>> transport unavailable, "
                    "attempts=%s, error_type=%s",
                    attempt + 1,
                    type(error).__name__,
                )
                raise DeepSeekExplanationError("DEEPSEEK_UNAVAILABLE") from error
            if response.status_code in {429, 500, 503} and attempt < self._settings.deepseek_max_retries:
                time.sleep(0.25 * (attempt + 1))
                continue
            if response.status_code != 200:
                raise DeepSeekExplanationError(_error_code_for_status(response.status_code))
            try:
                payload = response.json()
            except ValueError as error:
                raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_INVALID") from error
            if not isinstance(payload, dict):
                raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_INVALID")
            return payload
        raise DeepSeekExplanationError("DEEPSEEK_UNAVAILABLE")


def _system_prompt() -> str:
    """返回固定 JSON 指令，阻止模型将解释扩展为买卖建议或未提供事实。"""
    return """
你是基金雷达的受控解释助手。你只能解释用户输入 JSON 中的已验证事实，绝不使用外部知识、绝不猜测缺失信息。
不得给出买入、卖出、持有、止盈、止损、仓位、收益保证或确定性预测。不得改写或计算输入中的方向、概率、置信度。
必须输出一个 JSON 对象，且仅包含 overview、evidence、risk_notice、data_gap、disclaimer 五个字段。
overview、risk_notice、data_gap、disclaimer 均为简短中文字符串；
evidence 是 1 至 4 项数组，每项只含 label 和 detail 两个中文字符串。
disclaimer 必须明确“仅供信息参考，不构成交易建议”。若输入事实不足，data_gap 必须说明数据不足，不得补全。
JSON 示例：
{"overview":"...","evidence":[{"label":"数据截至日","detail":"..."}],"risk_notice":"...","data_gap":"...","disclaimer":"仅供信息参考，不构成交易建议。"}
""".strip()


def _parse_response(payload: dict[str, Any]) -> DeepSeekExplanationContent:
    """解析并严格白名单模型 JSON 输出，拒绝空内容、截断和未约束字段。"""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_INVALID")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_TRUNCATED")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_EMPTY")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_INVALID") from error
    if not isinstance(parsed, dict) or set(parsed) != {"overview", "evidence", "risk_notice", "data_gap", "disclaimer"}:
        raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_SCHEMA_INVALID")
    evidence = _normalize_evidence(parsed.get("evidence"))
    usage = payload.get("usage")
    provider_request_id = payload.get("id")
    return DeepSeekExplanationContent(
        provider_request_id=_optional_text(provider_request_id, 128),
        overview=_required_text(parsed.get("overview"), 1_200),
        evidence=evidence,
        risk_notice=_required_text(parsed.get("risk_notice"), 800),
        data_gap=_required_text(parsed.get("data_gap"), 800),
        disclaimer=_required_text(parsed.get("disclaimer"), 300),
        prompt_tokens=_optional_non_negative_int(usage.get("prompt_tokens") if isinstance(usage, dict) else None),
        completion_tokens=_optional_non_negative_int(
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        ),
    )


def _normalize_evidence(value: object) -> tuple[dict[str, str], ...]:
    """限制证据条数、字段与长度，避免模型输出任意嵌套结构进入数据库或页面。"""
    if not isinstance(value, list) or not 1 <= len(value) <= 4:
        raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_SCHEMA_INVALID")
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"label", "detail"}:
            raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_SCHEMA_INVALID")
        normalized.append(
            {"label": _required_text(item.get("label"), 80), "detail": _required_text(item.get("detail"), 400)}
        )
    return tuple(normalized)


def _required_text(value: object, max_length: int) -> str:
    """只接受长度受限的非空文本，避免透传未受控内容。"""
    if not isinstance(value, str):
        raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_SCHEMA_INVALID")
    clean_value = value.strip()
    if not clean_value or len(clean_value) > max_length:
        raise DeepSeekExplanationError("DEEPSEEK_RESPONSE_SCHEMA_INVALID")
    return clean_value


def _optional_text(value: object, max_length: int) -> str | None:
    """外部请求标识缺失时允许为空，存在时仍限制长度。"""
    if value is None:
        return None
    return _required_text(value, max_length)


def _optional_non_negative_int(value: object) -> int | None:
    """Token 用量仅接受非负整数，异常格式不影响解释正文。"""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _error_code_for_status(status_code: int) -> str:
    """将 DeepSeek HTTP 状态归一化，避免向上游传播账号、余额或请求细节。"""
    return {
        400: "DEEPSEEK_REQUEST_INVALID",
        401: "DEEPSEEK_AUTH_FAILED",
        402: "DEEPSEEK_BALANCE_INSUFFICIENT",
        422: "DEEPSEEK_REQUEST_INVALID",
        429: "DEEPSEEK_RATE_LIMITED",
        500: "DEEPSEEK_UNAVAILABLE",
        503: "DEEPSEEK_UNAVAILABLE",
    }.get(status_code, "DEEPSEEK_UNAVAILABLE")
