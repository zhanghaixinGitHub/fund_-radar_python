"""聊天结果的原文覆盖、拒绝伪造和停用付费回退；使用隔离样例，不调用供应商。"""

from datetime import datetime
from types import SimpleNamespace

import pytest
from app.core import deepseek_policy
from app.core.config import Settings
from app.services.fund_codex_analysis import reviewed_events


def data():
    return {
        "documents": [
            {
                "id": "doc-a",
                "title": "公告",
                "kind": "ANNOUNCEMENT",
                "body_scope": "FULL_TEXT",
                "body": "计划尚未执行。仍需股东大会批准。",
                "source_hash": "source",
                "date": "2026-10-14",
            }
        ]
    }


def review(scope="EXCERPTS", ranges=None, events=None):
    return {"documents": [{"id": "doc-a", "scope": scope, "ranges": ranges or [], "events": events or []}]}


def test_all_source_documents_must_have_explicit_review_scope():
    with pytest.raises(ValueError, match="MANUAL_DOCUMENT_COVERAGE_REQUIRED"):
        reviewed_events(data(), {"documents": []})


def test_excerpt_cannot_be_misrepresented_as_full_reading():
    with pytest.raises(ValueError, match="MANUAL_FULL_REVIEW_INCOMPLETE"):
        reviewed_events(data(), review("FULL_TEXT", [{"start": 0, "end": 8}]))


def test_excerpt_and_unread_documents_remain_explicit_limitations():
    events, coverage = reviewed_events(data(), review("EXCERPTS", [{"start": 0, "end": 8}]))
    assert not events and coverage[0]["reasonCode"] == "MANUAL_EXCERPT_REVIEW"
    events, coverage = reviewed_events(data(), review("NOT_REVIEWED"))
    assert not events and coverage[0]["reasonCode"] == "MANUAL_NOT_REVIEWED"


def test_quotes_outside_read_ranges_are_rejected_before_event_validation():
    with pytest.raises(ValueError, match="MANUAL_QUOTE_OUTSIDE_REVIEW"):
        reviewed_events(data(), review("EXCERPTS", [{"start": 0, "end": 8}], [{"quotes": ["仍需股东大会批准。"]}]))


def test_missing_sections_cannot_be_disguised_as_complete_reading():
    with pytest.raises(ValueError, match="MANUAL_FULL_REVIEW_INCOMPLETE"):
        reviewed_events(data(), review("FULL_TEXT", [{"start": 1, "end": len(data()["documents"][0]["body"])}]))


def test_public_result_preserves_all_used_conditions_without_transmitting_unused_originals():
    from copy import deepcopy

    from app.services.fund_information_analysis import referenced_facts

    facts = {
        "used": {"text": "计划尚未实施，仍需批准。", "source": {"url": "https://example.com/plan"}},
        "counter": {"text": "现金流弱于利润，尚有收款风险。"},
        "unused": {"text": "未采用的完整资料" * 100000},
    }
    original = deepcopy(facts)
    analysis = {"reasons": [{"refs": ["used"]}], "counterpoints": [{"refs": ["counter", "used"]}]}
    selected = referenced_facts(analysis, facts)
    assert selected == {"counter": facts["counter"], "used": facts["used"]}
    assert facts == original
    assert "尚未实施，仍需批准" in selected["used"]["text"]
    analysis["reasons"][0]["refs"].append("missing")
    with pytest.raises(ValueError, match="ANALYSIS_REFERENCE_INVALID"):
        referenced_facts(analysis, facts)


def test_manual_mode_is_default_and_private_key_does_not_enable_paid_calls(monkeypatch):
    assert Settings(_env_file=None).fund_002112_analysis_mode == "CODEX_MANUAL"
    settings = deepseek_policy.PrivateDeepSeekSettings(_env_file=None, api_key="test-only", enabled=False)
    monkeypatch.setattr(deepseek_policy, "PrivateDeepSeekSettings", lambda: settings)
    with (
        deepseek_policy.prediction_scope("002112", 1, "2026-10-14"),
        pytest.raises(ValueError, match="DEEPSEEK_DISABLED"),
    ):
        deepseek_policy.provider_settings("SYNTHESIS", {})


def test_ready_result_never_falls_back_to_paid_provider(monkeypatch):
    from contextlib import nullcontext

    from app.services import fund_codex_analysis as service

    result = SimpleNamespace(mappings=lambda: SimpleNamespace(first=lambda: None))
    connection = SimpleNamespace(execute=lambda *args, **kwargs: result)
    monkeypatch.setattr(service, "get_engine", lambda: SimpleNamespace(connect=lambda: nullcontext(connection)))
    with pytest.raises(ValueError, match="MANUAL_ANALYSIS_REQUIRED"):
        service.ready_result("2026-10-14")


@pytest.mark.parametrize(
    "field,value",
    [
        ("fund_code", "000001"),
        ("target_nav_date", "2026-10-15"),
        ("generation_source", "DEEPSEEK"),
        ("deadline_at", "2026-10-14T09:00:00+08:00"),
    ],
)
def test_ready_result_rejects_scope_source_or_deadline_mismatch(monkeypatch, field, value):
    from contextlib import nullcontext

    from app.services import fund_codex_analysis as service
    from app.services.direction_1d_protocol import canonical, digest

    payload = {
        "fund_code": "002112",
        "target_nav_date": "2026-10-14",
        "generation_source": service.SOURCE,
        "deadline_at": "2026-10-14T15:00:00+08:00",
        field: value,
    }
    output = {"payload_json": canonical(payload), "content_hash": digest(payload)}
    result = SimpleNamespace(mappings=lambda: SimpleNamespace(first=lambda: {"result": output}))
    connection = SimpleNamespace(execute=lambda *args, **kwargs: result)
    monkeypatch.setattr(service, "get_engine", lambda: SimpleNamespace(connect=lambda: nullcontext(connection)))
    monkeypatch.setattr(service.repo, "clock", lambda: datetime.fromisoformat("2026-10-14T10:00:00+08:00"))
    with pytest.raises(ValueError, match="MANUAL_RESULT_INVALID"):
        service.ready_result("2026-10-14")
