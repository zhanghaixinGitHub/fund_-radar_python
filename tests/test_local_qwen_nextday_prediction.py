"""当前一日诊断的重要边界：下一日不换基准、只读、全文及未知首次不重试。"""

from datetime import datetime

import pytest
from scripts.local_qwen_nextday_inference import render_input, run
from scripts.local_qwen_nextday_prediction import ZONE, forward_time, guard_sql, next_window, save


def test_next_session_keeps_unclosed_today_as_baseline():
    now = datetime(2026, 10, 13, 11, 35, tzinfo=ZONE)
    window = next_window(now)
    assert window["target_nav_date"] == "2026-10-14"
    assert window["base_nav_date"] == "2026-10-13"
    timing = forward_time(
        {
            "as_of": now.isoformat(),
            "window": window,
            "latest_nav_date": "2026-10-12",
            "nav": [{"nav_date": "2026-10-12"}],
            "companies": [],
            "market": {},
        }
    )
    assert timing["baseline_close_status"] == "NOT_YET_OCCURRED"
    assert timing["baseline_date"] != timing["latest_nav_date"]


def test_write_statements_are_blocked_before_database_execution():
    guard_sql("SELECT clock_timestamp()")
    guard_sql("SET TRANSACTION READ ONLY")
    for statement in (
        "INSERT INTO x VALUES (1)",
        "UPDATE x SET a=1",
        "DELETE FROM x",
        "WITH q AS (SELECT 1) UPDATE x SET a=1",
    ):
        with pytest.raises(ValueError, match="WRITE_FORBIDDEN"):
            guard_sql(statement)


def test_full_document_tail_and_empty_cell_layout_survive_input_render():
    data = {k: {} for k in ("window", "time_context", "report", "market")}
    data.update(
        fund_code="002112",
        fund_name="测试基金",
        as_of="2026-10-13T11:35:00+08:00",
        latest_nav_date="2026-10-12",
        companies=[],
        inventory=[],
        nav=[],
    )
    body = "完整原文" * 6000 + "末尾否定：尚未购买"
    content = render_input(
        data,
        {"D001": {"body": body, "title": "测试", "body_truncated": False}},
        {},
        {"D001": [{"cells": [[1, 1, 1, 4, "合计"], [1, 1, 5, 5, "85.73"]]}]},
    )
    assert body in content
    assert content.count(body) == 1
    assert '"合计"' in content


def test_unknown_first_result_is_saved_and_never_resent(tmp_path):
    save(tmp_path / "manifest.json", {})
    save(tmp_path / "request.json", {"model": "local-only"})
    calls = []

    def transport(request):
        calls.append(request)
        raise TimeoutError()

    # 本次测试不访问Ollama；运行后的元数据读回由测试替身提供。
    import scripts.local_qwen_nextday_inference as module

    original = module.local_api
    module.local_api = lambda *args, **kwargs: {"models": []}
    try:
        run(tmp_path, transport)
        with pytest.raises(ValueError, match="NO_RETRY"):
            run(tmp_path, transport)
    finally:
        module.local_api = original
    assert len(calls) == 1
    assert (tmp_path / "raw/first.json").exists()
