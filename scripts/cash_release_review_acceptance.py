"""本机真实报告的只读TCP验收；不新增回执，不启动训练，不输出服务Token或模型系数。"""

import argparse
import json
import socket
import sys
from contextlib import ExitStack
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from uuid import UUID

import httpx
import uvicorn
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.db.session import get_nav_preview_engine  # noqa: E402
from app.schemas.cash_prediction_check import CashPredictionCheckRequest  # noqa: E402
from app.schemas.cash_release_review import CashReleaseReview  # noqa: E402
from app.services.cash_reinvestment_research import BLOCKERS, FUNDS  # noqa: E402

from scripts.cash_prediction_attempt_acceptance import stop_http  # noqa: E402


def snapshot(engine, run_id):
    """只核对表计数及明确一份研究的指纹，不读取任何净值/样本/标签数值。"""
    with engine.connect() as conn, conn.begin():
        conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        counts = {
            table: conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
            for table in (
                "cash_sample_batch",
                "cash_sample",
                "cash_sample_label",
                "cash_research_run",
                "cash_prediction_attempt",
                "cash_forecast_result",
                "forecast_result",
                "analysis_model_release",
            )
        }
        report_hash = conn.execute(
            text("SELECT md5(report::text) FROM cash_research_run WHERE run_id=:id"), {"id": run_id}
        ).scalar_one()
    return {"table_counts": counts, "research_content_hash": report_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-run-id", type=UUID, required=True)
    parser.add_argument("--expected-report-hash", required=True)
    args = parser.parse_args()
    settings = get_settings()
    url = make_url(settings.ai_database_url)
    if url.host not in {"localhost", "127.0.0.1", "::1"} or url.database != "fund_ai":
        raise RuntimeError("acceptance is restricted to local fund_ai")
    engine = create_engine(
        url, hide_parameters=True, connect_args={"connect_timeout": 5, "options": "-c statement_timeout=5000"}
    )
    service_engine = get_nav_preview_engine()
    statements, summaries = [], []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)  # 不保存绑定参数；最后仅输出数量和核验结果。

    checks = 0
    event.listen(service_engine, "before_cursor_execute", capture)
    try:
        before = snapshot(engine, args.research_run_id)
        with socket.socket() as listener, ExitStack() as shutdown:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            server = uvicorn.Server(uvicorn.Config("app.main:app", log_level="warning", access_log=False))
            thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
            thread.start()
            shutdown.callback(stop_http, server, thread)
            deadline = monotonic() + 15
            while not server.started:
                if not thread.is_alive() or monotonic() > deadline:
                    raise RuntimeError("temporary HTTP server did not start")
                sleep(0.05)
            with httpx.Client(
                base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=15, trust_env=False
            ) as client:
                path = "/internal/v1/predictions/release-review"
                headers = {"X-Service-Token": settings.ai_service_token.get_secret_value()}
                for fund in FUNDS:
                    req = CashPredictionCheckRequest(
                        fundCode=fund, researchRunId=args.research_run_id, expectedReportHash=args.expected_report_hash
                    )
                    payload = req.model_dump(mode="json", by_alias=True)
                    first = client.post(path, json=payload, headers=headers)
                    second = client.post(path, json=payload, headers=headers)
                    assert first.status_code == second.status_code == 200
                    assert first.headers["cache-control"] == "no-store" and first.headers["X-Trace-Id"]
                    result = CashReleaseReview.model_validate(first.json())
                    repeated = CashReleaseReview.model_validate(second.json())
                    assert result.model_dump(exclude={"checked_at"}) == repeated.model_dump(exclude={"checked_at"})
                    assert set(BLOCKERS).issubset(result.blocking_codes)
                    assert result.report_hash == args.expected_report_hash and result.status == "BLOCKED"
                    assert not any(
                        (
                            result.publication_allowed,
                            result.policy_persisted,
                            result.database_written,
                            result.inference_executed,
                            result.independent_test_read,
                        )
                    )
                    coverage = [c for c in result.checks if c.code == "EXAM_COVERAGE"]
                    assert len(coverage) == 9 and all(c.status == "MISSING" and c.actual is None for c in coverage)
                    assert all(c.window_id == "VALIDATION_2024" for c in result.checks if c.code.startswith("ANNUAL_"))
                    summaries.append(
                        {
                            "fund": fund,
                            "policy_hash": result.policy_hash,
                            "policy_state": result.policy.approval_state,
                            "check_counts": result.check_counts,
                            "blocking_codes": result.blocking_codes,
                        }
                    )
                    checks += 2
                    # 原生成前检查也必须通过同一套报告完整性核验，但不能启用草案门槛或产生数字。
                    original = client.post("/internal/v1/predictions/generation-check", json=payload, headers=headers)
                    assert original.status_code == 200
                    original_check = original.json()
                    assert original_check["status"] == "GENERATION_BLOCKED"
                    assert original_check["up_probability"] is original_check["direction"] is None
                    assert not original_check["inference_executed"] and not original_check["database_written"]
                    assert set(original_check["blocking_codes"]).issubset(result.blocking_codes)
                    checks += 1
                queries_before_auth = len(statements)
                for auth in (
                    {},
                    {"X-Service-Token": "invalid-acceptance-token"},
                    {**headers, "Origin": "http://localhost"},
                ):
                    assert client.post(path, json=payload, headers=auth).status_code == 403
                    checks += 1
                for extra in ({"force": True}, {"includeTest": True}, {"policy": {}}, {"coverage": []}):
                    assert client.post(path, json={**payload, **extra}, headers=headers).status_code == 422
                    checks += 1
                assert len(statements) == queries_before_auth
                assert (
                    client.post(path, json={**payload, "expectedReportHash": "f" * 64}, headers=headers).status_code
                    == 409
                )
                assert (
                    client.post(path, json={**payload, "researchRunId": str(UUID(int=0))}, headers=headers).status_code
                    == 404
                )
                checks += 2
        after = snapshot(engine, args.research_run_id)
        assert before == after
        assert all(sql.lstrip().upper().startswith(("SET", "SELECT")) for sql in statements)
        assert all(
            not any(t in sql for t in ("unit_nav", "cash_sample", "cash_forecast", "analysis_model_release"))
            for sql in statements
        )
        checks += 3  # 数据快照一致、仅只读语句、无净值数值/样本/预测表读取。
        print(
            json.dumps(
                {
                    "http_and_integrity_checks": checks,
                    "checks": summaries,
                    "snapshot": after,
                    "captured_sql_count": len(statements),
                    "database_unchanged": True,
                    "temporary_server_stopped": not thread.is_alive(),
                },
                ensure_ascii=False,
            )
        )
    finally:
        event.remove(service_engine, "before_cursor_execute", capture)
        engine.dispose()


if __name__ == "__main__":
    main()
