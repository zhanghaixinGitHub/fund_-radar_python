"""本机真实批次的只读考试资料验收；2025只出现计划日期，不读数值、不训练或写库。"""

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
from app.db.session import get_nav_sample_storage_engine  # noqa: E402
from app.schemas.cash_exam_plan import CashExamPreparation, CashExamPreparationRequest  # noqa: E402

from scripts.cash_prediction_attempt_acceptance import stop_http  # noqa: E402
from scripts.cash_release_review_acceptance import snapshot  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-run-id", type=UUID, required=True)
    parser.add_argument("--expected-report-hash", required=True)
    parser.add_argument(
        "--verify-planned-research", action="store_true", help="增加真实草案拒绝与新绑定GET检查，不创建研究"
    )
    args = parser.parse_args()
    settings = get_settings()
    url = make_url(settings.ai_database_url)
    if url.host not in {"localhost", "127.0.0.1", "::1"} or url.database != "fund_ai":
        raise RuntimeError("acceptance is restricted to local fund_ai")
    engine = create_engine(
        url, hide_parameters=True, connect_args={"connect_timeout": 5, "options": "-c statement_timeout=5000"}
    )
    service_engine = get_nav_sample_storage_engine()
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)  # 不保存参数、Token、输入或答案正文。

    event.listen(service_engine, "before_cursor_execute", capture)
    try:
        before = snapshot(engine, args.research_run_id)
        with engine.connect() as conn, conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            batch_ids, dataset_hash, report_hash = conn.execute(
                text(
                    "SELECT report->'preparation'->'batch_ids', dataset_hash, report->>'report_hash' "
                    "FROM cash_research_run WHERE run_id=:id"
                ),
                {"id": args.research_run_id},
            ).one()
            frozen_count = conn.execute(text("SELECT count(*) FROM cash_policy_freeze")).scalar_one()
            planned_count = (
                conn.execute(text("SELECT count(*) FROM cash_planned_research_binding")).scalar_one()
                if args.verify_planned_research
                else None
            )
        assert report_hash == args.expected_report_hash and frozen_count == 0
        checks = 0
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
                base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=45, trust_env=False
            ) as client:
                headers = {"X-Service-Token": settings.ai_service_token.get_secret_value()}
                descriptor = client.get("/internal/v1/predictions/release-policy", headers=headers)
                assert descriptor.status_code == 200
                descriptor = descriptor.json()
                assert descriptor["policy"]["approval_state"] == "DRAFT" and not descriptor["approval_ready"]
                assert not statements  # 日期计划生成不依赖数据库题目好坏。
                plan_hash = descriptor["binding"]["exam_plan"]["plan_hash"]
                request = CashExamPreparationRequest(
                    batchIds=batch_ids, expectedDatasetHash=dataset_hash, expectedPlanHash=plan_hash
                )
                payload = request.model_dump(mode="json", by_alias=True)
                path = "/internal/v1/features/cash-reinvestment/exam-preparation"
                first = client.post(path, json=payload, headers=headers)
                repeated = client.post(path, json=payload, headers=headers)
                assert first.status_code == repeated.status_code == 200
                assert first.json() == repeated.json()
                assert first.headers["cache-control"] == "no-store" and first.headers["X-Trace-Id"]
                result = CashExamPreparation.model_validate(first.json())
                assert result.preparation.dataset_hash == dataset_hash and result.plan.plan_hash == plan_hash
                assert not any(
                    (
                        result.ex_ante_evidence,
                        result.publication_allowed,
                        result.model_fitted,
                        result.independent_test_read,
                        result.database_written,
                    )
                )
                checks += 3  # 日期计划、真实批次准备、重复请求完全一致。
                before_invalid = len(statements)
                for auth in (
                    {},
                    {"X-Service-Token": "invalid-acceptance-token"},
                    {**headers, "Origin": "http://localhost"},
                ):
                    assert client.post(path, json=payload, headers=auth).status_code == 403
                    checks += 1
                for extra in ({"includeTest": True}, {"force": True}, {"coverage": 1}, {"plan": {}}):
                    assert client.post(path, json={**payload, **extra}, headers=headers).status_code == 422
                    checks += 1
                assert (
                    client.post(path, json={**payload, "expectedPlanHash": "f" * 64}, headers=headers).status_code
                    == 409
                )
                assert len(statements) == before_invalid
                checks += 1
                missing = client.post(path, json={**payload, "batchIds": [str(UUID(int=0))]}, headers=headers)
                assert missing.status_code == 404
                conflict = client.post(path, json={**payload, "expectedDatasetHash": "f" * 64}, headers=headers)
                assert conflict.status_code == 409
                checks += 2
                old_review = client.post(
                    "/internal/v1/predictions/release-review",
                    json={
                        "fundCode": "008888",
                        "researchRunId": str(args.research_run_id),
                        "expectedReportHash": report_hash,
                    },
                    headers=headers,
                )
                assert old_review.status_code == 200
                old = old_review.json()
                assert not old["policy_persisted"] and not old["publication_allowed"]
                coverage = [c for c in old["checks"] if c["code"] == "EXAM_COVERAGE"]
                assert len(coverage) == 9 and all(c["status"] == "MISSING" for c in coverage)
                checks += 1  # 旧报告不会借本次事后准备补盖成已有事前冻结证据。
                if args.verify_planned_research:
                    # 前文已明确确认真实规则仍DRAFT；本验收绝不创建APPROVED配置或真实新研究。
                    planned_path = "/internal/v1/features/cash-reinvestment/planned-research-runs"
                    planned_payload = {
                        "requestKey": str(UUID(int=1)),
                        "batchIds": batch_ids,
                        "expectedDatasetHash": dataset_hash,
                        "policyFreezeId": str(UUID(int=0)),
                        "expectedPolicyFreezeHash": "f" * 64,
                    }
                    before_planned = len(statements)
                    rejected = client.post(planned_path, json=planned_payload, headers=headers)
                    assert (
                        rejected.status_code == 409
                        and rejected.json()["detail"]["code"] == "CASH_POLICY_APPROVAL_REQUIRED"
                    )
                    checks += 1
                    for auth in (
                        {},
                        {"X-Service-Token": "invalid-acceptance-token"},
                        {**headers, "Origin": "http://localhost"},
                    ):
                        assert client.post(planned_path, json=planned_payload, headers=auth).status_code == 403
                        assert client.get(planned_path + f"/{UUID(int=0)}", headers=auth).status_code == 403
                        checks += 2
                    for extra in (
                        {"researchRunId": str(args.research_run_id)},
                        {"includeTest": True},
                        {"evaluationStartedAt": "2024-01-01T00:00:00Z"},
                    ):
                        assert (
                            client.post(planned_path, json={**planned_payload, **extra}, headers=headers).status_code
                            == 422
                        )
                        checks += 1
                    assert len(statements) == before_planned
                    assert client.get(planned_path + f"/{UUID(int=0)}", headers=headers).status_code == 404
                    checks += 1
        after = snapshot(engine, args.research_run_id)
        assert before == after
        assert all(sql.lstrip().upper().startswith(("SET", "SELECT")) for sql in statements)
        assert all("nav_daily" not in sql and "forecast_result" not in sql for sql in statements)
        with engine.connect() as conn, conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            assert conn.execute(text("SELECT count(*) FROM cash_policy_freeze")).scalar_one() == frozen_count
            if args.verify_planned_research:
                assert (
                    conn.execute(text("SELECT count(*) FROM cash_planned_research_binding")).scalar_one()
                    == planned_count
                )
        checks += 3
        print(
            json.dumps(
                {
                    "http_and_integrity_checks": checks,
                    "plan_hash": plan_hash,
                    "batch_count": len(batch_ids),
                    "coverage": [g.model_dump(mode="json", exclude={"missing_cutoffs"}) for g in result.coverage],
                    "captured_sql_count": len(statements),
                    "snapshot": after,
                    "policy_freeze_count": frozen_count,
                    "planned_research_count": planned_count,
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
