"""本机真实HTTP拒绝回执验收；只新增三试点回执，不修改净值/样本/模型，不打印凭据。"""

import argparse
import json
import re
import socket
import sys
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from uuid import UUID, uuid5

import httpx
import uvicorn
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.cash_reinvestment_research import FUNDS  # noqa: E402


def stop_http(server, thread):
    """先让Uvicorn结束监听，再关闭Windows socket，避免中断挂起的accept。"""
    server.should_exit = True
    thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("temporary HTTP server did not stop")


def snapshot(engine, run_id, cutoff, *, include_forecasts=False):
    """仅对限定范围做完整性摘要；2025内容不离开数据库、不用于训练或评分。"""
    with engine.connect() as conn:
        nav = conn.execute(
            text("""WITH limited AS (
            SELECT n.* FROM nav_daily n
            WHERE fund_code IN ('001632', '006730', '008888') AND nav_date BETWEEN '2022-01-01' AND :cutoff
            ORDER BY fund_code, nav_date, source_id LIMIT 5001)
            SELECT count(*), md5(string_agg(row_to_json(limited)::text, '' ORDER BY fund_code, nav_date, source_id))
            FROM limited"""),
            {"cutoff": cutoff},
        ).one()
        if nav[0] > 5000:
            raise RuntimeError("NAV verification exceeds the fixed 5000-row bound")
        counts = {
            table: conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
            for table in (
                "cash_sample_batch",
                "cash_sample",
                "cash_sample_label",
                "cash_research_run",
                "forecast_result",
                "analysis_model_release",
            )
        }
        report = conn.execute(
            text("SELECT md5(report::text) FROM cash_research_run WHERE run_id=:run"), {"run": run_id}
        ).scalar_one()
        if include_forecasts:
            counts["cash_forecast_result"] = conn.execute(
                text("SELECT count(*) FROM cash_forecast_result")
            ).scalar_one()
    return {"nav_count": nav[0], "nav_hash": nav[1], "table_counts": counts, "research_content_hash": report}


def verify_forecast_rejections(client, headers, args):
    """真正调用生产授权解析器；没有人工授权开关，真实模型必须保持无结果。"""
    from app.services.cash_reinvestment_research import get_cash_research

    stored = get_cash_research(args.research_run_id)
    assert stored.report.report_hash == args.expected_report_hash
    model = next(w.model for w in stored.report.windows if w.window.window_id == "VALIDATION_2024")
    assert model is not None
    path = "/internal/v1/predictions/cash-forecasts"
    for fund in FUNDS:
        body = {
            "fundCode": fund,
            "cutoffDate": args.cutoff_date.isoformat(),
            "researchRunId": str(args.research_run_id),
            "expectedReportHash": args.expected_report_hash,
            "expectedModelHash": model.model_hash,
            "requestKey": str(uuid5(args.request_namespace, "forecast-" + fund)),
        }
        response = client.post(path, json=body, headers=headers)
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        result = response.json()
        assert result["status"] == "MODEL_NOT_RELEASED" and len(result["reason_codes"]) == 6
        assert result["forecast_id"] is result["up_probability"] is result["direction"] is None
        assert not result["created"]
    assert client.post(path, json={**body, "expectedModelHash": "f" * 64}, headers=headers).status_code == 409
    assert client.post(path, json={**body, "force": True}, headers=headers).status_code == 422
    assert client.post(path, json={**body, "cutoffDate": "2025-08-07"}, headers=headers).status_code == 422
    assert client.post(path, json={**body, "fundCode": "000001"}, headers=headers).status_code == 422
    assert client.post(path, json=body).status_code == 403
    assert client.post(path, json=body, headers={**headers, "Origin": "http://localhost"}).status_code == 403
    assert client.get(f"{path}/{uuid5(args.request_namespace, 'absent-forecast')}", headers=headers).status_code == 404
    return 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-run-id", type=UUID, required=True)
    parser.add_argument("--expected-report-hash", required=True)
    parser.add_argument("--cutoff-date", type=date.fromisoformat, required=True)
    parser.add_argument("--request-namespace", type=UUID, required=True, help="相同编号重跑只读回原回执，不追加新回执")
    parser.add_argument("--confirm-write-local-attempts", action="store_true")
    parser.add_argument(
        "--verify-forecast-endpoints", action="store_true", help="要求迁移16；另外核验真实生成请求被拒绝且不写结果"
    )
    args = parser.parse_args()
    settings = get_settings()
    url = make_url(settings.ai_database_url)
    if (
        not args.confirm_write_local_attempts
        or url.host not in {"localhost", "127.0.0.1", "::1"}
        or url.database != "fund_ai"
        or not re.fullmatch(r"[0-9a-f]{64}", args.expected_report_hash)
    ):
        raise RuntimeError("requires explicit local fund_ai write confirmation and a valid report hash")
    engine = create_engine(
        url, hide_parameters=True, connect_args={"connect_timeout": 5, "options": "-c statement_timeout=5000"}
    )
    receipts, checks = [], 0
    try:
        before = snapshot(
            engine, args.research_run_id, args.cutoff_date, include_forecasts=args.verify_forecast_endpoints
        )
        # 预先占有随机本机端口，避免抢占用户8000进程；使用真实TCP而非TestClient。
        with socket.socket() as listener, ExitStack() as shutdown:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config("app.main:app", log_level="warning", access_log=False))
            thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
            thread.start()
            shutdown.callback(stop_http, server, thread)
            deadline = monotonic() + 15
            while not server.started:
                if not thread.is_alive() or monotonic() > deadline:
                    raise RuntimeError("temporary HTTP server did not start")
                sleep(0.05)
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15, trust_env=False) as client:
                headers = {"X-Service-Token": settings.ai_service_token.get_secret_value()}
                path = "/internal/v1/predictions/generation-attempts"
                for fund in FUNDS:
                    body = {
                        "fundCode": fund,
                        "cutoffDate": args.cutoff_date.isoformat(),
                        "researchRunId": str(args.research_run_id),
                        "expectedReportHash": args.expected_report_hash,
                        "requestKey": str(uuid5(args.request_namespace, fund)),
                    }
                    first = client.post(path, json=body, headers=headers)
                    first.raise_for_status()
                    assert first.status_code in (200, 201)
                    result = first.json()
                    repeated = client.post(path, json=body, headers=headers)
                    read = client.get(f"{path}/{result['attempt_id']}", headers=headers)
                    assert repeated.status_code == read.status_code == 200
                    assert repeated.json() == read.json() == result
                    assert all(r.headers["cache-control"] == "no-store" for r in (first, repeated, read))
                    gate = result["check"]
                    assert result["historical_receipt"] and not result["forecast_created"]
                    assert gate["status"] == "GENERATION_BLOCKED" and gate["blocking_codes"]
                    assert gate["up_probability"] is gate["direction"] is None and not gate["inference_executed"]
                    checks += 3
                    receipts.append(
                        {"fund": fund, "attempt_id": result["attempt_id"], "reasons": gate["blocking_codes"]}
                    )
                for auth in (
                    {},
                    {"X-Service-Token": "invalid-acceptance-token"},
                    {**headers, "Origin": "http://localhost"},
                ):
                    assert client.post(path, json=body, headers=auth).status_code == 403
                    checks += 1
                assert client.post(path, json={**body, "force": True}, headers=headers).status_code == 422
                assert client.post(path, json={**body, "cutoffDate": "2025-08-07"}, headers=headers).status_code == 422
                assert client.post(path, json={**body, "fundCode": FUNDS[0]}, headers=headers).status_code == 409
                assert (
                    client.get(f"{path}/{uuid5(args.request_namespace, 'absent')}", headers=headers).status_code == 404
                )
                checks += 4
                if args.verify_forecast_endpoints:
                    checks += verify_forecast_rejections(client, headers, args)
                after = snapshot(
                    engine, args.research_run_id, args.cutoff_date, include_forecasts=args.verify_forecast_endpoints
                )
                assert before == after
                with engine.connect() as conn:
                    actual = conn.execute(
                        text("SELECT count(*) FROM cash_prediction_attempt WHERE request_key = ANY(:keys)"),
                        {"keys": [uuid5(args.request_namespace, fund) for fund in FUNDS]},
                    ).scalar_one()
                    assert actual == len(FUNDS)
        print(
            json.dumps(
                {"http_checks": checks, "receipts": receipts, "before_equals_after": True, "snapshot": after},
                ensure_ascii=False,
            )
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
