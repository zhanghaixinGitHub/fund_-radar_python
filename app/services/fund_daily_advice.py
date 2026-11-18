"""每日建议的正式生成与数据库读回；只生成报告，不调用交易入口。"""

from collections import Counter

from sqlalchemy import text

from app.commands.fund_prediction_followthrough import owner_scope
from app.commands.holding_advice_inventory import core_engine
from app.commands.verify_holding_advice import VERSION
from app.services.fund_exposure_common import now


def read_saved_advice(owner, receipt, *, backend=None, engine_factory=core_engine):
    """独立检查只读正式报告，绝不调用生成接口。"""
    engine, rows = engine_factory(), []
    try:
        with engine.connect() as connection:
            for item in receipt.get("items", []):
                code, report_id = item["fund_code"], item.get("report_id")
                if not report_id:
                    rows.append({"fund_code": code, "read_back": False, "status": item["status"]})
                    continue
                value = connection.execute(text(
                    "SELECT payload FROM portfolio_decision_report "
                    "WHERE user_id=:owner AND fund_code=:code AND report_id=:report"
                ), {"owner": owner, "code": code, "report": report_id}).scalar_one_or_none()
                matched = bool(value and value.get("strategyVersion") == VERSION
                               and value.get("fundCode") == code and value.get("reportId") == report_id
                               and value.get("decision") == item.get("decision")
                               and value.get("adviceState") == item.get("advice_state"))
                applicable, applicability = False, "UNVERIFIED"
                if matched and backend is not None:
                    try:
                        live = backend.get(backend.settings.core_service_base_url
                            + f"/internal/v1/holding-advice/users/{owner}/funds/{code}/current")
                        applicability = (live or {}).get("applicability", "UNVERIFIED")
                        applicable = bool(live and live.get("reportId") == report_id and applicability == "CURRENT")
                    except Exception:
                        applicability = "READ_FAILED"
                rows.append({"fund_code": code, "report_id": report_id, "read_back": matched,
                             "inputs_current": applicable, "applicability": applicability,
                             "status": item["status"]})
    finally:
        engine.dispose()
    return rows


def generate_advice(backend, scope, dependencies, *, engine_factory=core_engine):
    """同轮评分和四周期处理之后执行；未齐的依赖不能借旧建议冒充完成。

    每只独立失败，正式接口保存的原文必须与数据库一致。相同输入由正式接口
    复用原报告；这里保留真实生成时间，不改旧报告日期制造新增。
    """
    rows = []
    engine = engine_factory()
    try:
        for code in scope:
            row = {"fund_code": code, "checked_at": now().isoformat(), "read_back": False}
            try:
                if code not in owner_scope(backend.settings, backend.owner):
                    row.update(status="REMOVED_FROM_HOLDINGS")
                    rows.append(row)
                    continue
                pending = dependencies.get(code, [])
                row["pending_dependencies"] = pending
                # 资料缺项仍交正式规则核对个人风险条件；不能提前吞掉已触及限制的提示。
                response = backend.client.post(
                    backend.settings.core_service_base_url
                    + f"/internal/v1/holding-advice/users/{backend.owner}/funds/{code}/generate",
                    timeout=90,
                )
                response.raise_for_status()
                value = response.json()
                if value.get("fundCode") != code or value.get("strategyVersion") != VERSION:
                    raise ValueError("DAILY_ADVICE_IDENTITY_MISMATCH")
                with engine.connect() as connection:
                    saved = connection.execute(
                        text(
                            "SELECT payload FROM portfolio_decision_report "
                            "WHERE user_id=:owner AND fund_code=:code AND report_id=:report"
                        ),
                        {"owner": backend.owner, "code": code, "report": value["reportId"]},
                    ).scalar_one()
                if saved != value:
                    raise ValueError("DAILY_ADVICE_DATABASE_READBACK_MISMATCH")
                row.update(
                    read_back=True,
                    report_id=value["reportId"],
                    generated_at=value.get("generatedAt"),
                    advice_state=value.get("adviceState"),
                    decision=value.get("decision"),
                    missing=value.get("missingOptionalFactors", []),
                    status=value.get("adviceState") if value.get("adviceState") in {
                        "READY", "QUALIFIED", "INSUFFICIENT", "FAILED"
                    } else "FAILED",
                )
                if pending:
                    row["current_period_complete"] = False
                    # 依赖未齐只接纳明确带缺项的减仓提示，不能用旧周期给出完整买入/持有结论。
                    if not (row["status"] == "QUALIFIED" and row["decision"] == "REDUCE" and row["missing"]):
                        row.update(status="WAITING_INPUTS", missing=pending)
            except Exception as error:
                # 不保存可能带请求头或连接凭据的异常正文。
                row.update(status="FAILED", failure=type(error).__name__)
            rows.append(row)
    finally:
        engine.dispose()
    counts = dict(Counter(row["status"] for row in rows))
    complete = all(row["read_back"] and row["status"] != "FAILED"
                   and not row.get("pending_dependencies") for row in rows)
    return {
        "status": "SUCCEEDED" if complete else "ACTION_REQUIRED",
        "checked_at": now().isoformat(),
        "scope": list(scope),
        "counts": counts,
        "reports_read_back": sum(row["read_back"] for row in rows),
        "actionable_count": sum(row["status"] in {"READY", "QUALIFIED"} for row in rows),
        "items": rows,
    }
