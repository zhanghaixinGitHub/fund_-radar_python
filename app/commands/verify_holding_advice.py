"""经已授权的正式入口逐只保存建议，再只读核对原文和账务边界；不提交交易。"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import text

from app.commands.fund_prediction_followthrough import owner_scope
from app.commands.holding_advice_inventory import PROJECT, core_engine
from app.core.config import get_settings

VERSION = "HOLDING_ADVICE_V4_EVIDENCE"
DISPLAY_FIELDS = (
    "reportId",
    "fundCode",
    "strategyVersion",
    "generationStatus",
    "adviceState",
    "decision",
    "summary",
    "generatedAt",
    "validUntil",
    "dataAsOf",
    "qualityAssessment",
    "timingAssessment",
    "personalAssessment",
    "supportingEvidence",
    "opposingEvidence",
    "neutralEvidence",
    "facts",
    "missingOptionalFactors",
    "executionConstraints",
    "reassessmentConditions",
    "periodAssessments",
    "modelRefs",
    "preference",
    "defaultPreference",
)


def proof(engine, owner):
    queries = {
        "orders": "SELECT to_jsonb(t) data FROM sim_order t WHERE user_id=:u ORDER BY order_id",
        "plans": "SELECT to_jsonb(t) data FROM sim_plan t WHERE user_id=:u ORDER BY plan_id",
        "periods": (
            "SELECT to_jsonb(t) data FROM sim_plan_execution t JOIN sim_plan p USING(plan_id) "
            "WHERE p.user_id=:u ORDER BY execution_id"
        ),
        "ledger": "SELECT to_jsonb(t) data FROM sim_ledger_entry t WHERE user_id=:u ORDER BY entry_id",
        "position_principal": """SELECT jsonb_build_object('code',fund_code,
          'shares',snapshot->'shares','cost',snapshot->'cost',
          'totalBuy',snapshot->'totalBuy','totalSell',snapshot->'totalSell','realizedGain',snapshot->'realizedGain',
          'opening',opening_balance) data FROM sim_position
          WHERE user_id=:u ORDER BY fund_code""",
        "old_reports": (
            "SELECT to_jsonb(t) data FROM portfolio_decision_report t "
            "WHERE user_id=:u AND strategy_version<>:v ORDER BY report_id"
        ),
    }
    result = {}
    with engine.connect() as c:
        c.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        for name, query in queries.items():
            values = [r[0] for r in c.execute(text(query), {"u": owner, "v": VERSION})]
            result[name] = {
                "count": len(values),
                "sha256": hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            }
    return result


def run(codes, all_held=False):
    settings = get_settings()
    owner = json.loads((PROJECT / ".local-runs/automation-owner/account.json").read_text(encoding="utf-8"))[
        "owner_user_id"
    ]
    scope = owner_scope(settings, owner)
    selected = scope if all_held else codes
    if not selected or not set(selected) <= set(scope):
        raise ValueError("只允许验证本人当前正份额持仓。")
    folder = (
        PROJECT
        / ".local-runs"
        / ("holding-comprehensive-advice-" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d"))
    )
    folder.mkdir(parents=True, exist_ok=True)
    engine = core_engine()
    try:
        before = proof(engine, owner)
        rows = []
        with httpx.Client(
            timeout=90, trust_env=False, headers={"X-Service-Token": settings.ai_service_token.get_secret_value()}
        ) as client:
            for code in selected:
                if code not in owner_scope(settings, owner):
                    raise ValueError("持仓已变化，停止新增。")
                url = (
                    settings.core_service_base_url + f"/internal/v1/holding-advice/users/{owner}/funds/{code}/generate"
                )
                response = client.post(url)
                if response.status_code != 200:
                    rows.append({"fundCode": code, "httpStatus": response.status_code, "readBack": False})
                    break
                value = response.json()
                with engine.connect() as c:
                    saved = c.execute(
                        text(
                            "SELECT payload FROM portfolio_decision_report "
                            "WHERE user_id=:u AND fund_code=:c AND report_id=:id"
                        ),
                        {"u": owner, "c": code, "id": value["reportId"]},
                    ).scalar_one()
                if value != saved or value.get("strategyVersion") != VERSION or value.get("fundCode") != code:
                    raise ValueError("正式报告读回不一致，停止。")
                # 同输入再次请求必须复用原报告，不能偷偷刷新生成时间。
                again = client.post(url)
                reused = again.status_code == 200 and again.json() == saved
                rows.append(
                    {
                        "fundCode": code,
                        "httpStatus": response.status_code,
                        "reportId": value["reportId"],
                        "adviceState": value["adviceState"],
                        "decision": value["decision"],
                        "readBack": True,
                        "identicalInputReused": reused,
                        "missing": value["missingOptionalFactors"],
                    }
                )
                (folder / (code + "-display.json")).write_text(
                    json.dumps({k: value.get(k) for k in DISPLAY_FIELDS}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if not reused:
                    break
        after = proof(engine, owner)
        result = {
            "checkedAt": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "requested": len(selected),
            "readBack": sum(r["readBack"] for r in rows),
            "states": dict(Counter(r.get("adviceState", "FAILED") for r in rows)),
            "before": before,
            "after": after,
            "accountingUnchanged": before == after,
            "allReadBackAndReused": len(rows) == len(selected)
            and all(r["readBack"] and r.get("identicalInputReused") for r in rows),
            "rows": rows,
        }
        path = folder / ("all-held-verification.json" if all_held else "single-verification.json")
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "checkedAt",
                        "requested",
                        "readBack",
                        "states",
                        "accountingUnchanged",
                        "allReadBackAndReused",
                    )
                },
                ensure_ascii=False,
            )
        )
        return result
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", action="append", default=[])
    parser.add_argument("--all-held", action="store_true")
    args = parser.parse_args()
    result = run(args.code, args.all_held)
    raise SystemExit(0 if result["allReadBackAndReused"] and result["accountingUnchanged"] else 2)
