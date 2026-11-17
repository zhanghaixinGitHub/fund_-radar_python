"""本人当前持仓的建议输入只读清单；不触发采集、预测、评分或交易。"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from dotenv import dotenv_values
from sqlalchemy import URL, create_engine, text

from app.commands.fund_prediction_followthrough import owner_scope
from app.core.config import get_settings

PROJECT = Path(__file__).resolve().parents[2]


def core_engine():
    """凭据只留内存；所有现场核对使用数据库强制只读连接。"""
    cfg = dotenv_values(PROJECT.parent / "02-java/.env")
    return create_engine(
        URL.create(
            "postgresql+psycopg",
            username=cfg["FUND_CORE_DB_USERNAME"],
            password=cfg["FUND_CORE_DB_PASSWORD"],
            host=cfg.get("FUND_CORE_DB_HOST", "localhost"),
            port=int(cfg.get("FUND_CORE_DB_PORT", "54329")),
            database=cfg.get("FUND_CORE_DB_NAME", "fund_core"),
        ),
        hide_parameters=True,
        connect_args={"options": "-c default_transaction_read_only=on -c statement_timeout=5000"},
    )


def inventory():
    settings = get_settings()
    owner = json.loads((PROJECT / ".local-runs/automation-owner/account.json").read_text(encoding="utf-8"))[
        "owner_user_id"
    ]
    codes = owner_scope(settings, owner)
    engine = core_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            positions = {
                r["fund_code"]: r["snapshot"]
                for r in connection.execute(
                    text(
                        "SELECT fund_code,snapshot FROM sim_position "
                        "WHERE user_id=:owner AND (snapshot->>'shares')::numeric>0"
                    ),
                    {"owner": owner},
                ).mappings()
            }
            counts = {}
            for name, query in {
                "pending": (
                    "SELECT fund_code,side,count(*) n FROM sim_order "
                    "WHERE user_id=:owner AND status='PENDING' GROUP BY fund_code,side"
                ),
                "plans": (
                    "SELECT fund_code,count(*) n FROM sim_plan "
                    "WHERE user_id=:owner AND status='ACTIVE' GROUP BY fund_code"
                ),
                "rules": (
                    "SELECT fund_code,count(*) n FROM holding_rule_profile "
                    "WHERE user_id=:owner AND status='ACTIVE' GROUP BY fund_code"
                ),
            }.items():
                counts[name] = [dict(r) for r in connection.execute(text(query), {"owner": owner}).mappings()]
        if set(codes) != set(positions):
            raise ValueError("持仓范围在核对期间发生变化，请重新核对。")
        rows = []
        with httpx.Client(
            timeout=25, trust_env=False, headers={"X-Service-Token": settings.ai_service_token.get_secret_value()}
        ) as client:
            for code in codes:
                row = {"fundCode": code, "navDate": positions[code].get("navDate"), "interfaces": {}}
                for kind, path in {
                    "rating": f"/funds/{code}/rating",
                    "review": f"/multi-predictions/funds/{code}/review-v2",
                    "diagnosis": (
                        f"/portfolio-advice/{code}/diagnosis-facts"
                        f"?asOfDate={datetime.now(ZoneInfo('Asia/Shanghai')).date()}"
                    ),
                }.items():
                    response = client.get("http://127.0.0.1:8000/internal/v1" + path)
                    row["interfaces"][kind] = response.status_code
                    value = response.json() if response.status_code == 200 else None
                    if kind == "rating" and value:
                        row[kind] = {
                            k: value.get(k)
                            for k in ("status", "score", "asOfDate", "validUntil", "ratingRef", "limitations")
                        }
                    elif kind == "review" and value:
                        row[kind] = {"fundCode": value.get("fundCode"), "periods": value.get("periods", [])}
                    elif kind == "diagnosis" and value:
                        row[kind] = {
                            "overall": value.get("overall"),
                            "items": [
                                {k: item.get(k) for k in ("item", "verdict", "dataAsOfDate")}
                                for item in value.get("items", [])
                            ],
                        }
                for kind, values in counts.items():
                    row[kind] = [
                        {k: v for k, v in value.items() if k != "fund_code"}
                        for value in values
                        if value["fund_code"] == code
                    ]
                rows.append(row)
        if owner_scope(settings, owner) != codes:
            raise ValueError("持仓范围在接口核对期间发生变化，请重新核对。")
        return {
            "checkedAt": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "readOnly": True,
            "fundCount": len(rows),
            "ratings": dict(Counter(r.get("rating", {}).get("status", "UNAVAILABLE") for r in rows)),
            "periodKinds": dict(
                Counter(p.get("kind", "UNKNOWN") for r in rows for p in r.get("review", {}).get("periods", []))
            ),
            "rows": rows,
        }
    finally:
        engine.dispose()


if __name__ == "__main__":
    result = inventory()
    folder = (
        PROJECT
        / ".local-runs"
        / ("holding-comprehensive-advice-" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d"))
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "inventory.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ("checkedAt", "readOnly", "fundCount", "ratings", "periodKinds")}, ensure_ascii=False
        )
    )
