"""沪深300成交额的有限采集与冻结校验；最多四个年度请求，不写业务数据库。"""

from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from app.services.direction_1d_protocol import ZONE, calendar, digest
from app.services.direction_1d_training import read, write_new
from app.services.direction_market_data import client, metadata
from app.services.direction_training_artifacts import file_hash

INDEX = "000300.SH"
YEARS = (2021, 2022, 2023, 2024)


def validate_rows(rows: list[dict], old_prices: dict, year: int | None = None) -> dict:
    """严格核对既有交易日历、正成交额及全部旧收盘；缺失或修订不通过改旧包修复。"""
    expected = {str(d) for d in calendar()[0] if d.year in YEARS and (year is None or d.year == year)}
    result = {}
    for row in rows:
        day = row["date"]
        if row["index_code"] != INDEX or day not in expected or day in result:
            raise ValueError("ACTIVITY_IDENTITY_OR_DATE_INVALID")
        try:
            amount, close = Decimal(str(row["amount"])), Decimal(str(row["close"]))
        except InvalidOperation as error:
            raise ValueError("ACTIVITY_VALUE_INVALID") from error
        if (
            not all(v.is_finite() and v > 0 for v in (amount, close))
            or not np.isfinite([float(amount), float(close)]).all()
        ):
            raise ValueError("ACTIVITY_VALUE_INVALID")
        if day not in old_prices or close != Decimal(str(old_prices[day])):
            raise ValueError("ACTIVITY_OLD_CLOSE_CHANGED")
        result[day] = dict(row)
    if set(result) != expected:
        raise ValueError("ACTIVITY_HISTORY_INCOMPLETE")
    return result


def initialize(root: Path, base: Path, audit: Path, plan: Path) -> dict:
    """在任何请求前绑定本地存量审计、真实授权与旧价格快照，创建不可覆盖的四次预算。"""
    evidence = read(audit)
    source = evidence["source"]
    if (
        not source["enabled"]
        or not source["authorization_verified_at"]
        or not {"index_basic", "index_daily"}.issubset(source["authorized_api_names"])
        or source["retention_days"] <= 0
        or source["rate_limit_per_minute"] <= 0
        or evidence["local_index_amount_tables"]
    ):
        raise ValueError("ACTIVITY_SOURCE_OR_LOCAL_AUDIT_NOT_READY")
    now = datetime.now(ZONE)
    if not timedelta(0) <= now - datetime.fromisoformat(evidence["at"]) <= timedelta(days=1):
        raise ValueError("ACTIVITY_LOCAL_AUDIT_STALE")
    old = read(base / "base/prices.json")["prices"][INDEX]
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "local-audit.json", evidence)
    write_new(root / "old-prices.json", old)
    budget = {
        "created_at": now.isoformat(),
        "index_code": INDEX,
        "years": list(YEARS),
        "max_requests": 4,
        "plan_hash": file_hash(plan),
        "source": source,
        "base_study_hash": file_hash(base / "study.json"),
        "old_prices_hash": digest(old),
        "local_audit_hash": digest(evidence),
    }
    write_new(root / "budget.json", budget)
    write_new(root / "budget-receipt.json", {"hash": digest(budget)})
    return {"index_code": INDEX, "max_requests": 4, "old_price_count": len(old)}


def verify(root: Path) -> dict:
    """校验完整响应及封条，不调用供应商；用于研究包独立恢复。"""
    budget = read(root / "budget.json")
    if (
        digest(budget) != read(root / "budget-receipt.json")["hash"]
        or budget["years"] != list(YEARS)
        or budget["max_requests"] != 4
    ):
        raise ValueError("ACTIVITY_BUDGET_CHANGED")
    payload, seal = read(root / "activity-data.json"), read(root / "data-seal.json")
    if (
        digest(seal) != read(root / "data-seal-receipt.json")["hash"]
        or payload["budget_hash"] != digest(budget)
        or seal["code_hash"] != file_hash(Path(__file__))
    ):
        raise ValueError("ACTIVITY_SEAL_CHANGED")
    for name, expected in seal["files"].items():
        if Path(name).name != name or (root / name).is_symlink() or file_hash(root / name) != expected:
            raise ValueError("ACTIVITY_FROZEN_RESPONSE_CHANGED")
    old = read(root / "old-prices.json")
    if (
        digest(old) != budget["old_prices_hash"]
        or digest(read(root / "local-audit.json")) != budget["local_audit_hash"]
    ):
        raise ValueError("ACTIVITY_AUDIT_OR_PRICE_CHANGED")
    if {p.name for p in root.glob("*-reserved.json")} != {f"{y}-reserved.json" for y in YEARS}:
        raise ValueError("ACTIVITY_REQUEST_BUDGET_INVALID")
    rows, starts = [], []
    for year in YEARS:
        response, reserved = read(root / f"{year}.json"), read(root / f"{year}-reserved.json")
        if (
            response["status"] != "DOWNLOADED"
            or response["year"] != year
            or response["index_code"] != INDEX
            or reserved != response["reservation"]
        ):
            raise ValueError("ACTIVITY_RESPONSE_INVALID")
        starts.append(datetime.fromisoformat(reserved["at"]))
        if starts[-1] > datetime.fromisoformat(response["finished_at"]):
            raise ValueError("ACTIVITY_RESPONSE_TIME_INVALID")
        validate_rows(response["rows"], old, year)
        rows.extend(response["rows"])
    if validate_rows(rows, old) != payload["rows"] or payload["api_calls"] != 4:
        raise ValueError("ACTIVITY_COMBINED_DATA_CHANGED")
    expires = min(starts) + timedelta(days=budget["source"]["retention_days"])
    if datetime.fromisoformat(payload["source_expires_at"]) != expires or datetime.now(ZONE) >= expires:
        raise ValueError("ACTIVITY_SOURCE_RETENTION_INVALID")
    return payload


def acquire(root: Path) -> dict:
    """首个年度通过才继续；断点只复用完整回执，预算已占而未完成时停止，不盲目重试。"""
    if (root / "data-seal.json").exists():
        saved = verify(root)
        return {"status": "ALREADY_COMPLETED", "new_api_calls": 0, "saved_rows": len(saved["rows"])}
    budget = read(root / "budget.json")
    if (
        digest(budget) != read(root / "budget-receipt.json")["hash"]
        or budget["index_code"] != INDEX
        or budget["years"] != list(YEARS)
        or budget["max_requests"] != 4
    ):
        raise ValueError("ACTIVITY_BUDGET_CHANGED")
    meta = metadata([INDEX])
    if (
        meta["source_id"] != budget["source"]["source_id"]
        or meta["catalog"].get(INDEX, {}).get("display_name") != "沪深300"
    ):
        raise ValueError("ACTIVITY_SOURCE_OR_IDENTITY_CHANGED")
    rate = min(meta["rate_limit_per_minute"], budget["source"]["rate_limit_per_minute"])
    if rate <= 0:
        raise ValueError("ACTIVITY_RATE_LIMIT_INVALID")
    old = read(root / "old-prices.json")
    if digest(old) != budget["old_prices_hash"]:
        raise ValueError("ACTIVITY_OLD_PRICE_SNAPSHOT_CHANGED")
    rows, starts = [], []
    with client() as api:
        last = 0.0
        for year in YEARS:
            output, reserved = root / f"{year}.json", root / f"{year}-reserved.json"
            if not output.exists():
                if reserved.exists():
                    raise ValueError("ACTIVITY_UNCERTAIN_REQUEST_REQUIRES_AUDIT")
                sleep(max(0, max(0.4, 60 / rate) - (monotonic() - last)))
                reservation = {
                    "at": datetime.now(ZONE).isoformat(),
                    "api": "index_daily",
                    "index_code": INDEX,
                    "year": year,
                }
                write_new(reserved, reservation)
                last = monotonic()
                result = {"reservation": reservation, "index_code": INDEX, "year": year}
                try:
                    fetched = api.list_index_activity(INDEX, start_date=date(year, 1, 1), end_date=date(year, 12, 31))
                    values = [
                        {
                            "index_code": r.index_code,
                            "date": str(r.trade_date),
                            "close": str(r.close_price),
                            "amount": str(r.amount) if r.amount is not None else None,
                        }
                        for r in fetched
                    ]
                    result.update(status="DOWNLOADED", rows=values)
                except Exception as error:
                    result.update(
                        status="FAILED",
                        rows=[],
                        error_type=type(error).__name__,
                        permission_error=any(v in str(error) for v in ("权限", "permission", "积分")),
                    )
                result["finished_at"] = datetime.now(ZONE).isoformat()
                write_new(output, result)
            result = read(output)
            if result["status"] != "DOWNLOADED" or result["year"] != year or result["reservation"] != read(reserved):
                raise ValueError("ACTIVITY_REQUEST_FAILED_OR_CHANGED")
            validate_rows(result["rows"], old, year)
            rows.extend(result["rows"])
            starts.append(datetime.fromisoformat(result["reservation"]["at"]))
    payload = {
        "created_at": datetime.now(ZONE).isoformat(),
        "budget_hash": digest(budget),
        "index_code": INDEX,
        "amount_unit": "THOUSAND_CNY",
        "rows": validate_rows(rows, old),
        "api_calls": 4,
        "database_writes": 0,
        "historical_first_versions_verified": False,
        "source_expires_at": (min(starts) + timedelta(days=budget["source"]["retention_days"])).isoformat(),
    }
    write_new(root / "activity-data.json", payload)
    names = [
        "budget.json",
        "budget-receipt.json",
        "local-audit.json",
        "old-prices.json",
        "activity-data.json",
        *(f"{year}{suffix}.json" for year in YEARS for suffix in ("", "-reserved")),
    ]
    seal = {
        "created_at": datetime.now(ZONE).isoformat(),
        "files": {n: file_hash(root / n) for n in names},
        "code_hash": file_hash(Path(__file__)),
    }
    write_new(root / "data-seal.json", seal)
    write_new(root / "data-seal-receipt.json", {"hash": digest(seal)})
    verify(root)
    return {"status": "COMPLETED", "new_api_calls": 4, "rows": len(payload["rows"]), "old_close_mismatch_count": 0}
