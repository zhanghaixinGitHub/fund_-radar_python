"""按预先封存的指数清单补查日线；只写独立研究目录，不修改数据库。"""

import argparse
from datetime import date, datetime
from pathlib import Path
from time import monotonic, sleep

from app.services.direction_1d_protocol import ZONE, canonical, digest
from app.services.direction_1d_training import read, write_new
from app.services.direction_market_data import client, metadata, validate_prices
from app.services.direction_training_artifacts import file_hash


def acquire(folder: Path) -> dict:
    """每次请求先占用预算，失败不重试；中断后只复用完整回执，避免重复取数。"""
    plan = read(folder / "daily-plan.json")
    if read(folder / "daily-plan-receipt.json")["hash"] != digest(plan):
        raise ValueError("SECTOR_DAILY_PLAN_CHANGED")
    codes = plan["index_codes"]
    budget = read(folder / "budget.json")["index_daily_calls"]
    if len(codes) != len(set(codes)) or len(codes) * 4 > budget or budget > 52:
        raise ValueError("SECTOR_DAILY_BUDGET_INVALID")
    meta = metadata(codes)
    if not meta["rate_limit_per_minute"] or meta["rate_limit_per_minute"] <= 0:
        raise ValueError("SECTOR_RATE_LIMIT_MISSING")
    identities = {}
    for name, expected in plan["identity_files"].items():
        path = folder / name
        if path.parent != folder or path.is_symlink() or file_hash(path) != expected:
            raise ValueError("SECTOR_INDEX_IDENTITY_CHANGED")
        saved = read(path)
        identities.update(saved.get("catalog", {}))
        if saved.get("record"):
            identities[saved["record"]["index_code"]] = saved["record"]
    if any(identities.get(code, {}).get("index_code") != code for code in codes):
        raise ValueError("SECTOR_INDEX_IDENTITY_MISSING")
    # 完成包重入只校验并返回，不再次请求供应商或覆盖封存结果。
    if (folder / "new-prices.json").exists():
        saved = read(folder / "new-prices.json")
        if digest(saved) != read(folder / "new-prices-receipt.json")["hash"] or saved["plan_hash"] != digest(plan):
            raise ValueError("SECTOR_COMPLETED_DATA_CHANGED")
        for request in saved["requests"]:
            if file_hash(folder / request["file"]) != request["hash"]:
                raise ValueError("SECTOR_COMPLETED_RESPONSE_CHANGED")
        return {"status": "ALREADY_COMPLETED", "new_api_calls": 0, "saved_calls": len(saved["requests"])}
    prices, coverage, requests = {}, {}, []
    with client() as api:
        last = 0.0
        for code in codes:
            rows = []
            for year in (2021, 2022, 2023, 2024):
                output = folder / f"daily-{code}-{year}.json"
                reserved = folder / f"daily-{code}-{year}-reserved.json"
                if not output.exists():
                    if reserved.exists():
                        raise ValueError("SECTOR_UNCERTAIN_REQUEST_REQUIRES_AUDIT")
                    if len(list(folder.glob("daily-*-reserved.json"))) >= budget:
                        raise ValueError("SECTOR_DAILY_BUDGET_EXHAUSTED")
                    sleep(max(0, max(0.4, 60 / meta["rate_limit_per_minute"]) - (monotonic() - last)))
                    started = datetime.now(ZONE).isoformat()
                    write_new(reserved, {"api": "index_daily", "code": code, "year": year, "started_at": started})
                    last = monotonic()
                    result = {"api": "index_daily", "code": code, "year": year, "started_at": started}
                    try:
                        fetched = api.list_index_daily(code, start_date=date(year, 1, 1), end_date=date(year, 12, 31))
                        if any(r.trade_date.year != year or r.index_code != code for r in fetched):
                            raise ValueError("SECTOR_RESPONSE_IDENTITY_OR_YEAR_INVALID")
                        result.update(
                            status="DOWNLOADED",
                            prices=[{"date": str(r.trade_date), "close": str(r.close_price)} for r in fetched],
                        )
                    except Exception as error:
                        # 仅保存异常类别及明确权限判定，不将供应商响应或敏感配置带入证据。
                        result.update(
                            status="FAILED",
                            error_type=type(error).__name__,
                            permission_error=any(v in str(error) for v in ("权限", "permission", "积分")),
                            prices=[],
                        )
                    result["retrieved_at"] = datetime.now(ZONE).isoformat()
                    write_new(output, result)
                result = read(output)
                if (result["code"], result["year"]) != (code, year):
                    raise ValueError("SECTOR_CACHED_REQUEST_CHANGED")
                requests.append(
                    {k: v for k, v in result.items() if k != "prices"}
                    | {"rows": len(result["prices"]), "file": output.name, "hash": file_hash(output)}
                )
                if result["status"] != "DOWNLOADED" or not result["prices"]:
                    break
                rows.extend(result["prices"])
            prices[code], coverage[code] = validate_prices(rows, code)
            print(
                canonical({"index": code, "rows": len(prices[code]), "missing": len(coverage[code]["missing_dates"])}),
                flush=True,
            )
    result = {
        "created_at": datetime.now(ZONE).isoformat(),
        "prices": prices,
        "coverage": coverage,
        "requests": requests,
        "plan_hash": digest(plan),
        "database_writes": 0,
        "historical_first_versions_verified": False,
    }
    write_new(folder / "new-prices.json", result)
    write_new(folder / "new-prices-receipt.json", {"hash": digest(result)})
    return {
        "calls": len(requests),
        "indices": len(codes),
        "complete_indices": sum(not v["missing_dates"] for v in coverage.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    print(canonical(acquire(parser.parse_args().data_dir)))
