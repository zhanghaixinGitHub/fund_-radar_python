"""沪深A股市场广度：有界来源探测、逐日缓存、月度封存和上一交易日可得特征。"""

import re
from datetime import date
from decimal import Decimal
from time import monotonic, sleep

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_nav_preview_engine
from app.integrations.tushare_breadth import FIELDS, TushareBreadthClient
from app.models.fund import SourceRegistry
from app.repositories.fund_sync import TUSHARE_SOURCE_CODE
from app.repositories.market_reference_sync import require_tushare_source_capabilities
from app.schemas.direction_training import DirectionInput
from app.services.direction_training_artifacts import (
    ROOT,
    digest,
    file_hash,
    now,
    read_json,
    read_seal,
    seal,
    write_json,
)
from app.services.trading_calendar import load_calendar

EVIDENCE = ROOT / ".local-runs/direction-breadth-evidence-20260911"
PROBE_DAYS = ("2021-01-04", "2022-01-04", "2023-01-03", "2024-01-02", "2024-12-31")
RULES = {
    "universe": "HISTORICAL_DATE_SH_SZ_A_SHARE_DAILY_RECORDS_WITH_POSITIVE_VOLUME",
    "code_pattern": r"(?:60\d{4}|688\d{3})\.SH|(?:00\d{4}|30\d{4})\.SZ",
    "denominator": "POSITIVE_VOLUME_VALID_QUOTE_ROWS_INCLUDING_FLAT",
    "numerator": "OFFICIAL_PCT_CHG_STRICTLY_POSITIVE",
    "feature": "ARITHMETIC_MEAN_OF_DAILY_UP_FRACTIONS_LAST_5_SESSIONS_END_PREVIOUS_SESSION",
    "daily_minimum": 3000,
    "exchange_minimum": 1000,
    "valid_quote_fraction_minimum": 0.99,
    "missing_policy": "NO_IMPUTATION_NO_FORWARD_FILL_NO_COUNTING_SUSPENSIONS_AS_DOWN",
    "scope_exclusions": ["BEIJING", "B_SHARES", "CDR_689", "NON_STOCKS"],
}


def metadata():
    """只读原来源启用状态；本次daily权限由用户授权的小样本实测独立留证。"""
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        source = session.scalar(select(SourceRegistry).where(SourceRegistry.source_code == TUSHARE_SOURCE_CODE))
        if source is None or not source.enabled or source.authorization_verified_at is None:
            raise ValueError("BREADTH_SOURCE_DISABLED_OR_UNVERIFIED")
        require_tushare_source_capabilities(session, ("daily",))
        return {
            "source_code": source.source_code,
            "source_id": str(source.source_id),
            "rate_limit_per_minute": source.rate_limit_per_minute,
            "registered_api_names": source.authorized_api_names,
            "authorization_verified_at": str(source.authorization_verified_at),
            "daily_probe_scope": "USER_AUTHORIZED_LOCAL_RESEARCH_ONLY_NO_GLOBAL_CAPABILITY_WRITE",
        }


def client():
    settings = get_settings()
    return TushareBreadthClient(
        token=settings.tushare_token.get_secret_value(),
        api_url=settings.tushare_api_url,
        connect_timeout_seconds=5,
        read_timeout_seconds=20,
        max_retries=0,
        catalog_max_rows_per_query=8000,
        max_rows_per_query=6000,
    )


def days():
    return [str(d) for d in load_calendar().sessions if date(2021, 1, 1) <= d <= date(2024, 12, 31)]


def summarize(rows, day):
    """校验来源身份和字段；分母仅是当日可交易且报价有效的沪深普通A股。

    涨跌幅使用来源已按除权昨收计算的pct_chg，避免把分红除权误算成下跌。
    平盘进入分母。缺失/非正交易量单独计数；字段缺失、重复或错误日期拒绝整日。
    行数/两交易所最低数只是截断异常检查，不是对交易所全量覆盖的独立证明。
    """
    if not 1 <= len(rows) < 6000 or day not in days():
        raise ValueError("BREADTH_DATE_OR_RESPONSE_SIZE")
    seen, counts = (
        set(),
        {
            k: 0
            for k in (
                "received",
                "out_of_scope",
                "in_scope",
                "no_trade",
                "invalid_quote",
                "valid",
                "up",
                "flat",
                "down",
                "SH",
                "SZ",
            )
        },
    )
    for row in rows:
        if set(row) != set(FIELDS) or row["trade_date"] != day.replace("-", ""):
            raise ValueError("BREADTH_ROW_FIELDS_OR_DATE")
        code = row["ts_code"]
        if not isinstance(code, str) or not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", code) or code in seen:
            raise ValueError("BREADTH_IDENTITY_OR_DUPLICATE")
        seen.add(code)
        counts["received"] += 1
        if not re.fullmatch(RULES["code_pattern"], code):
            counts["out_of_scope"] += 1
            continue
        counts["in_scope"] += 1
        try:
            close, previous, change, volume = (Decimal(str(row[k])) for k in ("close", "pre_close", "pct_chg", "vol"))
        except Exception:
            counts["invalid_quote"] += 1
            continue
        if (
            not all(v.is_finite() for v in (close, previous, change, volume))
            or close <= 0
            or previous <= 0
            or volume < 0
        ):
            counts["invalid_quote"] += 1
            continue
        if volume == 0:
            counts["no_trade"] += 1
            continue
        # pct_chg常保留四位小数；容许最后一位舍入，拒绝明显不一致的涨跌口径。
        if abs((close / previous - 1) * 100 - change) > Decimal("0.001"):
            raise ValueError("BREADTH_PRICE_CHANGE_INCONSISTENT")
        counts["valid"] += 1
        counts[code[-2:]] += 1
        counts["up" if change > 0 else "down" if change < 0 else "flat"] += 1
    denominator = counts["in_scope"] - counts["no_trade"]
    valid = (
        counts["valid"] >= RULES["daily_minimum"]
        and min(counts["SH"], counts["SZ"]) >= RULES["exchange_minimum"]
        and counts["valid"] / max(1, denominator) >= RULES["valid_quote_fraction_minimum"]
    )
    return {
        "date": day,
        "status": "READY" if valid else "INSUFFICIENT_SOURCE_COVERAGE",
        "counts": counts,
        "up_fraction": counts["up"] / counts["valid"] if valid else None,
    }


def fetch(api, day):
    response = {"api": "daily", "trade_date": day, "fields": list(FIELDS), "retrieved_at": now()}
    try:
        rows = list(api.list_daily_market(date.fromisoformat(day)))
        summary = summarize(rows, day)
        response.update(status="DOWNLOADED", rows=rows, summary=summary)
    except Exception as error:
        # 只保存异常类型与供应商数字错误码，避免外部异常文字意外携带凭据。
        code = re.search(r"API code=(-?\d+)", str(error))
        response.update(
            status="FAILED",
            error_type=type(error).__name__,
            retryable=bool(getattr(error, "retryable", False)),
            provider_code=code.group(1) if code else None,
            reason=str(error) if isinstance(error, ValueError) and str(error).startswith("BREADTH_") else None,
            rows=[],
        )
    return response


def probe():
    EVIDENCE.mkdir(exist_ok=True)
    meta, files, summaries = metadata(), {}, []
    with client() as api:
        for day in PROBE_DAYS:
            name = f"breadth-response-{day}.json"
            path = EVIDENCE / name
            if not path.exists():
                response = fetch(api, day)
                write_json(path, response)
                sleep(max(0.35, 60 / (meta["rate_limit_per_minute"] or 200)))
            response = read_json(path)
            files[name] = file_hash(path)
            summaries.append({k: v for k, v in response.items() if k != "rows"})
            if response["status"] != "DOWNLOADED":
                break
    result = {
        "status": "PASSED"
        if len(summaries) == len(PROBE_DAYS) and all(r.get("summary", {}).get("status") == "READY" for r in summaries)
        else "FAILED",
        "metadata": meta,
        "rules": RULES,
        "results": summaries,
        "maximum_calls": len(PROBE_DAYS),
        "database_written": False,
        "historical_first_versions_verified": False,
    }
    files["breadth-probe.json"] = write_json(EVIDENCE / "breadth-probe.json", result)
    seal(EVIDENCE, "breadth-probe-seal.json", files, status=result["status"])
    return result


def augment(raw, daily):
    """保持原七列和样本身份，只追加截止日前上一交易日结束的五日广度均值。"""
    item, cal = DirectionInput.model_validate(raw), load_calendar()
    end = cal.at_or_before_index(item.cutoff) - 1
    if end < 4 or item.anchor != cal.sessions[end]:
        return None, "BREADTH_ANCHOR_OR_PREWARM"
    selected = [str(d) for d in cal.sessions[end - 4 : end + 1]]
    if any(d not in daily or daily[d]["status"] != "READY" for d in selected):
        return None, "BREADTH_MISSING_SESSION"
    value = sum(daily[d]["up_fraction"] for d in selected) / 5
    return {
        **item.model_dump(mode="json"),
        "x": [*item.x, value],
        "input_hash": digest(
            {"original": item.input_hash, "days": selected, "summaries": [daily[d] for d in selected], "feature": value}
        ),
    }, None


def validate_probe(folder):
    """探测记录同时绑定五个预定日期、字段、统计规则与原始响应。"""
    manifest = read_seal(folder, "breadth-probe-seal.json")
    probe = read_json(folder / "breadth-probe.json")
    if manifest["status"] != "PASSED" or probe["status"] != "PASSED" or probe["rules"] != RULES:
        raise ValueError("BREADTH_PROBE_NOT_PASSED")
    for day in PROBE_DAYS:
        response = read_json(folder / f"breadth-response-{day}.json")
        validate_response(response, day)
    return probe, manifest


def validate_response(response, day):
    if (response.get("api"), response.get("trade_date"), response.get("fields"), response.get("status")) != (
        "daily",
        day,
        list(FIELDS),
        "DOWNLOADED",
    ) or summarize(response["rows"], day) != response["summary"]:
        raise ValueError("BREADTH_RESPONSE_IDENTITY_OR_SUMMARY")
    return response["summary"]


def acquire(folder, protocol):
    """969日按日取历史截面，按月封存；最多10次可恢复失败的额外尝试。

    已成功的日响应原样复用；失败尝试保留为独立文件，不覆盖失败证据。
    每日最多一次重试，全研究最多10次；结构/数据质量失败不以重试改变输入。
    """
    import json

    meta = metadata()
    if meta["source_id"] != protocol["source_id"]:
        raise ValueError("BREADTH_SOURCE_ID_CHANGED")
    gap, last, retries = max(0.35, 60 / (meta["rate_limit_per_minute"] or 200)), 0.0, 0
    files, daily, requests = {}, {}, []
    with client() as api:
        for month in sorted({d[:7] for d in days()}):
            month_files = {}
            for day in (d for d in days() if d.startswith(month)):
                name = f"breadth-response-{day}.json"
                path = folder / name
                failure_path = folder / f"breadth-attempt-failed-{day}.json"
                if failure_path.exists():
                    retries += 1
                if not path.exists():
                    if failure_path.exists():
                        # 不重复消耗已申请的重试；中断后需要保留失败状态，不静默第三次调用。
                        raise ValueError("BREADTH_INTERRUPTED_RETRY_REQUIRES_AUDIT")
                    sleep(max(0, gap - (monotonic() - last)))
                    last = monotonic()
                    response = fetch(api, day)
                    if response["status"] == "FAILED" and response["retryable"] and retries < 10:
                        write_json(failure_path, response)
                        retries += 1
                        sleep(2)
                        last = monotonic()
                        response = fetch(api, day)
                    write_json(path, response)
                response = read_json(path)
                if failure_path.exists():
                    month_files[failure_path.name] = file_hash(failure_path)
                month_files[name] = file_hash(path)
                if response["status"] != "DOWNLOADED":
                    raise ValueError("BREADTH_ACQUISITION_FAILED_RESPONSE_SAVED")
                daily[day] = validate_response(response, day)
                requests.append(
                    {
                        "date": day,
                        "rows": len(response["rows"]),
                        "probe_reused": day in PROBE_DAYS,
                        "retried": failure_path.exists(),
                        "retrieved_at": response["retrieved_at"],
                    }
                )
            name = f"breadth-source-month-{month}.json"
            if (folder / name).exists():
                if read_seal(folder, name)["files"] != month_files:
                    raise ValueError("BREADTH_CACHED_MONTH_CHANGED")
            else:
                seal(folder, name, month_files, month=month, status="SEALED_DAILY_RESPONSES")
            files[name] = file_hash(folder / name)
            print(json.dumps({"month": month, "completed_sessions": len(daily), "retry_requests": retries}), flush=True)
    result = {
        "daily": daily,
        "rules": RULES,
        "requests": requests,
        "retries": retries,
        "source_metadata": meta,
        "database_written": False,
        "historical_first_versions_verified": False,
        "coverage": {
            "planned": len(days()),
            "received": len(daily),
            "ready": sum(r["status"] == "READY" for r in daily.values()),
        },
    }
    files["breadth-data.json"] = write_json(folder / "breadth-data.json", result)
    return result, files


def rebuild_daily(folder):
    """逐月验证原始行情并重新统计；不使用缓存摘要替代原始数据复核。"""
    daily = {}
    for month in sorted({d[:7] for d in days()}):
        read_seal(folder, f"breadth-source-month-{month}.json")
        for day in (d for d in days() if d.startswith(month)):
            daily[day] = validate_response(read_json(folder / f"breadth-response-{day}.json"), day)
    if daily != read_json(folder / "breadth-data.json")["daily"]:
        raise ValueError("BREADTH_DAILY_REBUILD_CHANGED")
    return daily
