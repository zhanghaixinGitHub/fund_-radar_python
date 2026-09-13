"""一日隔夜输入资格审计：核对SPX覆盖与跨时区对齐，不把市场收盘当作数据可用证明。"""

from bisect import bisect_right
from collections import Counter
from datetime import date, datetime, time, timedelta
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

from app.services.direction_1d_protocol import ZONE, calendar, canonical
from app.services.direction_1d_training import read, write_new
from app.services.direction_training_artifacts import file_hash

CALENDAR = Path(__file__).resolve().parents[1] / "data/calendars/nyse_cash_2021_2024_v1.json"
YEARS = (2021, 2022, 2023, 2024)
FIELDS = ["ts_code", "trade_date", "close", "pre_close", "pct_chg"]
RECIPE = {
    "candidate": "SPX_OVERNIGHT13",
    "reference": "ACTIVITY12",
    "new_feature_count": 1,
    "interval": "CN_T_15_EXCLUSIVE_TO_CN_U_08_INCLUSIVE",
    "value": "LAST_SPX_CLOSE_IN_INTERVAL_DIVIDED_BY_LAST_CLOSE_AT_OR_BEFORE_T15_MINUS_ONE",
    "unit": "DECIMAL_RETURN_NOT_PERCENTAGE_POINTS",
    "no_us_session": "ZERO_ONLY_IF_OFFICIAL_CALENDAR_HAS_NO_SESSION_IN_INTERVAL",
    "missing": "REJECT_NO_FORWARD_FILL_NO_QUESTION_DROPPING",
    "holiday": "ACCUMULATE_ALL_US_SESSIONS_DURING_CHINA_HOLIDAY",
    "timing": "MARKET_CLOSE_EVENT_IS_NOT_PROVIDER_AVAILABILITY",
    "historical_versions": "CURRENT_QUERY_NOT_HISTORICAL_FIRST_VERSION",
}


def sessions() -> dict[str, datetime]:
    """用官方现金市场假日及半日市日历计算收盘事件；带时区，不向开发年限外推。"""
    spec = read(CALENDAR)
    if spec["calendar_id"] != "NYSE_CASH_2021_2024_V1":
        raise ValueError("OVERNIGHT_CALENDAR_INVALID")
    start, end = date(2021, 1, 1), date(2024, 12, 31)
    holidays, early = (set(spec[k]) for k in ("holidays", "early_closes"))
    if holidays & early or any(not start <= date.fromisoformat(d) <= end for d in holidays | early):
        raise ValueError("OVERNIGHT_CALENDAR_DATES_INVALID")
    result = {}
    day = start
    zone = ZoneInfo("America/New_York")
    while day <= end:
        key = day.isoformat()
        if day.weekday() < 5 and key not in holidays:
            result[key] = datetime.combine(day, time(13 if key in early else 16), zone).astimezone(ZONE)
        day += timedelta(days=1)
    return result


def validate_history(payloads: list[dict]) -> dict[str, dict]:
    """拒绝重复、非法数值和意外年份；涨跌幅为百分数，按官方四位小数容许末位舍入。"""
    result = {}
    for payload in payloads:
        if payload.get("fields") != FIELDS or not 1 <= len(payload.get("items", [])) <= 366:
            raise ValueError("OVERNIGHT_FIELDS_OR_ROWS_INVALID")
        for values in payload["items"]:
            if len(values) != len(FIELDS):
                raise ValueError("OVERNIGHT_ROW_SHAPE_INVALID")
            row = dict(zip(FIELDS, values, strict=True))
            raw_date = row["trade_date"]
            if not isinstance(raw_date, str) or len(raw_date) != 8 or not raw_date.isdigit():
                raise ValueError("OVERNIGHT_DATE_INVALID")
            day = datetime.strptime(raw_date, "%Y%m%d").date()
            if row["ts_code"] != "SPX" or day.year not in YEARS or day.isoformat() in result:
                raise ValueError("OVERNIGHT_INDEX_DATE_OR_DUPLICATE")
            if any(type(row[k]) not in (int, float) or not isfinite(row[k]) for k in FIELDS[2:]):
                raise ValueError("OVERNIGHT_NUMBER_INVALID")
            if min(row["close"], row["pre_close"]) <= 0:
                raise ValueError("OVERNIGHT_NONPOSITIVE_PRICE")
            if abs((row["close"] / row["pre_close"] - 1) * 100 - row["pct_chg"]) > 0.0001:
                raise ValueError("OVERNIGHT_RETURN_INCONSISTENT")
            result[day.isoformat()] = row
    return dict(sorted(result.items()))


def coverage(history: dict[str, dict], market: dict[str, datetime]) -> dict:
    """和独立官方日历逐日对照；缺失数据不能被误识别为美股休市。"""
    missing, unexpected = sorted(market.keys() - history.keys()), sorted(history.keys() - market.keys())
    inconsistent = []
    ordered = list(history.items())
    # 前收盘价是额外交叉核对。仅日历完整时检查连续性，避免把缺日伪装成价格修订。
    if not missing and not unexpected:
        for (_, previous), (day, current) in zip(ordered, ordered[1:], strict=False):
            if abs(previous["close"] - current["pre_close"]) > 0.0001:
                inconsistent.append(day)
    return {
        "status": "COMPLETE" if not (missing or unexpected or inconsistent) else "INVALID",
        "expected": len(market),
        "received": len(history),
        "years": {
            str(y): {
                "expected": sum(d.startswith(str(y)) for d in market),
                "received": sum(d.startswith(str(y)) for d in history),
            }
            for y in YEARS
        },
        "missing": missing,
        "unexpected": unexpected,
        "previous_close_inconsistent": inconsistent,
    }


def align_dates(pairs: list[tuple[str, str]], history: dict[str, dict], market: dict[str, datetime]) -> list[dict]:
    """仅生成对齐诊断，不能直接喂模型；中国长假累计多个美股日，目标U日晚间数据被排除。"""
    if coverage(history, market)["status"] != "COMPLETE":
        raise ValueError("OVERNIGHT_COVERAGE_INCOMPLETE")
    cn_days = calendar()[0]
    following = dict(zip(cn_days[:-1], cn_days[1:], strict=True))
    if len(pairs) != len(set(pairs)):
        raise ValueError("OVERNIGHT_PAIR_DUPLICATE")
    ordered = sorted(market, key=market.get)
    closes = [market[d] for d in ordered]
    result = []
    for t_text, u_text in sorted(pairs):
        t, u = date.fromisoformat(t_text), date.fromisoformat(u_text)
        if not 2021 <= t.year <= u.year <= 2024 or following.get(t) != u:
            raise ValueError("OVERNIGHT_CN_PAIR_INVALID")
        start, cutoff = datetime.combine(t, time(15), ZONE), datetime.combine(u, time(8), ZONE)
        before, latest = bisect_right(closes, start) - 1, bisect_right(closes, cutoff) - 1
        if before < 0 or latest < before or cutoff > datetime(2024, 12, 31, 8, tzinfo=ZONE):
            raise ValueError("OVERNIGHT_BASE_OR_WINDOW_UNCOVERED")
        base, last = ordered[before], ordered[latest]
        selected = ordered[before + 1 : latest + 1]
        result.append(
            {
                "t": t_text,
                "u": u_text,
                "cutoff": cutoff.isoformat(),
                "baseline_us_date": base,
                "latest_us_date": last,
                "us_sessions": selected,
                "session_count": len(selected),
                "close_events": [market[d].isoformat() for d in selected],
                "diagnostic_return": history[last]["close"] / history[base]["close"] - 1,
                "input_available_at": None,
                "historical_first_version_verified": False,
                "status": "DIAGNOSTIC_ONLY_PROVIDER_TIMING_UNVERIFIED",
            }
        )
    return result


def inspect(base: Path, data: Path) -> dict:
    """离线复核固定四年请求回执、旧共同样本和对齐；不联网、不拟合、不读取保护年答案。"""
    receipt = read(data / "history-receipt.json")
    if receipt["status"] != "RECEIVED" or receipt["api_calls"] != 4 or len(receipt["requests"]) != 4:
        raise ValueError("OVERNIGHT_ACQUISITION_INCOMPLETE")
    probe, reservation = read(data / "probe-result.json"), read(data / "history-reserved.json")
    if (
        probe.get("status") != "PERMISSION_AVAILABLE_NOT_USAGE_OR_TIMING_PROOF"
        or reservation.get("api_name") != "index_global"
        or reservation.get("index") != "SPX"
        or reservation.get("years") != list(YEARS)
        or reservation.get("max_requests") != 4
    ):
        raise ValueError("OVERNIGHT_CAPABILITY_OR_RESERVATION_INVALID")
    payloads = []
    dependencies = [
        data / "history-receipt.json",
        data / "history-reserved.json",
        data / "source-audit.json",
        data / "probe-result.json",
        data / "probe-reserved.json",
        CALENDAR,
    ]
    project = Path(__file__).resolve().parents[2]
    dependencies.extend(
        project / n
        for n in (
            "app/services/direction_1d_protocol.py",
            "app/services/direction_1d_training.py",
            "app/services/direction_training_artifacts.py",
            "scripts/direction_1d_overnight_audit.py",
            "app/data/calendars/cn_a_share_2021_2025_v1.json",
            "app/data/calendars/cn_a_share_2026_v1.json",
        )
    )
    for year, item in zip(YEARS, receipt["requests"], strict=True):
        path, item_path = data / f"history-{year}.json", data / f"history-{year}-receipt.json"
        if (
            item != read(item_path)
            or item["year"] != year
            or item["http_status"] != 200
            or file_hash(path) != item["data_file_sha256"]
        ):
            raise ValueError("OVERNIGHT_RECEIPT_OR_HASH_INVALID")
        payload = read(path)
        if any(not str(v[1]).startswith(str(year)) for v in payload["items"]):
            raise ValueError("OVERNIGHT_REQUEST_YEAR_MISMATCH")
        payloads.append(payload)
        dependencies.extend([path, item_path, data / f"history-{year}-reserved.json"])
    history, market = validate_history(payloads), sessions()
    counts = coverage(history, market)
    groups, pairs, membership = {}, {}, {}
    for name in ("universe", "exam"):
        path = base / f"{name}.json"
        dependencies.append(path)
        rows = read(path)
        identities = [(r["fund_code"], r["t"], r["u"]) for r in rows]
        if len(identities) != len(set(identities)) or any(
            not "2021" <= r["t"][:4] <= r["u"][:4] <= "2024" for r in rows
        ):
            raise ValueError("OVERNIGHT_COHORT_INVALID")
        membership[name] = set(identities)
        pairs[name] = sorted({(r["t"], r["u"]) for r in rows})
        groups[name] = {
            "questions": len(rows),
            "funds": len({r["fund_code"] for r in rows}),
            "date_pairs": len(pairs[name]),
            "first_t": min(r["t"] for r in rows),
            "last_u": max(r["u"] for r in rows),
        }
    if not membership["exam"].issubset(membership["universe"]):
        raise ValueError("OVERNIGHT_EXAM_NOT_IN_UNIVERSE")
    aligned = align_dates(pairs["universe"], history, market) if counts["status"] == "COMPLETE" else []
    exam_dates = set(pairs["exam"])
    aligned_exam = [r for r in aligned if (r["t"], r["u"]) in exam_dates]
    for name, selected in (("universe", aligned), ("exam", aligned_exam)):
        groups[name].update(
            aligned_date_pairs=len(selected),
            sessions_per_interval={str(k): v for k, v in sorted(Counter(r["session_count"] for r in selected).items())},
            timing_verified_date_pairs=0,
        )
    # index_global已公布的响应字段不含首次发布/历史接收时刻。本版审计不接受手填的verified开关，
    # 也不提供“跳过检查”训练参数；将来接入可核验的历史回执，需要单独实现并验收证据链。
    return {
        "status": "BLOCKED_HISTORICAL_TIMING" if counts["status"] == "COMPLETE" else "BLOCKED_COVERAGE",
        "training_ready": False,
        "training_fits": 0,
        "recipe": RECIPE,
        "coverage": counts,
        "cohorts": groups,
        "availability": {
            "market_close_calendar_verified": True,
            "provider_readable_before_u08_verified": False,
            "historical_first_version_verified": False,
            "reason": "NO_HISTORICAL_PUBLICATION_OR_RECEIPT_TIMESTAMP",
        },
        "dependencies": {str(p.resolve()): file_hash(p) for p in dependencies},
        "alignment": aligned,
    }


def save_audit(base: Path, data: Path, output: Path) -> dict:
    """保存不能覆盖的资格审计，旧训练包只读；运行前建立新目录，重复调用拒绝。"""
    output.mkdir(parents=True, exist_ok=False)
    report = inspect(base, data)
    aligned = report.pop("alignment")
    write_new(output / "alignment-diagnostic.json", aligned)
    write_new(output / "readiness.json", report)
    write_new(
        output / "audit-receipt.json",
        {
            "at": datetime.now(ZONE).isoformat(),
            "base": str(base.resolve()),
            "data": str(data.resolve()),
            "files": {n: file_hash(output / n) for n in ("alignment-diagnostic.json", "readiness.json")},
            "code": {str(Path(__file__).resolve()): file_hash(Path(__file__))},
            "database_writes": 0,
            "training_fits": 0,
        },
    )
    return {k: report[k] for k in ("status", "training_ready", "coverage", "cohorts")}


def verify(output: Path) -> dict:
    """重算全部资格诊断并检查文件hash；复核不能凭修改报告字段把未证明的数据变成可训练。"""
    receipt = read(output / "audit-receipt.json")
    for name, expected in receipt["files"].items():
        if file_hash(output / name) != expected:
            raise ValueError("OVERNIGHT_AUDIT_FILE_CHANGED")
    for path, expected in receipt["code"].items():
        if file_hash(Path(path)) != expected:
            raise ValueError("OVERNIGHT_AUDIT_CODE_CHANGED")
    actual = inspect(Path(receipt["base"]), Path(receipt["data"]))
    alignment = actual.pop("alignment")
    if canonical(actual) != canonical(read(output / "readiness.json")) or alignment != read(
        output / "alignment-diagnostic.json"
    ):
        raise ValueError("OVERNIGHT_AUDIT_REPLAY_MISMATCH")
    return {"verification": "PASS", "status": actual["status"], "training_ready": False, "training_fits": 0}
