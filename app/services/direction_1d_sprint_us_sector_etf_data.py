"""半导体和金属矿业ETF成交价格的一日输入，历史开发与真实08:30前采集各自留证。"""

import hashlib
from datetime import date, datetime, time

import numpy as np

from app.integrations.sina_sprint_sector_etf import MAX_BYTES, fetch_etf, parse, url_for
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_fxi_overnight_data as timing
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_sector_data as slots

SYMBOLS = ("SOXX", "XME")
SOURCE = "SINA_US_SECTOR_ETF_QUOTE_V1"
SLOTS = ("0700", "0730", "0800")
required = timing.required
closed_before = timing.closed_before


def slot_for(at):
    """只有北京时间07:00至08:30允许采集，不把窗口外时刻塞入最后一个时槽。"""
    if at.tzinfo is None:
        raise ValueError("US_ETF_CAPTURE_TIMEZONE_REQUIRED")
    local = at.astimezone(base.ZONE)
    return slots.slot_for(local) if time(7) <= local.time() < time(8, 30) else None


def root():
    return base.ROOT / "round-56/sector-inputs"


def features(base_day, target, points):
    """各ETF只追加最新完整美股日开到收百分数，最后标记两源是否均可用。

    美股最近一次收盘若早于基准日T的15:00，则没有新的隔夜现金市场信息，回退原HK。
    历史OHLC异常行保持原价且回退，不能删掉基金问题或改价提高开发成绩。
    仅使用同一交易日开到收，避免跨日拆股制造巨额变化；不称总回报。
    """
    days = required(base_day, target)
    if set(points) != set(SYMBOLS) or any(day not in points[s] for s in SYMBOLS for day in days):
        raise ValueError("US_ETF_REQUIRED_DATE_MISSING")
    values, available = [], True
    for symbol in SYMBOLS:
        prior, latest = [points[symbol][day] for day in days]
        for row in (prior, latest):
            if any(
                type(row.get(k)) not in (int, float) or not np.isfinite(row[k]) or row[k] <= 0
                for k in ("open", "high", "low", "close")
            ):
                raise ValueError("US_ETF_FEATURE_PRICE_INVALID")
            if type(row.get("volume")) not in (int, float) or not np.isfinite(row["volume"]) or row["volume"] < 0:
                raise ValueError("US_ETF_FEATURE_VOLUME_INVALID")
            valid = (
                row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]
                and row["volume"] > 0
            )
            if row["available"] is not valid or row["unavailable_reason"] != (
                None if valid else "OHLC_RANGE_OR_NO_TRADE"
            ):
                raise ValueError("US_ETF_QUALITY_FLAG_CHANGED")
            available &= valid
        values += [100 * (latest["close"] / latest["open"] - 1)]
    china_close = datetime.combine(date.fromisoformat(base_day), time(15), base.ZONE)
    available &= overnight.sessions()[days[-1]] > china_close
    return np.clip(values, -20, 20).tolist() + [1.0] if available else [0.0] * 3


def history():
    """以原始HTTP字节重新解码并核对既有来源验收，不能改解析表绕过原始证据。"""
    folder = base.ROOT / "us-sector-etf-source-v1"
    plan, result, snapshot = (
        base.read(folder / name) for name in ("qualification-plan.json", "qualification-result.json", "history.json")
    )
    if (
        result["plan_hash"] != base.digest(plan)
        or snapshot["plan_hash"] != base.digest(plan)
        or result["history_hash"] != base.digest(snapshot)
    ):
        raise ValueError("US_ETF_HISTORY_MANIFEST_CHANGED")
    if result["status"] != "QUALIFIED_RAW_PRICE_HISTORY":
        raise ValueError("US_ETF_HISTORY_NOT_QUALIFIED")
    for name, digest in plan["code_hashes"].items():
        if hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("US_ETF_HISTORY_PARSER_CHANGED")
    combined = {}
    for symbol in SYMBOLS:
        label = symbol
        raw = (folder / f"response-{label}.bin").read_bytes()
        receipt, request = base.read(folder / f"receipt-{label}.json"), base.read(folder / f"request-{label}.json")
        if (
            request["url"] != url_for(symbol)
            or receipt["request_hash"] != base.digest(request)
            or receipt["status"] != "RECEIVED"
            or receipt["bytes"] != len(raw)
            or receipt["sha256"] != hashlib.sha256(raw).hexdigest()
            or receipt["sha256"] != plan["raw_hashes"][symbol]
        ):
            raise ValueError("US_ETF_HISTORY_RAW_CHANGED")
        combined[symbol] = parse(raw, symbol, "2026-09-14")
    if combined != snapshot["points"]:
        raise ValueError("US_ETF_HISTORY_PARSED_CHANGED")
    return combined


def load_piece(target, base_day, symbol, slot):
    """精确绑定证券、目标日期、请求、响应及实际收盘时刻；禁止跨来源冒用。"""
    if symbol not in SYMBOLS or slot not in SLOTS:
        raise ValueError("US_ETF_LIVE_ROUTE_INVALID")
    folder = root() / target / symbol
    meta = base.read(folder / f"raw/{slot}.json")
    request = base.read(folder / f"requests/{slot}.json")
    raw = (folder / f"raw/{slot}.bin").read_bytes()
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    received = datetime.fromisoformat(meta["received_at"])
    if (
        meta["request_hash"] != base.digest(request)
        or meta["source"] != SOURCE
        or meta["symbol"] != symbol
        or request["symbol"] != symbol
        or request["target"] != target
        or request["base"] != base_day
        or request["url"] != url_for(symbol)
        or request["required_dates"] != required(base_day, target)
        or not datetime.fromisoformat(request["at"]) <= received < deadline
        or not raw
        or len(raw) > MAX_BYTES
        or len(raw) != meta["bytes"]
        or hashlib.sha256(raw).hexdigest() != meta["body_sha256"]
    ):
        raise ValueError("US_ETF_LIVE_RAW_OR_TIME_CHANGED")
    closed_before(received, base_day, target)
    needed = required(base_day, target)
    points = parse(raw, symbol, needed[-1])
    if any(day not in points for day in needed):
        raise ValueError("US_ETF_LATEST_CLOSE_NOT_ARRIVED")
    return {day: points[day] for day in needed}, meta


def load(target):
    value = base.read(root() / target / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    created = datetime.fromisoformat(value["at"])
    if (
        value["target"] != target
        or value["source"] != SOURCE
        or created >= deadline
        or set(value["rows"]) != set(SYMBOLS)
    ):
        raise ValueError("US_ETF_LIVE_INPUT_SCOPE_INVALID")
    for symbol in SYMBOLS:
        ref = value["raw_refs"][symbol]
        points, meta = load_piece(target, value["base"], symbol, ref["slot"])
        if (
            points != value["rows"][symbol]
            or base.digest(meta) != ref["hash"]
            or datetime.fromisoformat(meta["received_at"]) > created
        ):
            raise ValueError("US_ETF_LIVE_PARSED_INPUT_CHANGED")
    features(value["base"], target, value["rows"])
    return value


def capture(at):
    """每个0700/0730/0800槽每只最多一次；晚到、缺最新行或中断均不伪造成功输入。"""
    slot = slot_for(at)
    target = str(at.astimezone(base.ZONE).date())
    days = list(map(str, base.calendar()[0]))
    if slot is None or target not in days:
        return None
    base_day = days[days.index(target) - 1]
    path = root() / target / "input.json"
    if path.exists():
        return load(target)
    needed = required(base_day, target)
    combined, refs = {}, {}
    for symbol in SYMBOLS:
        folder = root() / target / symbol
        for previous in SLOTS:
            if not (folder / f"raw/{previous}.json").exists():
                continue
            try:
                points, meta = load_piece(target, base_day, symbol, previous)
                combined[symbol], refs[symbol] = points, {"slot": previous, "hash": base.digest(meta)}
                break
            except ValueError:
                continue
        if symbol in combined:
            continue
        request_path = folder / f"requests/{slot}.json"
        if request_path.exists():
            continue
        request = {
            "at": base.now().isoformat(),
            "source": SOURCE,
            "symbol": symbol,
            "target": target,
            "base": base_day,
            "required_dates": needed,
            "url": url_for(symbol),
        }
        base.save(request_path, request)
        try:
            raw, headers = fetch_etf(symbol)
            received = base.now().isoformat()
            raw_path = folder / f"raw/{slot}.bin"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_path.open("xb") as output:
                output.write(raw)
            meta = {
                "source": SOURCE,
                "symbol": symbol,
                "received_at": received,
                "request_hash": base.digest(request),
                "body_sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "headers": headers,
            }
            base.save(folder / f"raw/{slot}.json", meta)
            points, meta = load_piece(target, base_day, symbol, slot)
            combined[symbol], refs[symbol] = points, {"slot": slot, "hash": base.digest(meta)}
        except Exception as exc:
            base.save(
                folder / f"failures/{slot}.json",
                {"at": base.now().isoformat(), "error": base.error_code(exc), "exception_type": type(exc).__name__},
            )
    if set(combined) != set(SYMBOLS):
        return None
    features(base_day, target, combined)
    created = base.now()
    if created >= datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE):
        return None
    value = {
        "at": created.isoformat(),
        "source": SOURCE,
        "target": target,
        "base": base_day,
        "raw_refs": refs,
        "rows": combined,
    }
    base.save(path, value)
    return load(target)
