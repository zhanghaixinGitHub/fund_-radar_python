"""CNYA发行方两种净值口径差；复用相同字段和最新现金时点规则，独立验证来源。

历史首发时间未获验证；未来必须取得CNYA自己的原始文件，不能复用FXI文件或伪造及时输入。
"""

import hashlib
from datetime import date, datetime, time
from pathlib import Path

from app.integrations.ishares_sprint_cnya import MAX_BYTES, URL, fetch_cnya
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_fxi_gap_data as parser
from app.services import direction_1d_sprint_fxi_overnight_data as timing

SOURCE = "ISHARES_CNYA_LATEST_COMPLETED_US_VALUATION_GAP"
SLOTS = ("0700", "0730", "0800")
parse = parser.parse
required = timing.required
features = timing.features
closed_before = timing.closed_before


def root():
    return base.ROOT / "round-43/cnya-inputs"


def history():
    """原字节、下载地址、请求链、解析器和快照必须一致，重算外层哈希不能替换原始字段。"""
    folder = base.ROOT / "cnya-data-feasibility-v1"
    plan, request, capture, result, snapshot = (
        base.read(folder / name)
        for name in ("plan.json", "request.json", "capture.json", "result.json", "history.json")
    )
    raw = (folder / "response.xls").read_bytes()
    if (
        plan["url"] != URL
        or request["url"] != URL
        or capture["url"] != URL
        or request["plan_hash"] != base.digest(plan)
        or capture["plan_hash"] != base.digest(plan)
        or capture["request_hash"] != base.digest(request)
        or result["plan_hash"] != base.digest(plan)
        or snapshot["plan_hash"] != base.digest(plan)
        or snapshot["capture_hash"] != base.digest(capture)
        or result["capture_hash"] != base.digest(capture)
        or result["history_hash"] != base.digest(snapshot)
        or len(raw) != capture["bytes"]
        or len(raw) > MAX_BYTES
        or hashlib.sha256(raw).hexdigest() != capture["sha256"]
        or result["status"] != "FEASIBLE_NOT_TRAINED"
        or hashlib.sha256((folder / "probe.py").read_bytes()).hexdigest() != plan["script_sha256"]
        or hashlib.sha256((base.PROJECT / "app/integrations/ishares_sprint_cnya.py").read_bytes()).hexdigest()
        != plan["client_sha256"]
        or hashlib.sha256(Path(parser.__file__).read_bytes()).hexdigest() != plan["parser_sha256"]
        or hashlib.sha256(Path(timing.__file__).read_bytes()).hexdigest() != plan["timing_sha256"]
        or datetime.fromisoformat(request["at"]) > datetime.fromisoformat(capture["at"])
    ):
        raise ValueError("CNYA_HISTORICAL_SOURCE_CHANGED")
    points = parse(raw)
    if points != snapshot["points"]:
        raise ValueError("CNYA_HISTORICAL_PARSED_CHANGED")
    return points


def load(target):
    """回读原始发行方文件和请求时点，拒绝只修改外层哈希的解析值或已过截止的响应。"""
    folder = root() / target
    value = base.read(folder / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if value["target"] != target or value["source"] != SOURCE or datetime.fromisoformat(value["at"]) >= deadline:
        raise ValueError("CNYA_LIVE_INPUT_SCOPE_OR_TIME_INVALID")
    needed = required(value["base"], target)
    slot = value["raw_ref"]["slot"]
    if slot not in SLOTS:
        raise ValueError("CNYA_LIVE_SLOT_INVALID")
    meta = base.read(folder / f"raw/{slot}.json")
    request = base.read(folder / f"requests/{slot}.json")
    if (
        value["raw_ref"]["hash"] != base.digest(meta)
        or meta["request_hash"] != base.digest(request)
        or request["target"] != target
        or request["base"] != value["base"]
        or request["url"] != URL
        or request["required_dates"] != needed
        or meta["source"] != SOURCE
        or not datetime.fromisoformat(request["at"]) <= datetime.fromisoformat(meta["received_at"]) < deadline
        or datetime.fromisoformat(meta["received_at"]) > datetime.fromisoformat(value["at"])
    ):
        raise ValueError("CNYA_LIVE_RESPONSE_CHANGED_OR_LATE")
    raw = (folder / f"raw/{slot}.xls").read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["body_sha256"] or len(raw) != meta["bytes"]:
        raise ValueError("CNYA_LIVE_RAW_CHANGED")
    closed_before(datetime.fromisoformat(meta["received_at"]), value["base"], target)
    points = parse(raw)
    if any(day not in points for day in needed) or {d: points[d] for d in needed} != value["rows"]:
        raise ValueError("CNYA_LIVE_PARSED_INPUT_CHANGED")
    return value


def capture(at):
    """每目标最多三个固定时槽，每槽一次原始文件请求；已成功则全基金共享，无自动重试。"""
    window = base.window(at)
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("CNYA_LIVE_WINDOW_CLOSED")
    target, base_day = window["target_nav_date"], window["base_nav_date"]
    folder = root() / target
    if (folder / "input.json").exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if base.now() >= deadline:
        raise ValueError("CNYA_LIVE_DEADLINE_REACHED")
    slot = "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"
    attempt = folder / f"requests/{slot}.json"
    if attempt.exists() or len(list((folder / "requests").glob("*.json"))) >= 3:
        raise ValueError("CNYA_LIVE_SLOT_OR_BUDGET_EXHAUSTED")
    closed_before(base.now(), base_day, target)
    needed = required(base_day, target)
    request = {"at": base.now().isoformat(), "url": URL, "target": target, "base": base_day, "required_dates": needed}
    base.save(attempt, request)
    try:
        body, headers = fetch_cnya()
        meta = {
            "received_at": base.now().isoformat(),
            "source": SOURCE,
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "headers": headers,
            "request_hash": base.digest(request),
        }
        raw_path = folder / f"raw/{slot}.xls"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("xb") as file:
            file.write(body)
        base.save(folder / f"raw/{slot}.json", meta)
        if datetime.fromisoformat(meta["received_at"]) >= deadline:
            raise ValueError("CNYA_LIVE_RESPONSE_LATE")
        points = parse(body)
        features(base_day, target, points)
        value = {
            "at": base.now().isoformat(),
            "target": target,
            "base": base_day,
            "source": SOURCE,
            "rows": {d: points[d] for d in needed},
            "raw_ref": {"slot": slot, "hash": base.digest(meta)},
        }
        if base.now() >= deadline:
            raise ValueError("CNYA_LIVE_DEADLINE_REACHED")
        base.save(folder / "input.json", value)
        checked = load(target)
        if base.now() >= deadline:
            raise ValueError("CNYA_LIVE_READBACK_LATE")
        return checked
    except Exception as exc:
        base.save(folder / f"errors-{slot}.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        raise
