"""FXI两种净值口径差及不可用标记，保留日期、来源和无效原因，不冒充市场价格。

非FV净值为发行方明确的--或无效零时，三项特征以可用标记区分占位零值。
所需原始日期缺失、负数、非有限值和结构变化均拒绝，不能用可用标记掩盖采集失败。
"""

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, time
from functools import lru_cache

import numpy as np

from app.integrations.ishares_sprint_fxi import MAX_BYTES, URL, fetch_fxi
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_vix_data as lagged

SOURCE = "ISHARES_FXI_NAV_GAP_PERSONAL_RESEARCH"
SLOTS = ("0700", "0730", "0800")
MAX_ROWS = 12000


def root():
    return base.ROOT / "round-31" / "fxi-inputs"


@lru_cache(maxsize=4)
def xml_rows(raw):
    """只提取一个原样Historical工作表；其他表的未转义字符不需要修补原始文件。

    只缓存不可变行，避免30只基金重复解析。拒绝实体/DTD、稀疏单元格及超出行数的文件。
    """
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError("FXI_RAW_SIZE_INVALID")
    blocks = re.findall(r'<ss:Worksheet ss:Name="Historical">.*?</ss:Worksheet>', raw.decode("utf-8-sig"), re.S)
    if len(blocks) != 1 or "<!" in blocks[0]:
        raise ValueError("FXI_HISTORICAL_SHEET_INVALID")
    ns = {"s": "urn:schemas-microsoft-com:office:spreadsheet"}
    try:
        tree = ET.fromstring('<ss:Workbook xmlns:ss="' + ns["s"] + '">' + blocks[0] + "</ss:Workbook>")
    except ET.ParseError:
        # 结构损坏属于采集内容无效，统一拒绝，不能按字段缺失继续训练或生成预测。
        raise ValueError("FXI_HISTORICAL_XML_INVALID") from None
    records = []
    for row in tree.findall("s:Worksheet/s:Table/s:Row", ns):
        cells = row.findall("s:Cell", ns)
        if (
            len(records) > MAX_ROWS
            or row.get("{" + ns["s"] + "}Index") is not None
            or any(c.get("{" + ns["s"] + "}Index") is not None or c.find("s:Data", ns) is None for c in cells)
        ):
            raise ValueError("FXI_ROW_LIMIT_OR_SPARSE_CELL")
        records.append(["".join(c.find("s:Data", ns).itertext()) for c in cells])
    if not records or records[0] != ["As Of", "NAV per Share", "Ex-Dividends", "Shares Outstanding", "Non-FV NAV"]:
        raise ValueError("FXI_HISTORICAL_FIELDS_CHANGED")
    output, prior = [], None
    for row in records[1:]:
        if len(row) != 5:
            raise ValueError("FXI_HISTORICAL_ROW_WIDTH")
        day = datetime.strptime(row[0], "%b %d, %Y").date().isoformat()
        if prior is not None and day >= prior:
            raise ValueError("FXI_DATE_ORDER_OR_DUPLICATE")
        prior = day
        if day < "2021-01-01":
            continue
        nav = float(row[1])
        nonfv = None if row[4] == "--" else float(row[4])
        if not math.isfinite(nav) or nav <= 0 or (nonfv is not None and (not math.isfinite(nonfv) or nonfv < 0)):
            raise ValueError("FXI_INVALID_NUMBER_IS_NOT_MISSING")
        reason = "EXPLICIT_MISSING" if nonfv is None else "INVALID_ZERO" if nonfv == 0 else None
        output.append((day, nav, nonfv, reason))
    if not output:
        raise ValueError("FXI_EMPTY_HISTORY")
    return tuple(output)


def parse(raw):
    return {
        day: {"nav_per_share": nav, "non_fv_nav": nonfv, "available": reason is None, "unavailable_reason": reason}
        for day, nav, nonfv, reason in xml_rows(raw)
    }


def required(base_day, target):
    """最近US现金收盘不晚于基准日15:00，及其前一现金日期；内部也检查国内相邻目标日。"""
    return lagged.required(base_day, target)


def features(base_day, target, points):
    needed = required(base_day, target)
    if any(day not in points for day in needed):
        raise ValueError("FXI_REQUIRED_ROW_ABSENT")
    available = True
    for day in needed:
        row = points[day]
        nav, nonfv = row["nav_per_share"], row["non_fv_nav"]
        if (
            isinstance(nav, bool)
            or not isinstance(nav, (float, int))
            or not math.isfinite(nav)
            or nav <= 0
            or (
                nonfv is not None
                and (
                    isinstance(nonfv, bool)
                    or not isinstance(nonfv, (float, int))
                    or not math.isfinite(nonfv)
                    or nonfv < 0
                )
            )
        ):
            raise ValueError("FXI_FEATURE_NUMBER_INVALID")
        reason = "EXPLICIT_MISSING" if nonfv is None else "INVALID_ZERO" if nonfv == 0 else None
        if row["available"] is not (reason is None) or row["unavailable_reason"] != reason:
            raise ValueError("FXI_AVAILABILITY_REASON_CHANGED")
        available &= reason is None
    if not available:
        return [0.0, 0.0, 0.0]
    prior, anchor = [100 * (points[d]["nav_per_share"] / points[d]["non_fv_nav"] - 1) for d in needed]
    return np.clip([anchor, anchor - prior], -20, 20).tolist() + [1.0]


def history():
    """重新验证发行方原始文件和独立缺失方案，保证缺失标记不是对已保存数据的事后修改。"""
    original, folder = base.ROOT / "fxi-data-feasibility-v1", base.ROOT / "fxi-masked-feasibility-v2"
    capture, original_plan = base.read(original / "result.json"), base.read(original / "plan.json")
    plan, result, snapshot = (base.read(folder / name) for name in ("plan.json", "result.json", "history.json"))
    raw = (original / "response.xls").read_bytes()
    if (
        capture["plan_hash"] != base.digest(original_plan)
        or original_plan["url"] != URL
        or hashlib.sha256(raw).hexdigest() != capture["sha256"]
        or len(raw) != capture["bytes"]
        or plan["raw_sha256"] != capture["sha256"]
        or plan["capture_hash"] != base.digest(capture)
        or result["plan_hash"] != base.digest(plan)
        or snapshot["plan_hash"] != base.digest(plan)
        or result["history_hash"] != base.digest(snapshot)
        or hashlib.sha256((folder / "check.py").read_bytes()).hexdigest() != plan["script_sha256"]
    ):
        raise ValueError("FXI_HISTORICAL_SOURCE_CHANGED")
    points = parse(raw)
    if points != snapshot["points"]:
        raise ValueError("FXI_HISTORICAL_PARSED_CHANGED")
    return points


def load(target):
    """回读原始发行方文件和请求时点，拒绝只修改外层哈希的解析值或已过截止的响应。"""
    folder = root() / target
    value = base.read(folder / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if value["target"] != target or value["source"] != SOURCE or datetime.fromisoformat(value["at"]) >= deadline:
        raise ValueError("FXI_LIVE_INPUT_SCOPE_OR_TIME_INVALID")
    needed = required(value["base"], target)
    slot = value["raw_ref"]["slot"]
    if slot not in SLOTS:
        raise ValueError("FXI_LIVE_SLOT_INVALID")
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
        raise ValueError("FXI_LIVE_RESPONSE_CHANGED_OR_LATE")
    raw = (folder / f"raw/{slot}.xls").read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["body_sha256"] or len(raw) != meta["bytes"]:
        raise ValueError("FXI_LIVE_RAW_CHANGED")
    points = parse(raw)
    if any(day not in points for day in needed) or {d: points[d] for d in needed} != value["rows"]:
        raise ValueError("FXI_LIVE_PARSED_INPUT_CHANGED")
    return value


def capture(at):
    """每目标最多三个固定时槽，每槽一次原始文件请求；已成功则全基金共享，无自动重试。"""
    window = base.window(at)
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("FXI_LIVE_WINDOW_CLOSED")
    target, base_day = window["target_nav_date"], window["base_nav_date"]
    folder = root() / target
    if (folder / "input.json").exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if base.now() >= deadline:
        raise ValueError("FXI_LIVE_DEADLINE_REACHED")
    slot = "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"
    attempt = folder / f"requests/{slot}.json"
    if attempt.exists() or len(list((folder / "requests").glob("*.json"))) >= 3:
        raise ValueError("FXI_LIVE_SLOT_OR_BUDGET_EXHAUSTED")
    needed = required(base_day, target)
    request = {"at": base.now().isoformat(), "url": URL, "target": target, "base": base_day, "required_dates": needed}
    base.save(attempt, request)
    try:
        body, headers = fetch_fxi()
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
            raise ValueError("FXI_LIVE_RESPONSE_LATE")
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
            raise ValueError("FXI_LIVE_DEADLINE_REACHED")
        base.save(folder / "input.json", value)
        checked = load(target)
        if base.now() >= deadline:
            raise ValueError("FXI_LIVE_READBACK_LATE")
        return checked
    except Exception as exc:
        base.save(folder / f"errors-{slot}.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        raise
