"""发行方FXI份额增减的一日输入，复用第31轮原始文件，不新增网络请求。

份额数量变化不能直接称为资金流入。显式缺失与结构跳变保留标记，缺日期直接失败。
"""

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, time

import numpy as np

from app.integrations.ishares_sprint_fxi import MAX_BYTES, URL
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_fxi_gap_data as parent
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_vix_data as lagged

SOURCE = "ISHARES_FXI_SHARES_PERSONAL_RESEARCH"


def root():
    return b.ROOT / "round-41/shares-inputs"


def parse(raw):
    """按已知五列工作表读取份额；份额数量不是金额，不能称为资金净流入。"""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError("SHARES_RAW_SIZE_INVALID")
    blocks = re.findall(r'<ss:Worksheet ss:Name="Historical">.*?</ss:Worksheet>', raw.decode("utf-8-sig"), re.S)
    if len(blocks) != 1 or "<!" in blocks[0]:
        raise ValueError("SHARES_SHEET_INVALID")
    ns = {"s": "urn:schemas-microsoft-com:office:spreadsheet"}
    tree = ET.fromstring('<ss:Workbook xmlns:ss="' + ns["s"] + '">' + blocks[0] + "</ss:Workbook>")
    records = []
    for row in tree.findall("s:Worksheet/s:Table/s:Row", ns):
        cells = row.findall("s:Cell", ns)
        if (
            len(records) > 12000
            or row.get("{" + ns["s"] + "}Index") is not None
            or any(c.get("{" + ns["s"] + "}Index") is not None or c.find("s:Data", ns) is None for c in cells)
        ):
            raise ValueError("SHARES_SPARSE_OR_ROW_LIMIT")
        records.append(["".join(c.find("s:Data", ns).itertext()) for c in cells])
    if not records or records[0] != ["As Of", "NAV per Share", "Ex-Dividends", "Shares Outstanding", "Non-FV NAV"]:
        raise ValueError("SHARES_FIELDS_CHANGED")
    points, prior = {}, None
    for row in records[1:]:
        if len(row) != 5:
            raise ValueError("SHARES_ROW_WIDTH")
        day = datetime.strptime(row[0], "%b %d, %Y").date().isoformat()
        if prior is not None and day >= prior:
            raise ValueError("SHARES_DATE_ORDER")
        prior = day
        if day < "2021-01-01":
            continue
        value = row[3]
        if value != "--" and not re.fullmatch(r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?", value):
            raise ValueError("SHARES_NUMBER_INVALID")
        shares = None if value == "--" else float(value.replace(",", ""))
        if shares is not None and (not math.isfinite(shares) or shares < 0):
            raise ValueError("SHARES_NUMBER_INVALID")
        reason = "EXPLICIT_MISSING" if shares is None else "INVALID_ZERO" if shares == 0 else None
        points[day] = {"shares": shares, "available": reason is None, "unavailable_reason": reason}
    if not points:
        raise ValueError("SHARES_EMPTY")
    return points


def required(base_day, target):
    """六个连续美股现金日期，最新收盘不晚于基准日15点；目标仍为相邻内地交易日。"""
    anchor = lagged.required(base_day, target)[-1]
    keys = list(overnight.sessions())
    index = keys.index(anchor)
    if index < 5:
        raise ValueError("SHARES_CALENDAR_UNCOVERED")
    return keys[index - 5 : index + 1]


def features(base_day, target, points):
    needed = required(base_day, target)
    if any(day not in points for day in needed):
        raise ValueError("SHARES_REQUIRED_DATE_ABSENT")
    values = [points[d]["shares"] for d in needed]
    available = True
    for day, value in zip(needed, values, strict=True):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        ):
            raise ValueError("SHARES_FEATURE_NUMBER_INVALID")
        reason = "EXPLICIT_MISSING" if value is None else "INVALID_ZERO" if value == 0 else None
        if points[day]["available"] is not (reason is None) or points[day]["unavailable_reason"] != reason:
            raise ValueError("SHARES_REASON_CHANGED")
        available &= reason is None
    if not available:
        return [0.0, 0.0, 0.0, 0.0]
    daily = [100 * (right / left - 1) for left, right in zip(values[:-1], values[1:], strict=True)]
    if any(abs(change) >= 50 for change in daily):
        return [0.0, 0.0, 1.0, 0.0]
    return np.clip([daily[-1], 100 * (values[-1] / values[0] - 1)], -50, 50).tolist() + [0.0, 1.0]


def history():
    """核对事前规则、原始来源和派生快照；重算份额值防止只重签JSON绕过原文件。"""
    folder, original = b.ROOT / "round-41", b.ROOT / "fxi-data-feasibility-v1"
    proposal = b.read(folder / "proposal-before-training.json")
    snapshot = b.read(folder / "shares-history.json")
    result = b.read(folder / "input-feasibility.json")
    capture, old_plan = b.read(original / "result.json"), b.read(original / "plan.json")
    raw = (original / "response.xls").read_bytes()
    if (
        capture["plan_hash"] != b.digest(old_plan)
        or old_plan["url"] != URL
        or hashlib.sha256(raw).hexdigest() != capture["sha256"]
        or len(raw) != capture["bytes"]
        or proposal["raw_sha256"] != capture["sha256"]
        or proposal["capture_hash"] != b.digest(capture)
        or hashlib.sha256((folder / "shares-feasibility.py").read_bytes()).hexdigest() != proposal["script_sha256"]
        or snapshot["proposal_hash"] != b.digest(proposal)
        or result["proposal_hash"] != b.digest(proposal)
        or result["history_hash"] != b.digest(snapshot)
        or result["status"] != "FEASIBLE_NOT_TRAINED"
    ):
        raise ValueError("SHARES_HISTORICAL_SOURCE_CHANGED")
    points = parse(raw)
    if points != snapshot["points"]:
        raise ValueError("SHARES_HISTORICAL_PARSED_CHANGED")
    return points


def parent_rows(target):
    """第31轮回读验证请求、响应和原始字节后，再从同一文件提取六个份额日期。"""
    value = parent.load(target)
    slot = value["raw_ref"]["slot"]
    meta = b.read(parent.root() / target / f"raw/{slot}.json")
    raw = (parent.root() / target / f"raw/{slot}.xls").read_bytes()
    if (
        value["raw_ref"]["hash"] != b.digest(meta)
        or hashlib.sha256(raw).hexdigest() != meta["body_sha256"]
        or len(raw) != meta["bytes"]
    ):
        raise ValueError("SHARES_PARENT_RAW_CHANGED")
    points = parse(raw)
    features(value["base"], target, points)
    return value, {d: points[d] for d in required(value["base"], target)}


def load(target):
    """份额快照、父输入与截止时点一起验证；父记录有效不代表可随意改写份额值。"""
    value = b.read(root() / target / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), b.ZONE)
    if value["target"] != target or value["source"] != SOURCE or datetime.fromisoformat(value["at"]) >= deadline:
        raise ValueError("SHARES_LIVE_SCOPE_OR_TIME_INVALID")
    parent_input, rows = parent_rows(target)
    if (
        value["parent_input_hash"] != b.digest(parent_input)
        or value["base"] != parent_input["base"]
        or value["rows"] != rows
        or value["raw_ref"] != parent_input["raw_ref"]
        or datetime.fromisoformat(parent_input["at"]) > datetime.fromisoformat(value["at"])
    ):
        raise ValueError("SHARES_LIVE_INPUT_CHANGED")
    return value


def capture(at):
    """仅复用已成功采集的父输入；父分支失败则本轮拒绝，不能另开请求重试。"""
    window = b.window(at)
    end = datetime.fromisoformat(b.read(b.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("SHARES_LIVE_WINDOW_CLOSED")
    target = window["target_nav_date"]
    path = root() / target / "input.json"
    if path.exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), b.ZONE)
    if b.now() >= deadline:
        raise ValueError("SHARES_LIVE_DEADLINE_REACHED")
    if not (parent.root() / target / "input.json").exists():
        raise ValueError("SHARES_PARENT_INPUT_UNAVAILABLE")
    parent_input, rows = parent_rows(target)
    if parent_input["base"] != window["base_nav_date"]:
        raise ValueError("SHARES_PARENT_BASE_CHANGED")
    value = {
        "at": b.now().isoformat(),
        "target": target,
        "base": parent_input["base"],
        "source": SOURCE,
        "rows": rows,
        "parent_input_hash": b.digest(parent_input),
        "raw_ref": parent_input["raw_ref"],
    }
    if b.now() >= deadline:
        raise ValueError("SHARES_LIVE_DEADLINE_REACHED")
    b.save(path, value)
    checked = load(target)
    if b.now() >= deadline:
        raise ValueError("SHARES_LIVE_READBACK_LATE")
    return checked
