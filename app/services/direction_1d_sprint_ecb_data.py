"""ECB参考汇率历史及真实到达证据；源日期最多到基金基准日T，不读取目标日汇率。"""

import csv
import hashlib
import io
import json
import math
import zipfile
from bisect import bisect_right
from datetime import date, datetime, time, timedelta
from functools import lru_cache

import numpy as np

from app.integrations.ecb_sprint_fx import MAX_BYTES, URL, fetch_ecb_fx
from app.services import direction_1d_sprint as base

SOURCE = "ECB_REFERENCE_FX_PERSONAL_RESEARCH"
SLOTS = ("0700", "0730", "0800")
MAX_UNCOMPRESSED = 15_000_000
CALENDAR = base.PROJECT / "app/data/calendars/ecb_target_2021_2026_sprint_v1.json"


def root():
    return base.ROOT / "round-26" / "ecb-inputs"


@lru_cache(maxsize=1)
def publication_dates():
    """按ECB长期TARGET规则重建应发布日期；缺数据不能被误当作节假日。"""
    spec = json.loads(CALENDAR.read_text(encoding="utf-8"))
    if spec["calendar_id"] != "ECB_TARGET_2021_2026_SPRINT_V1" or spec["years"] != [2021, 2026]:
        raise ValueError("ECB_CALENDAR_SCOPE_INVALID")
    days = []
    day = date(2021, 1, 1)
    while day.year <= 2026:
        if day.weekday() < 5 and str(day) not in spec["closing_days"][str(day.year)]:
            days.append(str(day))
        day += timedelta(days=1)
    return tuple(days)


@lru_cache(maxsize=4)
def zip_rows(raw):
    """有界解压单个CSV到内存；只提取美元与人民币的每欧元参考值，原始文件不修改。"""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError("ECB_FX_RAW_SIZE_INVALID")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if (
                len(infos) != 1
                or infos[0].filename != "eurofxref-hist.csv"
                or infos[0].file_size > MAX_UNCOMPRESSED
                or infos[0].flag_bits & 1
            ):
                raise ValueError("ECB_FX_ZIP_STRUCTURE_INVALID")
            content = archive.read(infos[0])
    except zipfile.BadZipFile as exc:
        raise ValueError("ECB_FX_ZIP_INVALID") from exc
    if len(content) > MAX_UNCOMPRESSED:
        raise ValueError("ECB_FX_UNCOMPRESSED_LIMIT")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), skipinitialspace=True)
    if not reader.fieldnames or any(reader.fieldnames.count(k) != 1 for k in ("Date", "USD", "CNY")):
        raise ValueError("ECB_FX_FIELDS_INVALID")
    rows, prior, count = [], "9999-12-31", 0
    expected = set(publication_dates())
    for item in reader:
        day = datetime.strptime(item["Date"], "%Y-%m-%d").date().isoformat()
        count += 1
        if day >= prior or count > 12000:
            raise ValueError("ECB_FX_DATE_ORDER_OR_LIMIT")
        prior = day
        if "2021-01-01" <= day <= "2026-12-31":
            if day not in expected:
                raise ValueError("ECB_FX_NONPUBLICATION_DATE")
            usd, cny = float(item["USD"]), float(item["CNY"])
            if not all(math.isfinite(v) and v > 0 for v in (usd, cny)):
                raise ValueError("ECB_FX_RATE_INVALID")
            rows.append((day, usd, cny))
    if not rows:
        raise ValueError("ECB_FX_EMPTY_RESPONSE")
    return tuple(rows)


def parse(raw):
    """缓存仅保存不可变元组，每次向调用方返回独立字典。"""
    return {day: {"usd_per_eur": usd, "cny_per_eur": cny} for day, usd, cny in zip_rows(raw)}


def required(base_day, target):
    """取基准日T及以前最近一个应发布日及前一应发布日，不向前跳过缺报日。

    ECB通常在欧洲16点左右发布T日参考值，此时内地已收盘但目标日U尚未开始。
    历史可得性仍是重建假设；真实预测必须在U日08:30前收到原始响应并保存答案。
    """
    days = list(map(str, base.calendar()[0]))
    if base_day not in days or days.index(base_day) + 1 >= len(days) or days[days.index(base_day) + 1] != target:
        raise ValueError("ECB_FX_TARGET_NOT_ADJACENT")
    publications = publication_dates()
    if not publications[0] <= base_day <= "2026-12-31":
        raise ValueError("ECB_FX_CALENDAR_UNCOVERED")
    index = bisect_right(publications, base_day) - 1
    if index < 1:
        raise ValueError("ECB_FX_ANCHOR_UNCOVERED")
    return [publications[index - 1], publications[index]]


def features(base_day, target, points):
    """两项输入均为百分点：每美元人民币参考值变化、每欧元美元参考值变化。

    CNY和USD源字段都以一欧元为基数，必须先用CNY/USD换算每美元人民币；
    正值表示一美元可换更多人民币。两项固定限幅正负20，不称为可成交汇率。
    """
    prior, anchor = required(base_day, target)
    if any(day not in points for day in (prior, anchor)):
        raise ValueError("ECB_FX_REQUIRED_DATE_MISSING")
    pu, pc = (float(points[prior][k]) for k in ("usd_per_eur", "cny_per_eur"))
    au, ac = (float(points[anchor][k]) for k in ("usd_per_eur", "cny_per_eur"))
    if not all(math.isfinite(v) and v > 0 for v in (pu, pc, au, ac)):
        raise ValueError("ECB_FX_FEATURE_RATE_INVALID")
    return np.clip([((ac / au) / (pc / pu) - 1) * 100, (au / pu - 1) * 100], -20, 20).tolist()


def history():
    """复用一次历史下载，验证响应、ZIP原始字节和派生字段一致，不重复联网。"""
    folder = base.ROOT / "ecb-fx-feasibility-v1"
    record, response = base.read(folder / "result.json"), base.read(folder / "response.json")
    raw = (folder / "response.zip").read_bytes()
    if record["capture_hash"] != base.digest(response) or hashlib.sha256(raw).hexdigest() != response["sha256"]:
        raise ValueError("ECB_FX_HISTORICAL_RAW_CHANGED")
    points = parse(raw)
    if points != record["rows"]:
        raise ValueError("ECB_FX_HISTORICAL_PARSED_CHANGED")
    return points


def load(target):
    """回读原始ZIP和请求时点，拒绝只修改外层哈希的解析值或已过截止的响应。"""
    folder = root() / target
    value = base.read(folder / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if value["target"] != target or value["source"] != SOURCE or datetime.fromisoformat(value["at"]) >= deadline:
        raise ValueError("ECB_FX_LIVE_INPUT_SCOPE_OR_TIME_INVALID")
    needed = required(value["base"], target)
    slot = value["raw_ref"]["slot"]
    if slot not in SLOTS:
        raise ValueError("ECB_FX_LIVE_SLOT_INVALID")
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
        raise ValueError("ECB_FX_LIVE_RESPONSE_CHANGED_OR_LATE")
    raw = (folder / f"raw/{slot}.zip").read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["body_sha256"] or len(raw) != meta["bytes"]:
        raise ValueError("ECB_FX_LIVE_RAW_CHANGED")
    points = parse(raw)
    if any(day not in points for day in needed) or {d: points[d] for d in needed} != value["rows"]:
        raise ValueError("ECB_FX_LIVE_PARSED_INPUT_CHANGED")
    return value


def capture(at):
    """每目标最多三个固定时槽，每槽一次原始文件请求；已成功则全基金共享，无自动重试。"""
    window = base.window(at)
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("ECB_FX_LIVE_WINDOW_CLOSED")
    target, base_day = window["target_nav_date"], window["base_nav_date"]
    folder = root() / target
    if (folder / "input.json").exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if base.now() >= deadline:
        raise ValueError("ECB_FX_LIVE_DEADLINE_REACHED")
    slot = "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"
    attempt = folder / f"requests/{slot}.json"
    if attempt.exists() or len(list((folder / "requests").glob("*.json"))) >= 3:
        raise ValueError("ECB_FX_LIVE_SLOT_OR_BUDGET_EXHAUSTED")
    needed = required(base_day, target)
    request = {"at": base.now().isoformat(), "url": URL, "target": target, "base": base_day, "required_dates": needed}
    base.save(attempt, request)
    try:
        body, headers = fetch_ecb_fx()
        meta = {
            "received_at": base.now().isoformat(),
            "source": SOURCE,
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "headers": headers,
            "request_hash": base.digest(request),
        }
        raw_path = folder / f"raw/{slot}.zip"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("xb") as file:
            file.write(body)
        base.save(folder / f"raw/{slot}.json", meta)
        if datetime.fromisoformat(meta["received_at"]) >= deadline:
            raise ValueError("ECB_FX_LIVE_RESPONSE_LATE")
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
            raise ValueError("ECB_FX_LIVE_DEADLINE_REACHED")
        base.save(folder / "input.json", value)
        checked = load(target)
        if base.now() >= deadline:
            raise ValueError("ECB_FX_LIVE_READBACK_LATE")
        return checked
    except Exception as exc:
        base.save(folder / f"errors-{slot}.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        raise
