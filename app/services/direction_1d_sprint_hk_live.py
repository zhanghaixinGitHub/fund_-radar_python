"""按目标日期独立保存港股输入；只有真实提前取得的响应能进入未来预测。

两只指数共用全基金输入，每个目标日最多三个时槽、每槽每指数一次请求。
任一请求失败保留记录，不能在同槽反复请求；下一时槽仍受总预算和08:30约束。
"""

import hashlib
import json
import time as sleep_time
from datetime import date, datetime, time, timedelta

from app.integrations.tushare_sprint_hk import fetch_hk
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_hk_data as data
from app.services import direction_1d_sprint_overnight as overnight


def root():
    return base.ROOT / "round-17" / "hk-inputs"


def load(target):
    """重读响应与解析值并核对日期和截止时刻，不接受只重写外层摘要的输入。"""
    value = base.read(root() / target / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if value["target"] != target or datetime.fromisoformat(value["at"]) >= deadline:
        raise ValueError("HK_LIVE_INPUT_LATE_OR_WRONG_TARGET")
    days, _ = base.calendar()
    day = date.fromisoformat(target)
    if day not in days or value["base"] != str(days[days.index(day) - 1]):
        raise ValueError("HK_LIVE_BASE_DATE_CHANGED")
    for code in data.INDICES:
        ref = value["raw_refs"][code]
        path = root() / target / "raw" / f"{ref['slot']}-{code}.json"
        raw = base.read(path)
        if base.digest(raw) != ref["hash"] or datetime.fromisoformat(raw["received_at"]) >= deadline:
            raise ValueError("HK_LIVE_RESPONSE_CHANGED_OR_LATE")
        if raw["end"] != value["base"] or raw["source_id"] != value["source_id"]:
            raise ValueError("HK_LIVE_RESPONSE_SCOPE_CHANGED")
        parsed = data.validate(code, json.dumps(raw["response"]).encode(), raw["start"], raw["end"])
        if parsed != value["rows"][code]:
            raise ValueError("HK_LIVE_PARSED_ROWS_CHANGED")
    return value


def capture(at):
    """窗口外不请求；请求结束后的实际接收时间必须仍在预测截止前。"""
    window = base.window(at)
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("HK_LIVE_WINDOW_CLOSED")
    target, base_day = window["target_nav_date"], window["base_nav_date"]
    folder = root() / target
    if (folder / "input.json").exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    slot = "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"
    start = date.fromisoformat(base_day) - timedelta(days=35)
    source = overnight.source()
    combined, refs, errors = {}, {}, {}
    for code in data.INDICES:
        raw_path, attempt = folder / f"raw/{slot}-{code}.json", folder / f"requests/{slot}-{code}.json"
        if base.now() >= deadline:
            raise ValueError("HK_LIVE_DEADLINE_REACHED")
        try:
            if not raw_path.exists():
                if attempt.exists() or len(list((folder / "requests").glob("*.json"))) >= 6:
                    raise ValueError("HK_LIVE_SLOT_ALREADY_ATTEMPTED_OR_BUDGET_EXHAUSTED")
                base.save(attempt, {"at": base.now().isoformat(), "code": code, "start": str(start), "end": base_day})
                try:
                    body = fetch_hk(code, start, date.fromisoformat(base_day))
                    base.save(
                        raw_path,
                        {
                            "received_at": base.now().isoformat(),
                            "source_id": str(source["source_id"]),
                            "start": str(start),
                            "end": base_day,
                            "response": json.loads(body),
                            "body_sha256": hashlib.sha256(body).hexdigest(),
                        },
                    )
                finally:
                    sleep_time.sleep(max(7, 60 / source["rate_limit_per_minute"]))
            raw = base.read(raw_path)
            if datetime.fromisoformat(raw["received_at"]) >= deadline:
                raise ValueError("HK_LIVE_RESPONSE_LATE")
            combined[code] = data.validate(code, json.dumps(raw["response"]).encode(), str(start), base_day)
            refs[code] = {"slot": slot, "hash": base.digest(raw)}
        except Exception as exc:
            errors[code] = base.error_code(exc)
    if errors:
        base.save(folder / f"errors-{slot}.json", {"at": base.now().isoformat(), "errors": errors}, replace=True)
        raise ValueError("HK_LIVE_INPUT_INCOMPLETE")
    if base.now() >= deadline:
        raise ValueError("HK_LIVE_DEADLINE_REACHED")
    value = {
        "at": base.now().isoformat(),
        "target": target,
        "base": base_day,
        "source_id": str(source["source_id"]),
        "rows": combined,
        "raw_refs": refs,
    }
    base.save(folder / "input.json", value)
    if base.now() >= deadline:
        raise ValueError("HK_LIVE_READBACK_LATE")
    return load(target)
