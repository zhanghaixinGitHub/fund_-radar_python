"""复用第26轮ECB原始ZIP，增加固定六种货币的每美元参考值变化，不新增请求。"""

import csv
import hashlib
import io
import math
import zipfile
from datetime import date, datetime, time
from functools import lru_cache

import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_ecb_data as core

CURRENCIES = ("JPY", "CHF", "AUD", "CAD", "GBP", "NOK")
SOURCE = core.SOURCE
URL = core.URL
MAX_BYTES = core.MAX_BYTES


def root():
    return base.ROOT / "fx-basket-feasibility-v1"


@lru_cache(maxsize=4)
def expanded_rows(raw):
    """先由既有校验器核对ZIP、日期与核心字段，再提取固定六列，缓存不可变元组。"""
    original = core.parse(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        content = archive.read("eurofxref-hist.csv")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), skipinitialspace=True)
    if any(reader.fieldnames.count(currency) != 1 for currency in CURRENCIES):
        raise ValueError("FX_BASKET_CURRENCY_FIELDS_INVALID")
    result = []
    for item in reader:
        day = item["Date"]
        if day not in original:
            continue
        values = tuple(float(item[currency]) for currency in CURRENCIES)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("FX_BASKET_CURRENCY_VALUE_INVALID")
        row = original[day]
        result.append((day, row["usd_per_eur"], row["cny_per_eur"], *values))
    if len(result) != len(original):
        raise ValueError("FX_BASKET_CORE_DATE_SET_CHANGED")
    return tuple(result)


def parse(raw):
    """所有源字段都以一欧元为基数；返回独立字典，不允许修改缓存影响其他基金。"""
    return {
        day: {"usd_per_eur": usd, "cny_per_eur": cny}
        | {f"{code.lower()}_per_eur": value for code, value in zip(CURRENCIES, values, strict=True)}
        for day, usd, cny, *values in expanded_rows(raw)
    }


def features(base_day, target, points):
    """保留原人民币和美元两项，按JPY/CHF/AUD/CAD/GBP/NOK固定顺序追加六项。

    每项先以外币每欧元值除美元每欧元值，再计算前后应发布日期的百分比变化；
    正值表示每美元可换更多该外币。全部固定限幅正负20个百分点，不扫描币种子集。
    """
    output = core.features(base_day, target, points)
    prior, anchor = core.required(base_day, target)
    for currency in CURRENCIES:
        key = f"{currency.lower()}_per_eur"
        p, a = float(points[prior][key]), float(points[anchor][key])
        if not all(math.isfinite(v) and v > 0 for v in (p, a)):
            raise ValueError("FX_BASKET_FEATURE_VALUE_INVALID")
        before = p / points[prior]["usd_per_eur"]
        after = a / points[anchor]["usd_per_eur"]
        output.append(float(np.clip((after / before - 1) * 100, -20, 20)))
    return output


def history():
    """从第26轮同一个历史ZIP派生新字段，不重新下载、不改变任何已冻结响应。"""
    original = core.history()
    folder = base.ROOT / "ecb-fx-feasibility-v1"
    response = base.read(folder / "response.json")
    raw = (folder / "response.zip").read_bytes()
    if hashlib.sha256(raw).hexdigest() != response["sha256"]:
        raise ValueError("FX_BASKET_HISTORICAL_RAW_CHANGED")
    points = parse(raw)
    if {day: {key: row[key] for key in ("usd_per_eur", "cny_per_eur")} for day, row in points.items()} != original:
        raise ValueError("FX_BASKET_HISTORICAL_CORE_CHANGED")
    if (root() / "history.json").exists():
        snapshot = base.read(root() / "history.json")
        if snapshot["raw_sha256"] != response["sha256"] or snapshot["rows"] != points:
            raise ValueError("FX_BASKET_HISTORICAL_SNAPSHOT_CHANGED")
    return points


def expand_input(value):
    """新字段只能来自旧分支已验证的同一原始响应；输入at仍表示旧响应派生记录的时间。

    新预测自身的保存、回读时间另行严格核验，不能把此处复用的旧at当作新预测时间。
    """
    folder = core.root() / value["target"]
    slot = value["raw_ref"]["slot"]
    metadata = base.read(folder / f"raw/{slot}.json")
    raw = (folder / f"raw/{slot}.zip").read_bytes()
    if value["raw_ref"]["hash"] != base.digest(metadata) or hashlib.sha256(raw).hexdigest() != metadata["body_sha256"]:
        raise ValueError("FX_BASKET_SHARED_RAW_CHANGED")
    all_rows = parse(raw)
    required = core.required(value["base"], value["target"])
    rows = {day: all_rows[day] for day in required}
    features(value["base"], value["target"], rows)
    return value | {"rows": rows, "core_input_hash": base.digest(value), "feature_currencies": list(CURRENCIES)}


def load(target):
    """既有load先核验原始接收、请求时槽、截止时间，再核对新币种字段。"""
    return expand_input(core.load(target))


def capture(at):
    """共用原三个时槽预算；若第26轮已有成功文件，追加特征不再次联网。"""
    original = core.capture(at)
    checked = load(original["target"])
    deadline = datetime.combine(date.fromisoformat(original["target"]), time(8, 30), base.ZONE)
    if base.now() >= deadline:
        raise ValueError("FX_BASKET_EXTRACTION_LATE")
    return checked
