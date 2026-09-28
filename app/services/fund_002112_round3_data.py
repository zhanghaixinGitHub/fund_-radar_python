"""002112 第三轮独立资料核查；入口绑定截至 2024 年的旧冻结包，绝不追随 ready。

本模块只有本地文件读操作和纯计算。输出由第三轮编排器写入自己的目录；不导入
旧训练器、采集器或数据库。日期级公开时间与原有保守的标签成熟时间分别保留。
"""

import hashlib
import json
import math
from collections import Counter, OrderedDict
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from app.services.direction_1d_protocol import ZONE, calendar, digest, features

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / ".local-runs/fund-exposure-002112"
RUN_ROOT = ROOT / "round3-runs"
OLD = ROOT / "training-runs/002112-b9fcc6eb85cce6e19466db3a"
SECOND = ROOT / "optimization-runs/round-20260925-v1"
DATA_HASH = "318d8b1fd2da152a5e8b72b9f2e0c2f020054ce7d1890228d5ef4be5b2d63d00"
SOURCE_HASH = "98fb1c51f2582687a09ac0c404d514c2ef74520e4cb63a9a9f89a300ef150611"
FUND = "002112"
COHORT = (FUND, "002170", "004237", "004605", "005187", "005312", "006038", "007509", "008960", "017493", "160323")
CLASSES = ("DOWN", "FLAT", "UP")
TIE_ORDER = ("FLAT", "UP", "DOWN")
FOLDS = {"2023Q2": ("2023-04-01", 50), "2023Q3": ("2023-07-01", 51), "2023Q4": ("2023-10-01", 60)}
FEATURES = (
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "max_drawdown_60d",
    "relative_position_60d",
    "consecutive_decline_days",
    "disclosed_nav_return_1d",
    "disclosed_nav_return_5d",
    "up_stock_nav_weight",
    "nav_weighted_amount_vs_previous20",
    "nav_weight_concentration",
    "disclosed_nav_weight",
    "reported_stock_nav_weight",
    "report_age_days",
    "full_disclosure",
    "csi300_return_1d",
    "csi300_return_5d",
    "csi500_return_1d",
    "csi500_return_5d",
)


def now():
    return datetime.now(ZONE).isoformat()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    """兼容旧封装并验证内容哈希；不会吞掉文件损坏或缺失。"""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and set(value) == {"hash", "payload"}:
        if digest(value["payload"]) != value["hash"]:
            raise ValueError("CONTENT_HASH_CHANGED")
        return value["payload"]
    return value


def write_once(path, value):
    """排他创建；已有产物只允许同内容复用，不覆盖阶段决定或账本。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError("IMMUTABLE_ARTIFACT_CHANGED:" + path.name)
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            {"hash": digest(value), "payload": value}, stream, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        stream.flush()
        import os

        os.fsync(stream.fileno())


def scope_day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value or value > "2024-12-31":
        raise ValueError("DATE_OUTSIDE_ALLOWED_FROZEN_SCOPE")
    return value


def direction(before, after):
    a, b = Decimal(str(before)), Decimal(str(after))
    if not a.is_finite() or not b.is_finite() or min(a, b) <= 0:
        raise ValueError("INVALID_NAV")
    return "UP" if b > a else "DOWN" if b < a else "FLAT"


def known_before(row, target):
    """后续修订版本以版本日期为准；缺公开日期绝不视为已知。"""
    dates = [row.get("ann_date")]
    dates.extend(row[k][:10] for k in ("revised_at", "version_publication_date") if row.get(k))
    return bool(dates[0]) and all(d < target for d in dates)


def nav_window(mapping, days, target):
    """找到 U 前最近的连续 61 日窗口；S 可早于 T，标签始终比较 U/T。"""
    i = days.index(target)
    for end in range(i - 1, 59, -1):
        needed = days[end - 60 : end + 1]
        if any(d not in mapping or not known_before(mapping[d], target) for d in needed):
            continue
        try:
            values = features([mapping[d]["nav"] for d in needed])
        except ValueError:
            continue
        return {
            "x": values,
            "S": days[end],
            "T": days[i - 1],
            "U": target,
            "lag_sessions": i - 1 - end,
            "nav_dates": needed,
            "nav_hashes": [mapping[d]["source_hash"] for d in needed],
            "last_publication": max(mapping[d]["ann_date"] for d in needed),
        }
    raise ValueError("NO_PRIOR_PUBLIC_CONTIGUOUS_NAV_WINDOW")


def public_day(report):
    """目录旧 available_at 明确由公开日加一天生成，只还原日期，不伪造 U 日时间。"""
    if report.get("published_date"):
        dates = [report["published_date"], report.get("source_publication_date", report["published_date"])]
        dates.extend(report[k][:10] for k in ("revised_at", "version_publication_date") if report.get(k))
        return max(dates)
    if report.get("available_at"):
        return (datetime.fromisoformat(report["available_at"]).date() - timedelta(days=1)).isoformat()
    raise ValueError("REPORT_PUBLICATION_UNKNOWN")


def report_key(report):
    full = report.get("full_stock_disclosure", report["report_type"] in ("ANNUAL", "HALF"))
    return report["report_end"], public_day(report), bool(full)


def select_report(reports, catalog, target):
    available = [r for r in reports if public_day(r) < target]
    if not available:
        raise ValueError("NO_PRIOR_PUBLIC_REPORT")
    key = max(map(report_key, available))
    choices = [r for r in available if report_key(r) == key]
    if len({r["raw"]["sha256"] for r in choices}) != 1:
        raise ValueError("REPORT_VERSION_CONFLICT")
    latest = max((report_key(r) for r in catalog if public_day(r) < target), default=None)
    if latest and latest > key:
        raise ValueError("LATEST_DISCLOSURE_MISSING_NO_FALLBACK")
    return choices[0]


def exposure(report, quotes, indices, needed):
    """与原 9+4 项计算公式一致，日期选择独立实现，不调用旧 calculate/initialize。"""
    age = (date.fromisoformat(needed[-1]) - date.fromisoformat(report["report_end"])).days
    total = float(report["disclosed_nav_pct"]) / 100
    if not 0 <= age <= 210 or not math.isfinite(total) or total <= 0:
        raise ValueError("REPORT_AGE_OR_WEIGHT_INVALID")
    weighted = weighted5 = up = activity = hhi = covered = 0.0
    seen = set()
    for h in report["holdings"]:
        code, weight = h["stock_code"], float(h["nav_weight_pct"]) / 100
        if code in seen or not math.isfinite(weight) or weight < 0:
            raise ValueError("HOLDING_IDENTITY_OR_WEIGHT_INVALID")
        seen.add(code)
        if weight == 0:  # 原文明确 0.00%；不由缺值推断为零。
            continue
        rows = []
        for day in needed:
            row = quotes(day).get("rows", {}).get(code)
            if row is None or any(
                type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in ("pct_chg", "amount")
            ):
                raise ValueError(f"POSITIVE_HOLDING_QUOTE_MISSING:{code}:{day}")
            rows.append(row)
        mean = sum(r["amount"] for r in rows[:-1]) / 20
        if mean <= 0:
            raise ValueError("AMOUNT_BASE_NOT_POSITIVE")
        covered += weight
        weighted += weight * rows[-1]["pct_chg"] / 100
        weighted5 += weight * (math.prod(1 + r["pct_chg"] / 100 for r in rows[-5:]) - 1)
        up += weight if rows[-1]["pct_chg"] > 0 else 0
        activity += weight * (rows[-1]["amount"] / mean - 1)
        hhi += weight**2
    if abs(covered / total - 1) > 1e-9:
        raise ValueError("DISCLOSED_WEIGHT_COVERAGE_INCOMPLETE")
    x = [
        weighted,
        weighted5,
        up,
        activity,
        hhi,
        total,
        float(report["stock_nav_pct"]) / 100,
        float(age),
        float(report["full_stock_disclosure"]),
    ]
    for code in ("000300.SH", "000905.SH"):
        try:
            vals = [indices[code]["rows"][day]["pct_chg"] / 100 for day in needed[-5:]]
        except (KeyError, TypeError):
            raise ValueError("INDEX_WINDOW_MISSING") from None
        x.extend([vals[-1], math.prod(1 + v for v in vals) - 1])
    if len(x) != 13 or not all(math.isfinite(v) for v in x):
        raise ValueError("EXPOSURE_NONFINITE")
    return x


def weights(rows):
    count = Counter(r["family"] for r in rows)
    return [len(rows) / (len(count) * count[r["family"]]) for r in rows]


def summary(rows):
    return {
        "rows": len(rows),
        "dates": len({r["target"] for r in rows}),
        "class_rows": {k: sum(r["actual_direction"] == k for r in rows) for k in CLASSES},
        "class_dates": {k: len({r["target"] for r in rows if r["actual_direction"] == k}) for k in CLASSES},
    }


def gate(rows, families):
    own = [r for r in rows if r["fund_code"] == FUND]
    pooled, target = summary(rows), summary(own)
    checks = {
        "pooled_dates_252": pooled["dates"] >= 252,
        "own_dates_252": target["dates"] >= 252,
        "each_class_30_dates": min(pooled["class_dates"].values()) >= 30,
        "own_three_classes": min(target["class_rows"].values()) > 0,
        "all_11_families": set(r["family"] for r in rows) == set(families.values())
        and len(set(families.values())) == 11,
        "no_duplicate_family_date": len({(r["family"], r["target"]) for r in rows}) == len(rows),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pooled": pooled,
        "own": target,
        "fund_rows": dict(Counter(r["fund_code"] for r in rows)),
    }


def eligible_at(row, start):
    return (
        row["target"] < start
        and row["label_publication"] < start
        and datetime.fromisoformat(row["mature_at"]) < datetime.fromisoformat(start + "T00:00:00+08:00")
    )


class Sources:
    """本轮只读来源清单。解析前先限定文件身份，原文只在已绑定回执指定路径打开。"""

    def __init__(self):
        self.files = {}
        self.receipts = {}

    def load(self, path, expected=None):
        path = Path(path).resolve()
        value = read(path)
        if expected and digest(value) != expected:
            raise ValueError("FROZEN_SOURCE_IDENTITY_CHANGED")
        self.files[str(path)] = file_hash(path)
        return value

    def raw(self, receipt):
        path = Path(receipt["raw_path"]) if receipt.get("raw_path") else ROOT / receipt["file"]
        path = path.resolve()
        if not path.is_relative_to(PROJECT / ".local-runs"):
            raise ValueError("RAW_PATH_OUTSIDE_LOCAL_RESEARCH")
        if receipt.get("expires_at") and datetime.fromisoformat(receipt["expires_at"]) <= datetime.now(ZONE):
            raise ValueError("SOURCE_EXPIRED")
        if str(path) not in self.files:
            if file_hash(path) != receipt["sha256"]:
                raise ValueError("RAW_SOURCE_CHANGED")
            self.files[str(path)] = receipt["sha256"]
        elif self.files[str(path)] != receipt["sha256"]:
            raise ValueError("RAW_SOURCE_VERSION_CONFLICT")
        self.receipts[digest(receipt)] = receipt
        return path


def construct(progress=print):
    """先完整检查原考试清单；任何缺日都会阻止整个真实实验，而非缩小分母。"""
    sources = Sources()
    spec = sources.load(OLD / "protocol.json")
    if spec["reference"]["sha256"] != DATA_HASH or spec["source_hash"] != SOURCE_HASH:
        raise ValueError("BASELINE_BINDING_CHANGED")
    old = sources.load(OLD / "dataset.json", DATA_HASH)
    if old["source_file"] != f"sources/{SOURCE_HASH}.json" or old["source_hash"] != SOURCE_HASH:
        raise ValueError("BASELINE_SOURCE_BINDING_CHANGED")
    snap = sources.load(ROOT / "training-ready" / old["source_file"], SOURCE_HASH)
    oldfolds = sources.load(SECOND / "folds.json")
    if set(snap["funds"]) != set(COHORT):
        raise ValueError("COHORT_CHANGED")
    families = {k: v["family"] for k, v in snap["funds"].items()}
    if len(set(families.values())) != 11:
        raise ValueError("DUPLICATE_FAMILY")
    indices = snap["indices"]
    days = sorted(indices["000300.SH"]["rows"])
    if days != sorted(indices["000905.SH"]["rows"]):
        raise ValueError("INDEX_CALENDAR_DISAGREEMENT")
    for day in days:
        scope_day(day)
    # 两指数共同有行情仍不足以定义休市日；再和旧冻结协议绑定的官方研究日历核对。
    calendar_path = PROJECT / "app/data/calendars/cn_a_share_2015_2020_research_v1.json"
    if file_hash(calendar_path) != spec["code"][calendar_path.relative_to(PROJECT).as_posix()]:
        raise ValueError("OFFICIAL_CALENDAR_CHANGED")
    older_calendar = sources.load(calendar_path)
    first, last = (date.fromisoformat(older_calendar[k]) for k in ("coverage_start", "coverage_end"))
    closed = set()
    for year in older_calendar["years"]:
        for left, right in year["closed_ranges"]:
            start, end = date.fromisoformat(left), date.fromisoformat(right)
            closed.update(start + timedelta(days=i) for i in range((end - start).days + 1))
    older_days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    expected_days = [str(day) for day in older_days if day.weekday() < 5 and day not in closed]
    expected_days.extend(str(day) for day in calendar()[0] if day <= date(2024, 12, 31))
    if [day for day in days if day >= str(first)] != expected_days:
        raise ValueError("OFFICIAL_CALENDAR_SOURCE_DISAGREEMENT")
    positions = {d: i for i, d in enumerate(days)}
    # 原文市场行情无基金净值标签。忽略 2024 后才开始的回执；混合年份行情只用于
    # 复验冻结包内的 <=2024 行，绝不访问 nav_daily、研究全目录或封存标签文件。
    for code, index in indices.items():
        reconstructed = {}
        for receipt in index["receipts"]:
            if receipt["params"].get("start_date", "") > "20241231":
                continue
            if receipt.get("api") != "index_daily":
                raise ValueError("INDEX_API_SCOPE")
            raw = json.loads(sources.raw(receipt).read_text(encoding="utf-8"))["data"]
            for item in raw["items"]:
                row = dict(zip(raw["fields"], item, strict=True))
                if row["ts_code"] != code:
                    raise ValueError("INDEX_CODE_CHANGED")
                day = datetime.strptime(row["trade_date"], "%Y%m%d").date().isoformat()
                if day <= "2024-12-31":
                    reconstructed[day] = {k: v for k, v in row.items() if k not in ("ts_code", "trade_date")}
        if reconstructed != index["rows"]:
            raise ValueError("INDEX_RAW_REPLAY_MISMATCH")
    oldquotes = snap["older_quotes"]
    reconstructed = {}
    for code, receipt in oldquotes["receipts"].items():
        if receipt.get("api") != "daily" or receipt["params"]["end_date"] > "20241231":
            raise ValueError("OLD_QUOTE_SCOPE")
        raw = json.loads(sources.raw(receipt).read_text(encoding="utf-8"))["data"]
        for item in raw["items"]:
            row = dict(zip(raw["fields"], item, strict=True))
            day = datetime.strptime(row.pop("trade_date"), "%Y%m%d").date().isoformat()
            if row.pop("ts_code") != code:
                raise ValueError("QUOTE_CODE_CHANGED")
            reconstructed.setdefault(day, {})[code] = row
    if reconstructed != {d: v["rows"] for d, v in oldquotes["days"].items()}:
        raise ValueError("OLD_QUOTE_RAW_REPLAY_MISMATCH")
    cache = OrderedDict()

    def quotes(day):
        scope_day(day)
        if day < "2021-01-01":
            return oldquotes["days"].get(day, {})
        if day not in cache:
            path = ROOT / "stock-days" / (day + ".json")
            if not path.exists():
                return {}
            value = sources.load(path)
            raw = json.loads(sources.raw(value["receipt"]).read_text(encoding="utf-8"))["data"]
            rows = {}
            for item in raw["items"]:
                r = dict(zip(raw["fields"], item, strict=True))
                if r.pop("trade_date") != day.replace("-", ""):
                    raise ValueError("QUOTE_DATE_CHANGED")
                code = r.pop("ts_code")
                if code in rows:
                    raise ValueError("QUOTE_DUPLICATE")
                rows[code] = r
            # 聚合文件仅含当时所需股票；旧训练器使用同一回执的完整 SH/SZ 原文。
            # 核对已有子集后复用原文，不把子集之外的股票误判为资料缺失。
            if any(rows.get(code) != r for code, r in value["rows"].items()):
                raise ValueError("QUOTE_RAW_REPLAY_MISMATCH")
            cache[day] = {
                "rows": {k: v for k, v in rows.items() if k.endswith((".SH", ".SZ"))},
                "receipt": value["receipt"],
            }
        cache.move_to_end(day)
        while len(cache) > 64:
            cache.popitem(last=False)
        return cache[day]

    navs = {}
    for code, fund in snap["funds"].items():
        rows = fund["nav"]["rows"]
        for r in rows:
            scope_day(r["date"])
        navs[code] = {r["date"]: r for r in rows}
        if len(navs[code]) != len(rows):
            raise ValueError("DUPLICATE_NAV_DATE")
        for report in fund["reports"]:
            sources.raw(report["raw"])
    built, excluded, formula_equal = {"train": [], "development": []}, [], 0
    for split in built:
        for n, r in enumerate(old[split]):
            u, t, code = scope_day(r["target"]), scope_day(r["base"]), r["fund_code"]
            if code not in COHORT or r["family"] != families[code]:
                raise ValueError("ROW_COHORT_CHANGED")
            if (split == "train" and u > "2023-12-31") or (
                split == "development" and (code != FUND or u[:4] != "2024")
            ):
                raise ValueError("ROW_SCOPE_CHANGED")
            if positions[u] == 0 or days[positions[u] - 1] != t:
                raise ValueError("TARGET_BASE_NOT_ADJACENT")
            if n % 500 == 0:
                progress(f"{split}: {n}/{len(old[split])}，真实拟合 0")
            mapping, fund = navs[code], snap["funds"][code]
            # 标签严格与冻结答案和来源双重比较；仅用于标签字段，从不拼入 x。
            label = direction(mapping[t]["nav"], mapping[u]["nav"])
            if label != r["actual_direction"] or direction(r["base_unit_nav"], r["target_unit_nav"]) != label:
                raise ValueError("EXACT_LABEL_CHANGED")
            try:
                window = nav_window(mapping, days, u)
                report = select_report(fund["reports"], fund["catalog"], u)
                needed = days[positions[t] - 20 : positions[t] + 1]
                x13 = exposure(report, quotes, indices, needed)
                if report["raw"]["sha256"] == r["exposure"]["report_raw_sha256"]:
                    if x13 != r["x"][7:]:
                        raise ValueError("EXPOSURE_FORMULA_CHANGED")
                    formula_equal += 1
                built[split].append(
                    {
                        "fund_code": code,
                        "family": r["family"],
                        "base": t,
                        "target": u,
                        "actual_direction": label,
                        "base_unit_nav": r["base_unit_nav"],
                        "target_unit_nav": r["target_unit_nav"],
                        "mature_at": r["mature_at"],
                        "label_publication": mapping[u]["ann_date"],
                        "nav": {k: v for k, v in window.items() if k != "x"},
                        "report_sha256": report["raw"]["sha256"],
                        "report_publication": public_day(report),
                        "report_end": report["report_end"],
                        "quote_dates": needed,
                        "x": window["x"] + x13,
                        "old_row_hash": digest(r),
                    }
                )
            except ValueError as exc:
                if str(exc).startswith(("EXPOSURE_FORMULA", "QUOTE_", "RAW_", "SOURCE_", "CONTENT_HASH")):
                    raise
                excluded.append({"split": split, "fund_code": code, "target": u, "reason": str(exc)})
    folds = []
    for name, (start, count) in (*FOLDS.items(), ("FULL", ("2024-01-01", 230))):
        original_dates = (
            [r["target"] for r in next(f for f in oldfolds if f["name"] == name)["exam_rows"]]
            if name != "FULL"
            else [r["target"] for r in old["development"]]
        )
        if len(original_dates) != count or len(set(original_dates)) != count:
            raise ValueError("ORIGINAL_EXAM_IDENTITY_CHANGED")
        training = [r for r in built["train"] if eligible_at(r, start)]
        exams = [
            r
            for r in (built["train"] if name != "FULL" else built["development"])
            if r["fund_code"] == FUND and r["target"] in original_dates
        ]
        absent = sorted(set(original_dates) - {r["target"] for r in exams})
        folds.append(
            {
                "name": name,
                "start": start,
                "expected_dates": original_dates,
                "train_ids": [[r["fund_code"], r["target"]] for r in training],
                "train_hash": digest(training),
                "weights": weights(training),
                "gate": gate(training, families),
                "missing_dates": absent,
                "exam_hash": digest(exams),
                "exam_count": len(exams),
            }
        )
    eligibility = {
        "passed": all(f["gate"]["passed"] and not f["missing_dates"] for f in folds),
        "folds": {
            f["name"]: {"gate": f["gate"], "missing_dates": f["missing_dates"], "exam_count": f["exam_count"]}
            for f in folds
        },
        "excluded": excluded,
        "formula_equal_rows": formula_equal,
        "nav_lag_sessions": {s: dict(Counter(r["nav"]["lag_sessions"] for r in rows)) for s, rows in built.items()},
        "real_fits": 0,
    }
    return (
        built,
        folds,
        eligibility,
        {
            "files": sources.files,
            "receipts": list(sources.receipts.values()),
            "source_hash": SOURCE_HASH,
            "old_dataset_hash": DATA_HASH,
            "label_read_scope": "fixed frozen NAV and labels through 2024 only",
        },
    )
