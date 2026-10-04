"""002112 晚间一日研究：完整可用历史、三个分支和固定 55/30/15 合成。

本脚本只写独立研究目录。历史公开日及收盘行情属于重建证据，不冒充当时首见
存档；不注册现用模型，不调用付费模型，不把人工判断用作训练答案。
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.metadata
import json
import math
import subprocess
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

PY = Path(__file__).resolve().parents[1]
RESEARCH = PY / ".local-runs/fund-exposure-002112"
SEM = RESEARCH / "semantic-holdings-optimization/20261001-v1"
SIGNAL = RESEARCH / "signal-optimization/20261001-v1"
ROOT = RESEARCH / "evening-news-fusion/20261003-v1"
ZONE = ZoneInfo("Asia/Shanghai")
CLASSES = ["DOWN", "FLAT", "UP"]
WEIGHTS = {"information": 0.55, "market": 0.30, "history": 0.15}
RECIPE = {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 10, "random_state": 0, "n_jobs": 2}
MAX_LABEL = "2026-09-30"
JOINT_START = "2024-01-02"
INVENTORY = (
    Path(r"C:\Users\a\.codex\visualizations\2026\10\02")
    / "01a0fc78-9a4f-7df0-90e5-703ebd5bb196/002112-asof/inventory.json"
)
NUM_FIELDS = {"revenue_yoy": 2, "parent_profit_yoy": 3, "order_ratio": 4, "buyback_ratio": 6}
FLAG_FIELDS = {
    "guidance_up": 0,
    "guidance_down": 1,
    "order_cancel": 5,
    "risk_new": 7,
    "risk_resolved": 8,
    "forecast": 9,
    "realized": 10,
    "material": 11,
}
INFO_NAMES = [
    "guidance_up_weight",
    "guidance_down_weight",
    "revenue_yoy",
    "parent_profit_yoy",
    "order_to_revenue",
    "order_cancel_weight",
    "buyback_execution_ratio",
    "risk_new_weight",
    "risk_resolved_weight",
    "forecast_weight",
    "realized_weight",
    "material_weight",
    "qualified_count",
    "direct_weight",
    "processed_count",
    "revenue_coverage",
    "profit_coverage",
    "order_coverage",
    "buyback_coverage",
    "report_age",
]
PUBLIC_ENUMS = {
    "direction": ["BENEFIT", "PRESSURE", "MIXED", "UNKNOWN"],
    "stage": ["REALIZED", "FORECAST", "PROPOSAL", "IMPLEMENTATION", "CANCELLED", "CORRECTION", "UNKNOWN"],
    "kind": ["EARNINGS", "ORDER", "BUYBACK", "FINANCING", "DIVIDEND", "GOVERNANCE", "POLICY", "BUSINESS", "OTHER"],
    "channel": ["PROFIT", "DEMAND", "COST", "FINANCING", "REGULATORY", "NONE"],
}
for window in [1, 5, 20]:
    for source in ["news", "policy"]:
        INFO_NAMES += [f"{source}_{window}_observed_count", f"{source}_{window}_direct_link_weight"]
        INFO_NAMES += [f"{source}_{window}_{field}_{v}" for field, values in PUBLIC_ENUMS.items() for v in values]
HISTORY_NAMES = [
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "return_60",
    "vol_5",
    "vol_20",
    "vol_60",
    "drawdown_60",
    "relative_position_60",
    "decline_streak",
    "nav_lag_sessions",
    "return_1_vs_5",
    "return_5_vs_20",
    "announced_next_dividend",
]
MARKET_NAMES = [
    "holding_return_1",
    "holding_return_5",
    "holding_up_weight",
    "holding_activity",
    "coverage_1",
    "coverage_5",
    "coverage_activity",
    "disclosed_weight",
    "stock_weight",
    "holding_concentration",
    "holding_age",
    "full_disclosure",
]
for code in ["000300.SH", "000905.SH"]:
    MARKET_NAMES += [f"{code}_{v}" for v in ["return_1", "return_5", "return_20", "vol_20", "amount_ratio"]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path: Path) -> list:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(s) for s in handle if s.strip()]


def write(path: Path, value) -> None:
    """新产物只允许创建，不静默覆盖中断记录或旧实验。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)


def write_lines(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def now() -> str:
    return datetime.now(ZONE).isoformat()


def moment(value: str) -> datetime:
    point = datetime.fromisoformat(value)
    if point.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED")
    return point.astimezone(ZONE)


def next_midnight(day: str) -> str:
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat() + "T00:00:00+08:00"


def available_nav(row: dict) -> str:
    """已明确的时间取严格约束；只有日期时等整日结束。近期本地实际收到时间也约束。

    早年批量导入时间不能伪称历史公开时间，改用公告日期重建并在产物中明确此限制。
    """
    points = [next_midnight(row["ann_date"])]
    if row.get("source_published_at"):
        points.append(moment(row["source_published_at"]).isoformat())
    if row["nav_date"] >= "2026-08-27":
        points.extend(moment(row[k]).isoformat() for k in ["created_at", "updated_at"] if row.get(k))
    return max(points)


def effective_index(sessions: list[str], stamp: str) -> int:
    """按交易日23点定位首次可用日，周末和23点后的消息进入下一个交易日晚间。"""
    point = moment(stamp)
    i = bisect.bisect_left(sessions, point.date().isoformat())
    if i < len(sessions) and point > moment(sessions[i] + "T23:00:00+08:00"):
        i += 1
    return i


def choose_report(reports: list, cutoff: str):
    eligible = [r for r in reports if r["available_at"] <= cutoff and r["fund_code"] == "002112"]
    return max(eligible, key=lambda r: (r["report_end"], r["available_at"])) if eligible else None


def holding_weights(report) -> dict:
    return {h["stock_code"]: float(h["nav_weight_pct"]) / 100 for h in report["holdings"]} if report else {}


def git_snapshot() -> dict:
    """只读记录三个工作区既有改动及字节摘要，不暂存、不还原、不提交。"""
    result = {}
    for repo in [PY, Path(r"C:\WebStormProject\workSpace05"), Path(r"C:\ideaProject\workSpace12")]:
        raw = subprocess.check_output(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=repo)
        entries = raw.decode("utf-8").split("\0")
        files = {}
        for entry in entries:
            if len(entry) < 4:
                continue
            p = repo / entry[3:]
            if p.is_file():
                files[str(p)] = sha(p)
        result[str(repo)] = {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip(),
            "status": raw.decode("utf-8"),
            "files": files,
        }
    return result


def load_calendar() -> tuple[list[str], list[Path]]:
    paths = [
        PY / "app/data/calendars" / n
        for n in ["cn_a_share_2015_2020_research_v1.json", "cn_a_share_2021_2025_v1.json", "cn_a_share_2026_v1.json"]
    ]
    sessions = []
    for path in paths:
        data = read(path)
        closed = {
            date.fromisoformat(a) + timedelta(days=i)
            for year in data["years"]
            for a, b in year["closed_ranges"]
            for i in range((date.fromisoformat(b) - date.fromisoformat(a)).days + 1)
        }
        start, end = map(date.fromisoformat, [data["coverage_start"], data["coverage_end"]])
        sessions += [
            (start + timedelta(days=i)).isoformat()
            for i in range((end - start).days + 1)
            if (start + timedelta(days=i)).weekday() < 5 and start + timedelta(days=i) not in closed
        ]
    assert sessions == sorted(set(sessions))
    return sessions, paths


def load_sources(root: Path) -> dict:
    """复用已校验正文和原始行情；只读查询本基金净值，保存本次冻结副本。"""
    from app.db.session import get_nav_preview_engine
    from sqlalchemy import text

    bundle_path = RESEARCH / "existing-data-experiment/20260930-v1/snapshot/normalized-inputs.json"
    paths = [
        bundle_path,
        INVENTORY,
        SEM / "reports.json",
        SEM / "feature-revision-r3/events.jsonl",
        SIGNAL / "features-r4/events.jsonl",
        SIGNAL / "features-r4/source-review.json",
    ]
    bundle = read(bundle_path)
    inventory = read(INVENTORY)
    sessions, calendar_paths = load_calendar()
    paths += calendar_paths
    with get_nav_preview_engine().connect() as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            nav_rows = (
                conn.execute(
                    text("""SELECT nav_date,ann_date,unit_nav,source_published_at,created_at,updated_at
                FROM nav_daily WHERE fund_code=:fund AND nav_date<=:end ORDER BY nav_date"""),
                    {"fund": "002112", "end": date.fromisoformat(MAX_LABEL)},
                )
                .mappings()
                .all()
            )
            nav_rows = json.loads(json.dumps([dict(r) for r in nav_rows], default=str))
    write(root / "nav-database-snapshot.json", {"received_at": now(), "read_only": True, "rows": nav_rows})
    nav = {}
    for row in nav_rows:
        if not row["ann_date"]:
            continue
        nav[row["nav_date"]] = {**row, "available_at": available_nav(row)}
        previous = bundle["nav"].get(row["nav_date"])
        # 旧记录中存在更晚的版本约束时继续保留；不能为增加样本而提前净值可用时间。
        if previous:
            if Decimal(previous["unit_nav"]) != Decimal(row["unit_nav"]):
                nav[row["nav_date"]]["available_at"] = max(
                    nav[row["nav_date"]]["available_at"], moment(row["updated_at"]).isoformat()
                )
            nav[row["nav_date"]]["available_at"] = max(nav[row["nav_date"]]["available_at"], previous["available_at"])
    # 只补旧完整行情包之后的收盘价；日期值/回执均保留。首见不全的部分仍属历史重建。
    recent = {}
    for path in sorted((RESEARCH / "quote-receipt-versions").glob("*.json")):
        receipt = read(path)["payload"]
        if receipt["api"] not in ("daily", "index_daily"):
            continue
        if receipt["params"].get("end_date", "") < "20260929":
            continue
        raw_path = RESEARCH / receipt["file"]
        if sha(raw_path) != receipt["sha256"]:
            raise ValueError("QUOTE_HASH_MISMATCH")
        raw = read(raw_path)["data"]
        paths.extend([path, raw_path])
        for values in raw["items"]:
            row = dict(zip(raw["fields"], values, strict=True))
            day = row["trade_date"]
            day = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
            if not "2026-09-29" <= day <= MAX_LABEL:
                continue
            key = (receipt["api"], row["ts_code"], day)
            previous = recent.get(key)
            if previous and any(previous.get(k) != row.get(k) for k in ["close", "pct_chg", "amount"]):
                row["revised_at"] = max(previous["received_at"], receipt["received_at"])
            if previous and previous["received_at"] < receipt["received_at"] and "revised_at" not in row:
                continue
            recent[key] = {**row, "received_at": receipt["received_at"], "source_sha256": receipt["sha256"]}
    for (api, code, day), row in recent.items():
        if api == "daily":
            bundle["stocks"].setdefault(day, {})[code] = row
        elif code in bundle["markets"]:
            bundle["markets"][code][day] = row
    result = {
        "sessions": sessions,
        "nav": nav,
        "markets": bundle["markets"],
        "stocks": bundle["stocks"],
        "reports": read(SEM / "reports.json"),
        "public_events": lines(SEM / "feature-revision-r3/events.jsonl"),
        "company_events": lines(SIGNAL / "features-r4/events.jsonl"),
        "dividends": inventory["dividends"],
    }
    write(root / "source-manifest.json", {"files": {str(p): sha(p) for p in sorted(set(paths))}})
    write(root / "sources.json", result)
    return result


def history_features(base: str, cutoff: str, source: dict):
    sessions, nav = source["sessions"], source["nav"]
    base_index = sessions.index(base)
    for end in range(base_index, max(59, base_index - 21), -1):
        days = sessions[end - 60 : end + 1]
        if len(days) != 61 or any(d not in nav or nav[d]["available_at"] > cutoff for d in days):
            continue
        values = np.array([float(nav[d]["unit_nav"]) for d in days])
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            continue
        returns = values[1:] / values[:-1] - 1
        lo, hi = float(min(values[1:])), float(max(values[1:]))
        streak = 0
        for r in reversed(returns):
            if r >= 0:
                break
            streak += 1
        r1, r5, r10, r20, r60 = [float(values[-1] / values[-1 - n] - 1) for n in [1, 5, 10, 20, 60]]
        target = sessions[base_index + 1]
        cash = sum(
            float(d["cash_dividend"])
            for d in source["dividends"]
            if d["ex_date"] == target and next_midnight(d["ann_date"]) <= cutoff
        )
        vector = [
            r1,
            r5,
            r10,
            r20,
            r60,
            float(returns[-5:].std()),
            float(returns[-20:].std()),
            float(returns.std()),
            float(min(values[1:] / np.maximum.accumulate(values[1:]) - 1)),
            (float(values[-1]) - lo) / (hi - lo) if hi > lo else None,
            float(streak),
            float(base_index - end),
            r1 - r5 / 5,
            r5 / 5 - r20 / 20,
            cash,
        ]
        return vector, {
            "nav_days": days,
            "latest_nav": days[-1],
            "lag_sessions": base_index - end,
            "maximum_available_at": max(nav[d]["available_at"] for d in days),
        }
    return None, {"reason": "NO_CONTIGUOUS_AVAILABLE_61_NAV_WINDOW"}


def quote_ok(row, day: str, cutoff: str) -> bool:
    """本轮收盘价重建假设为当日16点；已知晚修订必须后移，不用未来价格补缺失。"""
    return (
        bool(row) and max(day + "T16:00:00+08:00", row.get("revised_at") or "", row.get("available_at") or "") <= cutoff
    )


def market_features(base: str, cutoff: str, source: dict):
    sessions = source["sessions"]
    i = sessions.index(base)
    days = sessions[i - 20 : i + 1]
    report = choose_report(source["reports"], cutoff)
    weights = holding_weights(report)
    sums, covers = [0.0] * 4, [0.0] * 3
    details = []
    for code, weight in weights.items():
        qs = [source["stocks"].get(d, {}).get(code) for d in days]
        valid = [quote_ok(q, d, cutoff) for q, d in zip(qs, days, strict=True)]
        if valid[-1]:
            sums[0] += weight * float(qs[-1]["pct_chg"]) / 100
            sums[2] += weight * (qs[-1]["pct_chg"] > 0)
            covers[0] += weight
        if all(valid[-5:]):
            sums[1] += weight * (math.prod(1 + float(q["pct_chg"]) / 100 for q in qs[-5:]) - 1)
            covers[1] += weight
        if all(valid) and all(q.get("amount") is not None for q in qs):
            avg = sum(float(q["amount"]) for q in qs[:-1]) / 20
            if avg > 0:
                sums[3] += weight * (float(qs[-1]["amount"]) / avg - 1)
                covers[2] += weight
        details.append(
            {"code": code, "weight": weight, "missing_days": [d for d, ok in zip(days, valid, strict=True) if not ok]}
        )
    vector = [sums[k] if covers[c] else None for k, c in [(0, 0), (1, 1), (2, 0), (3, 2)]] + covers
    vector += [
        sum(weights.values()) if report else None,
        float(report["stock_nav_pct"]) / 100 if report else None,
        sum(w * w for w in weights.values()) if report else None,
        (date.fromisoformat(base) - date.fromisoformat(report["report_end"])).days if report else None,
        float(report["full_stock_disclosure"]) if report else None,
    ]
    for code in ["000300.SH", "000905.SH"]:
        qs = [source["markets"][code].get(d) for d in days]
        valid = [quote_ok(q, d, cutoff) for q, d in zip(qs, days, strict=True)]
        ret = [float(q["pct_chg"]) / 100 if ok else None for q, ok in zip(qs, valid, strict=True)]
        vector += [
            ret[-1],
            math.prod(1 + r for r in ret[-5:]) - 1 if all(valid[-5:]) else None,
            math.prod(1 + r for r in ret[-20:]) - 1 if all(valid[-20:]) else None,
            float(np.std(ret[-20:])) if all(valid[-20:]) else None,
            float(qs[-1]["amount"]) / (sum(float(q["amount"]) for q in qs[:-1]) / 20)
            if all(valid)
            and all(q.get("amount") is not None for q in qs)
            and sum(float(q["amount"]) for q in qs[:-1]) > 0
            else None,
        ]
    return vector, {
        "report_hash": report["raw"]["sha256"] if report else None,
        "report_available_at": report["available_at"] if report else None,
        "price_days": days,
        "holdings": details,
        "coverage_weight": covers[0],
    }


def information_features(base: str, cutoff: str, source: dict):
    """实际公司数字保留最新核实版；政策/新闻方向仅表示原文对象，绝不硬改为基金利好。"""
    report = choose_report(source["reports"], cutoff)
    current, sessions = holding_weights(report), source["sessions"]
    i = sessions.index(base)
    values = [None] * 20
    values[19] = (date.fromisoformat(base) - date.fromisoformat(report["report_end"])).days if report else None
    processed, qualified, public, proof = [], [], [], []
    for e in source["company_events"]:
        stamp = e["version_available_at"]
        if not stamp or stamp > cutoff or not e["links"]:
            continue
        age = i - effective_index(sessions, stamp)
        if not 0 <= age < 5:
            continue
        link = e["links"][0]
        weight = min(link["weight"], current.get(link["code"], 0))
        if weight <= 0 or e["status"].startswith("QUARANTINED") or e["status"] == "DUPLICATE_EVENT":
            continue
        processed.append(e)
        if e["status"] == "QUALIFIED":
            qualified.append((e, weight))
        proof.append(
            {
                "id": e["id"],
                "available_at": stamp,
                "weight": weight,
                "kind": "company",
                "age": age,
                "status": e["status"],
            }
        )
    if processed:
        for k in [*FLAG_FIELDS.values(), 12, 13, 14, 15, 16, 17, 18]:
            values[k] = 0.0
        values[12], values[14] = len(qualified), len(processed)
        for flag, ix in FLAG_FIELDS.items():
            weights = {}
            for e, w in qualified:
                if flag in e["flags"]:
                    weights[e["code"]] = max(weights.get(e["code"], 0), w)
            values[ix] = sum(weights.values())
        weights = {}
        for e, w in qualified:
            weights[e["code"]] = max(weights.get(e["code"], 0), w)
        values[13] = sum(weights.values())
        for (metric, ix), cover in zip(NUM_FIELDS.items(), [15, 16, 17, 18], strict=True):
            latest = {}
            for e, w in sorted(qualified, key=lambda x: (x[0]["version_available_at"], x[0]["id"])):
                if metric in e["numeric"]:
                    latest[e["code"]] = (e["numeric"][metric]["value"], w)
            denominator = sum(w for _, w in latest.values())
            values[cover] = denominator
            values[ix] = sum(v * w for v, w in latest.values()) / denominator if denominator else None
    for e in source["public_events"]:
        if e["source_kind"] not in ("news", "policy") or e["status"] != "SOURCE_GROUNDED" or e["repeated_facts"]:
            continue
        if e["available_at"] > cutoff:
            continue
        age = i - effective_index(sessions, e["available_at"])
        if not 0 <= age < 20:
            continue
        weight = sum(
            min(x["at_event_weight"], current.get(x["code"], 0))
            for x in e["links"]
            if x["basis"] != "APPLICATION_CONTEXT_ONLY"
        )
        public.append((e, age, weight))
        proof.append(
            {
                "id": e["id"],
                "available_at": e["available_at"],
                "weight": weight,
                "kind": e["source_kind"],
                "age": age,
                "status": e["status"],
            }
        )
    for window in [1, 5, 20]:
        for kind in ["news", "policy"]:
            rows = [(e, w) for e, age, w in public if age < window and e["source_kind"] == kind]
            # 数量只说明已收录材料；没有材料时方向等内容是未知，而不是“无利空”。
            values += [len(rows), sum(w for _, w in rows) if rows else None]
            values += [
                sum(e[field] == category for e, _ in rows) if rows else None
                for field, categories in PUBLIC_ENUMS.items()
                for category in categories
            ]
    assert len(values) == len(INFO_NAMES)
    return values, {
        "events": proof,
        "company_processed": len(processed),
        "company_qualified": len(qualified),
        "news": sum(e["source_kind"] == "news" for e, _, _ in public),
        "policy": sum(e["source_kind"] == "policy" for e, _, _ in public),
        "collection_complete": False,
        "historical_first_seen_proven": False,
    }


def build_dataset(source: dict) -> tuple[list, list, dict]:
    """逐日重建特征后独立挂接下一交易日标签；不随机切分，不以目标净值补输入。"""
    rows, audits, excluded = [], [], []
    sessions, nav = source["sessions"], source["nav"]
    for i, target in enumerate(sessions):
        if i == 0 or target > MAX_LABEL:
            continue
        base, cutoff = sessions[i - 1], sessions[i - 1] + "T23:00:00+08:00"
        if base not in nav or target not in nav:
            excluded.append({"base": base, "target": target, "reason": "LABEL_NAV_MISSING"})
            continue
        h, ha = history_features(base, cutoff, source)
        if h is None:
            excluded.append({"base": base, "target": target, **ha})
            continue
        a, b = Decimal(nav[base]["unit_nav"]), Decimal(nav[target]["unit_nav"])
        groups = {"history": h}
        audit = {"base": base, "target": target, "as_of": cutoff, "history": ha}
        if base >= JOINT_START:
            groups["market"], audit["market"] = market_features(base, cutoff, source)
            groups["information"], audit["information"] = information_features(base, cutoff, source)
        rows.append(
            {
                "base": base,
                "target": target,
                "as_of": cutoff,
                "groups": groups,
                "label": "UP" if b > a else "DOWN" if b < a else "FLAT",
                "return": float(b / a - 1),
                "label_mature_at": max(nav[base]["available_at"], nav[target]["available_at"]),
            }
        )
        audits.append(audit)
    summary = {
        "rows": len(rows),
        "first_target": rows[0]["target"],
        "last_target": rows[-1]["target"],
        "joint_rows": sum("information" in r["groups"] for r in rows),
        "excluded": excluded,
        "joint_by_year": dict(Counter(r["target"][:4] for r in rows if "information" in r["groups"])),
        "nav_records": len(nav),
        "documents": len(source["company_events"]),
        "public_source_counts": dict(Counter(e["source_kind"] for e in source["public_events"])),
        "information_days": {
            kind: sum(a.get("information", {}).get(kind, 0) > 0 for a in audits)
            for kind in ["company_qualified", "news", "policy"]
        },
    }
    return rows, audits, summary


def preprocess(x: np.ndarray, state=None):
    """预处理只在训练部分学习；全缺失列删除，其余用训练中位数并附加缺失标记。

    原输入及缺失原因始终保留 null。计算矩阵的中位数不是声称该历史事实已知。
    """
    if state is None:
        keep = ~np.all(np.isnan(x), axis=0)
        if not keep.any():
            raise ValueError("NO_OBSERVED_FEATURE")
        selected = x[:, keep]
        state = {"keep": keep, "median": np.nanmedian(selected, axis=0)}
    selected = x[:, state["keep"]]
    missing = np.isnan(selected)
    return np.concatenate([np.where(missing, state["median"], selected), missing.astype(float)], axis=1), state


def predict_expert(saved: dict, rows: list, branch: str) -> np.ndarray:
    x = np.array([r["groups"][branch] for r in rows], dtype=float)
    x, _ = preprocess(x, saved["preprocessing"])
    raw = saved["model"].predict_proba(x)
    aligned = np.zeros((len(rows), 3))
    for j, name in enumerate(saved["model"].classes_):
        aligned[:, CLASSES.index(name)] = raw[:, j]
    return aligned


def blend(branches: dict[str, np.ndarray], weights: dict = WEIGHTS) -> np.ndarray:
    if not math.isclose(sum(weights.values()), 1.0) or any(w < 0 for w in weights.values()):
        raise ValueError("INVALID_WEIGHTS")
    if any(k not in branches for k in weights):
        raise ValueError("MISSING_EXPERT_NO_SILENT_REWEIGHT")
    result = sum(weights[k] * branches[k] for k in weights)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=-1), 1.0):
        raise ValueError("INVALID_PROBABILITIES")
    return result


def fit_expert(root: Path, name: str, branch: str, rows: list, cutoff: str, evaluate: list) -> dict:
    training = [
        r for r in rows if branch in r["groups"] and r["label_mature_at"] < cutoff and r["target"] < cutoff[:10]
    ]
    if len(training) < 120:
        raise ValueError("INSUFFICIENT_TRAINING_ROWS")
    ledger = root / "fit-ledger.jsonl"
    previous = lines(ledger) if ledger.exists() else []
    if len(previous) >= 100 or any(e["id"] == name for e in previous):
        raise ValueError("FIT_BUDGET_OR_DUPLICATE")
    entry = {
        "id": name,
        "branch": branch,
        "at": now(),
        "cutoff": cutoff,
        "training_targets": [r["target"] for r in training],
        "count": len(training),
    }
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    x, state = preprocess(np.array([r["groups"][branch] for r in training], dtype=float))
    model = RandomForestClassifier(**RECIPE).fit(x, [r["label"] for r in training])
    saved = {
        "model": model,
        "preprocessing": state,
        "branch": branch,
        "cutoff": cutoff,
        "research_only": True,
        "historical_first_seen_proven": False,
        "row_count": len(training),
    }
    path = root / "models" / (name + ".joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("MODEL_ALREADY_EXISTS")
    joblib.dump(saved, path)
    probabilities = predict_expert(saved, evaluate, branch).tolist() if evaluate else []
    write(
        root / "fits" / (name + ".json"),
        {
            **entry,
            "model_sha256": sha(path),
            "probabilities": probabilities,
            "evaluation_targets": [r["target"] for r in evaluate],
        },
    )
    return saved


def metrics(records: list, method: str) -> dict:
    probabilities = np.array([r[method] for r in records])
    truths = np.array([CLASSES.index(r["label"]) for r in records])
    guesses = probabilities.argmax(axis=1)
    down = truths == 0
    return {
        "n": len(records),
        "correct": int((guesses == truths).sum()),
        "accuracy": float((guesses == truths).mean()),
        "down_recall": float((guesses[down] == 0).mean()) if down.any() else None,
        "brier": float(np.square(probabilities - np.eye(3)[truths]).sum(axis=1).mean()),
        "actual_counts": dict(Counter(r["label"] for r in records)),
    }


def freeze(root: Path) -> None:
    """固定唯一训练配方、对照、全量重训和评估日历；先冻结，后读取成绩。"""
    if root.exists():
        raise ValueError("RUN_DIRECTORY_EXISTS_USE_EXISTING_STAGE_OR_NEW_VERSION")
    root.mkdir(parents=True)
    write(root / "protection-before.json", git_snapshot())
    protocol = {
        "created_at": now(),
        "fund": "002112",
        "as_of": "D 23:00 Asia/Shanghai",
        "label": "sign(unit_nav[next_trading_day]/unit_nav[D]-1); exact Decimal comparison",
        "label_end": MAX_LABEL,
        "joint_start": JOINT_START,
        "weights": WEIGHTS,
        "recipe": RECIPE,
        "fixed_candidates": 1,
        "update_every_sessions": 20,
        "fit_cap": 100,
        "validation": "2025 and exposed 2026 expanding historical reconstruction; never untouched test",
        "baselines": ["history_only", "market_2/3_history_1/3", "always_up"],
        "full_fit": "each branch uses ALL its eligible history; labels mature before final cutoff",
        "manual_predictions_used_as_labels": False,
        "adoption": False,
        "new_paid_requests": 0,
        "limitations": [
            "历史收盘价按16点重建，原始首见版本不完整",
            "公告日期和已知版本约束不等于历史在线存档",
            "材料库非全互联网完整采集，部分资料正文为节选，可能有选择偏差",
            "海外市场和海外新闻缺少同口径完整历史，本轮无法复刻人工六小项评分",
            "55%为整体信息分支概率权重，不代表新闻贡献归因占比或已校准概率",
        ],
        "historical_screen": (
            "both years beat both baselines in accuracy and Brier; "
            "down recall no worse by >5pp; quarterly support >=half"
        ),
        "formal_gate": (
            "requires first-seen/version provenance, source coverage and "
            "untouched forward validation; fails with current evidence"
        ),
    }
    write(root / "protocol.json", protocol)
    (root / "code").mkdir()
    (root / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    source = load_sources(root)
    rows, audits, summary = build_dataset(source)
    write_lines(root / "dataset.jsonl", rows)
    write_lines(root / "daily-lineage.jsonl", audits)
    write(root / "coverage.json", summary)
    write(root / "feature-names.json", {"information": INFO_NAMES, "market": MARKET_NAMES, "history": HISTORY_NAMES})
    write(
        root / "freeze.json",
        {
            "files": {str(p): sha(p) for p in root.iterdir() if p.is_file()},
            "code": {str(Path(__file__)): sha(Path(__file__))},
            "packages": {n: importlib.metadata.version(n) for n in ["numpy", "scikit-learn", "joblib"]},
        },
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "excluded"}, ensure_ascii=False), flush=True)


def verify_freeze(root: Path) -> None:
    for item in [read(root / "freeze.json"), read(root / "source-manifest.json")]:
        for path, digest in item["files"].items():
            if sha(Path(path)) != digest:
                raise ValueError("FROZEN_SOURCE_CHANGED:" + path)
    for path, digest in read(root / "freeze.json")["code"].items():
        if sha(Path(path)) != digest:
            raise ValueError("FROZEN_CODE_CHANGED")


def train(root: Path) -> None:
    verify_freeze(root)
    if (root / "fit-ledger.jsonl").exists():
        raise ValueError("TRAIN_ALREADY_STARTED_DO_NOT_SILENTLY_REPEAT")
    rows = lines(root / "dataset.jsonl")
    output = []
    for year in ["2025", "2026"]:
        evaluation = [r for r in rows if r["target"].startswith(year) and "information" in r["groups"]]
        for start in range(0, len(evaluation), 20):
            block = evaluation[start : start + 20]
            cutoff = block[0]["as_of"]
            branches = {}
            for branch in WEIGHTS:
                name = year + f"_{start:03d}_" + branch
                saved = fit_expert(root, name, branch, rows, cutoff, block)
                branches[branch] = predict_expert(saved, block, branch)
            combined = blend(branches)
            price = blend(branches, {"market": 2 / 3, "history": 1 / 3})
            for i, row in enumerate(block):
                output.append(
                    {k: row[k] for k in ["base", "target", "as_of", "label", "return"]}
                    | {
                        "weighted_55": combined[i].tolist(),
                        "price_history": price[i].tolist(),
                        "history_only": branches["history"][i].tolist(),
                        "always_up": [0, 0, 1],
                        "branches": {b: p[i].tolist() for b, p in branches.items()},
                        "fit_prefix": year + f"_{start:03d}_",
                    }
                )
            print(f"完成 {year} 第 {start // 20 + 1} 组，累计历史判断 {len(output)} 日", flush=True)
    write_lines(root / "historical-predictions.jsonl", output)
    methods = ["weighted_55", "price_history", "history_only", "always_up"]
    scores = {}
    for period in ["2025", "2026", "ALL"]:
        group = [r for r in output if period == "ALL" or r["target"].startswith(period)]
        scores[period] = {m: metrics(group, m) for m in methods}
    for year in ["2025", "2026"]:
        for q in range(1, 5):
            group = [r for r in output if r["target"].startswith(year) and (int(r["target"][5:7]) - 1) // 3 + 1 == q]
            if group:
                scores[year + "Q" + str(q)] = {m: metrics(group, m) for m in methods}
    write(root / "scores.json", scores)
    # 分数不改变用户指定权重；最终三个分支重新拟合到所有已成熟历史，包括诊断年份。
    cutoff = now()
    final = {branch: fit_expert(root, "FULL_" + branch, branch, rows, cutoff, []) for branch in WEIGHTS}
    package = {
        "fund": "002112",
        "as_of_clock": "23:00 Asia/Shanghai",
        "horizon": "next_trading_day",
        "weights": WEIGHTS,
        "classes": CLASSES,
        "experts": final,
        "research_only": True,
        "historical_first_seen_proven": False,
        "features": read(root / "feature-names.json"),
        "trained_at": now(),
        "label_end": MAX_LABEL,
        "freeze_sha256": sha(root / "freeze.json"),
    }
    joblib.dump(package, root / "002112-evening-1d-news55.joblib")
    primary = []
    for year in ["2025", "2026"]:
        a = scores[year]["weighted_55"]
        for baseline in ["history_only", "price_history"]:
            b = scores[year][baseline]
            quarters = [k for k in scores if k.startswith(year + "Q")]
            quarter_support = sum(
                scores[k]["weighted_55"]["accuracy"] >= scores[k][baseline]["accuracy"] for k in quarters
            )
            primary.append(
                a["accuracy"] > b["accuracy"]
                and a["brier"] < b["brier"]
                and a["down_recall"] >= b["down_recall"] - 0.05
                and quarter_support >= math.ceil(len(quarters) / 2)
            )
    write(
        root / "decision.json",
        {
            "training_complete": True,
            "historical_screen_passed": all(primary),
            "production_eligible": False,
            "adopted": False,
            "forward_validated": False,
            "fits": len(lines(root / "fit-ledger.jsonl")),
            "final_training_rows": {k: v["row_count"] for k, v in final.items()},
            "reason": "历史版本与采集覆盖未通过正式证据要求，且2025/2026均已见；保留研究模型。",
        },
    )
    print(json.dumps({"scores": scores, "decision": read(root / "decision.json")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.stage == "prepare":
        freeze(args.root)
    else:
        train(args.root)


if __name__ == "__main__":
    main()
