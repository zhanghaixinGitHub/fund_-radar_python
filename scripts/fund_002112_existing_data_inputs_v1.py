"""002112 独立实验的只读输入适配。

构造阶段只计算历史 X 和标签成熟元数据，不生成目标答案。标签接口独立执行，
由训练执行者先记录访问并提供阶段凭据。所有原件只读；快照按内容寻址且禁止覆盖。
净值七项、披露组合九项公式分别对齐 direction_1d_protocol / fund_exposure_features。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("Asia/Shanghai")
PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT / ".local-runs/fund-exposure-002112"
PACKAGE = RESEARCH / "recent-input-completion/20260930-v1"
DEFAULT_ROOT = RESEARCH / "existing-data-experiment/20260930-v1"
VERSION = "FUND_002112_EXISTING_DATA_INPUTS_V1"
SECTORS = ["399998.SZ", "399986.SZ", "980017.SZ", "000819.SH", "980030.SZ"]
MARKETS = ["000300.SH", "000905.SH"]
N = [
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "max_drawdown_60d",
    "relative_position_60d",
    "consecutive_decline_days",
    "lag_sessions",
]
M = ["csi300_return_1d", "csi300_return_5d", "csi500_return_1d", "csi500_return_5d"]
H = [
    "disclosed_nav_return_1d",
    "disclosed_nav_return_5d",
    "up_stock_nav_weight",
    "nav_weighted_amount_vs_previous20",
    "nav_weight_concentration",
    "disclosed_nav_weight",
    "reported_stock_nav_weight",
    "report_age_days",
    "full_disclosure",
]
INDUSTRY_COLUMNS = [f"sector_{code.replace('.', '_')}_return_{n}d" for code in SECTORS for n in [1, 5, 20]]
F = [
    "top10_jaccard_change",
    "top10_nav_weight_change",
    "top3_nav_weight_change",
    "stock_nav_weight_change",
    "industry_nav_weight_concentration",
    "report_public_age_days",
    "c_share_net_assets_log1p",
    "manager_min_tenure_days",
    "manager_count",
    "fund_age_days",
]
G = [
    f"{kind}_count_{n}d"
    for kind in ["earnings", "buyback", "contract", "fund_manager_strategy", "policy", "industry_news"]
    for n in [5, 20]
]
CANDIDATE_GROUPS = {
    "A": ["N"],
    "B": ["N", "M"],
    "C": ["N", "H", "M"],
    "D": ["N", "M", "I"],
    "E": ["N", "H", "M", "I"],
    "F": ["N", "H", "M", "I", "F"],
    "G": ["N", "H", "M", "I", "G"],
    "H": ["N", "H", "M", "I"],
}
TRAIN_END = {"V1": "2024-12-31", "V2": "2025-06-30", "V3": "2025-09-30", "T": "2025-12-31", "FINAL": "2026-09-29"}
FIT_CUTOFF = {
    "V1": "2025-01-02T08:00:00+08:00",
    "V2": "2025-07-01T08:00:00+08:00",
    "V3": "2025-10-09T08:00:00+08:00",
    "T": "2026-01-05T08:00:00+08:00",
}
EVAL_RANGE = {
    "V1": ("2025-01-01", "2025-06-30"),
    "V2": ("2025-07-01", "2025-09-30"),
    "V3": ("2025-10-01", "2025-12-31"),
    "T": ("2026-01-01", "2026-09-29"),
}
EXPECTED = {
    "artifact-index.json": "325d908dd48ca13ccacd7f0852dfe245841daddf818807c2934bead4c9465afb",
    "input-date-manifest.json": "c852769501294624839a857fb2593648e085e1200d50589112421455297cff40",
    "input-provenance.json": "480c180414a3e4f38c612ecd7a13dde31270a53bf3c6f0a005dedb0c1176eee0",
    "sector-source-inputs.json": "8f139977c916d6f120fe55d3dd6c39c77eba1069efa6933be63becde25de37a3",
}


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path | str):
    """兼容已有带内容摘要的封装；校验后返回数据，不初始化任何业务服务。"""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and set(obj) == {"hash", "payload"}:
        if digest(obj["payload"]) != obj["hash"]:
            raise ValueError("SOURCE_PAYLOAD_HASH_MISMATCH")
        return obj["payload"]
    return obj


def save_new(path: Path, value) -> None:
    """独占发布新产物；完全相同内容可续用，任何不同内容均拒绝覆盖。"""
    raw = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"IMMUTABLE_OUTPUT_CONFLICT:{path.name}")
        return
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temp.write_bytes(raw)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class Snapshot:
    """原件复制前后与期望摘要三方一致才接受；内容寻址使断点恢复不覆盖。"""

    def __init__(self, root: Path):
        self.root = root
        self.sources: dict[str, dict] = {}

    def copy(self, path: Path | str, expected: str | None = None) -> Path:
        path = Path(path).resolve()
        if str(path) in self.sources:
            record = self.sources[str(path)]
            if expected and expected != record["sha256"]:
                raise ValueError("SOURCE_EXPECTATION_CONFLICT")
            return self.root / record["snapshot_path"]
        before = sha(path)
        if expected and before != expected:
            raise ValueError(f"SOURCE_EXPECTED_HASH_MISMATCH:{path}")
        raw = path.read_bytes()
        after = sha(path)
        if before != after or hashlib.sha256(raw).hexdigest() != before:
            raise ValueError(f"SOURCE_CHANGED_DURING_COPY:{path}")
        out = self.root / "snapshot/originals" / (before + path.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if sha(out) != before:
                raise ValueError("SNAPSHOT_CONTENT_CONFLICT")
        else:
            with out.open("xb") as stream:
                stream.write(raw)
        if sha(out) != before or sha(path) != before:
            raise ValueError("SOURCE_CHANGED_AFTER_COPY")
        self.sources[str(path)] = {
            "source_path": str(path),
            "snapshot_path": out.relative_to(self.root).as_posix(),
            "sha256": before,
            "bytes": len(raw),
            "before_after_equal": True,
        }
        return out

    def read(self, path: Path | str, expected: str | None = None):
        return read(self.copy(path, expected))


def at0800(day: str | date) -> datetime:
    return datetime.combine(date.fromisoformat(day) if isinstance(day, str) else day, time(8), ZONE)


def available_at(*constraints: str | None) -> str:
    """日期证据取下一自然日 08:00；明确时刻须含时区；多个约束取最晚值。"""
    points = []
    for value in constraints:
        if not value:
            continue
        if len(value) == 10:
            points.append(at0800(date.fromisoformat(value) + timedelta(days=1)))
        else:
            point = datetime.fromisoformat(value)
            if point.tzinfo is None:
                raise ValueError("SOURCE_TIMEZONE_REQUIRED")
            points.append(point.astimezone(ZONE))
    if not points:
        raise ValueError("SOURCE_PUBLICATION_UNKNOWN")
    return max(points).isoformat()


def nav_features(values: list) -> list[float]:
    """61 个实际净值计算原七项：波动为20日总体标准差，不年化。"""
    if len(values) != 61:
        raise ValueError("HISTORY_TOO_SHORT")
    v = [float(x) for x in values]
    if any(not math.isfinite(x) or x <= 0 for x in v):
        raise ValueError("INVALID_NAV")
    recent = v[1:]
    low, high = min(recent), max(recent)
    if low == high:
        raise ValueError("FLAT_FEATURE_WINDOW")
    returns = [v[i] / v[i - 1] - 1 for i in range(41, 61)]
    avg = sum(returns) / 20
    peak, drawdown = recent[0], 0.0
    for point in recent:
        peak = max(peak, point)
        drawdown = min(drawdown, point / peak - 1)
    decline = 0
    for i in range(60, 0, -1):
        if v[i] >= v[i - 1]:
            break
        decline += 1
    return [
        v[60] / v[55] - 1,
        v[60] / v[40] - 1,
        v[60] / v[0] - 1,
        math.sqrt(sum((r - avg) ** 2 for r in returns) / 20),
        drawdown,
        (v[60] - low) / (high - low),
        float(decline),
    ]


def choose_nav_window(sessions: list[str], nav: dict, target: str) -> tuple[str | None, int | None, list | None]:
    """只按可得时间与有效性寻找 S，最大20交易日；不接触目标方向。"""
    t_index = sessions.index(target) - 1
    cutoff = at0800(target)
    for lag in range(21):
        end = t_index - lag
        if end < 60:
            continue
        days = sessions[end - 60 : end + 1]
        if any(d not in nav or datetime.fromisoformat(nav[d]["available_at"]) > cutoff for d in days):
            continue
        try:
            return sessions[end], lag, nav_features([nav[d]["unit_nav"] for d in days]) + [float(lag)]
        except ValueError:
            continue
    return None, None, None


def select_report(reports: list[dict], cutoff: datetime) -> dict | None:
    """期末日优先、同一期再按版本时点；禁止份额不一致的报告进入候选。"""
    rs = [
        r
        for r in reports
        if r["fund_code"] == "002112"
        and r["fund_master_code"] == "001412"
        and datetime.fromisoformat(r["available_at"]) <= cutoff
    ]
    return max(rs, key=lambda r: (r["report_end"], r["available_at"], r["raw"]["sha256"])) if rs else None


def quote_available(row: dict, day: str, sessions: list[str], cutoff: datetime) -> bool:
    """收盘价按下一交易日08:00重建，已知较晚修订仍须等到该时刻。"""
    next_day = sessions[sessions.index(day) + 1]
    constraint = available_at(at0800(next_day).isoformat(), row.get("revised_at"), row.get("available_at"))
    return datetime.fromisoformat(constraint) <= cutoff


def holdings_features(report: dict | None, stocks: dict, days: list[str], sessions: list[str], cutoff: datetime):
    """九项保持原净资产权重分母；一只正权重股票缺任一21日行情则整组缺失。"""
    if report is None:
        return None, ["NO_AVAILABLE_REPORT"]
    age = (date.fromisoformat(days[-1]) - date.fromisoformat(report["report_end"])).days
    if not 0 <= age <= 210:
        return None, ["REPORT_STALE"]
    total = float(report["disclosed_nav_pct"]) / 100
    if total <= 0:
        return None, ["DISCLOSED_WEIGHT_NONPOSITIVE"]
    weighted = weighted5 = up_weight = activity = hhi = 0.0
    missing = []
    for holding in report["holdings"]:
        weight = float(holding["nav_weight_pct"]) / 100
        if weight <= 0:
            continue
        code = holding["stock_code"]
        quotes = [stocks.get(d, {}).get(code) for d in days]
        if any(r is None for r in quotes):
            missing.append("MISSING_QUOTE:" + code)
            continue
        if any(not quote_available(r, d, sessions, cutoff) for r, d in zip(quotes, days, strict=True)):
            missing.append("QUOTE_VERSION_NOT_AVAILABLE:" + code)
            continue
        if any(not math.isfinite(float(r[k])) for r in quotes for k in ["pct_chg", "amount"]):
            missing.append("INVALID_QUOTE:" + code)
            continue
        mean_amount = sum(r["amount"] for r in quotes[:-1]) / 20
        if mean_amount <= 0:
            missing.append("AMOUNT_BASE_NOT_POSITIVE:" + code)
            continue
        weighted += weight * quotes[-1]["pct_chg"] / 100
        weighted5 += weight * (math.prod(1 + r["pct_chg"] / 100 for r in quotes[-5:]) - 1)
        up_weight += weight if quotes[-1]["pct_chg"] > 0 else 0
        activity += weight * (quotes[-1]["amount"] / mean_amount - 1)
        hhi += weight**2
    if missing:
        return None, missing
    return [
        weighted,
        weighted5,
        up_weight,
        activity,
        hhi,
        total,
        float(report["stock_nav_pct"]) / 100,
        float(age),
        float(report["full_stock_disclosure"]),
    ], []


def report_extensions(report: dict | None, reports: list[dict], cutoff: datetime) -> dict:
    """只比较前一个不同报告期的前十大；不把跌出名单当成仓位归零。"""
    values = dict.fromkeys(F)
    if report is None:
        return values
    previous = select_report([r for r in reports if r["report_end"] < report["report_end"]], cutoff)

    def top(r, n):
        return sorted(r["holdings"], key=lambda h: h["reported_rank"])[:n]

    if previous:
        a, b = {h["stock_code"] for h in top(report, 10)}, {h["stock_code"] for h in top(previous, 10)}
        values[F[0]] = 1 - len(a & b) / len(a | b)
        for i, n in [(1, 10), (2, 3)]:
            values[F[i]] = (
                sum(float(h["nav_weight_pct"]) for h in top(report, n))
                - sum(float(h["nav_weight_pct"]) for h in top(previous, n))
            ) / 100
        values[F[3]] = (float(report["stock_nav_pct"]) - float(previous["stock_nav_pct"])) / 100
    industries = report.get("reported_industries")
    if industries:
        values[F[4]] = sum((float(i["nav_weight_pct"]) / 100) ** 2 for i in industries)
    # 取已声明公开日距U的自然日数；准入另由available_at约束，二者不得混用。
    values[F[5]] = float((cutoff.date() - date.fromisoformat(report["published_date"])).days)
    # 既有解析没有 C 份额资产、历史经理任免和成立日期字段；未知保持 None。
    return values


def choose_optional(rows: list[dict], names: list[str], raw_key: str) -> dict:
    """只见2024年基础E可用行的X，逐列80%覆盖、非恒定及联合60行规则。"""
    base = [
        r
        for r in rows
        if r["target_date"].startswith("2024") and all(r["groups"][g] is not None for g in CANDIDATE_GROUPS["E"])
    ]
    columns, decisions = [], {}
    for name in names:
        values = [r[raw_key][name] for r in base if r[raw_key][name] is not None]
        coverage = len(values) / len(base) if base else 0
        reason = (
            "COVERAGE_BELOW_80_PERCENT" if coverage < 0.8 else "CONSTANT_2024_INPUT" if len(set(values)) < 2 else None
        )
        decisions[name] = {
            "covered": len(values),
            "denominator": len(base),
            "coverage": coverage,
            "retained": reason is None,
            "reason": reason,
        }
        if reason is None:
            columns.append(name)
    joint = sum(all(r[raw_key][n] is not None for n in columns) for r in base) if columns else 0
    return {
        "enabled": bool(columns) and joint >= 60,
        "columns": columns,
        "joint_2024_rows": joint,
        "column_decisions": decisions,
        "selection_uses_target_labels": False,
    }


def build_rows(bundle: dict) -> tuple[list[dict], dict, dict]:
    sessions, nav, reports = bundle["sessions"], bundle["nav"], bundle["reports"]
    rows = []
    for target in bundle["targets"]:
        idx = sessions.index(target)
        base, cutoff = sessions[idx - 1], at0800(target)
        end, lag, n = choose_nav_window(sessions, nav, target)
        report = select_report(reports, cutoff)
        h, errors = holdings_features(report, bundle["stocks"], sessions[idx - 21 : idx], sessions, cutoff)
        m, industry = [], []
        for code in MARKETS:
            q = [bundle["markets"][code].get(d) for d in sessions[idx - 5 : idx]]
            if any(r is None for r in q) or any(
                not quote_available(r, d, sessions, cutoff) for r, d in zip(q, sessions[idx - 5 : idx], strict=True)
            ):
                m = None
                break
            returns = [float(r["pct_chg"]) / 100 for r in q]
            m.extend([returns[-1], math.prod(1 + v for v in returns) - 1])
        for code in SECTORS:
            series = bundle["sectors"][code]
            for span in [1, 5, 20]:
                ds = sessions[idx - span : idx]
                q = [series.get(d) for d in ds]
                if any(r is None for r in q) or any(
                    not quote_available(r, d, sessions, cutoff) for r, d in zip(q, ds, strict=True)
                ):
                    industry = None
                    break
                industry.append(float(math.prod(Decimal(r["close"]) / Decimal(r["pre_close"]) for r in q) - 1))
            if industry is None:
                break
        maturity = max(nav[target]["available_at"], nav[base]["available_at"])
        groups = {"N": n, "M": m, "H": h, "I": industry}
        availability = {
            g: {
                "available": v is not None,
                "reasons": [] if v is not None else errors if g == "H" else ["NO_ELIGIBLE_INPUT"],
            }
            for g, v in groups.items()
        }
        rows.append(
            {
                "target_date": target,
                "base_date": base,
                "nav_end_date": end,
                "as_of": cutoff.isoformat(),
                "lag_sessions": lag,
                "groups": groups,
                "availability": availability,
                "label_mature_at": maturity,
                "source_digest": bundle["source_digest"],
                "stock_no_trade_states": bundle.get("no_trade_states", {}).get(target, []),
                "report": {
                    k: report[k]
                    for k in [
                        "fund_code",
                        "fund_master_code",
                        "report_end",
                        "published_date",
                        "available_at",
                        "full_stock_disclosure",
                    ]
                }
                | {"raw_sha256": report["raw"]["sha256"]}
                if report
                else None,
                "raw_F": report_extensions(report, reports, cutoff),
                "raw_G": dict.fromkeys(G),
            }
        )
    decisions = {g: choose_optional(rows, cols, "raw_" + g) for g, cols in [("F", F), ("G", G)]}
    columns = {"N": N, "M": M, "H": H, "I": INDUSTRY_COLUMNS, **{g: decisions[g]["columns"] for g in ["F", "G"]}}
    for row in rows:
        for g in ["F", "G"]:
            raw = row.pop("raw_" + g)
            values = [raw[c] for c in columns[g]]
            ok = decisions[g]["enabled"] and all(v is not None for v in values)
            row["groups"][g] = values if ok else None
            row["availability"][g] = {
                "available": ok,
                "reasons": []
                if ok
                else ["OPTIONAL_BRANCH_DISABLED" if not decisions[g]["enabled"] else "OPTIONAL_FIELD_MISSING"],
            }
    return rows, columns, decisions


def build_snapshot(root: Path = DEFAULT_ROOT) -> dict:
    """一次构造输入快照，无网络、数据库或标签分类调用；ready由验收后另行发布。"""
    if (root / "handoff/data/ready.json").exists():
        raise ValueError("READY_SNAPSHOT_IMMUTABLE")
    snapshot = Snapshot(root)
    package = {name: snapshot.read(PACKAGE / name, expected) for name, expected in EXPECTED.items()}
    for name in [
        "input-acceptance.json",
        "nav-unresolved-evidence.json",
        "suspension-evidence.json",
        "validation-results.json",
        "request-ledger-final.json",
    ]:
        snapshot.copy(PACKAGE / name)
    provenance = package["input-provenance.json"]
    manifests = package["input-date-manifest.json"]["rows"]
    targets = [r["target_date"] for r in manifests]
    # 使用已验哈希的正式日历纯读取函数，不导入任何会初始化研究/数据库的入口。
    from app.services.direction_1d_protocol import calendar

    sessions, calendar_hash = calendar()
    sessions = [str(d) for d in sessions]
    for name in ["cn_a_share_2021_2025_v1.json", "cn_a_share_2026_v1.json"]:
        snapshot.copy(PROJECT / "app/data/calendars" / name)
    if calendar_hash != provenance["calendar_hash"] or targets != [
        d for d in sessions if "2024-01-02" <= d <= "2026-09-29"
    ]:
        raise ValueError("FROZEN_CALENDAR_MISMATCH")
    if len(targets) != 665:
        raise ValueError("TARGET_COUNT_MISMATCH")
    nav = {}
    originals = {}
    for key, source in provenance["nav"]["official_sources"].items():
        raw = snapshot.read(source["path"], key)
        originals[key] = {row["date"]: row for row in raw["dataList"]}
    for day, meta in provenance["nav"]["rows"].items():
        original = originals[meta["official_raw_sha256"]][day]
        if original["fundcode"] != "002112" or not meta["positive"] or not meta["matches_official"]:
            raise ValueError("NAV_IDENTITY_OR_ACCEPTANCE_INVALID")
        value = Decimal(str(original["netvalue"]))
        if not value.is_finite() or value <= 0:
            raise ValueError("NAV_INVALID")
        nav[day] = {
            "unit_nav": str(value),
            "available_at": available_at(meta["ann_date"], meta["source_published_at"]),
            "fund_code": "002112",
            "ann_date": meta["ann_date"],
            "source_published_at": meta["source_published_at"],
            "raw_sha256": meta["official_raw_sha256"],
            "content_hash": meta["content_hash"],
        }
    report_index = snapshot.read(RESEARCH / "report-result.json")
    reports = []
    for name in report_index["report_files"]:
        path = RESEARCH / "reports" / (name + ".json")
        item = read(path)
        # 加入2023年6月锚点供首个报告的不同期比较；更早报告不进入实验。
        if item["raw"]["sha256"] not in provenance["reports"] and not "2023-06-30" <= item["report_end"] < "2023-09-30":
            continue
        item = snapshot.read(path)
        raw = item["raw"]
        snapshot.copy(Path(raw["raw_path"]) if raw.get("raw_path") else RESEARCH / raw["file"], raw["sha256"])
        if (
            item["fund_code"] != "002112"
            or item["fund_master_code"] != "001412"
            or item["quality"] != "VERIFIED_TABLE_TOTALS"
        ):
            raise ValueError("REPORT_IDENTITY_OR_QUALITY_INVALID")
        verified = provenance["reports"].get(raw["sha256"])
        if verified and digest(item) != verified["payload_hash"]:
            raise ValueError("REPORT_PAYLOAD_CHANGED")
        item["available_at"] = available_at(item["published_date"], item["available_at"], item.get("revised_at"))
        reports.append(item)
    stocks = {}
    for day, meta in provenance["stock_dates"].items():
        cached = snapshot.read(meta["cache_path"], meta["cache_sha256"])
        snapshot.copy(meta["raw_path"], meta["raw_sha256"])
        # 原补齐包已逐条核验原始响应；这里只保留该日未来可能用到的披露股票。
        needed_codes = {h["stock_code"] for report in reports for h in report["holdings"]}
        stocks[day] = {code: value for code, value in cached["rows"].items() if code in needed_codes}
    markets = {}
    for code, meta in provenance["market"].items():
        cached = snapshot.read(meta["cache_path"], meta["cache_sha256"])
        for source in meta["sources"]:
            snapshot.copy(source["raw_path"], source["raw_sha256"])
        markets[code] = {d: cached["rows"][d] for d in meta["required_dates"]}
    sectors = package["sector-source-inputs.json"]
    for source in sectors["sources"]:
        snapshot.copy(source["path"], source["sha256"])
        if source.get("receipt_path"):
            snapshot.copy(source["receipt_path"])
    optional = snapshot.read(RESEARCH / "data-completion/20260930-v1/acceptance-final-v3.json")
    optional_audit = {}
    for name, reference in optional["inputs"].items():
        material = snapshot.read(reference["path"], reference["sha256"])
        rows = material.get("rows", [])
        counts = dict(sorted(Counter(str(r.get("published_date"))[:4] for r in rows).items()))
        optional_audit[name] = {
            "source": reference,
            "rows": len(rows),
            "publication_years": counts,
            "rows_published_2024": sum(str(r.get("published_date", "")).startswith("2024") for r in rows),
        }
    optional_audit["decision"] = "NO_AUDITED_2024_CATALOG_WINDOW_COVERAGE_G_VALUES_UNKNOWN_NOT_ZERO"
    save_new(root / "handoff/data/optional-cache-audit.json", optional_audit)
    source_manifest = {
        "version": VERSION,
        "sources": sorted(snapshot.sources.values(), key=lambda x: x["source_path"]),
        "calendar_hash": calendar_hash,
        "source_vintage": "HISTORICAL_RECONSTRUCTION_CURRENT_VERSION_RISK",
    }
    save_new(root / "source-manifest.json", source_manifest)
    bundle = {
        "sessions": sessions,
        "targets": targets,
        "nav": nav,
        "reports": reports,
        "stocks": stocks,
        "markets": markets,
        "sectors": sectors["series"],
        "source_digest": digest(source_manifest),
        "no_trade_states": {r["target_date"]: r["stock_no_trade_states"] for r in manifests},
    }
    save_new(root / "snapshot/nav-by-date.json", nav)
    save_new(root / "snapshot/normalized-inputs.json", bundle)
    rows, columns, decisions = build_rows(bundle)
    decisions["G"]["reason"] = optional_audit["decision"]
    decisions["F"]["unsupported_fields_reason"] = (
        "PARSED_REPORTS_LACK_C_SHARE_SIZE_MANAGER_HISTORY_AND_INCEPTION_FIELDS"
    )
    candidate_columns = {c: [name for g in groups for name in columns[g]] for c, groups in CANDIDATE_GROUPS.items()}
    payload = {
        "version": VERSION,
        "rows": rows,
        "group_columns": columns,
        "candidate_columns": candidate_columns,
        "optional_decisions": decisions,
        "input_digest": digest(rows),
    }
    save_new(root / "handoff/data/inputs.json", payload)
    save_new(
        root / "feature-spec.json",
        {
            "version": VERSION,
            "group_columns": columns,
            "candidate_columns": candidate_columns,
            "candidate_group_order": CANDIDATE_GROUPS,
        },
    )
    save_new(root / "optional-feature-decisions.json", decisions)
    coverage = {
        "targets": len(rows),
        "years": dict(Counter(r["target_date"][:4] for r in rows)),
        "group_available": {g: sum(r["groups"][g] is not None for r in rows) for g in columns},
        "candidate_input_available": {
            c: sum(all(r["groups"][g] is not None for g in groups) for r in rows)
            for c, groups in CANDIDATE_GROUPS.items()
        },
        "lag_sessions": dict(Counter(str(r["lag_sessions"]) for r in rows)),
        "labels_generated": False,
        "actual_new_fits": 0,
    }
    save_new(root / "feature-coverage.json", coverage)
    save_new(root / "coverage-before.json", package["input-date-manifest.json"])
    return coverage


def load_inputs(root: Path | str = DEFAULT_ROOT) -> dict:
    """读取稳定X，不读取nav快照或生成标签；校验交付摘要后返回普通字典。"""
    root = Path(root)
    pointer_path = root / "handoff/data/active-revision.json"
    if pointer_path.exists():
        pointer = read(pointer_path)
        ready = _bound_file(root, pointer["ready_path"], pointer["ready_sha256"])
        input_path = _bound_file(root, pointer["inputs_path"], pointer["inputs_sha256"])
        _bound_file(root, pointer["feature_spec_path"], pointer["feature_spec_sha256"])
        selected_ready = read(ready)
        if selected_ready["revision"] != pointer["revision"] or selected_ready["status"] != "READY":
            raise ValueError("ACTIVE_REVISION_NOT_READY")
        payload = read(input_path)
        # 指针、ready及输入三方绑定；不能通过改指针绕开ready中的真实产物摘要。
        for relative, expected in selected_ready["artifacts"].items():
            if relative in [pointer["inputs_path"], pointer["feature_spec_path"], "snapshot/nav-by-date.json"]:
                _bound_file(root, relative, expected)
        if selected_ready["artifacts"].get(pointer["inputs_path"]) != pointer["inputs_sha256"]:
            raise ValueError("ACTIVE_INPUT_NOT_BOUND_TO_READY")
        if selected_ready["artifacts"].get(pointer["feature_spec_path"]) != pointer["feature_spec_sha256"]:
            raise ValueError("ACTIVE_FEATURE_SPEC_NOT_BOUND_TO_READY")
        payload.update(
            {
                "revision": pointer["revision"],
                "data_ready_path": str(ready),
                "data_ready_sha256": pointer["ready_sha256"],
                "active_revision_sha256": sha(pointer_path),
            }
        )
    else:
        ready = root / "handoff/data/ready.json"
        payload = read(root / "handoff/data/inputs.json")
        payload.update(
            {
                "revision": "v1",
                "data_ready_path": str(ready),
                "data_ready_sha256": sha(ready) if ready.exists() else None,
                "active_revision_sha256": None,
            }
        )
    if payload["version"] != VERSION or payload["input_digest"] != digest(payload["rows"]):
        raise ValueError("INPUT_DIGEST_MISMATCH")
    if ready.exists() and not pointer_path.exists():
        for name, expected in read(ready)["artifacts"].items():
            if name in ["handoff/data/inputs.json", "snapshot/nav-by-date.json", "feature-spec.json"]:
                if sha(root / name) != expected:
                    raise ValueError("READY_ARTIFACT_CHANGED")
    return payload


def chinese_date(value: str) -> str:
    """读取报告/公告的明确中文日期，不从文件名、抓取日期反推公开日。"""
    match = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", re.sub(r"\s+", "", value))
    if not match:
        raise ValueError("EXPLICIT_CHINESE_DATE_REQUIRED")
    return date(*map(int, match.groups())).isoformat()


def pdf_pages(path: Path, limit: int = 14) -> list[str]:
    """仅提取限定页数供表格/身份适配；不分析净值方向或训练成绩。"""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(path)
    result = []
    try:
        for index in range(min(limit, len(document))):
            page = document[index]
            textpage = page.get_textpage()
            try:
                result.append(re.sub(r"\s+", "", textpage.get_text_range()))
            finally:
                textpage.close()
                page.close()
    finally:
        document.close()
    return result


def parse_report_state(report: dict, pages: list[str]) -> dict:
    """从同一份已验身份报告解析C列资产、合同成立日与经理表，绝不使用现库资产/离任值。

    年报财务表按“当年A/C、上年A/C”排序，因此固定取首年第二列；强制检查币种、
    份额列名和本期年份。经理表只取本基金任职行，证券从业年限不作为本基金任期。
    """
    text = "".join(pages)
    if "002112" not in text or "德邦鑫星价值" not in text:
        raise ValueError("REPORT_STATE_IDENTITY")
    financial = []
    for index, page in enumerate(pages):
        if "期末基金资产净值" not in page:
            continue
        prefix, suffix = page.split("期末基金资产净值", 1)
        if not re.search(r"德邦鑫星价值(?:灵活配置混合)?A德邦鑫星价值(?:灵活配置混合)?C", prefix):
            raise ValueError("REPORT_STATE_SHARE_COLUMN_ORDER")
        if "人民币元" not in prefix or report["report_end"][:4] + "年" not in prefix:
            raise ValueError("REPORT_STATE_PERIOD_OR_UNIT")
        match = re.match(r"([\d,]+\.\d{2})([\d,]+\.\d{2})", suffix)
        if not match:
            raise ValueError("REPORT_STATE_ASSET_ROW")
        a_value, c_value = [Decimal(v.replace(",", "")) for v in match.groups()]
        if min(a_value, c_value) <= 0:
            raise ValueError("REPORT_STATE_ASSET_NONPOSITIVE")
        financial.append(
            {
                "page": index + 1,
                "c_share_net_assets_cny": str(c_value),
                "a_share_net_assets_cny": str(a_value),
                "unit": "CNY",
                "row_excerpt": "期末基金资产净值" + match.group(),
                "column_rule": "CURRENT_PERIOD_A_THEN_C",
            }
        )
    if len(financial) != 1:
        raise ValueError("REPORT_STATE_ASSET_NOT_UNIQUE")
    inception = re.search(r"基金合同生效日(?:为)?(\d{4}年\d{1,2}月\d{1,2}日)", text)
    share_date = re.search(r"自(\d{4}年\d{1,2}月\d{1,2}日)起本基金增加C类基金份额", text)
    if not inception:
        raise ValueError("REPORT_STATE_INCEPTION_EVIDENCE")
    # 名单来自此次有限报告集已逐项核对的真实经理身份；不扫描其他基金的简历经历。
    manager_pattern = (
        r"(吴志鹏|揭诗琪|雷涛|陆阳|张铮烁)(本基金的基金经理|无)"
        r"(\d{4}年\d{1,2}月\d{1,2}日)(-|\d{4}年\d{1,2}月\d{1,2}日)"
    )
    managers = []
    for index, page in enumerate(pages):
        for name, role, begin, end in re.findall(manager_pattern, page):
            managers.append(
                {
                    "name": name,
                    "begin_date": chinese_date(begin),
                    "end_date": None if end == "-" else chinese_date(end),
                    "role": role,
                    "page": index + 1,
                }
            )
    if not managers or len({m["name"] for m in managers}) != len(managers):
        raise ValueError("REPORT_MANAGER_TABLE_MISSING_OR_DUPLICATE")
    return {
        "raw_sha256": report["raw"]["sha256"],
        "report_end": report["report_end"],
        "published_date": report["published_date"],
        "available_at": report["available_at"],
        "financial": financial[0],
        "fund_inception_date": chinese_date(inception.group(1)),
        "c_share_added_date": chinese_date(share_date.group(1)) if share_date else None,
        "managers": managers,
    }


def parse_manager_notice(document: dict, pages: list[str]) -> dict:
    """独立任免公告只采用本基金原文；CMS较晚发布时间/修订约束一并保留，不能回拨。"""
    text = "".join(pages)
    if "德邦鑫星价值" not in text or not pages:
        raise ValueError("MANAGER_NOTICE_FUND_IDENTITY")
    date_pattern = r"\d{4}年\d{1,2}月\d{1,2}日"
    announced = re.search(r"公告送出日期[:：](" + date_pattern + ")", text)
    if announced is None:
        matches = re.findall(date_pattern, text)
        if not matches:
            raise ValueError("MANAGER_NOTICE_PUBLICATION_UNKNOWN")
        paper_date = chinese_date(matches[-1])
    else:
        paper_date = chinese_date(announced.group(1))
    constraints = [paper_date]
    cms = {}
    for key in ["activationDate", "publishDate", "modificationDate"]:
        if document["catalog"].get(key):
            stamp = datetime.fromtimestamp(int(document["catalog"][key]) / 1000, ZONE).isoformat()
            cms[key] = stamp
            constraints.append(stamp)
    events = []
    for name, begin in re.findall(r"新任基金经理姓名(雷涛|陆阳|揭诗琪)任职日期(" + date_pattern + ")", text):
        events.append({"kind": "APPOINT", "name": name, "effective_date": chinese_date(begin)})
    leaving = re.search(r"离任基金经理姓名(吴志鹏|揭诗琪|雷涛|张铮烁).*?离任日期(" + date_pattern + ")", text)
    if leaving:
        events.append({"kind": "DEPART", "name": leaving.group(1), "effective_date": chinese_date(leaving.group(2))})
    if "暂停履行" in text:
        events.append({"kind": "PAUSE_DUTIES", "name": "揭诗琪", "effective_date": paper_date})
    elif "恢复履行" in text:
        effective = re.search(r"自(" + date_pattern + r")起恢复履行", text)
        if not effective:
            raise ValueError("MANAGER_RESUME_EFFECTIVE_UNKNOWN")
        events.append({"kind": "RESUME_DUTIES", "name": "揭诗琪", "effective_date": chinese_date(effective.group(1))})
    if not events:
        raise ValueError("MANAGER_NOTICE_UNSUPPORTED_SEMANTICS")
    return {
        "raw_sha256": document["receipt"]["sha256"],
        "declared_publication_date": paper_date,
        "available_at": available_at(*constraints),
        "cms_time_constraints": cms,
        "events": events,
        "version_risk": "CURRENT_ARCHIVE_NO_ORIGINAL_FIRST_SEEN_PROOF",
        "text_identity_verified": True,
    }


def state_extensions(report_state: dict, notices: list[dict], target: str) -> tuple[dict, dict]:
    """按U截止的已披露任职集合计算；暂停履职保留任职，离任只在其版本可得后生效。"""
    cutoff = at0800(target)
    if datetime.fromisoformat(report_state["available_at"]) > cutoff:
        raise ValueError("REPORT_STATE_NOT_AVAILABLE")
    roster = {
        m["name"]: m["begin_date"]
        for m in report_state["managers"]
        if m["begin_date"] <= target and (not m["end_date"] or target < m["end_date"])
    }
    applied = []
    # 旧期间的迟到公告不能覆盖更晚报告期已明确的任职集合。
    events = [
        (event, notice)
        for notice in notices
        if datetime.fromisoformat(notice["available_at"]) <= cutoff
        for event in notice["events"]
        if report_state["report_end"] < event["effective_date"] <= target
    ]
    for event, notice in sorted(events, key=lambda x: (x[0]["effective_date"], x[1]["available_at"], x[0]["name"])):
        if event["kind"] == "APPOINT":
            roster[event["name"]] = event["effective_date"]
        elif event["kind"] == "DEPART":
            roster.pop(event["name"], None)
        # 暂停/恢复是履职状态，不是任免；不删除在任经理或重新计算任期。
        applied.append(
            {
                "kind": event["kind"],
                "name": event["name"],
                "raw_sha256": notice["raw_sha256"],
                "available_at": notice["available_at"],
                "effective_date": event["effective_date"],
            }
        )
    tenure = [float((date.fromisoformat(target) - date.fromisoformat(start)).days) for start in roster.values()]
    values = {
        "c_share_net_assets_log1p": math.log1p(float(report_state["financial"]["c_share_net_assets_cny"])),
        "fund_age_days": float(
            (date.fromisoformat(target) - date.fromisoformat(report_state["fund_inception_date"])).days
        ),
        "manager_min_tenure_days": min(tenure) if tenure else None,
        "manager_count": float(len(roster)) if roster else None,
    }
    return values, {
        "report_raw_sha256": report_state["raw_sha256"],
        "report_available_at": report_state["available_at"],
        "manager_roster": roster,
        "applied_notices": applied,
        "fund_inception_date": report_state["fund_inception_date"],
        "c_share_added_date": report_state["c_share_added_date"],
        "manager_basis": "LATEST_AVAILABLE_REPORT_PLUS_AVAILABLE_POST_PERIOD_APPOINTMENT_NOTICES",
    }


def build_revision_v2(root: Path = DEFAULT_ROOT) -> dict:
    """在R1旁生成F修订输入；不发布ready/active、不读取标签、不覆盖R1原件。"""
    revision = root / "handoff/data/revisions/v2"
    if (revision / "ready.json").exists() or (root / "handoff/data/active-revision.json").exists():
        raise ValueError("REVISION_ALREADY_PUBLISHED")
    r1_ready = read(root / "handoff/data/ready.json")
    for relative, expected in r1_ready["artifacts"].items():
        _bound_file(root, relative, expected)
    for source, expected in r1_ready["code_sha256"].items():
        if sha(revision / "baseline-code" / Path(source).name) != expected:
            raise ValueError("R1_CODE_COPY_NOT_PRESERVED")
    snapshot = Snapshot(revision)
    bundle = read(root / "snapshot/normalized-inputs.json")
    source_manifest = read(root / "source-manifest.json")
    original_by_hash = {s["sha256"]: root / s["snapshot_path"] for s in source_manifest["sources"]}
    report_states = {}
    for report in bundle["reports"]:
        original = original_by_hash[report["raw"]["sha256"]]
        state = parse_report_state(report, pdf_pages(original))
        report_states[state["raw_sha256"]] = state
    if {s["fund_inception_date"] for s in report_states.values()} != {"2015-06-19"}:
        raise ValueError("FUND_INCEPTION_CONFLICT")
    if {s["c_share_added_date"] for s in report_states.values() if s["c_share_added_date"]} != {"2015-11-16"}:
        raise ValueError("C_SHARE_INCEPTION_CONFLICT")
    # 已在独立补核中定位的20条目录，仅选2023起且影响本次在任状态的公告。
    review = read(root / "handoff/data/excluded-f-fields-review.json")
    notices_by_hash = {}
    notice_references = []
    for entry in review["manager_notice_catalogue"]:
        if entry["publishedDate"] < "2023-01-01":
            continue
        name = entry["id"].removeprefix("fund-") + ".json"
        relative = Path("supplement/documents") / name
        live = RESEARCH / "materials-live" / relative
        source = live if live.exists() else RESEARCH / relative
        document = snapshot.read(source)
        receipt = document["receipt"]
        raw_path = Path(receipt["raw_path"]) if receipt.get("raw_path") else RESEARCH / receipt["file"]
        raw_copy = snapshot.copy(raw_path, receipt["sha256"])
        notice = parse_manager_notice(document, pdf_pages(raw_copy, limit=6))
        notice_references.append(
            {
                "document_id": entry["id"],
                "source_path": str(source),
                "raw_sha256": receipt["sha256"],
                "available_at": notice["available_at"],
            }
        )
        existing = notices_by_hash.get(receipt["sha256"])
        if existing:
            # 重复目录同一原件，保留更晚的约束而非挑最有利的较早条目。
            existing["available_at"] = max(existing["available_at"], notice["available_at"])
        else:
            notices_by_hash[receipt["sha256"]] = notice
    notices = list(notices_by_hash.values())
    source_increment = {
        "r1_source_manifest_sha256": sha(root / "source-manifest.json"),
        "new_sources": list(snapshot.sources.values()),
        "report_states": report_states,
        "manager_notices": notices,
        "notice_references": notice_references,
        "semantics": {
            "fund_age_days": "CALENDAR_DAYS_SINCE_FUND_CONTRACT_2015_06_19",
            "c_share_added_date": "IDENTITY_METADATA_ONLY_2015_11_16",
            "manager_count": "KNOWN_APPOINTED_NOT_ACTIVE_DUTY_COUNT",
            "manager_tenure": "CALENDAR_DAYS_SINCE_APPOINTMENT_NOT_SECURITIES_CAREER",
            "pause_resume": "DOES_NOT_TERMINATE_OR_RESTART_APPOINTMENT",
            "available_at": "MAX_DECLARED_PUBLICATION_AND_CMS_PUBLICATION_REVISION_CONSTRAINTS",
        },
    }
    save_new(revision / "source-increment.json", source_increment)
    payload = copy.deepcopy(read(root / "handoff/data/inputs.json"))
    row_evidence = []
    for row in payload["rows"]:
        row["raw_F"] = dict(zip(payload["group_columns"]["F"], row["groups"]["F"], strict=True))
        state = report_states[row["report"]["raw_sha256"]]
        extra, evidence = state_extensions(state, notices, row["target_date"])
        row["raw_F"].update(extra)
        row["extension_source_digest"] = digest(source_increment)
        row_evidence.append({"target_date": row["target_date"], **evidence})
    decision = choose_optional(payload["rows"], F, "raw_F")
    payload["group_columns"]["F"] = decision["columns"]
    for row in payload["rows"]:
        raw = row.pop("raw_F")
        values = [raw[c] for c in decision["columns"]]
        ok = decision["enabled"] and all(v is not None for v in values)
        row["groups"]["F"] = values if ok else None
        row["availability"]["F"] = {"available": ok, "reasons": [] if ok else ["OPTIONAL_FIELD_MISSING"]}
    payload["optional_decisions"]["F"] = decision
    payload["candidate_columns"] = {
        c: [name for g in groups for name in payload["group_columns"][g]] for c, groups in CANDIDATE_GROUPS.items()
    }
    payload["input_digest"] = digest(payload["rows"])
    payload["revision"] = "v2"
    save_new(revision / "inputs.json", payload)
    save_new(revision / "row-state-evidence.json", row_evidence)
    save_new(
        revision / "feature-spec.json",
        {
            "version": VERSION,
            "revision": "v2",
            "group_columns": payload["group_columns"],
            "candidate_columns": payload["candidate_columns"],
            "candidate_group_order": CANDIDATE_GROUPS,
            "added_field_semantics": source_increment["semantics"],
        },
    )
    save_new(revision / "optional-feature-decisions.json", payload["optional_decisions"])
    coverage = copy.deepcopy(read(root / "feature-coverage.json"))
    coverage["revision"] = "v2"
    coverage["group_available"]["F"] = sum(row["groups"]["F"] is not None for row in payload["rows"])
    coverage["candidate_input_available"]["F"] = sum(
        all(row["groups"][g] is not None for g in CANDIDATE_GROUPS["F"]) for row in payload["rows"]
    )
    coverage["f_columns"] = decision["columns"]
    coverage["f_candidate_dimension"] = len(payload["candidate_columns"]["F"])
    save_new(revision / "feature-coverage.json", coverage)
    return coverage


def _bound_file(root: Path, path: str, expected: str) -> Path:
    p = Path(path)
    p = p.resolve() if p.is_absolute() else (root / p).resolve()
    if not p.is_relative_to(root.resolve()) or sha(p) != expected:
        raise ValueError("LABEL_GATE_EVIDENCE_INVALID")
    return p


def classify_label(base, target) -> str:
    """仅在标签接口授权通过后调用；十进制精确比较，不使用展示收益舍入。"""
    a, b = Decimal(str(base)), Decimal(str(target))
    if not a.is_finite() or not b.is_finite() or a <= 0 or b <= 0:
        raise ValueError("INVALID_LABEL_NAV")
    return "UP" if b > a else "DOWN" if b < a else "FLAT"


def load_labels(
    root: Path | str, target_dates: list[str], *, stage: str, purpose: str, protocol_sha256: str, gate_path: Path | str
) -> dict:
    """分阶段读取真实标签；函数无写入，B须在调用前记录访问账本。

    训练同时检查目标日期边界和答案成熟时间。2026评价先验证冻结凭据，FINAL需先有
    固定检验结果；gate路径和所有凭据只能位于本实验目录。人工测试可用临时根目录。
    """
    root = Path(root)
    if stage not in TRAIN_END or purpose not in ["train", "evaluate"]:
        raise ValueError("INVALID_LABEL_STAGE")
    if sha(root / "protocol.json") != protocol_sha256:
        raise ValueError("PROTOCOL_NOT_FROZEN")
    gp = Path(gate_path).resolve()
    if not gp.is_relative_to(root.resolve()):
        raise ValueError("LABEL_GATE_OUTSIDE_EXPERIMENT")
    gate = read(gp)
    if any(
        gate.get(k) != v for k, v in {"stage": stage, "purpose": purpose, "protocol_sha256": protocol_sha256}.items()
    ):
        raise ValueError("LABEL_GATE_STAGE_MISMATCH")
    if len(target_dates) != len(set(target_dates)) or not set(target_dates).issubset(gate["allowed_target_dates"]):
        raise ValueError("LABEL_GATE_DATE_NOT_ALLOWED")
    if purpose == "train":
        cutoff = datetime.fromisoformat(gate["fit_cutoff"])
        if cutoff.tzinfo is None or any(not "2024-01-02" <= d <= TRAIN_END[stage] for d in target_dates):
            raise ValueError("TRAIN_LABEL_DATE_BOUNDARY")
        if stage != "FINAL" and cutoff != datetime.fromisoformat(FIT_CUTOFF[stage]):
            raise ValueError("TRAIN_LABEL_CUTOFF_BOUNDARY")
        if stage == "FINAL":
            _bound_file(root, gate["test_evaluation_path"], gate["test_evaluation_sha256"])
            # FINAL不能以调用时钟偷偷放宽冻结后的标签成熟边界。
            if cutoff != datetime.fromisoformat(read(root / "handoff/data/freeze-metadata.json")["frozen_at"]):
                raise ValueError("FINAL_FREEZE_CUTOFF_BOUNDARY")
    else:
        if stage not in EVAL_RANGE or any(not EVAL_RANGE[stage][0] <= d <= EVAL_RANGE[stage][1] for d in target_dates):
            raise ValueError("EVALUATION_LABEL_DATE_BOUNDARY")
        cutoff = datetime.fromisoformat(read(root / "handoff/data/freeze-metadata.json")["frozen_at"])
        if stage == "T":
            _bound_file(root, gate["prediction_freeze_path"], gate["prediction_freeze_sha256"])
    rows = {r["target_date"]: r for r in load_inputs(root)["rows"]}
    if not set(target_dates).issubset(rows):
        raise ValueError("LABEL_DATE_OUTSIDE_FROZEN_CALENDAR")
    nav = read(root / "snapshot/nav-by-date.json")
    labels, excluded, maturity = {}, {}, {}
    for target in target_dates:
        row = rows[target]
        base = row["base_date"]
        mature = max(
            datetime.fromisoformat(nav[target]["available_at"]), datetime.fromisoformat(nav[base]["available_at"])
        )
        maturity[target] = mature.isoformat()
        if mature > cutoff:
            excluded[target] = "LABEL_NOT_MATURE_AT_CUTOFF"
            continue
        labels[target] = classify_label(nav[base]["unit_nav"], nav[target]["unit_nav"])
    return {
        "stage": stage,
        "purpose": purpose,
        "labels": labels,
        "excluded": excluded,
        "maturity": maturity,
        "source_digest": sha(root / "snapshot/nav-by-date.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立只读输入快照；不执行真实标签读取或训练")
    parser.add_argument("command", choices=["build", "coverage"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = build_snapshot(args.root) if args.command == "build" else read(args.root / "feature-coverage.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
