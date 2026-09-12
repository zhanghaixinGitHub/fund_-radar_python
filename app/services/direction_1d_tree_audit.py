"""浅层树训练前只读审计：单位净值、公告、现金分红和旧模型错误，不据成绩删题。"""

from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import bindparam, text

from app.db.session import get_engine
from app.services import direction_1d_sector_study as sector
from app.services.direction_1d_protocol import ZONE, digest, input_days
from app.services.direction_1d_training import read, write_new
from app.services.direction_training_artifacts import file_hash


def collect(base: Path, output: Path) -> dict:
    """仅查冻结25只的2021—2024公开记录；数据库事务显式只读，未调用外部行情接口。"""
    sector.verify(base)
    spec = read(base / "study.json")
    codes = spec["fund_codes"]
    with get_engine().connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
        c.execute(text("SET TRANSACTION READ ONLY"))
        source = (
            c.execute(
                text("""SELECT source_id,source_code,enabled,authorization_verified_at,
              authorized_api_names,retention_days FROM source_registry WHERE source_code='TUSHARE_PRO_FUND'""")
            )
            .mappings()
            .one()
        )
        if (
            not source["enabled"]
            or not source["authorization_verified_at"]
            or "fund_div" not in source["authorized_api_names"]
        ):
            raise ValueError("TREE_AUDIT_SOURCE_NOT_AUTHORIZED")
        navs = (
            c.execute(
                text("""SELECT fund_code,nav_date,ann_date,unit_nav,accumulated_nav,adjusted_nav,
              accumulated_dividend,content_hash,source_published_at FROM nav_daily
              WHERE source_id=:source AND fund_code IN :codes AND nav_date BETWEEN '2021-01-01' AND '2024-12-31'
              ORDER BY fund_code,nav_date""").bindparams(bindparam("codes", expanding=True)),
                {"source": source["source_id"], "codes": codes},
            )
            .mappings()
            .all()
        )
        dividends = (
            c.execute(
                text("""SELECT fund_code,ann_date,implementation_ann_date,process_status,ex_date,nav_ex_date,
              cash_dividend,base_unit,content_hash FROM fund_dividend WHERE source_id=:source AND fund_code IN :codes
              AND (ex_date BETWEEN '2021-01-01' AND '2024-12-31' OR nav_ex_date BETWEEN '2021-01-01' AND '2024-12-31')
              ORDER BY fund_code,ex_date,nav_ex_date,content_hash""").bindparams(bindparam("codes", expanding=True)),
                {"source": source["source_id"], "codes": codes},
            )
            .mappings()
            .all()
        )
        observed_at = c.execute(text("SELECT clock_timestamp()")).scalar_one()
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "observed_at": observed_at,
        "source": dict(source),
        "fund_codes": codes,
        "base_study_hash": file_hash(base / "study.json"),
        "navs": [dict(r) for r in navs],
        "dividends": [dict(r) for r in dividends],
        "database_writes": 0,
        "api_calls": 0,
    }
    write_new(output / "public-db-snapshot.json", payload)
    result = analyze(base, read(output / "public-db-snapshot.json"))
    write_new(output / "audit.json", result)
    write_new(
        output / "audit-receipt.json",
        {
            "at": datetime.now(ZONE).isoformat(),
            "audit_hash": digest(result),
            "files": {name: file_hash(output / name) for name in ("public-db-snapshot.json", "audit.json")},
            "code_hash": file_hash(Path(__file__)),
        },
    )
    return result


def event_flags(row: dict, events: list[dict]) -> dict:
    """事件仅用于事后审计，不进入训练或改变单位净值答案；公告日不等于真实首次收取时间。"""
    days = {str(d) for d in input_days(date.fromisoformat(row["t"]))}
    target_events, input_events = [], []
    for e in events:
        if e["fund_code"] != row["fund_code"]:
            continue
        dates = {e.get("ex_date"), e.get("nav_ex_date")} - {None}
        if row["u"] in dates:
            target_events.append(e)
        if days & dates:
            input_events.append(e)
    return {
        "target_cash_event": bool(target_events),
        "input_cash_event": bool(input_events),
        "target_event_announced_by_t": any(e.get("ann_date") and e["ann_date"] <= row["t"] for e in target_events),
    }


def adjustment_changes(snapshot: dict) -> list[dict]:
    """复权/单位净值之比变化超过0.1%时列出线索；只交叉核对，不能认定折算或改标签。"""
    previous, result = {}, []
    for row in sorted(snapshot["navs"], key=lambda r: (r["fund_code"], r["nav_date"])):
        code = row["fund_code"]
        prior = previous.get(code)
        previous[code] = row
        if prior is None or not prior["adjusted_nav"] or not row["adjusted_nav"]:
            continue
        a = float(prior["adjusted_nav"]) / float(prior["unit_nav"])
        b = float(row["adjusted_nav"]) / float(row["unit_nav"])
        if a > 0 and abs(b / a - 1) > 0.001:
            result.append(
                {
                    "fund_code": code,
                    "date": row["nav_date"],
                    "ratio_change": b / a - 1,
                    "cash_event_matches": any(
                        e["fund_code"] == code and row["nav_date"] in (e["ex_date"], e["nav_ex_date"])
                        for e in snapshot["dividends"]
                    ),
                }
            )
    return result


def analyze(base: Path, snapshot: dict) -> dict:
    """对照原始hash与计算值，并输出全样本分层；高波动分界只取首个训练窗中位数。"""
    spec, exam = read(base / "study.json"), read(base / "common-exam.json")
    if snapshot["fund_codes"] != spec["fund_codes"] or snapshot["base_study_hash"] != file_hash(base / "study.json"):
        raise ValueError("TREE_AUDIT_SCOPE_OR_BASE_CHANGED")
    codes = set(spec["fund_codes"])
    current = {(r["fund_code"], r["nav_date"]): r for r in snapshot["navs"]}
    if len(current) != len(snapshot["navs"]) or {r["fund_code"] for r in snapshot["navs"]} != codes:
        raise ValueError("TREE_AUDIT_NAV_SCOPE_INVALID")
    history = read(base / "baseline/history.json")
    expected = {(f["fund_code"], r["date"]): r for f in history["funds"] if f["fund_code"] in codes for r in f["rows"]}
    mismatches = []
    for key, r in expected.items():
        now = current.get(key)
        if (
            now is None
            or r["source_hash"] != now["content_hash"]
            or float(r["nav"]) != float(now["unit_nav"])
            or r["ann_date"] != now["ann_date"]
        ):
            mismatches.append({"fund_code": key[0], "date": key[1]})
    if mismatches:
        raise ValueError("TREE_AUDIT_FROZEN_NAV_CHANGED:" + str(mismatches[:3]))
    q1fit = sector.selected_fit(base, "2024Q1")
    volatility_cut = float(np.median([r["x"][3] for r in q1fit]))
    x = np.asarray([sector.vector(r, "SPECIFIC11") for r in q1fit])
    corr = np.corrcoef(x, rowvar=False)
    names = sector.feature_names("SPECIFIC11")
    pairs = sorted(
        (
            {"left": names[i], "right": names[j], "correlation": float(corr[i, j])}
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if np.isfinite(corr[i, j])
        ),
        key=lambda r: -abs(r["correlation"]),
    )[:8]
    predictions = read(base / "main/predictions.json")
    models = read(base / "main/models.json")
    by_key = {(r["fund_code"], r["t"]): r for r in predictions}
    rows, lateness = [], Counter()
    for row in exam:
        p = by_key[row["fund_code"], row["t"]]
        if (
            p["y"] != row["y"]
            or p["u"] != row["u"]
            or abs(sector.predict(models["SPECIFIC11-" + row["quarter"]], [row])[0] - p["scores"]["SPECIFIC11"]) > 1e-12
        ):
            raise ValueError("TREE_AUDIT_LINEAR_SCORE_CHANGED")
        days = [str(d) for d in input_days(date.fromisoformat(row["t"]))]
        latest = max(current[row["fund_code"], d]["ann_date"] or d for d in days)
        if latest > row["u"]:
            raise ValueError("TREE_AUDIT_LATE_INPUT_IN_EXAM")
        lateness[(date.fromisoformat(latest) - date.fromisoformat(row["t"])).days] += 1
        flags = event_flags(row, snapshot["dividends"])
        rows.append(
            {
                **p,
                **flags,
                "input_ann_date": latest,
                "market_state": "MARKET_5D_UP" if row["market_input"]["x"][1] > 0 else "MARKET_5D_NON_UP",
                "volatility_state": "HIGH" if row["x"][3] > volatility_cut else "LOW",
            }
        )
    fields = (
        "target_cash_event",
        "input_cash_event",
        "target_event_announced_by_t",
        "market_state",
        "volatility_state",
        "quarter",
        "fund_code",
    )
    strata = {
        field: {
            str(v): sector.comparison.metrics([r for r in rows if r[field] == v], "SPECIFIC11")
            for v in sorted({r[field] for r in rows})
        }
        for field in fields
    }
    return {
        "kind": "PREFIT_DIAGNOSTIC_NOT_FEATURE_SELECTION",
        "audited_at": datetime.now(ZONE).isoformat(),
        "base_study_hash": snapshot["base_study_hash"],
        "scope": spec["fund_codes"],
        "snapshot_nav_count": len(current),
        "frozen_nav_count": len(expected),
        "nav_mismatch_count": 0,
        "cash_event_records": len(snapshot["dividends"]),
        "source_published_time_count": sum(r["source_published_at"] is not None for r in current.values()),
        "adjusted_nav_present_count": sum(r["adjusted_nav"] is not None for r in current.values()),
        "adjustment_factor_changes": adjustment_changes(snapshot),
        "exam_count": len(exam),
        "input_ann_calendar_days_after_t": dict(lateness),
        "volatility_cut_from_q1_fit": volatility_cut,
        "q1_feature_correlations": pairs,
        "strata": strata,
        "event_annotations": [
            {
                k: r[k]
                for k in ("fund_code", "t", "u", *fields[:3], "input_ann_date", "market_state", "volatility_state")
            }
            for r in rows
        ],
        "requires_data_repair": False,
        "rows_excluded_by_diagnostic": 0,
        "model_fits": 0,
        "limitations": [
            "现金分红仅来自当前已有表，未证明历史首次公告或全部折算事件覆盖",
            "事件与市场状态分层是描述性诊断，不按分数删题或选参数",
            "保留单位净值标签；分红日净值变化不等于投资总回报",
            "当前行与冻结行一致不证明供应商历史首版本正确",
        ],
    }
