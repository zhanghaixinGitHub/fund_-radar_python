"""输出可复核覆盖清单及保留原方案的现场证据，不向普通基金页面透出研究细节。"""

from collections import Counter, defaultdict
from datetime import datetime, time

from sqlalchemy import text

from app.db.session import get_engine
from app.services.direction_1d_protocol import ZONE, calendar, digest
from app.services.fund_exposure_common import ROOT, now, read, save
from app.services.fund_exposure_features import load_bundle, select_report
from app.services.fund_exposure_quotes import INDICES


def audit():
    """每家公司列出有效报告期间的所需日期、来源未返回日期和每日指标覆盖。"""
    bundle = load_bundle()
    rs = bundle["reports"]
    sessions, _ = calendar()
    codes = read(ROOT / "universe.json")["codes"]
    basic = {p.stem: read(p) for p in (ROOT / "stock-basic").glob("*.json")}
    coverage = {
        c: {"required": 0, "returned": 0, "missing_dates": [], "turnover_returned": 0, "turnover_missing_dates": []}
        for c in codes
    }
    for i, day in enumerate(sessions[:-1]):
        if day > now().date():
            break
        report = select_report(rs, datetime.combine(sessions[i + 1], time(8), ZONE))
        quotes = bundle["days"].get(str(day), {}).get("rows", {})
        for h in report["holdings"]:
            code = h["stock_code"]
            row = coverage[code]
            row["required"] += 1
            if code in quotes:
                row["returned"] += 1
            else:
                row["missing_dates"].append(str(day))
            b = basic.get(code, {}).get("rows", {}).get(str(day), {})
            if b.get("turnover_rate") is not None:
                row["turnover_returned"] += 1
            else:
                row["turnover_missing_dates"].append(str(day))
    report_periods = Counter((r["report_end"], r["report_type"]) for r in rs)
    expected = [("2020-09-30", "QUARTER"), ("2020-12-31", "QUARTER"), ("2020-12-31", "ANNUAL")]
    for year in range(2021, 2027):
        expected.extend(
            (f"{year}-{md}", "QUARTER") for md in ("03-31", "06-30", "09-30", "12-31") if year < 2026 or md <= "06-30"
        )
        expected.append((f"{year}-06-30", "HALF"))
        if year < 2026:
            expected.append((f"{year}-12-31", "ANNUAL"))
    names = defaultdict(list)
    for r in rs:
        for h in r["holdings"]:
            names[h["stock_code"]].append(
                {"name": h["stock_name"], "report_end": r["report_end"], "available_at": r["available_at"]}
            )
    for code in codes:
        coverage[code]["historical_names"] = names[code]
        coverage[code]["cause_of_nonreturn"] = "UNVERIFIED_NOT_ASSUMED_SUSPENSION_OR_ZERO"
    context_path = ROOT / "supplement" / "stock-context.json"
    known_suspensions = {}
    if context_path.exists():
        for item in read(context_path)["suspensions"]:
            # suspend_type S 表示停牌，R 是复牌；不能仅因出现一行就判断全天停牌。
            known_suspensions[item["stock_code"]] = {
                datetime.strptime(r["trade_date"], "%Y%m%d").date().isoformat(): {
                    "suspend_type": r["suspend_type"],
                    "suspend_timing": r.get("suspend_timing"),
                    "receipt_hash": item["receipt"]["sha256"],
                }
                for r in item["rows"]
                if r["suspend_type"] == "S"
            }
    confirmed_missing = 0
    for code, item in coverage.items():
        item["verified_suspension_dates"] = {
            d: known_suspensions.get(code, {})[d] for d in item["missing_dates"] if d in known_suspensions.get(code, {})
        }
        confirmed_missing += len(item["verified_suspension_dates"])
        if item["missing_dates"] and len(item["verified_suspension_dates"]) == len(item["missing_dates"]):
            item["cause_of_nonreturn"] = "PROVIDER_CONFIRMED_SUSPENSION_PRICE_REMAINS_MISSING"
    with get_engine().connect() as c:
        models = [
            dict(r)
            for r in c.execute(
                text(
                    "SELECT model_id,metadata->>'protocol' AS protocol,group_id "
                    "FROM direction_1d_model ORDER BY model_id"
                )
            ).mappings()
        ]
    study_path = ROOT / "study-current.json"
    study = read(ROOT / read(study_path)["file"]) if study_path.exists() else None
    stock_rows = sum(len(d["rows"]) for d in bundle["days"].values())
    result = {
        "at": now().isoformat(),
        "reports": len(rs),
        "missing_reports": [list(e) for e in expected if e not in report_periods],
        "duplicate_period_types": [list(k) for k, v in report_periods.items() if v > 1],
        "holding_rows": sum(r["holding_count"] for r in rs),
        "historical_stocks": len(codes),
        "stock_days": len(bundle["days"]),
        "stock_rows": stock_rows,
        "daily_basic_rows": sum(len(b["rows"]) for b in basic.values()),
        "basic_codes": len(basic),
        "coverage": coverage,
        "needed_stock_dates": sum(r["required"] for r in coverage.values()),
        "missing_stock_dates": sum(len(r["missing_dates"]) for r in coverage.values()),
        "missing_stock_dates_with_suspension_evidence": confirmed_missing,
        "missing_turnover_dates": sum(len(r["turnover_missing_dates"]) for r in coverage.values()),
        "indices": {
            c: {"name": INDICES[c], "dates": len(v["rows"]), "first": min(v["rows"]), "last": max(v["rows"])}
            for c, v in bundle["indices"].items()
        },
        "industry_policy_zh": (
            "保留报告当时的宽行业分类；用披露股票组合反映行业影响。"
            "没有逐股票历史细行业关系，不填造、不用今天的行业标签回填。"
        ),
        "model_registry_hash": digest(models),
        "model_registry_unchanged": digest(models) == study["model_registry_hash"] if study else None,
        "new_candidate_predictions": 0,
        "live_input_only_records": len(list((ROOT / "forward-inputs").glob("*.json"))),
        "new_purchase_cny": 0,
    }
    save(ROOT / "coverage.json", result, replace=True)
    return {k: v for k, v in result.items() if k != "coverage"}
