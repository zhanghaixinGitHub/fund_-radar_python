"""002112 资料任务内的固定十基金历史补充；采集、核对与训练严格分开。

共享任务队列及原持仓维护锁由调用方持有。历史研究文件只读，参照基金各自保存
报告、净值和版本；2025 标签及回算 2026 标签不进入本模块。
"""

import hashlib
import json
from collections import Counter, OrderedDict, defaultdict
from datetime import date, datetime, time
from pathlib import Path

from sqlalchemy import text

from app.db.session import get_engine
from app.integrations.public_fund_reports import PEERS, STORE, ReportClient, acquire_one, catalog
from app.integrations.tushare_sprint_stock_breadth_v2 import FIELDS, parse
from app.services.direction_1d_protocol import ZONE, calendar, digest
from app.services.direction_1d_training import build_samples, select_fit
from app.services.fund_exposure_common import ROOT, now, read, save
from app.services.fund_exposure_features import calculate
from app.services.fund_exposure_quotes import INDICES, Provider, permission, safe_error
from app.services.fund_exposure_supplement import SupplementProvider
from app.services.fund_materials_store import versioned_save


def expected_periods(code):
    """新设 C 类不伪造成立前的净值；017493 从其设立季度开始检查报告。"""
    start = "2022-12-31" if code == "017493" else "2020-09-30"
    periods = []
    for year in range(2020, 2025):
        for md in ("03-31", "06-30", "09-30", "12-31"):
            day = f"{year}-{md}"
            if start <= day <= "2024-09-30":
                periods.append((day, "QUARTER"))
        if start <= f"{year}-06-30" <= "2024-09-30":
            periods.append((f"{year}-06-30", "HALF"))
        if start <= f"{year}-12-31" <= "2023-12-31":
            periods.append((f"{year}-12-31", "ANNUAL"))
    return periods


def sync_reports(check_id, progress):
    result = {"created": 0, "updated": 0, "skipped": 0, "errors": [], "funds": {}}
    client = ReportClient(check_id)
    try:
        for ordinal, code in enumerate(PEERS):
            progress(ordinal, len(PEERS), code, f"补充 {code} {PEERS[code]} 的历史报告")
            manifest_path = STORE / code / "manifest.json"
            old = read(manifest_path) if manifest_path.exists() else {"documents": {}}
            documents, errors = {}, []
            try:
                entries, receipts = catalog(client, code)
                for i, entry in enumerate(entries):
                    progress(ordinal, len(PEERS), code, f"{code} 历史报告 {i + 1}/{len(entries)}")
                    try:
                        parsed, path, _ = acquire_one(client, code, entry)
                        previous = old["documents"].get(entry["ID"])
                        key = (
                            "skipped"
                            if previous and previous["sha256"] == parsed["raw"]["sha256"]
                            else "updated"
                            if previous
                            else "created"
                        )
                        result[key] += 1
                        documents[entry["ID"]] = {
                            "file": path.relative_to(STORE).as_posix(),
                            "sha256": parsed["raw"]["sha256"],
                            "report_end": parsed["report_end"],
                            "report_type": parsed["report_type"],
                            "listing_status": "LISTED",
                            "holding_count": parsed["holding_count"],
                            "previous_versions": list(
                                dict.fromkeys(
                                    (previous or {}).get("previous_versions", [])
                                    + (
                                        [previous["file"]]
                                        if previous and previous["file"] != path.relative_to(STORE).as_posix()
                                        else []
                                    )
                                )
                            ),
                        }
                    except Exception as exc:
                        client.invalidate_body(entry["ID"])
                        # 新一轮外呼失败不抹去此前已经核验的成果；新版本不能倒填历史。
                        previous = old["documents"].get(entry["ID"])
                        if previous:
                            documents[entry["ID"]] = {**previous, "recheck_status": "FAILED"}
                        errors.append({"fund_code": code, "article_id": entry["ID"], "reason": safe_error(exc)})
                listed = {e["ID"] for e in entries}
                disappeared = {
                    key: {**doc, "listing_status": "NOT_LISTED_ON_RECHECK"}
                    for key, doc in old["documents"].items()
                    if key not in listed
                }
                if disappeared:
                    errors.append(
                        {"fund_code": code, "reason": "REPORT_CATALOG_DISAPPEARED", "article_ids": list(disappeared)}
                    )
                present = {(d["report_end"], d["report_type"]) for d in documents.values()}
                missing = [list(p) for p in expected_periods(code) if p not in present]
                if missing:
                    errors.append({"fund_code": code, "reason": "REPORT_PERIODS_MISSING", "periods": missing})
                manifest = {
                    "fund_code": code,
                    "at": now().isoformat(),
                    "documents": documents,
                    "not_listed": disappeared,
                    "catalog": entries,
                    "catalog_receipts": receipts,
                    "errors": errors,
                    "check_id": check_id,
                    "missing_periods": missing,
                    "last_success_at": now().isoformat() if not errors else old.get("last_success_at"),
                }
                versioned_save(manifest_path, manifest)
            except Exception as exc:
                errors.append({"fund_code": code, "reason": safe_error(exc)})
            result["errors"].extend(errors)
            result["funds"][code] = {"reports": len(documents), "errors": errors}
    finally:
        result["new_requests"] = client.count
        client.close()
    return result


def nav_history():
    """只读提取 2021—2024 净值，日期和来源版本全部保留，不改正式净值表。"""
    funds = []
    with get_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        connection.execute(text("SET LOCAL statement_timeout='15000'"))
        source = (
            connection.execute(
                text(
                    "SELECT source_id,enabled,authorized_api_names FROM source_registry "
                    "WHERE source_code='TUSHARE_PRO_FUND'"
                )
            )
            .mappings()
            .one()
        )
        if not source["enabled"] or "fund_nav" not in source["authorized_api_names"]:
            raise ValueError("EXPOSURE_SOURCE_UNAVAILABLE")
        for code in PEERS:
            rows = connection.execute(
                text("""SELECT nav_date,unit_nav,ann_date,content_hash FROM nav_daily
                WHERE fund_code=:code AND source_id=:source AND nav_date BETWEEN '2021-01-01' AND '2024-12-31'
                ORDER BY nav_date"""),
                {"code": code, "source": source["source_id"]},
            ).mappings()
            values = [
                {
                    "date": str(r["nav_date"]),
                    "nav": str(r["unit_nav"]),
                    "ann_date": str(r["ann_date"]) if r["ann_date"] else None,
                    "source_hash": r["content_hash"],
                }
                for r in rows
            ]
            if not values:
                raise ValueError("EXPOSURE_PEER_NAV_MISSING")
            value = {"fund_code": code, "product_family_id": "PEER_" + code, "group_id": "CN_MIXED", "rows": values}
            versioned_save(STORE / code / "nav.json", value)
            funds.append(value)
    return {"funds": funds, "kind": "HISTORICAL_RECONSTRUCTION"}


class QuoteDays:
    """按日加载已保存的全市场原文；有限缓存，不复制或重下全部股票行情。

    SH/SZ 沿用原始核验器；BJ/HK 的历史身份或报价口径尚未获本方案核验，不能借用
    其他市场价格。缺失原文件才调用已授权日线接口，原文件存在但缺某股时另记缺口。
    """

    def __init__(self, *, allow_fetch=True):
        self.allow_fetch = allow_fetch
        self.index, self.provider = {}, None
        self.cache = OrderedDict()
        source = permission()
        if not {"daily", "index_daily"}.issubset(source["authorized_api_names"]):
            raise ValueError("EXPOSURE_API_NOT_AUTHORIZED")

    def get(self, day, default=None):
        if day not in self.cache:
            self.cache[day] = self._load(day)
        self.cache.move_to_end(day)
        while len(self.cache) > 64:
            self.cache.popitem(last=False)
        return self.cache[day]

    def _load(self, day):
        path = ROOT / "stock-days" / (day + ".json")
        receipt = read(path)["receipt"] if path.exists() else None
        rawpath = (
            (Path(receipt["raw_path"]) if receipt.get("raw_path") else ROOT / receipt["file"]) if receipt else None
        )
        if rawpath is None or not rawpath.exists():
            if not self.allow_fetch:
                raise ValueError("TRAINING_QUOTE_EVIDENCE_MISSING")
            self.provider = self.provider or Provider()
            value, receipt = self.provider.query("daily", {"trade_date": day.replace("-", "")}, FIELDS)
            rawpath = ROOT / receipt["file"]
        raw = rawpath.read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
            raise ValueError("EXPOSURE_RAW_HASH_MISMATCH")
        if datetime.fromisoformat(receipt["expires_at"]) <= now():
            raise ValueError("EXPOSURE_EVIDENCE_EXPIRED")
        parse(raw, day)
        data = json.loads(raw)["data"]
        rows = {
            item[0]: dict(zip(FIELDS[2:], item[2:], strict=True))
            for item in data["items"]
            if item[0].endswith((".SH", ".SZ"))
        }
        self.index[day] = {"raw_path": str(rawpath), "receipt": receipt, "stock_rows": len(rows)}
        return {"rows": rows, "receipt": receipt}


def build_coverage(progress):
    """按相同截止时刻拼齐输入，只生成独立检查资料，不拟合、登记或启用模型。"""
    history = nav_history()
    samples = build_samples(history)
    navs = {f["fund_code"]: {r["date"]: r for r in f["rows"]} for f in history["funds"]}
    sessions, _ = calendar()
    positions = {str(d): i for i, d in enumerate(sessions)}
    quote_days = QuoteDays()
    indices = {code: read(ROOT / "indices" / (code + ".json")) for code in INDICES}
    reports = {}
    for code in PEERS:
        path = STORE / code / "manifest.json"
        manifest = read(path) if path.exists() else {"documents": {}}
        files = {f for d in manifest["documents"].values() for f in [d["file"], *d.get("previous_versions", [])]}
        reports[code] = [read(STORE / f) for f in sorted(files)]
    rows, excluded, gaps = [], [], defaultdict(set)
    for i, row in enumerate(samples):
        code, target, base = row["fund_code"], row["u"], row["t"]
        if i % 100 == 0:
            progress(i, len(samples), code, f"按历史日期核对持仓和行情 {i}/{len(samples)}")
        position = positions[base]
        needed = [navs[code][str(d)] for d in sessions[position - 60 : position + 1]]
        reason = None
        if any(not n["ann_date"] for n in needed):
            reason = "NAV_PUBLICATION_UNKNOWN"
        elif any(n["ann_date"] > target for n in needed):
            reason = "NAV_NOT_PUBLIC_BY_TARGET"
        if reason:
            excluded.append({"fund_code": code, "base": base, "target": target, "reason": reason})
            continue
        try:
            exposure = calculate(
                {"reports": reports[code], "days": quote_days, "indices": indices},
                date.fromisoformat(base),
                datetime.combine(date.fromisoformat(target), time(8), ZONE),
            )
            if exposure["holdings_features"] is None or exposure["market_features"] is None:
                reason = (
                    "REPORT_NO_STOCK_EXPOSURE"
                    if exposure["disclosed_nav_weight"] == 0
                    else "QUOTES_OR_REPORT_INCOMPLETE"
                )
                excluded.append(
                    {
                        "fund_code": code,
                        "base": base,
                        "target": target,
                        "reason": reason,
                        "report_end": exposure["report_end"],
                        "missing": exposure["missing"],
                    }
                )
                for missing in exposure["missing"]:
                    gaps[missing["stock_code"]].update(missing["dates"])
                continue
            rows.append(
                {
                    **row,
                    "holdings_features": exposure["holdings_features"],
                    "market_features": exposure["market_features"],
                    "report_raw_sha256": exposure["report_raw_sha256"],
                    "report_end": exposure["report_end"],
                    "report_available_at": exposure["report_available_at"],
                    "quote_hashes": exposure["quote_hashes"],
                    "training_eligible": False,
                }
            )
        except ValueError as exc:
            if str(exc) != "EXPOSURE_NO_AVAILABLE_REPORT":
                # 原文损坏、过期或接口故障属于采集失败，不能吞掉后仅报样本数量为零。
                raise
            excluded.append({"fund_code": code, "base": base, "target": target, "reason": safe_error(exc)})
    fit = select_fit(rows, datetime(2024, 1, 1, tzinfo=ZONE))

    def counts(values):
        return {
            "rows": len(values),
            "dates": len({r["t"] for r in values}),
            "directions": {k: sum(r["actual_direction"] == k for r in values) for k in ("UP", "FLAT", "DOWN")},
        }

    gap_audit = explain_quote_gaps(gaps, progress)
    result = {
        "at": now().isoformat(),
        "kind": "PEER_DATA_COVERAGE_ONLY_NOT_A_TRAINING_RUN",
        "rows": rows,
        "excluded": excluded,
        "excluded_reasons": dict(Counter(r["reason"] for r in excluded)),
        "fit_504": counts(fit),
        "per_fund": {code: counts([r for r in fit if r["fund_code"] == code]) for code in PEERS},
        "development_2024": counts([r for r in rows if r["u"].startswith("2024")]),
        "quote_gaps": {code: sorted(days) for code, days in gaps.items()},
        "quote_gap_audit": gap_audit,
        "training_eligible": False,
        "training_runs": 0,
        "limitations": [
            "HISTORICAL_AVAILABILITY_RECONSTRUCTED",
            "COHORT_COMPATIBILITY_NOT_APPROVED",
            "PUBLIC_REPRINT_NOT_ISSUER_PDF",
            "BJ_HK_QUOTES_NOT_VERIFIED",
        ],
    }
    versioned_save(STORE / "quote-index.json", {"days": quote_days.index})
    key = digest(result)
    save(STORE / "datasets" / (key + ".json"), result)
    versioned_save(
        STORE / "coverage-current.json",
        {
            "file": f"datasets/{key}.json",
            "at": result["at"],
            "fit_504": result["fit_504"],
            "excluded_reasons": result["excluded_reasons"],
            "quote_gap_stocks": len(gaps),
        },
    )
    return {
        "fit_504": result["fit_504"],
        "excluded_reasons": result["excluded_reasons"],
        "quote_gap_stocks": len(gaps),
        "quote_days": len(quote_days.index),
        "unresolved_quote_stocks": gap_audit["unresolved_stocks"],
        "file": f"datasets/{key}.json",
    }


def explain_quote_gaps(gaps, progress):
    """用已授权的上市日期和停牌证据说明无报价原因；不补零或虚构停牌价格。

    仅询问本次报告实际涉及的缺失股票和日期范围。证据不全、空返回和未经核验的
    市场仍为未完成；已确认未上市或全天停牌的记录排除出候选资料，但不反复拉行情。
    """
    provider, details, unresolved = None, {}, []
    for i, (code, dates) in enumerate(sorted(gaps.items())):
        dates = sorted(dates)
        progress(i, len(gaps), code, f"核对 {code} 缺报价的原因 {i + 1}/{len(gaps)}")
        if not code.endswith((".SH", ".SZ")):
            details[code] = {"dates": dates, "status": "MARKET_NOT_VERIFIED"}
            unresolved.append(code)
            continue
        try:
            if provider is None:
                provider = SupplementProvider()
                provider.scope = {**provider.scope, "maximum_new_requests": 160}
            basic, br = provider.query("stock_basic", {"ts_code": code}, "ts_code,list_date,delist_date")
            suspended, sr = provider.query(
                "suspend_d",
                {"ts_code": code, "start_date": dates[0].replace("-", ""), "end_date": dates[-1].replace("-", "")},
            )

            def source_rows(value, expected_code=code):
                data = value["data"]
                rows = [dict(zip(data["fields"], row, strict=True)) for row in data["items"]]
                if any(r.get("ts_code") != expected_code for r in rows):
                    raise ValueError("REPORT_QUOTE_CONTEXT_IDENTITY_MISMATCH")
                return rows

            basics, pauses = source_rows(basic), source_rows(suspended)
            listed = min((str(r["list_date"]) for r in basics if r.get("list_date")), default="")
            full_pause = {
                str(r["trade_date"]) for r in pauses if r.get("suspend_type") == "S" and not r.get("suspend_timing")
            }
            explained = {
                d: "BEFORE_LISTING" if listed and d.replace("-", "") < listed else "FULL_DAY_SUSPENSION"
                for d in dates
                if (listed and d.replace("-", "") < listed) or d.replace("-", "") in full_pause
            }
            missing = [d for d in dates if d not in explained]
            details[code] = {
                "dates": dates,
                "explained": explained,
                "unresolved_dates": missing,
                "basic_receipt": br,
                "suspension_receipt": sr,
                "status": "UNRESOLVED" if missing else "KNOWN_NO_QUOTE",
            }
            if missing:
                unresolved.append(code)
        except Exception as exc:
            details[code] = {"dates": dates, "status": "FETCH_FAILED", "reason": safe_error(exc)}
            unresolved.append(code)
    audit = {"stocks": details, "unresolved_stocks": unresolved, "new_requests": provider.count if provider else 0}
    versioned_save(STORE / "quote-gap-audit.json", audit)
    return audit


def supplement(*, check_id, progress=lambda *args: None):
    """资料主任务中的同步步骤；失败也保留各基金成功文件，下一次只补未完成部分。"""
    plan = {
        "version": "PEER_MATERIALS_V1",
        "funds": PEERS,
        "report_start": "2020-09-30",
        "report_end": "2024-09-30",
        "nav_start": "2021-01-01",
        "nav_end": "2024-12-31",
        "new_purchase_cny": 0,
        "train_automatically": False,
    }
    plan_path = STORE / "plan.json"
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError("REPORT_PEER_PLAN_CHANGED")
    if not plan_path.exists():
        save(plan_path, plan)
    result = sync_reports(check_id, progress)
    progress(0, 1, "002112", "报告已保存，正在核对参照基金的历史净值和股票行情")
    try:
        result["coverage"] = build_coverage(progress)
        if result["coverage"]["unresolved_quote_stocks"]:
            result["errors"].append(
                {"reason": "REPORT_PEER_QUOTES_INCOMPLETE", "stocks": result["coverage"]["unresolved_quote_stocks"]}
            )
    except Exception as exc:
        result["errors"].append({"reason": safe_error(exc), "scope": "PEER_QUOTE_COVERAGE"})
    result["at"] = now().isoformat()
    result["check_id"] = check_id
    result["training_runs"] = 0
    versioned_save(STORE / "result.json", result)
    return result
