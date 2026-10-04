"""封装本轮近期输入验收；仅日期、来源、质量，不生成净值特征或方向标签。

逐日保留失败状态和原算法不兼容的停牌窗口；输出不是训练许可。
所有数据库连接只读，所有产物独占创建，现有缓存只读并核对原始响应。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pypdfium2 as pdfium
from app.integrations.tushare_sprint_stock_breadth_v2 import FIELDS, validate_quote_values
from app.services.direction_1d_protocol import calendar, digest
from app.services.fund_exposure_features import select_report
from app.services.fund_exposure_quotes import INDICES, reports
from sqlalchemy import text

from scripts.fund_002112_recent_acceptance_v1 import board, suspension_state
from scripts.fund_002112_recent_completion_v1 import NAV_DATES, OUT, ROOT, engine, now, read, save, sha


def receipt_file(rc):
    """复用回执必须验证原件摘要；共享缓存可能有不同的原件路径字段。"""
    p = Path(rc["raw_path"]) if rc.get("raw_path") else ROOT / rc["file"]
    if sha(p) != rc["sha256"]:
        raise ValueError("ORIGINAL_HASH_MISMATCH")
    if rc.get("expires_at") and datetime.fromisoformat(rc["expires_at"]) <= datetime.fromisoformat(now()):
        raise ValueError("ORIGINAL_RECEIPT_EXPIRED")
    return p


def verify_report(r):
    """复核份额身份和送出日，承接已通过表内合计核验的解析版本。"""
    p = receipt_file(r["raw"])
    doc = pdfium.PdfDocument(p)
    body = []
    try:
        # 年报和中报前四页通常是封面、提示、目录；份额代码在第 5 页基金基本情况。
        for ix in range(min(8, len(doc))):
            page = doc[ix]
            tp = page.get_textpage()
            try:
                body.append(tp.get_text_range())
            finally:
                tp.close()
                page.close()
    finally:
        doc.close()
    compact = re.sub(r"\s+", "", "".join(body))
    d = date.fromisoformat(r["published_date"])
    day_pattern = rf"{d.year}年0?{d.month}月0?{d.day}日"
    if "002112" not in compact or "德邦鑫星价值" not in compact or not re.search(day_pattern, compact):
        raise ValueError("REPORT_IDENTITY_OR_PUBLICATION_DATE")
    if r["fund_code"] != "002112" or r["fund_master_code"] != "001412" or r["quality"] != "VERIFIED_TABLE_TOTALS":
        raise ValueError("REPORT_PARSED_IDENTITY_OR_TOTALS")
    codes = [h["stock_code"] for h in r["holdings"]]
    if len(codes) != len(set(codes)) or not r["reported_industries"]:
        raise ValueError("REPORT_HOLDING_DUPLICATE_OR_INDUSTRY_MISSING")
    if any(not 0 <= Decimal(h["nav_weight_pct"]) <= 100 for h in r["holdings"]):
        raise ValueError("REPORT_WEIGHT_INVALID")
    return {
        "raw_path": str(p),
        "raw_sha256": sha(p),
        "payload_hash": digest(r),
        "report_end": r["report_end"],
        "published_date": r["published_date"],
        "available_at": r["available_at"],
        "title": r["title"],
        "quality": r["quality"],
        "first_pages_identity_and_publication_checked": True,
        "availability_basis": r["availability_basis"],
        "holdings": r["holdings"],
        "reported_industries": r["reported_industries"],
        "full_stock_disclosure": r["full_stock_disclosure"],
    }


def nav_metadata(required):
    """SQL 只返回有效性、相等性、日期及摘要，禁止返回净值数值或目标方向。

    官网原件为当前历史查询版本；值相同不等于证明首次发布时间。
    """
    official = read(ROOT / "supplement/official-nav.json")
    mapping = {r["date"]: r for r in official["rows"]}
    if len(mapping) != len(official["rows"]) or any(d not in mapping for d in required):
        raise ValueError("OFFICIAL_NAV_GAP_OR_DUPLICATE")
    originals, provenance, sources = {}, [], {}
    for day in required:
        item = mapping[day]
        rc = item["receipt"]
        key = rc["sha256"]
        if key not in originals:
            p = receipt_file(rc)
            raw = read(p)
            originals[key] = {r["date"]: r for r in raw["dataList"]}
            if len(originals[key]) != len(raw["dataList"]):
                raise ValueError("OFFICIAL_RAW_DUPLICATE")
            sources[key] = {"path": str(p), "receipt": rc}
        row = originals[key][day]
        if row["fundcode"] != "002112" or Decimal(str(row["netvalue"])) != Decimal(str(item["unit_nav"])):
            raise ValueError("OFFICIAL_NAV_RAW_CACHE_MISMATCH")
        provenance.append({"day": day, "official_value": str(item["unit_nav"])})
    e = engine()
    try:
        with e.connect() as c:
            rows = [
                dict(r)
                for r in c.execute(
                    text("""
                with official as (
                  select * from jsonb_to_recordset(cast(:records as jsonb)) as x(day text, official_value numeric)
                )
                select o.day as nav_date, n.ann_date::text, n.source_published_at::text,
                       n.source_id::text, n.content_hash,
                       n.unit_nav > 0 as positive, n.unit_nav = o.official_value as matches_official
                from official o left join nav_daily n on n.fund_code='002112' and n.nav_date=cast(o.day as date)
                order by o.day
            """),
                    {"records": json.dumps(provenance)},
                ).mappings()
            ]
    finally:
        e.dispose()
    bad = [r["nav_date"] for r in rows if not r["positive"] or not r["matches_official"] or not r["content_hash"]]
    if bad:
        save(OUT / "nav-value-validation-failure.json", {"at": now(), "dates": bad})
        raise ValueError("NAV_NUMERIC_IDENTITY_OR_CURRENT_VERSION_MISMATCH")
    for row in rows:
        row["official_raw_sha256"] = mapping[row["nav_date"]]["receipt"]["sha256"]
    return {
        "rows": {r["nav_date"]: r for r in rows},
        "official_sources": sources,
        "verified_dates": len(rows),
        "missing_or_nonpositive": [],
        "current_value_mismatches": [],
        "historical_first_publication_proven": False,
    }


def stock_metadata(needed):
    """核对每个需要的证券和交易日到原始日线，不能把停牌误判为下载失败。"""
    output, dates = {}, {}
    for day in sorted({d for _, d in needed}):
        p = ROOT / "stock-days" / (day + ".json")
        cached = read(p)
        rawp = receipt_file(cached["receipt"])
        data = read(rawp)["data"]
        if data["fields"] != FIELDS:
            raise ValueError("STOCK_RAW_SCHEMA")
        rawrows = {}
        for vals in data["items"]:
            row = dict(zip(FIELDS, vals, strict=True))
            code = row.pop("ts_code")
            if row.pop("trade_date") != day.replace("-", "") or code in rawrows:
                raise ValueError("STOCK_RAW_DATE_OR_IDENTITY")
            rawrows[code] = row
        dates[day] = {
            "cache_path": str(p),
            "cache_sha256": sha(p),
            "raw_path": str(rawp),
            "raw_sha256": sha(rawp),
            "received_at": cached["receipt"]["received_at"],
        }
        for code, d in needed:
            if d != day:
                continue
            value = cached["rows"].get(code)
            original = rawrows.get(code)
            if value is None:
                if original is not None:
                    raise ValueError("STOCK_CACHE_OMITTED_SOURCE_ROW")
                output[(code, d)] = None
                continue
            if value != original:
                raise ValueError("STOCK_CACHE_RAW_MISMATCH")
            validate_quote_values(value)
            output[(code, d)] = value
    return output, dates


def market_metadata(needed):
    """宽基指数按需要的日期逐行交叉核对原件；不受独立行业新增文件影响。"""
    result = {}
    for code in INDICES:
        p = ROOT / "indices" / (code + ".json")
        cached = read(p)
        if cached["code"] != code:
            raise ValueError("MARKET_INDEX_IDENTITY")
        rawrows, refs = {}, []
        for rc in cached["receipts"]:
            rawp = receipt_file(rc)
            data = read(rawp)["data"]
            if data["fields"] != FIELDS:
                raise ValueError("MARKET_SCHEMA")
            for values in data["items"]:
                row = dict(zip(FIELDS, values, strict=True))
                if row.pop("ts_code") != code:
                    raise ValueError("MARKET_SOURCE_IDENTITY")
                d = datetime.strptime(row.pop("trade_date"), "%Y%m%d").date().isoformat()
                if d not in needed:
                    continue
                if d in rawrows and rawrows[d] != row:
                    raise ValueError("MARKET_SOURCE_VERSION_CONFLICT")
                rawrows[d] = row
            refs.append({"raw_path": str(rawp), "raw_sha256": sha(rawp)})
        for d in needed:
            row = cached["rows"].get(d)
            if row is None or row != rawrows.get(d):
                raise ValueError("MARKET_GAP_OR_RAW_MISMATCH")
            validate_quote_values(row)
        result[code] = {
            "cache_path": str(p),
            "cache_sha256": sha(p),
            "sources": refs,
            "required_dates": sorted(needed),
            "complete": True,
        }
    return result


def run():
    """每行保留时间异常和停牌依赖；完整性不冒充训练、得分或采用结论。"""
    if (OUT / "input-acceptance.json").exists():
        raise ValueError("ACCEPTANCE_ALREADY_EXISTS_DO_NOT_OVERWRITE")
    ss, version = calendar()
    ss = list(map(str, ss))
    targets = [d for d in ss if "2024-01-01" <= d <= "2026-09-29"]
    rs = reports()
    selected, required_stock, required_market, required_nav, rows = {}, set(), set(), set(), []
    for target in targets:
        ix = ss.index(target)
        navdays, qdays = ss[ix - 61 : ix], ss[ix - 21 : ix]
        as_of = target + "T08:00:00+08:00"
        r = select_report(rs, datetime.fromisoformat(as_of))
        age = (date.fromisoformat(qdays[-1]) - date.fromisoformat(r["report_end"])).days
        if not 0 <= age <= 210 or not Decimal(r["disclosed_nav_pct"]) > 0:
            raise ValueError("REPORT_STALE_OR_NO_EXPOSURE")
        selected[r["raw"]["sha256"]] = r
        holdings = [h for h in r["holdings"] if Decimal(h["nav_weight_pct"]) > 0]
        pairs = [(h["stock_code"], day) for h in holdings for day in qdays]
        required_stock.update(pairs)
        required_nav.update([*navdays, target])
        required_market.update(qdays[-5:])
        rows.append(
            {
                "target_date": target,
                "as_of": as_of,
                "base_date": qdays[-1],
                "nav_dates": navdays,
                "target_nav_checked_for_existence_only": target,
                "report_raw_sha256": r["raw"]["sha256"],
                "holdings_window": qdays,
                "stock_pairs": pairs,
                "market_dates": qdays[-5:],
                "sector_row_reference": target,
            }
        )
    verified_reports = {k: verify_report(r) for k, r in selected.items()}
    nav = nav_metadata(sorted(required_nav))
    quotes, stock_sources = stock_metadata(required_stock)
    market = market_metadata(required_market)
    suspension = read(OUT / "suspension-evidence.json")
    events = {e["code"]: e for e in suspension["events"]}
    sector = read(OUT / "sector-source-inputs.json")
    if [r["target_date"] for r in sector["rows"]] != targets:
        raise ValueError("SECTOR_TARGET_ALIGNMENT")
    yearly = defaultdict(Counter)
    unexplained = []
    for row in rows:
        year = row["target_date"][:4]
        late = [
            d
            for d in row["nav_dates"]
            if nav["rows"][d]["ann_date"] and nav["rows"][d]["ann_date"] > row["target_date"]
        ]
        missing = []
        for code, d in row.pop("stock_pairs"):
            if quotes[(code, d)] is not None:
                continue
            event = events.get(code)
            if event is None or d not in event["dates"]:
                unexplained.append([code, d])
                state = {"state": "UNEXPLAINED_MISSING_QUOTE"}
            else:
                state = suspension_state(d, row["as_of"], event)
                if state["state"] != "OFFICIAL_SUSPENSION_NO_TRADE":
                    raise ValueError("SUSPENSION_NOT_KNOWN_AS_OF")
            missing.append({"code": code, "date": d, **state})
        row.update(
            nav_time_conflicts=late,
            stock_no_trade_states=missing,
            strict_existing_input_recipe_complete=not late and not missing,
            training_eligible=False,
            labels_read=False,
        )
        yearly[year].update(
            targets=1,
            nav_time_affected=int(bool(late)),
            holdings_no_trade_affected=int(bool(missing)),
            sector_complete=1,
            market_complete=1,
            existing_recipe_complete=int(not late and not missing),
        )
    if unexplained:
        save(OUT / "unexplained-stock-gaps.json", {"at": now(), "gaps": unexplained})
        raise ValueError("STOCK_GAPS_STILL_EXECUTABLE")
    affected_dates = sorted({d for row in rows for d in row["nav_time_conflicts"]})
    if affected_dates != sorted(NAV_DATES):
        raise ValueError("NAV_TIME_SCOPE_CHANGED_REVIEW_REQUIRED")
    sources = {
        "at": now(),
        "calendar_hash": version,
        "reports": verified_reports,
        "nav": nav,
        "stock_dates": stock_sources,
        "market": market,
        "sector_file": str(OUT / "sector-source-inputs.json"),
        "sector_sha256": sha(OUT / "sector-source-inputs.json"),
        "suspension_file": str(OUT / "suspension-evidence.json"),
        "suspension_sha256": sha(OUT / "suspension-evidence.json"),
    }
    save(OUT / "input-provenance.json", sources)
    save(
        OUT / "input-date-manifest.json",
        {
            "at": now(),
            "rows": rows,
            "source_provenance_sha256": sha(OUT / "input-provenance.json"),
            "source_vintage": "HISTORICAL_RECONSTRUCTION_NOT_ORIGINAL_VINTAGE_CERTIFICATION",
            "sealed_2025_and_backfilled_2026_labels_accessed": False,
            "actual_new_fits": 0,
            "personal_account_conditions": "UNKNOWN",
            "training_eligible": False,
        },
    )
    result = {
        "at": now(),
        "scope": [targets[0], targets[-1]],
        "target_count": len(rows),
        "nav_date_count": len(required_nav),
        "nav_value_missing": 0,
        "nav_current_value_conflicts": 0,
        "selected_reports": len(selected),
        "distinct_stocks": len({c for c, _ in required_stock}),
        "required_stock_date_pairs": len(required_stock),
        "verified_quote_pairs": sum(v is not None for v in quotes.values()),
        "official_no_trade_pairs": sum(v is None for v in quotes.values()),
        "unexplained_quote_gaps": 0,
        "market_pairs": len(required_market) * len(INDICES),
        "yearly": dict(yearly),
        "nav_time_conflict_dates": affected_dates,
        "nav_time_affected_targets": sum(bool(r["nav_time_conflicts"]) for r in rows),
        "holdings_no_trade_affected_targets": sum(bool(r["stock_no_trade_states"]) for r in rows),
        "existing_recipe_input_complete": sum(r["strict_existing_input_recipe_complete"] for r in rows),
        "all_targets_retained": True,
        "no_new_fit_or_admission": True,
        "public_first_version_evidence_status": "SEVEN_NAV_DATES_UNRESOLVED",
        "first_seen_revision_risk": "CURRENT_HISTORY_RECONSTRUCTION_RETAINS_VENDOR_REVISION_RISK",
        "historical_stock_industry_mapping": "NOT_INFERRED_FROM_CURRENT_CLASSIFICATION",
        "policy_and_all_news": "NOT_A_UNIVERSAL_GATE_PER_USER_NARROWED_SCOPE",
        "manifest_sha256": sha(OUT / "input-date-manifest.json"),
        "provenance_sha256": sha(OUT / "input-provenance.json"),
    }
    save(OUT / "input-acceptance.json", result)
    board(
        f"2024 年以来 {len(rows)} 个目标日期已逐日验收并全部保留：所需净值 {len(required_nav)} 日、"
        f"持仓原件 {len(selected)} 份、个股 {len(required_stock)} 个证券日。"
        f"当前净值与官网原件一致；按原输入规则完整 {result['existing_recipe_input_complete']} 日。"
        "其余日期按净值时间证据不足或停牌分别保留原因，未准入、未训练。",
        OUT / "input-acceptance.json",
    )
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    run()
