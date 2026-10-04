"""近期补数的独立验收：逐日来源、停牌语义与时间边界，不构造标签、不训练。"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pypdfium2 as pdfium
from app.services.direction_1d_protocol import ZONE, calendar, digest
from sqlalchemy import text

from scripts.fund_002112_recent_completion_v1 import OUT, ROOT, SECTORS, WEB, engine, now, read, save, sha


def board(message, evidence):
    """在原统一看板追加实际完成记录；重读最新字节，不覆盖其他会话新增记录。"""
    path = WEB / "docs_zhx/implementation/fund-radar.md"
    raw = path.read_bytes()
    content = raw.decode("utf-8-sig")
    anchor = "## 0. 当前状态（下次会话从这里开始）"
    if content.count(anchor) != 1:
        raise ValueError("BOARD_ANCHOR_CHANGED")
    entry = (
        f"\n\n> **{now()} M02 近期输入补齐。** {message}\n>\n> 证据：`{evidence}`。"
        "独立批次目录 `recent-input-completion/20260930-v1`；原研究、预算及旧停止结论不变，本轮拟合 0。\n"
    )
    updated = content.replace(anchor, anchor + entry, 1)
    if path.read_bytes() != raw:
        raise ValueError("BOARD_CHANGED_RETRY_FROM_LATEST")
    # 只在本批次确实完成一项时新增段落，旧看板正文逐字保留。
    path.write_text(updated, encoding="utf-8", newline="")


def parse_sector(raw, code):
    obj = json.loads(raw)
    if obj.get("code") != 0 or obj.get("data", {}).get("fields") != ["ts_code", "trade_date", "close", "pre_close"]:
        raise ValueError("SECTOR_RESPONSE_SCHEMA")
    rows = {}
    for c, d, close, previous in obj["data"]["items"]:
        day = datetime.strptime(d, "%Y%m%d").date().isoformat()
        if (
            c != code
            or day in rows
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (close, previous))
        ):
            raise ValueError("SECTOR_ROW_INVALID")
        rows[day] = {"close": str(close), "pre_close": str(previous)}
    return rows


def merge_series(existing, incoming):
    """重叠日期必须完全一致；后取版本冲突时保留两份原件而拒绝合并。"""
    for day, row in incoming.items():
        if day in existing and any(Decimal(existing[day][k]) != Decimal(row[k]) for k in row):
            raise ValueError("SECTOR_SOURCE_VERSION_CONFLICT")
        existing[day] = row


def sectors():
    target = OUT / "sector-acceptance.json"
    if target.exists():
        return read(target)
    folder = ROOT.parent / "direction-1d-sprint-20260914/sector-data-feasibility-v1"
    values = {c: {} for c in SECTORS}
    sources = []
    for code in SECTORS:
        for year in (2023, 2024, 2025, 2026):
            rc = read(folder / f"receipt-{code}-{year}.json")
            p = folder / f"response-{code}-{year}.json"
            if sha(p) != rc["sha256"]:
                raise ValueError("SECTOR_OLD_SOURCE_HASH")
            merge_series(values[code], parse_sector(p.read_bytes(), code))
            sources.append(
                {"path": str(p), "sha256": sha(p), "receipt_path": str(folder / f"receipt-{code}-{year}.json")}
            )
    for p in (ROOT.parent / "direction-1d-sprint-20260914/round-37/sector-inputs").glob("*/raw/*.response.json"):
        rc = read(p.with_name(p.name.replace(".response.json", ".json")))
        if sha(p) != rc["body_sha256"]:
            raise ValueError("SECTOR_LIVE_SOURCE_HASH")
        obj = json.loads(p.read_bytes())
        code = obj["data"]["items"][0][0]
        if code in values:
            merge_series(values[code], parse_sector(p.read_bytes(), code))
            sources.append({"path": str(p), "sha256": sha(p)})
    before = {c: set(v) for c, v in values.items()}
    for r in read(OUT / "sector-acquisition.json")["results"]:
        if r["status"] != "RECEIVED":
            raise ValueError("SECTOR_NEW_ACQUISITION_INCOMPLETE")
        rc = r["receipt"]
        p = OUT / rc["file"]
        if sha(p) != rc["sha256"]:
            raise ValueError("SECTOR_NEW_SOURCE_HASH")
        incoming = parse_sector(p.read_bytes(), r["code"])
        if any(not "2026-09-16" <= d <= "2026-09-29" for d in incoming):
            raise ValueError("SECTOR_NEW_SOURCE_OUTSIDE_RANGE")
        merge_series(values[r["code"]], incoming)
        sources.append({"path": str(p), "sha256": sha(p), "receipt": rc})
    ss = [str(d) for d in calendar()[0]]
    targets = [d for d in ss if "2024-01-01" <= d <= "2026-09-29"]
    required = sorted({d for t in targets for d in ss[ss.index(t) - 21 : ss.index(t)]})
    counts, gaps, discontinuities = {}, [], []
    for code, rows in values.items():
        gaps.extend((code, d) for d in required if d not in rows)
        for prior, day in zip(required, required[1:], strict=False):
            if (
                prior in rows
                and day in rows
                and abs(Decimal(rows[day]["pre_close"]) - Decimal(rows[prior]["close"])) > Decimal("0.001")
            ):
                discontinuities.append({"code": code, "date": day})
        counts[code] = {
            "required": len(required),
            "present": sum(d in rows for d in required),
            "new_required_rows": sum(d not in before[code] and d in rows for d in required),
            "new_rows_total": len(set(rows) - before[code]),
            "latest": max(rows),
        }
    if gaps or discontinuities:
        save(OUT / "sector-validation-failure.json", {"at": now(), "gaps": gaps, "discontinuities": discontinuities})
        raise ValueError("SECTOR_CONTINUITY_OR_COVERAGE_FAILED")
    rows_out = []
    for target_day in targets:
        ix = ss.index(target_day)
        needed = ss[ix - 21 : ix]
        facts = {}
        for code in SECTORS:
            prices = values[code]
            last = Decimal(prices[needed[-1]]["close"])
            facts[code] = {f"return_{n}d": str(last / Decimal(prices[needed[-n - 1]]["close"]) - 1) for n in (1, 5, 20)}
        rows_out.append(
            {
                "target_date": target_day,
                "as_of": target_day + "T08:00:00+08:00",
                "last_input_date": needed[-1],
                "lookback_start": needed[0],
                "sector_background": facts,
            }
        )
    e = engine()
    with e.connect() as c:
        identities = [
            dict(r)
            for r in c.execute(
                text(
                    "select index_code,display_name,category,list_date,expiry_date "
                    "from market_index_catalog where index_code=ANY(:codes)"
                ),
                {"codes": list(SECTORS)},
            ).mappings()
        ]
    e.dispose()
    if len(identities) != 5 or any(str(r["list_date"]) > required[0] or r["expiry_date"] for r in identities):
        raise ValueError("SECTOR_IDENTITY_INVALID")
    save(
        OUT / "sector-source-inputs.json",
        {
            "at": now(),
            "series": values,
            "sources": sources,
            "identities": identities,
            "rows": rows_out,
            "meaning": "INDEPENDENT_BACKGROUND_NOT_FUND_WEIGHTED_INDUSTRY",
            "source_vintage": "RECONSTRUCTED_HISTORY_WITH_CURRENT_VENDOR_REVISION_RISK",
            "labels": "NOT_ACCESSED_OR_GENERATED",
            "training_eligible": False,
        },
    )
    result = {
        "at": now(),
        "targets": len(targets),
        "counts": counts,
        "missing": gaps,
        "discontinuities": discontinuities,
        "added_required_rows": sum(v["new_required_rows"] for v in counts.values()),
        "added_total_rows": sum(v["new_rows_total"] for v in counts.values()),
        "source_count": len(sources),
        "input_path": str(OUT / "sector-source-inputs.json"),
        "input_sha256": sha(OUT / "sector-source-inputs.json"),
        "all_source_dates_before_target": True,
        "historical_original_vintage_proven": False,
    }
    save(target, result)
    board(
        "五类行业／主题序列补齐所需 40 条日线，另保存 9/29 的 5 条最新输入；"
        "665 个日期的 1/5/20 日行业背景输入已核对日期、身份、原始摘要及前收盘连续性，无缺日。"
        "历史供应商版本风险单列，不冒称基金行业加权持仓。",
        target,
    )
    print(json.dumps(result, ensure_ascii=False, default=str))
    return result


def pdf_pages(path):
    """在独立读取中完整提取小型停复牌公告；不重写共享文本缓存。"""
    doc = pdfium.PdfDocument(path)
    pages = []
    try:
        for page in doc:
            textpage = page.get_textpage()
            try:
                pages.append(textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
    finally:
        doc.close()
    return pages


def suspension_state(day, as_of, event):
    """停牌状态只使用截止时已可得的开始/进展公告，不借复牌公告预知终点。

    没有成交价格就是 None；不前填、不把收益置零、不对剩余持仓重新归一化。
    复牌信息在自身可得时间前只作后验核对材料，不进入历史输入。
    """
    cutoff = datetime.fromisoformat(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("SUSPENSION_AS_OF_TIMEZONE_REQUIRED")
    cutoff = cutoff.astimezone(ZONE)
    if date.fromisoformat(day) >= cutoff.date():
        raise ValueError("SUSPENSION_FUTURE_OR_UNFINISHED_DAY")
    if day < event["start_date"] or cutoff < datetime.fromisoformat(event["start_available_at"]):
        return {"state": "UNKNOWN", "price": None, "return": None, "resume_known": False}
    resume_known = cutoff >= datetime.fromisoformat(event["resume_available_at"])
    if resume_known and day >= event["resume_date"]:
        return {"state": "RESUMED_REQUIRE_OBSERVED_QUOTE", "price": None, "return": None, "resume_known": True}
    return {"state": "OFFICIAL_SUSPENSION_NO_TRADE", "price": None, "return": None, "resume_known": resume_known}


def suspensions():
    dest = OUT / "suspension-evidence.json"
    if dest.exists():
        return read(dest)
    scope = {
        "301486.SZ": {
            "start_date": "2025-04-08",
            "resume_date": "2025-04-22",
            "notice_ids": ["1223022285", "1223093997", "1223196932"],
        },
        "688313.SH": {
            "start_date": "2025-06-30",
            "resume_date": "2025-07-11",
            "notice_ids": ["1224016285", "1224083923", "1224131033"],
        },
    }
    context = read(ROOT / "supplement/stock-context.json")
    events = []
    for code, s in scope.items():
        catalog = read(ROOT / "supplement/company-announcements" / (code + ".json"))
        notices = []
        for item_id in s["notice_ids"]:
            item = next(r for r in catalog["rows"] if r["announcementId"] == item_id)
            rc = item["receipt"]
            if sha(ROOT / rc["file"]) != rc["sha256"]:
                raise ValueError("SUSPENSION_CATALOG_HASH")
            original_items = read(ROOT / rc["file"])["announcements"]
            if not any(
                r["announcementId"] == item_id and r["secCode"] == code[:6] and r["adjunctUrl"] == item["adjunctUrl"]
                for r in original_items
            ):
                raise ValueError("SUSPENSION_CATALOG_IDENTITY")
            url = "https://static.cninfo.com.cn/" + item["adjunctUrl"]
            receipt_path = ROOT / "supplement/cninfo-receipts" / (digest({"url": url, "params": None}) + ".json")
            pdf_receipt = read(receipt_path)
            pdf_path = ROOT / pdf_receipt["file"]
            if sha(pdf_path) != pdf_receipt["sha256"]:
                raise ValueError("SUSPENSION_ORIGINAL_HASH")
            pages = pdf_pages(pdf_path)
            body = re.sub(r"\s+", "", "\n".join(pages))
            if code[:6] not in body or ("致尚科技" if code.startswith("301") else "仕佳光子") not in body:
                raise ValueError("SUSPENSION_BODY_IDENTITY")
            kind = (
                "START" if item_id == s["notice_ids"][0] else "RESUME" if item_id == s["notice_ids"][-1] else "CONTINUE"
            )
            event_date = s["resume_date"] if kind == "RESUME" else s["start_date"]
            d = date.fromisoformat(event_date)
            needle = f"{d.year}年{d.month}月{d.day}日"
            if needle not in body or ("复牌" if kind == "RESUME" else "停牌") not in body:
                raise ValueError("SUSPENSION_BODY_DATE_OR_STAGE")
            pub_day = date.fromisoformat(item["announced_at_source"][:10])
            available = datetime.combine(pub_day + timedelta(days=1), time(8), ZONE).isoformat()
            snippets = [
                {"page": i + 1, "text": p} for i, p in enumerate(pages) if any(w in p for w in ["停牌", "复牌"])
            ]
            notice = {
                "id": item_id,
                "title": item["title_plain"],
                "kind": kind,
                "published_at_source": item["announced_at_source"],
                "available_at": available,
                "availability_basis": "SOURCE_CALENDAR_DAY_PLUS_ONE_0800_CONSERVATIVE_RECONSTRUCTION",
                "raw_path": str(pdf_path),
                "raw_sha256": sha(pdf_path),
                "url": url,
                "catalog_receipt": rc,
                "pdf_receipt": pdf_receipt,
                "snippets": snippets,
            }
            save(OUT / "suspension-notices" / (item_id + ".json"), notice)
            notices.append(notice)
        daily = next(r for r in context["suspensions"] if r["rows"][0]["ts_code"] == code)
        rc = daily["receipt"]
        if sha(ROOT / rc["file"]) != rc["sha256"]:
            raise ValueError("SUSPENSION_VENDOR_HASH")
        dates = [datetime.strptime(r["trade_date"], "%Y%m%d").date().isoformat() for r in daily["rows"]]
        required = [str(d) for d in calendar()[0] if s["start_date"] <= str(d) < s["resume_date"]]
        if set(dates) != set(required) or any(r["suspend_type"] != "S" for r in daily["rows"]):
            raise ValueError("SUSPENSION_VENDOR_OFFICIAL_DATE_CONFLICT")
        events.append(
            {
                "code": code,
                **s,
                "dates": sorted(dates),
                "notices": [r["id"] for r in notices],
                "start_available_at": notices[0]["available_at"],
                "resume_available_at": notices[-1]["available_at"],
                "vendor_receipt": rc,
                "state": "NO_TRADE_CONFIRMED_NOT_DOWNLOAD_FAILURE",
                "price_return_amount": None,
                "zero_or_forward_fill": False,
                "original_holdings_recipe_compatible": False,
            }
        )
    result = {
        "at": now(),
        "events": events,
        "missing_quote_days_explained": sum(len(e["dates"]) for e in events),
        "new_network_requests": 0,
        "notices_reused": 6,
        "state_policy": "PRESERVE_NONE_AND_EXPLICIT_SUSPENSION_DO_NOT_CHANGE_OLD_RECIPE",
        "future_resume_notice_used_before_publication": False,
    }
    save(dest, result)
    board(
        "两只股票的 19 个无行情日已用 6 份缓存停牌、进展及复牌原件与供应商逐日记录交叉核验，确认为无交易。"
        "新增独立状态规则保留空价格及停牌标记，不补零、不前填、不用未来复牌信息，不修改旧持仓算法；"
        "原算法受影响的日期仍保留。",
        dest,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["sectors", "suspensions"])
    args = parser.parse_args()
    {"sectors": sectors, "suspensions": suspensions}[args.step]()
