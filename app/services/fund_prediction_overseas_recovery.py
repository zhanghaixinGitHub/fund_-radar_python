"""复用已核实美日公开来源及官方汇率，严格区分普通收盘、盘后和当前盘中数据。"""

import json
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save
from app.services.material_content import content_digest


def fetch(url, base, root):
    cache = Path(root) / "overseas-receipts" / base / (digest(url) + ".json")
    if cache.exists():
        receipt = read(cache)
        raw = Path(receipt["raw_path"]).read_bytes()
        if content_digest(raw) != receipt["raw_sha256"]:
            raise ValueError("OVERSEAS_RAW_CHANGED")
        return raw, receipt
    with httpx.Client(
        timeout=httpx.Timeout(20, connect=5),
        trust_env=False,
        follow_redirects=False,
        headers={"Referer": "https://finance.sina.com.cn"},
    ) as client:
        response = client.get(url)
    raw = response.content
    if len(raw) > 3000000:
        raise ValueError("OVERSEAS_RESPONSE_SIZE")
    sha = content_digest(raw)
    path = Path(root) / "raw" / (sha + ".raw")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    receipt = {
        "url": url,
        "raw_path": str(path.resolve()),
        "raw_sha256": sha,
        "received_at": now().isoformat(),
        "http_status": response.status_code,
    }
    versioned_save(Path(root) / "overseas-attempts" / (digest(receipt) + ".json"), receipt)
    response.raise_for_status()
    versioned_save(cache, receipt)
    return raw, receipt


def us_close(raw, security, base):
    symbol = security[:-3].lower()
    match = re.fullmatch(r"var hq_str_gb_" + re.escape(symbol) + r'="([^"\r\n]+)";\s*', raw.decode("gb18030"))
    if not match:
        raise ValueError("US_QUOTE_SYMBOL_OR_FORMAT")
    fields = match[1].split(",")
    # 只取已核准的普通收盘、昨收、涨幅和普通收盘时刻字段。尾部扩展字段
    # 不属于本次收盘取值契约，不能额外要求它们与普通收盘价完全相同。
    if len(fields) != 36 or fields[29] != base[:4]:
        raise ValueError("US_QUOTE_LAYOUT")
    stamp = re.fullmatch(r"([A-Za-z]{3}) +(\d{1,2}) (\d{2}:\d{2}[AP]M) (EST|EDT)", fields[25])
    if not stamp:
        raise ValueError("US_REGULAR_CLOSE_TIME")
    at = datetime.strptime(base[:4] + " " + " ".join(stamp.groups()[:3]), "%Y %b %d %I:%M%p").replace(
        tzinfo=ZoneInfo("America/New_York")
    )
    if at.date().isoformat() != base or at.hour != 16 or at.tzname() != stamp[4]:
        raise ValueError("US_REGULAR_CLOSE_DATE")
    # 普通收盘字段的最后报告时刻可能晚于整点；保留原时刻，不取盘后字段。
    # closed_at沿用本基金已核NYSE普通交易完成边界，明确不是供应商精确成交时刻。
    completion = at.replace(minute=0)
    close, previous, pct = Decimal(fields[1]), Decimal(fields[26]), Decimal(fields[2])
    if (
        not all(v.is_finite() for v in (close, previous, pct))
        or min(close, previous) <= 0
        or abs((close / previous - 1) * 100 - pct) > Decimal(".015")
    ):
        raise ValueError("US_REGULAR_CLOSE_VALUE")
    return {
        "date": base,
        "close": str(close),
        "previous_close": str(previous),
        "change_pct": str(pct),
        "closed_at": completion.isoformat(),
        "observed_quote_time": fields[25],
        "session": "REGULAR_CLOSE",
        "close_timestamp_kind": "EXCHANGE_COMPLETION_BOUND",
        "currency": "USD",
        "valuation_market": "US",
    }


def jp_close(raw, security, base):
    symbol = "998405.T" if security == "TOPIX" else security[:-3] + ".T"
    soup = BeautifulSoup(raw, "html.parser")
    if symbol not in soup.get_text(" ", strip=True):
        raise ValueError("JP_HISTORY_IDENTITY")
    rows = {}
    for tr in soup.select("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"], recursive=False)]
        if not cells or not re.fullmatch(r"20\d{2}/\d{1,2}/\d{1,2}", cells[0]):
            continue
        day = datetime.strptime(cells[0], "%Y/%m/%d").date().isoformat()
        if day > base:
            continue
        if len(cells) < 5 or day in rows:
            raise ValueError("JP_HISTORY_LAYOUT")
        values = [Decimal(v.replace(",", "")) for v in cells[1:5]]
        op, high, low, close = values
        if not all(v.is_finite() and v > 0 for v in values) or high < max(op, close) or low > min(op, close):
            raise ValueError("JP_HISTORY_VALUE")
        rows[day] = {"date": day, "open": str(op), "high": str(high), "low": str(low), "close": str(close)}
    days = sorted(rows)
    if len(days) < 2 or days[-1] != base:
        raise ValueError("JP_HISTORY_BASE_MISSING")
    previous, last = rows[days[-2]], rows[base]
    close, prior = Decimal(last["close"]), Decimal(previous["close"])
    return {
        "date": base,
        "close": str(close),
        "previous_close": str(prior),
        "change_pct": str((close / prior - 1) * 100),
        "closed_at": base + "T15:30:00+09:00",
        "close_timestamp_kind": "EXCHANGE_COMPLETION_BOUND",
        "session": "REGULAR_CLOSE",
        "currency": "JPY",
        "valuation_market": "JP",
    }, {days[-2]: previous, base: last}


def official_fx(raw, base):
    soup = BeautifulSoup(raw, "html.parser")
    selected = []
    header_valid = False
    for tr in soup.select("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"], recursive=False)]
        if cells[:5] == ["日期", "美元", "欧元", "日元", "港元"]:
            header_valid = True
        if cells and cells[0] == base:
            selected.append(cells)
    if not header_valid or len(selected) != 1 or len(selected[0]) < 5:
        raise ValueError("OFFICIAL_FX_BASE_MISSING")
    row = selected[0]
    rates = []
    for currency, column in (("USD", 1), ("JPY", 3), ("HKD", 4)):
        amount = Decimal(row[column])
        if not amount.is_finite() or amount <= 0:
            raise ValueError("OFFICIAL_FX_VALUE")
        rates.append({"date": base, "currency": currency, "cny_per_unit": str(amount / 100)})
    return {
        "rates": rates,
        "official_rows": [{"row": row}],
        "conversion": "原表每百外币兑人民币，除以百换算每单位；不使用市场代理汇率",
    }


def refresh(material, base, root):
    evidence = material.get("calendar_evidence") or {}
    if evidence.get("policy") != "QDII_CNY_VERIFIED_SESSION_V1":
        return []
    documents = {d["id"]: d for d in material["documents"]}
    results = []

    def append(ref, kind, title, body, receipt):
        document = {
            "id": ref,
            "kind": kind,
            "title": title,
            "url": receipt["url"],
            "body": body,
            "body_hash": digest(body),
            "body_scope": "VERIFIED_SOURCE_TRANSFORM",
            "published_date": base,
            "available_at": receipt["received_at"],
            "codes": [material["fund_code"]],
            "raw_path": receipt["raw_path"],
            "raw_sha256": receipt["raw_sha256"],
        }
        if ref not in documents:
            material["documents"].append(document)
            documents[ref] = document

    for security, container in [(c["code"], c) for c in material.get("companies", [])] + list(
        material.get("market", {}).items()
    ):
        quote = (container.get("quote") or {}) if "code" in container else container
        if (
            "code" not in container
            and quote.get("date") == base
            and re.fullmatch(r"[15]\d{5}\.(SH|SZ)", security)
            and str(quote.get("source_ref", "")).startswith("reference-quote:" + security + ":")
        ):
            quote.update(
                currency="CNY",
                valuation_market="CN",
                session="REGULAR_CLOSE",
                closed_at=base + "T15:00:00+08:00",
                close_timestamp_kind="EXCHANGE_COMPLETION_BOUND",
            )
        if quote.get("date") == base or security in material.get("market_quote_exceptions", {}):
            continue
        original = documents.get(quote.get("source_ref"), {})
        url = original.get("url", "")
        parsed = urlparse(url)
        try:
            if (
                re.fullmatch(r"[A-Z.]{1,10}\.US", security)
                and parsed.hostname == "hq.sinajs.cn"
                and url == "https://hq.sinajs.cn/list=gb_" + security[:-3].lower()
            ):
                raw, receipt = fetch(url, base, root)
                current = us_close(raw, security, base)
                body = json.dumps(
                    {"regular_close": current, "after_hours_excluded": True}, ensure_ascii=False, sort_keys=True
                )
            elif (
                (re.fullmatch(r"\d{3}[\dA-Z]\.JP", security) or security == "TOPIX")
                and parsed.hostname == "finance.yahoo.co.jp"
                and parsed.path
                == "/quote/" + ("998405.T" if security == "TOPIX" else security[:-3] + ".T") + "/history"
            ):
                raw, receipt = fetch(url, base, root)
                current, history = jp_close(raw, security, base)
                body = json.dumps(history, ensure_ascii=False, sort_keys=True)
            else:
                results.append({"security": security, "status": "OVERSEAS_SOURCE_IDENTITY_REVIEW_REQUIRED"})
                continue
            ref = "overseas-close:" + security + ":" + receipt["raw_sha256"]
            append(ref, "MARKET_DATA", "本证券实际普通交易时段收盘 " + security, body, receipt)
            current["source_ref"] = ref
            if "code" in container:
                container["quote"] = current
            else:
                material["market"][security] = current
            results.append({"security": security, "status": "SUCCEEDED", "source": receipt})
        except Exception as error:
            results.append({"security": security, "status": "ACTION_REQUIRED", "failure": str(error)[:250]})
    try:
        prior = documents[evidence["fx_ref"]]
        if urlparse(prior["url"]).hostname != "www.safe.gov.cn":
            raise ValueError("OFFICIAL_FX_SOURCE_REVIEW_REQUIRED")
        url = "https://www.safe.gov.cn/AppStructured/hlw/RMBQuery.do?startDate=" + base + "&endDate=" + base
        raw, receipt = fetch(url, base, root)
        data = official_fx(raw, base)
        ref = "official-fx:" + receipt["raw_sha256"]
        append(
            ref,
            "FX_REFERENCE_DATA",
            "国家外汇管理局本基准日人民币中间价",
            json.dumps(data, ensure_ascii=False),
            receipt,
        )
        evidence["fx_ref"] = ref
        results.append({"kind": "OFFICIAL_FX", "status": "SUCCEEDED", "source": receipt})
    except Exception as error:
        results.append({"kind": "OFFICIAL_FX", "status": "ACTION_REQUIRED", "failure": str(error)[:250]})
    for security, quote in [(c["code"], c.get("quote") or {}) for c in material.get("companies", [])] + list(
        material.get("market", {}).items()
    ):
        ref = quote.get("source_ref", "")
        if quote.get("date") == base and ref.startswith("overseas-close:") and ref in documents:
            material["facts"]["current_overseas_quote:" + security] = {
                "category": "MARKET",
                "source_refs": [ref],
                "quotes": [documents[ref]["body"]],
            }
    return results
