"""复用包内已核新浪转债日线来源，补取本轮已收盘数据并保留原响应。"""

import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx

from app.services.direction_1d_protocol import calendar, digest
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save
from app.services.material_content import content_digest


def closed_rows(raw, security, base):
    text = raw.decode("utf-8")
    match = re.search(r"var _" + re.escape(security[:6]) + r"=\((\[.*\])\);\s*$", text, re.S)
    if not match:
        raise ValueError("BOND_QUOTE_SOURCE_FORMAT")
    rows = json.loads(match[1])
    if not isinstance(rows, list) or not 2 <= len(rows) <= 12:
        raise ValueError("BOND_QUOTE_ROWS")
    by_date = {}
    for row in rows:
        day = date.fromisoformat(row["day"])
        value = Decimal(row["close"])
        if str(day) in by_date or not value.is_finite() or value <= 0:
            raise ValueError("BOND_QUOTE_VALUE")
        by_date[str(day)] = row
    days, _ = calendar()
    index = days.index(date.fromisoformat(base))
    expected = [str(days[index - 1]), base]
    if not set(expected) <= set(by_date):
        raise ValueError("BOND_QUOTE_BASE_MISSING")
    return {day: by_date[day] for day in expected}


def refresh(material, base, root):
    fact = material.get("facts", {}).get("current_bond_prices")
    if not fact:
        return []
    documents = {d["id"]: d for d in material["documents"]}
    results, refs, quotes = [], [], []
    for old_ref in fact["source_refs"]:
        old = documents[old_ref]
        uri = urlparse(old["url"])
        query = parse_qs(uri.query)
        symbol = query.get("symbol", [""])[0]
        if (
            uri.scheme != "https"
            or uri.hostname != "quotes.sina.cn"
            or "CN_MarketDataService.getKLineData" not in uri.path
            or not re.fullmatch(r"(?:sh|sz)1\d{5}", symbol)
            or query.get("scale") != ["240"]
            or query.get("datalen") != ["8"]
        ):
            results.append({"source_ref": old_ref, "status": "SOURCE_REVIEW_REQUIRED"})
            continue
        security = symbol[2:] + "." + symbol[:2].upper()
        cache = Path(root) / "bond-receipts" / base / (security + ".json")
        try:
            if cache.exists():
                receipt = read(cache)
                raw = Path(receipt["raw_path"]).read_bytes()
                if content_digest(raw) != receipt["raw_sha256"]:
                    raise ValueError("BOND_QUOTE_RAW_CHANGED")
            else:
                with httpx.Client(timeout=httpx.Timeout(15, connect=4), trust_env=False) as client:
                    response = client.get(old["url"])
                    response.raise_for_status()
                    raw = response.content
                if len(raw) > 200000:
                    raise ValueError("BOND_QUOTE_RESPONSE_SIZE")
                sha = content_digest(raw)
                path = Path(root) / "raw" / (sha + ".raw")
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(raw)
                receipt = {
                    "url": old["url"],
                    "raw_path": str(path.resolve()),
                    "raw_sha256": sha,
                    "received_at": now().isoformat(),
                }
                closed_rows(raw, security, base)
                versioned_save(cache, receipt)
            rows = closed_rows(raw, security, base)
            body = json.dumps(rows, ensure_ascii=False, sort_keys=True)
            ref = "bond-quote:" + security + ":" + receipt["raw_sha256"]
            doc = {
                "id": ref,
                "kind": "MARKET_DATA",
                "title": "本转债实际基准日及前一交易日日线",
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
                material["documents"].append(doc)
                documents[ref] = doc
            refs.append(ref)
            quotes.append(body)
            results.append({"security": security, "status": "SUCCEEDED", "source": receipt})
        except Exception as error:
            results.append({"security": security, "status": "ACTION_REQUIRED", "failure": str(error)[:250]})
    if len(refs) == len(fact["source_refs"]):
        material["facts"]["current_bond_prices"] = {**fact, "source_refs": refs, "quotes": quotes}
    return results
