"""按既有港股历史日线来源补齐收盘价；不将盘中价或境内股票代码代入港股。"""

import json
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.services.direction_1d_protocol import calendar, digest
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save
from app.services.material_content import content_digest


def closed_rows(raw, base):
    content = raw.decode("ascii")
    year = datetime.fromisoformat(base).year
    if not content.startswith(f"daily_data_{year % 100:02d}="):
        raise ValueError("HK_HISTORY_YEAR_OR_FORMAT")
    rows = {}
    for match in re.finditer(r"(?m)^(\d{6}) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) (\d+)\\n", content):
        day = datetime.strptime(match[1], "%y%m%d").date().isoformat()
        values = [Decimal(match[i]) for i in range(2, 7)]
        op, close, high, low, volume = values
        if (
            day in rows
            or not all(v.is_finite() for v in values)
            or min(values[:4]) <= 0
            or high < max(op, close)
            or low > min(op, close)
            or volume < 0
        ):
            raise ValueError("HK_HISTORY_VALUE")
        rows[day] = dict(zip(("open", "close", "high", "low", "volume"), match.groups()[1:], strict=True))
    days, _ = calendar()
    index = days.index(datetime.fromisoformat(base).date())
    expected = [str(days[index - 1]), base]
    if not set(expected) <= set(rows):
        raise ValueError("HK_HISTORY_BASE_OR_PREVIOUS_SESSION_MISSING")
    return {day: rows[day] for day in expected}


def refresh(material, base, root):
    documents = {d["id"]: d for d in material["documents"]}
    results = []
    for company in material.get("companies", []):
        code = company["code"]
        if not re.fullmatch(r"\d{5}\.HK", code) or (company.get("quote") or {}).get("date") == base:
            continue
        old = documents.get((company.get("quote") or {}).get("source_ref"), {})
        parsed = urlparse(old.get("url", ""))
        if (
            parsed.scheme != "https"
            or parsed.hostname != "data.gtimg.cn"
            or not re.fullmatch(r"/flashdata/hk/daily/\d{2}/hk" + code[:5] + r"\.js", parsed.path)
        ):
            results.append({"code": code, "status": "HK_SOURCE_IDENTITY_REVIEW_REQUIRED"})
            continue
        url = re.sub(r"/daily/\d{2}/", "/daily/" + base[2:4] + "/", old["url"])
        cache = Path(root) / "hk-receipts" / base / (code + ".json")
        try:
            if cache.exists():
                receipt = read(cache)
                raw = Path(receipt["raw_path"]).read_bytes()
                if content_digest(raw) != receipt["raw_sha256"]:
                    raise ValueError("HK_HISTORY_RAW_CHANGED")
            else:
                with httpx.Client(timeout=httpx.Timeout(15, connect=4), trust_env=False) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    raw = response.content
                if len(raw) > 100000:
                    raise ValueError("HK_HISTORY_SIZE")
                closed_rows(raw, base)
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
                }
                versioned_save(cache, receipt)
            rows = closed_rows(raw, base)
            previous, last = list(rows.values())
            body = json.dumps(rows, ensure_ascii=False, sort_keys=True)
            ref = "hk-close:" + code + ":" + receipt["raw_sha256"]
            document = {
                "id": ref,
                "kind": "MARKET_DATA",
                "title": "本港股已收盘历史日线 " + code,
                "url": url,
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
            company["quote"] = {
                "date": base,
                "close": last["close"],
                "previous_close": previous["close"],
                "change_pct": str((Decimal(last["close"]) / Decimal(previous["close"]) - 1) * 100),
                "source_ref": ref,
                "currency": "HKD",
                "valuation_market": "HK",
                "session": "REGULAR_CLOSE",
                "closed_at": base + "T16:10:00+08:00",
                "close_timestamp_kind": "EXCHANGE_COMPLETION_BOUND",
            }
            results.append({"code": code, "status": "SUCCEEDED", "source": receipt})
        except Exception as error:
            results.append({"code": code, "status": "ACTION_REQUIRED", "failure": str(error)[:250]})
    return results
