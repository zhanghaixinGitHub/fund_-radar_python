"""按本期实际日期补上海黄金交易所现货行情，不用ETF场内价替代现货。"""

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now
from app.services.fund_materials_store import versioned_save
from app.services.material_content import content_digest, source_bytes


def gold_rows(raw, base):
    """只接受本期Au99.99实际收盘行，其他合约或旧日不能冒充本期。"""
    soup = BeautifulSoup(raw, "html.parser")
    rows = [[cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])] for tr in soup.select("tr")]
    selected = [row for row in rows if len(row) >= 8 and row[0] == base and row[1] == "Au99.99"]
    if len(selected) != 1:
        raise ValueError("GOLD_REQUIRED_DATE_NOT_RETURNED")
    row = selected[0]
    opening, high, low, close, change = [Decimal(x.replace(",", "")) for x in row[2:7]]
    pct = Decimal(row[7].removesuffix("%").replace(",", ""))
    if not all(v.is_finite() for v in [opening, high, low, close, change, pct]):
        raise ValueError("GOLD_SOURCE_NONFINITE")
    if min(opening, high, low, close) <= 0 or high < max(opening, low, close) or low > min(opening, close):
        raise ValueError("GOLD_SOURCE_PRICE_RANGE")
    previous = close - change
    if previous <= 0 or abs(change / previous * 100 - pct) > Decimal(".02"):
        raise ValueError("GOLD_SOURCE_RETURN_MISMATCH")
    return row, soup.get_text(" ", strip=True)


def refresh(material, base, output):
    fact = material.get("facts", {}).get("current_gold")
    if not fact:
        return {"status": "NOT_APPLICABLE", "attempts": []}
    documents = {d["id"]: d for d in material["documents"]}
    prior = [documents[ref] for ref in fact.get("source_refs", []) if ref in documents]
    if not any(urlparse(d["url"]).hostname in {"www.sge.com.cn", "sge.com.cn"} for d in prior):
        raise ValueError("GOLD_OFFICIAL_SOURCE_REVIEW_REQUIRED")
    for document in prior:
        observation = document.get("gold_observation", {})
        if observation.get("valuation_date") == base and observation.get("contract") == "Au99.99":
            raw = source_bytes(Path(document["raw_path"]))
            if content_digest(raw) != document["raw_sha256"]:
                raise ValueError("GOLD_CACHED_RAW_CHANGED")
            row, _ = gold_rows(raw, base)
            if row != observation.get("source_row"):
                raise ValueError("GOLD_CACHED_ROW_CHANGED")
            return {"status": "CURRENT_SOURCE_RETAINED", "attempts": []}
    output = Path(output)
    attempts = []
    start = str(date.fromisoformat(base) - timedelta(days=7))
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), trust_env=False, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.sge.com.cn/"}) as client:
        for host in ["www.sge.com.cn", "sge.com.cn"]:
            url = f"https://{host}/sjzx/quotation_daily_new?start_date={start}&end_date={base}"
            receipt = {"url": url, "received_at": now().isoformat(), "required_date": base}
            try:
                response = client.get(url)
                if response.url.scheme != "https" or response.url.host not in {"www.sge.com.cn", "sge.com.cn"}:
                    raise ValueError("GOLD_OFFICIAL_REDIRECT_SCOPE")
                raw = response.content
                if len(raw) > 3_000_000:
                    raise ValueError("GOLD_RESPONSE_SIZE")
                sha = content_digest(raw)
                path = output / "raw" / (sha + ".html")
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(raw)
                receipt.update(status=response.status_code, raw_path=str(path.resolve()), raw_sha256=sha)
                response.raise_for_status()
                row, body = gold_rows(raw, base)
                reference = "sge-gold:" + sha
                document = {
                    "id": reference, "kind": "MARKET_DATA", "title": "上金所本期Au99.99实际现货行情",
                    "url": url, "published_date": base, "available_at": receipt["received_at"],
                    "body": body, "body_hash": digest(body), "body_scope": "FULL_TEXT",
                    "codes": [material["fund_code"]], "raw_path": str(path.resolve()), "raw_sha256": sha,
                    "gold_observation": {"valuation_date": base, "contract": "Au99.99", "source_row": row},
                }
                if reference not in documents:
                    material["documents"].append(document)
                # 保留旧原件，当前事实仅引用这次真正取得的完整页面。
                material["facts"]["current_gold"] = {
                    **fact, "source_refs": [reference], "quotes": [body], "text": body,
                }
                receipt.update(result="ACQUIRED", row=row)
                attempts.append(receipt)
                versioned_save(output / "attempts" / (digest(receipt) + ".json"), receipt)
                return {"status": "ACQUIRED", "attempts": attempts}
            except (httpx.HTTPError, ValueError, ArithmeticError) as error:
                receipt.update(result="WAITING_SOURCE", failure=type(error).__name__ + ":" + str(error)[:180])
                attempts.append(receipt)
                versioned_save(output / "attempts" / (digest(receipt) + ".json"), receipt)
    return {"status": "WAITING_SOURCE", "attempts": attempts}
