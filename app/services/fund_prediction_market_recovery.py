"""补齐综合包已声明的境内对照行情；不猜测代码、不挪用旧报价日期。"""

import json
import math
import re
from datetime import date

from app.core.config import get_settings
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT
from app.services.fund_exposure_quotes import Provider

FIELDS = ["ts_code", "trade_date", "close", "pre_close", "pct_chg", "amount"]


def quote_row(value, code, base, api):
    """必须取得同一证券同一基准日的一条真实价格；空值不能沿用旧数。"""
    data = value.get("data") or {}
    if value.get("code") != 0 or data.get("fields") != FIELDS:
        raise ValueError("REFERENCE_QUOTE_SCHEMA")
    rows = data.get("items", [])
    if len(rows) != 1 or len(rows[0]) != len(FIELDS):
        raise ValueError("REFERENCE_QUOTE_MISSING_OR_DUPLICATE")
    row = dict(zip(FIELDS, rows[0], strict=True))
    if row["ts_code"] != code or row["trade_date"] != date.fromisoformat(base).strftime("%Y%m%d"):
        raise ValueError("REFERENCE_QUOTE_IDENTITY_OR_DATE")
    if any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) for k in FIELDS[2:5]):
        raise ValueError("REFERENCE_QUOTE_NUMERIC")
    amount = row["amount"]
    if amount is not None and (type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0):
        raise ValueError("REFERENCE_QUOTE_NUMERIC")
    if row["close"] <= 0 or row["pre_close"] <= 0:
        raise ValueError("REFERENCE_QUOTE_VALUE")
    # 场内基金的涨幅通常保留两位，指数保留四位；保留供应商数值及原响应。
    tolerance = 0.011 if api == "fund_daily" else 0.0002
    if abs(100 * (row["close"] / row["pre_close"] - 1) - row["pct_chg"]) > tolerance:
        raise ValueError("REFERENCE_QUOTE_RETURN_MISMATCH")
    return row


def refresh(material, base, *, provider=None):
    if not material.get("market"):
        return []
    provider = provider or Provider()
    receipts = []
    for code, previous in material.get("market", {}).items():
        if previous.get("date") == base:
            continue
        if re.fullmatch(r"0\d{5}\.(SH|CSI)|H\d{5}\.CSI", code):
            api = "index_daily"
        elif re.fullmatch(r"[15]\d{5}\.(SH|SZ)", code):
            api = "fund_daily"
        else:
            receipts.append({"code": code, "status": "MARKET_SPECIFIC_SOURCE_REQUIRED"})
            continue
        try:
            raw, receipt = provider.query(api, {"ts_code": code, "trade_date": base.replace("-", "")}, FIELDS)
            row = quote_row(raw, code, base, api)
            ref = "reference-quote:" + code + ":" + receipt["sha256"]
            body = json.dumps(row, ensure_ascii=False, sort_keys=True)
            document = {
                "id": ref,
                "kind": "MARKET_DATA",
                "title": "本对照证券实际基准日行情",
                "url": get_settings().tushare_api_url,
                "body": body,
                "body_hash": digest(body),
                "body_scope": "VERIFIED_SOURCE_TRANSFORM",
                "published_date": base,
                "available_at": receipt["received_at"],
                "codes": [material["fund_code"]],
                "raw_path": str((ROOT / receipt["file"]).resolve()),
                "raw_sha256": receipt["sha256"],
            }
            if not any(d["id"] == ref for d in material["documents"]):
                material["documents"].append(document)
            material["market"][code] = {
                **{
                    k: previous[k]
                    for k in ("valuation_market", "currency", "session", "close_timestamp_kind")
                    if k in previous
                },
                "date": base,
                "close": row["close"],
                "previous_close": row["pre_close"],
                "change_pct": row["pct_chg"],
                "turnover_thousand_cny": row["amount"],
                "source_ref": ref,
            }
            if previous.get("valuation_market") == "CN":
                material["market"][code]["closed_at"] = base + "T15:00:00+08:00"
            fact = "current_reference_quote:" + code
            material["facts"][fact] = {"category": "MARKET", "source_refs": [ref], "quotes": [body]}
            receipts.append({"code": code, "status": "SUCCEEDED", "source": receipt})
        except Exception as error:
            receipts.append({"code": code, "status": "ACTION_REQUIRED", "failure": str(error)[:250]})
    from app.services.fund_prediction_corporate_recovery import verified_codes

    non_trading = verified_codes(material, base) if material.get("market_quote_exceptions") else set()
    quotes = [
        *(c.get("quote") or {} for c in material.get("companies", []) if c["code"] not in non_trading),
        *material.get("market", {}).values(),
    ]
    if quotes and all(q.get("date") == base and q.get("change_pct") is not None for q in quotes):
        refs = [q.get("source_ref") for q in material["market"].values() if q.get("source_ref")]
        if refs:
            material["comprehensive_coverage"]["MARKET"] = {"status": "AVAILABLE", "source_refs": refs}
        documents = {d["id"]: d for d in material["documents"]}
        partial = material.get("source_manifest", {}).get("latest_partial_report", {}).get("report", {})
        selected = {h["stockCode"] for h in partial.get("holdings", [])[:5]}
        holding_quotes = [
            c.get("quote") or {}
            for c in material.get("companies", [])
            if c["code"] in selected and c["code"] not in non_trading
        ]
        for key, values in (
            ("current_market", list(material.get("market", {}).values())),
            ("current_holdings", holding_quotes),
        ):
            if key not in material["facts"]:
                continue
            refs = list(dict.fromkeys(q.get("source_ref") for q in values if q.get("source_ref") in documents))
            if refs and len(refs) == len(values):
                # 重建事实引用，旧版事实仍保留在旧资料包；不把旧涨幅改日期后沿用。
                material["facts"][key] = {
                    **material["facts"][key],
                    "category": "MARKET",
                    "source_refs": refs,
                    "quotes": [documents[ref]["body"] for ref in refs],
                }
    return receipts
