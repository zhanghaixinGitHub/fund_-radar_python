"""008764 在 2026-11-19 用户授权的参考汇率预测，仅用于该窗口。"""

import json
import re
from decimal import Decimal


def apply_reference_review(data, material):
    review = material.get("prediction_reference_fx_review")
    if review is None:
        return
    window = data["window"]
    if (window["base_nav_date"], window["target_nav_date"]) != ("2026-11-18", "2026-11-19"):
        return
    status = data["valuation_status"]
    if data["fund_code"] != "008764" or not status["supported"] or not status["market_ready"]:
        raise ValueError("VN_REFERENCE_FX_SCOPE_INVALID")
    if status["fx_ready"]:
        return
    if [gap["kind"] for gap in status["gaps"]] != ["FX"]:
        raise ValueError("VN_REFERENCE_FX_OTHER_GAPS")
    documents = {doc["id"]: doc for doc in data["documents"]}
    previous_cny_source = documents["vn-reference:cny17"]
    if (
        previous_cny_source["url"] != "https://www.chinamoney.com.cn/chinese/bkccpr/index.html?tab=2"
        or previous_cny_source["published_date"] != "2026-11-17"
    ):
        raise ValueError("VN_REFERENCE_FX_SOURCE_INVALID")
    previous_cny = Decimal(re.search(r"2026-11-17\s*\|\s*(\d+\.\d+)", previous_cny_source["body"])[1])
    current_source = documents[data["calendar_evidence"]["fx_ref"]]
    current_cny, = (
        Decimal(row["cny_per_unit"]) for row in json.loads(current_source["body"])["rates"]
        if (row["date"], row["currency"]) == ("2026-11-18", "USD")
    )
    if min(previous_cny, current_cny) <= 0:
        raise ValueError("VN_REFERENCE_FX_QUOTE_INVALID")
    rows = []
    for short_day, day in (("17", "2026-11-17"), ("18", "2026-11-18")):
        bank, wise = documents[f"vn-reference:hsbc{short_day}"], documents[f"vn-reference:wise{short_day}"]
        if (
            bank["url"] != "https://www.business.hsbc.com.vn/-/media/media/vietnam/pdfs/daily-fx-rates/"
                           f"2026/hsbc-rates-{short_day}-nov-2026.pdf"
            or wise["url"] != f"https://wise.com/us/currency-converter/usd-to-vnd-rate/history/{short_day}-11-2026"
            or any(doc["published_date"] != day or doc["codes"] != ["008764"] for doc in (bank, wise))
            or f"Rates applicable on {short_day}-Nov-26 8:36 AM" not in bank["body"]
            or f"November {short_day}, 2026" not in wise["body"]
        ):
            raise ValueError("VN_REFERENCE_FX_SOURCE_INVALID")
        bid, ask = re.search(r"Corporate Transfer ([\d,]+) ([\d,]+)", bank["body"]).groups()
        mid = re.search(r"1 USD = ([\d,]+) VND", wise["body"])[1]
        central = re.search(r"Official USD/VND Rate ([\d,]+)", bank["body"])[1]
        buy, sell, mid, central = (Decimal(value.replace(",", "")) for value in (bid, ask, mid, central))
        if not 0 < buy < sell or min(mid, central) <= 0:
            raise ValueError("VN_REFERENCE_FX_QUOTE_INVALID")
        rows.append(((buy + sell) / 2, mid, central))
    directions = {}
    checks = review["四周期"]
    multipliers = [current_cny / current / (previous_cny / previous)
                   for previous, current in zip(*rows, strict=True)]
    for horizon in ("T1", "T5_V1", "T20_V1", "M6_V1"):
        bounds = [Decimal(value) for value in checks[horizon]["当地股市压力检验假设百分点"]]
        returns = [(1 + move / 100) * multiplier - 1 for move in bounds
                   for multiplier in [Decimal(1), *multipliers]]
        if len(bounds) != 2 or not (max(returns) < 0 or min(returns) > 0):
            raise ValueError("VN_REFERENCE_FX_DIRECTION_NOT_STABLE")
        directions[horizon] = "UP" if min(returns) > 0 else "DOWN"
    data["valuation_status"] = {
        **status, "ready": True, "scope": "CURRENT_PREDICTION_REFERENCE_RATE_ONLY",
        "prediction_reference_fx": {
            "date": "2026-11-18", "source": "汇丰越南对公转账买卖中点",
            "usd_vnd": str(rows[-1][0]), "cross_checked": ["Wise", "越南央行参考中间价"],
            "directions": directions, "reference_only": True,
        },
    }
