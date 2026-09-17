"""美元高收益债和国债ETF的同日相对价格输入；不把价格差冒充纯信用利差。"""

import hashlib
import json
import math
from datetime import datetime
from html.parser import HTMLParser

from app.integrations import sina_sprint_credit_pair_etf as api
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_dollar_etf_data as single

IDENTITIES = {"HYG": ("464288513", "NYSE Arca"), "IEF": ("464287440", "NASDAQ")}


def root():
    return b.ROOT / "credit-pair-etf-feasibility-v1"


def require(condition, reason):
    if not condition:
        raise ValueError("CREDIT_PAIR_ETF_" + reason)


class IssuerFields(HTMLParser):
    """只解析发行方页面已有JSON属性，不运行脚本、不推测NAV或中间价为收盘价。"""

    def __init__(self):
        super().__init__()
        self.matches = []

    def handle_starttag(self, tag, attrs):
        if tag != "walrus-render-on-client":
            return
        for name, value in attrs:
            if name != "componentprops" or not value:
                continue
            try:
                payload = json.loads(value)
            except ValueError:
                continue
            points = payload.get("dataPoints") if isinstance(payload, dict) else None
            if isinstance(points, dict) and "closingPrice" in points:
                self.matches.append(points)


def issuer_observation(raw, symbol):
    require(symbol in IDENTITIES, "SYMBOL_INVALID")
    parser = IssuerFields()
    parser.feed(raw.decode("utf-8"))
    require(len(parser.matches) == 1, "ISSUER_PRICE_AMBIGUOUS")
    points = parser.matches[0]
    close, cusip, exchange, volume = (points[k] for k in ("closingPrice", "cusip", "exchange", "consolidatedVolume"))
    require(
        close["name"] == "closingPrice" and close["label"] == "Closing Price" and close["prefix"] == "$",
        "ISSUER_PRICE_KIND_CHANGED",
    )
    require((cusip["formattedValue"], exchange["formattedValue"]) == IDENTITIES[symbol], "ISSUER_IDENTITY_CHANGED")
    day = datetime.strptime(close["formattedAsOfDate"], "%b %d, %Y").date().isoformat()
    require(volume["formattedAsOfDate"] == close["formattedAsOfDate"], "ISSUER_VOLUME_DATE_CHANGED")
    value = float(close["formattedValue"].replace(",", ""))
    amount = float(volume["formattedValue"].replace(",", ""))
    require(math.isfinite(value) and value > 0 and math.isfinite(amount) and amount > 0, "ISSUER_VALUE_INVALID")
    return {"symbol": symbol, "date": day, "close": value, "volume": amount, "cusip": IDENTITIES[symbol][0]}


def features(t, u, points):
    """新增美国会话逐日开收对数收益，单位百分数；两腿都完整才可使用。

    第一个量是HYG收益减IEF收益，第二个量是IEF收益；分别限±20。
    两腿未做久期匹配，因此前者仅为相对价格变化。逐日比值不包含隔夜除息缺口。
    缺一腿、无新会话、任何OHLC异常均明确不可用，0值不能脱离available标记使用。
    """
    legs = {symbol: single.features(t, u, points.get(symbol, {})) for symbol in api.SYMBOLS}
    days = legs["HYG"]["new_us_dates"]
    require(legs["IEF"]["new_us_dates"] == days, "SESSION_ALIGNMENT_CHANGED")
    empty = {"available": False, "relative_log_pct": 0.0, "treasury_log_pct": 0.0, "new_us_dates": days}
    for symbol, feature in legs.items():
        if not feature["available"]:
            return empty | {"reason": symbol + "_" + feature["reason"]}
    raw = {
        symbol: 100 * sum(math.log(points[symbol][d]["close"] / points[symbol][d]["open"]) for d in days)
        for symbol in api.SYMBOLS
    }
    return {
        "available": True,
        "relative_log_pct": min(20.0, max(-20.0, raw["HYG"] - raw["IEF"])),
        "treasury_log_pct": min(20.0, max(-20.0, raw["IEF"])),
        "new_us_dates": days,
        "reason": None,
    }


def reconstruct():
    """四份已实际取得的原文重新核验；不同日期发行方收盘分别与供应商相同日期比较。"""
    p = b.read(root() / "qualification-plan.json")
    probe = b.read(root() / "probe-plan.json")
    require(b.digest(probe) == p["probe_plan_hash"], "PROBE_PLAN_CHANGED")
    for mapping in (probe["code"], p["code_hashes"]):
        for name, expected in mapping.items():
            require(hashlib.sha256((b.PROJECT / name).read_bytes()).hexdigest() == expected, "SOURCE_CODE_CHANGED")
    for query in probe["queries"]:
        key = query["key"]
        request = b.read(root() / "requests" / (key + ".json"))
        receipt = b.read(root() / "responses" / (key + ".json"))
        raw = (root() / "raw" / (key + ".bin")).read_bytes()
        expected = p["sources"][key]
        require(
            request["query"] == query and request["plan_hash"] == b.digest(probe) and request["attempt"] == 1,
            "REQUEST_CHANGED",
        )
        require(
            b.digest(request) == expected["request_hash"]
            and b.digest(receipt) == expected["receipt_hash"]
            and receipt["request_hash"] == b.digest(request)
            and receipt["http_status"] == 200
            and receipt["bytes"] == len(raw) <= query["max_bytes"]
            and receipt["sha256"] == expected["raw_sha256"] == hashlib.sha256(raw).hexdigest(),
            "RAW_OR_RECEIPT_CHANGED",
        )
        require(datetime.fromisoformat(request["at"]) <= datetime.fromisoformat(receipt["at"]), "SOURCE_TIME_CHANGED")
    points = {}
    for symbol in api.SYMBOLS:
        quotes = api.parse((root() / f"raw/sina-{symbol}.bin").read_bytes(), symbol, p["cutoff"])
        require(quotes == b.read(root() / f"parsed-{symbol}.json")["points"], "PARSED_HISTORY_CHANGED")
        observed = issuer_observation((root() / f"raw/issuer-{symbol}.bin").read_bytes(), symbol)
        require(observed == p["issuer_observations"][symbol], "ISSUER_OBSERVATION_CHANGED")
        require(quotes[observed["date"]]["close"] == observed["close"], "ISSUER_CLOSE_MISMATCH")
        points[symbol] = quotes
    return points


def history():
    p, result, snapshot = (
        b.read(root() / n) for n in ("qualification-plan.json", "qualification-result.json", "history.json")
    )
    require(
        result["status"] == "QUALIFIED_INTRADAY_RESEARCH_WITH_LIMITS"
        and result["plan_hash"] == snapshot["plan_hash"] == b.digest(p)
        and result["history_hash"] == b.digest(snapshot),
        "HISTORY_MANIFEST_CHANGED",
    )
    require(reconstruct() == snapshot["points"], "HISTORY_POINTS_CHANGED")
    return snapshot


def extend(market, t, u, points):
    return market | {"credit_pair": features(t, u, points)}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "credit_pair"}}
