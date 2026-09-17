"""黄金ETF的独立研究输入；只复用已取得的GLD日线，不接受未核验的铜价格。"""

import math
import re
from datetime import datetime

from app.integrations import sina_sprint_commodity_pair_etf as api
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_core_forward as core
from app.services import direction_1d_sprint_dollar_etf_data as single


def root():
    return b.ROOT / "gold-etf-feasibility-v1"


def require(condition, reason):
    if not condition:
        raise ValueError("GOLD_ETF_" + reason)


def issuer_observation(raw):
    """解析SSGA明确标注的主上市交易所收盘价及该表日期；NAV、中点不能替代。

    此处只证明近期收盘样本相等。官方主交易所高点与供应商全市场高点存在差异，
    不宣称所有OHLC/成交量、历史复权规则或首次公布时刻均独立核验。
    """
    text = raw.decode("utf-8")
    require('<meta name="ISIN" content="US78463V1070"' in text, "ISSUER_IDENTITY_CHANGED")
    require('<meta name="ticker" content="GLD' in text, "ISSUER_TICKER_CHANGED")
    sections = re.findall(r"<section\b[^>]*>(.*?)</section>", text, re.S)
    selected = [section for section in sections if re.search(r"<h2[^>]*>\s*Fund Market Price\b", section)]
    require(len(selected) == 1, "ISSUER_PRICE_SECTION_AMBIGUOUS")
    section = selected[0]
    dates = re.findall(r'<span class="date">as of ([A-Za-z]{3} \d{1,2} \d{4})</span>', section)
    prices = re.findall(r'<th[^>]*>\s*Closing Price\b.*?</th>\s*<td class="data">\$([\d,.]+)</td>', section, re.S)
    require(len(dates) == len(prices) == 1, "ISSUER_DATE_OR_CLOSE_AMBIGUOUS")
    day = datetime.strptime(dates[0], "%b %d %Y").date().isoformat()
    close = float(prices[0].replace(",", ""))
    require(math.isfinite(close) and close > 0, "ISSUER_CLOSE_INVALID")
    return {"symbol": "GLD", "isin": "US78463V1070", "date": day, "close": close, "kind": "EXCHANGE_CLOSING_PRICE"}


def features(t, u, points):
    """沿用已冻结的严格美股会话对齐及同日开收对数百分比，缺失不前填。"""
    return single.features(t, u, points)


def reconstruct():
    """从两份原始实收重建，逐项绑定来源URL、请求、响应、解析器和独立收盘样本。"""
    plan = b.read(root() / "qualification-plan.json")
    for name, digest in plan["code_hashes"].items():
        require(core.sha(b.PROJECT / name) == digest, "SOURCE_CODE_CHANGED")
    for source in plan["sources"]:
        source_root = b.ROOT / source["root"]
        source_plan = b.read(source_root / source["plan_file"])
        require(b.digest(source_plan) == source["plan_hash"], "SOURCE_PLAN_CHANGED")
        request = b.read(source_root / "requests" / (source["key"] + ".json"))
        receipt = b.read(source_root / "responses" / (source["key"] + ".json"))
        raw = source_root / "raw" / (source["key"] + ".bin")
        require(
            request["query"] in source_plan["queries"]
            and request["query"]["url"] == source["url"]
            and request["query"]["key"] == source["key"]
            and request["plan_hash"] == source["plan_hash"]
            and request["attempt"] == 1,
            "SOURCE_REQUEST_CHANGED",
        )
        require(
            b.digest(request) == source["request_hash"] == receipt["request_hash"]
            and b.digest(receipt) == source["receipt_hash"]
            and receipt["http_status"] == 200
            and receipt["sha256"] == source["raw_sha256"] == core.sha(raw)
            and receipt["bytes"] == raw.stat().st_size <= request["query"]["max_bytes"],
            "SOURCE_RAW_OR_RECEIPT_CHANGED",
        )
        require(datetime.fromisoformat(request["at"]) <= datetime.fromisoformat(receipt["at"]), "SOURCE_TIME_CHANGED")
    commodity = b.ROOT / "commodity-pair-etf-feasibility-v1"
    points = api.parse((commodity / "raw/sina-GLD.bin").read_bytes(), "GLD", plan["cutoff"])
    require(points == b.read(commodity / "parsed-GLD.json")["points"], "PARSED_HISTORY_CHANGED")
    observation = issuer_observation((commodity / "primary-check-v1/raw/ssga-GLD.bin").read_bytes())
    require(observation == plan["issuer_observation"], "ISSUER_OBSERVATION_CHANGED")
    require(points[observation["date"]]["close"] == observation["close"], "ISSUER_CLOSE_MISMATCH")
    return points


def history():
    plan, result, snapshot = (
        b.read(root() / name) for name in ("qualification-plan.json", "qualification-result.json", "history.json")
    )
    require(
        result["status"] == "QUALIFIED_INTRADAY_RESEARCH_WITH_LIMITS"
        and result["plan_hash"] == snapshot["plan_hash"] == b.digest(plan)
        and result["history_hash"] == b.digest(snapshot),
        "HISTORY_MANIFEST_CHANGED",
    )
    require(reconstruct() == snapshot["points"], "HISTORY_POINTS_CHANGED")
    return snapshot


def extend(market, t, u, points):
    return market | {"gold_etf": features(t, u, points)}


def original_row(row):
    return row | {"market": {key: value for key, value in row["market"].items() if key != "gold_etf"}}
