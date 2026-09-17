"""三天研究的亚洲开盘报价解析；只读字符串，严禁执行供应商JavaScript。"""

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
SYMBOLS = ("NKY", "KOSPI")
NAMES = {
    "NKY": {"日经225指数", "日经225", "日经指数"},
    "KOSPI": {"韩国KOSPI指数", "首尔综合", "韩国综合指数"},
}
ENVELOPE = re.compile(r'var hq_str_znb_(NKY|KOSPI)="([^"\\\r\n]*)";')


def decode(raw):
    """最多两条固定代码报价，GB18030严格解码；变量外多余语句、转义和重复代码均拒绝。"""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 32768:
        raise ValueError("ASIA_QUOTE_SIZE_INVALID")
    text = raw.decode("gb18030").strip()
    values = {}
    position = 0
    while position < len(text):
        match = ENVELOPE.match(text, position)
        if not match or match[1] in values:
            raise ValueError("ASIA_QUOTE_ENVELOPE_INVALID")
        fields = match[2].split(",")
        if not 12 <= len(fields) <= 32:
            raise ValueError("ASIA_QUOTE_FIELDS_INVALID")
        values[match[1]] = fields
        position = match.end()
        while position < len(text) and text[position].isspace():
            position += 1
    if set(values) != set(SYMBOLS):
        raise ValueError("ASIA_QUOTE_SYMBOL_MISSING")
    return values


def opening(symbol, fields, target, received_at):
    """按已保存的官方hq.gi字段取开盘8/前收9；行情时间按页面说明为北京时间。

    仅接受目标日08:00以后、实际接收以前的行情，实际接收还须早于08:30。
    返回百分比跳空供研究；不含目标基金结果，不把盘中当前价替代开盘价。
    """
    if symbol not in SYMBOLS or not 12 <= len(fields) <= 32 or fields[0] not in NAMES[symbol]:
        raise ValueError("ASIA_QUOTE_IDENTITY_INVALID")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields[6]) or not re.fullmatch(r"\d{2}:\d{2}:\d{2}", fields[7]):
        raise ValueError("ASIA_QUOTE_DATETIME_INVALID")
    received = datetime.fromisoformat(received_at)
    if received.tzinfo is None:
        raise ValueError("ASIA_RECEIPT_TIMEZONE_MISSING")
    received = received.astimezone(SHANGHAI)
    quote = datetime.fromisoformat(fields[6] + "T" + fields[7]).replace(tzinfo=SHANGHAI)
    earliest = datetime.fromisoformat(target + "T08:00:00").replace(tzinfo=SHANGHAI)
    deadline = datetime.fromisoformat(target + "T08:30:00").replace(tzinfo=SHANGHAI)
    if fields[6] != target or not earliest <= quote <= received < deadline:
        raise ValueError("ASIA_QUOTE_NOT_TIMELY_CURRENT_TARGET")
    prices = {
        key: float(fields[index])
        for key, index in {"current": 1, "open": 8, "pre_close": 9, "high": 10, "low": 11}.items()
    }
    if not all(math.isfinite(x) and x > 0 for x in prices.values()):
        raise ValueError("ASIA_PRICE_INVALID")
    if (
        not prices["low"]
        <= min(prices["open"], prices["current"])
        <= max(prices["open"], prices["current"])
        <= prices["high"]
    ):
        raise ValueError("ASIA_OHLC_RANGE_INVALID")
    return {
        "symbol": symbol,
        "name": fields[0],
        "date": target,
        "quote_at": quote.isoformat(),
        "received_at": received.isoformat(),
        "open": prices["open"],
        "pre_close": prices["pre_close"],
        "opening_gap_pct": 100 * (prices["open"] / prices["pre_close"] - 1),
    }
