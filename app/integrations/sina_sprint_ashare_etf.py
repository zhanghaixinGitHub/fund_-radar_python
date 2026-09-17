"""一日研究用独立沪深300海外ETF日线入口：仅ASHR，原有行情入口保持冻结。"""

import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
from time import monotonic

import httpx

SYMBOLS = ("ASHR",)
URL_TEMPLATE = "https://finance.sina.com.cn/staticdata/us/{symbol}"
MAX_BYTES = 200_000


def url_for(symbol: str) -> str:
    if symbol not in SYMBOLS:
        raise ValueError("US_ETF_SYMBOL_INVALID")
    return URL_TEMPLATE.format(symbol=symbol)


def fetch_etf(symbol: str) -> tuple[bytes, dict]:
    """单次只读GET，20秒总时限、200KB上限；无凭据、跳转或重试。"""
    url = url_for(symbol)
    started = monotonic()
    with httpx.Client(timeout=httpx.Timeout(15, connect=5), follow_redirects=False) as client:
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise ValueError("US_ETF_HTTP_FAILED")
            body = bytearray()
            for chunk in response.iter_bytes(65536):
                body.extend(chunk)
                if len(body) > MAX_BYTES or monotonic() - started > 20:
                    raise ValueError("US_ETF_RESPONSE_LIMIT")
            headers = {key: response.headers.get(key) for key in ("Date", "Last-Modified", "ETag", "Content-Type")}
    return bytes(body), headers


def decode(raw: bytes, symbol: str) -> list[dict]:
    """压缩价格是数据；仅把严格匹配的字符串送入隔离解码器，绝不执行响应里的JS语句。"""
    url_for(symbol)
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError("US_ETF_RAW_SIZE_INVALID")
    if not re.fullmatch(rb"\s*var KLC_K2_" + symbol.encode("ascii") + rb'="[A-Za-z0-9+/]+";\s*', raw):
        raise ValueError("US_ETF_DATA_ENVELOPE_INVALID")
    node = shutil.which("node")
    if node is None:
        raise ValueError("US_ETF_NODE_RUNTIME_MISSING")
    try:
        completed = subprocess.run(
            [
                node,
                "--max-old-space-size=128",
                str(Path(__file__).with_name("sina_sprint_ashare_decode.cjs")),
                symbol,
            ],
            input=raw,
            capture_output=True,
            timeout=10,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        values = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError("US_ETF_DECODE_FAILED") from exc
    if not isinstance(values, list) or not 1 <= len(values) <= 10000:
        raise ValueError("US_ETF_DECODED_ROWS_INVALID")
    return values


def parse(raw: bytes, symbol: str, end: str = "2026-12-31") -> dict:
    """保留2021起价格及OHLC质量标记；异常日不修造价格，特征层明确回退。

    文档称无复权；本分支只使用供应商原始开/收盘价变化，不能称总回报。
    OHLC范围异常可涉及四舍五入或来源不一致，因此标记整行不可用于新特征。
    """
    import math

    cutoff = date.fromisoformat(end).isoformat()
    points, previous_day = {}, ""
    for row in decode(raw, symbol):
        if not isinstance(row, dict) or not isinstance(row.get("date"), str):
            raise ValueError("US_ETF_ROW_INVALID")
        stamp = row["date"]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T00:00:00\.000Z", stamp):
            raise ValueError("US_ETF_DATE_INVALID")
        day = date.fromisoformat(stamp[:10]).isoformat()
        if day <= previous_day:
            raise ValueError("US_ETF_DUPLICATE_OR_UNSORTED_DATE")
        previous_day = day
        if day < "2021-01-01":
            continue
        if day > cutoff:
            raise ValueError("US_ETF_FUTURE_ROW")
        if any(
            type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
            for k in ("open", "high", "low", "close", "volume")
        ):
            raise ValueError("US_ETF_PRICE_NONFINITE")
        if min(row[k] for k in ("open", "high", "low", "close")) <= 0 or row["volume"] < 0:
            raise ValueError("US_ETF_PRICE_NONPOSITIVE")
        valid = (
            row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]
            and row["volume"] > 0
        )
        points[day] = {k: row[k] for k in ("open", "high", "low", "close", "volume")}
        points[day].update(available=valid, unavailable_reason=None if valid else "OHLC_RANGE_OR_NO_TRADE")
    return points
