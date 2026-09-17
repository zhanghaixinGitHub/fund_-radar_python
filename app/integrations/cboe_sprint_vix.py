"""个人一日研究的官方VIX只读客户端；固定地址、响应大小与时间上限，不自动重试。"""

from time import monotonic

import httpx

URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
MAX_BYTES = 2_000_000


def fetch_vix() -> tuple[bytes, dict]:
    """返回原始CSV与公开响应头；调用方须先登记请求次数，并记录实际接收时刻。"""
    began = monotonic()
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
        with client.stream("GET", URL) as response:
            if response.status_code != 200:
                raise ValueError("VIX_PROVIDER_HTTP_FAILED")
            body = bytearray()
            for chunk in response.iter_bytes(65536):
                body.extend(chunk)
                if len(body) > MAX_BYTES or monotonic() - began > 30:
                    raise ValueError("VIX_RESPONSE_LIMIT")
            headers = {
                key: response.headers.get(key)
                for key in ("Date", "Last-Modified", "ETag", "Content-Type", "Content-Length")
            }
    return bytes(body), headers
