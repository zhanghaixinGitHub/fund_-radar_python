"""个人一日研究的官方ECB参考汇率只读客户端；固定地址、响应大小与时间上限，不自动重试。"""

from time import monotonic

import httpx

URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
MAX_BYTES = 5_000_000


def fetch_ecb_fx() -> tuple[bytes, dict]:
    """返回原始ZIP与公开响应头；调用方须先登记请求次数，并记录实际接收时刻。"""
    began = monotonic()
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
        with client.stream("GET", URL) as response:
            if response.status_code != 200:
                raise ValueError("ECB_PROVIDER_HTTP_FAILED")
            body = bytearray()
            for chunk in response.iter_bytes(65536):
                body.extend(chunk)
                if len(body) > MAX_BYTES or monotonic() - began > 45:
                    raise ValueError("ECB_RESPONSE_LIMIT")
            headers = {
                key: response.headers.get(key)
                for key in ("Date", "Last-Modified", "ETag", "Content-Type", "Content-Length")
            }
    return bytes(body), headers
