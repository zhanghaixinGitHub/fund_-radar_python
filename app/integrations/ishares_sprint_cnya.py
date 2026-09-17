"""CNYA 美国基金发行方的公开历史文件；地址取自已核实产品页下载链接。"""

from time import monotonic

import httpx

URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document"
    "?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US"
    "&portfolioId=273318&component=fundDownload&userType=individual"
)
MAX_BYTES = 8_000_000


def fetch_cnya() -> tuple[bytes, dict]:
    """单次只读下载，最多8MB和45秒，不自动重试或跳转；调用方持久化实际请求时间。"""
    started = monotonic()
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
        with client.stream("GET", URL) as response:
            if response.status_code != 200:
                raise ValueError("CNYA_HTTP_FAILED")
            body = bytearray()
            for chunk in response.iter_bytes(65536):
                body.extend(chunk)
                if len(body) > MAX_BYTES or monotonic() - started > 45:
                    raise ValueError("CNYA_RESPONSE_LIMIT")
            headers = {k: response.headers.get(k) for k in ("Date", "Last-Modified", "ETag", "Content-Type")}
    return bytes(body), headers
