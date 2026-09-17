"""FXI发行方公开下载的只读客户端；固定已验证地址，不请求或推断交易所价格。"""

from time import monotonic

import httpx

URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document"
    "?appSubType=ISHARES&appType=PRODUCT_PAGE&component=fundDownload&locale=en_US"
    "&portfolioId=239536&targetSite=us-ishares&userType=individual"
)
MAX_BYTES = 8_000_000


def fetch_fxi() -> tuple[bytes, dict]:
    """最多8MB、45秒，无自动重试和重定向；调用方保存时槽、原始字节及实际接收证据。"""
    started = monotonic()
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
        with client.stream("GET", URL) as response:
            if response.status_code != 200:
                raise ValueError("FXI_HTTP_FAILED")
            body = bytearray()
            for chunk in response.iter_bytes(65536):
                body.extend(chunk)
                if len(body) > MAX_BYTES or monotonic() - started > 45:
                    raise ValueError("FXI_RESPONSE_LIMIT")
            headers = {k: response.headers.get(k) for k in ("Date", "Last-Modified", "ETag", "Content-Type")}
    return bytes(body), headers
