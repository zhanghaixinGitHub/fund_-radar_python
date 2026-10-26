"""经理资料的有界供应商适配；分页原件由上层留档，不能只取第一页。"""

import hashlib
import json
import re
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup

from app.core.config import get_settings
from app.integrations.tushare import TushareFundClient, TushareIntegrationError

FIELDS = "ts_code,ann_date,name,edu,begin_date,end_date,resume"


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ManagerSource:
    """最多20页且每页不超5000；重复页或保护上限不作为完整资料返回。"""

    def __init__(self):
        config = get_settings()
        self.client = TushareFundClient(
            token=config.tushare_token.get_secret_value(),
            api_url=config.tushare_api_url,
            connect_timeout_seconds=5,
            read_timeout_seconds=20,
            max_retries=2,
            catalog_max_rows_per_query=5000,
            request_rate_limit_per_minute=60,
        )
        self.pages: list[dict] = []

    def close(self):
        self.client.close()

    def managers(self, *, name: str | None = None, code: str | None = None) -> list[dict]:
        records, seen = [], set()
        for page in range(20):
            parameters = {"offset": page * 5000, "limit": 5000}
            parameters.update({"name": name} if name else {"ts_code": code})
            rows = list(self.client._query("fund_manager", params=parameters, fields=FIELDS))
            fingerprint = digest(rows)
            if rows and fingerprint in seen:
                raise TushareIntegrationError("fund_manager", "分页重复，未能完整取得经理资料")
            seen.add(fingerprint)
            if len(rows) > 5000 or any(
                (name and r.get("name") != name) or (code and r.get("ts_code") != code) for r in rows
            ):
                raise TushareIntegrationError("fund_manager", "经理来源范围不匹配")
            self.pages.append(
                {
                    "api": "fund_manager",
                    "parameters": parameters,
                    "receivedAt": datetime.now(UTC).isoformat(),
                    "hash": fingerprint,
                    "rows": rows,
                }
            )
            records.extend(rows)
            if len(rows) < 5000:
                return records
        raise TushareIntegrationError("fund_manager", "经理资料超过本次分页保护范围")

    def basics(self, codes: tuple[str, ...]):
        return self.client.list_fund_basics_by_ts_codes(codes)


def official_dbfund_profile(code: str, name: str) -> dict | None:
    """只读德邦官方具体产品页，要求代码、经理名单和职业简介三个锚点同时成立。

    当前仅适配明确的官方产品目录，不接受客户端URL，不以同名搜索结果归并人员。
    其他公司继续保留待核对，不套用该站点规则。
    """
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("基金代码不合法")
    url = f"https://www.dbfund.com.cn/products/hunhe/{code}/index.html"
    with httpx.Client(timeout=httpx.Timeout(15, connect=5), follow_redirects=False) as client:
        response = client.get(url)
        if response.status_code != 200 or len(response.content) > 2_000_000:
            return None
        document = BeautifulSoup(response.content, "html.parser")
    for tag in document(["script", "style"]):
        tag.decompose()
    fund_match = False
    for row in document.select("tr"):
        text = row.get_text(" ", strip=True)
        if "基金代码" in text and code in text and "基金经理" in text and name in text:
            fund_match = True
    if not fund_match:
        return None
    for heading in document.select(".name"):
        if heading.get_text(strip=True) != name:
            continue
        brief = heading.find_next(class_="brief")
        resume = brief.get_text(" ", strip=True) if brief else ""
        # 要求可核对的教育与有年月的职业经历；姓名或空简介本身不是身份证据。
        if "毕业于" not in resume or len(re.findall(r"20\d{2}年\d{1,2}月", resume)) < 2:
            return None
        return {
            "url": url,
            "code": code,
            "name": name,
            "resume": resume,
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "receivedAt": datetime.now(UTC).isoformat(),
            "content": response.text,
        }
    return None
