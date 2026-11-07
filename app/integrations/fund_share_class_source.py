"""从公开报告的身份表识别产品份额；名称只补充展示，不能用于猜测产品关系。"""

import json
import re
import time
import unicodedata

from bs4 import BeautifulSoup

from app.integrations.dbfund_reports import bounded_get
from app.integrations.market_fund_reports import MarketReportClient, read_source
from app.integrations.public_fund_reports import BODY_URL


def compact(value: str) -> str:
    """仅消除排版空白和全角字符，不删除业务名称或交易币种。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def parse_identity(body: str, code: str) -> tuple[str, tuple[str, ...]]:
    """只读取基金概况的身份表，拒绝缺失、重复、截断和目标 ETF 的代码。

    份额代码来自明确的交易代码行；主代码不能自动当作另一个份额。
    无分级表时，仅主代码与当前代码相同的完整概况可以确认单份额。
    """
    normalized = unicodedata.normalize("NFKC", body)
    starts = list(re.finditer(r"(?:基金(?:简介|产品概况|基本情况)|不动产基金产品基本情况)", normalized))
    for start in starts:
        section = normalized[start.end():]
        end = re.search(r"(?:2\.1\.1\s*目标基金|2\.2\s*|主要财务指标|§\s*3\s*)", section)
        if end is None:
            continue
        section = section[:end.start()]
        text = compact(section)
        main = re.search(r"基金(?:主代码|代码)[:：]?([0-9]{6})(?![0-9])", text)
        if main is None:
            continue
        # PDF 转载有时把行标题的“交易代码”拆到数字两侧。按每个下属字段分块，
        # 去除代码后必须恰好还原交易代码标题；前端/后端收费代码和名称块不能混入。
        lower = r"下\s*属\s*分\s*级\s*基\s*金"
        rows = []
        blocks = re.split(r"(?=" + lower + r")", section)
        for block in blocks:
            block = re.split(r"报告期末|注\s*[:：]", block)[0]
            numbers = re.findall(r"(?<![0-9])[0-9]{6}(?![0-9])", block)
            title = compact(re.sub(r"(?<![0-9])[0-9]{6}(?![0-9])", "", block)).rstrip(":：")
            if re.fullmatch(r"下属分级基金(?:的)?(?:交易|基金)代码", title) and numbers:
                rows.append(tuple(numbers))
        if len(rows) == 1:
            codes = rows[0]
            if not 1 <= len(codes) <= 30 or len(set(codes)) != len(codes) or code not in codes:
                raise ValueError("SHARE_IDENTITY_CODES_INVALID")
            return main[1], codes
        if not rows and "下属分级基金" not in text and main[1] == code and "基金管理人" in text:
            return main[1], (code,)
    raise ValueError("SHARE_IDENTITY_TABLE_UNAVAILABLE")


def parse_profile(raw: bytes, code: str) -> dict:
    """精确核对公开档案标题中的六位代码；未知类别保留名称，不猜为 A/C。"""
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    matched = re.match(r"(.+?)\(([0-9]{6})\)基金", unicodedata.normalize("NFKC", title))
    if not matched or matched[2] != code:
        raise ValueError("SHARE_PROFILE_IDENTITY_MISMATCH")
    name = compact(matched[1])
    # CNY 是币种，不是 Y 类；仅识别明确的末尾类别，不从产品简称中提取字母。
    label = re.search(
        r"(?<![A-Za-z])([A-Z])(?:类)?(?:[-(]?(人民币|美元现汇|美元现钞|美元|港币|CNY|USD|HKD)\)?)?$", name
    )
    prefix_currency = re.search(r"(人民币|美元现汇|美元现钞|美元|港币)(?:[-(]?)[A-Z]类?\)?$", name)
    return {
        "fundCode": code, "fundName": name,
        "shareClass": label[1] if label else None,
        "currency": (label[2] if label and label[2] else prefix_currency[1] if prefix_currency else None),
    }


class ShareClassSource(MarketReportClient):
    """沿用市场报告来源、归档和限额；额外只允许精确基金档案地址。"""

    def report(self, code: str, entry: dict) -> tuple[str, dict]:
        # 公告目录已精确绑定当前份额；转载证券列表可能只列主代码，最终仍须身份表含当前份额。
        params = {"client_source": "web_fund", "show_all": 1, "art_code": entry["id"]}
        raw = self._get(BODY_URL, params)
        value = json.loads(raw)
        data = value.get("data") or {}
        if value.get("success") != 1 or data.get("art_code") != entry["id"]:
            raise ValueError("SHARE_REPORT_IDENTITY_MISMATCH")
        if compact(data.get("notice_title", "")) != compact(entry["title"]):
            raise ValueError("SHARE_REPORT_TITLE_MISMATCH")
        body = data.get("notice_content", "")
        if not isinstance(body, str) or len(body) < 1500:
            raise ValueError("SHARE_REPORT_BODY_UNAVAILABLE")
        ref = self._archive(raw, ".json", BODY_URL, params)
        return body, ref | {"publishedOn": entry["published"], "title": entry["title"]}

    def profile(self, code: str) -> tuple[dict, dict]:
        if not re.fullmatch(r"[0-9]{6}", code):
            raise ValueError("SHARE_PROFILE_CODE_INVALID")
        url = f"https://fundf10.eastmoney.com/jbgk_{code}.html"
        if self.count >= 500 or self.failures >= 5:
            raise ValueError("SHARE_SOURCE_REQUEST_LIMIT")
        time.sleep(max(0, 0.35 - (time.monotonic() - self.last)))
        self.count += 1
        self.last = time.monotonic()
        raw = bounded_get(self.client, url, 1_000_000)
        return parse_profile(raw, code), self._archive(raw, ".html", url, {})

    def verify(self, ref: dict) -> bytes:
        return read_source(self.root, ref)
