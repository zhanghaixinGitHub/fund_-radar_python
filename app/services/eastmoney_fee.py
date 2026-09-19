"""天天基金 f10 费率页（jjfl）抓取与解析：申购取第一档优惠费率，赎回按持有自然天映射为费率分档。

页面为服务端渲染的 HTML（https://fundf10.eastmoney.com/jjfl_XXXXXX.html），
包含申购费率表（原费率与天天基金优惠费率两列）与赎回费率表（中文期限描述 + 费率）。
解析结果供 Java 核心服务落库 sim_fee_rule；无法可靠解析的档位（按年/月计、固定金额等）
抛出 MANUAL_REQUIRED，由后台人工维护兜底。
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import httpx
from bs4 import BeautifulSoup

from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_BYTES = 1_000_000  # 页面响应上限，防止异常响应撑爆内存
F10_BASE = "https://fundf10.eastmoney.com"
TERM_UNIT = re.compile(r"([0-9]+)\s*(天|个月|月|年)")
PERCENT = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
TITLE = re.compile(r"(.+?)\((\d{6})\)")

# 申购费率表与赎回费率表均以 box 内的表格呈现；赎回档位标题固定带锚点 shfl。
_PURCHASE_LABEL = "申购费率"
_REDEEM_LABEL = "赎回费率"


@dataclass(frozen=True)
class FeeBand:
    """单档赎回费率区间；min_days 含当天，max_days 为 None 表示最高档无上限。"""

    min_days: int
    max_days: int | None
    rate: Decimal


@dataclass(frozen=True)
class FeeProfile:
    """单基金费率档案；金额分档申购只取第一档（模拟场景申购金额均落在最低档）。"""

    fund_code: str
    fund_name: str
    purchase_rate: Decimal
    purchase_original_rate: Decimal
    discount_info: str | None
    redeem_bands: tuple[FeeBand, ...]
    data_source: str = "EASTMONEY_F10"


def parse_redeem_term(description: str) -> tuple[int, int | None]:
    """把赎回期限中文描述映射为 [min_days, max_days] 区间，max_days 为 None 表示无上限。

    只支持以「天」为单位的描述；出现年/月单位或无法识别的句式时要求人工维护。
    """
    matches = TERM_UNIT.findall(description)
    if not matches:
        raise ValueError(f"MANUAL_REQUIRED: 无法识别赎回期限描述「{description}」")
    if any(unit != "天" for _, unit in matches):
        raise ValueError(f"MANUAL_REQUIRED: 赎回期限按非自然天计期「{description}」")
    days = [int(number) for number, _ in matches]
    has_lower = "大于等于" in description or "大于" in description
    has_upper = "小于" in description or "不足" in description
    if has_lower and has_upper:
        return days[0], days[1] - 1
    if has_upper:
        return 0, days[0] - 1
    if "大于等于" in description:
        return days[0], None
    raise ValueError(f"MANUAL_REQUIRED: 无法识别赎回期限描述「{description}」")


def parse_jjfl_html(html: str, fund_code: str) -> FeeProfile:
    """解析 f10 费率页 HTML；页面标题中的基金代码必须与请求一致。"""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    matched = TITLE.match(title)
    if matched is None:
        raise ValueError(f"MANUAL_REQUIRED: 费率页标题无法解析「{title[:50]}」")
    fund_name, page_code = matched.group(1), matched.group(2)
    if page_code != fund_code:
        raise ValueError(f"FUND_CODE_MISMATCH: 页面代码 {page_code} 与请求 {fund_code} 不一致")

    purchase = _rate_table(soup, _PURCHASE_LABEL)
    redeem = _rate_table(soup, _REDEEM_LABEL)
    if purchase is None or redeem is None:
        raise ValueError(f"MANUAL_REQUIRED: 费率表缺失，申购={purchase is not None} 赎回={redeem is not None}")

    original, discounted = _purchase_rates(purchase)
    bands = tuple(_redeem_bands(redeem))
    _validate_bands(bands)
    discount = "天天基金优惠费率" if discounted < original else None
    return FeeProfile(
        fund_code=fund_code,
        fund_name=fund_name,
        purchase_rate=discounted,
        purchase_original_rate=original,
        discount_info=discount,
        redeem_bands=bands,
    )


def fetch_fee_profile(fund_code: str) -> FeeProfile:
    """抓取指定基金的 f10 费率页并解析；网络或页面异常均转为明确错误。"""
    url = f"{F10_BASE}/jjfl_{fund_code}.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": f"{F10_BASE}/jjfl_{fund_code}.html",
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=True) as client:
            response = client.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"F10_UNAVAILABLE: 天天基金费率页返回 {response.status_code}")
        if len(response.content) > MAX_BYTES:
            raise ValueError("F10_RESPONSE_TOO_LARGE: 页面响应超过大小上限")
        return parse_jjfl_html(response.text, fund_code)
    except ValueError:
        raise
    except httpx.HTTPError as error:
        logger.warning(
            "eastmoney_fee.fetch_fee_profile >>> f10 request failed, fundCode=%s, error=%s", fund_code, error
        )
        raise ValueError("F10_UNAVAILABLE: 天天基金费率页抓取失败") from error


def _rate_table(soup: BeautifulSoup, label_text: str):
    """定位标题为 label_text 的费率表；label 可能含锚点子元素，按文本前缀匹配。"""
    for label in soup.find_all("label", class_="left"):
        text = label.get_text(" ", strip=True)
        if text.startswith(label_text):
            box = label.find_parent("div", class_="box")
            table = box.find("table") if box else None
            if table is not None:
                return table
    return None


def _percent(cell_text: str) -> Decimal:
    """从单元格文本提取百分比费率；固定金额（每笔 N 元）等无法映射时要求人工。"""
    matched = PERCENT.search(cell_text)
    if matched is None:
        raise ValueError(f"MANUAL_REQUIRED: 费率单元格不是百分比「{cell_text[:40]}」")
    try:
        return Decimal(matched.group(1)) / Decimal(100)
    except InvalidOperation as error:
        raise ValueError(f"MANUAL_REQUIRED: 费率数值无法解析「{cell_text[:40]}」") from error


def _purchase_rates(table) -> tuple[Decimal, Decimal]:
    """解析申购表第一档：优惠费率列存在且低于原费率时用优惠费率，否则用原费率。"""
    rows = _data_rows(table)
    if not rows:
        raise ValueError("MANUAL_REQUIRED: 申购费率表为空")
    first = [cell.get_text(" ", strip=True) for cell in rows[0].find_all("td")]
    if len(first) < 2:
        raise ValueError("MANUAL_REQUIRED: 申购费率表结构异常")
    # 单元格形如「0.80% | 0.08%」时拆出原费率与优惠费率；只有一列时两者相同。
    parts = [part.strip() for part in first[1].split("|")]
    original = _percent(parts[0])
    discounted = _percent(parts[-1]) if len(parts) > 1 else original
    return original, discounted


def _redeem_bands(table) -> list[FeeBand]:
    rows = _data_rows(table)
    if not rows:
        raise ValueError("MANUAL_REQUIRED: 赎回费率表为空")
    bands = []
    for row in rows:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) < 2:
            continue
        min_days, max_days = parse_redeem_term(cells[0])
        bands.append(FeeBand(min_days, max_days, _percent(cells[1])))
    return bands


def _data_rows(table) -> list:
    """只取表体数据行，跳过表头 th 行。"""
    body = table.find("tbody")
    rows = body.find_all("tr") if body else table.find_all("tr")
    return [row for row in rows if row.find("td") is not None]


def _validate_bands(bands: tuple[FeeBand, ...]) -> None:
    """校验分档：下界必须从 0 开始逐档递增，且最高档无上限。"""
    for index, band in enumerate(bands):
        expected_min = 0 if index == 0 else bands[index - 1].max_days + 1
        if band.min_days != expected_min:
            raise ValueError(f"MANUAL_REQUIRED: 赎回分档不连续，第 {index + 1} 档起于 {band.min_days} 天")
    if bands and bands[-1].max_days is not None:
        raise ValueError("MANUAL_REQUIRED: 最高赎回档必须有上界为空")
