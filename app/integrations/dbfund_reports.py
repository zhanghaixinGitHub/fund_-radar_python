"""德邦官网公开报告目录和 PDF 持仓解析，限定单基金与正常公开读取。"""

import hashlib
import io
import re
import time
from datetime import date, datetime, timedelta
from datetime import time as day_time
from decimal import Decimal
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_exposure_common import ROOT, blob, initialize, now, read, save
from app.services.fund_materials_store import versioned_save

ORIGIN = "https://www.dbfund.com.cn"
PRODUCT = ORIGIN + "/products/hunhe/002112/index.html"
CATALOG = ORIGIN + "/common-web/cms/content/getContents"
PARSER_VERSION = "DBFUND_REPORT_HOLDINGS_V2"
EXTENDED_PARSER_VERSION = "DBFUND_EXTENDED_HOLDINGS_V1"
MAX_PDF_BYTES = 20_000_000


def official_url(value: str) -> str:
    """只读取已知官网 PDF，拒绝跨域跳转和非报告附件。"""
    url = urljoin(ORIGIN, value)
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "www.dbfund.com.cn":
        raise ValueError("REPORT_HOST_NOT_ALLOWED")
    if not parsed.path.startswith("/upload/pdf/") or not parsed.path.lower().endswith(".pdf"):
        raise ValueError("REPORT_URL_NOT_PDF")
    return url


def bounded_get(client: httpx.Client, url: str, limit: int, **kwargs) -> bytes:
    """限制响应大小和超时；不跟随未知跳转，失败由上层逐项记录。"""
    with client.stream("GET", url, **kwargs) as response:
        response.raise_for_status()
        raw = bytearray()
        for part in response.iter_bytes():
            raw.extend(part)
            if len(raw) > limit:
                raise ValueError("REPORT_RESPONSE_TOO_LARGE")
    return bytes(raw)


def catalog(client: httpx.Client, *, extended_history=False) -> list[dict]:
    """读取同一基金的公开目录；扩展历史单独留证，不改冻结研究的目录和规则。"""
    plan = initialize()
    raw = bounded_get(client, PRODUCT, 1_000_000)
    blob(raw, "html")
    soup = BeautifulSoup(raw, "html.parser")
    board = soup.select_one(".fund-board-container.information_dqgg")
    if board is None:
        raise ValueError("REPORT_CATALOG_NOT_FOUND")
    category = board.get("information_categoryid")
    more = board.select_one('a[href^="javascript:getMore"]')
    match = re.search(r",(\d+),(\d+),'[^']+','[^']+',(\d+)\)", more["href"] if more else "")
    if not match or not category:
        raise ValueError("REPORT_PAGINATION_UNKNOWN")
    pages, page_size, expected = map(int, match.groups())
    if not 1 <= pages <= plan["maximum_report_pages"] or expected > plan["maximum_reports"]:
        raise ValueError("REPORT_CATALOG_LIMIT")
    result = {}
    returned = 0
    for page in range(1, pages + 1):
        raw = bounded_get(
            client,
            CATALOG,
            2_000_000,
            params={
                "categoryId": category,
                "pageNumber": page,
                "pageSize": page_size,
            },
        )
        blob(raw, "json")
        import json

        data = json.loads(raw)
        items = data.get("contents")
        if not isinstance(items, list) or not 0 < len(items) <= page_size:
            raise ValueError("REPORT_CATALOG_SCHEMA_OR_EMPTY")
        returned += len(items)
        for item in items:
            title = re.sub(r"\s+", "", item["title"])
            year = re.search(r"(20\d{2})年", title)
            first_year = 2015 if extended_history else plan["report_year_start"]
            if "德邦鑫星价值" not in title or not year or int(year[1]) < first_year:
                continue
            if not re.search(r"(季度报告|中期报告|半年度报告|年度报告)", title) or "摘要" in title:
                continue
            # 2021 起点使用当时已经公开的 2020 三季报作锚；更早报告不属于本轮目标。
            if (
                not extended_history
                and int(year[1]) == 2020
                and not re.search(r"(第?[34三四]季度报告|年度报告)", title)
            ):
                continue
            url = official_url(item["url"])
            result[url] = {
                "title": title,
                "url": url,
                "content_id": item["contentId"],
                # CMS 日期可能是网站迁移日期，仅保留原值；历史可得性以报告送出日期保守重建。
                "catalog_activation_ms": item.get("activationDate"),
                "catalog_publish_ms": item.get("publishDate"),
                "catalog_received_at": now().isoformat(),
            }
        time.sleep(0.25)
    if returned != expected:
        raise ValueError("REPORT_CATALOG_COUNT_CHANGED")
    value = sorted(result.values(), key=lambda r: r["title"])
    directory = ROOT / "readiness-extended" if extended_history else ROOT
    versioned_save(directory / "catalog.json", {"at": now().isoformat(), "total_returned": returned, "reports": value})
    return value


def parse_text(
    pages: list[str],
    title: str,
    *,
    fund_code="002112",
    fund_name="德邦鑫星价值",
    master_code="001412",
    derive_missing_weights=False,
) -> dict:
    """提取基金整体股票表；严格核验连续序号、比例分母和完整披露合计。"""
    first = re.sub(r"\s+", "", "\n".join(pages[:3]))
    # 其他基金的公开报告可能用中文日期；只转换日期字段，不改动正文金额或持仓。
    if fund_code != "002112":
        chinese = re.search(
            r"送出日期[：:]?([〇零0一二三四五六七八九]{4})年([一二三四五六七八九十]+)月([一二三四五六七八九十]+)日",
            first,
        )
        if chinese:
            digits = {c: str(i) for i, c in enumerate("零一二三四五六七八九")}
            digits["〇"] = "0"
            digits["0"] = "0"

            def number(value):
                if "十" in value:
                    left, right = value.split("十")
                    return (
                        int(digits[left]) * 10 + int(digits[right] if right else 0)
                        if left
                        else 10 + int(digits[right] if right else 0)
                    )
                return int(digits[value])

            year_text, month_text, day_text = chinese.groups()
            first = first.replace(
                chinese[0],
                f"送出日期:{''.join(digits[c] for c in year_text)}年{number(month_text)}月{number(day_text)}日",
            )
    sent = re.search(r"(?:报告送出日期|送出日期)[：:]?(\d{4})年(\d{1,2})月(\d{1,2})日", first)
    if not sent:
        raise ValueError("REPORT_PUBLICATION_DATE_MISSING")
    published = date(*map(int, sent.groups()))
    title_for_period = title
    if fund_code != "002112":
        # 富国等管理人将年份写成“二0二三”，只归一化年份，不改报告期或原始标题。
        translation = str.maketrans("〇零一二三四五六七八九", "00123456789")
        title_for_period = re.sub(r"[〇零0一二三四五六七八九]{4}(?=年)", lambda m: m[0].translate(translation), title)
    title_year = re.search(r"(20\d{2})年", title_for_period)
    if not title_year:
        raise ValueError("REPORT_YEAR_MISSING")
    year = int(title_year[1])
    quarter = re.search(r"第?([1234一二三四])季度", title)
    q = int(quarter[1]) if quarter and quarter[1].isdigit() else ("一二三四".index(quarter[1]) + 1 if quarter else None)
    kind = "QUARTER" if q else "HALF" if "中期" in title or "半年度" in title else "ANNUAL"
    month, day = (
        {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[q] if q else ((6, 30) if kind == "HALF" else (12, 31))
    )
    end = date(year, month, day)
    if not end < published <= end + timedelta(days=125):
        raise ValueError("REPORT_PUBLICATION_DATE_INVALID")
    text = "\n".join(pages)
    if fund_name not in re.sub(r"\s+", "", text) or fund_code not in text:
        raise ValueError("REPORT_FUND_IDENTITY_MISMATCH")
    # 去掉行内空白但保留换行，以便识别跨页表格。中间的标题行不会变成股票记录。
    lines = [re.sub(r"[ \t\u3000]+", " ", line).strip() for line in text.splitlines()]
    clean = "\n".join(lines)
    starts = list(re.finditer(r"(?:5|7|8)\.3\s+(?:期末|报告期末)[^\n]*股票\s*投\s*资\s*明\s*细", clean))
    sections = []
    for start in starts:
        following = clean[start.end() :]
        stop = re.search(r"\n(?:5|7|8)\.4\s", following)
        section = following[: stop.start()] if stop else ""
        if section and not re.search(r"\.{4,}", clean[start.start() : start.end()]):
            sections.append((start.start(), section))
    if not sections:
        raise ValueError("REPORT_HOLDINGS_SECTION_MISSING")
    section_start, section = sections[-1]
    # 早期报告有金额但百分比印为横线。只有独立扩展研究可从同份报告的基金总净资产
    # 推导比例；保留原始缺省符号及计算依据，绝不把横线填成零或借用 A 类单独净资产。
    denominator = None
    if derive_missing_weights and re.search(r"(?m)^\s*\d+\s+\d{6}[^\n]+\.\d{2}\s+[－—-]\s*$", section):
        equity = re.search(r"(?m)^所有者权益合计\s+([\d,]+\.\d{2})", clean)
        if equity:
            denominator = Decimal(equity[1].replace(",", ""))
            share_nav = re.search(r"期末基金资产净值\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})", clean)
            if (
                not share_nav
                or denominator <= 0
                or sum(Decimal(n.replace(",", "")) for n in share_nav.groups()) != denominator
            ):
                raise ValueError("REPORT_FUND_NAV_DENOMINATOR_MISMATCH")
    weight_token = r"(?:\d+(?:\.\d+)?|[－—-])" if derive_missing_weights else r"\d+(?:\.\d+)?"
    pattern = (
        r"(?m)^\s*(\d{1,4})\s+(\d{5,6})\s+(.+?)\s+([\d,]+(?:\.\d+)?)\s+([\d,]+\.\d{2})\s+("
        + weight_token
        + r")\s*%?\s*$"
    )
    holdings = []
    ranks = []
    for m in re.finditer(pattern, section):
        rank, code, name, quantity, value, weight = m.groups()
        missing_weight = weight in {"－", "—", "-"}
        if missing_weight:
            if denominator is None:
                raise ValueError("REPORT_WEIGHT_DENOMINATOR_MISSING")
            weight = str(Decimal(value.replace(",", "")) / denominator * 100)
            # 若横线对应显著仓位，说明不是本规则能解释的格式；停止而非猜测。
            if Decimal(weight) >= Decimal("0.005"):
                raise ValueError("REPORT_WEIGHT_DASH_NOT_ROUNDING_SCALE")
        if fund_code == "002112" and (len(code) != 6 or not code.startswith(("0", "3", "6"))):
            raise ValueError("REPORT_SECURITY_MARKET_UNVERIFIED")
        market = (
            ".HK"
            if len(code) == 5
            else ".SH"
            if code.startswith("6")
            else ".SZ"
            if code.startswith(("0", "3"))
            else ".BJ"
            if code.startswith(("4", "8", "92"))
            else None
        )
        if market is None:
            raise ValueError("REPORT_SECURITY_MARKET_UNVERIFIED")
        row = {
            "reported_rank": int(rank),
            "stock_code": code + market,
            "stock_name": name.replace(" ", ""),
            "quantity_shares": str(Decimal(quantity.replace(",", ""))),
            "market_value_cny": str(Decimal(value.replace(",", ""))),
            "nav_weight_pct": str(Decimal(weight)),
        }
        if not Decimal(0) <= Decimal(weight) <= Decimal(100):
            raise ValueError("REPORT_WEIGHT_INVALID")
        holdings.append(row)
        if missing_weight:
            row.update(
                reported_nav_weight_pct=None,
                weight_basis="PUBLISHED_VALUE_DIVIDED_BY_PUBLISHED_FUND_NAV",
                denominator_nav_cny=str(denominator),
            )
        ranks.append(int(rank))
    prefix = clean[:section_start]
    industry_start = list(re.finditer(r"(?:5|7|8)\.2\s+(?:报告期末|期末)?按行业分类", prefix))
    industry = prefix[industry_start[-1].start() :] if industry_start else ""
    totals = list(re.finditer(r"合计\s+([\d,]+\.\d{2})\s+([\d.]+)", industry))
    no_stocks = bool(re.search(r"未(?:持有|投资)(?:任何)?股票", section))
    if not totals and not no_stocks:
        raise ValueError("REPORT_STOCK_NAV_RATIO_MISSING")
    stock_pct = Decimal(totals[-1][2]) if totals else Decimal(0)
    stock_value = Decimal(totals[-1][1].replace(",", "")) if totals else Decimal(0)
    # 2020 年报按市值采用并列序号，例如两个第 28 名。只允许相邻、同金额并列，
    # 不改写原序号；金额总和仍须与报告严格相符，以免把漏行误判为并列。
    ranks_valid = not ranks or ranks[0] == 1
    for i in range(1, len(ranks)):
        tied = ranks[i] == ranks[i - 1] and holdings[i]["market_value_cny"] == holdings[i - 1]["market_value_cny"]
        ranks_valid = ranks_valid and (ranks[i] == ranks[i - 1] + 1 or tied)
    if not ranks_valid or len({r["stock_code"] for r in holdings}) != len(holdings):
        raise ValueError("REPORT_HOLDINGS_RANK_OR_DUPLICATE")
    if not holdings and not no_stocks:
        raise ValueError("REPORT_HOLDINGS_PARSE_EMPTY")
    if kind == "QUARTER" and len(holdings) > 10:
        raise ValueError("REPORT_QUARTER_HOLDING_LIMIT")
    covered = sum((Decimal(r["nav_weight_pct"]) for r in holdings), Decimal(0))
    amount = sum((Decimal(r["market_value_cny"]) for r in holdings), Decimal(0))
    full = kind != "QUARTER"
    # 完整持仓用金额做严格核对，权重只接受每行四舍五入能解释的误差。
    if full and (
        abs(amount - stock_value) > Decimal("0.02") or abs(covered - stock_pct) > Decimal("0.005") * (len(holdings) + 1)
    ):
        raise ValueError("REPORT_FULL_HOLDING_TOTAL_MISMATCH")
    if covered > stock_pct + Decimal("0.005") * (len(holdings) + 1):
        raise ValueError("REPORT_PARTIAL_WEIGHT_EXCEEDS_STOCK")
    available = datetime.combine(published + timedelta(days=1), day_time(8), ZONE)
    asset_starts = list(re.finditer(r"(?:5|7|8)\.1\s+(?:报告期末|期末)基金资产组合情况", prefix))
    asset_text = (
        prefix[asset_starts[-1].start() : industry_start[-1].start()] if asset_starts and industry_start else ""
    )
    assets = {}
    for asset_name in (
        "权益投资",
        "固定收益投资",
        "买入返售金融资产",
        "银行存款和结算备付金合计",
        "其他各项资产",
        "其他资产",
    ):
        m = re.search(re.escape(asset_name) + r"\s+([\d, ]+\.\d{2}|-)\s+([\d.]+|-)", asset_text)
        if m:
            assets[asset_name] = {
                "value_cny": "0" if m[1] == "-" else m[1].replace(",", "").replace(" ", ""),
                "total_asset_pct": "0" if m[2] == "-" else m[2],
                "denominator": "TOTAL_ASSET",
            }
    industries = []
    for m in re.finditer(
        r"(?m)^([A-S])\s+([\u4e00-\u9fff、，\s]+?)\s+([\d, ]+\.\d{2})\s+([\d.]+|[－—-])\s*$", industry
    ):
        industries.append(
            {
                "industry_code": m[1],
                "name": re.sub(r"\s+", "", m[2]),
                "value_cny": m[3].replace(",", "").replace(" ", ""),
                "nav_weight_pct": None if m[4] in {"－", "—", "-"} else m[4],
            }
        )
    if abs(sum((Decimal(r["value_cny"]) for r in industries), Decimal(0)) - stock_value) > Decimal("0.02"):
        raise ValueError("REPORT_INDUSTRY_TOTAL_MISMATCH")
    return {
        "fund_code": fund_code,
        "fund_master_code": master_code,
        "title": title,
        "report_end": str(end),
        "published_date": str(published),
        "available_at": available.isoformat(),
        "availability_basis": "REPORT_DECLARED_DATE_NEXT_DAY_0800_RECONSTRUCTION",
        "report_type": kind,
        "full_stock_disclosure": full,
        "stock_nav_pct": str(stock_pct),
        "stock_value_cny": str(stock_value),
        "disclosed_nav_pct": str(covered),
        "holding_count": len(holdings),
        "holdings": holdings,
        "assets": assets,
        "reported_industries": industries,
        "parser_version": EXTENDED_PARSER_VERSION if derive_missing_weights else PARSER_VERSION,
        "quality": "VERIFIED_TABLE_TOTALS",
    }


def acquire(*, incremental=False, extended_history=False, progress=lambda *args: None) -> dict:
    """缓存原始 PDF；扩展研究只取 C 类成立后的历史，独立保存，不改变正式资料范围。"""
    initialize()
    directory = ROOT / "readiness-extended" if extended_history else ROOT
    parser_version = EXTENDED_PARSER_VERSION if extended_history else PARSER_VERSION
    results, errors = [], []
    previous = read(directory / "report-result.json") if (directory / "report-result.json").exists() else {}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": PRODUCT}
    with httpx.Client(timeout=httpx.Timeout(25, connect=5), headers=headers, follow_redirects=False) as client:
        entries = catalog(client, extended_history=True) if extended_history else catalog(client)
        if extended_history:
            from app.integrations.public_fund_reports import report_period

            entries = [e for e in entries if (p := report_period(e["title"])) and "2015-12-31" <= p[0] <= "2024-09-30"]
        for ordinal, entry in enumerate(entries):
            progress(ordinal, len(entries), "002112", "核对本基金历史报告：" + entry["title"])
            key = digest(entry["url"])
            receipt = directory / "report-receipts" / f"{key}.json"
            receipt_source = receipt
            if extended_history and not receipt.exists():
                receipt_source = ROOT / "report-receipts" / f"{key}.json"
            try:
                if receipt_source.exists():
                    saved = read(receipt_source)
                    raw = (ROOT / saved["file"]).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != saved["sha256"]:
                        raise ValueError("REPORT_CACHED_PDF_CHANGED")
                    last_check = datetime.fromisoformat(saved.get("checked_at", saved["received_at"]))
                    catalog_key = {k: v for k, v in entry.items() if k != "catalog_received_at"}
                    changed = saved.get("catalog_hash") and saved["catalog_hash"] != digest(catalog_key)
                    if changed or (not incremental and now() - last_check >= timedelta(days=7)):
                        # 官网可能在原链接替换修订版；每周重新核验，旧版本和原首次取得时间均保留。
                        fresh = bounded_get(client, entry["url"], MAX_PDF_BYTES)
                        if not fresh.startswith(b"%PDF-"):
                            raise ValueError("REPORT_RESPONSE_NOT_PDF")
                        sha, name = blob(fresh, "pdf")
                        archive = directory / "report-receipt-versions" / (digest(saved) + ".json")
                        if not archive.exists():
                            save(archive, saved)
                        received = saved["received_at"] if sha == saved["sha256"] else now().isoformat()
                        revised = saved.get("revised_after_receipt", False) or sha != saved["sha256"]
                        saved = {
                            "url": entry["url"],
                            "sha256": sha,
                            "file": name,
                            "received_at": received,
                            "checked_at": now().isoformat(),
                        }
                        if extended_history:
                            saved["revised_after_receipt"] = revised
                        save(receipt, saved, replace=True)
                        raw = fresh
                    saved["catalog_hash"] = digest(catalog_key)
                    save(receipt, saved, replace=True)
                else:
                    raw = bounded_get(client, entry["url"], MAX_PDF_BYTES)
                    if not raw.startswith(b"%PDF-"):
                        raise ValueError("REPORT_RESPONSE_NOT_PDF")
                    sha, name = blob(raw, "pdf")
                    saved = {"url": entry["url"], "sha256": sha, "file": name, "received_at": now().isoformat()}
                    save(receipt, saved)
                    time.sleep(0.3)
                pdf = PdfReader(io.BytesIO(raw))
                path = directory / "reports" / f"{saved['sha256']}-{parser_version}.json"
                if path.exists():
                    results.append(read(path))
                    continue
                if len(pdf.pages) > 180:
                    raise ValueError("REPORT_PAGE_LIMIT")
                pages = [page.extract_text() for page in pdf.pages]
                parsed = (
                    parse_text(pages, entry["title"], derive_missing_weights=True)
                    if extended_history
                    else parse_text(pages, entry["title"])
                )
                if incremental and not extended_history:
                    # 本次才得到的新报告或更正版不能回填成过去已知；历史研究成果仍保留原记录。
                    parsed["available_at"] = max(parsed["available_at"], saved["received_at"])
                if extended_history and saved.get("revised_after_receipt"):
                    parsed["available_at"] = max(parsed["available_at"], saved["received_at"])
                parsed.update({"source": entry, "raw": saved, "parsed_at": now().isoformat(), "pages": len(pages)})
                save(path, parsed)
                results.append(parsed)
                print(f"REPORT_OK {parsed['report_end']} {parsed['report_type']} {parsed['holding_count']}", flush=True)
            except Exception as exc:
                # 公开报告错误码可记录；网络异常只记录类型，避免输出完整请求上下文。
                reason = (
                    str(exc) if isinstance(exc, ValueError) and str(exc).startswith("REPORT_") else type(exc).__name__
                )
                errors.append({"title": entry["title"], "url": entry["url"], "reason": reason})
                print(f"REPORT_ERROR {entry['title']} {reason}", flush=True)
    result = {
        "at": now().isoformat(),
        "reports": len(results),
        "errors": errors,
        "report_files": [f"{r['raw']['sha256']}-{parser_version}" for r in results],
        "new_purchase_cny": 0,
    }
    # 新来源缺失或失败不能移除已核验的历史报告；不同原文哈希继续保留。
    prior_files = previous.get("report_files", [])
    if extended_history:
        # 同一原文的解析修复只保留当前解析为入口；旧文件留存但不作为重复报告。
        current_hashes = {r["raw"]["sha256"] for r in results}
        prior_files = [
            f for f in prior_files if read(directory / "reports" / (f + ".json"))["raw"]["sha256"] not in current_hashes
        ]
    result["report_files"] = list(dict.fromkeys(prior_files + result["report_files"]))
    result["reports"] = len(result["report_files"])
    listed_urls = {entry["url"] for entry in entries}
    result["not_listed_on_recheck"] = [
        name
        for name in result["report_files"]
        if read(directory / "reports" / (name + ".json"))["raw"]["url"] not in listed_urls
    ]
    versioned_save(directory / "report-result.json", result)
    return result
