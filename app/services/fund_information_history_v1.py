"""公告历史补齐 V1：独立研究目录、有限公开读取、三层覆盖与保守语义。

不导入训练入口，不生成标签/方向，不改旧事实或旧实验。目录完整仅表示在
登记来源与标题范围内分页齐全；正文与语义仍须分别核验。
"""

import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

ZONE = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[2] / ".local-runs/fund-exposure-002112"
OLD = ROOT / "information-research/20260928-v1"
OUT = ROOT / "information-research/20260928-history-v1"
QUERY = "https://www.cninfo.com.cn/new/hisAnnouncement/query"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    # 旧来源使用 payload 封装；完整性通过外层文件哈希单独固定。
    return value["payload"] if isinstance(value, dict) and set(value) == {"hash", "payload"} else value


def save(path, value):
    """排他保存，恢复时只复用相同内容；绝不覆盖另一版证据。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError("IMMUTABLE_ARTIFACT_CONFLICT:" + path.name)
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def normalize(text):
    """锚点偏移以去空白文本计，原 PDF 和完整抽取页仍保留供独立重放。"""
    return re.sub(r"\s+", "", text)


def classify(title):
    """只认明确标题，辅助意见不计作公司新事件；快报与正式结果分开。"""
    if any(w in title for w in ("核查意见", "法律意见", "审核意见", "独立董事", "监事会", "提示性")):
        return {"category": None, "stage": "SUPPORTING_DOCUMENT"}
    if "回购" in title and not any(
        w in title for w in ("股权激励", "限制性股票", "回购注销", "债券", "质押式", "股东持股")
    ):
        category = "BUYBACK"
    elif any(w in title for w in ("重大合同", "重大经营合同", "重大销售合同")):
        category = "MAJOR_CONTRACT"
    elif any(w in title for w in ("业绩预告", "业绩快报", "年度报告", "季度报告", "半年度报告")):
        category = "PERFORMANCE"
    else:
        return {"category": None, "stage": "OUTSIDE_REGISTERED_TITLE_SCOPE"}
    if any(w in title for w in ("更正", "修正", "修订")):
        stage = "CORRECTION"
    elif any(w in title for w in ("终止", "取消")):
        stage = "TERMINATED"
    elif any(w in title for w in ("完成", "完毕")):
        stage = "COMPLETED"
    elif "业绩预告" in title:
        stage = "FORECAST"
    elif "业绩快报" in title:
        stage = "PRELIMINARY_RESULT"
    elif category == "PERFORMANCE":
        stage = "REPORTED_RESULT"
    elif "进展" in title or "首次回购" in title:
        stage = "PROGRESS"
    elif any(w in title for w in ("方案", "计划", "提议", "报告书", "决议")):
        stage = "PLAN_OR_APPROVAL"
    elif category == "MAJOR_CONTRACT" and "签订" in title:
        stage = "SIGNED_DISCLOSURE"
    else:
        stage = "DISCLOSED"
    return {"category": category, "stage": stage}


def catalog_rows(pages, stock, start, end):
    """逐页核总数、身份和时间。官网 totalpages 不可信，按总数推导页数。"""
    if not pages:
        raise ValueError("EMPTY_PAGE_SET")
    total = pages[0][1].get("totalAnnouncement")
    if not isinstance(total, int) or total < 0:
        raise ValueError("INVALID_TOTAL")
    expected = list(range(1, max(1, (total + 29) // 30) + 1))
    if [p for p, _ in pages] != expected:
        raise ValueError("MISSING_OR_DUPLICATE_PAGE")
    rows, ids = [], set()
    for page, value in pages:
        if value.get("totalAnnouncement") != total:
            raise ValueError("CATALOG_TOTAL_CHANGED")
        items = value.get("announcements") or []
        if len(items) != min(30, max(0, total - (page - 1) * 30)):
            raise ValueError("PAGE_SIZE_INCOMPLETE")
        for item in items:
            key = item["announcementId"]
            if key in ids:
                raise ValueError("DUPLICATE_ANNOUNCEMENT")
            ids.add(key)
            if stock not in item["secCode"].split(","):
                raise ValueError("WRONG_COMPANY")
            day = datetime.fromtimestamp(item["announcementTime"] / 1000, ZONE).date().isoformat()
            if not start <= day <= end:
                raise ValueError("DATE_OUTSIDE_WINDOW")
            title = BeautifulSoup(item["announcementTitle"], "html.parser").get_text()
            rows.append({**item, "title_plain": title, "published_date": day, **classify(title)})
    return rows


def covered(intervals, start, end):
    cursor = date.fromisoformat(start)
    for lo, hi in sorted(intervals):
        if date.fromisoformat(hi) < cursor:
            continue
        if date.fromisoformat(lo) > cursor:
            return False
        cursor = date.fromisoformat(hi) + timedelta(days=1)
        if cursor > date.fromisoformat(end):
            return True
    return False


class BoundedPublicReader:
    """每次请求先记不可退款槽位。网络失败/中断保留，不自动重试或追加预算。

    仅允许登记的公开查询和 GET 原件；不跟随跳转、不带账号、限制响应字节。
    同一进程串行读取；调用方还必须取得目录锁，避免两个进程消费同一预算。
    """

    def __init__(self, directory, limits):
        self.directory, self.limits = Path(directory), limits
        self.client = httpx.Client(timeout=httpx.Timeout(25, connect=5), follow_redirects=False)
        self.last = 0.0

    def close(self):
        self.client.close()

    def fetch(self, url, *, params=None, group="company", maximum_bytes=4_000_000):
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc not in {"www.cninfo.com.cn", "static.cninfo.com.cn", "www.nhsa.gov.cn"}
            or parsed.username
            or parsed.password
        ):
            raise ValueError("PUBLIC_URL_NOT_ALLOWED")
        if params is not None and url.split("?", 1)[0] not in {
            QUERY,
            "https://www.nhsa.gov.cn/module/web/jpage/dataproxy.jsp",
        }:
            raise ValueError("POST_NOT_PUBLIC_QUERY")
        key = digest({"url": url, "params": params})
        target = self.directory / "requests" / (key + ".json")
        receipt = self.directory / "receipts" / (key + ".json")
        if receipt.exists():
            saved = read(receipt)
            if not saved["ok"]:
                raise ValueError("PREVIOUS_PUBLIC_REQUEST_FAILED:" + saved["reason"])
            if sha(saved["path"]) != saved["sha256"]:
                raise ValueError("PUBLIC_RAW_CHANGED")
            return Path(saved["path"]).read_bytes(), saved
        if target.exists():
            raise ValueError("INTERRUPTED_PUBLIC_REQUEST_NO_AUTORETRY")
        consumed = sum(read(p)["group"] == group for p in (self.directory / "requests").glob("*.json"))
        if consumed >= self.limits[group]:
            raise ValueError("PUBLIC_REQUEST_LIMIT:" + group)
        save(target, {"url": url, "params": params, "group": group, "at": datetime.now(ZONE).isoformat()})
        try:
            time.sleep(max(0, 0.65 - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            with self.client.stream(
                "POST" if params is not None else "GET",
                url,
                data=params,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://" + parsed.netloc + "/"},
            ) as response:
                response.raise_for_status()
                raw = bytearray()
                for part in response.iter_bytes():
                    raw.extend(part)
                    if len(raw) > maximum_bytes:
                        raise ValueError("PUBLIC_RESPONSE_SIZE_LIMIT")
            blob = self.directory / "raw" / key
            blob.parent.mkdir(parents=True, exist_ok=True)
            with blob.open("xb") as stream:
                stream.write(raw)
            result = {
                "ok": True,
                "url": url,
                "params": params,
                "path": str(blob),
                "sha256": sha(blob),
                "received_at": datetime.now(ZONE).isoformat(),
                "group": group,
            }
        except (httpx.HTTPError, ValueError) as exc:
            result = {
                "ok": False,
                "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                "group": group,
            }
            save(receipt, result)
            raise ValueError("PUBLIC_READ_STOP:" + result["reason"]) from exc
        save(receipt, result)
        return bytes(raw), result


def anchor(pages, phrase):
    """引用必须在一页中唯一出现；页码从 1 开始，不跨页拼凑表格数值。"""
    quote = normalize(phrase)
    hits = [(i + 1, normalize(p).index(quote)) for i, p in enumerate(pages) if quote in normalize(p)]
    if not quote or len(hits) != 1 or normalize(pages[hits[0][0] - 1]).count(quote) != 1:
        raise ValueError("ANCHOR_NOT_UNIQUE")
    page, offset = hits[0]
    return {"page": page, "normalized_offset": offset, "text": quote}


def explicit_money(value, unit):
    """保留原单位并用十进制换算人民币；不猜币种，不跨币种合计。"""
    factors = {"元": Decimal(1), "万元": Decimal(10000), "百万元": Decimal(1000000), "亿元": Decimal(100000000)}
    amount = Decimal(str(value).replace(",", "").replace("，", ""))
    if not amount.is_finite() or unit not in factors:
        raise ValueError("UNKNOWN_MONEY_UNIT_OR_VALUE")
    return {"value": str(amount), "unit": unit, "currency": "CNY", "cny": str(amount * factors[unit])}


def buyback_candidates(pages):
    """只抽明确的累计执行段；候选需逐条复核，不把计划额度/月内量当累计量。

    累计支付不可跨公告相加；这里不输出预测输入，未标明截止日/单位就不抽。
    """
    result = []
    amount = re.compile(
        r"(?:支付(?:的)?(?:总)?金额|成交(?:的)?总金额|成交总额|已支付的总金额)"
        r"(?:为|约为|合计为|合计)?(?:人民币)?([\d,]+(?:\.\d+)?)(亿元|万元|元)"
    )
    for number, raw in enumerate(pages, 1):
        text = normalize(raw)
        for match in re.finditer(r"截至(\d{4})年(\d{1,2})月(\d{1,2})日[^。；]{0,650}", text):
            sentence = match[0]
            if (
                "累计" not in sentence
                or "回购" not in sentence
                or any(w in sentence for w in ("拟回购", "计划回购", "不超过人民币", "不低于人民币"))
            ):
                continue
            hits = list(amount.finditer(sentence))
            if len(hits) != 1:
                continue
            found = hits[0]
            # 元/股是价格，不能当支付金额；已回购 0 股也不能据此补造金额 0。
            if sentence[found.end() :].startswith("/股"):
                continue
            result.append(
                {
                    "metric": "BUYBACK_CUMULATIVE_PAID",
                    "basis": "CUMULATIVE_EXECUTED_NOT_ADDITIVE",
                    "as_of": date(*map(int, match.groups())).isoformat(),
                    "amount": explicit_money(found[1], found[2]),
                    "fee_basis": "EXCLUDES_FEES"
                    if "不含交易费用" in sentence or "不含交易手续费" in sentence
                    else "UNRESOLVED",
                    "anchor": {"page": number, "normalized_offset": match.start(), "text": sentence},
                    "verification": "CANDIDATE_REQUIRES_INDIVIDUAL_REVIEW",
                }
            )
    return result


def pdf_revision(metadata, published):
    """晚于目录日的创建/修改元数据留下冲突待审，不能倒填或要求历史下载存档。"""
    issues = []
    for name in ("CreationDate", "ModDate"):
        value = metadata.get(name, "") or ""
        match = re.match(r"(?:D:)?(\d{4})(\d{2})(\d{2})", str(value))
        if match:
            try:
                day = date(*map(int, match.groups())).isoformat()
            except ValueError:
                issues.append({"field": name, "reason": "INVALID_PDF_DATE"})
                continue
            if day > published:
                issues.append({"field": name, "date": day, "reason": "PDF_METADATA_AFTER_CATALOG_DATE"})
    return issues
