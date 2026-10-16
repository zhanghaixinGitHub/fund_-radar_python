"""本基金已核验失效报告的精确备用来源，不进行开放式搜索或跨基金替换。"""

import hashlib
import re
from datetime import datetime, timedelta

from bs4 import BeautifulSoup

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, blob, now, read
from app.services.fund_materials_store import versioned_save

TITLE = "德邦鑫星价值灵活配置混合型证券投资基金2020年第1季度报告"
PRIMARY = "https://www.dbfund.com.cn/upload/pdf/2020-04-21-084255-" + TITLE + ".pdf"
REPRINT = "https://www.yilucaifu.com/fund/fundAnnounceDetail/102006090/102006091/0000000000000sdnrv.html"


def recover_report(client, entry, failure):
    """仅官网404的指定报告可回退；转载标识、正文日期、基金代码和表格仍逐项检查。

    保存转载原始HTML及实际接收时间，不能将其写成官网PDF或声称历史首次抓取。
    原官网失败附在回执中，后续同步复用30天内的转载原文但仍重验摘要和业务内容。
    """
    from app.integrations.dbfund_reports import bounded_get, parse_text

    if entry.get("url") != PRIMARY or entry.get("title") != TITLE or failure.get("http_status") != 404:
        raise ValueError("REPORT_RECOVERY_SCOPE_INVALID")
    cache = ROOT / "readiness-extended/report-recovery" / (digest(PRIMARY) + ".json")
    old = read(cache) if cache.exists() else None
    if old and now() - datetime.fromisoformat(old["checked_at"]) < timedelta(days=30):
        if old["url"] != REPRINT or old["primary_failure"]["url"] != PRIMARY:
            raise ValueError("REPORT_RECOVERY_SOURCE_CHANGED")
        raw = (ROOT / old["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != old["sha256"]:
            raise ValueError("REPORT_RECOVERY_RAW_CHANGED")
        receipt = old
    else:
        raw = bounded_get(client, REPRINT, 2_000_000)
        sha, filename = blob(raw, "html")
        receipt = {
            "url": REPRINT,
            "sha256": sha,
            "file": filename,
            "received_at": old["received_at"] if old and old["sha256"] == sha else now().isoformat(),
            "checked_at": now().isoformat(),
            "source": "PUBLIC_ISSUER_REPORT_REPRINT",
            "format": "HTML",
            "historical_first_seen_verified": False,
            "primary_failure": failure,
            "revised_after_receipt": bool(old and (old.get("revised_after_receipt") or old["sha256"] != sha)),
        }
    soup = BeautifulSoup(raw, "html.parser")
    wrapper = soup.select_one("div.content.bordernone")
    title = wrapper.select_one("h2") if wrapper else None
    info = wrapper.select_one("p.oif") if wrapper else None
    body = wrapper.select_one("div.text-content") if wrapper else None
    if (
        not title
        or re.sub(r"\s+", "", title.text) != "鑫星价值：2020年第1季度报告"
        or not info
        or "时间：2020-04-21" not in info.text
        or "源自：基金公司网站" not in info.text
        or not body
    ):
        raise ValueError("REPORT_RECOVERY_TITLE_OR_PUBLICATION_MISMATCH")
    parsed = parse_text([body.get_text("\n", strip=True)], TITLE, derive_missing_weights=True)
    if (
        parsed["fund_code"] != "002112"
        or parsed["report_end"] != "2020-03-31"
        or parsed["report_type"] != "QUARTER"
        or parsed["published_date"] != "2020-04-21"
    ):
        raise ValueError("REPORT_RECOVERY_IDENTITY_OR_PERIOD_MISMATCH")
    if receipt.get("revised_after_receipt"):
        # 已取得后的正文变化不能回填成2020年已知；旧版本仍由回执版本目录留存。
        parsed["available_at"] = max(parsed["available_at"], receipt["received_at"])
    parsed.update(
        {"source": {**entry, "recovery_url": REPRINT}, "raw": receipt, "parsed_at": now().isoformat(), "pages": None}
    )
    versioned_save(cache, receipt)
    return parsed
