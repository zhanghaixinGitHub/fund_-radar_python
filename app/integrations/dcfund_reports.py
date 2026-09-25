"""大成景恒的官网报告补充，调用公开披露页自身的只读分页查询。"""

import hashlib
import json
import re
import time
from datetime import datetime, timedelta

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, blob, now, read
from app.services.fund_materials_store import versioned_save

ORIGIN = "https://www.dcfund.com.cn"


def catalog(client):
    """固定 006038、历史范围和最多十页；空基金参数不可能扩为全市场。"""
    from app.integrations.public_fund_reports import STORE, report_period

    if hasattr(client, "dc_catalog"):
        return client.dc_catalog
    entries, expected = {}, None
    for number in range(1, 11):
        params = {
            "funcNo": "742003",
            "curtPageNo": number,
            "numPerPage": 20,
            "product_code": "006038",
            "ann_type": 0,
            "key_word": "景恒",
            "start_date": "2020-10-01",
            "end_date": "2024-12-31",
            "select_time": "",
        }
        url = ORIGIN + "/servlet/json"
        path = STORE / "receipts" / (digest({"dc_catalog": params}) + ".json")
        old = read(path) if path.exists() else None
        if old and now() - datetime.fromisoformat(old["checked_at"]) < timedelta(days=30):
            raw = (ROOT / old["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != old["sha256"]:
                raise ValueError("REPORT_CACHED_BODY_CHANGED")
            receipt = old
        else:
            if client.count >= 600:
                raise ValueError("REPORT_REQUEST_BUDGET")
            time.sleep(max(0, 0.4 - (time.monotonic() - client.last)))
            client.count += 1
            client.last = time.monotonic()
            # 网站前端使用 POST 提交查询条件；本调用不写基金公司数据。
            with client.client.stream(
                "POST", url, data=params, headers={"Referer": ORIGIN + "/main/aboutus/information/index.shtml"}
            ) as response:
                response.raise_for_status()
                raw = bytearray()
                for part in response.iter_bytes():
                    raw.extend(part)
                    if len(raw) > 2_000_000:
                        raise ValueError("REPORT_RESPONSE_TOO_LARGE")
                raw = bytes(raw)
            value = json.loads(raw)
            if value.get("error_no") != "0":
                raise ValueError("REPORT_PUBLIC_SOURCE_REJECTED")
            sha, filename = blob(raw, "json")
            receipt = {
                "url": url,
                "params": params,
                "sha256": sha,
                "file": filename,
                "checked_at": now().isoformat(),
                "source": "DC_ISSUER_CATALOG",
            }
            versioned_save(path, receipt)
        value = json.loads(raw)
        data = value["results"][0]
        if expected is None:
            expected = data["totalRows"]
        if data["currentPage"] != number or data["totalRows"] != expected or not 0 < expected <= 200:
            raise ValueError("REPORT_CATALOG_COUNT_CHANGED")
        for item in data["data"]:
            period = report_period(item["title"])
            if "大成景恒" not in item["title"] or not period:
                raise ValueError("REPORT_CATALOG_FUND_MISMATCH")
            if period in entries:
                raise ValueError("REPORT_REPRINT_PERIOD_AMBIGUOUS")
            attachment = item["attachment_url"]
            if not re.fullmatch(r"/home/working/download/\d{6}/[a-zA-Z0-9]+\.pdf", attachment):
                raise ValueError("REPORT_URL_NOT_PDF")
            entries[period] = {
                "url": ORIGIN + attachment,
                "title": item["title"],
                "published_date": item["pub_date"][:10],
                "catalog_receipt": receipt,
            }
        if len(entries) == expected:
            client.dc_catalog = entries
            return entries
        if not data["data"]:
            raise ValueError("REPORT_CATALOG_INCOMPLETE")
    raise ValueError("REPORT_CATALOG_INCOMPLETE")


def pdf(client, code, entry):
    """官网目录与原公告的报告期相同才读取 PDF，正文仍走共享身份和金额校验。"""
    if code != "006038":
        raise ValueError("REPORT_PEER_SCOPE_INVALID")
    item = catalog(client).get((entry["report_end"], entry["report_type"]))
    if not item:
        raise ValueError("REPORT_ISSUER_PERIOD_NOT_FOUND")
    raw, receipt = client.issuer_pdf(code, entry["ID"], url=item["url"])
    return raw, {**receipt, "catalog_receipt": item["catalog_receipt"]}, item
