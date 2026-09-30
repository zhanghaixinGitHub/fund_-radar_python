"""近期披露持仓公司的公告事实；独立于历史研究准入，不把公告标题当经营成果。"""

import json
import re
import unicodedata
from datetime import datetime, timedelta
from urllib.parse import urlparse

from app.integrations.cninfo_exposure import acquire_announcements, acquire_attachments
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, read
from app.services.fund_exposure_runtime import execution_lock
from app.services.fund_exposure_supplement import verified_bytes
from app.services.fund_materials import snapshot
from app.services.fund_materials_store import source_path

RULE = "DISCLOSED_HOLDING_COMPANY_PUBLICATION_V1"
# 首次实测目录有207份近期原件。文档处理与网络请求分别限额，复用已保存原件，
# 不能因“每次只取最新30份”永久饿死较早待核验资料；超过250份仍明确部分覆盖。
LIMITS = {"days": 30, "companies": 30, "documents": 250, "catalog_requests": 100, "body_requests": 60}


def compact(value):
    """只统一版面空白和全半角，不移除实质词语或修改证券代码。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def publication_title_anchor(item, text):
    """核对标题实质文字，仅兼容明确的排版差异与目录前置证券简称。

    书名号和左右引号不改变标题字词；括号及其中的草案、修订、适用条件必须保留。
    去掉目录开头的简称时，正文仍须通过独立公司身份检查，且剩余标题至少八字。
    不删除年份、次数、否定词或“的”等字，也不采用相似度阈值。
    """
    def normalize(value):
        return compact(value).translate(str.maketrans("", "", "《》“”‘’"))

    title = normalize(item["title_plain"])
    body = normalize(text[:4000])
    if title and title in body:
        return {"rule": "EXACT_WORDS_WITH_LAYOUT_NORMALIZATION", "anchor": title}
    name = normalize(item["secName"])
    if name and title.startswith(name):
        remainder = title[len(name):].removeprefix(":")
        if len(remainder) >= 8 and remainder in body:
            return {"rule": "CATALOG_SECURITY_NAME_PREFIX", "anchor": remainder, "security_name": name}
    return None


def legacy_catalog_evidence(item, document, stock_code):
    """用旧目录原始回执补证旧版缺失的摘要字段，不改旧原件，也不接受已存在但不匹配的摘要。

    旧目录行、原始响应行、当前目录四项版本元数据必须完全一致；原响应字节重验摘要。
    此处只证明已留存原件与目录身份的绑定，不宣称 URL 从未发生未公告的字节修订。
    """
    if document.get("catalog_hash") is not None:
        return None
    path = ROOT / "supplement/company-announcements" / (stock_code + ".json")
    if not path.exists():
        return None
    catalog = read(path)
    old = next((r for r in catalog["rows"] if r["announcementId"] == item["announcementId"]), None)
    fields = ("adjunctUrl", "adjunctSize", "announcementTitle", "announcementTime")
    if not old or any(old.get(k) != item.get(k) for k in fields):
        return None
    receipt = old.get("receipt", {})
    if receipt.get("url") != "https://www.cninfo.com.cn/new/hisAnnouncement/query":
        return None
    raw = json.loads(verified_bytes(receipt))
    row = next((r for r in (raw.get("announcements") or []) if r.get("announcementId") == item["announcementId"]), None)
    if not row or any(row.get(k) != item.get(k) for k in fields):
        return None
    if (
        row.get("secCode") != stock_code.split(".")[0]
        or document.get("title") != item["title_plain"]
        or document.get("announced_at_source") != item["announced_at_source"]
    ):
        return None
    return {
        "metadata_hash": digest({k: row.get(k) for k in fields}),
        "receipt": receipt,
        "original_row": row,
        "kind": "ORIGINAL_CATALOG_RECEIPT_REPLAY",
    }


def company_identity_evidence(stock_code):
    """复用已留档的证券代码与法定全名，不凭简称猜测没有代码页眉的公司身份。"""
    path = ROOT / "supplement/stock-context.json"
    if not path.exists():
        return None
    context = read(path)
    row = context.get("stock_basic", {}).get(stock_code)
    if (
        not row
        or row.get("ts_code") != stock_code
        or row.get("symbol") != stock_code.split(".")[0]
        or not row.get("fullname")
        or len(compact(row["fullname"])) < 8
    ):
        return None
    return {
        "row": row,
        "context_hash": digest(context),
        "as_of": context.get("at"),
        "meaning": "CURRENT_SECURITY_IDENTITY_NOT_HISTORICAL_INDUSTRY",
    }


def inspect_company(item, document, holding, report, checked_at, *, legacy_catalog=None, identity_evidence=None):
    """核验目录、原件、公司身份、标题及来源显示日，输出仅为“披露了文件”。

    公开日精度仅到日；目录显示时间与正文签署日期不强行合并，不认定历史首次公开。
    原件字节由调用方核对。股权激励草案不等于实施，合同不等于收入，回购也不等于利好。
    """
    code = holding["stockCode"].split(".")[0]
    day = datetime.fromisoformat(item["announced_at_source"]).date()
    receipt = document["receipt"]
    received = datetime.fromisoformat(receipt["received_at"])
    parsed = urlparse(receipt["url"])
    metadata = {k: item.get(k) for k in ("adjunctUrl", "adjunctSize", "announcementTitle", "announcementTime")}
    text = compact("\n".join(document.get("pages", [])))
    title_anchor = publication_title_anchor(item, text)
    identity = bool(re.search(r"(?:证券|股票)代码[:：]?" + code + r"(?!\d)", text[:1200]))
    identity = identity or (compact(item["secName"]) in text[:600] and code in text[:600])
    if identity_evidence:
        source_identity = identity_evidence["row"]
        identity = identity or (
            source_identity.get("ts_code") == holding["stockCode"]
            and source_identity.get("symbol") == code
            and len(compact(source_identity.get("fullname", ""))) >= 8
            and compact(source_identity["fullname"]) in text[:600]
        )
    checks = {
        "BODY_NOT_EXTRACTED": document.get("text_status") == "TEXT_EXTRACTED",
        "DOCUMENT_ID_MISMATCH": document.get("announcement_id") == item["announcementId"],
        "STOCK_CODE_MISMATCH": document.get("stock_code") == code and item.get("secCode") == code,
        "CATALOG_VERSION_MISMATCH": document.get("catalog_hash") == digest(metadata)
        or (
            document.get("catalog_hash") is None
            and legacy_catalog is not None
            and legacy_catalog.get("metadata_hash") == digest(metadata)
        ),
        "SOURCE_URL_MISMATCH": parsed.scheme == "https"
        and parsed.netloc == "static.cninfo.com.cn"
        and parsed.path.lower() == "/" + item["adjunctUrl"].lower(),
        "SOURCE_DATE_MISMATCH": parsed.path.split("/")[2:3] == [day.isoformat()],
        "RECEIPT_TIME_INVALID": received.tzinfo is not None and received <= checked_at,
        "OUTSIDE_RECENT_WINDOW": checked_at.date() - timedelta(days=LIMITS["days"]) <= day <= checked_at.date(),
        "NOT_LISTED_ON_RECHECK": item.get("listing_status", "LISTED") == "LISTED",
        "TITLE_ANCHOR_MISSING": title_anchor is not None,
        "COMPANY_IDENTITY_ANCHOR_MISSING": identity,
        "BODY_TOO_SHORT": len(text) >= 100,
    }
    failed = [reason for reason, passed in checks.items() if not passed]
    reviewed = None
    if failed and set(failed) <= {
        "BODY_NOT_EXTRACTED", "TITLE_ANCHOR_MISSING", "COMPANY_IDENTITY_ANCHOR_MISSING", "BODY_TOO_SHORT",
    }:
        # 只兼容同一版本已逐份核验的正文/OCR版式问题，目录、原件、时间和公司代码门槛不变。
        from app.services.company_publication_review import reviewed_publication

        reviewed = reviewed_publication(item, document, checked_at)
        if reviewed:
            failed = []
    if failed:
        raise ValueError("COMPANY_PUBLICATION_NOT_VERIFIED:" + ",".join(failed))
    relation = (
        f"{holding['stockName']}（{code}）见基金截至 {report['endDate']} 的披露持仓，"
        f"报告于 {report['publishedDate']} 公开，占基金净资产 {holding['weightPct']}%；不代表当前实际仓位"
    )
    return {
        "event_id": digest({"source": "CNINFO_PUBLIC_DISCLOSURE", "id": item["announcementId"]}),
        "content_hash": receipt["sha256"],
        "evidence_hash": digest({"item": item, "document": document, **({"review": reviewed} if reviewed else {})}),
        "title": f"{holding['stockName']}：{item['title_plain']}",
        "source_name": "巨潮资讯",
        "source_url": receipt["url"],
        "published_date": day.isoformat(),
        "publication_precision": "DAY",
        "first_received_at": receipt["received_at"],
        "effective_at": None,
        "stage": "已披露文件",
        "summary": reviewed["business_boundary"] if reviewed else
        "公司已披露该文件；事项是否生效、实施或形成收入，需结合原文进一步核实。",
        "business_eligible": True,
        "prediction_eligible": False,
        "training_eligible": False,
        "relation": relation,
        "evidence": {
            "catalog": item,
            "document": document,
            "holding": holding,
            "report_date": report["endDate"],
            "report_published_date": report["publishedDate"],
            "rule": RULE,
            "legacy_catalog_evidence": legacy_catalog,
            "company_identity_evidence": identity_evidence,
            "publication_title_anchor": title_anchor,
            "verified_publication_review": reviewed,
        },
    }


def collect_company_news(checked_at, progress=lambda *args: None):
    """复用原巨潮采集器和资料锁，最多 30 公司、250 近期文件；先定范围，再核验原文。"""
    data = snapshot("002112")
    if not data or data.get("fundCode") != "002112":
        raise ValueError("COMPANY_NEWS_HOLDINGS_MISSING")
    candidates = [r for r in data["reports"] if r["endDate"] <= r["publishedDate"] <= str(checked_at.date())]
    if not candidates:
        raise ValueError("COMPANY_NEWS_REPORT_DATE_UNKNOWN")
    report = max(candidates, key=lambda r: (r["endDate"], r["publishedDate"], r.get("fullDisclosure", False)))
    age = (checked_at.date() - datetime.fromisoformat(report["endDate"]).date()).days
    holdings = {h["stockCode"]: h for h in report["holdings"]}
    if age > 210 or not 1 <= len(holdings) <= LIMITS["companies"] or len(holdings) != len(report["holdings"]):
        raise ValueError("COMPANY_NEWS_HOLDINGS_SCOPE_INVALID")
    lower = str(checked_at.date() - timedelta(days=LIMITS["days"]))
    scope = {
        "windows": {c: [[lower, str(checked_at.date())]] for c in holdings},
        "cutoff": str(checked_at.date()),
        "check_id": checked_at.isoformat(),
        "maximum_pages_per_window": 5,
        "maximum_new_requests": LIMITS["catalog_requests"],
        "training_eligible": False,
    }
    with execution_lock() as locked:
        if not locked:
            raise ValueError("COMPANY_NEWS_MATERIALS_BUSY")
        collected = acquire_announcements(live_scope=scope, progress=progress)
        errors = list(collected.get("errors", []))
        entries = []
        for code in sorted(holdings):
            path = source_path(ROOT, "supplement/company-announcements/" + code + ".json")
            if not path.exists():
                errors.append({"code": code, "reason": "COMPANY_CATALOG_MISSING"})
                continue
            catalog = read(path)
            # 来源失败时保留旧页面，但本轮不能拿旧目录冒称重新核对成功。
            if catalog.get("scope_hash") != digest(scope):
                errors.append({"code": code, "reason": "COMPANY_CATALOG_NOT_CURRENT"})
                continue
            for item in catalog["rows"]:
                if lower <= item["announced_at_source"][:10] <= str(checked_at.date()):
                    entries.append((code, item))
        entries.sort(key=lambda pair: (pair[1]["announced_at_source"], pair[1]["announcementId"]), reverse=True)
        selected = entries[: LIMITS["documents"]]
        attached = acquire_attachments(
            live=True,
            progress=progress,
            announcement_ids={item["announcementId"] for _, item in selected},
            request_limit=LIMITS["body_requests"],
        )
        errors.extend(attached.get("errors", []))
        facts = []
        for code, item in selected:
            try:
                document = read(
                    source_path(ROOT, "supplement/company-documents/" + digest(item["announcementId"]) + ".json")
                )
                verified_bytes(document["receipt"])
                # 下载发生在范围冻结后，按实际完成时间验收，不能伪装成冻结前取得。
                from app.services.fund_exposure_common import now

                legacy = legacy_catalog_evidence(item, document, code)
                identity = company_identity_evidence(code)
                facts.append(
                    inspect_company(
                        item, document, holdings[code], report, now(), legacy_catalog=legacy, identity_evidence=identity
                    )
                )
            except (ValueError, KeyError, OSError, IndexError) as error:
                reason = str(error) if isinstance(error, ValueError) else type(error).__name__
                errors.append({"id": item["announcementId"], "reason": reason})
        return {
            "facts": facts,
            "errors": errors,
            "limited": len(entries) > len(selected),
            "coverage": {
                "companies": len(holdings),
                "catalog_documents": len(entries),
                "selected_documents": len(selected),
                "verified_documents": len(facts),
                "rejected_documents": len(selected) - len(facts),
                "report_date": report["endDate"],
                "report_published_date": report["publishedDate"],
                "limits": LIMITS,
            },
        }
