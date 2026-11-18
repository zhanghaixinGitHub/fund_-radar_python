"""共享公司新原文进入未批准候选；不写正式预测，不把别的公司或同行数据带入基金。"""

import re
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from app.services.direction_1d_protocol import digest, window
from app.services.fund_exposure_common import ROOT, now, read
from app.services.fund_materials_store import versioned_save
from app.services.fund_workflow_materials import workflow_root
from app.services.material_content import source_bytes, content_digest, location_exists


def _require(ok, message):
    if not ok:
        raise ValueError("COMPANY_SHARED_ORIGINAL_" + message)


def _companies(material):
    return {row["code"] for row in material.get("companies", [])
            if re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", row.get("code", ""))}


def _validated(record, security, at, raw_root):
    from app.integrations.dbfund_supplement import article_text

    _require(record["stock_code"] == security[:6], "ISSUER_MISMATCH")
    receipt = record["receipt"]
    url = urlparse(receipt["url"])
    _require(url.scheme == "https" and url.netloc == "static.cninfo.com.cn"
             and url.path.lower().endswith(".pdf"), "SOURCE_URL")
    raw_path = (Path(raw_root) / receipt["file"]).resolve()
    _require(raw_path.is_relative_to(Path(raw_root).resolve()), "SOURCE_PATH")
    raw = source_bytes(raw_path)
    _require(content_digest(raw) == receipt["sha256"], "SOURCE_HASH")
    available = datetime.fromisoformat(receipt["received_at"])
    _require(available.tzinfo is not None and available <= at, "ACTUAL_AVAILABILITY")
    published = record["announced_at_source"][:10]
    _require(published <= str(available.date()) and published <= str(at.date()), "PUBLICATION_DATE")
    pages, status = article_text(raw, True)
    _require(status == "TEXT_EXTRACTED" and bool(pages), "FULL_TEXT_REQUIRED")
    body = "\n".join(pages)
    has_code = bool(re.search(r"(?<!\d)" + security[:6] + r"(?!\d)", body[:20000]))
    # 问询回复封面可能只写公司全称，不印证券代码；须同时重放真实目录的
    # 证券、公告id、附件URL和正文公司名，不能只凭标题或人工声称关联。
    proof = record.get("catalog_proof")
    if proof:
        catalog_raw = source_bytes(Path(proof["path"]))
        _require(content_digest(catalog_raw) == proof["sha256"], "CATALOG_SOURCE_HASH")
        entries = [row for row in json.loads(catalog_raw).get("announcements", [])
                   if str(row["announcementId"]) == str(record["announcement_id"])]
        _require(len(entries) == 1 and entries[0]["secCode"] == security[:6]
                 and "https://static.cninfo.com.cn/" + entries[0]["adjunctUrl"] == receipt["url"], "CATALOG_ISSUER_BINDING")
        short_name = re.sub(r"\s|<[^>]*>", "", entries[0]["secName"])
        has_code = has_code or len(short_name) >= 2 and short_name in re.sub(r"\s", "", body[:20000])
    _require(has_code, "BODY_ISSUER")
    return dict(id="shared-company:" + record["announcement_id"] + ":" + receipt["sha256"],
                kind="ANNOUNCEMENT", title=record["title"], url=receipt["url"],
                body=body, body_hash=digest(body), body_scope="FULL_TEXT", published_date=published,
                available_at=available.isoformat(), issuer_code=security,
                raw_path=str(raw_path), raw_sha256=receipt["sha256"],
                availability_note="实际取得时间；不是原披露日已取得的证明")


def register(material, security, record, *, root=None, raw_root=ROOT, at=None):
    """从现有公司collector收据登记共享原件及下一实际窗口待办，保持历史不动。"""
    at, root = at or now(), workflow_root(root)
    _require(security in _companies(material), "UNRELATED_SECURITY")
    document = _validated(record, security, at, raw_root)
    path = root / "shared-company-originals" / (security + ".json")
    previous = read(path) if location_exists(path) else {"security": security, "items": {}}
    _require(previous.get("security") == security, "INDEX_IDENTITY")
    key = record["announcement_id"] + ":" + document["raw_sha256"]
    if key in previous["items"]:
        _require(previous["items"][key]["document"] == document, "INDEX_CONTENT_CHANGED")
        return dict(status="UNCHANGED", security=security, document_id=document["id"], index_path=str(path))
    item = dict(document=document, source_record=deepcopy(record), raw_root=str(Path(raw_root).resolve()),
                registered_at=at.isoformat(), origin_fund_code=material["fund_code"],
                origin_material_hash=digest(material), approved=False)
    updated = deepcopy(previous)
    updated["items"][key] = item
    versioned_save(path, updated)
    todo_path = root / "company-original-reviews" / (material["fund_code"] + ".json")
    todo = read(todo_path) if location_exists(todo_path) else dict(fund_code=material["fund_code"], items={})
    todo["items"][document["id"]] = dict(status="PENDING_INDEPENDENT_REVIEW", security=security,
        registered_at=at.isoformat(), available_at=document["available_at"], raw_sha256=document["raw_sha256"],
        required_window=window(at), existing_predictions_changed=False,
        action="在下一有效资料窗口读取全文、重审四周期及建议；不继承旧方向，不倒填已保存历史")
    versioned_save(todo_path, todo)
    return dict(status="REGISTERED_UNREVIEWED", security=security, document_id=document["id"],
                index_path=str(path), next_window_todo=str(todo_path), window=window(at))


def refresh(material, *, root=None, checked_at=None):
    """即使已有财报和当日报价，也读取新的关联公司事项；只追加未批准候选。"""
    at, root = checked_at or now(), workflow_root(root)
    result = {"status": "UNCHANGED", "documents_added": [], "already_present": [], "approved": False}
    for security in sorted(_companies(material)):
        path = root / "shared-company-originals" / (security + ".json")
        if not location_exists(path):
            continue
        index = read(path)
        _require(index["security"] == security, "INDEX_IDENTITY")
        for item in index["items"].values():
            document = _validated(item["source_record"], security, at, item["raw_root"])
            _require(document == item["document"] and item.get("approved") is False, "INDEX_CONTENT_CHANGED")
            # 同字节原文已经以别的来源id入包时，不重复插入正文。
            existing = next((d for d in material["documents"] if d.get("raw_sha256") == document["raw_sha256"]
                             and d.get("issuer_code") in {None, security}), None)
            if existing:
                result["already_present"].append(existing["id"])
                continue
            document = dict(document, codes=[material["fund_code"]])
            material["documents"].append(document)
            material.setdefault("source_manifest", {}).setdefault("shared_company_originals", {})[document["id"]] = {
                "security": security, "raw_sha256": document["raw_sha256"],
                "available_at": document["available_at"], "review_status": "UNREVIEWED"}
            result["documents_added"].append(document["id"])
    if result["documents_added"]:
        result["status"] = "NEW_ORIGINALS_REVIEW_REQUIRED"
    return result
