"""公告证券列表未覆盖本份额时的人工身份复核，绑定目录、正文及管理人原件。

转载接口的证券列表不是完整适用名单。这里不自动扩大范围；本机审阅须给出
连续原文及明确的适用层级，任何正文、基金身份或来源变化均重新核验。
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import text

from app.db.session import get_engine
from app.integrations.market_fund_reports import read_source
from app.integrations.public_fund_reports import BODY_URL, CATALOG_URL
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save

SCOPES = {
    "MANAGER_SERVICE",
    "MANAGER_GOVERNANCE",
    "EXPLICIT_FUND_CODE",
    "EXPLICIT_PRODUCT_NAME",
    "MANAGER_CONDITIONAL_VALUATION",
}


def _compact(value):
    # 工商名称的全角括号和数据库半角括号只属排版差异；不删除名称文字、
    # 不折叠品牌简称，避免把另一家管理人也认作相同法律主体。
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _profile(code):
    """管理人以当前库内本份额档案为准，不能从待核验公告反推。"""
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text("""SELECT management_company_name, content_hash FROM fund_profile
                    WHERE fund_code=:code ORDER BY updated_at DESC,fund_profile_id LIMIT 1"""),
                {"code": code},
            )
            .mappings()
            .first()
        )
    if not row or not row["management_company_name"]:
        raise ValueError("MARKET_NEWS_MANAGER_PROFILE_REQUIRED")
    return dict(row)


def _entry(entry):
    return {key: entry[key] for key in ("id", "title", "published")}


def _catalog(root, code, entry, source):
    """目录必须是该份额的真实归档响应，标题、编号及公开日期逐项一致。"""
    if source.get("url") != CATALOG_URL or source.get("params", {}).get("fundcode") != code:
        raise ValueError("MARKET_NEWS_IDENTITY_CATALOG")
    value = json.loads(read_source(root, source))
    rows = value.get("Data")
    if value.get("ErrCode") != 0 or not isinstance(rows, list):
        raise ValueError("MARKET_NEWS_IDENTITY_CATALOG")
    matches = [row for row in rows if row.get("ID") == entry["id"]]
    if len(matches) != 1 or (
        matches[0].get("FUNDCODE") != code
        or matches[0].get("TITLE") != entry["title"]
        or str(matches[0].get("PUBLISHDATEDesc", ""))[:10] != entry["published"]
    ):
        raise ValueError("MARKET_NEWS_IDENTITY_CATALOG")


def _validate(root, code, entry, source, value):
    current = now()
    profile = _profile(code)
    raw = json.loads(read_source(root, source))
    data = raw.get("data") or {}
    body = data.get("notice_content", "")
    reviewed = datetime.fromisoformat(value["reviewed_at"])
    if (
        not re.fullmatch(r"[0-9]{6}", code)
        or source.get("url") != BODY_URL
        or source.get("params", {}).get("art_code") != entry["id"]
        or raw.get("success") != 1
        or data.get("art_code") != entry["id"]
        or not isinstance(body, str)
        or len(body.strip()) < 50
        or value.get("fund_code") != code
        or value.get("entry") != _entry(entry)
        or value.get("source_hash") != source["sha256"]
        or value.get("manager") != profile["management_company_name"]
        or value.get("profile_hash") != profile["content_hash"]
        or value.get("scope") not in SCOPES
        or reviewed.tzinfo is None
        or reviewed > current
        or entry["published"] > current.date().isoformat()
        or len(value.get("note", "").strip()) < 80
        or not 1 <= len(value.get("quotes", [])) <= 10
        or any(not isinstance(q, str) or len(q.strip()) < 15 or q not in body for q in value["quotes"])
    ):
        raise ValueError("MARKET_NEWS_IDENTITY_REVIEW_INVALID")
    for ref in (source, value["catalog_source"]):
        received = datetime.fromisoformat(ref["received_at"])
        if received.tzinfo is None or received > current or received > reviewed:
            raise ValueError("MARKET_NEWS_IDENTITY_SOURCE_TIME")
    _catalog(root, code, entry, value["catalog_source"])
    manager = _compact(profile["management_company_name"])
    proof = value.get("identity_proof")
    document = None
    if proof:
        material = read(Path(proof["materials_path"]))
        document = next((d for d in material["documents"] if d["id"] == proof["document_id"]), None)
        if (
            material.get("fund_code") != code
            or digest(material) != proof["materials_hash"]
            or not document
            or code not in document.get("codes", [])
            or digest(document["body"]) != document["body_hash"]
            or proof["quote"] not in document["body"]
            or len(proof["quote"].strip()) < 20
        ):
            raise ValueError("MARKET_NEWS_IDENTITY_PRODUCT_PROOF")
        available = datetime.fromisoformat(document["available_at"])
        if available.tzinfo is None or available > reviewed or document["published_date"] > str(current.date()):
            raise ValueError("MARKET_NEWS_IDENTITY_SOURCE_TIME")
    title_body = _compact(data.get("notice_title", "") + body)
    if manager not in title_body:
        anchor = value.get("manager_anchor", "")
        if (
            not document
            or len(anchor) < 5
            or _compact(anchor) not in _compact(body)
            or _compact(anchor) not in _compact(proof["quote"])
            or manager not in _compact(document["body"])
        ):
            raise ValueError("MARKET_NEWS_MANAGER_IDENTITY_UNPROVEN")
    if value["scope"] == "EXPLICIT_FUND_CODE" and not any(
        re.search(r"(?<!\d)" + code + r"(?!\d)", q) for q in value["quotes"]
    ):
        raise ValueError("MARKET_NEWS_EXPLICIT_FUND_REQUIRED")
    if value["scope"] == "EXPLICIT_PRODUCT_NAME":
        product = value.get("product_name", "")
        if (
            not document
            or len(product) < 12
            or not re.search(r"(?<!\d)" + code + r"(?!\d)", document["body"])
            or _compact(product) not in _compact(proof["quote"])
            or not any(_compact(product) in _compact(q) for q in value["quotes"])
        ):
            raise ValueError("MARKET_NEWS_EXPLICIT_PRODUCT_REQUIRED")
    if value["scope"] == "MANAGER_CONDITIONAL_VALUATION":
        # 管理人的条件式估值政策可以是本基金目录中的真实背景，但不能变成
        # “本基金当前持有该股票/已按此调整估值”的事实。当前仅接已核实的
        # 国泰官网份额档案格式，以及明确排除ETF的旗下基金股票估值公告。
        product = value.get("product_name", "")
        uri = urlparse(document.get("url", "")) if document else None
        if (
            not document
            or document.get("kind") != "FUND_PROFILE"
            or uri.scheme != "https"
            or uri.hostname != "e.gtfund.com"
            or uri.path != "/etrade/Jijin/view/id/" + code
            or len(product) < 12
            or re.search(r"ETF|交易型开放式", product, re.I)
            or _compact(product) not in _compact(proof["quote"])
            or not re.search(r"(?<!\d)" + code + r"(?!\d)", document["body"])
            or manager not in _compact(document["body"])
            or not re.fullmatch(re.escape(manager) + r"调整长期停牌股票估值方法的公告", _compact(entry["title"]))
            or "本公司旗下基金(ETF基金除外)持有的" not in _compact(body)
            or not re.search(r"证券代码[:：]\d{6}", _compact(body))
            or not any("本公司旗下基金(ETF基金除外)持有的" in _compact(q) for q in value["quotes"])
            or value.get("does_not_confirm_current_portfolio_application") is not True
        ):
            raise ValueError("MARKET_NEWS_CONDITIONAL_VALUATION_UNPROVEN")
    if value["scope"] in {"MANAGER_SERVICE", "MANAGER_GOVERNANCE"}:
        # 仅服务和机构治理类别允许管理人层级；部分基金清单、投资及估值事项不能借此放行。
        title = _compact(entry["title"])
        terms = (
            # 反诈声明、客服时段和官网升级属于管理人服务背景，也须逐篇人工复核。
            # 这里只补足类别词，不代替上方的管理人、目录、原件及引用校验；
            # 估值、部分基金交易、经理等经济事项仍不能借服务类型扩大适用范围。
            r"系统(?:升级)?维护|服务器维护|服务提示|直销业务迁移|基金销售业务|第三方互联网平台|办公地址变更"
            r"|谨防金融诈骗|防范金融诈骗|客服.*人工服务|官网.*(?:升级|维护)"
            r"|直销平台.*(?:完善|更新).*身份资料"
            if value["scope"] == "MANAGER_SERVICE"
            else r"设立.*分公司|高级管理人员|高管|参与上市公司治理.*(?:指引|规则)"
        )
        if re.search(r"旗下部分基金|关联交易|估值方法|基金经理|获配|承销证券", title) or not re.search(
            terms, title + _compact(body[:400])
        ):
            raise ValueError("MARKET_NEWS_MANAGER_SCOPE_UNPROVEN")
    return value


def record_applicability_review(
    code,
    entry,
    source,
    catalog_source,
    scope,
    note,
    quotes,
    *,
    root,
    identity_proof=None,
    manager_anchor=None,
    product_name=None,
):
    """本机人工复核入口；管理人服务/治理仅作背景，不等于本基金经济事项。"""
    root = Path(root)
    profile = _profile(code)
    value = {
        "fund_code": code,
        "entry": _entry(entry),
        "source_hash": source["sha256"],
        "catalog_source": catalog_source,
        "manager": profile["management_company_name"],
        "profile_hash": profile["content_hash"],
        "scope": scope,
        "note": note,
        "quotes": quotes,
        "identity_proof": identity_proof,
        "manager_anchor": manager_anchor,
        "product_name": product_name,
        "reviewed_at": now().isoformat(),
        "does_not_determine_direction": True,
    }
    if scope == "MANAGER_CONDITIONAL_VALUATION":
        value["does_not_confirm_current_portfolio_application"] = True
    _validate(root, code, entry, source, value)
    versioned_save(root / "identity-reviews" / code / (source["sha256"] + ".json"), value)
    return value


def checked_applicability(root, code, entry, source):
    """采集器只消费核验通过且仍匹配本正文的复核，缺回执仍按原身份规则拒绝。"""
    path = Path(root) / "identity-reviews" / code / (source["sha256"] + ".json")
    if not path.exists():
        return None
    value = _validate(Path(root), code, entry, source, read(path))
    if entry.get("catalog_source"):
        _catalog(Path(root), code, entry, entry["catalog_source"])
    return value
