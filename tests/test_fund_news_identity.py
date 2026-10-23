"""公告缺失证券代码时必须有逐篇身份凭据，不能扩大管理人或部分基金的适用范围。"""

import json
from datetime import datetime, timedelta

import pytest
from app.integrations import market_fund_reports
from app.integrations.public_fund_reports import BODY_URL, CATALOG_URL
from app.services import fund_market_news as news
from app.services import fund_news_identity as identity
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save

CODE, ARTICLE = "010737", "AN202610151830508374"
TITLE = "某某基金管理有限公司关于网上交易平台系统维护的通知"
BODY = (
    "某某基金管理有限公司通知所有网上交易客户，本公司网上交易平台将于周末进行系统维护，"
    "期间可能无法登录。请投资者提前安排相关业务。"
)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    tmp_path = tmp_path / "workflow" / "news"
    instant = datetime.fromisoformat("2026-10-21T10:00:00+08:00")
    monkeypatch.setattr(news, "now", lambda: instant)
    monkeypatch.setattr(market_fund_reports, "now", lambda: instant)
    monkeypatch.setattr(identity, "now", lambda: instant)
    monkeypatch.setattr(
        identity,
        "_profile",
        lambda _: {
            "management_company_name": "某某基金管理有限公司",
            "content_hash": "profile-1",
        },
    )
    client = news.MarketNewsClient(tmp_path)
    entry = {"id": ARTICLE, "title": TITLE, "published": "2026-10-15"}
    raw = json.dumps(
        {
            "success": 1,
            "data": {
                "art_code": ARTICLE,
                "notice_title": TITLE,
                "notice_content": BODY,
                "security": [{"stock": "000001"}],
            },
        },
        ensure_ascii=False,
    ).encode()
    source = client._archive(raw, ".json", BODY_URL, {"art_code": ARTICLE})
    catalog = client._archive(
        json.dumps(
            {
                "ErrCode": 0,
                "Data": [
                    {
                        "ID": ARTICLE,
                        "FUNDCODE": CODE,
                        "TITLE": TITLE,
                        "PUBLISHDATEDesc": "2026-10-15",
                    }
                ],
            }
        ).encode(),
        ".json",
        CATALOG_URL,
        {"fundcode": CODE},
    )
    yield tmp_path, client, entry, source, catalog, instant
    client.close()


def review(sources, **changes):
    root, _, entry, source, catalog, _ = sources
    args = dict(
        code=CODE,
        entry=entry,
        source=source,
        catalog_source=catalog,
        scope="MANAGER_SERVICE",
        note="已阅读全文核实本公告仅涉及该管理人网上交易系统的维护及客户服务，不能推定本基金投资组合、净值估值或管理团队发生变化。当前基金档案管理人、基金专属目录与公告标题及正文一致，作为渠道服务背景保留。",
        quotes=[BODY],
        root=root,
    )
    args.update(changes)
    return identity.record_applicability_review(**args)


def test_missing_security_requires_review_and_preserves_failed_receipt(sources):
    root, client, entry, source, _, _ = sources
    with pytest.raises(ValueError, match="BODY_UNVERIFIED"):
        client.news_body(CODE, entry, {"entry": entry, "source": source})
    assert read(root / "body-receipts" / CODE / (ARTICLE + ".json")) == source
    receipt = review(sources)
    actual = client.news_body(CODE, entry, {"entry": entry, "source": source})
    assert actual["applicability_review"] == receipt
    assert receipt["scope"] == "MANAGER_SERVICE"


def test_cross_fund_catalog_and_manager_cannot_be_reused(sources, monkeypatch):
    with pytest.raises(ValueError, match="CATALOG"):
        review(sources, code="005284")
    monkeypatch.setattr(
        identity,
        "_profile",
        lambda _: {
            "management_company_name": "其他基金管理有限公司",
            "content_hash": "profile-2",
        },
    )
    with pytest.raises(ValueError, match="MANAGER_IDENTITY"):
        review(sources)


def test_review_changes_and_forged_quote_are_rejected(sources, monkeypatch):
    root, _, entry, source, _, instant = sources
    review(sources)
    with pytest.raises(ValueError, match="REVIEW_INVALID"):
        identity.checked_applicability(root, CODE, {**entry, "title": "其他标题"}, source)
    with pytest.raises(ValueError, match="REVIEW_INVALID"):
        review(sources, quotes=["不能把没有出现在真实原文中的文字作为身份关联依据。"])
    monkeypatch.setattr(identity, "now", lambda: instant - timedelta(hours=1))
    with pytest.raises(ValueError, match="REVIEW_INVALID"):
        identity.checked_applicability(root, CODE, entry, source)


def test_partial_fund_investment_announcement_cannot_be_manager_context(sources):
    root, client, _, source, _, _ = sources
    entry = {
        "id": ARTICLE,
        "title": "某某基金管理有限公司关于旗下部分基金关联交易事项的公告",
        "published": "2026-10-15",
    }
    catalog = client._archive(
        json.dumps(
            {
                "ErrCode": 0,
                "Data": [
                    {
                        "ID": ARTICLE,
                        "FUNDCODE": CODE,
                        "TITLE": entry["title"],
                        "PUBLISHDATEDesc": "2026-10-15",
                    }
                ],
            }
        ).encode(),
        ".json",
        CATALOG_URL,
        {"fundcode": CODE},
    )
    with pytest.raises(ValueError, match="MANAGER_SCOPE_UNPROVEN"):
        review(sources, entry=entry, catalog_source=catalog)
    with pytest.raises(ValueError, match="EXPLICIT_FUND_REQUIRED"):
        review(sources, entry=entry, catalog_source=catalog, scope="EXPLICIT_FUND_CODE")


def test_changed_raw_body_cannot_consume_old_identity_review(sources):
    root, client, entry, _, _, _ = sources
    review(sources)
    source = client._archive(
        json.dumps(
            {
                "success": 1,
                "data": {
                    "art_code": ARTICLE,
                    "notice_title": TITLE,
                    "notice_content": BODY + "内容已更新。",
                    "security": [{"stock": "000001"}],
                },
            }
        ).encode(),
        ".json",
        BODY_URL,
        {"art_code": ARTICLE},
    )
    assert identity.checked_applicability(root, CODE, entry, source) is None


@pytest.mark.parametrize(
    ("title", "scope"),
    [
        ("某某基金管理有限公司提醒投资者谨防金融诈骗的公告", "MANAGER_SERVICE"),
        ("某某基金管理有限公司国庆期间暂停客服人工服务的通知", "MANAGER_SERVICE"),
        ("某某基金管理有限公司关于官网完成升级正式上线的公告", "MANAGER_SERVICE"),
        ("某某基金管理有限公司关于直销系统升级维护暂停网上交易服务的通知", "MANAGER_SERVICE"),
        ("某某基金管理有限公司提醒直销平台投资者完善、更新身份资料的公告", "MANAGER_SERVICE"),
        ("某某基金管理有限公司参与上市公司治理管理工作指引", "MANAGER_GOVERNANCE"),
    ],
)
def test_reviewed_manager_notices_still_require_matching_catalog_and_original(sources, title, scope):
    """服务和治理类别扩充不放宽逐份额目录、原件和人工复核的身份边界。"""
    root, client, _, _, _, _ = sources
    entry = {"id": ARTICLE, "title": title, "published": "2026-10-15"}
    source = client._archive(
        json.dumps(
            {
                "success": 1,
                "data": {
                    "art_code": ARTICLE,
                    "notice_title": title,
                    "notice_content": BODY,
                    "security": [{"stock": "000001"}],
                },
            },
            ensure_ascii=False,
        ).encode(),
        ".json",
        BODY_URL,
        {"art_code": ARTICLE},
    )
    catalog = client._archive(
        json.dumps(
            {"ErrCode": 0, "Data": [{"ID": ARTICLE, "FUNDCODE": CODE, "TITLE": title, "PUBLISHDATEDesc": "2026-10-15"}]}
        ).encode(),
        ".json",
        CATALOG_URL,
        {"fundcode": CODE},
    )
    assert identity.checked_applicability(root, CODE, entry, source) is None
    result = review(sources, entry=entry, source=source, catalog_source=catalog, scope=scope)
    assert result["does_not_determine_direction"] is True
    with pytest.raises(ValueError, match="CATALOG"):
        review(sources, entry=entry, source=source, catalog_source=catalog, scope=scope, code="005284")
    with pytest.raises(ValueError, match="REVIEW_INVALID"):
        review(
            sources,
            entry=entry,
            source=source,
            catalog_source=catalog,
            scope=scope,
            quotes=["未出现在原件中的服务范围说明不能成为复核依据。"],
        )


@pytest.mark.parametrize(
    "title",
    [
        "某某基金管理有限公司关于客服维护及旗下部分基金关联交易的公告",
        "某某基金管理有限公司关于官网升级及调整股票估值方法的公告",
    ],
)
def test_new_service_terms_cannot_hide_product_economic_events(sources, title):
    """复合标题含投资或估值事项时，服务关键词不能将其泛化到旗下基金。"""
    _, client, _, _, _, _ = sources
    entry = {"id": ARTICLE, "title": title, "published": "2026-10-15"}
    catalog = client._archive(
        json.dumps(
            {"ErrCode": 0, "Data": [{"ID": ARTICLE, "FUNDCODE": CODE, "TITLE": title, "PUBLISHDATEDesc": "2026-10-15"}]}
        ).encode(),
        ".json",
        CATALOG_URL,
        {"fundcode": CODE},
    )
    with pytest.raises(ValueError, match="MANAGER_SCOPE_UNPROVEN"):
        review(sources, entry=entry, catalog_source=catalog)


def test_manager_legal_name_width_difference_is_typography(sources, monkeypatch):
    """同一法律名称的括号宽度可归一化，但不能删除名称或换管理人。"""
    assert identity._compact("摩根基金管理（中国）有限公司") == identity._compact("摩根基金管理(中国)有限公司")
    assert identity._compact("摩根基金管理（中国）有限公司") != identity._compact("摩根资产管理有限公司")


def test_conditional_valuation_never_confirms_current_portfolio(sources):
    """条件式政策须有官网份额档案，ETF及缺条件原文均拒绝，现有服务类别不放行。"""
    root, client, _, _, _, instant = sources
    product = "某某有色金属行业指数证券投资基金"
    title = "某某基金管理有限公司调整长期停牌股票估值方法的公告"
    body = (
        "某某基金管理有限公司与托管银行商定，对本公司旗下基金（ETF基金除外）持有的"
        "星帅尔（证券代码：002860）股票进行估值调整，按指数收益法估值。"
        "待股票体现活跃市场交易特征后恢复采用收盘价格，届时不再另行公告。"
    )
    entry = {"id": ARTICLE, "title": title, "published": "2026-10-15"}
    source = client._archive(
        json.dumps(
            {"success": 1, "data": {"art_code": ARTICLE, "notice_title": title, "notice_content": body}},
            ensure_ascii=False,
        ).encode(),
        ".json",
        BODY_URL,
        {"art_code": ARTICLE},
    )
    catalog = client._archive(
        json.dumps(
            {"ErrCode": 0, "Data": [{"ID": ARTICLE, "FUNDCODE": CODE, "TITLE": title, "PUBLISHDATEDesc": "2026-10-15"}]}
        ).encode(),
        ".json",
        CATALOG_URL,
        {"fundcode": CODE},
    )
    profile_body = f"某某基金管理有限公司\n基金全称 {product}\n基金代码 {CODE}\n普通开放式"
    document = {
        "id": "official-profile",
        "kind": "FUND_PROFILE",
        "codes": [CODE],
        "url": "https://e.gtfund.com/etrade/Jijin/view/id/" + CODE,
        "body": profile_body,
        "body_hash": digest(profile_body),
        "published_date": "2026-10-15",
        "available_at": instant.isoformat(),
    }
    material = {"fund_code": CODE, "documents": [document]}
    path = root / "conditional-material.json"
    versioned_save(path, material)
    proof = {
        "materials_path": str(path),
        "materials_hash": digest(material),
        "document_id": document["id"],
        "quote": profile_body,
    }
    args = dict(
        entry=entry,
        source=source,
        catalog_source=catalog,
        quotes=[body],
        scope="MANAGER_CONDITIONAL_VALUATION",
        identity_proof=proof,
        product_name=product,
    )
    actual = review(sources, **args)
    assert actual["does_not_confirm_current_portfolio_application"] is True
    assert actual["does_not_determine_direction"] is True
    with pytest.raises(ValueError, match="MANAGER_SCOPE_UNPROVEN"):
        review(sources, **{**args, "scope": "MANAGER_SERVICE"})
    with pytest.raises(ValueError, match="CONDITIONAL_VALUATION_UNPROVEN"):
        review(sources, **{**args, "identity_proof": None})
    with pytest.raises(ValueError, match="CONDITIONAL_VALUATION_UNPROVEN"):
        review(sources, **{**args, "product_name": "某某有色金属行业交易型开放式指数证券投资基金"})
    changed = {**actual, "does_not_confirm_current_portfolio_application": False}
    versioned_save(root / "identity-reviews" / CODE / (source["sha256"] + ".json"), changed)
    with pytest.raises(ValueError, match="CONDITIONAL_VALUATION_UNPROVEN"):
        identity.checked_applicability(root, CODE, entry, source)


def test_news_only_link_preserves_company_gaps_and_invalidates_old_source(sources):
    root, client, entry, source, _, _ = sources
    review(sources)
    item = client.news_body(CODE, entry, {"entry": entry, "source": source})
    event = {"fund_code": CODE, "article_id": ARTICLE, "source_hash": source["sha256"]}
    versioned_save(
        root / "funds" / (CODE + ".json"),
        {
            "fund_code": CODE,
            "status": "SUCCEEDED",
            "catalog_complete": True,
            "items": {ARTICLE: item},
            "candidate_events": [event],
        },
    )
    material = {
        "fund_code": CODE,
        "source_hash": "a" * 64,
        "gaps": ["本次公告目录或原文尚未核验完整", "公司经营资料缺项"],
        "companies": [{"stockCode": "000001.SZ", "quote": None}],
        "reports": [{"endDate": "2026-06-30"}],
        "prediction_ready": True,
        "rating_ready": True,
    }
    versioned_save(root.parent / "materials" / (CODE + ".json"), material)
    result = news.refresh_material_news_link(CODE, root=root.parent)
    assert result["gaps"] == ["公司经营资料缺项"]
    assert result["companies"] == material["companies"] and result["reports"] == material["reports"]
    assert result["source_hash"] != material["source_hash"]
    assert result["candidate_events"] == [event]
    assert not result["prediction_ready"] and not result["rating_ready"]
    assert news.refresh_material_news_link(CODE, root=root.parent)["source_hash"] == result["source_hash"]
