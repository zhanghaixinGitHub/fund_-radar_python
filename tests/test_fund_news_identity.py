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


def _specific_notice(sources, title, body):
    """独立构造本份额目录和原响应，后续测试仍经完整身份及摘要核验。"""
    _, client, _, _, _, _ = sources
    entry = {"id": ARTICLE, "title": title, "published": "2026-10-15"}
    source = client._archive(
        json.dumps({"success": 1, "data": {"art_code": ARTICLE, "notice_title": title,
                    "notice_content": body}}, ensure_ascii=False).encode(),
        ".json", BODY_URL, {"art_code": ARTICLE},
    )
    catalog = client._archive(
        json.dumps({"ErrCode": 0, "Data": [{"ID": ARTICLE, "FUNDCODE": CODE,
                    "TITLE": title, "PUBLISHDATEDesc": "2026-10-15"}]}).encode(),
        ".json", CATALOG_URL, {"fundcode": CODE},
    )
    return {"entry": entry, "source": source, "catalog_source": catalog, "quotes": [body]}


@pytest.mark.parametrize("economic_change", [False, True])
def test_privacy_notice_is_only_exact_manager_service_background(sources, economic_change):
    title = "某某基金管理有限公司隐私政策更新提示"
    body = ("某某基金管理有限公司更新客户隐私政策、第三方信息共享清单、个人信息收集清单，"
            "以上隐私政策自发布之日起生效。客户应阅读条款，有疑问按声明方式联系。")
    if economic_change:
        body += "本次同时调整基金经理。"
    args = _specific_notice(sources, title, body)
    if economic_change:
        with pytest.raises(ValueError, match="MANAGER_SCOPE_UNPROVEN"):
            review(sources, **args)
    else:
        assert review(sources, **args)["does_not_determine_direction"] is True


@pytest.mark.parametrize(
    "problem", [None, "ETF", "wrong_share", "unrelated_title", "missing_recovery", "tampered_flag"]
)
def test_report_bound_valuation_background_does_not_assert_portfolio(sources, problem):
    root, _, _, _, _, instant = sources
    product = "某某成长混合型证券投资基金"
    if problem == "ETF":
        product = "某某行业交易型开放式指数证券投资基金"
    title = "某某基金管理有限公司关于旗下基金调整长期停牌股票估值方法的提示性公告"
    body = ("某某基金管理有限公司经与基金托管人协商一致，对本公司旗下基金所持有的"
            "中国稀土（股票代码：000831）按指数收益法估值。"
            "待该股票的交易体现活跃市场交易特征后，恢复采用交易所当日收盘价。")
    if problem == "unrelated_title":
        title = "某某基金管理有限公司关于旗下部分基金调整费率的公告"
    if problem == "missing_recovery":
        body = body.split("待该股票")[0]
    args = _specific_notice(sources, title, body)
    own = (f"某某基金管理有限公司\n基金全称 {product}\n下属分级基金的交易代码 010736 "
           f"{CODE if problem != 'wrong_share' else '999999'}")
    document = {"id": "own-report", "kind": "REPORT", "codes": [CODE], "body": own,
                "body_hash": digest(own), "published_date": "2026-10-15", "available_at": instant.isoformat()}
    material = {"fund_code": CODE, "documents": [document]}
    path = root / "own-report-material.json"
    versioned_save(path, material)
    args.update(scope="MANAGER_CONDITIONAL_VALUATION", product_name=product,
                identity_proof={"materials_path": str(path), "materials_hash": digest(material),
                                "document_id": document["id"], "quote": own})
    if problem not in {None, "tampered_flag"}:
        with pytest.raises(ValueError, match="CONDITIONAL_VALUATION_UNPROVEN"):
            review(sources, **args)
    else:
        receipt = review(sources, **args)
        assert receipt["does_not_confirm_current_portfolio_application"] is True
        if problem == "tampered_flag":
            receipt["does_not_confirm_current_portfolio_application"] = False
            versioned_save(root / "identity-reviews" / CODE / (args["source"]["sha256"] + ".json"), receipt)
            with pytest.raises(ValueError, match="CONDITIONAL_VALUATION_UNPROVEN"):
                identity.checked_applicability(root, CODE, args["entry"], args["source"])


@pytest.mark.parametrize("partial_fund", [False, True])
def test_manager_system_shutdown_is_service_but_partial_fund_scope_is_rejected(sources, partial_fund):
    root, client, _, _, _, _ = sources
    title = "某某基金管理有限公司关于系统停机维护的通知"
    if partial_fund:
        title = "某某基金管理有限公司关于旗下部分基金系统停机维护的通知"
    body = (
        "某某基金管理有限公司为提升系统性能与稳定性，拟定于周六13:00至18:00进行系统停机维护，"
        "届时将暂停网上交易系统（官网、APP、微信）对外服务，敬请提前安排相关业务。"
    )
    entry = {"id": ARTICLE, "title": title, "published": "2026-10-15"}
    source = client._archive(
        json.dumps({"success": 1, "data": {"art_code": ARTICLE, "notice_title": title,
                    "notice_content": body, "security": [{"stock": "000001"}]}}, ensure_ascii=False).encode(),
        ".json", BODY_URL, {"art_code": ARTICLE},
    )
    catalog = client._archive(
        json.dumps({"ErrCode": 0, "Data": [{"ID": ARTICLE, "FUNDCODE": CODE,
                    "TITLE": title, "PUBLISHDATEDesc": "2026-10-15"}]}, ensure_ascii=False).encode(),
        ".json", CATALOG_URL, {"fundcode": CODE},
    )
    kwargs = {"entry": entry, "source": source, "catalog_source": catalog, "quotes": [body]}
    if partial_fund:
        with pytest.raises(ValueError, match="MANAGER_SCOPE_UNPROVEN"):
            review(sources, **kwargs)
    else:
        assert review(sources, **kwargs)["does_not_determine_direction"] is True


@pytest.mark.parametrize("economic_change", [False, True])
def test_exact_manager_impersonation_warning_is_service_background_only(sources, economic_change):
    root, client, _, _, _, _ = sources
    title = "关于警惕不法分子假冒某某基金管理有限公司进行非法证券活动的重要提示"
    body = (
        "某某基金管理有限公司发现不法分子假冒公司开展非法证券活动。官方信息请以官网为准，"
        "请通过官方渠道咨询，正规金融机构不会承诺保本保收益，请投资者谨防诈骗，注意保护个人信息。"
    )
    if economic_change:
        body += "本基金经理发生变更，具体投资策略随之调整。"
    entry = {"id": ARTICLE, "title": title, "published": "2026-10-15"}
    source = client._archive(
        json.dumps(
            {
                "success": 1,
                "data": {
                    "art_code": ARTICLE,
                    "notice_title": title,
                    "notice_content": body,
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
            {
                "ErrCode": 0,
                "Data": [{"ID": ARTICLE, "FUNDCODE": CODE, "TITLE": title, "PUBLISHDATEDesc": "2026-10-15"}],
            },
            ensure_ascii=False,
        ).encode(),
        ".json",
        CATALOG_URL,
        {"fundcode": CODE},
    )
    kwargs = {"entry": entry, "source": source, "catalog_source": catalog, "quotes": [body[:160]]}
    if economic_change:
        with pytest.raises(ValueError, match="MANAGER_SCOPE_UNPROVEN"):
            review(sources, **kwargs)
    else:
        result = review(sources, **kwargs)
        assert result["scope"] == "MANAGER_SERVICE" and result["does_not_determine_direction"] is True


def test_reviewed_original_reused_when_catalog_listing_day_differs_from_body_day(sources, monkeypatch):
    root, client, entry, source, catalog, _ = sources
    raw = json.loads(market_fund_reports.read_source(root, source))
    raw["data"]["notice_date"] = "2026-10-14 00:00:00"
    source = client._archive(json.dumps(raw, ensure_ascii=False).encode(), ".json", BODY_URL, {"art_code": ARTICLE})
    versioned_save(root / "body-receipts" / CODE / (ARTICLE + ".json"), source)
    review(sources, source=source)

    def unexpected_download(*args):
        pytest.fail("已绑定本目录和原件的逐篇凭据不应再次下载相同正文")

    monkeypatch.setattr(client, "_get", unexpected_download)
    item = client.news_body(CODE, entry)
    assert item["source"]["sha256"] == source["sha256"] and item["received_at"] == source["received_at"]


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
        ("某某基金管理有限公司关于系统升级暂停网上交易系统服务的通知", "MANAGER_SERVICE"),
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


@pytest.mark.parametrize("missing", [None, "最近12个月内董事变更超过百分之五十", "公司董事会成员", "向监管机构备案"])
def test_manager_board_change_requires_exact_title_and_governance_original(sources, missing):
    """董事变动须有真实治理正文；通过只补充管理人背景，不证明基金经理变动。"""
    root, client, _, _, _, _ = sources
    title = "某某基金管理有限公司关于公司董事变更的公告"
    body = (
        "某某基金管理有限公司公告最近12个月内董事变更超过百分之五十，"
        "公司董事会成员已经股东会决议确认，上述变动按规定向监管机构备案。"
        "本公告仅说明管理人董事会人员组成，不代表任何基金经理发生变动。"
    )
    if missing:
        body = body.replace(missing, "相关事项")
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
    args = dict(entry=entry, source=source, catalog_source=catalog, scope="MANAGER_GOVERNANCE", quotes=[body])
    if missing:
        with pytest.raises(ValueError, match="MANAGER_SCOPE_UNPROVEN"):
            review(sources, **args)
    else:
        result = review(sources, **args)
        assert result["does_not_determine_direction"] is True
        assert client.news_body(CODE, entry, {"entry": entry, "source": source})["applicability_review"] == result
        with pytest.raises(ValueError, match="CATALOG"):
            review(sources, **{**args, "code": "005284"})


@pytest.mark.parametrize("problem", [None, "other_master", "other_share", "other_row", "missing_header"])
def test_auditor_table_binds_master_code_and_share_to_same_report(sources, problem):
    """跨行表格只按同一报告的产品、主代码及份额交叉核验，不串行借码。"""
    root, client, _, _, _, _ = sources
    product = "某某成长精选灵活配置混合型证券投资基金"
    parts = ["某某成长精选灵活配置混合", "型证券投资基金"]
    short, master = "某某成长精选", "000001"
    row = f"{parts[0]}  {short}  {master}\n\n{parts[1]}"
    title = "某某基金管理有限公司关于旗下基金改聘会计师事务所公告"
    body = (
        "某某基金管理有限公司按董事会决议改聘会计师事务所。\n"
        f"基金名称 基金简称 基金主代码\n{row}\n改聘日期及托管通知均以公告正文为准。"
    )
    master_quote = "基金主代码 000001"
    share_quote = f"下属分级基金的交 000001 {CODE}\n易代码"
    proof_quote = f"基金名称 {product}\n基金简称 {short}\n{master_quote}\n{share_quote}"
    document = {
        "id": "original-report",
        "kind": "REPORT",
        "codes": [CODE],
        "body": proof_quote,
        "body_hash": digest(proof_quote),
        "available_at": "2026-10-15T09:00:00+08:00",
        "published_date": "2026-10-14",
    }
    material = {"fund_code": CODE, "documents": [document]}
    path = root / "product-proof.json"
    versioned_save(path, material)
    table = {
        "name_parts": parts,
        "short_name": short,
        "master_code": master,
        "row_quote": row,
        "master_quote": master_quote,
        "share_quote": share_quote,
    }
    if problem == "other_master":
        table["master_code"] = "000002"
    elif problem == "other_share":
        table["share_quote"] = share_quote.replace(CODE, "005284")
    elif problem == "other_row":
        table["row_quote"] = row.replace(master, "000002")
    elif problem == "missing_header":
        body = body.replace("基金名称 基金简称 基金主代码", "基金名单")
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
    args = dict(
        entry=entry,
        source=source,
        catalog_source=catalog,
        scope="EXPLICIT_PRODUCT_NAME",
        quotes=[body],
        product_name=product,
        identity_proof={
            "materials_path": str(path),
            "materials_hash": digest(material),
            "document_id": document["id"],
            "quote": proof_quote,
            "product_table": table,
        },
    )
    if problem:
        with pytest.raises(ValueError, match="EXPLICIT_PRODUCT_REQUIRED"):
            review(sources, **args)
    else:
        receipt = review(sources, **args)
        assert receipt["scope"] == "EXPLICIT_PRODUCT_NAME"
        assert client.news_body(CODE, entry, {"entry": entry, "source": source})["applicability_review"] == receipt


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


@pytest.mark.parametrize("missing", [None, "不开通基金转换业务的基金", "处于正常开放期的其他基金之间开通基金转换业务"])
def test_conditional_conversion_needs_full_scope_and_never_confirms_current_service(sources, missing):
    """目录中的条件式转换公告只作背景，不能靠管理人名称推定具体份额已开通。"""
    root, client, _, _, _, instant = sources
    title = "某某基金管理有限公司关于旗下部分基金开通基金转换业务的公告"
    body = (
        "某某基金管理有限公司自明日起办理转换业务。"
        "（1）不开通基金转换业务的基金列于本表。"
        "（2）不开通同一基金不同类别基金份额之间转换业务的基金另列。"
        "（3）除上述（1）、（2）项所列基金及ETF外，本公司处于正常开放期的其他基金之间开通基金转换业务。"
    )
    if missing:
        body = body.replace(missing, "另行约定")
    entry = {"id": ARTICLE, "title": title, "published": "2026-10-15"}
    source = client._archive(
        json.dumps(
            {"success": 1, "data": {"art_code": ARTICLE, "notice_title": title, "notice_content": body}}
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
    product = "某某中证农牧渔交易型开放式指数证券投资基金联接基金"
    proof_body = "某某基金管理有限公司 基金名称" + product + " 份额交易代码" + CODE
    document = {
        "id": "own-report",
        "kind": "REPORT",
        "codes": [CODE],
        "body": proof_body,
        "body_hash": digest(proof_body),
        "available_at": instant.isoformat(),
        "published_date": "2026-10-15",
    }
    material = {"fund_code": CODE, "documents": [document]}
    path = root / "conversion-proof.json"
    versioned_save(path, material)
    proof = {
        "materials_path": str(path),
        "materials_hash": digest(material),
        "document_id": document["id"],
        "quote": proof_body,
    }
    args = dict(
        entry=entry,
        source=source,
        catalog_source=catalog,
        quotes=[body],
        scope="MANAGER_CONDITIONAL_CONVERSION",
        identity_proof=proof,
        product_name=product,
    )
    if missing:
        with pytest.raises(ValueError, match="CONDITIONAL_CONVERSION_UNPROVEN"):
            review(sources, **args)
    else:
        receipt = review(sources, **args)
        assert receipt["does_not_confirm_current_portfolio_application"] is True
        assert receipt["does_not_determine_direction"] is True
        with pytest.raises(ValueError, match="CONDITIONAL_CONVERSION_UNPROVEN"):
            review(sources, **{**args, "identity_proof": None})
        changed = {**receipt, "does_not_confirm_current_portfolio_application": False}
        versioned_save(root / "identity-reviews" / CODE / (source["sha256"] + ".json"), changed)
        with pytest.raises(ValueError, match="CONDITIONAL_CONVERSION_UNPROVEN"):
            identity.checked_applicability(root, CODE, entry, source)


def test_unchanged_notice_reuses_older_verified_original_without_backdating(sources, monkeypatch):
    """日期跨日不使同版本原件失效，新的覆盖仍由目录核验，身份回执照常检查。"""
    root, client, entry, source, _, instant = sources
    review(sources)
    later = instant + timedelta(days=2)
    monkeypatch.setattr(news, "now", lambda: later)
    monkeypatch.setattr(identity, "now", lambda: later)
    monkeypatch.setattr(client, "_get", lambda *_: pytest.fail("目录版本未变不重复下载原件"))
    actual = client.news_body(CODE, entry, {"entry": entry, "source": source})
    assert actual["received_at"] == source["received_at"] and actual["source"] == source
    assert actual["applicability_review"] is not None


def test_old_failed_identity_original_is_reused_but_not_accepted_without_review(sources, monkeypatch):
    root, client, entry, source, _, _ = sources
    original = json.loads((root / source["file"]).read_bytes())
    original["data"]["notice_date"] = entry["published"]
    source = client._archive(json.dumps(original).encode(), ".json", BODY_URL, {"art_code": ARTICLE})
    versioned_save(root / "body-receipts" / CODE / (ARTICLE + ".json"), source)
    monkeypatch.setattr(client, "_get", lambda *_: pytest.fail("已留存原件不应因身份未通过而再下载"))
    with pytest.raises(ValueError, match="BODY_UNVERIFIED"):
        client.news_body(CODE, entry)


def test_revised_catalog_uses_matching_new_original_receipt_before_retrying_network(sources, monkeypatch):
    root, client, entry, source, _, _ = sources
    raw = json.loads((root / source["file"]).read_bytes())
    raw["data"]["notice_date"] = entry["published"]
    current = client._archive(json.dumps(raw).encode(), ".json", BODY_URL, {"art_code": ARTICLE})
    review(sources, source=current)
    versioned_save(root / "body-receipts" / CODE / (ARTICLE + ".json"), current)
    old_entry = {**entry, "published": "2026-10-14"}
    monkeypatch.setattr(client, "_get", lambda *_: pytest.fail("已取得对应修订目录的真实原文，不重复请求"))
    item = client.news_body(CODE, entry, {"entry": old_entry, "source": source})
    assert item["source"] == current and item["received_at"] == current["received_at"]


def test_trade_state_rows_keep_each_share_code_in_its_original_columns():
    """实际三种列布局及CRLF均可核验，缺尾数或借邻行仍拒绝。"""
    from pathlib import Path

    rows = json.loads(
        (Path(__file__).parent / "fixtures/notice-share-layout-20261029/018590-rows.json").read_text("utf-8")
    )
    assert len(rows) == 11
    for row in rows:
        body = "交易状态表\n基金简称 基金代码 开放申购 开放赎回\n" + row["row"]
        args = ("018590", row["product"], row["proof"], row["document"], body, row["title"], "银华基金管理股份有限公司")
        assert identity._trade_state_table_matches(*args), row["id"]
        assert not identity._trade_state_table_matches("014047", *args[1:])
        changed = json.loads(json.dumps(row["proof"]))
        changed["trade_state_table"]["row_quote"] = row["row"].replace("01859", "01404")
        changed_body = "交易状态表\n基金简称 开放申购 开放赎回\n" + changed["trade_state_table"]["row_quote"]
        assert not identity._trade_state_table_matches(
            "018590", row["product"], changed, row["document"], changed_body, row["title"], "银华基金管理股份有限公司"
        )


def test_allocation_product_name_cannot_borrow_a_neighbouring_row():
    product = "南方中证500交易型开放式指数证券投资基金联接基金(LOF)"
    row = " 南方中证 500 交易型开放式指数证券    171      17,100    优先配售\n      投资基金联接基金（LOF）"
    proof = {"allocation_table": {"row_quote": row}}
    body = "基金名称 获配数量 获配金额 投资类型\n" + row
    title = "南方基金管理股份有限公司关于旗下基金投资关联方承销可转换公司债券的关联交易公告"
    assert identity._allocation_product_table_matches(
        product, proof, {"kind": "REPORT"}, body, title, "南方基金管理股份有限公司"
    )
    assert not identity._allocation_product_table_matches(
        product.replace("500", "A500"), proof, {"kind": "REPORT"}, body, title, "南方基金管理股份有限公司"
    )
    assert not identity._allocation_product_table_matches(
        product, proof, {"kind": "REPORT"}, body + row, title, "南方基金管理股份有限公司"
    )


def test_benchmark_product_name_uses_only_same_row_display_columns():
    manager = "前海开源基金管理有限公司"
    title = manager + "关于旗下部分基金调整业绩比较基准并修订基金合同等法律文件的公告"
    product = "前海开源黄金交易型开放式证券投资基金联接基金"
    row = "    前海开源黄金交易型开放  黄金价格基准权重调整\n32\n\n    式证券投资基金联接基金  现金类权重调整"
    body = "序号 基金全称 原业绩比较基准 调整后新业绩比较基准\n" + row
    proof = {"benchmark_table": {"row_quote": row, "column_start": 4, "column_end": 26, "ordinal": "32"}}
    args = (proof, {"kind": "REPORT"}, body, title, manager)
    assert identity._benchmark_product_table_matches(product, *args)
    assert not identity._benchmark_product_table_matches(product.replace("联接", "发起式联接"), *args)
    assert not identity._benchmark_product_table_matches(product, proof, {"kind": "REPORT"}, body + row, title, manager)
    changed = json.loads(json.dumps(proof))
    changed["benchmark_table"]["row_quote"] = row + "\n33    另一基金名称"
    assert not identity._benchmark_product_table_matches(
        product, changed, {"kind": "REPORT"}, body + "\n33    另一基金名称", title, manager
    )
    assert not identity._benchmark_product_table_matches(product, proof, {"kind": "FUND_PROFILE"}, body, title, manager)
    assert not identity._benchmark_product_table_matches(
        product, proof, {"kind": "REPORT"}, body, "普通业务公告", manager
    )


@pytest.mark.parametrize("product,accepted", [("南方原油证券投资基金", True), ("南方原油基金", False)])
def test_short_legal_product_still_requires_independent_share_original(sources, product, accepted):
    root, client, _, _, _, instant = sources
    title = "某某基金管理有限公司关于旗下部分基金修订基金合同的公告"
    body = (f"某某基金管理有限公司公告：{product}修订基金合同，本产品以实际公告条款为准。"
            "投资者应阅读与本份额对应的完整合同，本公告的产品适用性不能仅由转载证券列表推断。")
    entry = {"id": ARTICLE, "title": title, "published": "2026-10-15"}
    raw = {"success": 1, "data": {"art_code": ARTICLE, "notice_title": title, "notice_content": body}}
    source = client._archive(json.dumps(raw, ensure_ascii=False).encode(), ".json", BODY_URL, {"art_code": ARTICLE})
    rows = {"ErrCode": 0, "Data": [{"ID": ARTICLE, "FUNDCODE": CODE, "TITLE": title,
                                   "PUBLISHDATEDesc": "2026-10-15"}]}
    catalog = client._archive(json.dumps(rows).encode(), ".json", CATALOG_URL, {"fundcode": CODE})
    proof_body = f"{product}独立季度报告，本份额基金代码为{CODE}，主代码及其他份额代码另行披露。"
    document = {"id": "own-report", "kind": "REPORT", "codes": [CODE], "body": proof_body,
                "body_hash": digest(proof_body), "available_at": instant.isoformat(), "published_date": "2026-10-15"}
    material = {"fund_code": CODE, "documents": [document]}
    path = root / "own-report.json"
    versioned_save(path, material)
    args = dict(entry=entry, source=source, catalog_source=catalog, scope="EXPLICIT_PRODUCT_NAME",
                quotes=[body], product_name=product, identity_proof={"materials_path": str(path),
                "materials_hash": digest(material), "document_id": document["id"], "quote": proof_body})
    if accepted:
        assert review(sources, **args)["does_not_determine_direction"]
        with pytest.raises(ValueError):
            review(sources, **{**args, "identity_proof": None})
    else:
        with pytest.raises(ValueError, match="EXPLICIT_PRODUCT_REQUIRED"):
            review(sources, **args)
