"""本轮实际遇到的有限公告格式兼容；身份核验不等于经济影响或交易状态确认。"""

from copy import deepcopy

from app.services import fund_market_news as news
from app.services import fund_news_identity as identity
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from tests.test_fund_news_identity import ARTICLE, CODE, review, sources  # noqa: F401


def test_current_news_link_does_not_revalidate_expired_archived_items(sources):  # noqa: F811
    root, client, entry, source, _, _ = sources
    review(sources)
    item = client.news_body(CODE, entry, {"entry": entry, "source": source})
    event = {"fund_code": CODE, "article_id": "AN202609011830508374", "source_hash": "a" * 64}
    old = {"entry": {"published": "2026-09-01"}, "source": {"invalid_old_source_must_not_be_opened": True}}
    versioned_save(
        root / "funds" / (CODE + ".json"),
        {
            "fund_code": CODE,
            "status": "SUCCEEDED",
            "catalog_complete": True,
            "coverage": {"start": "2026-10-01", "end": "2026-10-21"},
            "items": {ARTICLE: item, event["article_id"]: old},
            "candidate_events": [event],
        },
    )
    versioned_save(
        root.parent / "materials" / (CODE + ".json"),
        {
            "fund_code": CODE,
            "source_hash": "b" * 64,
            "gaps": [],
        },
    )
    result = news.refresh_material_news_link(CODE, root=root.parent)
    assert set(result["news_link_manifest"]) == {ARTICLE}
    assert result["candidate_events"] == [event]
    assert event["article_id"] in read(root / "funds" / (CODE + ".json"))["items"]


def test_conditional_valuation_uses_independent_share_proof_and_restoration_terms():
    product = "南方测试指数证券投资基金联接基金"
    manager = "南方基金管理股份有限公司"
    proof = {"quote": product + " 008164 " + manager}
    doc = {"kind": "REPORT", "body": proof["quote"]}
    body = (
        manager + "自2026年11月5日起，对旗下基金（ETF基金除外）持有的002129采用指数收益法，复牌后恢复按市场价格估值。"
    )
    title = "关于南方基金管理股份有限公司旗下基金调整停牌股票估值方法的提示性公告"
    assert identity._november_valuation_background("008164", product, proof, doc, body, title, manager)
    assert not identity._november_valuation_background("008163", product, proof, doc, body, title, manager)
    assert not identity._november_valuation_background(
        "008164", product, proof, doc, body.replace("市场价格", "其他方法"), title, manager
    )
    assert not identity._november_valuation_background("008164", product, proof, doc, body, title, "其他管理人")


def test_missing_c_code_in_reprint_only_allows_product_identity_with_own_share_binding():
    product = "银华体育文化灵活配置混合型证券投资基金"
    short = "银华体育文化灵活配置混合"
    master_quote = "基金主代码 003397"
    share_quote = "下属分级基金的交易代码 003397 018590"
    own = f"{product} 基金简称 {short} {master_quote} {share_quote}"
    row = "              A 类：\n银华体育文    00339\n化灵活配置    7\n混合          C 类：\n"
    proof = {
        "quote": own,
        "trade_state_table": {
            "short_name": short,
            "master_code": "003397",
            "master_quote": master_quote,
            "share_quote": share_quote,
            "row_quote": row,
        },
    }
    doc = {"kind": "REPORT", "body": own}
    body = "交易状态表 基金简称 开放申购 开放赎回\n" + row
    title = "银华基金管理股份有限公司旗下基金交易状态表"
    args = ("018590", product, proof, doc, body, title, "银华基金管理股份有限公司")
    assert not identity._trade_state_table_matches(*args)
    proof["trade_state_table"]["class_code_missing_in_text"] = True
    assert identity._trade_state_table_matches(*args)
    changed = deepcopy(proof)
    changed["trade_state_table"]["row_quote"] = row + "              999999\n"
    assert not identity._trade_state_table_matches(
        "018590", product, changed, doc, body + "              999999\n", title, "银华基金管理股份有限公司"
    )
