"""隔离验证完整金额穿透、错误ETF拒绝及缓存不能绕过原件摘要。"""

import hashlib
import json
from copy import deepcopy
from decimal import Decimal

import pytest
from app.services import fund_rating_passive_holdings as module
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS


class Source:
    def __init__(self, text):
        self.text = text

    def pages(self, _digest):
        return [self.text]


def test_etf_columns_interleave_name_but_identity_and_full_amount_are_required():
    legal = "隔离新能源交易型开放式指数证券投资基金"
    body = ("2.1.1目标基金基本情况基金名称" + legal + "基金主代码1599992.2基金产品说明"
            "7.2期末投资目标基金明细金额单位:人民币元序号基金名称基金类型运作方式管理人公允价值"
            "隔离新能源交易型开管理人\n1 交易型开放股票型放式公司900.0090.00式指数证券投资基金7.3股票")
    target = {"fund_code": "159999", "legal_name": legal}
    assert module._fund_holding(Source(body), "fixture", target, "900")["amount_cny"] == "900"
    with pytest.raises(ValueError, match="AMOUNT_OR_NAME"):
        module._fund_holding(Source(body), "fixture", target, "800")
    with pytest.raises(ValueError, match="RELATIONSHIP"):
        module._fund_holding(Source(body), "fixture", {**target, "fund_code": "159998"}, "900")
    with pytest.raises(ValueError, match="AMOUNT_OR_NAME"):
        module._fund_holding(Source(body.replace("式指数证券投资基金7.3", "式混合基金7.3")),
                             "fixture", target, "900")


def test_nested_fund_investment_table_keeps_single_rank_and_full_amount_boundary():
    legal = '隔离五百交易型开放式指数证券投资基金'
    own = ('2.1.1目标基金基本情况基金名称'+legal+'基金主代码5999992.2基金产品说明\n'
           '7.10 本报告期投资基金情况\n'
           '7.10.1 期末按公允价值占基金资产净值比例大小排序的前十名基金投资明细\n'
           '金额单位：人民币元\n序号 基金名称 公允价值\n'
           +legal+'\n1 股票型 管理人 8,387,658,\n550.98 93.57\n7.11 本基金投资股指期货的投资政策')
    target = {'fund_code': '599999', 'legal_name': legal}
    result = module._fund_holding(Source(own), 'isolated', target, '8387658550.98')
    assert result['amount_cny'] == '8387658550.98' and result['table_rank_rows'] == ['1']
    with pytest.raises(ValueError, match='COMPLETE_TARGET_FUND_TABLE'):
        module._fund_holding(Source(own.replace('7.11', '2 其他基金 100.00\n7.11')),
                             'isolated', target, '8387658550.98')


def test_money_tail_cannot_be_borrowed_from_percentage_or_another_fund():
    legal = "隔离新能源交易型开放式指数证券投资基金"
    body = ("2.1.1目标基金基本情况基金名称"+legal+"基金主代码1599992.2基金产品说明\n"
            "7.10 期末按公允价值占基金资产净值比例大小排序的前十名基金投资明细\n"
            "金额单位:人民币元序号基金名称\n"+legal+"\n1 股票型 3,753,463,923. 90.77\n 公司 64\n7.11 投资政策")
    target = {"fund_code": "159999", "legal_name": legal}
    result = module._fund_holding(Source(body), "fixture", target, "3753463923.64")
    assert result["reported_amount_fragments"] == ["3,753,463,923.", "64"]
    with pytest.raises(ValueError, match="AMOUNT_OR_NAME"):
        module._fund_holding(Source(body), "fixture", target, "3753463923.77")
    with pytest.raises(ValueError, match="COMPLETE_TARGET_FUND_TABLE"):
        module._fund_holding(Source(body.replace("7.11", "2 其他基金 1.00\n7.11")),
                             "fixture", target, "3753463923.64")


def test_index_name_and_last_amount_group_can_wrap_inside_interleaved_columns():
    legal = "隔离柏瑞中证稀土产业交易型开放式指数证券投资基金"
    body = ("2.1.1目标基金基本情况基金名称" + legal + "基金主代码1599992.2基金产品说明\n"
            "7.10 期末按公允价值占基金资产净值比例大小排序的前十名基金投资明细\n"
            "金额单位:人民币元序号基金名称\n隔离柏瑞\n中证稀土 隔离柏瑞\n"
            "1 产业交易 指数型 交易型开 基金管理 552,793,4 94.70\n"
            "型开放式 放式 有限公司 69.37\n指数证券\n投资基金\n7.11 投资政策")
    target = {"fund_code": "159999", "legal_name": legal}
    proof = module._fund_holding(Source(body), "fixture", target, "552793469.37")
    assert proof["reported_amount_fragments"] == ["552,793,4", "69.37"]
    changed = body.replace("中证稀土 隔离柏瑞", "中证食品 隔离柏瑞")
    with pytest.raises(ValueError, match="AMOUNT_OR_NAME"):
        module._fund_holding(Source(changed), "fixture", target, "552793469.37")


def test_target_table_short_name_requires_target_original_field_and_exact_full_fund_amount():
    legal = '隔离消费电子交易型开放式指数证券投资基金'
    abbreviated = '隔离消费电子ETF'
    own = ('2.1.1目标基金基本情况基金名称' + legal + '基金主代码1599992.2基金产品说明\n'
           '7.12.2 报告期末按公允价值占基金资产净值比例大小排序的所有基金投资明细\n'
           '序号 基金代码 基金名称 公允价值（元）\n1 159999 '+abbreviated+' 900.00 90.00\n7.13 投资组合报告附注')

    class Reports:
        def pages(self, key):
            return [own if key == 'own' else legal + '基金简称' + abbreviated + '场内简称隔离ETF基金主代码159999']

    target = {'fund_code': '159999', 'legal_name': legal, 'source_hash': 'target'}
    proof = module._fund_holding(Reports(), 'own', target, '900')
    assert proof['table_abbreviation_verified_from_target_report'] == abbreviated
    with pytest.raises(ValueError, match='AMOUNT_OR_NAME'):
        module._fund_holding(Reports(), 'own', target, '800')


@pytest.fixture
def isolated_inputs(tmp_path, monkeypatch):
    sources = {}

    def archive(name, payload):
        path = tmp_path/"sources"/name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        sources[sha] = {"file": "sources/"+name, "sha256": sha, "published_on": "2026-08-31"}
        return sha

    report = archive("report.json", {"data": {
        "notice_content": "999999隔离指数证券投资基金跟踪隔离指数。本基金采用完全复制方法。"
    }})
    metadata = archive("index.json", {"code": 0, "data": {"fields": ["ts_code", "name", "publisher"],
                         "items": [["fixture.CSI", "隔离", "中证指数有限公司"]]}})
    weights = archive("weight.json", {"code": 0, "data": {
        "fields": ["index_code", "con_code", "trade_date", "weight"],
        "items": [["fixture.CSI", "a", "20260630", "25"], ["fixture.CSI", "b", "20260630", "75"]]}})
    monkeypatch.setattr(module, "portfolio", lambda *a, **k: {
        "balance": {"fund_amount": "0"}, "stock_total": "100", "nonsecurity_substitution_increment": "0",
        "positions": [{"stock_code": "a", "market_value_cny": "70"},
                      {"stock_code": "b", "market_value_cny": "30"}],
    })
    monkeypatch.setattr(module, "_mappings", lambda *a: (
        {"a": "issuer_a", "b": "issuer_b"}, {"a": "C35", "b": "C39"}
    ))
    return {"version": module.VERSION, "basis": PASSIVE_BASIS, "fund_code": "999999",
            "legal_name": "隔离指数证券投资基金", "as_of_date": "2026-09-30", "holding_date": "2026-06-30",
            "sources": sources, "primary_ref": report,
            "index": {"code": "fixture.CSI", "name": "隔离", "metadata": metadata, "weights": weights}}, tmp_path


def test_independent_holdings_uses_original_absolute_r5_formula_without_fabricated_peers(isolated_inputs):
    inputs, root = isolated_inputs
    result = module.calculate(inputs, root=root)
    assert Decimal(result["score"]) == 55
    assert set(result["subscores"]) == {"issuer_replication", "industry_replication"}
    assert "peers" not in inputs and "score" not in inputs
    assert result["stock_count"] == 2
    assert "NO_LIQUIDITY_OR_FULL_SCORE" in result["scope"]


def test_existing_holdings_cache_still_rejects_corrupt_original(isolated_inputs):
    inputs, root = isolated_inputs
    module.calculate(inputs, root=root)
    (root/"sources"/"report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_HASH"):
        module.calculate(inputs, root=root)


def test_historical_weight_date_and_unproved_index_alias_are_rejected(isolated_inputs):
    inputs, root = isolated_inputs
    changed = deepcopy(inputs)
    changed["holding_date"] = "2026-07-01"
    with pytest.raises(ValueError, match="WEIGHT_IDENTITY_DATE"):
        module.calculate(changed, root=root)
    changed = deepcopy(inputs)
    changed["index"]["contract_name"] = "隔离指数证券投资基金跟踪隔离"
    with pytest.raises(ValueError, match="ALIAS_PROOF_REQUIRED"):
        module.calculate(changed, root=root)


def test_derived_summary_cannot_replace_financial_report_or_weight_original(isolated_inputs):
    inputs, root = isolated_inputs
    for digest in [inputs["primary_ref"], inputs["index"]["weights"]]:
        changed = deepcopy(inputs)
        changed["sources"][digest]["archive_kind"] = "DERIVED"
        with pytest.raises(ValueError, match="FINANCIAL_ORIGINAL_REQUIRED"):
            module.calculate(changed, root=root)
