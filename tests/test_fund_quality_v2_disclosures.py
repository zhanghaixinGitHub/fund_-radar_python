"""原文解析的反例与边界；测试数据不是校准产品或来源证明。"""

import json
from datetime import date
from decimal import Decimal

import pytest
from app.services.fund_quality_v2_disclosures import holder_pressure, manager_history, report_period


@pytest.mark.parametrize("mutation", [None, "missing_catalog", "bad_hash", "business_failure", "wrong_code",
                                     "wrong_date", "wrong_attachment", "http", "wrong_path", "fake_host"])
def test_cjhx_periodic_pdf_requires_matching_actual_issuer_catalog(tmp_path, monkeypatch, mutation):
    """官网白名单不替代原响应中的公告归属、附件和公开日；工程假样本不作校准。"""
    import hashlib
    import json
    from datetime import date

    from app.services import fund_quality_v2_disclosures as module

    folder = tmp_path / "sources"
    folder.mkdir()
    pdf = folder / "report.pdf"
    pdf.write_bytes(b"test-only-pdf-reader-substitute")
    title = "测试股票型证券投资基金2025年年度报告"
    url = "https://www.cjhxfund.com/plat_files/uploadProdAd/importFile/20260330/123456/987654.pdf"
    entry = {"main_product_code": "123456", "product_code": "123457", "pub_date": "2026-03-30",
             "title": title, "attachment_url": url.split(".com", 1)[1]}
    payload = {"error_no": "0", "results": [{"data": [entry]}]}
    if mutation == "business_failure":
        payload["error_no"] = "-1"
    elif mutation == "wrong_code":
        entry["main_product_code"] = "654321"
    elif mutation == "wrong_date":
        entry["pub_date"] = "2026-03-31"
    elif mutation == "wrong_attachment":
        entry["attachment_url"] += "-another.pdf"
    raw = json.dumps(payload).encode()
    catalog = folder / "catalog.json"
    catalog.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    ref = {"label": "historical-annual", "file": "sources/report.pdf", "url": url, "sha256": "a" * 64,
           "published_on": "2026-03-30", "report_end": "2025-12-31", "title": title,
           "issuer_catalog_sha256": sha}
    catalog_ref = {"label": "issuer-periodic-catalog", "file": "sources/catalog.json", "sha256": sha,
                   "url": "https://www.cjhxfund.com/servlet/json", "params": {"funcNo": "742006"}}
    if mutation == "bad_hash":
        catalog.write_bytes(raw + b" ")
    elif mutation == "http":
        ref["url"] = ref["url"].replace("https:", "http:")
    elif mutation == "wrong_path":
        ref["url"] = "https://www.cjhxfund.com/unverified/report.pdf"
    elif mutation == "fake_host":
        ref["url"] = ref["url"].replace("www.cjhxfund.com", "www.cjhxfund.com.example.org")
    source = (title + "\n123456\n本报告期自2025年1月1日至2025年12月31日\n报告送出日期：2026年3月30日")
    monkeypatch.setattr(module, "public_document", lambda *args: ([source], None))
    refs = [ref] if mutation == "missing_catalog" else [ref, catalog_ref]
    result = module.source_documents("123456", refs, tmp_path, date(2026, 4, 30))["report"]
    if mutation is None:
        assert "error" not in result and result["economic_period"] == ["2025-01-01", "2025-12-31"]
        assert "www.cjhxfund.com" not in module.HOSTS
    else:
        assert "error" in result
HEADING = "11.1 报告期内单一投资者持有基金份额比例达到或超过20%的情况\n"


@pytest.mark.parametrize(
    "statement",
    [
        "无。",
        "本基金本报告期无单一投资者持有基金份额达到或超过20%的情况。",
        "报告期内单一投资者持有基金份额比例不存在达到或超过20%的情况。",
        "本报告期内，本基金未出现单一投资者持有基金份额比例达到或超过本基金总份额20%的情况。",
        "本基金本报告期内不存在持有基金份额比例达到或者超过20%的单一投资者的情况。",
        "本基金本报告期内无需要披露的单一投资者持有基金份额比例达到或超过20%的情况。",
    ],
)
def test_explicit_period_absence_is_a_real_zero(statement):
    assert holder_pressure(HEADING + statement)[0] == 0


def test_other_negative_statement_is_not_absence_of_large_holder():
    with pytest.raises(ValueError):
        holder_pressure(HEADING + "本报告期发生单一投资者持有基金份额达到或超过20%的情况，未出现大额赎回。")


def test_unnumbered_holder_title_requires_parent_and_direct_period_statement():
    body = (
        "影响投资者决策的其他重要信息\n报告期内单一投资者持有基金份额比例达到或超过 20%的情况\n"
        "注：本基金本报告期无单一投资者持有基金份额达到或超过20%的情况。\n"
        "影响投资者决策的其他重要信息\n其他股东变更资料。"
    )
    value, excerpt, rows = holder_pressure(body)
    assert value == 0 and rows == [] and "其他股东变更" not in excerpt
    for invalid in [body.replace("影响投资者决策的其他重要信息\n", "", 1),
                    body.replace("注：本基金本报告期无单一投资者持有基金份额达到或超过20%的情况。",
                                 "本报告期未出现大额赎回，但存在单一投资者持有超过20%份额。")]:
        with pytest.raises(ValueError):
            holder_pressure(invalid)


def test_zero_redemption_must_be_uniquely_reconciled_and_end_ratio_is_not_peak():
    text = (
        HEADING
        + "期初 申购 赎回 持有份额 份额占比（%）\n机构 1 20260101-20260630\n100.00 20.00 - 120.00 10.00\n产品特有风险"
    )
    value, _, rows = holder_pressure(text)
    assert value == 1 and rows[0]["redemption"] == "0.00"
    assert rows[0]["end_percent"] == "10.00"
    with pytest.raises(ValueError, match="RECONCILED"):
        holder_pressure(text.replace("120.00", "200.00"))


def test_net_redemption_is_redemption_above_subscription():
    text = (
        HEADING + "期初 申购 赎回 持有份额 份额占比（%）\n机构 1 20260101~20260630\n"
        "100.00 20.00 50.00 70.00 25.00\n产品特有风险"
    )
    assert holder_pressure(text)[0] == 2


@pytest.mark.parametrize(("buy", "sell", "expected"), [("100.00", "100.00", 1), ("100.00", "120.00", 2)])
def test_gross_flows_uniquely_define_net_change_without_filling_missing_balances(buy, sell, expected):
    text = HEADING + f"期初 申购 赎回\n机构1 20260101-20260630\n- {buy} {sell} - -\n产品特有风险"
    value, _, rows = holder_pressure(text)
    assert value == expected and rows[0]["begin"] is None and rows[0]["end"] is None
    assert rows[0]["reconciliation"] == "NET_CHANGE_FROM_GROSS_FLOWS_BALANCES_NOT_DISCLOSED"


def test_chinese_date_and_split_iso_date_still_require_flow_reconciliation():
    """真实版式的日期规范化不能吞掉份额数值或改变赎回判断。"""
    for interval in ("2026 年 1 月 1 日至2026 年 1 月 26 日", "2026-0\n1-01 至2026-0\n1-26"):
        value, _, rows = holder_pressure(HEADING + "期初 申购 赎回\n" + interval + "\n100.00 - 20.00 80.00 18.00")
        assert value == 2 and rows[0]["from"] == "2026-01-01" and rows[0]["to"] == "2026-01-26"


@pytest.mark.parametrize("prefix", ["", "1", "招募说明书（更新）", "测试基金管理有限责任公司"])
def test_product_level_prospectus_needs_exact_report_name_and_both_institutions(tmp_path, monkeypatch, prefix):
    from app.services import fund_quality_v2_disclosures as module

    report = (
        "基金名称测试景气精选股票型证券投资基金 基金简称测试股票 基金主代码123456 "
        "基金管理人测试基金管理有限责任公司 基金托管人测试银行股份有限公司"
    )
    prospectus = (
        prefix + "测试景气精选股票型证券投资基金更新的招募说明书 基金管理人：测试基金管理有限责任公司 "
        "基金托管人：测试银行股份有限公司"
    )
    refs = [
        {"label": label, "file": label, "sha256": label, "url": "https://www.cmfchina.com/" + label}
        for label in ("half", "prospectus")
    ]
    for ref in refs:
        (tmp_path / ref["file"]).write_bytes(b"original")
    monkeypatch.setattr(
        module,
        "public_document",
        lambda raw, ref, code, title: ([report if ref["label"] == "half" else prospectus], None),
    )
    docs = module.source_documents("123456", refs, tmp_path)
    assert docs["prospectus"]["identity_source_hashes"] == ["half"]
    prospectus = prospectus.replace("测试景气精选", "测试其他")
    assert (
        module.source_documents("123456", refs, tmp_path)["prospectus"]["error"]
        == "DOCUMENT_PRODUCT_NAME_BINDING_UNVERIFIED"
    )


def test_restricted_bond_is_included_and_must_fit_actual_bond_balance():
    from app.services.fund_quality_v2_restrictions import restricted_securities

    text = """6.4.12.1 因认购新发流通受限证券
    本基金未持有认购新发流通受限股票。
    6.4.12.1.2 受限证券类别：债券
    118071 测试转债 2026年6月23日 1个月内 新债未上市 100.00 100.00 770 76,999.83 77,001.18 -
    6.4.12.2 期末持有的暂时停牌等流通受限股票
    本基金未持有暂时停牌等流通受限股票。
    6.4.12.3 期末债券正回购
    6.4.12.3.1 银行间市场债券正回购
    本基金无因银行间市场债券正回购交易而抵押的债券。
    6.4.12.3.2 交易所市场债券正回购
    本基金无因交易所市场债券正回购交易而抵押的债券。
    6.4.12.4 期末参与转融通证券出借业务的证券
    本基金未持有参与转融通证券出借业务的证券。
    6.4.13 风险
    """
    assert restricted_securities([text], [], "100000")["restricted_amount"] == "77001.18"
    with pytest.raises(ValueError, match="BOND_BALANCE_CONFLICT"):
        restricted_securities([text], [], "77000")


def test_historical_total_is_not_bank_deposit_subtotal(tmp_path, monkeypatch):
    from app.services import fund_quality_v2_disclosures as module

    source = """测试基金123456，2026年3月31日
    5.1 报告期末基金资产组合情况 金额单位：人民币元
    1 权益投资 100.00 50.00
    2 银行存款和结算备付金合计 80.00 40.00
    3 其他资产 20.00 10.00
    4 合计 200.00 100.00
    5.2 报告期末按行业分类
    """
    (tmp_path / "original").write_bytes(b"TEST_ONLY")
    monkeypatch.setattr(module, "public_document", lambda *a: ([source], None))
    ref = {
        "label": "historical-quarter",
        "file": "original",
        "sha256": "TEST_ONLY",
        "report_end": "2026-03-31",
        "published_on": "2026-04-20",
        "url": "TEST_ONLY",
    }
    result = module.historical_report_facts("123456", [ref], tmp_path, date(2023, 9, 30), date(2026, 9, 30))["reports"][
        0
    ]
    assert result["total_assets"] == result["known_amount_sum"] == "200.00" and "gap" not in result


def test_selected_enhanced_index_title_cannot_be_classified_as_passive(monkeypatch, tmp_path):
    from app.services import fund_quality_v2_disclosures as module

    title = "测试沪深300指数精选增强型证券投资基金产品资料概要 编制日期2026年6月1日"
    doc = {"text": title, "pages": [title], "ref": {"sha256": "TEST_ONLY", "url": "TEST_ONLY"}}
    monkeypatch.setattr(module, "source_documents", lambda *a: {"summary": doc})
    result = module.extract_disclosures("123456", [], tmp_path, date(2026, 9, 30))
    assert result["facts"]["classification"]["kind"] == "INDEX_ENHANCED"


def test_a_share_cost_uses_external_subscription_and_does_not_add_basic_fees_again(monkeypatch, tmp_path):
    from app.services import fund_quality_v2_disclosures as module

    summary = (
        "测试基金（A类份额）基金产品资料概要 123456 交易币种人民币 送出日期2026年7月29日 "
        "三、投资本基金涉及的费用 基金销售相关费用 M<100万元1.2% 申购费（前收费） N≥180天0% "
        "基金运作相关费用 管理费0.15% 托管费0.05% 基金运作综合费率（年化）0.20% 四、风险"
    )
    prospectus = "A类基金份额：不从本类别基金资产中计提销售服务费。"
    docs = {
        label: {"pages": [body], "text": body, "ref": {"sha256": label, "url": "TEST_ONLY"}}
        for label, body in (("summary", summary), ("prospectus", prospectus))
    }
    monkeypatch.setattr(module, "source_documents", lambda *a: docs)
    result = module.cost_from_sources("123456", [], tmp_path, date(2026, 9, 30))
    assert abs(Decimal(result["value"]) - (Decimal(".012") / Decimal("1.012") + Decimal(".002"))) < Decimal("1e-26")
    assert set(result["source_hashes"]) == {"summary", "prospectus"}


def test_net_flow_can_be_known_while_both_gross_flows_remain_unknown():
    value, _, rows = holder_pressure(
        HEADING + "期初 申购 赎回 持有份额 份额占比（%）\n机构 1 20260101-20260630\n100.00 - - 100.00 25.00"
    )
    assert value == 1
    assert rows[0]["purchase"] is None and rows[0]["redemption"] is None
    assert rows[0]["net_flow"] == "0.00"
    assert rows[0]["reconciliation"] == "NET_FLOW_ONLY_GROSS_FLOWS_NOT_DISCLOSED"
    with pytest.raises(ValueError, match="VALUE_MISSING"):
        holder_pressure(HEADING + "期初 申购 赎回\n20260101-20260630\n100.00 - 100.00 - -")


def test_a_share_alias_needs_exact_code_and_header_and_ah_zero_sales_statement(tmp_path, monkeypatch):
    from app.services import fund_quality_v2_disclosures as module

    source = (
        "测试债券基金（测试债券A）基金产品资料概要更新 编制日期2026年5月28日 送出日期2026年5月29日 "
        "下属基金简称测试债券A 下属基金代码123456 交易币种人民币 "
        "三、投资本基金涉及的费用 M<100万元0.80% 申购费（前收费） 赎回费N>=7天0.00% "
        "基金运作综合费率（年化）0.18% 四、风险"
    )
    prospectus = "本基金A类、H类不收取销售服务费，C类销售服务费年费率为0.4%。"
    docs = {
        label: {"text": body, "pages": [body], "ref": {"sha256": label, "url": "TEST_ONLY"}}
        for label, body in (("summary", source), ("prospectus", prospectus))
    }
    monkeypatch.setattr(module, "source_documents", lambda *a: docs)
    result = module.cost_from_sources("123456", [], tmp_path, date(2026, 9, 30))
    assert Decimal(result["value"]) == Decimal("0.008") / Decimal("1.008") + Decimal("0.0018")
    docs["summary"]["text"] = source.replace("下属基金代码123456", "下属基金代码999999")
    with pytest.raises(ValueError, match="SHARE_DOCUMENT_IDENTITY"):
        module.cost_from_sources("123456", [], tmp_path, date(2026, 9, 30))


def test_multiple_threshold_intervals_are_one_holder_row_and_decimal_line_break_is_not_missing():
    value, _, rows = holder_pressure(
        HEADING + "期初 申购 赎回\n机构 1 2026-03-09至2026-03-23;2026-03-26至2026-04-19\n"
        "142,312,\n643.99 － 74,901,3\n91.57 67,411,252.42 9.91%\n产品特有风险"
    )
    assert value == 2 and len(rows) == 1 and len(rows[0]["intervals"]) == 2
    assert rows[0]["redemption"] == "74901391.57"
    value, _, rows = holder_pressure(
        HEADING + "期初 申购 赎回\n20260101-20260331\n78,482,3\n46.81 203,571.\n21 78,685,9\n18.02 － －"
    )
    assert value == 2 and rows[0]["end"] == "0.00" and rows[0]["end_percent"] is None


def test_combined_summary_uses_c_share_own_transaction_table_and_composite_rate(tmp_path, monkeypatch):
    from app.services import fund_quality_v2_disclosures as module

    source = (
        "测试黄金联接基金产品资料概要更新 编制日期2026年6月1日 送出日期2026年6月2日 "
        "基金代码C123456 交易币种人民币\n三、投资本基金涉及的费用\n"
        "测试黄金联接A：\n申购费（前收费）M<100万元0.60% 赎回费N≥366天0%\n"
        "测试黄金联接C：\n申购费（前收本基金C类费）--份额无申购费 赎回费N≥7天0%\n"
        "（二）基金运作相关费用 投资于目标ETF的部分不收取管理费。"
        "本基金投资于目标ETF的部分不收取托管费。\n（三）基金运作综合费用测算\n"
        "测试黄金联接A：\n基金运作综合费率（年化）0.71%\n测试黄金联接C：\n基金运作综合费率（年化）1.06%\n四、风险"
    )
    document = {"text": source, "pages": [source], "ref": {"sha256": "test-summary", "url": "TEST_ONLY"}}
    monkeypatch.setattr(module, "source_documents", lambda *a: {"summary": document})
    result = module.cost_from_sources("123456", [], tmp_path, date(2026, 9, 30))
    assert result["value"] == "0.0106"
    assert result["fee_proof"]["selected_share_label"] == "测试黄金联接C"
    # C类自己的赎回费未知时，A类的零费率不能补给C类。
    document["text"] = source.replace("赎回费N≥7天0%", "赎回费N≥7天－")
    with pytest.raises(ValueError, match="REDEMPTION_ZERO"):
        module.cost_from_sources("123456", [], tmp_path, date(2026, 9, 30))


def test_dedicated_c_renminbi_summary_is_not_a_combined_share_fee_table(tmp_path, monkeypatch):
    from app.services import fund_quality_v2_disclosures as module

    source = (
        "测试海外指数基金（QDII）（测试海外指数C人民币）基金产品资料概要更新 "
        "编制日期2026年6月25日 送出日期2026年6月26日 下属基金简称测试海外指数C人民币下属基金代码123456 交易币种人民币\n"
        "三、投资本基金涉及的费用 本基金C类基金份额不收取申购费 赎回费N≥7天0% "
        "基金运作综合费率（年化）基金运作综合费率1.33% 四、风险"
    )
    monkeypatch.setattr(
        module,
        "source_documents",
        lambda *a: {"summary": {"text": source, "pages": [source], "ref": {"sha256": "test-only", "url": "TEST_ONLY"}}},
    )
    assert module.cost_from_sources("123456", [], tmp_path, date(2026, 9, 30))["value"] == "0.0133"


@pytest.mark.parametrize("declared_master", ["123456", "999999"])
def test_product_manager_history_needs_original_master_and_never_expands_share_nav(
    tmp_path, monkeypatch, declared_master
):
    from app.services import fund_quality_v2_disclosures as module

    body = "测试股票基金2026年中期报告 基金主代码123456 下属份额654321 基金经理测试甲 基金合同生效日2020年1月1日"
    doc = {"text": body, "pages": [body], "ref": {"sha256": "test-report", "url": "TEST_ONLY"}}
    monkeypatch.setattr(module, "source_documents", lambda *a: {"half": doc})
    raw = {
        "code": 0,
        "data": {
            "fields": ["ts_code", "ann_date", "name", "begin_date", "end_date"],
            "items": [[declared_master + ".OF", "20200101", "测试甲", "20200101", None]],
        },
    }
    (tmp_path / "manager.json").write_text(json.dumps(raw), encoding="utf-8")
    refs = [
        {
            "label": "manager-master-api",
            "file": "manager.json",
            "ts_code": declared_master + ".OF",
            "master_code": declared_master,
            "sha256": "test-manager",
            "acquired_at": "2026-10-16T00:00:00+08:00",
        }
    ]
    result = module.extract_disclosures("654321", refs, tmp_path, date(2026, 9, 30))
    if declared_master == "123456":
        assert result["metrics"]["team_months"]["value"] == "36"
        assert set(result["metrics"]["team_months"]["source_hashes"]) == {"test-report", "test-manager"}
        assert "own_nav_coverage" not in result["facts"]
    else:
        assert "team_months" not in result["metrics"]
        assert result["facts"]["product_manager_history"]["error"] == "MANAGER_MASTER_NOT_BOUND_BY_ORIGINAL_PRODUCT"


def test_report_period_accepts_explicit_same_year_omission_but_not_reversed_cross_year_guess():
    assert report_period("本报告期自2025年1月1日起至12月31日止。") == (date(2025, 1, 1), date(2025, 12, 31))
    with pytest.raises(ValueError, match="REVERSED"):
        report_period("本报告期自2025年10月1日起至3月31日止。")


def test_manager_gap_does_not_become_full_coverage():
    records = [
        {"manager_name": "测试甲", "begin_date": "2020-01-01", "end_date": "2024-01-01"},
        {"manager_name": "测试乙", "begin_date": "2024-02-01", "end_date": None},
    ]
    with pytest.raises(ValueError, match="GAP"):
        manager_history(records, date(2023, 9, 30), date(2026, 9, 30))


def test_official_change_effective_date_and_role_duration():
    records = [
        {"manager_name": "测试甲", "begin_date": "2020-01-01", "end_date": "2024-01-01"},
        {"manager_name": "测试乙", "begin_date": "2024-01-01", "end_date": None},
    ]
    months, coverage, teams = manager_history(records, date(2023, 9, 30), date(2026, 9, 30))
    assert coverage == Decimal(1) and months < 36 and teams[-1]["members"] == ["测试乙"]


def test_report_date_is_not_download_or_publication_date():
    text = "本报告期自2026年1月1日起至2026年6月30日止。送出日期2026年8月31日。"
    assert report_period(text) == (date(2026, 1, 1), date(2026, 6, 30))


@pytest.mark.parametrize("publication", [None, "2026-10-20"])
def test_historical_future_or_unknown_publication_never_produces_asset_fact(tmp_path, publication):
    from app.services.fund_quality_v2_disclosures import historical_report_facts

    ref = {"label": "historical-quarter", "report_end": "2026-09-30", "published_on": publication,
           "sha256": "TEST_ONLY", "file": "must-not-read", "url": "TEST_ONLY"}
    row = historical_report_facts("123456", [ref], tmp_path, date(2023, 9, 30), date(2026, 9, 30))["reports"][0]
    assert row["error"] == ("HISTORICAL_REPORT_PUBLICATION_DATE_UNVERIFIED" if publication is None
                            else "HISTORICAL_REPORT_PUBLISHED_AFTER_AS_OF")
    assert "total_assets" not in row
