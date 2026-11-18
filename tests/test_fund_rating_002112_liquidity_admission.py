import json
from hashlib import sha256

import pytest
from app.services.fund_rating_002112_liquidity_admission import operating_decision

NAME = "甲灵活配置混合型证券投资基金"


def report(name=NAME, mode="契约型开放式", basis="", tail=""):
    return (
        name
        + "2026年中期报告2.1基金基本情况基金名称"
        + name
        + "基金主代码000001基金运作方式"
        + mode
        + "基金合同生效日2014年1月1日2.2基金产品说明主动股票投资6.4.1基金基本情况"
        + name
        + basis
        + "6.4.2会计报表的编制基础"
        + tail
    )


def decision(body, name=NAME, **kwargs):
    return operating_decision(body, legal_name=name, as_of_date="2026-10-30", **kwargs)


def test_periodic_open_name_overrides_generic_open_table():
    name = "银华战略新兴灵活配置定期开放混合型发起式证券投资基金"
    result = decision(report(name), name)
    assert result["status"] == "EXCLUDED"


def test_manager_other_periodic_funds_and_redemption_fee_tenor_do_not_exclude_ordinary_product():
    result = decision(report(tail="管理人另有乙一年定期开放基金。本基金赎回费随持有期限递减。"))
    assert result["status"] == "ADMITTED"


def test_holding_period_in_own_mode_cannot_be_ordinary():
    assert decision(report(mode="契约型开放式,三年持有期"))["status"] == "EXCLUDED"


def test_initial_closed_period_requires_actual_effective_conversion_not_arithmetic():
    body = report(mode="契约型开放式,初始封闭期三年", basis="2014年成立,封闭期三年。")
    assert decision(body)["status"] == "PENDING"


def test_actual_past_conversion_is_distinct_from_future_conversion():
    mode = "契约型开放式,初始封闭期三年"
    assert (
        decision(report(mode=mode, basis="本基金自2017年9月19日起转型为上市开放式基金(LOF)。"))["status"] == "ADMITTED"
    )
    assert (
        decision(report(mode=mode, basis="本基金自2027年9月19日起转型为上市开放式基金(LOF)。"))["status"] == "PENDING"
    )


def test_reordered_pdf_table_can_use_own_financial_legal_statement():
    assert decision(report(mode="", basis="本基金为契约型开放式基金,存续期限不定。"))["status"] == "ADMITTED"


def test_contractual_or_lof_label_alone_is_not_daily_open_evidence():
    assert decision(report(mode="契约型"))["status"] == "PENDING"


def test_explicit_same_product_summary_proves_each_open_day():
    extra = NAME + "基金代码000001运作方式普通开放式开放频率每个开放日"
    assert decision(report(mode="契约型"), supplement=extra)["status"] == "ADMITTED"
    with pytest.raises(ValueError, match="SUPPLEMENT_IDENTITY"):
        decision(report(mode="契约型"), supplement="别的基金运作方式普通开放式开放频率每个开放日")


def test_other_product_report_is_rejected():
    with pytest.raises(ValueError, match="LEGAL_IDENTITY"):
        decision(report(name="乙基金"))


def test_missing_own_basic_section_is_pending_not_inferred_from_manager_list():
    assert decision(NAME + "管理人另有基金运作方式契约型开放式")["status"] == "PENDING"


def test_actual_body_can_have_renumbered_headings_without_using_table_of_contents():
    body = report().replace("2.1基金基本情况", "1.1基金基本情况").replace("2.2基金产品说明", "1.2基金产品说明")
    assert decision("目录2.1基金基本情况......5" + body)["status"] == "ADMITTED"


def test_own_financial_statement_periodic_mode_overrides_generic_table():
    assert decision(report(basis="本基金为契约型定期开放式基金。"))["status"] == "EXCLUDED"


@pytest.fixture
def compiled_input(tmp_path, monkeypatch):
    def archive(value, kind, published=None):
        raw = json.dumps(value, ensure_ascii=False).encode()
        digest = sha256(raw).hexdigest()
        folder = tmp_path / "sources"
        folder.mkdir(exist_ok=True)
        (folder / (digest + ".json")).write_bytes(raw)
        return digest, {
            "file": "sources/" + digest + ".json",
            "sha256": digest,
            "format": "json",
            "archive_kind": kind,
            "published_on": published,
        }

    upstream, upstream_ref = archive({"as_of_date": "2026-10-30"}, "DERIVED")
    body = report() + "本报告期自2026年1月1日起至2026年6月30日止。"
    original, original_ref = archive({"data": {"notice_content": body}}, "ORIGINAL", "2026-08-28")
    from app.services import fund_rating_active_monthly_admission

    monkeypatch.setattr(
        fund_rating_active_monthly_admission,
        "replay_monthly_admission",
        lambda inputs, *, root: {"status": "EXPOSURE_ADMISSION_COMPLETE", "structure_codes": ["000001"]},
    )
    return tmp_path, {
        "version": "R5_002112_MONTHLY_LIQUIDITY_ADMISSION_V1",
        "fund_code": "002112",
        "as_of_date": "2026-10-30",
        "research_on": "2026-11-17",
        "monthly_admission_ref": upstream,
        "sources": {upstream: upstream_ref, original: original_ref},
        "portfolios": {
            "000001": {
                "source_ref": original,
                "share_codes": ["000001"],
                "legal_name": NAME,
                "period": "2026-06-30",
                "published_on": "2026-08-28",
            }
        },
    }


def test_production_compilation_reads_original_and_keeps_graph_binding(compiled_input):
    from app.services.fund_rating_002112_liquidity_admission import compile_admission

    root, inputs = compiled_input
    result = compile_admission(inputs, root=root)
    assert result["admitted_codes"] == ["000001"]
    assert result["monthly_admission_ref"] in result["used_source_refs"]


def test_missing_peer_is_not_silently_removed(compiled_input):
    from app.services.fund_rating_002112_liquidity_admission import compile_admission

    root, inputs = compiled_input
    inputs["portfolios"].clear()
    with pytest.raises(ValueError, match="CANNOT_REMOVE_PEER"):
        compile_admission(inputs, root=root)


def test_report_period_cannot_be_relabelled(compiled_input):
    from app.services.fund_rating_002112_liquidity_admission import compile_admission

    root, inputs = compiled_input
    inputs["portfolios"]["000001"]["period"] = "2026-03-31"
    with pytest.raises(ValueError, match="ACTUAL_REPORT_PERIOD"):
        compile_admission(inputs, root=root)


def test_changed_original_bytes_are_rejected(compiled_input):
    from app.services.fund_rating_002112_liquidity_admission import compile_admission

    root, inputs = compiled_input
    digest = inputs["portfolios"]["000001"]["source_ref"]
    (root / inputs["sources"][digest]["file"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_HASH"):
        compile_admission(inputs, root=root)


def test_later_legal_mode_cannot_be_admitted_as_historical_trading_evidence(compiled_input):
    from app.services.fund_rating_002112_liquidity_admission import compile_admission

    root, inputs = compiled_input
    raw = json.dumps({"data": {"notice_content": NAME + "000001运作方式普通开放式开放频率每个开放日"}}).encode()
    digest = sha256(raw).hexdigest()
    (root / "sources" / (digest + ".json")).write_bytes(raw)
    inputs["sources"][digest] = {
        "file": "sources/" + digest + ".json",
        "sha256": digest,
        "format": "json",
        "archive_kind": "ORIGINAL",
        "published_on": "2026-10-31",
        "economic_role": "HISTORICAL_PRICE",
    }
    inputs["portfolios"]["000001"]["operation_supplement_ref"] = digest
    with pytest.raises(ValueError, match="SUPPLEMENT_NOT_HISTORICAL_PRICE"):
        compile_admission(inputs, root=root)
