"""014156标准候选的增量拒绝边界；合成源仅留pytest目录，不进入正式请求。"""

import hashlib
import json
from decimal import Decimal as D

import pytest
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_enhanced_performance import (
    DIVIDEND_FIELDS,
    _dividends,
    _iso,
    _material_review,
    _read,
    _verify_contracts,
    check_binding,
    standard_growth,
)


def source(tmp_path, raw):
    folder = tmp_path / "sources"
    folder.mkdir(exist_ok=True)
    data = json.dumps(raw).encode()
    digest = hashlib.sha256(data).hexdigest()
    path = folder / (digest + ".json")
    path.write_bytes(data)
    ref = {
        "file": path.relative_to(tmp_path).as_posix(),
        "sha256": digest,
        "published_on": "2026-09-01",
        "params": {"ts_code": "008778.OF"},
    }
    return Originals({"sources": {digest: ref}, "as_of_date": "2026-09-30"}, tmp_path), ref


def raw_dividends(rows):
    return {
        "code": 0,
        "data": {
            "fields": DIVIDEND_FIELDS,
            "items": [[r[k] for k in DIVIDEND_FIELDS] for r in rows],
            "has_more": False,
            "count": len(rows),
        },
    }


def event(**overrides):
    return {
        "ts_code": "008778.OF",
        "ann_date": "20260901",
        "imp_anndate": "20260901",
        "ex_date": "20260902",
        "div_proc": "实施",
        "div_cash": "0.10",
        "base_unit": "1000",
        "record_date": "20260901",
        "pay_date": "20260903",
        **overrides,
    }


def dividends(originals, ref, curve=None):
    member = {"code": "008778", "ticker": "008778.OF", "dividend_source": ref}
    return _dividends(
        member,
        originals,
        ["2026-09-01", "2026-09-02"],
        curve if curve is not None else {"2026-09-02": {"unitMoney": "分红"}},
        [],
    )


def test_equal_decimal_duplicate_is_one_event(tmp_path):
    originals, ref = source(tmp_path, raw_dividends([event(), event(div_cash="0.100")]))
    values, records = dividends(originals, ref)
    assert values == {"2026-09-02": D("0.10")} and len(records) == 1


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"record_date": "20260831"}, "ENHANCED_DIVIDEND_SAME_DATE_CONFLICT"),
        ({"pay_date": "20260904"}, "ENHANCED_DIVIDEND_SAME_DATE_CONFLICT"),
        ({"div_cash": "0.20"}, "ENHANCED_DIVIDEND_SAME_DATE_CONFLICT"),
        ({"ann_date": "20261001"}, "ENHANCED_DIVIDEND_ANNOUNCEMENT_CUTOFF"),
        ({"ts_code": "008779.OF"}, "ENHANCED_DIVIDEND_WRONG_SHARE"),
    ],
)
def test_ambiguous_or_future_cash_events_rejected(tmp_path, changes, error):
    originals, ref = source(tmp_path, raw_dividends([event(), event(**changes)]))
    with pytest.raises(ValueError, match=error):
        dividends(originals, ref)


def test_unknown_status_missing_date_is_not_silently_absence(tmp_path):
    originals, ref = source(tmp_path, raw_dividends([event(div_proc="未知", ex_date=None)]))
    with pytest.raises(ValueError, match="ENHANCED_DIVIDEND_UNKNOWN_STATUS"):
        dividends(originals, ref, {})


@pytest.mark.parametrize("change", [{"has_more": True}, {"has_more": None}, {"count": 1}, {"count": None}])
def test_empty_response_requires_complete_page_proof(tmp_path, change):
    raw = raw_dividends([])
    raw["data"].update(change)
    originals, ref = source(tmp_path, raw)
    with pytest.raises(ValueError, match="ENHANCED_DIVIDEND_PAGE_INCOMPLETE"):
        dividends(originals, ref, {})


def test_original_capture_request_must_be_this_share(tmp_path):
    originals, ref = source(tmp_path, raw_dividends([]))
    ref = {**ref, "params": {"ts_code": "008779.OF"}}
    with pytest.raises(ValueError, match="ENHANCED_DIVIDEND_REQUEST_SHARE"):
        dividends(originals, ref, {})


def test_original_reader_never_opens_nested_external_path(tmp_path):
    originals, ref = source(tmp_path, {"TEST_ONLY": True})
    assert _read({**ref, "path": "Z:/never-read-outside-sources.txt"}, originals) == originals.raw[ref["sha256"]]


@pytest.mark.parametrize("value", ["2026/09/30", "2026-W40-3", "20261301", None, True])
def test_unknown_or_invalid_date_encoding_rejected(value):
    with pytest.raises(ValueError):
        _iso(value)


def test_iso_and_compact_dates_have_same_economic_meaning():
    assert _iso("2026-09-30") == _iso("20260930") == "2026-09-30"


def test_frozen_previous_nav_growth_handles_market_change_on_ex_day():
    wealth, _ = standard_growth(["2026-09-01", "2026-09-02"], [D("1"), D(".95")], {"2026-09-02": D(".1")})
    assert abs(wealth[-1] / wealth[0] - D(".95") / D(".9")) < D("1e-26")
    assert abs((wealth[-1] / wealth[0] - 1) - D(".05")) > D(".001")


@pytest.mark.parametrize("state", [None, True, "False", 0])
def test_material_event_state_never_defaulted_to_false(state):
    with pytest.raises(ValueError, match="ENHANCED_EXPLICIT_MATERIAL_REVIEW_REQUIRED"):
        _material_review({"material_event_pending": state}, None)


def test_contract_period_is_read_from_body_not_assigned_label(tmp_path):
    refs = {}
    all_sources = {}
    for period in ["2023-12-31", "2024-12-31", "2025-12-31", "2026-06-30"]:
        text = (
            "本报告期自2023年1月1日起至2023年12月31日止。基金名称国泰海通中证500指数增强型证券投资基金"
            "基金简称国泰海通500增强基金主代码014155下属代码014156业绩比较基准中证500指数收益率×95%"
            "+银行活期存款利率×5%风险收益特征量化增强。测试标识" + period
        )
        originals, ref = source(tmp_path, {"data": {"notice_content": text}})
        refs[period] = ref
        all_sources.update(originals.sources)
    originals = Originals({"sources": all_sources, "as_of_date": "2026-09-30"}, tmp_path)
    with pytest.raises(ValueError, match="ENHANCED_ACTUAL_CONTRACT_PERIOD:014156:2024-12-31"):
        _verify_contracts(
            {"target": {"code": "014156", "contracts": refs}, "peers": [], "as_of_date": "2026-09-30"}, originals
        )


@pytest.mark.parametrize("route", ["基本面选股", "量化选股"])
def test_same_index_enhanced_group_does_not_add_quantitative_only_gate(tmp_path, route):
    refs, all_sources = {}, {}
    for period in ["2023-12-31", "2024-12-31", "2025-12-31", "2026-06-30"]:
        year = period[:4]
        month, day = map(int, period[5:].split("-"))
        text = (
            f"本报告期自{year}年1月1日起至{year}年{month}月{day}日止。"
            "基金名称国泰海通中证500指数增强型证券投资基金基金简称国泰海通500增强"
            "基金主代码014155下属代码014156业绩比较基准中证500指数收益率×95%"
            f"+银行活期存款利率×5%风险收益特征。本基金采用{route}实现指数增强。"
        )
        originals, ref = source(tmp_path, {"data": {"notice_content": text}})
        refs[period] = ref
        all_sources.update(originals.sources)
    originals = Originals({"sources": all_sources, "as_of_date": "2026-09-30"}, tmp_path)
    result = _verify_contracts(
        {"target": {"code": "014156", "contracts": refs}, "peers": [], "as_of_date": "2026-09-30"}, originals
    )
    assert len(result) == 4


@pytest.mark.parametrize("change", [{"share_class": "A"}, {"primary_ref": "another-report"}])
def test_current_fund_binding_rejects_foreign_share_or_primary(change):
    bundle = {
        "family": "ACTIVE_EQUITY",
        "fund_code": "014156",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "identity": {"legal_name": "本法律产品", "manager_name": "本管理人"},
        "original_report_archive": {"sha256": "own-report"},
    }
    raw = {
        "fund_code": "014156",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "legal_name": "本法律产品",
        "manager_name": "本管理人",
        "primary_ref": "own-report",
        **change,
    }
    with pytest.raises(ValueError, match="ENHANCED_CURRENT_BINDING"):
        check_binding(bundle, raw)
