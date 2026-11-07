"""直接指数适配的经济口径及拒绝边界；测试数据均为隔离假设，不写真实评分。"""

import calendar
import hashlib
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal

import pytest
from app.services.fund_rating_direct_index import LIABILITIES, Originals, _liabilities
from app.services.fund_rating_direct_index_math import capacity, performance, replication, share_flow


def test_flat_equal_returns_have_zero_excess_and_risk():
    ends = []
    for index in range(37):
        year, month = 2023 + index // 12, index % 12 + 1
        ends.append(str(date(year, month, calendar.monthrange(year, month)[1])))
    result = performance(ends, ends, ["1"] * 37, ["10"] * 37)
    for key in ["return_1y", "return_3y", "extra_drawdown", "extra_downside", "tracking_error"]:
        assert Decimal(result[key]) == 0
    assert len(result["monthly_success"]) == 36 and len(result["annual_success"]) == 25
    with pytest.raises(ValueError, match="COMMON_WINDOW"):
        performance(ends, ends[:-1], ["1"] * 37, ["10"] * 37)


def test_replication_uses_full_stock_and_real_issuer_not_cash_or_coarse_industry():
    positions = [{"stock_code": "a", "market_value_cny": "70"}, {"stock_code": "b", "market_value_cny": "30"}]
    result = replication(positions, "100", {"a": ".25", "b": ".75"},
                         {"a": "issuer_a", "b": "issuer_b"}, {"a": "C35", "b": "C39"})
    assert Decimal(result["issuer_replication"]) == Decimal(".45")
    assert Decimal(result["industry_replication"]) == Decimal(".45")
    with pytest.raises(ValueError, match="STOCK_TOTAL"):
        replication(positions, "110", {"a": ".25", "b": ".75"}, {"a": "x", "b": "y"}, {"a": "C35", "b": "C39"})
    with pytest.raises(ValueError, match="MISSING"):
        replication(positions, "100", {"a": ".25", "b": ".75"}, {"a": "x"}, {"a": "C35", "b": "C39"})


def test_capacity_converts_provider_thousand_yuan_and_never_imputes_missing_day():
    days = [str(date(2026, 7, 1) + timedelta(days=i)) for i in range(60)]
    position = [{"stock_code": "fixture", "market_value_cny": "5000000"}]
    market = {"fixture": {d: "10000" for d in days}}
    assert Decimal(capacity(position, "5000000", days, market)["K"]) == Decimal(".5")
    market["fixture"].pop(days[15])
    with pytest.raises(ValueError, match="DAYS_MISSING"):
        capacity(position, "5000000", days, market)


def test_dash_settlement_is_only_zero_when_complete_nonnegative_liabilities_prove_it():
    rows = [name + (" 10.00 5.00" if name == "应付赎回款" else " － －") for name in LIABILITIES]
    pages = ["\n".join(rows).replace("衍生金融负债 － －", "衍生金融负债 6.4.7.3 － －") + "\n负债合计 10.00 5.00"]
    assert _liabilities(pages, "10")["point_values"]["应付清算款"] == "0.00"
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        _liabilities(pages, "11")
    with pytest.raises(ValueError, match="ROW"):
        _liabilities([pages[0].replace("应付清算款 － －", "")], "10")


def test_share_flow_dash_is_not_a_blanket_zero():
    rows = [{"share": "C", "opening": "100", "subscribed": "20", "redeemed": "50", "closing": "70", "split": None}]
    assert share_flow(rows)[0]["split"] == "0"
    rows[0]["closing"] = "80"
    with pytest.raises(ValueError, match="NOT_RECONCILED"):
        share_flow(rows)


def test_source_tampering_and_path_escape_fail_before_parsing(tmp_path):
    file = tmp_path / "sources" / "fixture.json"
    file.parent.mkdir()
    file.write_text('{"fixture": true}', encoding="utf-8")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    raw = {"as_of_date": "2026-09-30", "sources": {digest: {"file": "sources/fixture.json", "sha256": digest}}}
    assert Originals(raw, tmp_path).json(digest) == {"fixture": True}
    escaped = deepcopy(raw)
    escaped["sources"][digest]["file"] = "../fixture.json"
    with pytest.raises(ValueError, match="SOURCE_SCOPE"):
        Originals(escaped, tmp_path)
    file.write_text('{"fixture": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_HASH"):
        Originals(raw, tmp_path)


def test_existing_compilation_cache_does_not_bypass_changed_original(tmp_path, monkeypatch):
    from app.services import fund_rating_direct_index as module

    file = tmp_path / "sources" / "fixture.json"
    file.parent.mkdir()
    file.write_bytes(b'{}')
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    raw = {"as_of_date": "2026-09-30", "sources": {digest: {"file": "sources/fixture.json", "sha256": digest}}}
    monkeypatch.setattr(module, "_compile_cached", lambda *args: {"isolated_cached_observation": True})
    assert module.compile_observations(raw, root=tmp_path)["isolated_cached_observation"]
    file.write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError, match="SOURCE_HASH"):
        module.compile_observations(raw, root=tmp_path)


def test_physical_page_prefix_is_verified_for_every_body_page_and_preserves_stock_digits():
    from app.services.fund_rating_direct_index import physical_page_prefix

    pages = ["隔离封面", " 2\n1 600000 测试股份 100 1,000.00 10.00\n",
             "3\n2 000001 另一个股票 57 1,583.46 0.00\n", "4\n7.4 报告期变动"]
    parsed = physical_page_prefix(pages)
    assert parsed[1].startswith("1 600000") and "57 1,583.46 0.00" in parsed[2]
    assert parsed[3].startswith("7.4")
    with pytest.raises(ValueError, match="PREFIX_UNVERIFIED"):
        physical_page_prefix([*pages[:2], pages[2].replace("3\n", "30\n"), pages[3]])
