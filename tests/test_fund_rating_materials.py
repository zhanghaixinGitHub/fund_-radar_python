"""资料适配边界的隔离测试；测试数值不是基金真实评级。"""

import hashlib
from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import pytest
from app.integrations.fund_rating_public import collect_risk_free, risk_free_intervals
from app.services.fund_exposure_common import read, save
from app.services.fund_rating_disclosures import appointed_teams, key_information, statement_amount
from app.services.fund_rating_inputs import empty_evidence
from app.services.fund_rating_materials import _documents, _raw, archive_materials, attach_materials
from app.services.fund_rating_metrics import equity_structure, holdings_metrics, standard_cost


def test_balance_receivable_is_not_cash_and_unknown_holder_blocks():
    positions = [{"issuer": "a", "industry": "i", "amount": "80", "saleable_amount": "80", "turnover_20d": [100] * 20}]
    # 80股票+10现金+30应收-20总负债=100净资产；其中五日应付仅5，不能拿它替代总负债。
    balance = {"other_assets": {"cash": "10", "receivable": "30"}, "liabilities": "20", "total_assets": "120"}
    value = holdings_metrics(positions, 100, 10, 5, ".1", balance=balance)
    assert value["liquid_5d"] == Decimal(".55")
    assert equity_structure(positions)["issuer_hhi"] == 1
    with pytest.raises(ValueError, match="NUMBER_REQUIRED"):
        holdings_metrics(positions, 100, 10, 5, None, balance=balance)
    with pytest.raises(ValueError, match="ASSETS_NOT_RECONCILED"):
        holdings_metrics(positions, 101, 10, 5, ".1", balance=balance)
    with pytest.raises(ValueError, match="BALANCE_CONFLICT"):
        holdings_metrics(positions, 100, 41, 5, ".1", balance=balance)


def test_comprehensive_cost_does_not_double_count_basic_fees():
    assert standard_cost(".007", ".0015", ".001", 0, 0, mode="EXTERNAL", operating_cost=".0096") == Decimal(".0096")
    with pytest.raises(ValueError, match="OPERATING_COST_RANGE"):
        standard_cost(".007", ".0015", ".001", 0, 0, mode="EXTERNAL", operating_cost=".009")


@pytest.mark.parametrize("redemption", ["0.5%", "0.05%"])
def test_nonzero_redemption_rate_is_not_read_as_zero_prefix(redemption):
    text = (
        "德邦鑫星价值001412002112交易币种人民币送出日期：2026年6月26日"
        "基金经理陆阳开始担任本基金基金经理的日期2024年1月30日"
        "管理费0.7%托管费0.15%销售服务费0.1%基金运作综合费率（年化）持有期间0.96%"
        "不收取申购费用N≥30日" + redemption
    )
    with pytest.raises(ValueError, match="FEE_SCENARIO_UNVERIFIED"):
        key_information([text])


def test_statement_selects_current_amount_not_note_number_or_prior_period():
    assert statement_amount("货币资金 6.4.7.1 1,234.56 9,876.54", "货币资金") == Decimal("1234.56")
    with pytest.raises(ValueError, match="ROW_AMBIGUOUS"):
        statement_amount("货币资金 12.34\n货币资金 23.45", "货币资金")
    with pytest.raises(ValueError, match="AMOUNT_MISSING"):
        statement_amount("货币资金 - -", "货币资金")


def test_raw_hash_time_and_path_are_checked(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"pdf")
    receipt = {
        "file": "a.pdf",
        "sha256": hashlib.sha256(b"pdf").hexdigest(),
        "received_at": "2026-10-10T00:00:00+00:00",
    }
    frozen = datetime(2026, 10, 14, tzinfo=UTC)
    assert _raw(root, receipt, frozen) == b"pdf"
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        _raw(root, {**receipt, "sha256": "0" * 64}, frozen)
    with pytest.raises(ValueError, match="PATH_SIZE"):
        _raw(root, {**receipt, "file": "../../a.pdf"}, frozen)
    with pytest.raises(ValueError, match="FUTURE"):
        _raw(root, receipt, datetime(2026, 10, 9, tzinfo=UTC))


def test_partial_preparation_cannot_admit_or_create_scores():
    member = {
        "admitted": False,
        "representative": False,
        "metrics": {},
        "public": empty_evidence({}, "ACTIVE_EQUITY", date(2026, 9, 30)),
    }
    material = {
        "facts": {},
        "diagnostics": {"sharpe_proxy": 2},
        "dimensions": {},
        "blockers": [{"code": "MISSING", "dimension": "liquidity", "message": "持有人资料不足"}],
    }
    attach_materials(member, material)
    assert not member["admitted"] and not member["metrics"] and not member["representative"]
    assert len(member["public"]["dimensions"]) == 8
    assert all(d["grade"] is None for d in member["public"]["dimensions"])
    assert "sharpe_proxy" not in str(member["public"])


def test_risk_free_uses_interval_start_yield_and_actual_days():
    data = {
        "currency": "CNY",
        "unit": "PERCENT_PER_YEAR",
        "rows": {
            "2026-09-25": {"threeMonth": "1.5"},
            "2026-09-28": {"threeMonth": "99"},
        },
    }
    values = risk_free_intervals(data, [date(2026, 9, 25), date(2026, 9, 28)], term="threeMonth")
    assert values == [(1 + Decimal(".015")) ** (Decimal(3) / 365) - 1]
    with pytest.raises(ValueError, match="DATE_MISSING"):
        risk_free_intervals(data, [date(2026, 9, 24), date(2026, 9, 28)], term="threeMonth")


def test_public_collection_failure_keeps_previous_complete_pointer(tmp_path):
    save(tmp_path / "risk-free.json", {"previous": True})

    def respond(request):
        assert request.url.host == "yield.chinabond.com.cn"
        return httpx.Response(200, json={"flag": "1", "heList": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError, match="SOURCE_EMPTY"):
            collect_risk_free(date(2026, 9, 28), date(2026, 9, 30), tmp_path, client=client)
    assert read(tmp_path / "risk-free.json") == {"previous": True}
    assert list((tmp_path / "receipts").glob("*.json"))  # 失败原文/回执仍保留。


def test_formal_appointments_reconstruct_changes_and_do_not_count_current_start_as_last_change():
    pages = ["本基金历任基金经理：甲乙先生，2020年1月1日至2024年2月1日管理本基金。"]
    info = {"current_manager": "丙丁", "manager_start": "2023-01-01"}
    value = appointed_teams(pages, info, date(2023, 9, 30), date(2026, 9, 30))
    assert value["team_changes"] == 1
    assert value["teams"][-1] == {"effective_date": "2024-02-01", "members": ["丙丁"]}


def test_archived_sources_are_immutable_and_survive_research_source_removal(tmp_path):
    origin, target = tmp_path / "case", tmp_path / "ratings"
    origin.mkdir()
    source = origin / "report.pdf"
    source.write_bytes(b"official raw bytes")
    receipt = {
        "file": "report.pdf",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "received_at": "2026-10-10T00:00:00+00:00",
    }
    material = {"source_root": str(origin), "sources": [receipt, receipt]}
    frozen = datetime(2026, 10, 14, tzinfo=UTC)
    refs = archive_materials(material, target, frozen)
    assert len(refs) == 1
    assert archive_materials(material, target, frozen) == refs
    saved = target / refs[0]["file"]
    saved.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="ARCHIVED_SOURCE_CORRUPT"):
        archive_materials(material, target, frozen)
    assert saved.read_bytes() == b"corrupted"  # 发现异常不能覆盖原证据伪装修复。


def test_future_legal_document_is_not_used_for_historical_valuation(tmp_path):
    title = "德邦鑫星价值C份额产品资料概要"
    save(tmp_path / "supplement/public-result.json", {"documents": [{"title": title, "file": "summary.json"}]})
    stamp = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp() * 1000)
    save(tmp_path / "summary.json", {"catalog": {"title": title, "activationDate": stamp}})
    assert _documents(tmp_path, date(2026, 9, 30)) == []
    assert len(_documents(tmp_path, date(2026, 10, 1))) == 1


def test_public_dates_distinguish_document_publication_from_verification():
    member = {"admitted": False, "metrics": {}, "public": empty_evidence({}, "ACTIVE_EQUITY", date(2026, 9, 30))}
    attach_materials(
        member,
        {
            "blockers": [],
            "dimensions": {},
            "facts": {
                "key_information": {"published_on": "2026-06-26", "checked_on": "2026-09-24"},
                "history": {"dates": ["2023-09-28", "2026-09-30"], "missing_dates": []},
            },
        },
    )
    assert member["public"]["dataDates"]["fees"] == "2026-09-24"
    assert member["public"]["dataDates"]["nav"] == "2026-09-30"
    assert member["public"]["comparison"]["period"] == "2023-09-28 至 2026-09-30"
    assert not member["admitted"]
