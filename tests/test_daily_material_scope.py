"""持仓采集边界：空范围、来源错误均不能回退全市场。"""

from types import SimpleNamespace

import pytest
from app.services.fund_market_workflow import MarketSyncWorkflow
from app.services.tushare_fund_sync import TushareFundSyncService


def test_workflow_fetches_current_holdings_once_and_excludes_other_catalog_funds(monkeypatch):
    from app.commands import fund_prediction_followthrough

    calls = []
    monkeypatch.setattr(fund_prediction_followthrough, "owner_scope", lambda *args: calls.append(args) or ["000002"])
    workflow = MarketSyncWorkflow(
        preparation=object(), owner_user_id="owner",
        scope_provider=lambda: [{"fund_code": "000001"}, {"fund_code": "000002"}],
    )
    assert workflow.scope() == [{"fund_code": "000002"}]
    assert workflow.scope() == [{"fund_code": "000002"}]
    assert len(calls) == 1


@pytest.mark.parametrize("held", [[], ["000003"]])
def test_empty_or_unmapped_holdings_never_use_catalog_fallback(monkeypatch, held):
    from app.commands import fund_prediction_followthrough

    monkeypatch.setattr(fund_prediction_followthrough, "owner_scope", lambda *args: held)
    workflow = MarketSyncWorkflow(
        preparation=object(), owner_user_id="owner", scope_provider=lambda: [{"fund_code": "000001"}],
    )
    if held:
        with pytest.raises(ValueError, match="HELD_FUND_NOT_IN_ACTIVE_CATALOG"):
            workflow.scope()
    else:
        assert workflow.scope() == []


def test_source_target_filter_keeps_empty_scope_empty_and_requires_mapping():
    targets = [SimpleNamespace(fund_code=code) for code in ("000001", "000002")]
    assert TushareFundSyncService._held_targets(targets, ()) == []
    assert TushareFundSyncService._held_targets(targets, ("000002",)) == targets[1:]
    with pytest.raises(ValueError, match="HELD_FUND_SOURCE_MAPPING_MISSING"):
        TushareFundSyncService._held_targets(targets, ("000003",))


@pytest.mark.parametrize("codes", [(), ("000002",)])
def test_explicit_fee_scope_never_reads_broader_fee_directory(monkeypatch, codes):
    from app.services.simulation_fee_sync import SimulationFeeSyncService

    worker = object.__new__(SimulationFeeSyncService)
    saved = []
    monkeypatch.setattr(worker, "_sync_one", saved.append)
    result = worker.sync(fund_codes=codes, progress_reporter=lambda *args: None)
    assert result.total == result.updated == len(codes)
    assert saved == list(codes)
