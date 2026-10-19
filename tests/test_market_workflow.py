"""市场/关注资料只按本轮证据放行；替身不联网、不生成真实预测。"""

from datetime import datetime

import pytest
from app.services.fund_exposure_common import save
from app.services.fund_market_preparation import MarketFundPreparation
from app.services.fund_market_workflow import MarketSyncWorkflow


class Preparation:
    def sync_news(self, code, **kwargs):
        if code == "000001":
            raise ValueError("source failure")
        return {"status": "SUCCEEDED", "catalog_complete": True}

    def sync_holdings(self, funds, **kwargs):
        return {"status": "PARTIAL_SUCCESS", "funds": [{"fund_code": "000002", "status": "SUCCEEDED"}]}

    def sync_material(self, fund, news, **kwargs):
        return {"fund_code": fund["fund_code"], "status": "SUCCEEDED", "gaps": []}

    def prepare_inputs(self, funds, **kwargs):
        # 故意提供旧包已齐备的结果；本轮失败的基金仍必须被拦截。
        return {
            "items": [
                {"fund_code": f["fund_code"], "prediction_ready": True, "gaps": [], "status": "READY"} for f in funds
            ]
        }

    def close(self):
        pass


def test_failure_does_not_abort_other_funds_or_reuse_old_readiness():
    workflow = MarketSyncWorkflow(
        preparation=Preparation(), scope_provider=lambda: [{"fund_code": "000001"}, {"fund_code": "000002"}]
    )

    def progress(*_):
        pass

    assert workflow.news(progress)["status"] == "PARTIAL_SUCCESS"
    assert len(workflow.news_results) == 2
    assert workflow.materials(progress)["status"] == "PARTIAL_SUCCESS"
    result = workflow.inputs(progress)
    assert result["created"] == 1 and result["skipped"] == 1
    assert workflow.prediction_codes() == ["000002"]
    assert workflow.input_results["000001"]["status"] == "WAITING_DATA"


@pytest.mark.parametrize("prior_status", ["SUCCEEDED", "PARTIAL_SUCCESS"])
def test_daily_workflow_obeys_weekly_cadence_and_preserves_per_fund_failures(tmp_path, monkeypatch, prior_status):
    from app.services import market_holdings_maintenance as holdings

    checked = "2026-11-02"
    save(
        tmp_path / "state.json",
        {
            "status": prior_status,
            "last_checked_on": checked,
            "finished_at": checked + "T00:31:00+08:00",
            "funds": [
                {"fund_code": "000001", "status": "PARTIAL_SUCCESS", "errors": ["test-gap"]},
                {"fund_code": "000002", "status": "SUCCEEDED"},
            ],
        },
    )

    class Maintenance:
        root = tmp_path

        def check(self, *, funds, force=False):
            assert not force
            return {"status": "NOT_DUE", "last_checked_on": checked, "downloads": 0}

    monkeypatch.setattr(holdings, "MarketHoldingsMaintenance", Maintenance)
    preparation = MarketFundPreparation(
        root=tmp_path, clock=lambda: datetime.fromisoformat("2026-11-04T00:30:00+08:00")
    )
    result = preparation.sync_holdings([{"fund_code": c} for c in ("000001", "000002", "000003")])
    assert result["status"] == "NOT_DUE" and result["reused"] and not result["performed_now"]
    assert result["source_checked_at"] == checked + "T00:31:00+08:00"
    assert result["funds"][0]["status"] == "PARTIAL_SUCCESS"
    assert {f["fund_code"] for f in result["funds"]} == {"000001", "000002"}


@pytest.mark.parametrize("status,checked", [("RUNNING", "2026-11-02"), ("SUCCEEDED", "2026-11-06")])
def test_unfinished_or_future_receipt_cannot_be_reused(tmp_path, monkeypatch, status, checked):
    from app.services import market_holdings_maintenance as holdings

    save(
        tmp_path / "state.json",
        {
            "status": status,
            "last_checked_on": checked,
            "finished_at": checked + "T00:31:00+08:00",
            "funds": [{"fund_code": "000002", "status": "SUCCEEDED"}],
        },
    )

    class Maintenance:
        root = tmp_path

        def check(self, *, funds, force=False):
            assert not force
            return {"status": "NOT_DUE", "last_checked_on": checked}

    monkeypatch.setattr(holdings, "MarketHoldingsMaintenance", Maintenance)
    preparation = MarketFundPreparation(
        root=tmp_path, clock=lambda: datetime.fromisoformat("2026-11-04T00:30:00+08:00")
    )
    result = preparation.sync_holdings([{"fund_code": "000002"}])
    assert not result["reused"] and result["funds"] == []


def test_verified_period_receipt_allows_material_processing_without_new_holdings_download():
    class CachedPreparation(Preparation):
        def sync_holdings(self, funds, **kwargs):
            return {
                "status": "NOT_DUE",
                "reused": True,
                "source_checked_at": "2026-11-02T00:31:00+08:00",
                "funds": [{"fund_code": "000002", "status": "SUCCEEDED"}],
            }

    workflow = MarketSyncWorkflow(preparation=CachedPreparation(), scope_provider=lambda: [{"fund_code": "000002"}])
    result = workflow.materials(lambda *_: None)
    assert result["status"] == "SUCCEEDED" and result["holdings_reused"]
    assert result["holdings_check"] == "NOT_DUE"
