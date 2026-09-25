"""资料同步闭环：断点、增量、版本、权限、互斥和原子发布；外部来源使用可控替身。"""

from contextlib import contextmanager
from datetime import datetime
from threading import Event
from uuid import UUID

import pytest
from app.services import fund_materials_sync as sync
from app.services.fund_exposure_common import read, save
from app.services.fund_materials_store import merge_catalog, versioned_save
from app.services.sync_jobs import LocalSyncJobManager, SyncJobInProgressError
from fastapi.testclient import TestClient
from tests.test_sync_all_jobs import wait_finished


@contextmanager
def available_lock():
    yield True


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "ROOT", tmp_path)
    monkeypatch.setattr(sync, "LIVE", tmp_path / "materials-live")
    monkeypatch.setattr(sync, "now", lambda: datetime.fromisoformat("2026-09-25T18:00:00+08:00"))
    monkeypatch.setattr(sync, "execution_lock", available_lock)
    save(tmp_path / "supplement/plan.json", {"end_date": "20260924"})
    save(tmp_path / "report-result.json", {"report_files": ["original"]})
    calls = []

    def operation(name, result):
        def run(*args, **kwargs):
            calls.append(name)
            return result

        return run

    monkeypatch.setattr(sync, "acquire", operation("reports", {"reports": 1, "errors": []}))
    monkeypatch.setattr(sync, "acquire_quotes", operation("quotes", {"errors": []}))
    monkeypatch.setattr(sync, "current_scope", lambda *a: {"windows": {"300308.SZ": []}})
    monkeypatch.setattr(sync, "update_financials", operation("financials", {"errors": [], "created": 1}))
    monkeypatch.setattr(sync, "acquire_announcements", operation("announcements", {"errors": []}))
    monkeypatch.setattr(sync, "acquire_attachments", operation("attachments", {"errors": []}))
    monkeypatch.setattr(sync, "acquire_documents", operation("documents", {"errors": []}))
    monkeypatch.setattr(sync, "supplement_peers", operation("peers", {"errors": [], "created": 1}))
    monkeypatch.setattr(sync, "prepare_training_materials", operation("training_materials", {"errors": []}))
    monkeypatch.setattr(sync, "build", operation("build", {"file": "page"}))
    return tmp_path, calls


def test_complete_run_persists_success_and_rechecks_sources_on_next_manual_run(pipeline):
    root, calls = pipeline
    result = sync.FundMaterialsSyncService().sync("002112")
    assert result["status"] == "SUCCEEDED"
    assert calls[-1] == "build"
    state = read(root / "materials-live/state.json")
    assert state["last_complete_through"] == "2026-09-25"
    assert state["last_success_at"] and len(state["stages"]) == 9
    assert read(root / "supplement/plan.json")["end_date"] == "20260924"
    calls.clear()
    assert sync.FundMaterialsSyncService().sync("002112")["status"] == "SUCCEEDED"
    assert calls == [
        "reports",
        "quotes",
        "financials",
        "announcements",
        "attachments",
        "documents",
        "peers",
        "training_materials",
        "build",
    ]


def test_partial_failure_preserves_completed_stages_and_retry_only_unfinished(pipeline, monkeypatch):
    root, calls = pipeline
    good = sync.acquire_announcements
    monkeypatch.setattr(sync, "acquire_announcements", lambda **kw: {"errors": [{"reason": "ReadTimeout"}]})
    result = sync.FundMaterialsSyncService().sync("002112")
    assert result["status"] == "PARTIAL_SUCCESS"
    assert "build" not in calls
    assert read(root / "materials-live/state.json")["last_success_at"] is None
    monkeypatch.setattr(sync, "acquire_announcements", good)
    calls.clear()
    assert sync.FundMaterialsSyncService().sync("002112")["status"] == "SUCCEEDED"
    assert calls == ["announcements", "attachments", "build"]


def test_publish_failure_keeps_checkpoint_and_retry_rebuilds_only(pipeline, monkeypatch):
    _, calls = pipeline
    good = sync.build
    monkeypatch.setattr(sync, "build", lambda **kw: (_ for _ in ()).throw(ValueError("MATERIAL_INVALID")))
    assert sync.FundMaterialsSyncService().sync("002112")["status"] == "PARTIAL_SUCCESS"
    monkeypatch.setattr(sync, "build", good)
    calls.clear()
    assert sync.FundMaterialsSyncService().sync("002112")["status"] == "SUCCEEDED"
    assert calls == ["build"]


def test_peer_partial_failure_updates_002112_page_and_retries_peer_step(pipeline, monkeypatch):
    root, calls = pipeline
    good = sync.supplement_peers
    monkeypatch.setattr(
        sync,
        "supplement_peers",
        lambda **kw: {"created": 2, "errors": [{"fund_code": "005187", "reason": "REPORT_PERIODS_MISSING"}]},
    )
    result = sync.FundMaterialsSyncService().sync("002112")
    assert result["status"] == "PARTIAL_SUCCESS"
    assert calls[-1] == "build"
    assert "详情已更新" in result["message"]
    assert read(root / "materials-live/state.json")["last_success_at"] is None
    assert read(root / "materials-live/state.json")["last_complete_through"] == "2026-09-25"
    monkeypatch.setattr(sync, "supplement_peers", good)
    calls.clear()
    assert sync.FundMaterialsSyncService().sync("002112")["status"] == "SUCCEEDED"
    assert calls == ["peers", "training_materials", "build"]


def test_real_materials_pipeline_keeps_one_click_running_until_peer_stage_finishes(pipeline, monkeypatch):
    from tests.test_sync_all_jobs import make_manager

    entered, release = Event(), Event()

    def peer_stage(**kwargs):
        entered.set()
        assert release.wait(3)
        return {"errors": [{"reason": "REPORT_PERIODS_MISSING"}]}

    monkeypatch.setattr(sync, "supplement_peers", peer_stage)
    calls = []
    manager = make_manager(calls)
    manager._materials_service_factory = sync.FundMaterialsSyncService
    try:
        parent = manager.start_all()
        assert entered.wait(3)
        assert manager.get_job(parent.job_id).status == "RUNNING"
        assert manager.get_latest_job("FUND_MATERIALS").status == "RUNNING"
        assert "MARKET_NAV_INCREMENTAL" not in calls
        with pytest.raises(SyncJobInProgressError):
            manager.start_fund_materials("002112")
        release.set()
        assert wait_finished(manager, parent.job_id).status == "PARTIAL_SUCCESS"
        assert manager.get_latest_job("FUND_MATERIALS").status == "PARTIAL_SUCCESS"
        assert "MULTI_PREDICTIONS" in calls
    finally:
        release.set()
        manager.close()


@pytest.mark.parametrize("code", [None, "", " ", "008888", "002112.OF"])
def test_scope_never_defaults_to_another_fund(code):
    with pytest.raises(ValueError):
        sync.FundMaterialsSyncService().sync(code)


def test_shared_maintenance_lock_prevents_source_calls(pipeline, monkeypatch):
    @contextmanager
    def busy():
        yield False

    monkeypatch.setattr(sync, "execution_lock", busy)
    assert sync.FundMaterialsSyncService().sync("002112")["status"] == "FAILED"
    assert pipeline[1] == []


def test_catalog_disappearance_is_preserved_as_unconfirmed_withdrawal(tmp_path):
    old = [{"id": "old", "date": "2026-09-24"}, {"id": "outside", "date": "2021-01-01"}]
    rows = merge_catalog(old, [{"id": "new", "date": "2026-09-25"}], "id", "date", [["2026-09-01", "2026-09-25"]])
    assert rows[0]["listing_status"] == "NOT_LISTED_ON_RECHECK"
    assert "listing_status" not in rows[1]
    p = tmp_path / "current.json"
    versioned_save(p, {"rows": old})
    versioned_save(p, {"rows": rows})
    assert len(list((tmp_path / "versions").glob("*.json"))) == 2
    assert read(p)["rows"] == rows


def test_manager_duplicate_and_restart_restore_interruption(tmp_path):
    entered, release = Event(), Event()

    class Service:
        def sync(self, code, *, progress_reporter):
            entered.set()
            assert release.wait(5)
            return {"status": "SUCCEEDED", "message": "完成", "created": 0, "updated": 0, "skipped": 1, "errors": []}

    path = tmp_path / "jobs.json"
    manager = LocalSyncJobManager(materials_service_factory=Service, state_path=path)
    try:
        job = manager.start_fund_materials("002112")
        assert entered.wait(3)
        with pytest.raises(SyncJobInProgressError):
            manager.start_fund_materials("002112")
        with pytest.raises(SyncJobInProgressError):
            manager.start_all()
        backup = tmp_path / "restart.json"
        backup.write_bytes(path.read_bytes())
        restored = LocalSyncJobManager(state_path=backup)
        assert restored.get_job(job.job_id).error_code == "SYNC_INTERRUPTED"
        restored.close()
        release.set()
        assert wait_finished(manager, job.job_id).status == "SUCCEEDED"
        restored = LocalSyncJobManager(state_path=path)
        assert restored.get_latest_job("FUND_MATERIALS").status == "SUCCEEDED"
        restored.close()
    finally:
        release.set()
        manager.close()


def test_internal_http_scope_and_token_and_real_queue(pipeline, monkeypatch):
    from app.api.routes import funds
    from app.core.config import get_settings
    from app.main import create_application

    manager = LocalSyncJobManager(materials_service_factory=sync.FundMaterialsSyncService)
    monkeypatch.setenv("AI_SERVICE_TOKEN", "materials-test-only")
    get_settings.cache_clear()
    monkeypatch.setattr(funds, "get_sync_job_manager", lambda: manager)
    headers = {"X-Service-Token": "materials-test-only"}
    path = "/internal/v1/funds/sync-jobs/fund-materials"
    try:
        with TestClient(create_application()) as client:
            assert client.post(path + "?fundCode=002112").status_code == 403
            for code in (None, "", "008888"):
                assert (
                    client.post(path, headers=headers, params={} if code is None else {"fundCode": code}).status_code
                    == 422
                )
            response = client.post(path, headers=headers, params={"fundCode": "002112"})
            assert response.status_code == 202
            result = wait_finished(manager, UUID(response.json()["job_id"]))
            assert result.status == "SUCCEEDED" and result.fund_codes == ("002112",)
            assert client.get(path + "/latest", headers=headers).json()["status"] == "SUCCEEDED"
    finally:
        manager.close()
        get_settings.cache_clear()


def test_no_new_attachment_never_downloads_historical_pdf(tmp_path, monkeypatch):
    from app.integrations import cninfo_exposure as module
    from app.services.direction_1d_protocol import digest

    monkeypatch.setattr(module, "SUPPLEMENT", tmp_path / "supplement")
    monkeypatch.setattr(module, "LIVE", tmp_path / "materials-live")
    save(tmp_path / "supplement/attachment-limits.json", {})
    item = {
        "announcementId": "a",
        "adjunctUrl": "a.PDF",
        "title_plain": "公告",
        "secCode": "300308",
        "announced_at_source": "2026-09-25",
        "announcementTitle": "公告",
        "adjunctSize": 1,
    }
    save(tmp_path / "materials-live/supplement/company-announcements/300308.SZ.json", {"rows": [item]})
    raw = tmp_path / "raw/a.pdf"
    raw.parent.mkdir()
    raw.write_bytes(b"%PDF-test")
    stored = {
        "title": "公告",
        "text_status": "TEXT_EXTRACTED",
        "receipt": {"url": "https://static.cninfo.com.cn/a.PDF", "file": "raw/a.pdf"},
    }
    save(tmp_path / "supplement/company-documents" / (digest("a") + ".json"), stored)

    class Client:
        def fetch(self, *a, **kw):
            raise AssertionError("unchanged historical PDF must not be downloaded")

        def close(self):
            pass

    monkeypatch.setattr(module, "PublicClient", Client)
    result = module.acquire_attachments(live=True)
    assert result["errors"] == [] and result["skipped"] == 1 and result["created"] == 0


def test_financial_empty_is_not_zero_and_successful_units_resume(tmp_path_factory, monkeypatch):
    from app.services import fund_exposure_field_enrichment as module
    from app.services.fund_exposure_supplement import FINANCIAL_APIS, OPERATING_APIS

    # Windows 临时目录保持短路径，避免夹具名称加版本哈希超过系统路径上限。
    tmp_path = tmp_path_factory.mktemp("fin")

    monkeypatch.setattr(module, "SUPPLEMENT", tmp_path / "supplement")
    monkeypatch.setattr(module, "LIVE", tmp_path / "materials-live")
    monkeypatch.setattr(module, "plan", lambda: {"end_date": "20260924"})
    save(tmp_path / "supplement/field-permission-probes.json", {"probes": []})
    calls = []

    class Provider:
        def __init__(self, **kw):
            pass

        def query(self, api, params, fields=""):
            calls.append(api)
            return {"data": {"fields": ["ts_code", "end_date", "ann_date"], "items": []}}, {"received_at": "2026-09-25"}

    monkeypatch.setattr(module, "SupplementProvider", Provider)
    old = {"ts_code": "300308.SZ", "end_date": "20260630", "ann_date": "20260820", "revenue": 123}
    save(tmp_path / "supplement/enriched-financials/300308.SZ/income.json", {"rows": [old]})
    first = module.update_financials(["300308.SZ"], "20260925", check_id="first")
    assert first["errors"] == [] and first["empty"] == len(FINANCIAL_APIS + OPERATING_APIS)
    income = read(tmp_path / "materials-live/supplement/enriched-financials/300308.SZ/income.json")
    assert income["rows"] == [old] and income["query_status"] == "SOURCE_RETURNED_EMPTY"
    calls.clear()
    repeated = module.update_financials(["300308.SZ"], "20260925", check_id="first")
    assert calls == [] and repeated["skipped"] == 10


def test_cms_index_refresh_does_not_mean_pdf_was_revised():
    from app.integrations.dbfund_supplement import same_document_catalog

    old = {"contentId": "a", "title": "公告", "url": "/a.pdf", "indexTime": "yesterday"}
    assert same_document_catalog(old, {**old, "indexTime": "today"})
    assert not same_document_catalog(old, {**old, "url": "/revised.pdf"})
    assert not same_document_catalog(old, {**old, "title": "更正公告"})


def test_financial_publication_versions_survive_and_unchanged_recheck_has_no_updates(tmp_path_factory, monkeypatch):
    from app.services import fund_exposure_field_enrichment as module

    root = tmp_path_factory.mktemp("finver")
    monkeypatch.setattr(module, "SUPPLEMENT", root / "supplement")
    monkeypatch.setattr(module, "LIVE", root / "materials-live")
    monkeypatch.setattr(module, "plan", lambda: {"end_date": "20260924"})
    save(root / "supplement/field-permission-probes.json", {"probes": []})
    fields = ["ts_code", "end_date", "ann_date", "revenue"]
    rows = [
        ["300308.SZ", "20260630", "20260820", 123],
        ["300308.SZ", "20260630", "20260821", 124],
    ]

    class Provider:
        def __init__(self, **kw):
            pass

        def query(self, api, params, fields=""):
            return {
                "data": {
                    "fields": ["ts_code", "end_date", "ann_date", "revenue"],
                    "items": rows if api == "income" else [],
                }
            }, {"received_at": "2026-09-25"}

    monkeypatch.setattr(module, "SupplementProvider", Provider)
    baseline = [dict(zip(fields, values, strict=True)) for values in rows]
    save(root / "supplement/enriched-financials/300308.SZ/income.json", {"rows": baseline})
    for check_id in ("first", "second"):
        result = module.update_financials(["300308.SZ"], "20260925", check_id=check_id)
        assert result["errors"] == []
        assert result["created"] == result["updated"] == 0
        current = read(root / "materials-live/supplement/enriched-financials/300308.SZ/income.json")
        assert current["rows"] == baseline
    # 同一来源版本真正更正时只计一条更新，旧值仍在版本目录中。
    rows[1][-1] = 125
    result = module.update_financials(["300308.SZ"], "20260925", check_id="corrected")
    assert result["updated"] == 1 and result["created"] == 0
    archives = (root / "materials-live/supplement/enriched-financials/300308.SZ/versions").glob("*.json")
    assert any(read(path)["value"].get("rows") == baseline for path in archives)


def test_newly_disclosed_report_refreshes_company_scope(monkeypatch):
    from app.services import fund_exposure_features

    seen = []

    def select(_reports, at):
        seen.append(at.date().isoformat())
        code = "600001.SH" if at.day < 25 else "300308.SZ"
        return {"report_end": "2026-06-30", "holdings": [{"stock_code": code}]}

    monkeypatch.setattr(sync, "reports", lambda: [{"source": "newly-disclosed"}])
    monkeypatch.setattr(fund_exposure_features, "select_report", select)
    scope = sync.current_scope("20260925", "2026-09-24")
    assert "2026-09-25" in seen
    assert "600001.SH" in scope["windows"] and "300308.SZ" in scope["windows"]
    assert scope["windows"]["300308.SZ"][-1][1] == "2026-09-25"
    assert scope["training_eligible"] is False


def test_page_publish_preserves_actual_dates_and_previous_snapshot_on_failure(tmp_path, monkeypatch):
    from app.services import fund_exposure_common as common
    from app.services import fund_materials_build as builder

    root = tmp_path / "research"
    save(root / "supplement/plan.json", {"fund_code": "002112", "end_date": "20260924"})
    save(root / "report-result.json", {"report_files": ["r"]})
    save(
        root / "reports/r.json",
        {
            "title": "中期报告",
            "report_end": "2026-06-30",
            "published_date": "2026-08-31",
            "full_stock_disclosure": True,
            "stock_nav_pct": "10",
            "disclosed_nav_pct": "10",
            "holdings": [
                {"stock_code": "300308.SZ", "stock_name": "公司", "nav_weight_pct": "10", "market_value_cny": "100"}
            ],
            "raw": {"sha256": "r", "url": "https://www.dbfund.com.cn/r.pdf"},
            "assets": {},
            "reported_industries": [],
        },
    )
    save(root / "stock-days/2026-09-24.json", {"rows": {"300308.SZ": {"close": 100, "pct_chg": 1}}})
    save(root / "stock-days/2026-09-25.json", {"rows": {}})
    target = tmp_path / "page.json"
    builder.build(root, target, checked_through="20260925")
    original = target.read_bytes()
    page = read(target)
    assert page["asOfDate"] == "2026-09-24"
    assert page["companies"][0]["quote"]["date"] == "2026-09-24"
    assert page["checkedThrough"] == "2026-09-25"

    def fail_replace(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(common.os, "replace", fail_replace)
    with pytest.raises(OSError):
        builder.build(root, target, checked_through="20260925")
    assert target.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
