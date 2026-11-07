"""运行适配的真实边界：旧进程契约、来源预算、跨基金拒绝和逐字原件校验。"""

from datetime import datetime
from types import SimpleNamespace

import pytest
from app.services import fund_prediction_recovery as recovery
from app.services.fund_daily_runtime import LocalDailyBackend
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save


def test_one_day_clock_skew_retries_same_owner_target_once_and_retains_first(tmp_path):
    """只恢复真实的新鲜度拒绝，不调整日期、容差、账号或预测内容。"""
    import json

    import httpx

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(400, json={"message": "CLOCK_SKEW"})
        return httpx.Response(200, json={"status": "PREDICTED"})

    b = object.__new__(LocalDailyBackend)
    b.settings = SimpleNamespace(core_service_base_url="http://local")
    with httpx.Client(transport=httpx.MockTransport(handler)) as b.client:
        assert b.archive_one_day("123456", "bound-owner", "2026-11-03", tmp_path) == {"status": "PREDICTED"}
    assert requests == [{"targetNavDate": "2026-11-03", "ownerUserId": "bound-owner"}] * 2
    assert read(tmp_path / "owner-archive-first-failure.json")["reason"] == "CLOCK_SKEW"


@pytest.mark.parametrize("reason,count", [("CLOCK_SKEW", 2), ("SOURCE_IDENTITY_CONFLICT", 1)])
def test_one_day_archive_retry_is_bounded_and_does_not_retry_identity_rejection(tmp_path, reason, count):
    import httpx

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(400, json={"message": reason})

    b = object.__new__(LocalDailyBackend)
    b.settings = SimpleNamespace(core_service_base_url="http://local")
    with httpx.Client(transport=httpx.MockTransport(handler)) as b.client:
        with pytest.raises(httpx.HTTPStatusError):
            b.archive_one_day("123456", "bound-owner", "2026-11-03", tmp_path)
    assert len(requests) == count


def test_uncertain_submission_does_not_submit_again():
    b = object.__new__(LocalDailyBackend)
    b.latest_batch = lambda: {"job_id": "new", "status": "RUNNING"}
    assert b.recover_submission({"previous": {"job_id": "old"}})["job_id"] == "new"
    b.latest_batch = lambda: {"job_id": "old", "status": "SUCCEEDED"}
    with pytest.raises(ValueError, match="UNCERTAIN_SUBMISSION"):
        b.recover_submission({"previous": {"job_id": "old"}})


def test_approval_cannot_select_old_package_after_new_original_was_collected(tmp_path, monkeypatch):
    """新原件和旧包不能分走两条成功链；缺该原件即在正式用途批准前拒绝。"""
    from app.services import fund_market_review

    b = object.__new__(LocalDailyBackend)
    b.directory = tmp_path
    versioned_save(tmp_path / "funds/123456/companies/8035.JP/jp-financials/latest.json", {
        "status": "ORIGINAL_PARSED_READING_REQUIRED", "raw_sha256": "new-original",
    })
    selected = tmp_path / "selected.json"
    versioned_save(selected, {"fund_code": "123456", "documents": [{"raw_sha256": "old-original"}]})
    approved = []
    monkeypatch.setattr(fund_market_review, "approve_comprehensive", lambda *a, **k: approved.append(a))
    with pytest.raises(ValueError, match="CURRENT_JP_FINANCIAL_ORIGINAL_NOT_IN_REVIEWED_MATERIALS"):
        b.approve("123456", {"materials_path": str(selected), "note": "本期实际阅读说明"})
    assert approved == []


def test_old_service_without_materials_only_contract_never_submits():
    calls = []

    class Client:
        def post(self, url, *, json):
            calls.append(json)
            return SimpleNamespace(status_code=422, json=lambda: {"detail": [{"loc": ["body", "otherField"]}]})

    b = object.__new__(LocalDailyBackend)
    b.client, b.base = Client(), "http://127.0.0.1:8000"
    with pytest.raises(ValueError, match="CONTRACT_MISSING"):
        b.start_or_reuse({"status": "SUCCEEDED"})
    assert len(calls) == 1 and calls[0]["materialsOnly"] is not True


def test_new_holding_uses_own_report_and_actual_quote_without_fake_financials(tmp_path, monkeypatch):
    from app.services import fund_market_preparation as preparation

    at = datetime.fromisoformat("2026-10-29T00:40:00+08:00")
    code, existing, new = "123456", "600001.SH", "300308.SZ"
    material = {
        "fund_code": code,
        "documents": [],
        "companies": [
            {
                "code": existing,
                "name": "既有证券",
                "financials": {"period": "2026-06-30"},
                "quote": {"date": "2026-10-27"},
            }
        ],
    }
    source = tmp_path / "old.json"
    versioned_save(source, material)
    receipt = tmp_path / "handoff.json"
    versioned_save(receipt, {"materials_path": str(source)})
    versioned_save(
        tmp_path / "news/funds" / (code + ".json"), {"status": "SUCCEEDED", "coverage": {"end": "2026-10-29"}}
    )
    monkeypatch.setattr(recovery, "now", lambda: at)
    monkeypatch.setattr(recovery, "workflow_root", lambda: tmp_path)
    monkeypatch.setattr(recovery, "receipt_path", lambda _: receipt)
    monkeypatch.setattr(recovery, "replay_originals", lambda *args: [])
    monkeypatch.setattr(preparation, "market_scope", lambda: [{"fund_code": code, "fund_name": "本基金"}])
    monkeypatch.setattr(
        recovery,
        "snapshot",
        lambda _: {
            "reports": [
                {
                    "id": "own-new",
                    "endDate": "2026-09-30",
                    "publishedDate": "2026-10-28",
                    "holdings": [{"stockCode": new, "weightPct": 1.2}],
                }
            ]
        },
    )
    calls = []
    monkeypatch.setattr(recovery, "public_financial_originals", lambda security, *args: calls.append(security) or [])
    monkeypatch.setattr(recovery, "get_settings", lambda: SimpleNamespace(tushare_api_url="https://api.tushare.pro"))

    class Collector:
        def company(self, security):
            return {
                "stockCode": security,
                "quote": {"date": "2026-10-28", "close": 11, "pre_close": 10, "pct_chg": 10},
                "financials": {},
                "sources": [
                    {"api": "daily", "file": "original.json", "sha256": "a" * 64, "received_at": at.isoformat()}
                ],
                "gaps": ["financial permission missing"],
            }

    output = tmp_path / "run"
    value = recovery.recover(code, output, {"base_nav_date": "2026-10-28"}, collector=Collector())
    assert set(value["securities"]) == {existing, new} and calls == [new]
    candidate = read(__import__("pathlib").Path(value["materials_path"]))
    assert candidate["companies"][0]["quote"]["date"] == "2026-10-28"
    assert candidate["companies"][0]["financials"] == material["companies"][0]["financials"]
    assert read(source) == material  # 原包保留，没有把新增季度仓位拼到旧全仓。
    assert candidate.get("report") is None


def test_cross_fund_materials_rejected_before_collection(tmp_path, monkeypatch):
    from app.services import fund_market_preparation as preparation

    code = "123456"
    source, receipt = tmp_path / "source.json", tmp_path / "receipt.json"
    versioned_save(source, {"fund_code": "654321"})
    versioned_save(receipt, {"materials_path": str(source)})
    versioned_save(
        tmp_path / "news/funds" / (code + ".json"), {"status": "SUCCEEDED", "coverage": {"end": "2999-12-31"}}
    )
    monkeypatch.setattr(recovery, "workflow_root", lambda: tmp_path)
    monkeypatch.setattr(recovery, "receipt_path", lambda _: receipt)
    monkeypatch.setattr(recovery, "replay_originals", lambda *args: [])
    monkeypatch.setattr(preparation, "market_scope", lambda: [{"fund_code": code}])
    with pytest.raises(ValueError, match="CROSS_FUND"):
        recovery.recover(code, tmp_path / "run", {"base_nav_date": "2026-10-28"})


def test_external_failure_reopens_when_real_input_changes(tmp_path, monkeypatch):
    import json

    from app.services import fund_daily_runtime as runtime

    monkeypatch.setattr(runtime, "workflow_root", lambda: tmp_path)
    material = tmp_path / "materials/123456.json"
    versioned_save(material, {"source_hash": "original-source"})
    blocked = tmp_path / "blocked.json"
    blocked.write_text(
        json.dumps([{"fund_code": "123456", "horizon": "T1", "reason": "SOURCE_UNAVAILABLE"}]), encoding="utf-8"
    )
    b = object.__new__(LocalDailyBackend)
    b.directory = tmp_path / "run"
    first = b.current_blocks(blocked)
    assert len(json.loads(first.read_text(encoding="utf-8"))) == 1
    versioned_save(material, {"source_hash": "new-official-original"})
    second = b.current_blocks(blocked)
    assert json.loads(second.read_text(encoding="utf-8")) == []
    assert json.loads(blocked.read_text(encoding="utf-8"))[0]["reason"] == "SOURCE_UNAVAILABLE"


def test_run_mutex_is_shared_and_released(tmp_path, monkeypatch):
    from app.commands import fund_daily_prediction as command

    monkeypatch.setattr(command, "PROJECT", tmp_path)
    with command.run_lock():
        with pytest.raises(ValueError, match="ALREADY_ACTIVE"):
            with command.run_lock():
                pass
    with command.run_lock():
        pass


def test_legacy_jsonb_negative_zero_requires_exact_original_hash(tmp_path):
    from copy import deepcopy

    from app.services.direction_1d_protocol import digest
    from app.services.fund_prediction_snapshot_compat import checked_snapshot

    material = {"fund_code": "123456", "companies": [{"quote": {"change_pct": -0.0}}]}
    path = tmp_path / "material.json"
    versioned_save(path, material)
    original = {**material, "materials_path": str(path), "materials_hash": digest(material)}
    stored = deepcopy(original)
    stored["companies"][0]["quote"]["change_pct"] = 0.0
    assert digest(stored) != digest(original)
    restored = checked_snapshot(stored, digest(original))
    assert digest(restored) == digest(original)
    assert digest(stored) != digest(original)  # 没有改写传入数据库对象。
    stored["fund_code"] = "654321"
    with pytest.raises(ValueError, match="MATERIALS_MISMATCH"):
        checked_snapshot(stored, digest(original))


def test_only_purpose_review_metadata_is_excluded_from_economic_input_identity():
    from app.services.fund_prediction_snapshot_compat import materials_input_identity

    original = {
        "fund_code": "123456",
        "workflow_source_hash": "source",
        "companies": [{"close": 10}],
        "workflow_review_note": "first note",
        "workflow_reviewed_at": "first-time",
    }
    renewed = {**original, "workflow_review_note": "new note", "workflow_reviewed_at": "new-time"}
    assert materials_input_identity(original) == materials_input_identity(renewed)
    assert materials_input_identity(original) != materials_input_identity({**renewed, "companies": [{"close": 11}]})
    assert materials_input_identity(original) != materials_input_identity(
        {**renewed, "workflow_source_hash": "new-source"}
    )


def test_missing_services_start_hidden_and_running_services_are_reused(tmp_path, monkeypatch):
    from app.services import fund_daily_services as services
    from pydantic import SecretStr

    python, java = tmp_path / "python-project", tmp_path / "java-project"
    executable = python / ".venv/Scripts/python.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    jdk = java / ".tools/jdk17/configured/bin/java.exe"
    jdk.parent.mkdir(parents=True)
    jdk.touch()
    monkeypatch.setattr(services, "PYTHON", python)
    monkeypatch.setattr(services, "JAVA", java)
    monkeypatch.setattr(
        services,
        "get_settings",
        lambda: SimpleNamespace(core_service_base_url="http://127.0.0.1:8080", ai_service_token=SecretStr("test-only")),
    )
    active, launches = set(), []
    monkeypatch.setattr(services, "owned_process", lambda port, _: {"ProcessId": port} if port in active else None)

    def powershell(script, *, capture=True):
        if "Start-Process" in script:
            assert "-WindowStyle Hidden" in script and "-RedirectStandardOutput" in script
            assert capture is False
            launches.append(script)
            active.add(8080 if f"-WorkingDirectory '{java}'" in script else 8000)
            return {"Id": 1}
        assert "Get-CimInstance Win32_Process" in script and "$_.Name -match" in script
        return None

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, _):
            return SimpleNamespace(status_code=200, json=lambda: {"status": "UP"})

    monkeypatch.setattr(services, "powershell", powershell)
    monkeypatch.setattr(services.httpx, "Client", Client)
    first = services.ensure_services(tmp_path / "run")
    assert len(launches) == 2 and all(r["started"] and r["ready"] for r in first)
    second = services.ensure_services(tmp_path / "run")
    assert len(launches) == 2 and all(not r["started"] and r["ready"] for r in second)


def test_missing_port_is_a_successful_empty_probe_and_foreign_owner_still_rejected(monkeypatch):
    from pathlib import Path

    from app.services import fund_daily_services as services

    calls = []

    def query(script):
        calls.append(script)
        assert "-LocalPort" not in script
        assert "Get-NetTCPConnection -State Listen -ErrorAction Stop" in script
        assert "Where-Object { $_.LocalPort -eq 8000 }" in script
        return None

    monkeypatch.setattr(services, "powershell", query)
    assert services.owned_process(8000, Path("C:/expected")) is None
    assert len(calls) == 1
    monkeypatch.setattr(
        services,
        "powershell",
        lambda _: {"ProcessId": 9, "ExecutablePath": "C:/other/python.exe", "CommandLine": "python -m other"},
    )
    with pytest.raises(ValueError, match="PORT_OWNED_BY_OTHER_PROJECT"):
        services.owned_process(8000, Path("C:/expected"))
