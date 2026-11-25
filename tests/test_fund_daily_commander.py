from copy import deepcopy
from datetime import datetime
from hashlib import sha256

import pytest
from app.services.fund_daily_commander import Commander, business_result, identity, merge_worker_receipt
from app.services.fund_daily_prediction import DailyPredictionRun
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from tests.test_fund_daily_prediction import Backend

AT = datetime.fromisoformat("2026-11-24T01:00:00+08:00")


def state(codes=None):
    return {"owner_user_id": "owner", "supplement_run_id": "same-run", "scope": codes or ["000001", "000002"],
            "checked_at": AT.isoformat(), "window": {"base_nav_date": "2026-11-23", "target_nav_date": "2026-11-24",
                                                   "deadline_at": "2026-11-24T15:00:00+08:00"}}


def audit(s, outcome="CONTINUE"):
    return {"status": "CONTINUE" if outcome == "CONTINUE" else "SUCCEEDED", "funds": {
        code: {"counts": {outcome: 1}, "remaining": [{"field": "NAV", "required_date": "2026-11-23"}]
               if outcome == "CONTINUE" else []} for code in s["scope"]}}


def receipt(s, code):
    return {"identity": identity(s), "fund_code": code,
            "ratings": {"readback": [{"fundCode": code, "status": "SCORED", "score": "70"}],
                        "checklist_month_covered": [code],
                        "newRule": {"items": [{"fundCode": code, "status": "SCORED"}]}},
            "periods": [{"horizon": h, "review_complete": True, "counts_as_dated_forecast": True}
                        for h in ("T1", "T5_V1", "T20_V1", "M6_V1")],
            "advice": {"items": [{"fund_code": code, "read_back": True, "status": "READY",
                                  "checklist_current_predictions": True, "inputs_current": True}]}}


def commander(tmp_path, s, a=None):
    return Commander(tmp_path, s, AT, audit=a or audit(s))


def live_check(c, status="SUCCEEDED", issues=None):
    versioned_save(c.directory / "commander-live-check.json", {
        "commander_identity": c.binding,
        "commander_fingerprint": c.snapshot()["verification_fingerprint"],
        "status": status, "issues": issues or [],
    })


def test_failed_acceptance_returns_same_fund_and_finish_is_rejected(tmp_path):
    s = state()
    c = commander(tmp_path, s)
    c.change("dispatch", code="000001", worker="original-worker")
    result = c.change("accept", code="000001")
    assert result["funds"][0]["action"] == "RETURN_TO_WORKER"
    assert result["funds"][0]["remaining"][0]["field"] == "NAV"
    assert c.change("finish")["can_finish"] is False
    assert read(tmp_path / "commander.json")["events"][-1]["action"] == "FINISH_REJECTED"
    restored = commander(tmp_path, s)
    assert restored.snapshot()["funds"][0]["action"] == "RETURN_TO_WORKER"
    assert restored.snapshot()["funds"][1]["action"] == "DISPATCH"


def test_concurrency_duplicate_worker_and_release_require_reason(tmp_path):
    s = state([f"{n:06d}" for n in range(1, 7)])
    c = commander(tmp_path, s)
    for n in range(1, 6):
        c.change("dispatch", code=f"{n:06d}", worker=f"worker-{n}")
    with pytest.raises(ValueError, match="SUPPLEMENT_LIMIT"):
        c.change("dispatch", code="000006", worker="worker-6")
    with pytest.raises(ValueError, match="DUPLICATE"):
        c.change("dispatch", code="000006", worker="worker-1")
    with pytest.raises(ValueError, match="REASON_REQUIRED"):
        c.change("release", code="000001")
    c.change("release", code="000001", reason="已确认原窗口结束，交回原基金未完项")
    c.change("dispatch", code="000006", worker="worker-6")
    c.change("checkpoint", reason="保存当前进度，剩余原队列未完成")
    assert read(tmp_path / "commander.json")["status"] == "INTERRUPTED"


def test_identity_changes_and_expired_window_cannot_be_relabelled(tmp_path):
    s = state()
    c = commander(tmp_path, s)
    c.change("checkpoint", reason="测试恢复")
    changed = deepcopy(s)
    changed["scope"] = ["000001"]
    with pytest.raises(ValueError, match="IDENTITY_CHANGED"):
        commander(tmp_path, changed)
    expired = Commander(tmp_path, s, AT.replace(hour=16), audit=audit(s))
    assert expired.snapshot()["status"] == "MISSED_DEADLINE"
    with pytest.raises(ValueError, match="WINDOW_EXPIRED"):
        expired.change("dispatch", code="000001", worker="worker")


def test_each_fund_must_be_accepted_and_changed_inputs_invalidate_acceptance(tmp_path):
    s = state()
    for code in s["scope"]:
        versioned_save(tmp_path / "fund-business" / code / "business-status.json", receipt(s, code))
    c = commander(tmp_path, s, audit(s, "REUSED_VERIFIED"))
    assert not c.snapshot()["can_finish"]
    for code in s["scope"]:
        c.change("accept", code=code)
    assert not c.change("finish")["can_finish"]
    assert c.snapshot()["ready_for_final_check"]
    live_check(c)
    assert c.change("finish")["status"] == "SUCCEEDED"
    updated = receipt(s, "000001")
    updated["ratings"]["newRule"]["items"][0]["status"] = "WAITING_DATA"
    versioned_save(tmp_path / "fund-business/000001/business-status.json", updated)
    assert c.snapshot()["funds"][0]["action"] == "VERIFY_RETURN"
    assert not c.change("finish")["can_finish"]


def test_old_scores_prepared_inputs_and_missing_advice_do_not_count_as_complete():
    s = state()
    r = receipt(s, "000001")
    r["ratings"]["newRule"]["items"][0].update(status="WAITING_DATA", blockers=[{"state": "TASK_LIMIT"}])
    r["periods"][0].update(review_complete=False, counts_as_dated_forecast=False)
    r["advice"]["items"][0]["inputs_current"] = False
    b = business_result(r, s, "000001")
    assert not b["rating_complete"] and not b["reviews_saved"] and not b["advice_complete"]
    r["identity"]["job_id"] = "old-job"
    assert not business_result(r, s, "000001")["receipt_bound"]


def test_actual_abstention_can_be_reported_without_inventing_direction(tmp_path):
    s = state(["000001"])
    r = receipt(s, "000001")
    for period in r["periods"]:
        period.update(status="REVIEWED_ABSTAIN", counts_as_dated_forecast=False,
                      source_reasons=["主因与反证无法确定净方向"], data_blocked=False)
    r["advice"]["items"][0].update(status="INSUFFICIENT", missing=["五日无明确方向"])
    versioned_save(tmp_path / "fund-business/000001/business-status.json", r)
    c = commander(tmp_path, s, audit(s, "REUSED_VERIFIED"))
    c.change("accept", code="000001")
    live_check(c, "INCOMPLETE", ["ADVICE_INPUTS_PENDING"])
    result = c.change("finish")
    assert result["status"] == "DONE_WITH_GAPS"
    assert result["predictions_complete"] == result["advice_complete"] == 0


def test_initialization_does_not_start_global_collection_scoring_or_prediction(tmp_path):
    backend = Backend()
    result = DailyPredictionRun(tmp_path, backend).advance(initialize_only=True)
    assert result["status"] == "READY_FOR_DISPATCH"
    assert result["scope"] == backend.codes
    assert backend.recovers == backend.batch_reads == [] and not backend.saved


def test_business_dependencies_do_not_block_entering_single_fund_business(tmp_path):
    s = state(["000001"])
    a = {"status": "CONTINUE", "funds": {"000001": {"counts": {"CONTINUE": 1}, "remaining": [
        {"field": "advice_predictions", "outcome": "CONTINUE"}]}}}
    c = commander(tmp_path, s, a)
    assert c.change("accept", code="000001")["funds"][0]["action"] == "RUN_FUND_BUSINESS"


def test_initialized_ledger_is_auditable_and_evidence_change_is_returned(tmp_path, monkeypatch):
    from app.commands import fund_daily_check
    from app.services import material_content
    from app.services.fund_daily_commander import initialize_material_ledger

    s = state(["000001"])
    task = {"fund_code": "000001", "field": "NAV", "required_date": "2026-11-23"}
    monkeypatch.setattr(fund_daily_check, "PROJECT", tmp_path)
    monkeypatch.setattr(material_content, "source_bytes", lambda path: (
        b"cached parsed text" if path.name == "official.json" else path.read_bytes()
    ))

    class Supplements:
        previous = {}

        def prepare(self, current):
            versioned_save(tmp_path / "supplement-catalog.json", {
                "job_id": "same-run", "window": s["window"], "tasks": [task]})

    initialize_material_ledger(tmp_path, s, Supplements())
    assert fund_daily_check.read_supplement_work(tmp_path, s, AT)["counts"] == {"CONTINUE": 1}
    proof = tmp_path / "official.json"
    proof.write_text('{"nav": 1.1}', encoding="utf-8")
    package = {"identity": identity(s), "fund_code": "000001", "reviewed_by": "commander", "items": [{
        **task, "status": "NEW_VERIFIED", "summary": "同份额同日净值已核", "attempted_source": "官方净值",
        "attempted_at": AT.isoformat(), "next_source": "后续净值版本变化后复核", "observations": {"nav": 1.1},
        "evidence_ref": str(proof), "evidence_sha256": sha256(proof.read_bytes()).hexdigest()}]}
    merge_worker_receipt(tmp_path, s, package, project=tmp_path)
    assert fund_daily_check.read_supplement_work(tmp_path, s, AT)["counts"] == {"NEW_VERIFIED": 1}
    proof.write_text('{"nav": 9.9}', encoding="utf-8")
    assert fund_daily_check.read_supplement_work(tmp_path, s, AT)["counts"] == {"CONTINUE": 1}


def test_single_fund_aggregation_preserves_original_results_for_other_funds(tmp_path):
    from app.commands.fund_daily_fund import merge_fund_result, supplement_input

    s = state()
    s["advice"] = {"items": [{"fund_code": code, "report_id": f"old-{code}"} for code in s["scope"]]}
    s["ratings"] = {"readback": [{"fundCode": code, "score": 62} for code in s["scope"]]}
    for code in s["scope"]:
        updated = receipt(s, code)
        updated["advice"]["items"][0]["report_id"] = f"new-{code}"
        versioned_save(tmp_path / "fund-business" / code / "business-status.json", updated)
    ledger = tmp_path / "supplement-funds/000001.json"
    versioned_save(ledger, {"items": ["original"]})
    original_input = supplement_input(tmp_path, "000001")
    a = read(tmp_path / "fund-business/000001/business-status.json")
    # 模拟评分中另一只补件与建议汇总已更新，还有一份独立回执尚未合入。
    s["supplements"] = {"pending_item_count": 7}
    s["advice"]["items"][1]["report_id"] = "current-000002"
    versioned_save(tmp_path / "run.json", s)

    result = merge_fund_result(tmp_path, a, original_input)
    assert result["supplements"]["pending_item_count"] == 7
    assert {row["report_id"] for row in result["advice"]["items"]} == {"new-000001", "current-000002"}
    assert {"fundCode": "000002", "score": 62} in result["ratings"]["readback"]
    assert result["advice"]["status"] == "PARTIAL_SUCCESS"
    versioned_save(ledger, {"items": ["own-input-changed"]})
    with pytest.raises(ValueError, match="INPUT_CHANGED_BEFORE_MERGE"):
        merge_fund_result(tmp_path, a, original_input)
    assert read(tmp_path / "run.json") == result
    rejected = read(tmp_path / "fund-business/000001/business-status.json")
    assert not business_result(rejected, s, "000001")["receipt_bound"]


def test_worker_merge_preserves_other_funds_and_rejects_self_report_or_wrong_dates(tmp_path):
    s = state()
    tasks = [{"fund_code": c, "field": "NAV", "required_date": "2026-11-23", "next_source": "官方净值"}
             for c in s["scope"]]
    versioned_save(tmp_path / "supplement-catalog.json", {"job_id": "same-run", "window": s["window"], "tasks": tasks})
    old = [{**row, "status": "NOT_ATTEMPTED", "summary": "未派工"} for row in tasks]
    versioned_save(tmp_path / "source-rechecks.json", {"job_id": "same-run", "items": old})
    proof = tmp_path / "official.json"
    proof.write_text('{"date":"2026-11-23","fund":"000001","nav":"1.1"}', encoding="utf-8")
    row = {**tasks[0], "status": "NEW_VERIFIED", "summary": "已核本份额同日正式净值",
           "attempted_source": "官方来源", "attempted_at": AT.isoformat(), "observations": {"nav": "1.1"},
           "evidence_ref": str(proof), "evidence_sha256": sha256(proof.read_bytes()).hexdigest()}
    package = {"identity": identity(s), "fund_code": "000001", "reviewed_by": "commander", "items": [row]}
    invalid = deepcopy(package)
    invalid["items"][0]["attempted_at"] = "2026-11-23T01:00:00+08:00"
    with pytest.raises(ValueError, match="ITEM_DATE"):
        merge_worker_receipt(tmp_path, s, invalid, project=tmp_path)
    invalid = deepcopy(package)
    invalid["items"][0]["evidence_sha256"] = "wrong"
    with pytest.raises(ValueError, match="EVIDENCE_HASH"):
        merge_worker_receipt(tmp_path, s, invalid, project=tmp_path)
    result = merge_worker_receipt(tmp_path, s, package, project=tmp_path)
    assert result["item_count"] == 2 and result["pending_item_count"] == 1
    assert read(tmp_path / "supplement-funds/000002.json")["items"] == [old[1]]
    assert read(tmp_path / "run.json")["scope"] == s["scope"]


def test_single_fund_business_uses_one_fund_and_does_not_publish_preparation(tmp_path, monkeypatch):
    from app.commands.fund_daily_fund import advance_fund
    from app.services import fund_daily_advice

    s = state()
    calls = []

    class SingleBackend:
        def context(self):
            return s["checked_at"], s["owner_user_id"], s["scope"], s["window"]

        def prepare_monthly(self, codes):
            assert codes == ["000001"]
            return {}

        def ratings(self, terminal, scoped):
            assert scoped["scope"] == ["000001"] and s["scope"] == ["000001", "000002"]
            return receipt(s, "000001")["ratings"]

        def periods(self, code, window):
            return [{"horizon": h, "status": "READY", "review_complete": False} for h in sorted(
                {"T1", "T5_V1", "T20_V1", "M6_V1"})]

        def read_back(self, *args):
            return False

        def freeze(self, code, horizon, folder):
            calls.append(horizon)
            if horizon == "M6_V1":
                raise ValueError("ONE_PERIOD_NEEDS_SOURCE_REPAIR")
            return {"generationStatus": "WAITING_ANALYSIS", "evaluation": None}

        def import_and_archive(self, *args):
            pytest.fail("只有准备记录不能发布")

        def advice(self, scoped, periods):
            assert scoped["scope"] == ["000001"]
            return {"items": [{"fund_code": "000001", "read_back": True, "status": "WAITING_INPUTS"}]}

    monkeypatch.setattr(fund_daily_advice, "read_saved_advice", lambda *a, **kw: [
        {"fund_code": "000001", "read_back": True, "inputs_current": True}])
    result = advance_fund(tmp_path, s, "000001", SingleBackend(), publish=True)
    assert len(calls) == 4 and len(result["prepared"]) == 3
    assert result["failures"][0]["horizon"] == "M6_V1"
    assert not business_result(result, s, "000001")["predictions_complete"]
    assert read(tmp_path / "fund-business/000001/business-status.json")["identity"]["scope"] == s["scope"]


@pytest.mark.parametrize("advice_current", [True, False])
def test_live_verification_reads_current_business_and_rejects_changed_advice(tmp_path, monkeypatch, advice_current):
    from app.commands import fund_daily_check
    from app.commands import fund_daily_commander as command
    from app.services import fund_daily_advice
    from app.services.direction_1d_protocol import digest
    from app.services.fund_daily_commander import initialize_material_ledger

    s = state(["000001"])
    s.update(scope_hash=digest(s["scope"]), workflow_version="HOLDINGS_END_TO_END_V1")
    monkeypatch.setattr(fund_daily_check, "PROJECT", tmp_path)
    monkeypatch.setattr(command, "now", lambda: AT)

    class Supplements:
        previous = {}

        def prepare(self, current):
            versioned_save(tmp_path / "supplement-catalog.json", {"job_id": "same-run", "window": s["window"],
                "tasks": [{"fund_code": "000001", "field": "NAV", "required_date": "2026-11-23"}]})

    initialize_material_ledger(tmp_path, s, Supplements())
    proof = tmp_path / "official.json"
    proof.write_text('{"nav": 1.1}', encoding="utf-8")
    merge_worker_receipt(tmp_path, s, {"identity": identity(s), "fund_code": "000001", "reviewed_by": "commander",
        "items": [{"fund_code": "000001", "field": "NAV", "required_date": "2026-11-23", "status": "NEW_VERIFIED",
                   "summary": "正式净值", "attempted_source": "官方", "attempted_at": AT.isoformat(),
                   "next_source": "版本变化后重查", "observations": {"nav": 1.1}, "evidence_ref": str(proof),
                   "evidence_sha256": sha256(proof.read_bytes()).hexdigest()}]}, project=tmp_path)
    r = receipt(s, "000001")
    r["ratings"]["readback"][0]["asOfDate"] = "2026-10-31"
    versioned_save(tmp_path / "fund-business/000001/business-status.json", r)
    counts = dict(total=0, completed=0, checked=0, created=0, pending=0, failed=0, remaining=0)

    class LiveBackend:
        def context(self):
            return AT.isoformat(), "owner", s["scope"], s["window"]

        def get(self, url):
            return r["ratings"]["readback"][0]

        def periods(self, code, window):
            return r["periods"]

        def prediction_outcomes(self, scope):
            return {"status": "SUCCEEDED", "summary": counts}

        def one_day_outcomes(self, **kwargs):
            return {"status": "SUCCEEDED", "summary": counts, "scope": s["scope"], "owner_user_id": "owner",
                    "dueBefore": "2026-11-24", "checked_at": AT.isoformat()}

        def one_day_outcome_status(self):
            return counts

        def guard(self, output, blocked):
            return {"analysis_version": "COMPREHENSIVE_REVIEW_V2", "status": "CLOSED", "all_reviews_saved": True,
                    "owner_user_id": "owner", "window": s["window"]}

    monkeypatch.setattr(fund_daily_advice, "read_saved_advice", lambda *a, **kw: [{
        "fund_code": "000001", "read_back": True, "inputs_current": advice_current}])
    command.verify_live(tmp_path, s, LiveBackend())
    c = Commander(tmp_path, s, AT)
    c.change("accept", code="000001")
    result = c.change("finish")
    assert result["can_finish"] is advice_current
    assert read(tmp_path / "run.json")["scope"] == ["000001"]
    check = read(tmp_path / "commander-live-check.json")
    assert (check["status"] == "SUCCEEDED") is advice_current
    if not advice_current:
        assert "ADVICE_INPUTS_CHANGED_OR_UNVERIFIED" in check["issues"]
