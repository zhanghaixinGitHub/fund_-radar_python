"""通用定时续办验收：动态范围、资料屏障、断点幂等、账号归档和真实结束条件。"""

from copy import deepcopy

import pytest
from app.services.direction_1d_protocol import digest
from app.services.fund_daily_prediction import DailyPredictionRun
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from app.services.fund_prediction_recovery import normalized_materials


@pytest.fixture(autouse=True)
def legacy_policy_for_this_module(monkeypatch):
    """本文件验证旧路径；不读取正式库当前采用状态。"""
    monkeypatch.setattr("app.services.prediction_review_v2_publication.adoption_status", lambda: None)


class Backend:
    def __init__(self, codes=None):
        self.codes = codes or [f"{i:06d}" for i in range(1, 8)]
        self.target, self.owner = "2026-10-29", "bound-owner"
        self.batch_reads, self.recovers, self.saved = [], [], set()
        self.terminal, self.outcomes = True, {"status": "SUCCEEDED", "summary": {"checked": 3}}
        self.fail_code, self.freeze_failure, self.archive = None, None, True
        self.actions, self.external = set(), set()

    def context(self):
        return (
            self.target + "T00:30:00+08:00",
            self.owner,
            self.codes,
            {"target_nav_date": self.target, "base_nav_date": "2026-10-28"},
        )

    def batch_status(self, batch):
        assert batch["job_id"] == "original"
        self.batch_reads.append(batch["job_id"])
        return {"terminal": self.terminal, "prediction_outcomes": self.outcomes}

    def one_day_outcomes(self):
        return {"status": "SUCCEEDED"}

    def prediction_outcomes(self, scope):
        assert scope == self.codes
        return self.outcomes

    def guard(self, path, blocked):
        funds = [{"fund_code": c, "actions": ["fix"] if c in self.actions else [], "horizons": []} for c in self.codes]
        pending = [
            c
            for c in self.codes
            if c in self.actions
            or any((c, h) not in self.saved and (c, h) not in self.external for h in ("T1", "T5_V1"))
        ]
        value = {
            "status": "ACTION_REQUIRED" if pending else "CLOSED",
            "pending_funds": pending,
            "all_predictions_generated": all((c, h) in self.saved for c in self.codes for h in ("T1", "T5_V1")),
            "funds": funds,
        }
        versioned_save(path, value)
        return value

    def recover_materials(self, code, output, current_window):
        self.recovers.append(code)
        if code == self.fail_code:
            raise ValueError("PARSER_STILL_NEEDS_REPAIR")
        return {"status": "MATERIALS_CHECKED"}

    def approve(self, code, review):
        pass

    def periods(self, code, current_window):
        return [
            {
                "horizon": h,
                "status": "ACTION_REQUIRED"
                if code in self.actions
                else "EXTERNAL_BLOCKED"
                if (code, h) in self.external
                else "READY",
            }
            for h in ("T1", "T5_V1")
        ]

    def read_back(self, code, horizon, current_window):
        return (code, horizon) in self.saved

    def freeze(self, code, horizon, folder):
        if (code, horizon) == self.freeze_failure:
            raise ValueError("BAD_PERIOD_CALENDAR")
        folder.mkdir(parents=True, exist_ok=True)
        return {"path": str(folder / "input.json")}

    def import_and_archive(self, code, horizon, folder, owner, current_window):
        if self.archive:
            self.saved.add((code, horizon))
        return {"imported": True}


def reviews(root, codes):
    for code in codes:
        for horizon in ("T1", "T5_V1"):
            versioned_save(root / "funds" / code / horizon / "review.json", {})


def test_new_run_reads_dynamic_holdings_without_using_any_sync_batch(tmp_path):
    b = Backend()
    reviews(tmp_path, b.codes)
    run = DailyPredictionRun(tmp_path, b)
    first = run.advance()
    assert first["status"] == "CLOSED"
    assert first["supplement_run_id"] and "batch" not in first
    assert len(b.saved) == 14
    b.codes.append("999999")
    reviews(tmp_path, ["999999"])
    resumed = run.advance()
    assert resumed["status"] == "CLOSED"
    assert resumed["supplement_run_id"] == first["supplement_run_id"]
    assert ("999999", "T5_V1") in b.saved
    assert not b.batch_reads and not (tmp_path / "submission-intent.json").exists()


def legacy_run(root, backend, job_id="original"):
    versioned_save(root / "run.json", {
        "owner_user_id": backend.owner, "window": backend.context()[3], "funds": {},
        "first_failures": [], "batch": {"job_id": job_id},
    })


def test_no_analysis_before_all_children_terminal(tmp_path):
    b = Backend()
    b.terminal = False
    legacy_run(tmp_path, b)
    value = DailyPredictionRun(tmp_path, b).advance()
    assert value["status"] == "WAITING_MATERIALS"
    assert not b.recovers and not b.saved


def test_waiting_resume_polls_original_job_and_rechecks_materials_only_after_terminal(tmp_path):
    class CountingBackend(Backend):
        def __init__(self):
            super().__init__()
            self.guards = 0

        def guard(self, path, blocked):
            self.guards += 1
            return super().guard(path, blocked)

    b = CountingBackend()
    b.terminal = False
    legacy_run(tmp_path, b)
    run = DailyPredictionRun(tmp_path, b)
    assert run.advance()["status"] == "WAITING_MATERIALS"
    assert b.guards == 1 and b.batch_reads == ["original"]
    b.codes.append("999999")
    assert run.advance()["scope"] == b.codes
    assert b.guards == 1 and b.batch_reads == ["original"] * 2 and not b.recovers
    b.terminal = True
    result = run.advance()
    assert b.guards == 3  # 终态后的资料检查和正式结束检查均不能省略。
    assert "999999" in b.recovers and result["status"] == "ACTION_REQUIRED"


def test_missing_initial_followthrough_is_rebuilt_even_while_waiting(tmp_path):
    b = Backend()
    b.terminal = False
    legacy_run(tmp_path, b)
    run = DailyPredictionRun(tmp_path, b)
    run.advance()
    (tmp_path / "followthrough.json").unlink()
    assert run.advance()["status"] == "WAITING_MATERIALS"
    assert read(tmp_path / "followthrough.json")["funds"]
    assert b.batch_reads == ["original"] * 2 and not b.recovers


def test_one_parser_failure_does_not_stop_other_funds_or_become_external(tmp_path):
    b = Backend()
    b.fail_code = b.codes[0]
    reviews(tmp_path, b.codes)
    result = DailyPredictionRun(tmp_path, b).advance()
    assert result["status"] == "ACTION_REQUIRED"
    assert len(b.saved) == 12
    assert result["first_failures"][0]["reason"] == "PARSER_STILL_NEEDS_REPAIR"


def test_one_freeze_failure_keeps_other_horizon_running(tmp_path):
    b = Backend(["123456"])
    b.freeze_failure = ("123456", "T1")
    reviews(tmp_path, b.codes)
    DailyPredictionRun(tmp_path, b).advance()
    assert ("123456", "T5_V1") in b.saved and ("123456", "T1") not in b.saved


def test_one_owner_http_rejection_keeps_other_horizon_and_fund_running(tmp_path):
    """本人归档拒绝不等于其余周期失败，重试也不覆盖首次HTTP回执。"""
    import httpx

    b = Backend(["123456", "654321"])
    original = b.import_and_archive

    def archive(code, horizon, *args):
        if (code, horizon) == ("123456", "T1"):
            request = httpx.Request("POST", "http://local/owner-archive")
            httpx.Response(400, request=request).raise_for_status()
        return original(code, horizon, *args)

    b.import_and_archive = archive
    reviews(tmp_path, b.codes)
    result = DailyPredictionRun(tmp_path, b).advance()
    assert ("123456", "T1") not in b.saved
    assert {("123456", "T5_V1"), ("654321", "T1"), ("654321", "T5_V1")} <= b.saved
    failure = result["first_failures"][0]
    assert failure["fund_code"] == "123456" and failure["horizon"] == "T1"
    assert failure["type"] == "HTTPStatusError"


def test_import_without_owner_archive_is_not_complete(tmp_path):
    b = Backend(["123456"])
    b.archive = False
    reviews(tmp_path, b.codes)
    result = DailyPredictionRun(tmp_path, b).advance()
    assert result["status"] == "ACTION_REQUIRED"
    assert result["funds"]["123456"]["periods"][0]["failure"] == "OWNER_ARCHIVE_READBACK_MISSING"


def test_missing_review_stays_action_with_real_input_path(tmp_path):
    b = Backend(["123456"])
    result = DailyPredictionRun(tmp_path, b).advance()
    assert result["status"] == "ACTION_REQUIRED"
    assert result["funds"]["123456"]["periods"][0]["status"] == "READ_AND_JUDGE"
    assert not b.saved


def test_advice_runs_after_ratings_and_cannot_close_with_pending_dependencies(tmp_path):
    b = Backend(["123456"])
    reviews(tmp_path, b.codes)
    steps = []
    b.ratings = lambda terminal, state: steps.append("ratings") or {"status": "SUCCEEDED"}

    def advice(state, final):
        assert state["ratings"]["status"] == "SUCCEEDED"
        assert final["all_predictions_generated"]
        steps.append("advice")
        return {"status": "ACTION_REQUIRED", "items": []}

    b.advice = advice
    result = DailyPredictionRun(tmp_path, b).advance()
    assert steps == ["ratings", "advice"]
    assert result["status"] == "ADVICE_PENDING"
    assert read(tmp_path / "advice-status.json")["status"] == "ACTION_REQUIRED"


def test_missing_outcomes_is_not_zero_or_complete(tmp_path):
    b = Backend(["123456"])
    b.outcomes = None
    reviews(tmp_path, b.codes)
    assert DailyPredictionRun(tmp_path, b).advance()["status"] == "OUTCOME_STEP_MISSING"


def test_missed_window_is_pinned_and_owner_mismatch_rejected(tmp_path):
    b = Backend(["123456"])
    run = DailyPredictionRun(tmp_path, b)
    run.advance()
    b.target = "2026-10-30"
    assert run.advance()["status"] == "MISSED_DEADLINE"
    assert read(tmp_path / "run.json")["window"]["target_nav_date"] == "2026-10-29"
    b.owner = "different-owner"
    with pytest.raises(ValueError, match="OWNER_BINDING"):
        run.advance()


def test_external_closure_does_not_mean_all_predictions_generated(tmp_path):
    b = Backend(["123456"])
    b.external = {("123456", "T1"), ("123456", "T5_V1")}
    result = DailyPredictionRun(tmp_path, b).advance()
    assert result["status"] == "CLOSED" and not result["all_predictions_generated"]


def test_review_target_and_fund_cannot_be_reused_for_another_window(tmp_path):
    b = Backend(["123456"])
    result = DailyPredictionRun(tmp_path, b).advance(
        reviews={"123456": {"fund_code": "654321", "target_nav_date": b.target}}
    )
    assert result["status"] == "ACTION_REQUIRED"
    assert result["first_failures"][0]["reason"] == "DAILY_REVIEW_IDENTITY_OR_TARGET_MISMATCH"


def test_uncertain_submit_only_recovers_original(tmp_path):
    b = Backend(["123456"])
    versioned_save(tmp_path / "submission-intent.json", {"previous": {"job_id": "old"}})
    DailyPredictionRun(tmp_path, b).advance()
    assert not b.batch_reads


def test_nul_normalization_preserves_original_and_character_offsets(tmp_path):
    import hashlib

    original = tmp_path / "original.pdf"
    original.write_bytes(b"original immutable bytes")
    material = {
        "documents": [
            {
                "id": "d",
                "body": "收入\x00增长",
                "body_hash": digest("收入\x00增长"),
                "raw_path": str(original),
                "raw_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            }
        ],
        "facts": {"f": {"quotes": ["收入\x00增长"]}},
    }
    before = deepcopy(material)
    normalized = normalized_materials(material)
    assert material == before and original.read_bytes() == b"original immutable bytes"
    assert normalized["documents"][0]["body"] == "收入\ufffd增长"
    assert len(normalized["documents"][0]["body"]) == len(before["documents"][0]["body"])
    assert normalized["facts"]["f"]["quotes"][0] == normalized["documents"][0]["body"]
    original.write_bytes(b"changed")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        normalized_materials(material)


def test_old_saved_prediction_cannot_close_changed_materials():
    from app.services.fund_prediction_followthrough import verify_outputs

    code = "123456"
    current = {"base_nav_date": "2026-10-28", "target_nav_date": "2026-10-29"}
    record = {
        "id": "saved",
        "fund_code": code,
        "horizon": "T1",
        "base_date": current["base_nav_date"],
        "target_date": current["target_nav_date"],
        "end_date": current["target_nav_date"],
        "owner_verified": True,
        "snapshot_verified": True,
        "materials_hash": "old",
    }
    result = verify_outputs(code, current, {}, [record], [], materials_hash="new")
    assert result["horizons"][0]["status"] == "ACTION_REQUIRED"
    result = verify_outputs(code, current, {}, [record], [], materials_hash="old")
    assert result["horizons"][0]["status"] == "SAVED_AND_READ_BACK"


def test_failed_prepare_does_not_leave_empty_freeze_directory(tmp_path, monkeypatch):
    from app.services import fund_codex_analysis as analysis

    monkeypatch.setattr(analysis.repo, "clock", lambda: None)

    def missing(*args):
        raise ValueError("OWN_NAV_NOT_AVAILABLE")

    monkeypatch.setattr(analysis, "prepare", missing)
    folder = tmp_path / "freeze"
    with pytest.raises(ValueError, match="OWN_NAV_NOT_AVAILABLE"):
        analysis.freeze_packet(folder, "123456")
    assert not folder.exists()


def test_legacy_predecessor_finishes_then_continues_without_new_batch(tmp_path):
    b = Backend(["123456"])
    original_status = b.batch_status
    b.batch_status = lambda batch: (
        {"terminal": False, "existing_job_finished": True}
        if batch["job_id"] == "predecessor"
        else original_status(batch)
    )
    versioned_save(
        tmp_path / "run.json",
        {
            "owner_user_id": b.owner,
            "window": b.context()[3],
            "funds": {},
            "first_failures": [],
            "batch": {"job_id": "predecessor", "reused": True},
        },
    )
    result = DailyPredictionRun(tmp_path, b).advance()
    assert result["status"] == "ACTION_REQUIRED" and b.recovers == ["123456"]
    assert result["supplement_run_id"] and "batch" not in result
    assert not b.batch_reads and result["preceding_jobs"][0]["job_id"] == "predecessor"


def test_fund_checklist_pending_keeps_source_date_and_horizon_by_use(tmp_path, monkeypatch):
    from datetime import datetime

    from app.services import fund_daily_profiles as checklists

    monkeypatch.setattr(checklists, "now", lambda: datetime.fromisoformat("2026-11-20T17:00:00+08:00"))
    fund = "016008"
    profile = {**checklists.profiles()[fund], "name": "招商消费电子联接C"}
    root = tmp_path / "run"
    current = {"base_nav_date": "2026-11-20", "target_nav_date": "2026-11-23",
               "deadline_at": "2026-11-23T15:00:00+08:00"}
    state = {"scope": [fund], "window": current, "funds": {}}
    checklists.save_pending(root, state, records={fund: profile})
    tasks = {row["field"]: row for row in read(root / "supplement-tasks.json")["tasks"]}
    assert tasks["rating_return"]["required_date"] == "2026-10-31"
    assert "doc_id=119" in tasks["NAV"]["next_source"]
    assert "jjgg_159779.html" in tasks["prediction_target_159779"]["next_source"]
    assert "jjgg_016008.html" in tasks["rating_cost"]["next_source"]
    assert tasks["prediction_t20_v1"]["affects"] == ["T20_V1"]
    assert "caict.ac.cn" in tasks["prediction_t20_v1"]["next_source"]
    assert tasks["advice_funding"]["collection"] == "系统读回及本人确认"
    assert tasks["outcome_nav"]["required_date"] is None
    document = (tmp_path / "fund-supplements" / f"{fund}.md").read_text(encoding="utf-8")
    assert "## 预测结果核验\n\n- **□ 待核实**　到期预测" in document
    assert "## 四周期预测" in document and "931494" in document


def test_source_recheck_updates_only_the_matching_daily_checklist(tmp_path, monkeypatch):
    from datetime import datetime

    from app.services import fund_daily_profiles as checklists

    monkeypatch.setattr(checklists, "now", lambda: datetime.fromisoformat("2026-11-20T00:31:00+08:00"))
    fund = "021740"
    root = tmp_path / "run"
    current = {"base_nav_date": "2026-11-19", "target_nav_date": "2026-11-20",
               "deadline_at": "2026-11-20T15:00:00+08:00"}
    state = {"scope": [fund], "window": current, "funds": {}}
    source_review = {
        "day": "2026-11-20", "base_nav_date": "2026-11-19", "target_nav_date": "2026-11-20",
        "items": [{
            "fund_code": fund, "field": "NAV", "status": "NOT_PUBLISHED",
            "attempted_at": "2026-11-20T00:30:00+08:00", "attempted_source": "本份额原净值入口",
            "summary": "11月19日本份额同日净值尚未披露", "next_source": "基金管理人正式净值页面复查",
            "retry_at": "2026-11-20T10:15:00+08:00",
        }],
    }
    versioned_save(root / "source-rechecks.json", source_review)
    records = {fund: {"name": "黄金联接C", "template": "domestic"}}
    document = tmp_path / "fund-supplements" / f"{fund}.md"
    tasks = checklists.save_pending(root, state, records=records)
    nav = next(task for task in tasks if task["field"] == "NAV")
    assert nav["source_review"]["status"] == "NOT_PUBLISHED"
    assert nav["next_source"] == "基金管理人正式净值页面复查"
    assert nav["retry_at"] == "2026-11-20T10:15:00+08:00"
    assert "□ 待核实**　本份额正式净值" in document.read_text(encoding="utf-8")
    assert "本次取件 所需日期尚未披露" in document.read_text(encoding="utf-8")
    source_review["target_nav_date"] = "2026-11-23"
    versioned_save(root / "source-rechecks.json", source_review)
    tasks = checklists.save_pending(root, state, records=records)
    assert all("source_review" not in task for task in tasks)
    assert "本次取件" not in document.read_text(encoding="utf-8")


def test_checklist_reuses_verified_evidence_but_not_an_old_rating_alone(tmp_path, monkeypatch):
    from datetime import datetime

    from app.services import fund_daily_profiles as checklists

    when = "2026-11-21T00:31:00+08:00"
    monkeypatch.setattr(checklists, "now", lambda: datetime.fromisoformat(when))
    fund = "008888"
    root = tmp_path / "run"
    window = {"base_nav_date": "2026-11-20", "target_nav_date": "2026-11-23",
              "deadline_at": "2026-11-23T15:00:00+08:00"}
    state = {
        "scope": [fund], "window": window, "funds": {},
        "ratings": {"checklist_day": when[:10], "checklist_month_covered": [fund], "readback": [
            {"fundCode": fund, "dimensions": [{"key": "holdings", "score": "99.83", "sources": ["中报"]}]}
        ]},
    }
    base = {"fund_code": fund, "attempted_at": when, "attempted_source": "本份额正式来源",
            "summary": "核对日期、份额和口径", "evidence_ref": "材料归档回执"}
    versioned_save(root / "source-rechecks.json", {
        "day": when[:10], "base_nav_date": window["base_nav_date"], "target_nav_date": window["target_nav_date"],
        "items": [
            {**base, "field": "rating_liquidity", "status": "REUSED_VERIFIED", "required_date": "2026-10-31"},
            {**base, "field": "NAV", "status": "NEW_VERIFIED", "required_date": "2026-11-20"},
            {**base, "field": "rating_cost", "status": "REUSED_VERIFIED", "required_date": "2026-09-30"},
            {**base, "field": "rating_management", "status": "REUSED_VERIFIED", "required_date": "2026-10-31",
             "evidence_ref": ""},
        ],
    })
    tasks = checklists.save_pending(root, state, records={fund: {"name": "半导体联接C", "template": "domestic"}})
    fields = {task["field"] for task in tasks}
    assert {"rating_holdings", "rating_cost", "rating_management"} <= fields
    assert fields.isdisjoint({"rating_liquidity", "NAV"})
    document = (tmp_path / "fund-supplements" / f"{fund}.md").read_text(encoding="utf-8")
    assert "□ 待核实**　持仓结构" in document
    assert "✓ 已核实**　规模与流动性" in document and "本次核对 已核实并复用" in document
    assert "✓ 已核实**　本份额正式净值" in document and "本次核对 已核实并补齐" in document


def test_checklist_does_not_use_morning_check_time_as_retry_time(tmp_path, monkeypatch):
    from datetime import datetime

    from app.services import fund_daily_profiles as checklists

    monkeypatch.setattr(checklists, "now", lambda: datetime.fromisoformat("2026-11-21T00:31:00+08:00"))
    fund = "021740"
    state = {"scope": [fund], "window": {"base_nav_date": "2026-11-20", "target_nav_date": "2026-11-23",
                                         "deadline_at": "2026-11-23T15:00:00+08:00"}, "funds": {}}
    pending = checklists.save_pending(tmp_path / "daily-prediction-20261121", state,
                                     records={fund: {"name": "黄金联接C", "template": "domestic"}})
    assert next(task for task in pending if task["field"] == "NAV")["retry_at"] is None


def test_refresh_single_checklist_reads_latest_profile_without_entering_business_run(tmp_path, monkeypatch):
    import sys
    from datetime import datetime

    from app.commands import fund_daily_prediction as command
    from app.services import fund_daily_profiles as checklists

    at = datetime.fromisoformat("2026-11-20T00:31:00+08:00")
    records = tmp_path / ".local-runs"
    directory = records / "daily-prediction-20261120"
    fund = "021740"
    state = {"scope": [fund], "window": {"base_nav_date": "2026-11-19", "target_nav_date": "2026-11-20",
                                         "deadline_at": "2026-11-20T15:00:00+08:00"}, "funds": {}}
    versioned_save(directory / "run.json", state)
    versioned_save(directory / "fund-profiles.json", {fund: {"name": "黄金联接C", "report_source": "旧公告入口"}})
    monkeypatch.setattr(command, "PROJECT", tmp_path)
    monkeypatch.setattr(command, "now", lambda: at)
    monkeypatch.setattr(checklists, "now", lambda: at)
    monkeypatch.setattr(checklists, "profiles", lambda: {fund: {"report_source": "已核验的新公告入口"}})
    monkeypatch.setattr("app.services.prediction_trial_v2.storage_check", lambda _: None)
    monkeypatch.setattr("app.services.fund_daily_services.ensure_services", lambda _: pytest.fail("误入业务续办"))
    monkeypatch.setattr(
        sys, "argv", ["fund_daily_prediction", "--run-dir", str(directory), "--refresh-checklist", fund]
    )

    command.main()

    checklist = records / "fund-supplements" / f"{fund}.md"
    assert "已核验的新公告入口" in checklist.read_text(encoding="utf-8")
    assert read(directory / "run.json") == state
