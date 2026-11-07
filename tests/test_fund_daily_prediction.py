"""通用定时续办验收：动态范围、资料屏障、断点幂等、账号归档和真实结束条件。"""

from copy import deepcopy

import pytest
from app.services.direction_1d_protocol import digest
from app.services.fund_daily_prediction import DailyPredictionRun
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from app.services.fund_prediction_recovery import normalized_materials


class Backend:
    def __init__(self, codes=None):
        self.codes = codes or [f"{i:06d}" for i in range(1, 8)]
        self.target, self.owner = "2026-10-29", "bound-owner"
        self.submissions, self.recovers, self.saved = 0, [], set()
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

    def latest_batch(self):
        return {"job_id": "old", "status": "SUCCEEDED"}

    def start_or_reuse(self, previous):
        self.submissions += 1
        return {"job_id": "original"}

    def recover_submission(self, intent):
        return {"job_id": "original"}

    def batch_status(self, batch):
        assert batch["job_id"] == "original"
        return {"terminal": self.terminal, "prediction_outcomes": self.outcomes}

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


def test_more_than_two_and_dynamic_new_fund_and_no_duplicate_submission(tmp_path):
    b = Backend()
    reviews(tmp_path, b.codes)
    run = DailyPredictionRun(tmp_path, b)
    assert run.advance()["status"] == "CLOSED"
    assert len(b.saved) == 14
    b.codes.append("999999")
    reviews(tmp_path, ["999999"])
    assert run.advance()["status"] == "CLOSED"
    assert ("999999", "T5_V1") in b.saved
    assert b.submissions == 1


def test_no_analysis_before_all_children_terminal(tmp_path):
    b = Backend()
    b.terminal = False
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
    run = DailyPredictionRun(tmp_path, b)
    assert run.advance()["status"] == "WAITING_MATERIALS"
    assert b.guards == 1 and b.submissions == 1
    b.codes.append("999999")
    assert run.advance()["scope"] == b.codes
    assert b.guards == 1 and b.submissions == 1 and not b.recovers
    b.terminal = True
    result = run.advance()
    assert b.guards == 3  # 终态后的资料检查和正式结束检查均不能省略。
    assert "999999" in b.recovers and result["status"] == "ACTION_REQUIRED"


def test_missing_initial_followthrough_is_rebuilt_even_while_waiting(tmp_path):
    b = Backend()
    b.terminal = False
    run = DailyPredictionRun(tmp_path, b)
    run.advance()
    (tmp_path / "followthrough.json").unlink()
    assert run.advance()["status"] == "WAITING_MATERIALS"
    assert read(tmp_path / "followthrough.json")["funds"]
    assert b.submissions == 1 and not b.recovers


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
    assert b.submissions == 0


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


def test_existing_non_materials_job_must_finish_then_submit_pure_batch(tmp_path):
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
    b.terminal = False
    result = DailyPredictionRun(tmp_path, b).advance()
    assert result["status"] == "WAITING_MATERIALS" and not b.recovers
    assert b.submissions == 1 and result["preceding_jobs"][0]["job_id"] == "predecessor"
