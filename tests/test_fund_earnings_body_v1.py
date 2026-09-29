"""原件补齐必须精确承接缺件，不能把网络失败伪装成新批次重试。"""

import pytest
from app.services.fund_earnings_body_v1 import BodyContinuation, body_purpose, reconcile_pending
from app.services.fund_information_history_v1 import save


def fixture_work(tmp_path, reason="DOCUMENT_COUNT_LIMIT", positions=(2, 3)):
    rows = [{"announcementId": str(n), "title_plain": "2021年第一季度报告"} for n in range(1, 4)]
    save(tmp_path / "body-worklist.json", rows)
    save(tmp_path / "documents/1.json", {"row": rows[0], "body_saved": True})
    save(tmp_path / "documents/2.json", {"row": rows[1], "body_saved": False, "reason": reason})
    items = [{"position": n, "row": rows[n - 1]} for n in positions]
    save(tmp_path / "pending-body-worklist.json", {"count": len(items), "items": items})
    return items


def test_exact_limit_and_unattempted_continuation(tmp_path):
    expected = fixture_work(tmp_path)
    assert reconcile_pending(tmp_path) == expected


@pytest.mark.parametrize("positions", [(3, 2), (2,), (2, 2, 3), (1, 2, 3)])
def test_no_reorder_omission_duplicate_or_completed_replay(tmp_path, positions):
    fixture_work(tmp_path, positions=positions)
    with pytest.raises(ValueError, match="EXACT_RECONCILIATION"):
        reconcile_pending(tmp_path)


@pytest.mark.parametrize("reason", ["PUBLIC_READ_STOP:ReadTimeout", "INTERRUPTED_PUBLIC_REQUEST_NO_AUTORETRY"])
def test_no_network_failure_retry_by_new_batch_name(tmp_path, reason):
    fixture_work(tmp_path, reason=reason)
    with pytest.raises(ValueError, match="FAILED_SOURCE_REQUIRES"):
        reconcile_pending(tmp_path)


def test_auditor_attachment_not_report_and_correction_kept():
    assert body_purpose({"title_plain": "西部超导2021年度审计报告"})
    assert body_purpose({"title_plain": "2021年度报告更正公告"}) is None


def test_stopped_collect_refused_before_client_created(tmp_path, monkeypatch):
    batch = BodyContinuation("20260929-earnings-v7")
    batch.out = tmp_path
    save(tmp_path / "collection-stop.json", {"do_not_resume_collect": True})
    monkeypatch.setattr(batch, "check_plan", lambda: {})
    with pytest.raises(ValueError, match="ALREADY_STOPPED"):
        batch.collect()


def test_pending_metadata_change_is_rejected(tmp_path):
    fixture_work(tmp_path)
    document = tmp_path / "documents/2.json"
    document.write_text('{"row": {}, "body_saved": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="METADATA_CHANGED"):
        reconcile_pending(tmp_path)
