"""验证旧停止来源在窗口、公告身份和缓存入口保持有效。"""

import pytest
from app.services.fund_earnings_batch_v5 import EarningsBatchV5, document_stop, window_stop_collisions
from app.services.fund_information_history_v1 import save


def entry():
    return {
        "stock": "688122",
        "published_date": "2023-03-25",
        "document_id": "old",
        "url": "https://static.cninfo.com.cn/finalpage/old.PDF",
    }


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ("2023-03-25", "2023-03-25", 1),
        ("2023-03-01", "2023-03-25", 1),
        ("2023-03-25", "2023-04-01", 1),
        ("2023-03-26", "2023-04-01", 0),
    ],
)
def test_public_day_boundary_keeps_stop(start, end, expected):
    assert len(window_stop_collisions([{"stock": "688122", "start": start, "end": end}], [entry()])) == expected


def test_other_company_same_day_is_independent():
    assert window_stop_collisions([{"stock": "002027", "start": "2023-03-01", "end": "2023-04-01"}], [entry()]) == []


@pytest.mark.parametrize("key,url", [("old", "renamed.PDF"), ("new", "finalpage/old.PDF")])
def test_either_identity_or_url_preserves_stop(key, url):
    assert document_stop({"announcementId": key, "adjunctUrl": url}, [entry()])


def test_independent_original_is_not_blocked():
    assert document_stop({"announcementId": "new", "adjunctUrl": "new.PDF"}, [entry()]) == []


def test_completed_batch_cannot_resume_network(tmp_path, monkeypatch):
    b = EarningsBatchV5("20260929-earnings-v19")
    b.out = tmp_path
    save(tmp_path / "collection-stop.json", {"do_not_resume_collect": True})
    monkeypatch.setattr(b, "check_plan", lambda: {})
    with pytest.raises(ValueError, match="BATCH_STOPPED"):
        b.collect()


def test_resume_cannot_change_predecessor(tmp_path, monkeypatch):
    b = EarningsBatchV5("20260929-earnings-v19")
    b.out = tmp_path
    save(tmp_path / "plan.json", {})
    monkeypatch.setattr(b, "check_plan", lambda: {"previous_run": "C:/data/20260929-earnings-v18", "windows": [1]})
    assert b.prepare("20260929-earnings-v18")["reused_frozen_plan"]
    with pytest.raises(ValueError, match="PREDECESSOR_CHANGED"):
        b.prepare("20260929-earnings-v17")
