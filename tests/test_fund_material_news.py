"""资料要点保留原文日期、基金归属和不完整状态，不取得预测资格。"""

from types import SimpleNamespace

import pytest
from app.services import fund_material_news as news
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import save


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "get_settings", lambda: SimpleNamespace(fund_insight_directory=str(tmp_path)))
    value = {
        "fund_code": "012835",
        "checked_at": "2026-10-16T10:00:00+08:00",
        "evidence_cutoff": "2026-10-15",
        "complete": False,
        "limitations": ["部分公告原件暂缺。"],
        "items": [
            {
                "event_id": "event",
                "content_hash": "a" * 64,
                "title": "公司：2026年中期报告",
                "source_name": "公司公开披露",
                "source_url": "https://example.com/report.pdf",
                "published_date": "2026-08-31",
                "first_received_at": "2026-10-16T09:00:00+08:00",
                "relation": "持仓截至2026-06-30，不代表实时持仓。",
                "business_eligible": True,
                "prediction_eligible": False,
                "training_eligible": False,
                "verification_mode": "SOURCE_AND_NUMERIC_CROSSCHECK",
                "points": ["报告期为2026年上半年，原文数字已核对。"],
            }
        ],
    }

    def publish(data):
        root = tmp_path / "012835/material-news"
        key = digest(data)
        save(root / "snapshots" / (key + ".json"), data)
        save(root / "current.json", {"hash": key}, replace=True)
        return root / "snapshots" / (key + ".json")

    return value, publish


def test_own_snapshot_is_read_without_changing_incomplete_state(snapshot):
    value, publish = snapshot
    publish(value)
    result = news.current("012835")
    assert result["items"][0]["publishedDate"] == "2026-08-31"
    assert result["items"][0]["firstReceivedAt"].startswith("2026-10-16")
    assert not result["complete"]
    assert news.current("017960")["items"] == []
    assert "verification_mode" not in result["items"][0]


@pytest.mark.parametrize("fault", ["wrong_fund", "future_date", "training", "source_url", "unverified"])
def test_invalid_qualification_or_identity_is_rejected(snapshot, fault):
    value, publish = snapshot
    if fault == "wrong_fund":
        value["fund_code"] = "017960"
    elif fault == "future_date":
        value["items"][0]["published_date"] = "2026-10-16"
    elif fault == "training":
        value["items"][0]["training_eligible"] = True
    elif fault == "source_url":
        value["items"][0]["source_url"] = "file:///private"
    else:
        value["items"][0]["verification_mode"] = "UNVERIFIED"
    publish(value)
    with pytest.raises(ValueError):
        news.current("012835")


def test_tampering_does_not_return_a_successful_empty_result(snapshot):
    value, publish = snapshot
    path = publish(value)
    save(path, {**value, "items": []}, replace=True)
    with pytest.raises(ValueError, match="SNAPSHOT_INVALID"):
        news.current("012835")
