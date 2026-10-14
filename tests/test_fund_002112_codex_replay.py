"""历史盲测的时间边界和统一揭晓门禁；使用小替身，不读真实目标答案。"""

from copy import deepcopy

import pytest
from scripts import fund_002112_codex_replay_v1 as replay


def sample():
    return {
        "as_of": "2026-08-04T10:00:00+08:00",
        "window": {"base_nav_date": "2026-08-03", "target_nav_date": "2026-08-04"},
        "latest_nav_date": "2026-08-03",
        "nav": [{"nav_date": "2026-08-03", "unit_nav": "1"}],
        "report": {"available_at": "2026-07-22T08:00:00+08:00"},
        "time_proof": {"nav_max_available_at": "2026-08-04T08:00:00+08:00"},
        "companies": [{"quote": {"date": "2026-08-03"}}],
        "market": {},
        "documents": [],
    }


def test_valid_historical_cutoff():
    replay.temporal_guard(sample())


@pytest.mark.parametrize("kind", ["nav", "quote", "report", "document", "answer"])
def test_reject_late_input_and_answer(kind):
    value = deepcopy(sample())
    if kind == "nav":
        value["nav"][0]["nav_date"] = "2026-08-04"
    elif kind == "quote":
        value["companies"][0]["quote"]["date"] = "2026-08-04"
    elif kind == "report":
        value["report"]["available_at"] = "2026-08-05T08:00:00+08:00"
    elif kind == "document":
        value["documents"] = [{"date": "2026-08-04", "available_at": "2026-08-04T10:01:00+08:00"}]
    else:
        value["nested"] = {"actual_direction": "UP"}
    with pytest.raises(ValueError):
        replay.temporal_guard(value)


def test_score_gate_prevents_reading_answer(monkeypatch):
    monkeypatch.setattr(replay, "checked", lambda: {})
    monkeypatch.setattr(replay, "pending", lambda p: ["unreviewed"])
    monkeypatch.setattr(replay.data, "load_sources", lambda: pytest.fail("答案源不应在门禁通过前读取"))
    with pytest.raises(ValueError, match="ALL_FIRST_REVIEWS_REQUIRED"):
        replay.score()


def test_first_file_cannot_be_overwritten(tmp_path):
    path = tmp_path / "first.json"
    replay.save(path, {"direction": "UP"})
    with pytest.raises(FileExistsError):
        replay.save(path, {"direction": "DOWN"})
    assert replay.read(path)["direction"] == "UP"


def test_network_is_forbidden():
    with pytest.raises(RuntimeError, match="NETWORK_FORBIDDEN"):
        replay.no_network()
