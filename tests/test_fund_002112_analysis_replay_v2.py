"""冻结选样、失败分母与评分先后边界，拒绝把旧实验的代码哈希改成新代码。"""

from datetime import date, timedelta

import pytest
from scripts import fund_002112_analysis_replay_v2 as replay


def test_selection_only_uses_phase_and_date_not_answers():
    days = [str(date(2026, 1, 1) + timedelta(days=i)) for i in range(80)]
    rows = [{"target": t, "phase": "MORNING_0830", "label": "UP"} for t in days]
    result = replay.select_targets(rows, days[-20:])
    for row in rows:
        row["label"] = "DOWN"
        row["return"] = -999
    assert replay.select_targets(list(reversed(rows)), days[-20:]) == result
    assert result == {"regression": days[-20:], "expanded": days[-60:-20]}


def test_scoring_does_not_load_answers_until_every_target_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "OUT", tmp_path)
    monkeypatch.setattr(replay, "checked_protocol", lambda: {"targets": ["2026-08-26"]})
    monkeypatch.setattr(replay.data, "load_sources", lambda: pytest.fail("不能提前读取答案"))
    with pytest.raises(ValueError, match="ALL_PREDICTIONS_REQUIRED"):
        replay.score()


def test_new_protocol_rejects_changed_code_without_rewriting_hash(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_text("before", encoding="utf-8")
    expected = replay.data.sha(path)
    replay.data.save(tmp_path / "protocol.json", {"code_hashes": {str(path): expected}})
    path.write_text("after", encoding="utf-8")
    monkeypatch.setattr(replay, "OUT", tmp_path)
    with pytest.raises(ValueError, match="FROZEN_FILE_CHANGED"):
        replay.checked_protocol()
    assert replay.data.read(tmp_path / "protocol.json")["code_hashes"][str(path)] == expected
