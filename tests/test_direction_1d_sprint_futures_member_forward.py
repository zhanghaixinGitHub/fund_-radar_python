"""新层必须保留全部旧分支，并在父链前完成有时限的映射和排名采集。"""

from datetime import date, datetime
from pathlib import Path

import pytest
from app.services import direction_1d_sprint_futures_member_forward as forward


def test_all39_branches_include_intermediate118(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(forward.base, "ROOT", tmp_path)

    def read(path):
        number = int(Path(path).relative_to(tmp_path).parts[0].split("-")[1])
        seen.append(number)
        count = 2 if number in (92, 113) else 1
        return {"answers": {f"r{number}_{i}": {"prediction": i % 2} for i in range(count)}}

    monkeypatch.setattr(forward.base, "read", read)
    parent = {"u": "2026-09-17", "code": "synthetic", "answers": {f"core{i}": {} for i in range(9)}}
    combined = forward.combined_answers(parent, {"answers": {"r119": {}}}, {"answers": {"r120": {}}})
    assert len(combined) == 39 and seen == [92, *range(94, 119)]
    assert "r118_0" in combined and "r119" in combined and "r120" in combined


def test_early_capture_order_and_original_parent_budget(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(forward.runtime, "plan", lambda: {})
    monkeypatch.setattr(forward.base, "ROOT", tmp_path)
    monkeypatch.setattr(forward.base, "now", lambda: datetime.fromisoformat("2026-09-17T08:00:00+08:00"))
    monkeypatch.setattr(forward.base, "calendar", lambda: ([date(2026, 9, 16), date(2026, 9, 17)], "cal"))
    monkeypatch.setattr(forward.base, "read", lambda path: {"path": str(path)})
    monkeypatch.setattr(forward.live.quote_live, "live_root", lambda: tmp_path / "quote")
    snapshot = tmp_path / "quote/2026-09-17/snapshot.json"

    def quote(*args):
        events.append("original_quote_capture")
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text("synthetic", encoding="utf-8")

    monkeypatch.setattr(forward.live.quote_live, "capture", quote)
    monkeypatch.setattr(forward.live, "capture", lambda *args: events.append("single_member_capture"))

    def parent():
        events.append("parent_run")
        raise RuntimeError("STOP_BEFORE_CONTEXT")

    monkeypatch.setattr(forward.prior_forward, "run", parent)
    with pytest.raises(RuntimeError, match="STOP_BEFORE_CONTEXT"):
        forward.run()
    assert events == ["original_quote_capture", "single_member_capture", "parent_run"]
