"""联合特征必须同日且两源齐全；未来证据时间取较晚实际收到时间。"""

from copy import deepcopy

import pytest
from app.services import direction_1d_sprint_option_futures_data as s


def test_combination_keeps_both_and_does_not_fill_missing():
    c = {"2026-09-16": {"date": "2026-09-16", "log_put_call_volume_change": 0.1}}
    f = {"2026-09-16": {"intraday_return_pct": -0.2}}
    assert s.combine("2026-09-16", c, f) == c["2026-09-16"] | f["2026-09-16"]
    assert s.combine("2026-09-16", c, {}) is None
    assert s.combine("2026-09-16", {}, f) is None
    c["2026-09-16"]["date"] = "2026-09-15"
    with pytest.raises(ValueError, match="COMBINATION_DATE_MISMATCH"):
        s.combine("2026-09-16", c, f)


def test_live_retains_both_actual_receipts_and_later_time(monkeypatch):
    c = {
        "at": "2026-09-17T08:16:00+08:00",
        "previous_history_at": "2026-09-16T11:01:00+08:00",
        "available": True,
        "snapshot": {"points": {"2026-09-16": {"date": "2026-09-16", "log_put_call_volume_change": 0.1}}},
    }
    f = {
        "at": "2026-09-17T08:15:00+08:00",
        "available": True,
        "snapshot": {"points": {"2026-09-16": {"intraday_return_pct": -0.2}}},
    }
    monkeypatch.setattr(s.change, "live", lambda *_: deepcopy(c))
    monkeypatch.setattr(s.intra, "live", lambda *_: deepcopy(f))
    v = s.live("2026-09-16", "2026-09-17", "r97plan", "r95plan")
    assert (
        v["at"] == c["at"]
        and v["change_source_hash"] == s.base.digest(c)
        and v["intraday_source_hash"] == s.base.digest(f)
    )
    assert v["available"] and v["new_source_requests"] == 0
    f["available"] = False
    assert not s.live("2026-09-16", "2026-09-17", "r97plan", "r95plan")["available"]
