"""窗口内模型完成可用；未来登记、过期及错误时间仍拒绝。"""

from datetime import datetime

import pytest
from app.services.direction_1d_selection import available_at_prediction


@pytest.mark.parametrize(
    ("trained", "registered", "expires", "expected"),
    [
        ("2026-09-11T23:23:23", "2026-09-11T23:23:41", "2027-01-01T00:00:00", True),
        ("2026-09-12T05:59:00", "2026-09-12T06:00:00", "2027-01-01T00:00:00", True),
        ("2026-09-12T06:00:01", "2026-09-12T06:00:02", "2027-01-01T00:00:00", False),
        ("2026-09-11T23:23:23", "2026-09-12T06:00:01", "2027-01-01T00:00:00", False),
        ("2026-09-11T23:23:23", "2026-09-11T23:23:00", "2027-01-01T00:00:00", False),
        ("2026-09-11T23:23:23", "2026-09-11T23:23:41", "2026-09-12T06:00:00", False),
    ],
)
def test_real_availability_replaces_window_open_gate(trained, registered, expires, expected):
    def at(value):
        return datetime.fromisoformat(value + "+08:00")

    model = {"trained_at": at(trained), "registered_at": at(registered), "expires_at": at(expires)}
    assert available_at_prediction(model, at("2026-09-12T06:00:00")) is expected
