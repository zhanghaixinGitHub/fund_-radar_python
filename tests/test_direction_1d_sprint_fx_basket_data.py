"""验证新增币种使用每美元口径、相同原始响应和相同到达时间预算。"""

from datetime import datetime

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_fx_basket_data as d

from test_direction_1d_sprint_ecb_data import zipped

CSV = (
    "Date,USD,CNY,JPY,CHF,AUD,CAD,GBP,NOK\n"
    "2026-09-16,9,90,9999,9,9,9,9,9\n"
    "2026-09-15,1.25,9.0,187.5,0.8,1.8,1.7,0.8,12\n"
    "2026-09-14,1.2,8.4,168,0.9,1.8,1.7,0.8,12\n"
    "2026-09-11,1.1,7.7,165,0.9,1.8,1.7,0.8,12\n"
    "2026-09-10,1.1,7.7,165,0.9,1.8,1.7,0.8,12\n"
)


def body():
    return zipped(CSV)


def test_currency_order_units_and_core_features_preserved():
    points = d.parse(body())
    result = d.features("2026-09-15", "2026-09-16", points)
    assert len(result) == 8
    assert result[:2] == d.core.features("2026-09-15", "2026-09-16", points)
    assert result[2] == pytest.approx((150 / 140 - 1) * 100)
    assert result[3] == pytest.approx(((0.8 / 1.25) / (0.9 / 1.2) - 1) * 100)
    points["2026-09-16"]["jpy_per_eur"] = 1
    assert d.features("2026-09-15", "2026-09-16", points) == result
    assert d.parse(body())["2026-09-16"]["jpy_per_eur"] == 9999


@pytest.mark.parametrize("replacement", ["nan", "-1", "0", "N/A"])
def test_missing_or_invalid_new_currency_is_rejected(replacement):
    raw = zipped(CSV.replace("187.5", replacement))
    assert d.core.parse(raw)["2026-09-15"]["usd_per_eur"] == 1.25
    with pytest.raises(ValueError):
        d.parse(raw)


def test_missing_field_does_not_change_old_parser():
    raw = zipped(CSV.replace("JPY", "JPY_OTHER"))
    assert d.core.parse(raw)
    with pytest.raises(ValueError, match="CURRENCY_FIELDS_INVALID"):
        d.parse(raw)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    calendar = b.calendar()
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    clock = [datetime(2026, 9, 16, 7, 10, tzinfo=b.ZONE)]
    monkeypatch.setattr(b, "now", lambda: clock[0])
    monkeypatch.setattr(
        b, "window", lambda at: {"status": "OPEN", "base_nav_date": "2026-09-15", "target_nav_date": "2026-09-16"}
    )
    b.save(tmp_path / "protocol.json", {"deadline_at": "2026-09-17T12:11:38+08:00"})
    calls = []
    monkeypatch.setattr(d.core, "fetch_ecb_fx", lambda: (calls.append(1) or body(), {}))
    return clock, calls


def test_old_and_new_branches_share_exactly_one_successful_request(ready):
    _, calls = ready
    original = d.core.capture(b.now())
    original_bytes = (d.core.root() / "2026-09-16/input.json").read_bytes()
    expanded = d.capture(b.now())
    assert calls == [1]
    assert expanded["core_input_hash"] == b.digest(original)
    assert expanded["at"] == original["at"] and expanded["raw_ref"] == original["raw_ref"]
    assert len(expanded["rows"]["2026-09-15"]) == 8
    assert d.load("2026-09-16") == expanded
    assert (d.core.root() / "2026-09-16/input.json").read_bytes() == original_bytes


def test_changed_shared_raw_is_rejected(ready):
    d.capture(b.now())
    raw = d.core.root() / "2026-09-16/raw/0700.zip"
    raw.write_bytes(zipped(CSV.replace("187.5", "188.5")))
    with pytest.raises(ValueError, match="LIVE_RAW_CHANGED"):
        d.load("2026-09-16")


def test_extraction_finishing_at_cutoff_is_rejected(ready, monkeypatch):
    clock, _ = ready
    d.core.capture(b.now())
    real_load = d.load

    def late(target):
        value = real_load(target)
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return value

    monkeypatch.setattr(d, "load", late)
    with pytest.raises(ValueError, match="EXTRACTION_LATE"):
        d.capture(b.now())


def test_missing_expected_reference_date_cannot_use_older_date():
    points = d.parse(body())
    del points["2026-09-15"]
    with pytest.raises(ValueError, match="REQUIRED_DATE_MISSING"):
        d.features("2026-09-15", "2026-09-16", points)
