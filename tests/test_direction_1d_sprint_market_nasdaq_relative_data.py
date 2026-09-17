"""纳指原始响应、时间对齐、无查询消费、首次实际回读及篡改拒绝。"""

from datetime import datetime

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_nasdaq_relative_data as d

from test_direction_1d_sprint_market_only import market
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def raw_source(ready_fixture):
    clock, source, _ = ready_fixture
    source["market"] = market()
    b.save(d.parent.root() / "2026-09-16/input.json", source)
    aligned = d.parent.overnight.alignment("2026-09-15", "2026-09-16")
    rows = [["IXIC", day.replace("-", ""), 100.0 + i, 99.0 + i] for i, day in enumerate(aligned["required_us_dates"])]
    value = {
        "received_at": "2026-09-16T07:01:00+08:00",
        "alignment": aligned,
        "response": {"code": 0, "data": {"fields": d.ixic.FIELDS, "items": rows}},
    }
    folder = b.ROOT / "round-06/live/2026-09-16"
    b.save(
        folder / "0700-reserved.json", {"at": "2026-09-16T07:00:00+08:00", "maximum_requests": 1, "alignment": aligned}
    )
    b.save(folder / "0700-response.json", value)
    return folder, value


def test_existing_raw_response_is_consumed_once_and_bound_to_parent(parent_ready, monkeypatch):  # noqa: F811
    folder, value = raw_source(parent_ready)
    monkeypatch.setattr(d.ixic, "fetch_ixic", lambda *args: pytest.fail("must not call provider"))
    result = d.capture(b.now())
    assert result["ixic_hash"] == b.digest(value)
    assert result["market"]["features"][3] == pytest.approx(0, abs=1e-12)
    path = d.root() / "2026-09-16/input.json"
    before = path.read_bytes()
    d.capture(b.now())
    assert path.read_bytes() == before
    assert d.live_market(parent_ready[1])["nasdaq_input_hash"] == b.digest(result)


@pytest.mark.parametrize(
    "fault", ["future_receipt", "wrong_alignment", "bad_price", "missing_day", "bad_business_code"]
)
def test_invalid_raw_cannot_create_input(parent_ready, fault):  # noqa: F811
    folder, value = raw_source(parent_ready)
    if fault == "future_receipt":
        value["received_at"] = "2026-09-16T08:30:00+08:00"
    elif fault == "wrong_alignment":
        value["alignment"] = value["alignment"] | {"base": "2026-09-14"}
    elif fault == "bad_price":
        value["response"]["data"]["items"][0][2] = 0
    elif fault == "missing_day":
        value["response"]["data"]["items"].pop()
    else:
        value["response"]["code"] = 40203
    b.save(folder / "0700-response.json", value, replace=True)
    assert d.capture(b.now()) is None
    assert not (d.root() / "2026-09-16/input.json").exists()
    assert (d.root() / "2026-09-16/0700-rejected.json").exists()


def test_late_readback_and_changed_response_fail_closed(parent_ready):  # noqa: F811
    folder, value = raw_source(parent_ready)
    d.capture(b.now())
    value["response"]["data"]["items"][-1][2] += 1
    b.save(folder / "0700-response.json", value, replace=True)
    with pytest.raises(ValueError):
        d.load("2026-09-16")


def test_recorded_readback_at_cutoff_is_rejected(parent_ready):  # noqa: F811
    raw_source(parent_ready)
    d.capture(b.now())
    p = d.root() / "2026-09-16/receipt.json"
    b.save(p, b.read(p) | {"readback_at": "2026-09-16T08:30:00+08:00"}, replace=True)
    with pytest.raises(ValueError, match="INPUT_TIME_INVALID"):
        d.load("2026-09-16")


def test_before_window_or_missing_response_never_fetches(parent_ready, monkeypatch):  # noqa: F811
    monkeypatch.setattr(d.ixic, "fetch_ixic", lambda *args: pytest.fail("must not fetch"))
    assert d.capture(datetime(2026, 9, 16, 6, 59, tzinfo=b.ZONE)) is None
    assert d.capture(b.now()) is None
    assert d.capture(datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)) is None
