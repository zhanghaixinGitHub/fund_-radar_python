"""实际来源二次解析时仍绑定原始字节、父分布和实收时间，不允许历史补填。"""

from copy import deepcopy

import pytest
from app.services import direction_1d_sprint_stock_tail_live as s


@pytest.fixture
def actual_source(monkeypatch, tmp_path):
    monkeypatch.setattr(s.b, "ROOT", tmp_path)
    monkeypatch.setattr(s.parent, "root", lambda: tmp_path / "live")
    t, u = "2026-09-16", "2026-09-17"
    path = s.parent.root() / u
    path.mkdir(parents=True)
    raw = b"actual-source-test"
    (path / "raw.bin").write_bytes(raw)
    receipt = {"raw": {"bytes": len(raw), "sha256": s.hashlib.sha256(raw).hexdigest()}}
    s.b.save(path / "response.json", receipt)
    s.b.save(tmp_path / "round-116/plan.json", {"model": "tail"})
    s.b.save(tmp_path / "round-109/plan.json", {"model": "breadth"})
    original = {"date": t, "included_set_sha256": "same-set"}
    source = {
        "at": "2026-09-17T08:15:00+08:00",
        "received_at": "2026-09-17T08:14:58+08:00",
        "receipt_hash": s.b.digest(receipt),
        "points": {t: original},
        "available": True,
    }
    row = {"date": t, "source_distribution_hash": s.b.digest(original), "tail_balance": 0.2}
    calls = []

    def load(a, z, ph):
        calls.append((a, z, ph))
        return deepcopy(source)

    monkeypatch.setattr(s.parent, "load_live", load)
    monkeypatch.setattr(s.features, "parse", lambda *args: deepcopy(row))
    return source, row, calls, t, u, s.b.digest({"model": "tail"})


def test_preserves_actual_time_parent_identity_and_zero_new_requests(actual_source):
    source, row, calls, t, u, ph = actual_source
    got = s.load_live(t, u, ph)
    assert calls == [(t, u, s.b.digest({"model": "breadth"}))]
    assert got["parent_source_hash"] == s.b.digest(source) and got["at"] == source["at"]
    assert got["points"] == {t: row}
    assert got["new_source_requests"] == got["reserved_request_slots"] == 0


def test_missing_source_stays_missing_without_reading_historical_t(actual_source):
    source, _, _, t, u, ph = actual_source
    source.update(available=False, points={}, received_at=None)
    (s.parent.root() / u / "raw.bin").unlink()
    got = s.load_live(t, u, ph)
    assert not got["available"] and got["points"] == {} and got["received_at"] is None


@pytest.mark.parametrize("change", ["raw", "receipt", "distribution", "plan"])
def test_source_change_rejected(actual_source, change):
    _, row, _, t, u, ph = actual_source
    if change == "raw":
        (s.parent.root() / u / "raw.bin").write_bytes(b"changed-price-same-universe")
    elif change == "receipt":
        s.b.save(s.parent.root() / u / "response.json", {"changed": True}, replace=True)
    elif change == "distribution":
        row["source_distribution_hash"] = "changed"
    else:
        ph = "changed"
    with pytest.raises(ValueError):
        s.load_live(t, u, ph)
