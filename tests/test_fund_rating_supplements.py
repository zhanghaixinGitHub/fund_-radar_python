import hashlib

import pytest
from app.services.fund_rating_supplements import record, report


def test_receipt_requires_evidence_and_preserves_newer_state_on_replay(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"fixture holding detail")
    database = tmp_path / "supplements.sqlite3"
    identity = {"fund_code": "123456", "period": "2026-09", "field": "full_holdings", "owner": "fixture"}
    ready = {
        **identity,
        "state": "READY_TO_PARSE",
        "note": "原件到手，待整理",
        "next_at": "2026-01-01T00:00:00+08:00",
        "sources": [{"file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
    }
    assert record(database, ready, sources_root=tmp_path)
    assert not record(database, ready, sources_root=tmp_path)
    assert report(database)["ready_to_parse"] == report(database)["due"] == 1

    with pytest.raises(ValueError, match="EVIDENCE_REQUIRED"):
        record(database, {**identity, "state": "VERIFIED", "note": "已核"}, sources_root=tmp_path)
    assert report(database)["open"] == 1

    closed = {**ready, "state": "VERIFIED", "note": "已核原件对应字段"}
    assert record(database, closed, sources_root=tmp_path)
    assert not record(database, ready, sources_root=tmp_path)
    assert report(database)["open"] == report(database)["due"] == 0
