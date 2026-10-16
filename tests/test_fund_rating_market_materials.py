"""部分资料的时点和原件边界，防止后补数据进入历史冻结批次。"""

import hashlib
import json
from datetime import UTC, date, datetime

import pytest
from app.services.fund_rating_market_materials import load


def test_later_materials_are_excluded_and_original_tampering_rejects(tmp_path):
    (tmp_path / "sources").mkdir()
    (tmp_path / "assessments").mkdir()
    original = tmp_path / "sources" / "report.json"
    original.write_bytes(b"original")
    item = {
        "fund_code": "001234",
        "as_of_date": "2026-09-30",
        "verified_at": "2026-10-15T01:00:00+00:00",
        "source_root": str(tmp_path.resolve()),
        "sources": [{"file": "sources/report.json", "sha256": hashlib.sha256(b"original").hexdigest()}],
    }
    raw = json.dumps({"version": 1, "as_of_date": "2026-09-30", "records": {"001234": item}}).encode()
    packet = tmp_path / "assessments" / "packet.json"
    packet.write_bytes(raw)
    (tmp_path / "market-materials-current.json").write_text(
        json.dumps({"file": "assessments/packet.json", "sha256": hashlib.sha256(raw).hexdigest()}), encoding="utf8"
    )
    assert load(tmp_path, date(2026, 9, 30), datetime(2026, 10, 14, tzinfo=UTC)) == {}
    assert load(tmp_path, date(2026, 9, 30), datetime(2026, 10, 16, tzinfo=UTC)) == {"001234": item}
    original.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="MATERIALS_ORIGINAL"):
        load(tmp_path, date(2026, 9, 30), datetime(2026, 10, 16, tzinfo=UTC))
