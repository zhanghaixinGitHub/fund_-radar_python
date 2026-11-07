"""大派生清单的技术上限仍有界；默认原件上限和摘要校验不得失效。"""

import hashlib

import pytest
from app.services.fund_rating_reviewed_snapshot import _read


def test_default_limit_remains_and_explicit_component_read_still_checks_hash(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    raw = b"x" * 25_000_001
    digest = hashlib.sha256(raw).hexdigest()
    p = sources / (digest + ".txt")
    p.write_bytes(raw)
    ref = {"file": "sources/" + p.name, "sha256": digest, "kind": "DERIVED_MANDATE_REVIEW"}
    with pytest.raises(ValueError, match="ARCHIVE_PATH"):
        _read(ref, tmp_path)
    assert _read(ref, tmp_path, max_bytes=50_000_000) is None
    with pytest.raises(ValueError, match="ARCHIVE_HASH"):
        _read({**ref, "sha256": "0" * 64}, tmp_path, max_bytes=50_000_000)
    with pytest.raises(ValueError, match="SIZE_BOUND"):
        _read(ref, tmp_path, max_bytes=50_000_001)
