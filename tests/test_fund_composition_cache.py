"""隔离SQLite验证持久缓存；不连接业务库，不读取真实原件。"""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.services import fund_composition as composition
from app.services import fund_composition_cache as cache
from sqlalchemy import create_engine, text


@pytest.fixture
def store(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + str(tmp_path / "cache.sqlite3"))
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE fund_composition_cache(cache_key text PRIMARY KEY,payload text,created_at text)")
        )
    monkeypatch.setattr(cache, "get_engine", lambda: engine)
    monkeypatch.setattr(
        cache,
        "get_settings",
        lambda: SimpleNamespace(fund_material_directory=tmp_path, fund_insight_directory=tmp_path),
    )
    versions = {}
    monkeypatch.setattr(cache, "_identity", lambda path: versions.get(str(Path(path).resolve())))
    calls = []
    source = tmp_path / "source.json"
    versions[str(source.resolve())] = "v1"

    def compute(codes):
        calls.append(list(codes))
        cache.observe(source)
        return {
            "version": 1,
            "funds": [{"fundCode": code, "source": versions[str(source.resolve())]} for code in codes],
            "themeDefinitions": [],
        }

    monkeypatch.setattr(composition, "batch", compute)
    yield engine, calls, versions, source
    engine.dispose()


def test_persisted_hit_skips_revalidation_and_preserves_requested_order(store):
    engine, calls, versions, source = store
    first = cache.batch(["000001", "000002"])
    engine.dispose()  # 新连接仍从库复用，不依赖进程内结果缓存。
    second = cache.batch(["000002", "000001"])
    assert len(calls) == 1
    assert second["funds"] == list(reversed(first["funds"]))


def test_source_change_and_new_holdings_invalidate_cache(store):
    _, calls, versions, source = store
    cache.batch(["000001"])
    versions[str(source.resolve())] = "v2"
    assert cache.batch(["000001"])["funds"][0]["source"] == "v2"
    cache.batch(["000001", "000002"])
    assert len(calls) == 3


def test_missing_source_appears_and_classification_changes_invalidate(store):
    _, calls, versions, source = store
    cache.batch(["000001"])
    root = source.parent / "market-workflow/comprehensive-inputs/000001.json"
    versions[str(root.resolve())] = "receipt-added"
    cache.batch(["000001"])
    versions[str((composition.PROJECT / "app/data/fund_security_themes_v1.json").resolve())] = "new-themes"
    cache.batch(["000001"])
    assert len(calls) == 3


def test_sources_changing_during_build_are_not_saved(store, monkeypatch):
    engine, _, versions, source = store

    def compute(codes):
        cache.observe(source)
        versions[str(source.resolve())] = "changed-mid-read"
        return {"funds": []}

    monkeypatch.setattr(composition, "batch", compute)
    with pytest.raises(ValueError, match="CHANGED_DURING_READ"):
        cache.batch(["000001"])
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM fund_composition_cache")) == 0
    assert not cache.active()


def test_actual_bytes_change_detected_even_with_same_length(tmp_path, monkeypatch):
    from app.services import material_content

    source = tmp_path / "source.json"
    source.write_bytes(b"old")
    monkeypatch.setattr(material_content, "source_bytes", lambda path: path.read_bytes())
    first = cache._identity(source)
    source.write_bytes(b"new")
    assert first == hashlib.sha256(b"old").hexdigest()
    assert cache._identity(source) != first


def test_read_reuses_verified_payload_within_build(tmp_path, monkeypatch):
    from app.services import fund_exposure_common

    calls = []
    monkeypatch.setattr(cache, "_identity", lambda path: "version")
    monkeypatch.setattr(fund_exposure_common, "read", lambda path: calls.append(path) or {"verified": True})
    token = cache._context.set({"sources": {}, "values": {}})
    try:
        path = tmp_path / "payload.json"
        assert cache.read(path) is cache.read(path)
        assert len(calls) == 1
    finally:
        cache._context.reset(token)
