"""大JSON兼容修复保留精确词元、首次失败和不可变有效正文。"""
import json

import pytest
from app.models.material_content import MaterialContent, MaterialLocation
from app.services.material_content import MaterialRepository, compact_json, material_read_cache, opaque_content
from sqlalchemy import create_engine


@pytest.fixture
def repo(tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'content.db'))
    MaterialContent.__table__.create(engine)
    MaterialLocation.__table__.create(engine)
    yield MaterialRepository(engine)
    engine.dispose()


def test_large_whitespace_compaction_keeps_escaped_quotes_strings_and_numbers():
    # 大量小段空白覆盖实际内存压力形状；精确小数不经float重序列化。
    row = r'{ "value": 123456789.1234567890123, "text": "表格 \"原话\" 和 \\路径" }'
    source = '[\n' + ',\n'.join([row] * 30_000) + '\n]'
    compact = compact_json(source)
    assert '123456789.1234567890123' in compact
    assert json.loads(source) == json.loads(compact)
    assert '表格 ' in compact and '\\"原话' in compact


def test_memory_placeholder_recovers_same_source_and_retains_first_failure(tmp_path, repo):
    path = tmp_path / 'fund.json'
    raw = b'{ "money": 0.1234567890123456789, "reports": [{"id":"old"}] }'
    path.write_bytes(raw)
    failed = opaque_content(raw, '.json', 'MemoryError')
    repo.put(path, failed)
    recovered = repo.ingest(path)
    assert recovered.kind == 'JSON' and recovered.complete
    assert recovered.source_hash == failed.source_hash
    assert recovered.extracted['initial_extraction']['content_hash'] == failed.content_hash
    assert recovered.extracted['initial_extraction']['metadata']['extraction_issue'] == 'MemoryError'
    assert '0.1234567890123456789' in recovered.decode()
    assert json.loads(recovered.decode())['reports'] == [{'id':'old'}]
    assert repo.ingest(path).content_hash == recovered.content_hash


def test_other_failure_and_invalid_json_are_not_silently_replaced(tmp_path, repo):
    path = tmp_path / 'broken.json'
    raw = b'{ broken json '
    path.write_bytes(raw)
    failed = opaque_content(raw, '.json', 'MemoryError')
    repo.put(path, failed)
    with pytest.raises(ValueError, match='REQUIRES_COMPLETE_CONTENT'):
        repo.ingest(path)
    assert repo.get(path).content_hash == failed.content_hash
    other = tmp_path / 'other.json'
    other.write_bytes(b'{"ok":1}')
    placeholder = opaque_content(other.read_bytes(), '.json', 'PermissionError')
    repo.put(other, placeholder)
    assert repo.ingest(other).kind == 'UNSUPPORTED'


def test_recovered_body_above_cache_limit_does_not_leave_small_failure_cached(tmp_path, repo):
    path = tmp_path / 'large.json'
    raw = b'{"body":"' + b'x' * (32 * 1024 * 1024 + 100) + b'"}'
    path.write_bytes(raw)
    failed = opaque_content(raw, '.json', 'MemoryError')
    repo.put(path, failed)

    @material_read_cache
    def restore_and_read():
        assert repo.get(path).kind == 'UNSUPPORTED'
        assert repo.get(source_hash=failed.source_hash).kind == 'UNSUPPORTED'
        recovered = repo.ingest(path)
        assert recovered.kind == 'JSON'
        assert repo.get(path).content_hash == recovered.content_hash
        assert repo.get(source_hash=failed.source_hash).content_hash == recovered.content_hash
        assert len(repo.get(path).extracted['text']) > 32 * 1024 * 1024

    restore_and_read()
