"""早到的旧交易日响应不能永久遮蔽后来实际公布的新收盘。"""

import httpx
import pytest
from app.services import fund_prediction_overseas_recovery as recovery
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from app.services.material_content import content_digest

URL = "https://hq.sinajs.cn/list=gb_aapl"
BASE = "2026-11-16"


def quote(day):
    fields = ["name", "100.00", "0.00", "2026-11-17 08:00:00", "0"] + ["0"] * 31
    fields[25], fields[26], fields[29] = f"Nov {day} 04:00PM EST", "100.00", "2026"
    return ('var hq_str_gb_aapl="' + ",".join(fields) + '";').encode()


def validator(raw):
    return recovery.us_close(raw, "AAPL.US", BASE)


def cache(root, raw, *, correct_hash=True):
    original = root / "raw" / (content_digest(raw) + ".raw")
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(raw)
    pointer = root / "overseas-receipts" / BASE / (digest(URL) + ".json")
    receipt = {
        "url": URL,
        "raw_path": str(original),
        "raw_sha256": content_digest(raw) if correct_hash else "0" * 64,
        "received_at": "2026-11-17T00:30:00+08:00",
        "http_status": 200,
    }
    versioned_save(pointer, receipt)
    return original, pointer


def network(monkeypatch, responses):
    calls = []
    real_client = httpx.Client

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=responses[len(calls) - 1])

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(recovery.httpx, "Client", lambda **kwargs: real_client(transport=transport))
    return calls


def test_stale_cached_day_is_refetched_and_original_is_preserved(monkeypatch, tmp_path):
    original, pointer = cache(tmp_path, quote(13))
    calls = network(monkeypatch, [quote(16)])
    raw, receipt = recovery.fetch(URL, BASE, tmp_path, validator=validator)
    assert len(calls) == 1 and validator(raw)["date"] == BASE
    assert original.read_bytes() == quote(13)
    assert read(pointer) == receipt
    assert receipt["raw_sha256"] == content_digest(quote(16))


def test_early_response_is_archived_but_not_reused_as_target_day(monkeypatch, tmp_path):
    calls = network(monkeypatch, [quote(13), quote(16)])
    with pytest.raises(ValueError, match="CLOSE_DATE"):
        recovery.fetch(URL, BASE, tmp_path, validator=validator)
    pointer = tmp_path / "overseas-receipts" / BASE / (digest(URL) + ".json")
    assert not pointer.exists()
    assert (tmp_path / "raw" / (content_digest(quote(13)) + ".raw")).read_bytes() == quote(13)
    raw, _ = recovery.fetch(URL, BASE, tmp_path, validator=validator)
    assert len(calls) == 2 and validator(raw)["date"] == BASE


def test_verified_cached_day_does_not_refetch(monkeypatch, tmp_path):
    cache(tmp_path, quote(16))
    calls = network(monkeypatch, [])
    raw, _ = recovery.fetch(URL, BASE, tmp_path, validator=validator)
    assert not calls and validator(raw)["date"] == BASE


def test_cached_non_numeric_quote_is_refetched(monkeypatch, tmp_path):
    cache(tmp_path, quote(16).replace(b"100.00", b"N/A", 1))
    calls = network(monkeypatch, [quote(16)])
    raw, _ = recovery.fetch(URL, BASE, tmp_path, validator=validator)
    assert len(calls) == 1 and validator(raw)["close"] == "100.00"


def test_corrupt_cached_original_fails_without_network_fallback(monkeypatch, tmp_path):
    cache(tmp_path, quote(13), correct_hash=False)
    calls = network(monkeypatch, [])
    with pytest.raises(ValueError, match="RAW_CHANGED"):
        recovery.fetch(URL, BASE, tmp_path, validator=validator)
    assert not calls
