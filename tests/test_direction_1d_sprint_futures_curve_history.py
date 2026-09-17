"""验证原响应证据链，损坏或时序颠倒不能靠重算派生数组掩盖。"""

import hashlib

import pytest
from app.services import direction_1d_sprint_futures_curve_history as s


@pytest.fixture
def evidence():
    raw = b'{"code":0}'
    req = {"at": "2026-09-16T12:00:00+08:00"}
    meta = {
        "received_at": "2026-09-16T12:00:01+08:00",
        "request_hash": s.base.digest(req),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "http_status": 200,
    }
    result = {"at": "2026-09-16T12:00:02+08:00", "response_hash": s.base.digest(meta)}
    return [req, meta, result, raw]


def test_original_source_chain(evidence):
    s.response_chain(*evidence)


@pytest.mark.parametrize(
    "change", ["body", "request", "response", "bytes", "receive_before_request", "parse_before_receive"]
)
def test_tampering_and_impossible_timing_rejected(evidence, change):
    if change == "body":
        evidence[-1] = b"changed"
    elif change == "request":
        evidence[0]["new_field"] = 1
    elif change == "response":
        evidence[1]["new_field"] = 1
    elif change == "bytes":
        evidence[1]["bytes"] = 99
    elif change == "receive_before_request":
        evidence[1]["received_at"] = "2026-09-16T11:59:59+08:00"
        evidence[2]["response_hash"] = s.base.digest(evidence[1])
    else:
        evidence[2]["at"] = "2026-09-16T12:00:00+08:00"
    with pytest.raises(ValueError):
        s.response_chain(*evidence)
