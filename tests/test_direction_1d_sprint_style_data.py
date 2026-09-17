"""风格数据不能用别的指数、缺失或不连续价格冒充完整历史。"""

import json

import pytest
from app.services import direction_1d_sprint_style_data as d


def response(code="RUT"):
    return {
        "code": 0,
        "data": {
            "fields": d.FIELDS,
            "items": [
                [code, "20250106", 102, 100],
                [code, "20250107", 103, 102],
                [code, "20250108", 104, 103],
            ],
        },
    }


@pytest.fixture(autouse=True)
def sessions(monkeypatch):
    monkeypatch.setattr(d.overnight, "sessions", lambda: ["2025-01-06", "2025-01-07", "2025-01-08"])


def test_two_indices_are_validated_independently():
    for code in d.INDICES:
        result = d.validate(code, json.dumps(response(code)), "2025-01-06", "2025-01-08")
        assert len(result) == 3 and result["2025-01-08"]["close"] == 104


@pytest.mark.parametrize("kind", ("wrong_code", "missing", "duplicate", "bad_close", "broken_chain"))
def test_bad_source_records_cannot_be_silently_used(kind):
    payload = response()
    rows = payload["data"]["items"]
    if kind == "wrong_code":
        rows[0][0] = "SPX"
    elif kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows.append(rows[0])
    elif kind == "bad_close":
        rows[0][2] = float("nan")
    else:
        rows[1][3] = 500
    with pytest.raises(ValueError):
        d.validate("RUT", json.dumps(payload), "2025-01-06", "2025-01-08")


def test_reserved_failed_attempt_cannot_issue_a_second_request(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "root", lambda: tmp_path)
    monkeypatch.setattr(d, "plan", lambda: {"source_id": "source", "end": "2026-09-11", "max_requests": 12})
    monkeypatch.setattr(d.regression, "active", lambda: None)
    monkeypatch.setattr(d.overnight, "source", lambda: {"source_id": "source"})
    monkeypatch.setattr(d, "fetch_style", lambda *args: pytest.fail("unexpected repeated request"))
    d.base.save(tmp_path / "requests/RUT-2025.json", {"status": "attempted"})
    with pytest.raises(ValueError, match="NO_AUTO_RETRY"):
        d.query("RUT", 2025)
