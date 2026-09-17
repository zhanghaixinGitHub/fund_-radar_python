"""港股历史只读采集的日期、价格链及重复请求保护。"""

import json

import pytest
from app.services import direction_1d_sprint_hk_data as d


def payload():
    return {
        "code": 0,
        "data": {"fields": d.FIELDS, "items": [["HSI", "20250102", 101, 100], ["HSI", "20250103", 102, 101]]},
    }


def test_valid_rows_keep_source_date_and_previous_close():
    result = d.validate("HSI", json.dumps(payload()), "2025-01-02", "2025-01-03")
    assert result["2025-01-03"] == {"close": 102, "pre_close": 101}


@pytest.mark.parametrize("kind", ("symbol", "future", "duplicate", "price", "chain", "empty"))
def test_unusable_vendor_rows_are_rejected(kind):
    value = payload()
    rows = value["data"]["items"]
    if kind == "symbol":
        rows[0][0] = "SPX"
    elif kind == "future":
        rows[1][1] = "20250106"
    elif kind == "duplicate":
        rows.append(rows[0])
    elif kind == "price":
        rows[0][2] = float("nan")
    elif kind == "chain":
        rows[1][3] = 500
    else:
        rows.clear()
    with pytest.raises(ValueError):
        d.validate("HSI", json.dumps(value), "2025-01-02", "2025-01-03")


def test_failed_attempt_is_not_retried(tmp_path, monkeypatch):
    spec = {"key": "HSI-test", "code": "HSI", "start": "2025-01-02", "end": "2025-01-03"}
    monkeypatch.setattr(d, "root", lambda kind: tmp_path)
    monkeypatch.setattr(d, "plan", lambda kind: {"queries": [spec], "source_id": "source", "max_requests": 1})
    monkeypatch.setattr(d.regression, "active", lambda: None)
    monkeypatch.setattr(
        d.overnight, "source", lambda: {"source_id": "source", "authorized_api_names": ["index_global"]}
    )
    monkeypatch.setattr(d, "fetch_hk", lambda *args: pytest.fail("unexpected provider request"))
    d.base.save(tmp_path / "requests/HSI-test.json", {"attempted": True})
    with pytest.raises(ValueError, match="NO_AUTO_RETRY"):
        d.query("probe", spec)
