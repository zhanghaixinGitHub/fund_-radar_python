"""公告日期审计边界：合成响应、只读范围、重复预算及零模型训练。"""

from copy import deepcopy
from datetime import datetime, timedelta

import pytest
from app.integrations.tushare import TushareIntegrationError
from app.services import direction_1d_ann_audit as audit
from app.services.direction_1d_protocol import ZONE, digest


def scope():
    codes = ["001632", "008888"]
    return {
        "codes": codes,
        "scope_hash": digest(codes),
        "verified_at": datetime.now(ZONE).isoformat(),
        "identity_basis": "EXACT_USER_CONFIRMED_ACTIVE_ACCOUNT",
    }


@pytest.mark.parametrize("case", ["duplicate", "foreign_hash", "admin", "stale", "future", "invalid_code", "empty"])
def test_wrong_or_stale_owner_scope_is_rejected(case):
    s = scope()
    if case == "duplicate":
        s["codes"].append(s["codes"][0])
    elif case == "foreign_hash":
        s["scope_hash"] = "other"
    elif case == "admin":
        s["identity_basis"] = "FIRST_ADMIN_ACCOUNT"
    elif case == "stale":
        s["verified_at"] = (datetime.now(ZONE) - timedelta(hours=2)).isoformat()
    elif case == "future":
        s["verified_at"] = (datetime.now(ZONE) + timedelta(hours=2)).isoformat()
    elif case == "invalid_code":
        s["codes"] = ["bad"]
    else:
        s["codes"] = []
    with pytest.raises(ValueError):
        audit.validate_scope(s)


def test_verified_scope_preserved_and_not_extended():
    s = scope()
    assert audit.validate_scope(s) == s["codes"]


def raw():
    return {
        "ts_code": "001632.OF",
        "nav_date": "20240930",
        "ann_date": "20241025",
        "unit_nav": "1.1234",
        "accum_nav": "1.3456",
        "net_asset": "1000000",
        "total_netasset": None,
        "accum_div": None,
        "adj_nav": None,
    }


def test_later_ann_date_is_preserved_without_claiming_first_publication():
    source = raw()
    before = deepcopy(source)
    rows = audit.validate_probe([source], "001632", "001632.OF")
    assert str(rows[0]["ann_date"]) == "2024-10-25"
    assert str(rows[0]["nav_date"]) == "2024-09-30"
    assert source == before
    assert "received_at" not in rows[0] and "source_published_at" not in rows[0]


@pytest.mark.parametrize("case", ["code", "date", "duplicate", "empty", "limit", "zero", "nan", "wrong_mapping"])
def test_invalid_raw_probe_is_not_adopted(case):
    row, code = raw(), "001632.OF"
    rows = [row]
    if case == "code":
        row["ts_code"] = "008888.OF"
    elif case == "date":
        row["nav_date"] = "20250102"
    elif case == "duplicate":
        rows.append(deepcopy(row))
    elif case == "empty":
        rows = []
    elif case == "limit":
        rows *= 20
    elif case == "zero":
        row["unit_nav"] = "0"
    elif case == "nan":
        row["unit_nav"] = "NaN"
    else:
        code = "008888.OF"
    with pytest.raises((ValueError, TushareIntegrationError)):
        audit.validate_probe(rows, "001632", code)


def test_missing_announcement_is_not_replaced_with_nav_date():
    row = raw()
    row["ann_date"] = None
    assert audit.validate_probe([row], "001632", "001632.OF")[0]["ann_date"] is None


def test_delay_count_is_calendar_days_and_quarter_end_is_only_a_clue():
    rows = [
        {"nav_date": "2024-09-30", "ann_date": "2024-10-25", "net_asset": "100", "source_published_at": None},
        {"nav_date": "2024-09-27", "ann_date": "2024-10-08", "net_asset": None, "source_published_at": None},
        {"nav_date": "2024-10-08", "ann_date": None, "net_asset": None, "source_published_at": None},
    ]
    s = audit.summary_rows(rows)
    assert s["ann_delay_gt_7_days"] == 2 and s["late_at_quarter_end"] == 1
    assert s["late_with_net_asset"] == 1 and s["ann_null"] == 1 and s["max_delay_days"] == 25
    assert "first_publication_proven" not in s


@pytest.mark.parametrize("case", ["disabled", "api", "retention", "rate"])
def test_missing_source_authorization_is_rejected(case):
    source = {
        "source_code": "TUSHARE_PRO_FUND",
        "enabled": True,
        "authorization_verified_at": "now",
        "authorized_api_names": ["fund_nav"],
        "retention_days": 365,
        "rate_limit_per_minute": 200,
    }
    if case == "disabled":
        source["enabled"] = False
    elif case == "api":
        source["authorized_api_names"] = []
    elif case == "retention":
        source["retention_days"] = 0
    else:
        source["rate_limit_per_minute"] = 0
    with pytest.raises(ValueError, match="SOURCE_UNAVAILABLE"):
        audit.validate_source(source)


def test_completed_probe_does_not_access_database_or_provider(tmp_path, monkeypatch):
    audit.write_new(tmp_path / "probe-completion.json", {})
    monkeypatch.setattr(audit, "verify_inputs", lambda _: {})
    monkeypatch.setattr(audit, "verify_probes", lambda *a: None)
    monkeypatch.setattr(audit, "get_engine", lambda: pytest.fail("不得访问数据库"))
    monkeypatch.setattr(audit, "client", lambda: pytest.fail("不得重复调用来源"))
    assert audit.probe(tmp_path) == {"status": "ALREADY_COMPLETED", "new_api_calls": 0}


def test_probe_budget_cannot_gain_extra_request(tmp_path):
    audit.write_new(tmp_path / "probe-completion.json", {"api_calls": 3, "files": {}})
    with pytest.raises(ValueError, match="BUDGET_CHANGED"):
        audit.verify_probes(tmp_path, {"probe_codes": ["001632", "008888"]})


def test_invalid_scope_never_opens_database(monkeypatch):
    s = scope()
    s["identity_basis"] = "FIRST_ADMIN_ACCOUNT"
    monkeypatch.setattr(audit, "get_engine", lambda: pytest.fail("不能查询管理员或他人数据"))
    with pytest.raises(ValueError, match="SCOPE_INVALID"):
        audit.collect(s)


@pytest.mark.parametrize("case", ["fit_budget", "date", "expired", "input", "code"])
def test_frozen_audit_boundaries_cannot_be_expanded(tmp_path, monkeypatch, case):
    monkeypatch.setattr(audit, "fingerprint", lambda: {"code": "known"})
    audit.write_new(tmp_path / "input.json", {"scope": 43})
    spec = {
        "fingerprint": audit.fingerprint(),
        "max_fits": 0,
        "max_requests": audit.MAX_REQUESTS,
        "probe_start": audit.START,
        "probe_end": audit.END,
        "model_released": False,
        "source_expires_at": (datetime.now(ZONE) + timedelta(days=1)).isoformat(),
        "input_files": {"input.json": audit.file_hash(tmp_path / "input.json")},
    }
    if case == "fit_budget":
        spec["max_fits"] = 1
    elif case == "date":
        spec["probe_end"] = "2025-01-02"
    elif case == "expired":
        spec["source_expires_at"] = (datetime.now(ZONE) - timedelta(days=1)).isoformat()
    elif case == "code":
        spec["fingerprint"] = {"code": "unknown"}
    else:
        (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    audit.write_new(tmp_path / "audit-spec.json", spec)
    audit.write_new(tmp_path / "audit-spec-receipt.json", {"hash": digest(spec)})
    with pytest.raises(ValueError):
        audit.verify_inputs(tmp_path)
