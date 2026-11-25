from datetime import date, datetime
from pathlib import Path

from app.commands.fund_daily_check import assess, read_supplement_work
from app.services.fund_daily_supplements import DailySupplements, pending, task_key, verified
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from tests.test_daily_holdings_check import completed


def test_resume_checks_exact_dates_and_keeps_unresolved_original_rows(tmp_path, monkeypatch):
    import app.services.fund_daily_supplements as module

    directory = tmp_path / "night"
    scope = ["000001", "000002"]
    window = {"base_nav_date": "2026-11-20", "target_nav_date": "2026-11-23"}
    instant = datetime.fromisoformat("2026-11-23T00:50:00+08:00")
    state = {"checked_at": instant.isoformat(), "scope": scope, "window": window, "funds": {},
             "batch": {"job_id": "original"}}
    profiles = {code: {"name": code} for code in scope}
    old = {"fund_code": scope[0], "field": "NAV", "required_date": "2026-11-19", "next_source": "旧缺项原渠道"}
    current = {**old, "required_date": "2026-11-20"}
    other = {**current, "fund_code": scope[1]}
    versioned_save(directory / "supplement-tasks.json", {"window": window, "tasks": [old]})
    monkeypatch.setattr(module, "now", lambda: instant)
    monkeypatch.setattr(module, "calendar", lambda: ([date(2026, 11, 20)], "verified-calendar"))
    monkeypatch.setattr(module, "save_pending", lambda *args, **kwargs: [current, other])
    monkeypatch.setattr(DailySupplements, "check_storage", lambda self: None)
    monkeypatch.setattr(DailySupplements, "original", lambda self, value: Path(value))
    monkeypatch.setattr(DailySupplements, "material_for_check", lambda *args: (None, {}))
    seen = []

    def acquire(self, task, *args):
        seen.append(task_key(task))
        if task == old:
            return pending("MISMATCH", "实际查询", "所需日期未覆盖", next_source="查管理人同日净值")
        return verified("正式净值库", "已核对同日净值", {"raw": "evidence"})

    monkeypatch.setattr(DailySupplements, "acquire", acquire)
    result = DailySupplements(directory, state, profiles).process(state)
    assert seen == [task_key(item) for item in (old, current, other)]
    assert result["job_id"] == "original"
    assert result["item_count"] == 3
    assert result["pending_item_count"] == 1
    tasks = read(directory / "supplement-tasks.json")["tasks"]
    assert [task_key(row) for row in tasks] == [task_key(old)]
    assert tasks[0]["source_review"]["summary"] == "所需日期未覆盖"
    assert tasks[0]["next_source"] == "查管理人同日净值"
    seen.clear()
    again = DailySupplements(directory, state, profiles).process(state)
    assert seen == [task_key(old)]
    assert again["item_count"] == 3
    assert len(read(directory / "source-rechecks.json")["items"]) == 3


def test_morning_report_flags_unfinished_original_supplements():
    state, guard, codes = completed()
    state["window"]["base_nav_date"] = "2026-11-10"
    state["batch"] = {"job_id": "original"}
    receipt = {
        "day": "2026-11-11", "checked_at": "2026-11-11T01:23:25+08:00", "job_id": "original",
        "base_nav_date": state["window"]["base_nav_date"], "target_nav_date": state["window"]["target_nav_date"],
        "funds": {code: {"REVIEW_REQUIRED": 1} for code in codes},
        "status": "ACTION_REQUIRED", "pending_item_count": len(codes),
    }
    report = assess(state, guard, codes, [], "2026-11-11T09:00:00+08:00", supplements=receipt)
    assert "SUPPLEMENT_ITEMS_PENDING" in report["issues"]
    assert report["supplements"] == receipt
    receipt["job_id"] = "another"
    report = assess(state, guard, codes, [], "2026-11-11T09:00:00+08:00", supplements=receipt)
    assert "SUPPLEMENT_RECEIPT_MISMATCH" in report["issues"]


def test_direct_run_reuses_evidence_and_only_calls_recovery_for_missing_fields(tmp_path, monkeypatch):
    from app.commands import fund_daily_check
    from app.services import fund_daily_supplements as module

    scope = ["000001", "000002"]
    base, day = "2026-11-20", "2026-11-23"
    instant = datetime.fromisoformat(day + "T00:50:00+08:00")
    window = {"base_nav_date": base, "target_nav_date": day}
    state = {"checked_at": instant.isoformat(), "scope": scope, "window": window, "funds": {},
             "supplement_run_id": "night"}
    tasks = [{"fund_code": code, "field": field, "required_date": base} for code in scope
             for field in ("CLOSE:AAA", "FX", "NAV")]
    profiles = {code: {"name": code} for code in scope}
    monkeypatch.setattr(fund_daily_check, "PROJECT", tmp_path)
    monkeypatch.setattr(module, "now", lambda: instant)
    monkeypatch.setattr(module, "calendar", lambda: ([], "calendar"))
    monkeypatch.setattr(module, "save_pending", lambda *_, fund_code=None, **__: [
        row for row in tasks if fund_code is None or row["fund_code"] == fund_code
    ])
    monkeypatch.setattr("app.services.fund_materials.snapshot", lambda _: {})
    monkeypatch.setattr(DailySupplements, "check_storage", lambda self: None)
    monkeypatch.setattr(DailySupplements, "material_for_check",
                        lambda self, code, item: (tmp_path / f"{code}.json", {"fund_code": code}))
    for code in scope:
        versioned_save(tmp_path / f"{code}.json", {"fund_code": code})
    calls = []

    def acquire(self, task, *args, reuse_only=False):
        code, field, _ = task_key(task)
        if reuse_only and code == scope[1] and field == "FX":
            return None
        if not reuse_only:
            calls.append((code, field))
        return verified("核对资料", "份额和日期一致", {"source": code})

    def recover(code, output, window, *, missing_fields):
        calls.append((code, sorted(missing_fields)))
        return {"materials_path": str(tmp_path / f"{code}.json")}

    monkeypatch.setattr(DailySupplements, "acquire", acquire)
    supplements = DailySupplements(tmp_path / "night", state, profiles)
    result = supplements.process(state, recover_materials=recover)
    assert result["job_id"] == "night" and result["status"] == "SUCCEEDED"
    assert calls == [(scope[0], "NAV"), (scope[1], "NAV"), (scope[1], ["FX"]), (scope[1], "FX")]
    assert read_supplement_work(tmp_path / "night", state, instant)["status"] == "SUCCEEDED"
    calls.clear()
    assert supplements.process(state, refresh_dependents=True)["status"] == "SUCCEEDED"
    assert calls == []


def test_supplement_work_returns_unattempted_expired_and_unproven_items(tmp_path, monkeypatch):
    from app.commands import fund_daily_check

    monkeypatch.setattr(fund_daily_check, "PROJECT", tmp_path)
    day = "2026-11-23"
    instant = datetime.fromisoformat(day + "T00:50:00+08:00")
    code = "000001"
    window = {"base_nav_date": "2026-11-20", "target_nav_date": day}
    state = {"scope": [code], "window": window, "batch": {"job_id": "original"}, "checked_at": instant.isoformat()}
    header = {"job_id": "original", "window": window}
    proof = tmp_path / "original.txt"
    proof.write_text("received source evidence", encoding="utf-8")
    common = {"fund_code": code, "required_date": day, "attempted_at": instant.isoformat(),
              "attempted_source": "actual source", "next_source": "official record", "summary": "result",
              "evidence_ref": str(proof)}
    rows = [
        {**common, "field": "verified", "status": "REUSED_VERIFIED"},
        {**common, "field": "missing_proof", "status": "NEW_VERIFIED", "evidence_ref": "missing.txt"},
        {**common, "field": "not_checked", "status": "NOT_ATTEMPTED"},
        {**common, "field": "future", "status": "NOT_PUBLISHED", "retry_at": day + "T11:00:00+08:00"},
        {**common, "field": "expired", "status": "NOT_PUBLISHED", "retry_at": day + "T00:40:00+08:00"},
    ]
    versioned_save(tmp_path / "supplement-catalog.json", {**header, "tasks": rows})
    versioned_save(tmp_path / "supplement-tasks.json", {**header, "tasks": rows[2:]})
    versioned_save(tmp_path / "source-rechecks.json", {**header, **window, "day": day, "items": rows})
    versioned_save(tmp_path / "supplement-funds" / f"{code}.json",
                   {**header, "day": day, "fund_code": code, "items": rows})
    versioned_save(tmp_path / "supplement-execution.json", {
        **header, **window, "day": day, "item_count": len(rows), "pending_item_count": 3,
        "counts": {"REUSED_VERIFIED": 1, "NEW_VERIFIED": 1, "NOT_ATTEMPTED": 1, "NOT_PUBLISHED": 2},
    })

    report = read_supplement_work(tmp_path, state, instant, code)
    assert report["status"] == "CONTINUE"
    assert report["counts"] == {"REUSED_VERIFIED": 1, "CONTINUE": 3, "WAITING": 1}
    assert {row["field"]: row["outcome"] for row in report["funds"][code]["remaining"]} == {
        "missing_proof": "CONTINUE", "not_checked": "CONTINUE", "future": "WAITING", "expired": "CONTINUE",
    }


def test_shibor_only_uses_exact_dates_and_waits_for_the_publication(monkeypatch):
    from app.services import fund_daily_supplements as module

    day = "2026-11-23"
    base = "2026-11-20"
    before = datetime.fromisoformat(day + "T10:59:59+08:00")
    after = datetime.fromisoformat(day + "T11:00:01+08:00")
    runner = object.__new__(DailySupplements)
    runner.day = day
    runner.window = {"base_nav_date": base}
    calls = []

    def lookup(url, params):
        calls.append(params)
        return {"head": {"rep_code": "200"}, "records": [
            {"showDateCN": base, "ON": "1.20", "1W": "1.35"},
            {"showDateCN": day, "ON": "1.21", "1W": "1.36"},
            {"showDateCN": "2026-11-19", "ON": "9.0", "1W": "9.0"},
        ]}, {"raw_path": "official.json"}

    runner.request_json = lookup
    monkeypatch.setattr(module, "now", lambda: before)
    result = runner.official_shibor(day)
    assert result["status"] == "NOT_PUBLISHED"
    assert result["retry_at"] == day + "T11:00:00+08:00"
    assert not calls
    monkeypatch.setattr(module, "now", lambda: after)
    result = runner.official_shibor(day)
    assert result["status"] == "NEW_VERIFIED"
    assert set(result["observations"]["rates_pct"]) == {base, day}
    assert calls == [{"startDate": base, "endDate": day}]


def test_complete_etf_market_data_does_not_verify_benchmark_tracking():
    runner = object.__new__(DailySupplements)
    runner.day = "2026-11-23"
    runner.profiles = {"001632": {"target_funds": ["159736"]}}
    observations = {
        "fund_share": [["159736.SZ", "20261119", 120], ["159736.SZ", "20261120", 123]],
        "fund_nav": [["159736.SZ", "20261122", "20261120", 1.02]],
        "fund_daily": [["159736.SZ", "20261120", 1.03, 1.03, 0.0, 1000.0]],
    }
    runner.provider_response = lambda api, symbol, params, fields: (
        {"code": 0, "data": {"fields": fields, "items": observations[api]}}, {"raw_path": api + ".json"}
    )
    result = runner.target_market(
        {"target_fund": "159736", "check_key": "prediction_target_159736"},
        "001632", "2026-11-20", [date(2026, 11, 19), date(2026, 11, 20)],
    )
    assert result["status"] == "REVIEW_REQUIRED"
    assert "跟踪偏离" in result["summary"]
    assert result["observations"]["shares"]["change_ten_thousand"] == "3"
    assert "market_premium_pct" in result["observations"]


def test_repo_month_coverage_uses_observed_term():
    from app.services.fund_daily_source_routes import cfets_repo_rows

    at = datetime.fromisoformat("2026-11-23T11:20:00+08:00")
    _, rows = cfets_repo_rows({
        "head": {"rep_code": "200"}, "data": {"showDateCN": "2026-11-23 11:15"},
        "records": [
            {"date": "26-11-23", "productCode": "R007", "weightedRate": "1.40", "avgPrd": "7"},
            {"date": "26-11-23", "productCode": "R014", "weightedRate": "1.50", "avgPrd": "14"},
        ],
    }, "2026-11-23", at)
    assert not rows["R007"]["covers_month_start"]
    assert rows["R014"]["covers_month_start"]
