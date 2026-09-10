"""专项数据契约：新日期范围、旧数学、未来隔离及不可覆盖研究包。"""

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from app.repositories.cash_reinvestment_samples import CashNavPoint
from app.schemas.cash_reinvestment_samples import CashSampleRequest
from app.schemas.direction_training import DirectionAnswer, DirectionInput
from app.services import direction_training_artifacts as artifacts
from app.services import direction_training_dataset as data
from app.services.direction_training_inventory import local_snapshot
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.trading_calendar import load_calendar
from pydantic import ValidationError
from tests.test_cash_reinvestment_samples import dividend, preview, rows_for
from tests.test_trading_nav_window import SOURCE


def test_new_formula_matches_old_cash_label_and_lag_does_not_extend_twenty_sessions():
    cutoff = date(2023, 6, 20)
    nav = rows_for()
    event = dividend(date(2023, 6, 21))
    old = preview(nav, (event,))
    item, issues = data.build_input("006730", cutoff, SOURCE, nav, (event,))
    answer, problems = data.build_answer("006730", cutoff, nav, (event,), include_value=True)
    assert not issues and not problems
    assert item.anchor == date(2023, 6, 19) < cutoff
    assert item.x == tuple(float(old.feature_payload.metrics[k]) for k in FEATURE_NAMES)
    assert answer.end == old.label_end_date
    assert answer.future_return == old.offline_label.future_return_20d
    assert answer.y == old.offline_label.label_up_20d
    meta, _ = data.build_answer("006730", cutoff, nav, (event,), include_value=False)
    assert set(meta) == {"fund", "cutoff", "end", "available_at"}


def test_future_nav_and_unannounced_dividend_do_not_change_historical_input():
    cutoff = date(2023, 6, 20)
    nav = rows_for()
    before, _ = data.build_input("006730", cutoff, SOURCE, nav, ())
    changed = tuple(replace(p, unit_nav=Decimal("99")) if p.nav_date > cutoff else p for p in nav)
    unknown = dividend(date(2023, 6, 19), ann_date=date(2023, 6, 21))
    after, _ = data.build_input("006730", cutoff, SOURCE, changed, (unknown,))
    assert before == after
    late = tuple(replace(p, ann_date=date(2023, 7, 1)) if p.nav_date == before.anchor else p for p in nav)
    item, issues = data.build_input("006730", cutoff, SOURCE, late, ())
    assert item is None and issues


def test_2021_is_only_extended_by_new_contract_not_old_api():
    cutoff = date(2021, 8, 20)
    nav = tuple(
        CashNavPoint(d, d, Decimal(10) + Decimal(i) / 100)
        for i, d in enumerate(load_calendar().sessions)
        if date(2021, 1, 1) <= d <= date(2021, 9, 30)
    )
    item, issues = data.build_input("001632", cutoff, SOURCE, nav, ())
    assert item and not issues
    with pytest.raises(ValidationError):
        CashSampleRequest(fundCode="001632", cutoffDate=cutoff)


def test_protected_period_rejected_before_building_values(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not inspect values")

    monkeypatch.setattr(data, "build_cash_return_series", forbidden)
    with pytest.raises(ValueError, match="PROTECTED"):
        data.build_answer("001632", date(2025, 1, 2), (), (), include_value=True)
    answer, issues = data.build_answer("001632", date(2024, 12, 20), (), (), include_value=True)
    assert answer is None and issues == ["LABEL_CROSSES_PROTECTED_PERIOD"]
    with pytest.raises(ValidationError):
        DirectionInput(
            fund="001632",
            cutoff="2025-01-02",
            anchor="2024-12-31",
            available_at="2024-12-31",
            x=[0] * 7,
            input_hash="a" * 64,
        )


def test_flat_answer_is_non_up_and_direction_must_match_return():
    cutoff = date(2023, 6, 20)
    nav = tuple(replace(p, unit_nav=Decimal(10)) for p in rows_for())
    answer, issues = data.build_answer("001632", cutoff, nav, (), include_value=True)
    assert not issues and answer.y == 0 and Decimal(answer.future_return) == 0
    with pytest.raises(ValidationError):
        DirectionAnswer(**{**answer.model_dump(), "y": 1})


def test_calendar_exclusion_is_planned_before_source_availability():
    total, planned = data.exam_dates(date(2023, 12, 31), date(2024, 3, 31))
    assert len(total) == 58 and len(planned) == 38
    assert all(load_calendar().future_sessions(d)[-1] <= date(2024, 3, 31) for d in planned)


def test_coverage_retains_missing_inputs_and_late_labels(monkeypatch):
    monkeypatch.setattr(data, "WINDOWS", (("W", "2022-12-31", "2023-06-30", "2023-09-30"),))
    monkeypatch.setattr(data, "restore_source", lambda snapshot, fund: (SOURCE, (), ()))
    late = date(2023, 8, 1)

    def item(fund, cutoff, *args):
        if cutoff == date(2023, 7, 3):
            return None, ["MISSING_NAV"]
        return DirectionInput(
            fund=fund, cutoff=cutoff, anchor=cutoff, available_at=cutoff, x=[1] * 7, input_hash="a" * 64
        ), []

    def answer(fund, cutoff, *args, **kwargs):
        return {"available_at": str(date(2023, 10, 1) if cutoff == late else cutoff + timedelta(days=30))}, []

    monkeypatch.setattr(data, "build_input", item)
    monkeypatch.setattr(data, "build_answer", answer)
    report, _, rows = data.build_coverage({})
    group = report["windows"]["W"]["funds"]["001632"]["EXAM"]
    assert group["planned"] == 44 and group["usable"] < 44
    assert group["issue_counts_nonexclusive"]["MISSING_NAV"] == 1
    assert group["issue_counts_nonexclusive"]["ANSWER_NOT_MATURE_BY_STAGE_END"] >= 1
    assert all(not r["label"] or "y" not in r["label"] for r in rows)


def test_artifact_seal_corruption_overwrite_path_and_budget(tmp_path, monkeypatch):
    files = {"result.json": artifacts.write_json(tmp_path / "result.json", {"n": 1})}
    artifacts.seal(tmp_path, "complete.json", files, status="VERIFIED")
    assert artifacts.read_seal(tmp_path, "complete.json")["status"] == "VERIFIED"
    with pytest.raises(FileExistsError):
        artifacts.write_json(tmp_path / "result.json", {"n": 2})
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="HASH"):
        artifacts.read_seal(tmp_path, "complete.json")
    with pytest.raises(ValueError, match="PATH"):
        artifacts.verify_files(tmp_path, {"../outside.json": "a" * 64})
    monkeypatch.setattr(artifacts, "MAX_ROWS", 2)
    with pytest.raises(ValueError, match="BUDGET"):
        artifacts.write_jsonl(tmp_path / "large.jsonl", [{}, {}, {}])
    with pytest.raises(ValueError):
        artifacts.run_folder(UUID(int=1), prefix="../escape")


def test_local_database_scope_rejected_before_connection(monkeypatch):
    from types import SimpleNamespace

    from app.services import direction_training_inventory as inventory

    engine = SimpleNamespace(url=SimpleNamespace(host="remote.example", database="fund_ai"))
    monkeypatch.setattr(inventory, "get_nav_sample_storage_engine", lambda: engine)
    with pytest.raises(ValueError, match="LOCAL_FUND_AI_REQUIRED"), local_snapshot():
        pytest.fail("must not connect")
