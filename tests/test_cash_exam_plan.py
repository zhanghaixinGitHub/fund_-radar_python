"""独立日历分母、资料覆盖、旧快照兼容及2025保护；不对真实测试集评分。"""

import copy
from dataclasses import replace
from datetime import date, timedelta
from decimal import localcontext

import pytest
from app.api.routes import cash_reinvestment_storage as api
from app.schemas.cash_exam_plan import CashExamPlan, CashExamPreparation, CashExamPreparationRequest
from app.schemas.cash_policy_freeze import CashPlannedRuleBinding
from app.services import cash_exam_preparation as service
from app.services import cash_policy_freeze as freeze
from app.services import cash_release_policy as policy_source
from app.services.cash_exam_plan import build_cash_exam_plan, validate_cash_exam_plan
from app.services.cash_reinvestment_storage import cash_hash
from app.services.historical_nav_storage import HistoricalNavStorageError
from app.services.trading_calendar import load_calendar
from pydantic import ValidationError
from tests.test_cash_policy_freeze import row_for
from tests.test_cash_reinvestment_research import data as data
from tests.test_cash_reinvestment_research import small_dataset
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

URL = "/internal/v1/features/cash-reinvestment/exam-preparation"


def request_for(data=None):
    data = data or small_dataset()
    return CashExamPreparationRequest(
        batchIds=data.report.batch_ids,
        expectedDatasetHash=data.report.dataset_hash,
        expectedPlanHash=build_cash_exam_plan().plan_hash,
    )


def test_calendar_only_plan_is_fixed_and_covers_every_possible_cutoff():
    plan = build_cash_exam_plan()
    assert plan == build_cash_exam_plan()
    assert plan.fund_codes == ("001632", "006730", "008888")
    assert [len(w.planned_cutoffs) for w in plan.windows] == [44, 40, 222, 223]
    calendar = load_calendar()
    for window in plan.windows:
        eligible, purged = [], []
        end_index = calendar.at_or_before_index(window.end_date)
        for index, day in enumerate(calendar.sessions):
            if window.start_date <= day <= window.end_date:
                (eligible if index + 20 <= end_index else purged).append(day)
        assert window.planned_cutoffs == tuple(eligible)
        assert window.boundary_purged_cutoffs == tuple(purged)
    assert plan.plan_hash == cash_hash(plan.model_dump(mode="json", exclude={"plan_hash"}))


def test_descriptor_binds_plan_without_database_or_approval(monkeypatch, client):
    monkeypatch.setattr(freeze, "get_nav_sample_storage_engine", lambda: pytest.fail("no DB"))
    response = client.get("/internal/v1/predictions/release-policy", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["binding"]["exam_plan"]["plan_hash"] == build_cash_exam_plan().plan_hash
    descriptor = freeze.describe_cash_policy()
    assert isinstance(descriptor.binding, CashPlannedRuleBinding)
    assert not descriptor.approval_ready and not descriptor.publication_allowed


@pytest.mark.parametrize("mutation", ["date", "fund", "calendar", "window", "hash"])
def test_even_rehashed_plan_cannot_remove_or_change_dates(mutation):
    raw = build_cash_exam_plan().model_dump(mode="json")
    if mutation == "date":
        raw["windows"][0]["planned_cutoffs"].pop(0)
    elif mutation == "fund":
        raw["fund_codes"].reverse()
    elif mutation == "calendar":
        raw["calendar_hash"] = "e" * 64
    elif mutation == "window":
        raw["windows"][0]["window_id"] = "CHOSEN_WINNER"
    raw["plan_hash"] = "f" * 64 if mutation == "hash" else cash_hash({k: v for k, v in raw.items() if k != "plan_hash"})
    with pytest.raises(ValueError):
        validate_cash_exam_plan(CashExamPlan.model_validate(raw))


@pytest.mark.parametrize("legacy", [False, True])
def test_historical_snapshots_do_not_depend_on_todays_approval_file(monkeypatch, tmp_path, legacy):
    row = row_for()
    if legacy:
        row.snapshot["binding"].pop("exam_plan")
        row.snapshot["binding"]["version"] = "CASH_RELEASE_RULE_BINDING_V1"
        row.binding_hash = cash_hash(row.snapshot["binding"])
        row.content_hash = freeze._content_hash(row)
    before = copy.deepcopy(row.snapshot)
    monkeypatch.setattr(policy_source, "POLICY_PATH", tmp_path / "missing.json")
    frozen = freeze.restore_policy_freeze(row)
    assert frozen.snapshot.model_dump(mode="json") == before
    if legacy:
        with pytest.raises(HistoricalNavStorageError) as error:
            freeze.validate_policy_binding(frozen, frozen.snapshot.policy)
        assert error.value.code == "CASH_POLICY_FREEZE_MISMATCH"


def test_changed_date_list_cannot_be_smuggled_into_rehashed_freeze():
    row = row_for()
    plan = row.snapshot["binding"]["exam_plan"]
    plan["windows"][0]["planned_cutoffs"].pop(0)
    plan["plan_hash"] = cash_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    row.binding_hash = cash_hash(row.snapshot["binding"])
    row.content_hash = freeze._content_hash(row)
    with pytest.raises(HistoricalNavStorageError) as error:
        freeze.restore_policy_freeze(row)
    assert error.value.code == "CASH_POLICY_FREEZE_CORRUPTED"


def test_preparation_uses_research_selection_but_never_reads_x_y(data):
    plan = build_cash_exam_plan()
    original = service.summarize_cash_exam_preparation(data, plan)
    poisoned = replace(data, rows=tuple(replace(r, x=object(), y=object()) for r in reversed(data.rows)))
    with localcontext() as context:
        context.prec = 6
        after = service.summarize_cash_exam_preparation(poisoned, plan)
    assert original == after
    assert len(original.coverage) == 12 and not original.ex_ante_evidence and not original.model_fitted
    protected = original.coverage[-3:]
    assert all(c.status == "TEST_PERIOD_PROTECTED" and c.coverage is None and c.usable_count is None for c in protected)
    assert all(c.missing_cutoffs is None and c.late_label_count is None for c in protected)


def test_reducing_selected_data_never_shrinks_denominator(data):
    plan = build_cash_exam_plan()
    original = service.summarize_cash_exam_preparation(data, plan)
    subset = replace(data, rows=tuple(r for r in data.rows if r.fund_code == "006730" and r.as_of_date.day <= 15))
    partial = service.summarize_cash_exam_preparation(subset, plan)
    assert partial.plan == original.plan
    for full, fewer in zip(original.coverage[:9], partial.coverage[:9], strict=True):
        assert full.planned_count == fewer.planned_count and full.usable_count >= fewer.usable_count
        if fewer.fund_code != "006730":
            assert fewer.usable_count == fewer.coverage == 0
            assert len(fewer.missing_cutoffs) == fewer.planned_count


def test_delayed_label_counts_as_missing_not_denominator_reduction(data):
    plan = build_cash_exam_plan()
    first = plan.windows[0]
    row = next(r for r in data.rows if r.fund_code == "001632" and r.as_of_date == first.planned_cutoffs[0])
    delayed = replace(row, label_available_at=first.end_date + timedelta(days=1))
    changed = replace(data, rows=tuple(delayed if r is row else r for r in data.rows))
    before = service.summarize_cash_exam_preparation(data, plan).coverage[0]
    after = service.summarize_cash_exam_preparation(changed, plan).coverage[0]
    assert before.planned_count == after.planned_count
    assert after.usable_count == before.usable_count - 1
    assert after.late_label_count == before.late_label_count + 1
    assert row.as_of_date in after.missing_cutoffs


@pytest.mark.parametrize("change", ["duplicate", "fund", "cutoff", "future"])
def test_bad_matrix_identity_or_test_dates_fail_closed(data, change):
    row = data.rows[0]
    if change == "duplicate":
        changed = replace(data, rows=(*data.rows, row))
    else:
        updates = {
            "fund": {"fund_code": "999999"},
            "cutoff": {"as_of_date": row.as_of_date - timedelta(days=1)},
            "future": {"label_available_at": date(2025, 1, 1)},
        }[change]
        changed = replace(data, rows=(replace(row, **updates), *data.rows[1:]))
    with pytest.raises(ValueError):
        service.summarize_cash_exam_preparation(changed, build_cash_exam_plan())


def test_plan_mismatch_stops_before_data_load(monkeypatch):
    monkeypatch.setattr(service, "load_cash_dataset", lambda _: pytest.fail("plan must be checked first"))
    with pytest.raises(HistoricalNavStorageError) as error:
        service.prepare_cash_exam_data(request_for().model_copy(update={"expected_plan_hash": "f" * 64}))
    assert error.value.code == "EXAM_PLAN_HASH_MISMATCH"


def test_dataset_hash_conflict_and_http_success(monkeypatch, client):
    data = small_dataset()
    monkeypatch.setattr(service, "load_cash_dataset", lambda _: data)
    request = request_for(data)
    first = client.post(URL, headers=HEADERS, json=request.model_dump(mode="json", by_alias=True))
    assert first.status_code == 200 and first.headers["cache-control"] == "no-store"
    result = CashExamPreparation.model_validate(first.json())
    assert not result.database_written and not result.independent_test_read and not result.publication_allowed
    conflict = request.model_copy(update={"expected_dataset_hash": "f" * 64})
    second = client.post(URL, headers=HEADERS, json=conflict.model_dump(mode="json", by_alias=True))
    assert second.status_code == 409 and second.json()["detail"]["code"] == "DATASET_HASH_MISMATCH"


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "bad"}, {**HEADERS, "Origin": "http://localhost"}])
def test_http_auth_before_preparation(client, monkeypatch, headers):
    monkeypatch.setattr(api, "prepare_cash_exam_data", lambda _: pytest.fail("no data load"))
    response = client.post(URL, headers=headers, json=request_for().model_dump(mode="json", by_alias=True))
    assert response.status_code == 403


@pytest.mark.parametrize(
    "extra", [{"includeTest": True}, {"coverage": 1}, {"force": True}, {"plan": {}}, {"approved": True}]
)
def test_client_cannot_supply_test_switch_or_coverage(client, monkeypatch, extra):
    monkeypatch.setattr(api, "prepare_cash_exam_data", lambda _: pytest.fail("no data load"))
    response = client.post(URL, headers=HEADERS, json={**request_for().model_dump(mode="json", by_alias=True), **extra})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field", ["ex_ante_evidence", "publication_allowed", "model_fitted", "independent_test_read", "database_written"]
)
def test_response_cannot_claim_side_effect_or_prior_freeze(field):
    result = service.summarize_cash_exam_preparation(small_dataset(), build_cash_exam_plan())
    with pytest.raises(ValidationError):
        CashExamPreparation.model_validate({**result.model_dump(mode="json"), field: True})


def test_response_cannot_replace_missing_fund_with_duplicate_group(data):
    result = service.summarize_cash_exam_preparation(data, build_cash_exam_plan()).model_dump(mode="json")
    result["coverage"][1] = result["coverage"][0]
    with pytest.raises(ValidationError):
        CashExamPreparation.model_validate(result)
