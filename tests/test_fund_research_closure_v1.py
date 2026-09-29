"""覆盖切片不能制造缺口完成；拟合必须等所有独立条件通过。"""

import pytest
from app.services.fund_research_closure_v1 import admission_status, project_coverage, split_windows


def test_split_leap_day_and_overlap_preserve_union():
    rows = [
        {"stock": "000001", "start": "2020-02-01", "end": "2020-03-15"},
        {"stock": "000001", "start": "2020-03-10", "end": "2020-03-31"},
    ]
    assert split_windows(rows) == [
        {"stock": "000001", "start": "2020-02-01", "end": "2020-03-01"},
        {"stock": "000001", "start": "2020-03-02", "end": "2020-03-31"},
    ]


def test_gap_day_or_failed_page_cannot_become_complete():
    rows = [
        {"fund_code": "002112", "target": "2023-05-01", "catalog_complete": False, "remaining_companies": ["000001"]}
    ]
    windows = split_windows([{"stock": "000001", "start": "2023-04-01", "end": "2023-04-30"}], 15)
    catalogs = [{"window": w, "catalog_complete": True} for w in windows]
    assert project_coverage(rows, catalogs)[0]["catalog_complete"]
    catalogs[1]["catalog_complete"] = False
    assert not project_coverage(rows, catalogs)[0]["catalog_complete"]


def test_large_company_group_is_not_dropped():
    gaps = [{"stock": f"{i:06}", "start": "2023-04-01", "end": "2023-04-30"} for i in range(199)]
    assert len(split_windows(gaps)) == 199


@pytest.mark.parametrize("start,end", [("2025-01-01", "2025-01-02"), ("2023-01-02", "2023-01-01")])
def test_out_of_scope_queries_rejected(start, end):
    with pytest.raises(ValueError):
        split_windows([{"stock": "000001", "start": start, "end": end}])


@pytest.mark.parametrize("missing", ["catalog_complete", "body_complete", "semantics_complete", "protocol_frozen"])
def test_any_missing_admission_condition_blocks_fit(missing):
    args = dict(catalog_complete=True, body_complete=True, semantics_complete=True, protocol_frozen=True, fit_budget=12)
    args[missing] = False
    assert not admission_status(**args)["fit_execution_allowed"]


@pytest.mark.parametrize("budget", [0, 13, True])
def test_unbounded_or_invalid_budget_rejected(budget):
    assert not admission_status(
        catalog_complete=True, body_complete=True, semantics_complete=True, protocol_frozen=True, fit_budget=budget
    )["fit_execution_allowed"]
