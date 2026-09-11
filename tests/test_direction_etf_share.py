"""份额日期可用性、拆分排除、同样本对照和真实隔离进程的数值验证。"""

import sys
from copy import deepcopy
from datetime import date

import pytest
from app.services import direction_etf_share_data as d
from app.services import direction_etf_share_models as m
from app.services import direction_etf_share_runner as r
from app.services import direction_rolling_models as rolling
from app.services.direction_linear_models import restore as old_restore
from app.services.direction_linear_models import validate_job as old_validate
from app.services.direction_linear_protocol import planned_dates, specification_for, study_windows
from app.services.direction_training_artifacts import digest
from app.services.direction_training_process import run_process
from app.services.trading_calendar import load_calendar


def input_row(fund, day, i=0):
    calendar = load_calendar()
    return {
        "fund": fund,
        "cutoff": str(day),
        "anchor": str(calendar.sessions[calendar.at_or_before_index(day) - 1]),
        "available_at": str(day),
        "x": [(i * j + 3) % 19 / 19 for j in range(1, 8)],
        "input_hash": "a" * 64,
    }


@pytest.fixture
def payload():
    calendar = load_calendar()
    days = [day for day in calendar.sessions if day >= date(2021, 10, 13)][:270]
    fit = []
    for fund in r.FUNDS:
        for i, day in enumerate(days):
            future = calendar.future_sessions(day, 21)
            fit.append(
                {
                    "input": input_row(fund, day, i),
                    "answer": {
                        "fund": fund,
                        "cutoff": str(day),
                        "end": str(future[-2]),
                        "available_at": str(future[-1]),
                        "y": i % 2,
                        "future_return": "0.01" if i % 2 else "-0.01",
                    },
                }
            )
    return {
        "version": m.VERSION,
        "branch": "REFERENCE",
        "window": study_windows(m.VERSION)[0],
        "fit": fit,
        "exam": [input_row(f, day) for f in r.FUNDS for day in (date(2023, 1, 3), date(2023, 3, 31))],
    }


def history():
    return {
        fund: {
            day.strftime("%Y%m%d"): str(100 + i)
            for i, day in enumerate(load_calendar().sessions)
            if day.year in d.YEARS
        }
        for fund in r.FUNDS
    }


def test_feature_uses_second_prior_session_and_five_session_change():
    shares = history()
    evidence, reason = d.feature(shares, "001632", "2023-01-12")
    assert reason is None
    assert evidence["dates"] == ["20230103", "20230104", "20230105", "20230106", "20230109", "20230110"]
    assert evidence["assumed_available_at"] == "2023-01-11"
    assert evidence["value"] == pytest.approx(float(evidence["shares_wan"][-1]) / float(evidence["shares_wan"][0]) - 1)
    shares["001632"]["20230111"] = "999999999"
    shares["001632"]["20230112"] = "999999999"
    assert d.feature(shares, "001632", "2023-01-12")[0] == evidence


def test_missing_middle_date_is_not_filled():
    shares = history()
    del shares["001632"]["20230106"]
    assert d.feature(shares, "001632", "2023-01-12") == (None, "ETF_SHARE_HISTORY_UNAVAILABLE")


def test_known_split_window_is_removed_without_adjusting_other_etfs():
    shares = history()
    assert d.feature(shares, "006730", "2022-09-01") == (None, "ETF_SHARE_SPLIT_WINDOW")
    assert d.feature(shares, "001632", "2022-09-01")[0] is not None
    assert d.feature(shares, "006730", "2022-09-13")[0] is not None


@pytest.mark.parametrize("amount", ["0", "-1", "NaN", "Infinity"])
def test_source_rejects_non_positive_and_non_finite_amounts(amount):
    with pytest.raises(ValueError, match="ETF_SHARE_SOURCE_VALUE"):
        d.normalize([["159736.SZ", "20230103", amount]], "001632")


def test_source_deduplication_calendar_and_listing_audit():
    rows = [
        ["159736.SZ", day, amount]
        for day, amount in (("20230103", "100.0"), ("20230103", "100"), ("20221231", "50"), ("20210924", "25"))
    ]
    kept, audit = d.normalize(rows, "001632")
    assert kept == {"20230103": "100"}
    assert audit["non_calendar_dates"] == ["20221231"]
    assert audit["pre_listing_dates"] == ["20210924"]
    with pytest.raises(ValueError, match="ETF_SHARE_SOURCE_CONFLICT"):
        d.normalize([*rows, ["159736.SZ", "20230103", "101"]], "001632")
    with pytest.raises(ValueError, match="ETF_SHARE_SOURCE_IDENTITY"):
        d.normalize([["159995.SZ", "20230103", "100"]], "001632")
    with pytest.raises(ValueError, match="ETF_SHARE_SOURCE_VALUE"):
        d.normalize([["159736.SZ", "20250103", "100"]], "001632")


@pytest.mark.parametrize("cutoff", ["2023-01-01", "2025-01-03"])
def test_feature_rejects_out_of_plan_dates(cutoff):
    with pytest.raises(ValueError, match="ETF_SHARE_FEATURE_CUTOFF"):
        d.feature(history(), "001632", cutoff)


@pytest.mark.parametrize(
    "mutation",
    [
        "future_label",
        "publication",
        "exam_label",
        "exam_quarter",
        "anchor",
        "duplicate",
        "unbalanced",
        "dimension",
        "window",
        "extra_field",
    ],
)
def test_worker_rejects_leakage_and_unplanned_contract(payload, mutation):
    if mutation == "future_label":
        payload["fit"][0]["answer"]["available_at"] = "2023-01-03"
    elif mutation == "publication":
        payload["fit"][0]["answer"]["available_at"] = payload["fit"][0]["answer"]["end"]
    elif mutation == "exam_label":
        payload["exam"][0]["y"] = 1
    elif mutation == "exam_quarter":
        payload["exam"][1] = input_row(r.FUNDS[0], date(2023, 4, 3))
    elif mutation == "anchor":
        payload["exam"][1]["anchor"] = payload["exam"][1]["cutoff"]
    elif mutation == "duplicate":
        payload["exam"][1] = payload["exam"][0]
    elif mutation == "unbalanced":
        payload["fit"].pop()
    elif mutation == "dimension":
        payload["exam"][0]["x"].append(0.0)
    elif mutation == "window":
        payload["window"]["fit_end"] = "2023-01-03"
    else:
        payload["answers"] = []
    with pytest.raises(ValueError):
        m.validate_job(payload)


def test_worker_preserves_original_algorithm_and_rejects_tampered_model(payload):
    output = run_process(payload, command=[sys.executable, "-m", "scripts.direction_etf_share_worker"])
    assert output["status"] == "PREDICTED"
    old = rolling.execute_job(
        {
            "version": rolling.VERSION,
            "month": "2023-01",
            "recipe": "ALL",
            "fit": payload["fit"],
            "exam": [i for i in payload["exam"] if i["cutoff"].startswith("2023-01")],
        }
    )
    for key in ("mean", "scale", "coefficients", "intercept", "solver_iterations", "train_hash", "train_counts"):
        assert output["model"][key] == old["model"][key]
    assert output["scores"][::2] == old["scores"]
    assert output["model"]["train_hash"] == digest(payload["fit"])
    bad = deepcopy(output["model"])
    bad["intercept"] += 1
    with pytest.raises(ValueError, match="ETF_SHARE_MODEL_HASH"):
        m.restore(bad)
    bad["hash"] = digest({k: v for k, v in bad.items() if k != "hash"})
    bad["fit_end"] = "2023-01-03"
    with pytest.raises(ValueError, match="ETF_SHARE_MODEL_PROTOCOL"):
        m.restore(bad)


def test_eighth_feature_works_in_real_worker_and_is_bound_in_input_hash(payload):
    shares = history()
    payload["branch"] = "ETF_SHARE_CHANGE"
    for row in payload["fit"]:
        item = row["input"]
        ev, _ = d.feature(shares, item["fund"], item["cutoff"])
        # 人工数据集避开拆分只为验证八列worker，不复用任何真实研究结果。
        ev = ev or {"value": 0.0, "dates": [], "etf": d.ETF_CODES[item["fund"]]}
        row["input"] = d.add_feature(item, ev)
    payload["exam"] = [d.add_feature(i, d.feature(shares, i["fund"], i["cutoff"])[0]) for i in payload["exam"]]
    output = run_process(payload, command=[sys.executable, "-m", "scripts.direction_etf_share_worker"])
    assert output["status"] == "PREDICTED"
    assert len(output["model"]["coefficients"]) == 8
    assert output["model"]["train_hash"] == digest(payload["fit"])
    assert all(i["input_hash"] != "a" * 64 for i in payload["exam"])
    assert output["scores"] == m.predict_model(output["model"], m.validate_job(payload)[1])


def test_old_entrypoints_cannot_load_new_time_or_feature_contract(payload):
    with pytest.raises(ValueError, match="ETF_SHARE_REQUIRES_DEDICATED_WORKER"):
        old_validate(payload)
    with pytest.raises(ValueError, match="ETF_SHARE_REQUIRES_DEDICATED_RESTORE"):
        old_restore({"version": m.VERSION})
    with pytest.raises(ValueError):
        specification_for(m.VERSION)


@pytest.mark.parametrize("mutation", ["permission", "truncated", "year"])
def test_history_rejects_denial_truncation_and_wrong_request_year(tmp_path, monkeypatch, mutation):
    record = {
        "api_name": "fund_share",
        "business_code": 0,
        "http_status": 200,
        "params": {"ts_code": "159736.SZ", "start_date": "20210101", "end_date": "20211231"},
        "data": {
            "fields": ["ts_code", "trade_date", "fd_share"],
            "items": [["159736.SZ", "20210927", "100"]],
            "has_more": False,
        },
    }
    if mutation == "permission":
        record["business_code"] = 40203
    elif mutation == "truncated":
        record["data"]["has_more"] = True
    else:
        record["data"]["items"][0][1] = "20220927"
    monkeypatch.setattr(d, "read_json", lambda path: record)
    with pytest.raises(ValueError, match="ETF_SHARE_SOURCE_"):
        d.load_history(tmp_path)


def test_common_availability_removes_same_dates_from_all_funds_and_both_groups(payload, tmp_path, monkeypatch):
    windows = study_windows(m.VERSION)
    exams = {w["name"]: [input_row(f, day) for f in r.FUNDS for day in planned_dates(m.VERSION, w)] for w in windows}
    pool = [*payload["fit"], *({"input": i} for rows in exams.values() for i in rows)]
    missing = payload["fit"][0]["input"]["cutoff"]
    missing_exam = exams[windows[0]["name"]][0]["cutoff"]
    monkeypatch.setattr(r, "load_history", lambda folder: ({}, {}))
    monkeypatch.setattr(r, "read_jsonl", lambda path: pool)
    monkeypatch.setattr(r, "select_fit", lambda *args: deepcopy(payload["fit"]))
    monkeypatch.setattr(
        r,
        "feature",
        lambda hist, fund, cutoff: (
            (None, "ETF_SHARE_HISTORY_UNAVAILABLE")
            if fund == r.FUNDS[0] and cutoff in (missing, missing_exam)
            else ({"value": 0.125, "etf": d.ETF_CODES[fund]}, None)
        ),
    )

    def prepared(path):
        name = path.name.removeprefix("source-linear-prepared-").removesuffix(".json")
        window = next(w for w in windows if w["name"] == name)
        return {
            "complete": {"exam": {"CLEAN": exams[name]}},
            "planned": [str(day) for day in planned_dates(m.VERSION, window)],
        }

    monkeypatch.setattr(r, "read_json", prepared)
    bundles, _ = r.build_bundles(tmp_path)
    for bundle in bundles.values():
        a, b = (bundle["jobs"][branch] for branch in r.BRANCHES)
        assert len(a["fit"]) == len(b["fit"]) == 269 * 3
        assert all(row["input"]["cutoff"] != missing for row in a["fit"])
        assert all(i["cutoff"] != missing_exam for i in a["exam"])
        for left, right in zip(a["fit"], b["fit"], strict=True):
            assert left["answer"] == right["answer"]
            assert left["input"]["x"] == right["input"]["x"][:7]
            assert left["input"]["cutoff"] == right["input"]["cutoff"]
        assert [(i["fund"], i["cutoff"]) for i in a["exam"]] == [(i["fund"], i["cutoff"]) for i in b["exam"]]


@pytest.mark.parametrize("marker", ["etf-share-replay-started.json", "etf-share-replay-origin.json"])
def test_replay_budget_is_one_and_cannot_replay_a_replay(tmp_path, monkeypatch, marker):
    (tmp_path / marker).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(r, "new_folder", lambda: pytest.fail("must not create another run"))
    with pytest.raises(ValueError, match="ETF_SHARE_REPLAY_ALREADY_USED"):
        r.replay(tmp_path)
