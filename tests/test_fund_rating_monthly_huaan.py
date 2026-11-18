from types import SimpleNamespace

import pytest
from app.services.fund_rating_monthly_huaan import deduplicate_fx_sources, pending_performance_bundle


@pytest.mark.parametrize("failure", [None, "conflict", "partial_overlap", "duplicate"])
def test_fx_original_superset_only(failure):
    def html(rows):
        return (
            '<table id="InfoTable"><tr><th>日期</th><th>美元</th></tr>'
            + "".join("<tr><td>" + d + "</td><td>" + v + "</td></tr>" for d, v in rows)
            + "</table>"
        ).encode()

    old = [("2026-09-30", "710")]
    new = [("2026-09-30", "711" if failure == "conflict" else "710"), ("2026-10-30", "712")]
    if failure == "partial_overlap":
        old.insert(0, ("2026-01-01", "709"))
    if failure == "duplicate":
        new.append(new[0])
    source = SimpleNamespace(raw={"old": html(old), "new": html(new)})
    inputs = {"fx_refs": ["old", "new"]}
    if failure:
        with pytest.raises(ValueError):
            deduplicate_fx_sources(source, inputs)
    else:
        assert deduplicate_fx_sources(source, inputs) == ["new"]
        assert inputs["fx_refs"] == ["old", "new"]
        assert "old" in source.raw


@pytest.mark.parametrize("failure", [None, "period", "fund", "component"])
def test_pending_archive_keeps_new_admission_and_calendar(monkeypatch, tmp_path, failure):
    import app.services.fund_rating_monthly_huaan as module
    from app.services import fund_rating_huaan_qdii_relative as relative
    from app.services import fund_rating_pipeline as pipeline

    bundle = {
        "fund_code": "014978",
        "as_of_date": "2026-10-30",
        "monthly_performance_followup": {"pending_performance_source_archive": {"sha256": "exact"}},
    }
    raw = {
        "fund_code": "014978",
        "as_of_date": "2026-10-30",
        "members": {"019524": {}},
        "calendar_ref": "real-727",
        "fx_refs": [],
    }
    pending = {
        "fund_code": "014978",
        "as_of_date": "2026-10-30",
        **{k: {"inputs": dict(raw)} for k in ["huaan_qdii_relative_calculation", "huaan_qdii_benchmark_calculation"]},
    }
    if failure == "period":
        pending["as_of_date"] = "2026-09-30"
    if failure == "fund":
        pending["fund_code"] = "008764"
    if failure == "component":
        pending["huaan_qdii_relative_calculation"]["inputs"]["as_of_date"] = "2026-09-30"
    monkeypatch.setattr(pipeline, "_load", lambda root, ref: pending)
    monkeypatch.setattr(pipeline, "archive_json", lambda root, inputs: {"sha256": "current-corrected-inputs"})
    monkeypatch.setattr(relative, "Originals", lambda inputs, root: None)
    monkeypatch.setattr(module, "deduplicate_fx_sources", lambda source, inputs: inputs["fx_refs"])
    monkeypatch.setattr(
        module, "_shared_inputs", lambda value, root: value["huaan_qdii_relative_calculation"]["inputs"]
    )
    if failure:
        with pytest.raises(ValueError):
            pending_performance_bundle(bundle, root=tmp_path)
    else:
        result = pending_performance_bundle(bundle, root=tmp_path)
        assert result["huaan_qdii_relative_calculation"]["inputs"]["calendar_ref"] == "real-727"
        assert "019524" in result["huaan_qdii_relative_calculation"]["inputs"]["members"]
        assert result is not pending
        assert result["huaan_qdii_relative_calculation"]["input_archive"]["sha256"] == "current-corrected-inputs"


@pytest.mark.parametrize("missing", [False, True])
def test_publisher_actual_algorithm_point_cannot_be_missing(missing):
    import json

    from app.services.fund_rating_monthly_huaan import _publisher_values

    days = ["2024-03-29", "2024-04-01"]
    rows = [{"year": 2024, "month": 2, "day": 29, "value": 0}]
    if not missing:
        rows.append({"year": 2024, "month": 3, "day": 1, "value": 1})
    raw = json.dumps([{"titleText": "收益率走势", "chartsDatas": rows}]).encode()
    if missing:
        with pytest.raises(ValueError, match="REQUIRED_DATES_MISSING:2024-04-01"):
            _publisher_values(raw, days)
    else:
        assert len(_publisher_values(raw, days)) == 2


@pytest.mark.parametrize("failure", [None, "economic", "protocol"])
def test_shared_inputs_never_hide_economic_disagreement(monkeypatch, tmp_path, failure):
    from app.services import fund_rating_huaan_qdii_relative as relative
    from app.services.fund_rating_monthly_huaan import _shared_inputs

    left = {
        "members": {"019524": {}},
        "calendar_ref": "727",
        "research_on": "2026-11-17T16:00:00+08:00",
        "benchmark_protocol_ref": "a",
        "source_registry": {"x": {"sha256": "x"}},
    }
    right = {**left, "research_on": "2026-11-17T16:01:00+08:00", "benchmark_protocol_ref": "b"}
    if failure == "economic":
        right["calendar_ref"] = "730"

    def original(value, root):
        return SimpleNamespace(
            json=lambda ref: {
                "annual_cash_rate": ".004" if failure == "protocol" and ref == "b" else ".0035",
                "frozen_at": ref,
            }
        )

    monkeypatch.setattr(relative, "Originals", original)
    bundle = {
        "huaan_qdii_relative_calculation": {"inputs": left},
        "huaan_qdii_benchmark_calculation": {"inputs": right},
    }
    if failure:
        with pytest.raises(ValueError):
            _shared_inputs(bundle, root=tmp_path)
    else:
        result = _shared_inputs(bundle, root=tmp_path)
        assert result["members"] == {"019524": {}}
        assert result["research_on"] == right["research_on"]
        assert left["research_on"] == "2026-11-17T16:00:00+08:00"


def test_new_missing_publisher_point_preserves_four_components(monkeypatch, tmp_path):
    import json

    import app.services.fund_rating_monthly_huaan as module
    from app.services import fund_rating_huaan_qdii_relative as relative
    from app.services import fund_rating_pipeline_prepare as prepare

    days = ["2024-03-29", "2024-04-01"]
    inputs = {"calendar_ref": "calendar", "window_start": days[0]}
    pending = {"huaan_qdii_relative_calculation": {"inputs": inputs}}
    bundle = {
        "fund_code": "014978",
        "as_of_date": days[-1],
        "four_real_components": {"management": 55, "fee": 75, "holdings": 100, "liquidity": 75},
        "raw_native_row_count": 730,
    }
    monkeypatch.setattr(module, "pending_performance_bundle", lambda value, root: pending)
    monkeypatch.setattr(
        relative, "Originals", lambda inputs, root: SimpleNamespace(json=lambda ref: {"history": {"dates": days}})
    )
    monkeypatch.setattr(prepare, "canonical_source", lambda root, ref: ref)

    def compile_forbidden(*args, **kwargs):
        raise AssertionError("missing required point must never reach complete compiler")

    monkeypatch.setattr(module, "compile_monthly_performance", compile_forbidden)
    raw = json.dumps(
        [{"titleText": "收益率走势", "chartsDatas": [{"year": 2024, "month": 2, "day": 29, "value": 0}]}]
    ).encode()
    result = module.resume_monthly_performance(
        bundle, root=tmp_path, output=tmp_path / "attempt", fetch=lambda url, params: (raw, 200)
    )
    assert result["status"] == "MATERIALS_PENDING"
    saved = json.loads(__import__("pathlib").Path(result["candidate_path"]).read_text(encoding="utf-8"))
    assert saved == bundle
    assert "2024-04-01" in result["missing"][0]["reason"]
    assert (tmp_path / "attempt" / "publisher-refresh-receipt.json").exists()


def test_eight_components_use_new_calculations_and_real_weighted_total(monkeypatch, tmp_path):
    from decimal import Decimal

    import app.services.fund_rating_monthly_huaan as module
    from app.services import (
        fund_rating_huaan_qdii_benchmark as benchmark,
    )
    from app.services import (
        fund_rating_huaan_qdii_disclosure as disclosure,
    )
    from app.services import (
        fund_rating_huaan_qdii_relative as relative,
    )
    from app.services import (
        fund_rating_pipeline as pipeline,
    )
    from app.services import (
        fund_rating_reviewed_snapshot as reviewed,
    )

    inputs = {"fund_code": "014978"}
    bundle = {
        "fund_code": "014978",
        "as_of_date": "2026-10-30",
        "monthly_disclosure_only": True,
        "monthly_performance_followup": {"pending_performance_source_archive": {"sha256": "preserved"}},
        "huaan_qdii_disclosure_calculation": {
            "inputs": {"management_inputs": {"real": True}, "monthly_fee_inputs": {"real": True}}
        },
        "old_scalar_return": "99",
    }
    monkeypatch.setattr(module, "_shared_inputs", lambda value, root: inputs)
    monkeypatch.setattr(pipeline, "archive_json", lambda root, value: {"sha256": "new-current"})
    monkeypatch.setattr(relative, "calculate", lambda value, root: {"dimensions": {"return": "60", "stability": "70"}})
    monkeypatch.setattr(benchmark, "calculate", lambda value, root: {"dimensions": {"risk": "80", "efficiency": "90"}})
    monkeypatch.setattr(
        disclosure,
        "calculate",
        lambda value, root: {"dimensions": {"management": "55", "cost": "75", "holdings": "100", "liquidity": "75"}},
    )
    monkeypatch.setattr(disclosure, "replay_snapshot", lambda *args, **kwargs: None)
    totals = []

    def final_snapshot(candidate, root):
        document = candidate["reviewed_document"]
        assert document["dimensions"]["return"] == "60"
        dimensions = {k: {"lower": v, "upper": v, "candidate": v} for k, v in document["dimensions"].items()}
        actual = disclosure.completed_total(candidate, dimensions, document["outer_weights_percent"], root=root)
        expected = sum(
            Decimal(str(document["outer_weights_percent"][k])) * Decimal(str(v)) / 100
            for k, v in document["dimensions"].items()
        ).quantize(Decimal(".01"))
        assert Decimal(actual["score"]) == expected
        totals.append(actual["score"])
        return {"status": "SCORED", "score": actual["score"]}

    monkeypatch.setattr(reviewed, "calculate", final_snapshot)
    from app.services.fund_rating_monthly import compile_monthly_bundle

    bundle["huaan_qdii_relative_calculation"] = {"inputs": inputs}
    bundle["huaan_qdii_benchmark_calculation"] = {"inputs": inputs}
    result = compile_monthly_bundle(bundle, root=tmp_path)
    entry = result["entries"][0]
    assert entry["monthly_disclosure_only"] is False
    assert entry["huaan_qdii_disclosure_calculation"]["inputs"]["version"] == disclosure.MONTHLY_COMPLETE_VERSION
    assert totals
    assert entry["monthly_performance_followup"]["pending_performance_source_archive"] == {"sha256": "preserved"}
    assert bundle["monthly_disclosure_only"] is True


def test_storage_error_stops_resume_without_followup_writes(monkeypatch, tmp_path):
    import app.services.fund_rating_monthly_huaan as module
    from app.services import fund_materials_store as store
    from app.services import fund_rating_huaan_qdii_relative as relative

    monkeypatch.setattr(
        module,
        "pending_performance_bundle",
        lambda value, root: {
            "huaan_qdii_relative_calculation": {"inputs": {"calendar_ref": "calendar", "window_start": "2023-10-31"}}
        },
    )
    monkeypatch.setattr(
        relative,
        "Originals",
        lambda value, root: SimpleNamespace(json=lambda ref: {"history": {"dates": ["2026-10-30"]}}),
    )
    writes = []
    monkeypatch.setattr(store, "versioned_save", lambda *args, **kwargs: writes.append(args))

    def storage_failure(*args):
        raise OSError("simulated D storage write failure")

    monkeypatch.setattr(module, "_atomic_bytes", storage_failure)
    with pytest.raises(OSError, match="storage write failure"):
        module.resume_monthly_performance(
            {"fund_code": "014978"}, root=tmp_path, output=tmp_path / "attempt", fetch=lambda url, params: (b"[]", 200)
        )
    assert writes == []


@pytest.mark.parametrize("mode", ["complete", "no_next", "wrong_share", "later_month_point"])
def test_month_end_proof_only_from_actual_same_share_cross_month(monkeypatch, tmp_path, mode):
    import json

    from app.services import fund_rating_month_end as month_end
    from app.services.fund_rating_monthly_huaan import _attach_month_end_proof

    rows = [{"year": 2026, "month": 9, "day": 30, "value": 10}]
    if mode != "no_next":
        rows.append({"year": 2026, "month": 10, "day": 2, "value": 11})
    if mode == "later_month_point":
        rows.append({"year": 2026, "month": 9, "day": 31, "value": 10.5})
    value = [{"titleText": "收益率走势", "chartsDatas": rows}]
    raw = json.dumps(value).encode()
    ref = {
        "sha256": "actual-response-hash",
        "url": "https://www.huaan.com.cn/fund/FundPerformance/fundPerformanceChart.do",
        "status_code": 200,
        "params": {
            "fundcodes": "040046" if mode == "wrong_share" else "014978",
            "selecttype": "ZS",
            "beginDate": "2023-10-31",
            "endDate": "2026/11/17",
        },
    }
    monkeypatch.setattr(month_end, "Originals", lambda *args, **kwargs: SimpleNamespace(json=lambda digest: value))
    bundle = {"fund_code": "014978", "share_class": "C", "as_of_date": "2026-10-30"}
    _attach_month_end_proof(bundle, raw, ref, root=tmp_path)
    if mode == "complete":
        assert bundle["monthly_valuation_proof"]["next_valuation_date"] == "2026-11-02"
        assert bundle["monthly_valuation_proof"]["issuer_nav_source_ref"] == ref
        assert bundle["monthly_valuation_proof"]["month_end"] == "2026-10-31"
    else:
        assert "monthly_valuation_proof" not in bundle
        assert bundle["monthly_valuation_proof_pending"]
