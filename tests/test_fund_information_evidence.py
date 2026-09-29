"""事实层防止时间穿越、缺口填零、虚构金额和一事多文误加总。"""

from datetime import datetime

import pytest
from app.services.fund_002112_zero_fit_review import digest, file_hash, read_json
from app.services.fund_information_evidence import (
    available_at,
    classification,
    information_features,
    make_fact,
    store_bundle,
    validate,
    window_covered,
)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("关于回购公司股份方案的公告", ("BUYBACK", "PLAN")),
        ("关于回购股份进展的公告", ("BUYBACK", "PROGRESS")),
        ("关于回购股份完成的公告", ("BUYBACK", "COMPLETED")),
        ("关于回购注销部分限制性股票的公告", None),
        ("关于签订重大合同的公告", ("MAJOR_CONTRACT", "SIGNED")),
        ("2023年度业绩预告修正公告", ("PERFORMANCE", "CORRECTION")),
        ("2023年度业绩预告", ("PERFORMANCE", "FORECAST")),
        ("监事会关于年度报告的审核意见", None),
    ],
)
def test_type_and_stage_are_facts_not_sentiment(title, expected):
    assert classification(title) == expected


@pytest.fixture
def fact(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text("公告：回购计划尚未实施", encoding="utf-8")
    return make_fact(
        category="BUYBACK",
        stage="PLAN",
        entity_type="COMPANY",
        entity_id="000001",
        document_id="CNINFO:123",
        title="回购计划",
        published="2023-03-31",
        source={
            "url": "https://example.com/123",
            "path": str(raw),
            "sha256": file_hash(raw),
            "use_basis": "test",
            "retention_basis": "test",
        },
        anchors=[{"text": "回购计划尚未实施"}],
    )


def test_day_only_is_next_morning_not_midnight(fact):
    assert available_at(fact).isoformat() == "2023-04-01T08:00:00+08:00"
    assert available_at(fact) > datetime.fromisoformat("2023-04-01T00:00:00+08:00")


def test_unknown_and_revised_publication(fact):
    assert available_at({**fact, "revision_unresolved": True}) is None
    assert available_at({**fact, "published_date": None}) is None
    assert available_at({**fact, "version_publication_date": "2023-04-15"}).isoformat().startswith("2023-04-16")


def test_coverage_hole_cannot_be_filled():
    assert not window_covered([["2023-03-01", "2023-03-10"], ["2023-03-12", "2023-03-31"]], "2023-03-01", "2023-03-31")
    assert window_covered([["2023-03-01", "2023-03-10"], ["2023-03-11", "2023-03-31"]], "2023-03-01", "2023-03-31")


def test_missing_news_does_not_mean_zero_event(fact):
    with pytest.raises(ValueError, match="COVERAGE_INCOMPLETE"):
        information_features(
            [fact], cutoff="2023-04-03T08:00:00+08:00", companies=["000001"], report_sha256="a" * 64, coverage={}
        )


def test_fact_has_no_calibrated_score(fact):
    with pytest.raises(ValueError, match="INVENT_PREDICTION"):
        validate({**fact, "prediction_score": 0.8})
    assert fact["quantities"] == []


def test_amount_without_units_rejected(fact):
    with pytest.raises(ValueError, match="QUANTITY"):
        validate({**fact, "quantities": [{"value": "1", "unit": "UNKNOWN"}]})


def test_immutable_bundle_dedup_and_readback(tmp_path, fact):
    manifest = store_bundle(tmp_path, [fact, fact], {}, {})
    assert manifest["facts"] == 1 and manifest["readback_passed"]
    assert read_json(manifest["path"])["facts"][0]["stage"] == "PLAN"
    assert store_bundle(tmp_path, [fact], {}, {}) == manifest
    other = {**fact, "stage": "COMPLETED"}
    other["revision"] = digest({k: v for k, v in other.items() if k != "revision"})
    with pytest.raises(ValueError, match="CONFLICTING"):
        store_bundle(tmp_path, [fact, other], {}, {})


def test_revision_change_requires_new_digest(fact):
    with pytest.raises(ValueError, match="REVISION_CHANGED"):
        validate({**fact, "stage": "COMPLETED"})


def test_source_tampering_detected(fact):
    from pathlib import Path

    Path(fact["source"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        validate(fact)
