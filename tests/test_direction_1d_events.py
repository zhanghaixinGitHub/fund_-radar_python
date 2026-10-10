"""事件版的方向边界：不让公告条数、重复内容、缺失信息或净值连跌冒充利好。"""

import copy

import pytest
from app.services import direction_1d_events as event
from app.services import direction_1d_information as info

SESSIONS = [f"2026-09-{n:02}" for n in range(1, 31)] + ["2026-10-08", "2026-10-09"]
CUTOFF = "2026-10-09T10:00:00+08:00"


def document(**changes):
    return {
        "id": "one",
        "kind": "ANNOUNCEMENT",
        "title": "2026年第三季度业绩预告",
        "facts": ["预计归属于母公司股东的净利润同比增长50%。"],
        "direction": "BENEFIT",
        "stage": "FORECAST",
        "published_date": "2026-10-08",
        "available_at": "2026-10-08T22:00:00+08:00",
        "source_hash": "a" * 64,
        "source_url": "https://example.org/one.pdf",
        "links": [
            {
                "basis": "ISSUER",
                "code": "600001.SH",
                "weight": 0.09,
                "report_hash": "b" * 64,
                "report_available_at": "2026-08-31T00:00:00+08:00",
            }
        ],
        **changes,
    }


def fixture(documents=None):
    evidence = event.build_evidence(
        {"events": documents if documents is not None else [document()]}, "2026-10-08", CUTOFF, SESSIONS
    )
    full = {
        "numeric": [0.0] * 87,
        "event_evidence": evidence,
        "holding_coverage": 0.8,
        "counts": {"ANNOUNCEMENT": 1, "NEWS": 0, "POLICY": 0},
        "text": "公告",
    }
    model = {
        "feature_version": event.VERSION,
        "features": event.FEATURES,
        "fund_code": "002112",
        "protocol": "DIRECTION_1D_V2",
        "group_id": event.GROUPS["MORNING_0830"],
        "policy": event.POLICY,
        "classes": ["DOWN", "UP"],
        "coef": [0.1] * 8,
        "intercept": 0,
    }
    return model, full


def test_counts_missing_flags_nav_falls_and_text_do_not_drive_direction():
    model, full = fixture()
    original = event.predict(model, full)
    changed = copy.deepcopy(full)
    changed["numeric"][:7] = [-100] * 7
    changed["numeric"][29:] = [9999] * 58
    changed["counts"]["ANNOUNCEMENT"] = 10000
    changed["text"] = "利好" * 1000
    assert event.predict(model, changed) == original
    assert original["direction"] == "UP"


def test_repeated_files_and_changed_ids_never_accumulate():
    _, one = fixture()
    _, many = fixture([document(), document(id="copy"), document(id="third")])
    assert one["event_evidence"]["values"] == many["event_evidence"]["values"]
    assert len(many["event_evidence"]["events"]) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"published_date": "2026-09-01", "available_at": CUTOFF},
        {"available_at": "2026-10-09T11:00:00+08:00"},
        {"published_date": "2026-10-10"},
        {"direction": "UNKNOWN"},
        {"stage": "PROPOSAL"},
        {"facts": ["请投资者注意风险。"]},
        {"title": "董事会会议决议"},
        {"links": []},
        {"source_hash": "wrong"},
    ],
)
def test_unverifiable_event_cannot_create_direction(changes):
    model, full = fixture([document(**changes)])
    result = event.predict(model, full)
    assert result["status"] == "ABSTAINED"
    assert result["direction"] is result["score"] is result["class_scores"] is None
    assert "NO_DIRECTIONAL_EVENT" in result["decision"]["reason_codes"]


def test_age_and_disclosed_weight_affect_same_concrete_event():
    _, fresh = fixture()
    older = document(published_date="2026-09-29")
    _, aged = fixture([older])
    assert 0 < aged["event_evidence"]["values"][1] < fresh["event_evidence"]["values"][1]
    weak = document()
    weak["links"][0]["weight"] = 0.01
    _, low = fixture([weak])
    assert low["event_evidence"]["values"][1] < fresh["event_evidence"]["values"][1]


def test_negative_content_cannot_become_positive_and_conflicts_abstain():
    negative = document(direction="PRESSURE", facts=["预计净利润同比下降50%。"])
    model, full = fixture([negative])
    assert event.predict(model, full)["direction"] == "DOWN"
    _, mixed = fixture([negative, document(id="positive")])
    assert "EVENT_CONFLICT" in event.predict(model, mixed)["decision"]["reason_codes"]
    model, full = fixture()
    full["numeric"][7] = -0.02
    assert "MARKET_EVENT_CONFLICT" in event.predict(model, full)["decision"]["reason_codes"]


@pytest.mark.parametrize("kind", ["POLICY", "NEWS"])
def test_policy_and_news_require_content_time_and_real_holding_link(kind):
    model, full = fixture(
        [
            document(
                kind=kind,
                title="产品出口政策实施",
                stage="IMPLEMENTATION",
                facts=["出口关税已下调，相关公司产品出口成本下降。"],
            )
        ]
    )
    assert event.predict(model, full)["direction"] == "UP"
    no_link = document(kind=kind)
    no_link["links"][0]["basis"] = "INDUSTRY_CONTEXT"
    _, full = fixture([no_link])
    assert event.predict(model, full)["status"] == "ABSTAINED"


def test_old_model_dispatch_stays_separate():
    model, full = fixture()
    model["validation_status"] = "INDEPENDENT_VALIDATION_PASSED"
    info.validate_model(model)
    assert info.predict(model, full) == event.predict(model, full)
    model["coef"][0] = -0.1
    with pytest.raises(ValueError, match="EVENT_MODEL_INVALID"):
        info.validate_model(model)


def test_unvalidated_research_model_abstains_even_with_bullish_event():
    model, full = fixture()
    assert event.predict(model, full)["direction"] == "UP"
    result = info.predict(model, full)
    assert result["status"] == "ABSTAINED"
    assert "VALIDATION_INSUFFICIENT" in result["decision"]["reason_codes"]
    assert info.explain(model, full)["direction"] is None
