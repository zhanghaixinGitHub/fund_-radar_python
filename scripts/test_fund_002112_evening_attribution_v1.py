"""四组拆分、证据权重与时间边界的行为验证。"""

from copy import deepcopy

import numpy as np
import pytest

from scripts import fund_002112_evening_attribution_v1 as m


def rows():
    base = {
        "target": "2025-01-06",
        "as_of": "2025-01-05T23:00:00+08:00",
        "base": "2025-01-03",
        "label": "UP",
        "groups": {"market": [1, None], "history": [3], "information": list(range(188))},
    }
    both = deepcopy(base)
    both["groups"]["information"] = [1000 + i for i in range(211)]
    return base, both


def evidence():
    row, _ = rows()
    sessions = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    company = {
        "id": "event",
        "code": "A",
        "version_available_at": "2025-01-04T08:00:00+08:00",
        "status": "QUALIFIED",
        "numeric": {"profit_yoy": {"value": 0, "quote": "同比持平"}},
    }
    proof = {
        "target": row["target"],
        "as_of": row["as_of"],
        "public_events": [],
        "company_events": [{"id": "event", "available_at": company["version_available_at"], "weight": 0.04}],
    }
    return row, proof, {"event": company}, sessions


def test_factorial_projection_separates_both_treatments():
    base, both = rows()
    before = deepcopy((base, both))
    names = m.prior.feature_names("C_EVENT")
    for group in m.GROUPS:
        result = m.project_row(base, both, group)
        assert len(result["groups"]["information"]) == len(m.feature_names(group))
        assert result["label"] == base["label"] and result["as_of"] == base["as_of"]
        assert result["groups"]["market"] == base["groups"]["market"]
    timing = m.project_row(base, both, "TIMING")["groups"]["information"]
    assert not set(m.REACTION_FIELDS) & set(m.feature_names("TIMING"))
    assert timing == [
        v for name, v in zip(names, both["groups"]["information"], strict=True) if name not in m.REACTION_FIELDS
    ]
    reaction = m.project_row(base, both, "REACTION")["groups"]["information"]
    assert reaction[:188] == base["groups"]["information"]
    assert reaction[188:] == [both["groups"]["information"][names.index(n)] for n in m.REACTION_FIELDS]
    assert (base, both) == before


def test_projection_rejects_misaligned_labels_or_prices():
    base, both = rows()
    both["label"] = "DOWN"
    with pytest.raises(AssertionError):
        m.project_row(base, both, "REACTION")
    both["label"] = "UP"
    both["groups"]["market"] = [0]
    with pytest.raises(AssertionError):
        m.project_row(base, both, "TIMING")


def test_quality_does_not_read_answer_or_price_and_accepts_true_zero():
    row, proof, company, sessions = evidence()
    first = m.quality(row, proof, company, sessions)
    assert first["state"] == "STRONG" and first["information_weight"] == 0.55
    row["label"] = "DOWN"
    row["return"] = -9.9
    row["groups"]["market"] = [999]
    assert m.quality(row, proof, company, sessions) == first


@pytest.mark.parametrize("change", ["missing_number", "missing_quote", "small_holding", "old", "unqualified"])
def test_weak_evidence_cannot_be_strong(change):
    row, proof, companies, sessions = evidence()
    event = companies["event"]
    if change == "missing_number":
        event["numeric"]["profit_yoy"]["value"] = None
    elif change == "missing_quote":
        event["numeric"]["profit_yoy"]["quote"] = ""
    elif change == "small_holding":
        proof["company_events"][0]["weight"] = 0.02
    elif change == "old":
        sessions[:0] = ["2024-12-30", "2024-12-31"]
        event["version_available_at"] = "2024-12-30T08:00:00+08:00"
        proof["company_events"][0]["available_at"] = event["version_available_at"]
    else:
        event["status"] = "DUPLICATE_EVENT"
    assert m.quality(row, proof, companies, sessions)["state"] == "CONTEXT"


def test_repeated_same_company_not_counted_twice_for_coverage():
    row, proof, companies, sessions = evidence()
    proof["company_events"][0]["weight"] = 0.02
    proof["company_events"].append(deepcopy(proof["company_events"][0]))
    result = m.quality(row, proof, companies, sessions)
    assert result["strong_coverage"] == 0.02 and result["state"] == "CONTEXT"


def test_future_evidence_is_rejected():
    row, proof, companies, sessions = evidence()
    companies["event"]["version_available_at"] = "2025-01-06T08:00:00+08:00"
    proof["company_events"][0]["available_at"] = companies["event"]["version_available_at"]
    with pytest.raises(AssertionError):
        m.quality(row, proof, companies, sessions)


def test_context_and_no_evidence_fallback_keep_missing_values():
    row, proof, companies, sessions = evidence()
    proof["company_events"] = []
    proof["public_events"] = [{"available_at": "2025-01-04T00:00:00+08:00"}]
    assert m.quality(row, proof, companies, sessions)["state"] == "CONTEXT"
    proof["public_events"] = []
    state = m.quality(row, proof, companies, sessions)
    assert state["state"] == "NONE" and state["information_weight"] == 0
    assert row["groups"]["market"][1] is None
    p = m.blend([1, 0, 0], [0.3, 0, 0.7], [0.6, 0, 0.4], "QUALITY_GATE", state)
    assert np.allclose(p, [0.4, 0, 0.6])


@pytest.mark.parametrize("policy", m.POLICIES)
def test_fusion_conserves_market_history_ratio_and_probability(policy):
    w = m.weights_for(policy, {"state": "CONTEXT"})
    assert sum(w.values()) == pytest.approx(1) and w["market"] == pytest.approx(2 * w["history"])
    p = m.blend([0.2, 0.1, 0.7], [0.6, 0.1, 0.3], [0.1, 0.1, 0.8], policy, {"state": "CONTEXT"})
    assert sum(p) == pytest.approx(1)


def test_selection_ties_are_fixed_and_do_not_use_2026():
    score = {
        "A": {"correct": 100, "brier": 0.50, "2026_correct": 0},
        "B": {"correct": 99, "brier": 0.40, "2026_correct": 999},
    }
    assert m.choose(score, {"A": 5, "B": 1}) == "A"
    score["B"]["correct"] = 100
    assert m.choose(score, {"A": 5, "B": 1}) == "B"
    score["A"]["brier"] = 0.40
    assert m.choose(score, {"A": 5, "B": 1}) == "B"
