"""自动评价的收益方向、类型、缺项与批次边界，不依赖真实数据库或外网。"""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_narrative import PROFILES, REVISION, render
from app.services.fund_rating_narrative_facts import _summary_source
from app.services.fund_rating_narrative_store import enrich, load, prepare

KEYS = ("return", "risk", "efficiency", "stability", "management", "holdings", "liquidity", "cost")


def item():
    return {
        "status": "SCORED",
        "score": "75.65",
        "grade": None,
        "ratingRef": "a" * 64,
        "summary": "旧摘要",
        "dataDates": {"holdings": "2026-06-30"},
        "dimensions": [
            {"key": k, "score": "75.00", "explanation": "既有适用依据", "metrics": [], "sources": []} for k in KEYS
        ],
    }


def facts():
    return {
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "share_class": "C",
        "fee_date": "2026-09-03",
        "index_name": "农牧渔",
        "performance": {
            "period": "2025-09-30 至 2026-09-30",
            "fund_return": "-.1282",
            "reference_return": "-.1452",
            "drawdown": ".2887",
            "reference_drawdown": ".2962",
            "tracking_error": ".0083",
            "months_at_or_above": 7,
            "month_count": 12,
        },
        "execution": {
            "tracking_error_limit": ".04",
            "contract_date": "2026-06-30",
            "transition_date": "2025-10-10",
            "post_transition_tracking_error": ".00828",
        },
        "management": {"paragraphs": ["陈建华自2021-12-06管理本产品，转型后接续。"]},
        "holdings": {
            "target_etf_nav_fraction": ".9467",
            "major_holdings_count": 10,
            "matched_major_holdings_count": 10,
        },
        "liquidity": {
            "review_complete": True,
            "reported_redemption_alert": False,
            "reported_redemption_shares": "234994584.68",
            "deferred_redemption": True,
        },
        "cost": {"annual_cost_per_10000": "101", "annual_total_rate": ".0101"},
    }


def dimension(output, key):
    return next(d for d in output["dimensions"] if d["key"] == key)


@pytest.mark.parametrize(
    ("fund", "reference", "expected"),
    [
        ("-.12", "-.14", "比基准少跌"),
        ("-.14", "-.12", "跌幅大于基准"),
        (".12", ".14", "取得正收益，但仍落后"),
        (".14", ".12", "取得正收益，同时领先"),
        ("0", "-.1", "期间收益持平"),
        (".1", ".1", "期间回报持平"),
    ],
)
def test_return_conclusion_uses_actual_signs_not_score(fund, reference, expected):
    evidence = facts()
    evidence["performance"].update(fund_return=fund, reference_return=reference)
    actual = render(item(), evidence, "INDEX")
    assert expected in dimension(actual, "return")["assessment"]["conclusion"]
    assert actual["score"] == "75.65" and actual["grade"] is None


def test_full_eight_dimension_content_preserves_dates_scores_and_inputs():
    original, evidence = item(), facts()
    before = deepcopy((original, evidence))
    actual = render(original, evidence, "INDEX")
    assert all(d["assessment"]["conclusion"] and d["assessment"]["paragraphs"] for d in actual["dimensions"])
    assert "0.75个百分点" in dimension(actual, "risk")["explanation"]
    assert dimension(actual, "stability")["metrics"][0]["value"] == "7/12"
    assert "2025-10-10" in dimension(actual, "efficiency")["explanation"]
    assert "不是分散程度" in dimension(actual, "holdings")["explanation"]
    assert dimension(actual, "cost")["metrics"][0]["value"] == "101.00"
    assert dimension(actual, "holdings")["metrics"][0]["period"] == "2026-06-30"
    assert actual["dataDates"]["nav"] == "2026-09-30"
    assert [d["score"] for d in actual["dimensions"]] == [d["score"] for d in original["dimensions"]]
    assert (original, evidence) == before


def test_active_fund_does_not_receive_passive_tracking_or_matching_conclusions():
    evidence = facts()
    evidence["execution"]["description"] = "已核股票仓位与现金要求。"
    evidence["execution"].update(
        stock_total_ratio=".89",
        stock_nav_ratio=".90",
        cash_nav_ratio=".06",
        stock_min=".80",
        stock_max=".95",
        cash_min=".05",
    )
    evidence["holdings"]["major_holdings_nav_fraction"] = ".52"
    actual = render(item(), evidence, "ACTIVE_EQUITY")
    assert "合同仓位与策略执行" in dimension(actual, "efficiency")["explanation"]
    assert "跟踪误差" not in dimension(actual, "efficiency")["explanation"]
    execution = dimension(actual, "efficiency")
    assert [(m["name"], m["value"], m["unit"], m["period"]) for m in execution["metrics"]] == [
        ("股票占总资产", "89.00", "%", "2026-06-30"),
        ("股票占净资产", "90.00", "%", "2026-06-30"),
        ("合同认可现金占净资产", "6.00", "%", "2026-06-30"),
    ]
    assert not any(p.startswith("截至") for p in execution["assessment"]["paragraphs"])
    assert "80.00%至95.00%" in dimension(actual, "efficiency")["explanation"]
    assert "现金下限为净资产的5.00%" in execution["explanation"]
    assert "合同配置执行不等于收益保证，也不直接证明选股或择时能力。" in execution["assessment"]["limitations"]
    assert dimension(actual, "holdings")["metrics"][0]["value"] == "52.00"
    assert "成分匹配" not in dimension(actual, "holdings")["explanation"]


@pytest.mark.parametrize("family", ["BOND", "SHORT_BOND", "MONEY", "QDII_BOND", "QDII_COMMODITY", "FOF"])
def test_specialized_types_keep_their_own_basis_and_no_equity_replication(family):
    actual = render(item(), {"as_of_date": "2026-09-30"}, family)
    text = dimension(actual, "holdings")["explanation"]
    assert PROFILES[family][0] in text and "既有适用依据" in text
    assert "10只" not in text and "跟踪误差" not in text


@pytest.mark.parametrize("bad", [None, True, "NaN", "Infinity", "invalid", "-2"])
def test_invalid_return_never_becomes_zero_or_an_invented_conclusion(bad):
    evidence = facts()
    evidence["performance"]["fund_return"] = bad
    actual = render(item(), evidence, "INDEX")
    assert "期末金额" not in dimension(actual, "return")["explanation"]
    assert actual["summary"] == "旧摘要"


def test_missing_score_or_month_evidence_cannot_become_an_affirmative_assessment():
    original = item()
    original["dimensions"][5]["score"] = None
    evidence = facts()
    evidence["performance"].pop("month_count")
    actual = render(original, evidence, "INDEX")
    assert "尚无确定分" in dimension(actual, "holdings")["assessment"]["conclusion"]
    assert not dimension(actual, "holdings")["metrics"]
    assert "7个月" not in dimension(actual, "stability")["explanation"]


def stored_fixture():
    bundle = {"fund_code": "000009", "family": "INDEX", "as_of_date": "2026-09-30"}
    snapshot = {"members": [{"fund_code": "000009", "bundle": bundle, "evidence_hash": digest(bundle)}]}
    result = SimpleNamespace(
        fund_code="000009",
        batch_id="a" * 64,
        status="SCORED",
        score="75.65",
        grade=None,
        dimension_values={},
        public_evidence={},
    )
    return result, snapshot


def test_store_is_bound_to_formal_snapshot_and_get_never_compiles(tmp_path, monkeypatch):
    from app.services import fund_rating_narrative_facts as compiler

    result, snapshot = stored_fixture()
    monkeypatch.setattr(compiler, "compile_facts", lambda *a, **kw: facts())
    before = deepcopy(snapshot)
    assert prepare(result, snapshot, root=tmp_path)["state"] == "GENERATED"
    assert prepare(result, snapshot, root=tmp_path)["state"] == "REUSED"
    monkeypatch.setattr(compiler, "compile_facts", lambda *a, **kw: pytest.fail("GET compiled originals"))
    assert dimension(enrich(item(), result, snapshot, root=tmp_path), "return")["assessment"]
    snapshot["members"][0]["bundle"]["as_of_date"] = "2026-10-31"
    with pytest.raises(ValueError, match="BINDING"):
        load(result, snapshot, root=tmp_path)
    assert before["members"][0]["bundle"]["as_of_date"] == "2026-09-30"


@pytest.mark.parametrize("status", ["STALE", "WITHDRAWN", "UNSUPPORTED", "NOT_RATED"])
def test_withheld_rating_does_not_even_read_old_fact_file(tmp_path, monkeypatch, status):
    from app.services import fund_rating_narrative_store as store

    original = item()
    original["status"] = status
    monkeypatch.setattr(store, "load", lambda *a, **kw: pytest.fail("read old narrative"))
    result, snapshot = stored_fixture()
    assert all(d["assessment"] is None for d in enrich(original, result, snapshot, root=tmp_path)["dimensions"])


def test_failed_generation_preserves_old_artifact_and_all_scores(tmp_path, monkeypatch):
    from app.services import fund_rating_narrative_facts as compiler

    result, snapshot = stored_fixture()
    monkeypatch.setattr(compiler, "compile_facts", lambda *a, **kw: facts())
    prepare(result, snapshot, root=tmp_path)
    path = next((tmp_path / "narratives").glob("*.json"))
    before = path.read_bytes()

    def fail(*a, **kw):
        raise ValueError("missing source")

    monkeypatch.setattr(compiler, "compile_facts", fail)
    with pytest.raises(ValueError, match="missing source"):
        prepare(result, snapshot, root=tmp_path, force=True)
    assert path.read_bytes() == before and result.score == "75.65" and result.grade is None


def test_foreign_share_and_tampered_facts_cannot_reuse_cache(tmp_path, monkeypatch):
    from app.services import fund_rating_narrative_facts as compiler

    result, snapshot = stored_fixture()
    monkeypatch.setattr(compiler, "compile_facts", lambda *a, **kw: facts())
    prepare(result, snapshot, root=tmp_path)
    result.fund_code = "000010"
    assert load(result, snapshot, root=tmp_path) is None
    result.fund_code = "000009"
    path = next((tmp_path / "narratives").glob("*.json"))
    packet = json.loads(path.read_text(encoding="utf-8"))
    packet["facts"]["performance"]["fund_return"] = "100"
    path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(ValueError, match="BINDING_OR_CONTENT"):
        load(result, snapshot, root=tmp_path)
    assert packet["revision"] == REVISION


def test_supplemental_summary_rejects_future_publication():
    source = SimpleNamespace(
        sources={"h": {"kind": "json"}},
        pages=lambda h: ["000009 C类资料概要"],
        json=lambda h: {"data": {"notice_date": "2026-10-01"}},
    )
    inputs = {"fund_code": "000009", "share_class": "C", "as_of_date": "2026-09-30", "target_refs": {"summary": "h"}}
    with pytest.raises(ValueError, match="PUBLICATION_DATE"):
        _summary_source(inputs, source)
