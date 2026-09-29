"""新授权运行器的预算和防误用检查；测试用替身，不消耗真实基金拟合额度。"""

import hashlib

import pytest
from app.services import fund_002112_information_experiment as ex
from app.services.fund_002112_information_admission import label, publication


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RUN", tmp_path)
    ex.write(tmp_path / "protocol.json", {"maximum_real_fits": 24})
    for h in ex.HYPOTHESES:
        ex.write(tmp_path / h / "freeze.json", {"hypothesis": h})
    return tmp_path


def nav(day, value, ann=None, **extra):
    return {"date": day, "nav": value, "ann_date": ann or day, "source_hash": "a" * 64, **extra}


@pytest.mark.parametrize(
    ("a", "b", "direction"), [("1", "1.00000001", "UP"), ("1.00000001", "1", "DOWN"), ("1.0", "1.00", "FLAT")]
)
def test_exact_decimal_labels(a, b, direction):
    r = label(nav("2020-01-02", a, "2020-01-07"), nav("2020-01-03", b, "2020-01-04"), ["2020-01-02", "2020-01-03"])
    assert r["actual_direction"] == direction
    assert r["mature_at"] == "2020-01-08T08:00:00+08:00"


def test_revision_never_advances_publication():
    assert publication(nav("2020-01-02", "1", "2020-01-03", revised_at="2020-02-01")) == "2020-02-01"
    with pytest.raises(ValueError, match="UNKNOWN"):
        publication(nav("2020-01-02", "1", revision_known=True))


@pytest.mark.parametrize("day", ["2025-01-02", "2026-01-02", "2023-01-02"])
def test_new_label_scope(day):
    with pytest.raises(ValueError, match="SCOPE"):
        label(nav("2020-01-01", "1"), nav(day, "1"), ["2020-01-01", day])


def test_reserved_budget_is_not_refunded(run):
    ex.reserve(ex.HYPOTHESES[0], ex.SLOTS[0], "token")
    assert ex.status()["consumed"] == 1
    assert ex.status()["actual_fits"] == 0
    with pytest.raises(ValueError, match="CONSUMED"):
        ex.reserve(ex.HYPOTHESES[0], ex.SLOTS[0], "token2")
    assert ex.ledger()[0]["permit_sha256"] == hashlib.sha256(b"token").hexdigest()


def test_global_budget_cannot_be_borrowed(run, monkeypatch):
    monkeypatch.setattr(ex, "completed", lambda *a: {})
    for h in ex.HYPOTHESES:
        ex.write(run / h / "historical-decision.json", {"passed": True})
        for s in ex.SLOTS:
            ex.reserve(h, s, "token")
    assert ex.status()["consumed"] == 24
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        ex.reserve(ex.HYPOTHESES[1], ex.SLOTS[0], "token")


def test_full_requires_historical_gate(run, monkeypatch):
    monkeypatch.setattr(ex, "completed", lambda *a: {})
    for s in ex.SLOTS[:6]:
        ex.reserve(ex.HYPOTHESES[0], s, "token")
    ex.write(run / ex.HYPOTHESES[0] / "historical-decision.json", {"passed": False})
    with pytest.raises(ValueError, match="GATE_REQUIRED"):
        ex.reserve(ex.HYPOTHESES[0], ex.SLOTS[6], "token")
    assert len(ex.ledger()) == 6


def test_orphan_trace_blocks_deleted_last_budget(run):
    ex.write(run / ex.HYPOTHESES[0] / "classifier-started" / (ex.SLOTS[0] + ".json"), {})
    with pytest.raises(ValueError, match="ORPHAN"):
        ex.ledger()


def test_input_missing_is_not_zero_filled():
    with pytest.raises(ValueError, match="INPUT"):
        ex.matrix([{"x": [1]}], ["a", "b"])


def test_immutable_result_cannot_be_overwritten(tmp_path):
    ex.once(tmp_path / "data.json", {"v": 1})
    ex.once(tmp_path / "data.json", {"v": 1})
    with pytest.raises(ValueError, match="IMMUTABLE"):
        ex.once(tmp_path / "data.json", {"v": 2})


def test_resume_uses_checkpoint_without_second_fit(run, monkeypatch):
    h, slot = ex.HYPOTHESES[0], ex.SLOTS[0]
    ex.reserve(h, slot, "token")
    ex.write(run / h / "completed" / (slot + ".json"), {})
    monkeypatch.setattr(ex, "load", lambda h: ({}, {}))
    calls = []
    monkeypatch.setattr(ex, "child", lambda h, s, **kw: calls.append(kw))
    result = ex.run(h, interrupt_after=1)
    assert result["status"] == "VALID_CHECKPOINT_SAVED"
    assert calls == [{"restore": True}]
    assert len(ex.ledger()) == 1


def test_consumed_partial_fit_stops_without_retry(run, monkeypatch):
    h = ex.HYPOTHESES[0]
    ex.reserve(h, ex.SLOTS[0], "token")
    monkeypatch.setattr(ex, "load", lambda h: ({}, {}))
    monkeypatch.setattr(ex, "child", lambda *a, **kw: pytest.fail("Must not fit again"))
    assert ex.run(h)["status"] == "STOP_ENGINEERING_FAILURE"
    assert len(ex.ledger()) == 1
