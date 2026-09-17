"""成交额训练的一次性衔接边界：父训练必须完整成功，输入绑定只允许替换预先声明的哈希。"""

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest


@pytest.fixture
def worker():
    path = (
        Path(__file__).resolve().parents[1]
        / ".local-runs/direction-1d-sprint-20260914/round-110/continue_after_parent.py"
    )
    spec = importlib.util.spec_from_file_location("turnover_continuation_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binding_changes_only_declared_hash_and_keeps_training_logic(worker):
    template = 'PROPOSAL_HASH = "' + worker.MARKER + '"\nC=0.1\nthreshold=0.5\n'
    actual = worker.rendered_model(template, "a" * 64)
    assert actual == 'PROPOSAL_HASH = "' + "a" * 64 + '"\nC=0.1\nthreshold=0.5\n'


@pytest.mark.parametrize("value", ["a" * 63, "G" * 64, '";change_recipe();#'])
def test_invalid_binding_hash_cannot_be_written_as_code(worker, value):
    with pytest.raises(AssertionError, match="PROPOSAL_HASH_INVALID"):
        worker.rendered_model(worker.MARKER, value)


@pytest.mark.parametrize("phase", ["FAILED_NO_RETRY", "SOURCE_FAILED_NO_TRAINING", "STOPPED_AT_DEADLINE"])
def test_parent_failure_is_terminal_for_child_without_retraining(worker, monkeypatch, phase):
    monkeypatch.setattr(worker.b, "read", lambda _: {"phase": phase})
    monkeypatch.setattr(worker.s.previous, "models", lambda: pytest.fail("failed parent cannot load training models"))
    assert worker.parent_state()[0] == "FAILED"


def test_parent_completion_must_match_real_saved_model(worker, monkeypatch):
    phase = worker.s.previous.root() / "execution-status.json"
    result = {"actual": "saved model result"}
    values = {
        phase: {"phase": "TRAINED_ANALYZED_PENDING_WINDOWS_CUTOVER"},
        worker.s.previous.root() / "training-completion.json": {"result_hash": "wrong", "new_fits": 27},
    }
    monkeypatch.setattr(worker.b, "read", lambda path: values[path])
    monkeypatch.setattr(worker.s.previous, "models", lambda: (result, {}))
    with pytest.raises(AssertionError):
        worker.parent_state()
    values[worker.s.previous.root() / "training-completion.json"]["result_hash"] = worker.b.digest(result)
    assert worker.parent_state()[0] == "READY"


def test_deadline_stops_before_any_parent_check_or_fit(worker, monkeypatch):
    observed = []
    monkeypatch.setattr(worker, "check", lambda: {})
    monkeypatch.setattr(worker.b, "now", lambda: worker.END + timedelta(seconds=1))
    monkeypatch.setattr(worker.b, "save", lambda path, value, **kw: observed.append(value))
    monkeypatch.setattr(worker, "parent_state", lambda: pytest.fail("deadline must stop first"))
    monkeypatch.setattr(worker, "execute", lambda _: pytest.fail("no fits after deadline"))
    worker.main()
    assert observed[-1]["phase"] == "STOPPED_AT_DEADLINE"
    assert observed[-1]["training_started"] is False
