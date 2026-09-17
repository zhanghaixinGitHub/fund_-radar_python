"""验收接续进程的停止边界：数据缺失、依赖变化或截止不能启动训练。"""

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / ".local-runs/direction-1d-sprint-20260914/round-121/after_source.py"
    spec = importlib.util.spec_from_file_location("r117_pipeline_guard_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = tmp_path / "round-121"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(mod, "SOURCE", source)
    monkeypatch.setattr(mod, "PROJECT", tmp_path)
    monkeypatch.setattr(mod.b, "ROOT", tmp_path)
    now = [datetime.fromisoformat("2026-09-16T20:00:00+08:00")]
    monkeypatch.setattr(mod.b, "now", lambda: now[0])
    plan = {
        "script_sha256": mod.core.sha(script),
        "code_hashes": {},
        "artifact_hashes": {},
        "deadline": "2026-09-17T12:11:38+08:00",
        "original_rows_hash": "fixed-rows",
    }
    mod.b.save(root / "pipeline-plan.json", plan)
    executed = []
    monkeypatch.setattr(mod, "execute", lambda *args: executed.append(str(args[0])))
    return mod, plan, now, executed


def test_missing_source_waits_without_executing_qualification_or_fit(pipeline, monkeypatch):
    mod, _, _, executed = pipeline

    def observe_wait(seconds):
        assert seconds == 30 and executed == []
        assert mod.b.read(mod.ROOT / "pipeline-progress.json")["stage"] == "WAITING_FOR_EXISTING_BOUNDED_ACQUISITION"
        raise RuntimeError("observed wait only")

    monkeypatch.setattr(mod.time, "sleep", observe_wait)
    with pytest.raises(RuntimeError, match="observed wait only"):
        mod.main()


def test_acquisition_failure_never_retries_or_starts_training(pipeline):
    mod, _, _, executed = pipeline
    mod.b.save(mod.SOURCE / "last-error.json", {"synthetic": "failed"})
    with pytest.raises(ValueError, match="SOURCE_ACQUISITION_FAILED_NO_RETRY"):
        mod.main()
    assert executed == []


def test_deadline_stops_even_if_completed_history_is_present(pipeline):
    mod, _, now, executed = pipeline
    now[0] = datetime.fromisoformat("2026-09-17T12:11:38+08:00")
    mod.b.save(mod.SOURCE / "history.json", {"available": True})
    with pytest.raises(ValueError, match="DEADLINE_REACHED"):
        mod.main()
    assert executed == [] and not (mod.ROOT / "pipeline-attempt.json").exists()


def test_changed_dependency_rejected_before_source_use(pipeline):
    mod, plan, _, executed = pipeline
    path = mod.PROJECT / "dependency.py"
    path.write_text("changed", encoding="utf-8")
    plan["code_hashes"]["dependency.py"] = "not-its-hash"
    mod.b.save(mod.ROOT / "pipeline-plan.json", plan, replace=True)
    with pytest.raises(ValueError, match="DEPENDENCY_CHANGED"):
        mod.main()
    assert executed == []


def test_partial_qualification_cannot_freeze_identity_or_train(pipeline, monkeypatch):
    mod, _, _, executed = pipeline
    mod.b.save(mod.SOURCE / "history.json", {"available": True})

    def partial(script, plan):
        executed.append(str(script))
        mod.b.save(
            mod.SOURCE / "qualification-result.json",
            {
                "status": "QUALIFIED_SH_SZ_MONEYFLOW_WITH_LIMITS",
                "source_days": 1382,
                "original_rows_hash": "fixed-rows",
            },
        )

    monkeypatch.setattr(mod, "execute", partial)
    with pytest.raises(ValueError, match="SOURCE_NOT_QUALIFIED"):
        mod.main()
    assert len(executed) == 1 and not (mod.ROOT / "identity-plan.json").exists()
