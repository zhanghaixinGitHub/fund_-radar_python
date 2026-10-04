"""新实验人工边界测试；任何监督拟合均为随机生成 X/人工标签，单独记账。"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from scripts import fund_002112_existing_data_fit_v1 as f
from scripts import fund_002112_existing_data_v1 as r


def synthetic_fit(x, y, candidate, **kwargs):
    """每次人工监督调用都记录，不混入真实研究预算。"""
    log = os.environ.get("SYNTHETIC_FIT_LOG")
    if log:
        r.append(
            Path(log),
            {
                "at": r.now(),
                "data": "ARTIFICIAL_ONLY",
                "rows": len(x),
                "candidate": candidate,
                "x_digest": f.digest(x),
                "pid": os.getpid(),
            },
        )
    return f.fit_estimator(x, y, candidate, **kwargs)


def artificial(n=150):
    x = np.random.default_rng(9).normal(size=(n, 8)).tolist()
    y = ["UP" if row[0] > 0 else "DOWN" for row in x]
    return x, y


def row(**updates):
    result = {
        "U": "2025-01-02",
        "T": "2024-12-31",
        "S": "2024-12-30",
        "cutoff": "2025-01-02T08:00:00+08:00",
        "available": dict.fromkeys("NMHIFG", True),
        "reasons": {},
        "features": {k: 1.0 for k in "NMHIFG"},
        "lag_sessions": 1,
        "holdings_available": True,
    }
    result.update(updates)
    return result


def test_fixed_algorithms_and_dependencies():
    assert f.LR_PARAMS == {"C": 1.0, "solver": "lbfgs", "tol": 1e-8, "max_iter": 1000, "random_state": 0}
    assert f.H_PARAMS["early_stopping"] is False
    assert f.dependencies("C") == ["A", "C"]
    assert f.dependencies("F") == ["A", "D", "F"]
    assert set(f.GROUPS) == set("ABCDEFGH")


def test_fallback_never_uses_labels_or_fills_missing():
    names = {g: [g] for g in "NMHIFG"}
    x = row()
    x["available"]["H"] = False
    x["features"]["H"] = None
    x["reasons"]["H"] = "停牌"
    original = copy.deepcopy(x)
    assert f.route(x, "E", dict.fromkeys("ADE"), names) == ("D", "H:停牌")
    x["label"] = "UP"
    assert f.route(x, "E", dict.fromkeys("ADE"), names)[0] == "D"
    x.pop("label")
    assert x == original
    x["available"]["I"] = False
    assert f.route(x, "E", dict.fromkeys("ADE"), names)[0] == "A"
    x["available"]["N"] = False
    assert f.route(x, "E", dict.fromkeys("ADE"), names)[0] is None


def test_unknown_event_is_missing_not_zero():
    names = {g: [g] for g in "NMHIFG"}
    x = row()
    x["available"]["G"] = False
    x["features"]["G"] = None
    assert not f.usable(x, "G", names)
    assert f.route(x, "G", dict.fromkeys("ADG"), names)[0] == "D"


def test_scaler_only_training_two_classes_and_tie():
    x, y = artificial()
    x = np.asarray(x)
    x[:, 7] = 2
    model = synthetic_fit(x.tolist(), y, "A")
    assert np.allclose(model["scaler"].mean_, x.mean(axis=0))
    assert model["zero_variance_columns"] == [7]
    before = model["scaler"].mean_.copy()
    p = f.probabilities(model, [[999.0] * 8])
    assert np.array_equal(before, model["scaler"].mean_)
    assert model["unlearnable_classes"] == ["FLAT"] and p[0, 0] == 0
    assert f.predict_label([0.5, 0.5, 0]) == "FLAT"
    assert f.predict_label([0, 0.5, 0.5]) == "DOWN"


@pytest.mark.parametrize("candidate", ["A", "H"])
def test_save_load_and_independent_refit(tmp_path, candidate):
    x, y = artificial()
    model = synthetic_fit(x, y, candidate)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    assert f.compare_models(model, joblib.load(path), x)["passed"]
    script = (
        "import json,os,joblib;from pathlib import Path;"
        "from scripts.test_fund_002112_existing_data_v1 import artificial,synthetic_fit;"
        "x,y=artificial();m=synthetic_fit(x,y,os.environ['SYNTHETIC_CANDIDATE']);"
        "joblib.dump(m,os.environ['SYNTHETIC_MODEL_PATH'])"
    )
    env = {**os.environ, "SYNTHETIC_CANDIDATE": candidate, "SYNTHETIC_MODEL_PATH": str(tmp_path / "replay.joblib")}
    subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script], cwd=r.PY, env=env, check=True)
    assert f.compare_models(model, joblib.load(tmp_path / "replay.joblib"), x)["passed"]


def test_maturity_before_fit_even_when_target_inside_year():
    names = {g: [g] for g in "NMHIFG"}
    rows = [row(U="2024-12-31"), row(U="2024-12-30"), row(U="2025-01-02")]
    labels = {
        "labels": {x["U"]: "UP" for x in rows},
        "maturity": {
            "2024-12-31": "2025-01-14T08:00:00+08:00",
            "2024-12-30": "2024-12-31T08:00:00+08:00",
            "2025-01-02": "2025-01-03T08:00:00+08:00",
        },
    }
    chosen = r.training_rows(rows, labels, "V1", "2025-01-02T08:00:00+08:00", "A", names)
    assert [x["U"] for x in chosen] == ["2024-12-30"]


def test_budget_48_failure_consumed_and_old_unchanged(tmp_path):
    old = tmp_path / "old-budget.json"
    old.write_text('{"consumed":70}')
    original = r.sha(old)
    root = tmp_path / "new"
    budget = r.Budget(root)
    for bucket, maximum in r.BUCKETS.items():
        for i in range(maximum):
            budget.reserve(f"{bucket}-{i}", bucket, {"test": True})
    assert budget.state()["consumed"] == 48
    # 无 FINISHED 也消费，不能退款或同名重跑。
    with pytest.raises(RuntimeError):
        budget.reserve("49", "retry", {})
    with pytest.raises(RuntimeError):
        budget.reserve("validation-0", "validation", {})
    assert r.sha(old) == original


def test_atomic_budget_reservation_across_processes(tmp_path):
    root = tmp_path / "atomic"
    script = (
        "import os;from pathlib import Path;from scripts.fund_002112_existing_data_v1 import Budget;"
        "Budget(Path(os.environ['SYNTHETIC_BUDGET_ROOT'])).reserve('same','validation',{})"
    )
    env = {**os.environ, "SYNTHETIC_BUDGET_ROOT": str(root)}
    processes = [
        subprocess.Popen(
            [sys.executable, "-X", "utf8", "-B", "-c", script],
            cwd=r.PY,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    codes = []
    for p in processes:
        p.communicate(timeout=30)
        codes.append(p.returncode)
    assert sorted(codes) == [0, 1]
    assert r.Budget(root).state()["consumed"] == 1


def test_checkpoint_changed_blocks_resume(tmp_path):
    p = tmp_path / "old-snapshot.txt"
    p.write_text("protected")
    r.checkpoint(tmp_path, "sample", [p])
    p.write_text("modified")
    with pytest.raises(ValueError, match="CHECKPOINT_CHANGED"):
        r.require_checkpoint(tmp_path, "sample")


def test_test_labels_require_frozen_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "verify_frozen", lambda root: {})
    r.save(tmp_path / "protocol.json", {})
    with pytest.raises(FileNotFoundError):
        r.label_access(tmp_path, "T", "evaluate", ["2026-01-02"], [row(U="2026-01-02")])
    assert not (tmp_path / "label-access-ledger.jsonl").exists()


def test_metrics_keep_flat_and_full_calendar_mask():
    dates = [f"2026-01-{d:02d}" for d in range(1, 21)]
    a = [
        {"U": d, "actual": "FLAT" if i == 0 else "UP", "predicted": "UP", "probabilities": [0, 0, 1]}
        for i, d in enumerate(dates)
        if i not in (5, 6, 7)
    ]
    b = [{**x, "predicted": "DOWN" if i % 2 else "UP"} for i, x in enumerate(a)]
    result = f.paired(a, b, dates)
    assert result["n"] == 17 and result["bootstrap"]["missing_mask_count"] == 3
    assert result["baseline"]["per_class"]["FLAT"]["n"] == 1
    assert result["baseline"]["per_class"]["DOWN"]["recall"] is None
    assert result["net_correct"] == result["paired_cells"]["candidate_only"] - result["paired_cells"]["baseline_only"]


def test_selection_uses_fixed_tiebreak_and_no_test_labels():
    def pair(gain, drop=0):
        return {
            "n": 100,
            "net_correct": gain,
            "candidate": {"per_class": {c: {"recall": 0.5 - drop} for c in ("UP", "DOWN")}},
            "baseline": {"per_class": {c: {"recall": 0.5} for c in ("UP", "DOWN")}},
        }

    comparisons = {c: pair(2) for c in "BCDEFGH"}
    folds = {s: copy.deepcopy(comparisons) for s in ("V1", "V2", "V3")}
    counts = dict(zip("BCDEFGH", [12, 21, 27, 36, 42, 48, 36], strict=True))
    result = f.choose_candidate(comparisons, folds, counts, set("BCDEFGH"))
    assert result["selected"] == "B" and result["status"] == "VALIDATION_GAIN"
    for c in comparisons:
        comparisons[c] = pair(-1)
    result = f.choose_candidate(comparisons, folds, counts, set("BCDEFGH"))
    assert result["selected"] == "B" and result["status"] == "NO_VALIDATION_GAIN"


def test_real_worker_rejects_unreserved_or_other_root(tmp_path):
    path = tmp_path / "job.json"
    r.save(path, {"root": str(tmp_path), "plan": "unreserved"})
    with pytest.raises(ValueError, match="REAL_WORKER_ROOT_MISMATCH"):
        r.worker(path)


def test_go_and_transitive_source_hash_checks(tmp_path):
    artifact = tmp_path / "snapshot/artificial.txt"
    artifact.parent.mkdir()
    artifact.write_text("ARTIFICIAL_ONLY")
    source = tmp_path / "code.py"
    source.write_text("# synthetic source")
    r.save(
        tmp_path / "handoff/data/ready.json",
        {
            "status": "READY",
            "experiment_id": r.EXPERIMENT,
            "artifacts": {"snapshot/artificial.txt": r.sha(artifact)},
            "code_sha256": {str(source): r.sha(source)},
        },
    )
    r.save(tmp_path / "code-manifest.json", {"files": {str(source): r.sha(source)}})
    r.save(
        tmp_path / "protocol.json",
        {
            "spec_sha256": r.sha(r.DOC),
            "code_manifest_sha256": r.sha(tmp_path / "code-manifest.json"),
            "data_ready_sha256": r.sha(tmp_path / "handoff/data/ready.json"),
        },
    )
    with pytest.raises(FileNotFoundError):
        r.verify_frozen(tmp_path)
    r.save(
        tmp_path / "handoff/coordinator-train-release.json",
        {
            "status": "GO",
            "protocol_sha256": r.sha(tmp_path / "protocol.json"),
            "code_manifest_sha256": r.sha(tmp_path / "code-manifest.json"),
            "data_ready_sha256": r.sha(tmp_path / "handoff/data/ready.json"),
        },
    )
    r.verify_frozen(tmp_path)
    artifact.write_text("altered after freeze")
    with pytest.raises(RuntimeError, match="A_ARTIFACT_HASH_MISMATCH"):
        r.verify_frozen(tmp_path)
    artifact.write_text("ARTIFICIAL_ONLY")
    source.write_text("changed code")
    with pytest.raises(RuntimeError, match="FROZEN_CODE_CHANGED"):
        r.verify_frozen(tmp_path)


def test_worker_claim_prevents_same_reservation_fitting_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "verify_frozen", lambda root: {})
    x, y = artificial()
    path = tmp_path / "fit-attempts/001/job.json"
    r.save(
        path,
        {
            "root": str(tmp_path),
            "plan": "synthetic-claim",
            "candidate": "A",
            "x": x,
            "y": y,
            "probe_x": x,
            "extended_iterations": False,
        },
    )
    r.Budget(tmp_path).reserve("synthetic-claim", "validation", {"job_sha256": r.sha(path)})
    original = f.fit_estimator

    def wrapped(x, y, candidate, **kwargs):
        log = os.environ.get("SYNTHETIC_FIT_LOG")
        if log:
            r.append(
                Path(log), {"at": r.now(), "data": "ARTIFICIAL_ONLY", "test": "worker_claim", "candidate": candidate}
            )
        return original(x, y, candidate, **kwargs)

    monkeypatch.setattr(f, "fit_estimator", wrapped)
    r.worker(path)
    with pytest.raises(FileExistsError):
        r.worker(path)
    assert r.Budget(tmp_path).state()["consumed"] == 1


def test_active_revision_binds_real_ready_and_preserves_first_freeze(tmp_path, monkeypatch):
    old_ready = tmp_path / "handoff/data/ready.json"
    new_ready = tmp_path / "handoff/data/revisions/v2/ready.json"
    r.save(old_ready, {"status": "READY", "revision": "r1"})
    r.save(new_ready, {"status": "READY", "revision": "v2"})
    old_sha = r.sha(old_ready)
    r.save(tmp_path / "handoff/data/active-revision.json", {"revision": "v2", "ready_sha256": r.sha(new_ready)})
    fake_api = SimpleNamespace(
        load_inputs=lambda root: {"data_ready_path": str(new_ready), "data_ready_sha256": r.sha(new_ready)}
    )
    monkeypatch.setattr(r.importlib, "import_module", lambda module: fake_api)
    path, pointer = r.active_data_ready(tmp_path)
    assert path == new_ready and pointer
    r.save(tmp_path / "protocol.json", {"data_ready_sha256": old_sha})
    r.save(tmp_path / "code-manifest.json", {"r1": "code"})
    r.save(tmp_path / "validation-results.json", {"r1": "passed"})
    r.save(tmp_path / "handoff/training/ready-for-review.json", {"r1": "waiting"})
    protocol_sha = r.sha(tmp_path / "protocol.json")
    r.supersede_freeze(tmp_path)
    archive = tmp_path / "handoff/training/revisions/r1-superseded"
    assert r.sha(archive / "protocol.json") == protocol_sha
    assert r.sha(old_ready) == old_sha
    assert r.Budget(tmp_path).state()["consumed"] == 0
    assert r.read(tmp_path / "freeze-supersession.json")["supersedes_protocol_sha256"] == protocol_sha


def test_no_protocol_supersession_after_real_reservation(tmp_path):
    r.Budget(tmp_path).reserve("consumed-before-crash", "validation", {})
    with pytest.raises(RuntimeError, match="CANNOT_SUPERSEDE_AFTER_REAL_ACCESS"):
        r.supersede_freeze(tmp_path)


@pytest.mark.parametrize("failed_candidate", ["A", "B"])
def test_exhausted_retry_keeps_failure_and_next_branch_without_orphan_job(tmp_path, monkeypatch, failed_candidate):
    """用人工回执注入不收敛，重试桶已满时不得写第二次任务或挪用其他桶。"""
    names = {g: [g] for g in "NMHIFG"}
    days = [f"2024-{month:02d}-{day:02d}" for month in range(1, 5) for day in range(1, 21)]
    rows = [
        row(U=d, source_digest="ARTIFICIAL", features={g: float(i % (j + 2)) for j, g in enumerate("NMHIFG")})
        for i, d in enumerate(days)
    ]
    rows.append(row(source_digest="ARTIFICIAL"))
    labels = {
        "labels": {d: "UP" if i % 2 else "DOWN" for i, d in enumerate(days)},
        "maturity": {d: d + "T08:00:00+08:00" for d in days},
        "excluded": {},
    }
    r.save(tmp_path / "protocol.json", {"input_digest": "ARTIFICIAL"})
    r.save(tmp_path / "code-manifest.json", {})
    budget = r.Budget(tmp_path)
    for i in range(3):
        budget.reserve(f"artificial-used-retry-{i}", "retry", {"simulation_only": True})
    monkeypatch.setattr(r, "label_access", lambda *args: labels)
    worker_candidates = []

    def run(command, **kwargs):
        path = Path(command[-1])
        job = r.read(path)
        worker_candidates.append(job["candidate"])
        if job["candidate"] == failed_candidate:
            r.save(path.parent / "result.json", {"status": "NON_CONVERGED", "warnings": ["ARTIFICIAL_INJECTION"]})
        else:
            bundle = synthetic_fit(job["x"], job["y"], job["candidate"])
            joblib.dump(bundle, path.parent / "model.joblib")
            r.save(
                path.parent / "result.json",
                {
                    "status": "SUCCESS",
                    "params": bundle["params"],
                    "unlearnable_classes": bundle["unlearnable_classes"],
                    "zero_variance_columns": bundle["zero_variance_columns"],
                    "structure_sha256": f.digest(f.model_structure(bundle)),
                },
            )
        return SimpleNamespace(returncode=0, stdout="ARTIFICIAL_ONLY", stderr="")

    monkeypatch.setattr(r.subprocess, "run", run)
    if failed_candidate == "A":
        with pytest.raises(RuntimeError, match="COMMON_BASELINE_CANNOT_FIT"):
            r.fit_stage(tmp_path, "V1", ["A", "C"], rows, names)
        assert worker_candidates == ["A"]
    else:
        refs = r.fit_stage(tmp_path, "V1", ["B", "C"], rows, names)
        assert refs["C"]["status"] == "SUCCESS" and worker_candidates == ["B", "C"]
        assert f.route(rows[0], "B", dict.fromkeys("AC"), names)[0] == "A"
        # 再续跑整阶段只复用失败和成功回执，不再次消费或创建job。
        r.fit_stage(tmp_path, "V1", ["B", "C"], rows, names)
        assert worker_candidates == ["B", "C"]
    failed = r.read(tmp_path / "models/v1" / (failed_candidate + ".json"))
    assert failed["status"] == "FAILED" and failed["reason"] == "RETRY_BUDGET_EXHAUSTED"
    assert failed["retry_attempted"] is False
    state = budget.state()
    assert state["consumed_by_bucket"]["retry"] == 3
    assert state["consumed_by_bucket"]["validation"] == len(worker_candidates)
    jobs = list((tmp_path / "fit-attempts").glob("*/job.json"))
    assert len(jobs) == len(worker_candidates)
    reserved = {entry["plan"]: entry for entry in state["reservations"]}
    assert all(
        r.read(path)["plan"] in reserved and r.sha(path) == reserved[r.read(path)["plan"]]["job_sha256"]
        for path in jobs
    )
    assert not (tmp_path / "fit-attempts" / f"{state['consumed'] + 1:03d}").exists()


def test_synthetic_full_validation_test_final_resume(tmp_path, monkeypatch):
    """完整编排用人工日历/数据；替身只取代外部输入、只读数据库及进程启动。"""
    names = {g: [g] for g in "NMHI"} | {"F": [], "G": []}
    dates = []
    # 每年人工 150 日，故不借用真实665日或真实标签；覆盖全部三个验证分段。
    for year in (2024, 2025, 2026):
        for month in range(1, 10 if year == 2026 else 13):
            dates += [f"{year}-{month:02d}-{day:02d}" for day in range(1, 16)]
    rng = np.random.default_rng(85)
    rows = [
        row(
            U=d,
            T=d,
            S=d,
            cutoff=d + "T08:00:00+08:00",
            source_digest="SYNTHETIC",
            label_mature_at=d + "T08:00:00+08:00",
            features=dict(zip("NMHI", rng.normal(size=4), strict=True)),
            available=dict.fromkeys("NMHI", True) | {"F": False, "G": False},
        )
        for d in dates
    ]
    all_labels = {d: "UP" if i % 3 == 0 else "DOWN" for i, d in enumerate(dates)}
    protocol = {"input_digest": "SYNTHETIC", "calendar": dates, "final_label_cutoff": "2026-09-30T08:00:00+08:00"}
    r.save(tmp_path / "protocol.json", protocol)
    r.save(tmp_path / "code-manifest.json", {})
    r.save(tmp_path / "handoff/coordinator-train-release.json", {"status": "SYNTHETIC_TEST_ONLY"})
    monkeypatch.setattr(r, "verify_frozen", lambda root, **kw: protocol)
    monkeypatch.setattr(r, "inputs", lambda root: ({"optional_decisions": "synthetic"}, rows, names))

    def labels(root, stage, purpose, targets, _rows):
        if purpose == "evaluate" and stage == "T":
            r.require_checkpoint(root, "predict-test")
        if stage == "FINAL":
            r.require_checkpoint(root, "evaluate-test")
        result = {
            "labels": {d: all_labels[d] for d in targets},
            "maturity": {d: d + "T08:00:00+08:00" for d in targets},
            "excluded": {},
            "stage": stage,
            "purpose": purpose,
        }
        path = root / "labels" / (stage.lower() + "-" + purpose + ".json")
        if not path.exists():
            r.save(path, result)
        return result

    monkeypatch.setattr(r, "label_access", labels)

    def run(command, **kwargs):
        job_path = Path(command[-1])
        job = r.read(job_path)
        bundle = synthetic_fit(job["x"], job["y"], job["candidate"], retry=job["extended_iterations"])
        joblib.dump(bundle, job_path.parent / "model.joblib")
        r.save(
            job_path.parent / "result.json",
            {
                "status": "SUCCESS",
                "pid": len(command),
                "params": bundle["params"],
                "unlearnable_classes": bundle["unlearnable_classes"],
                "zero_variance_columns": bundle["zero_variance_columns"],
                "structure_sha256": f.digest(f.model_structure(bundle)),
            },
        )
        return SimpleNamespace(returncode=0, stdout="ARTIFICIAL_ONLY", stderr="")

    monkeypatch.setattr(r.subprocess, "run", run)
    monkeypatch.setattr(r, "actual_prediction_comparison", lambda *args: {"synthetic_test": True})
    r.train_validation(tmp_path)
    r.select_and_replay(tmp_path)
    r.predict_test(tmp_path)
    before = r.Budget(tmp_path).state()["consumed"]
    r.predict_test(tmp_path)
    assert r.Budget(tmp_path).state()["consumed"] == before
    r.evaluate_test(tmp_path)
    r.fit_final(tmp_path)
    before = r.Budget(tmp_path).state()["consumed"]
    r.fit_final(tmp_path)
    assert r.Budget(tmp_path).state()["consumed"] == before <= 48
    assert r.read(tmp_path / "final-training-summary.json")["research_only"]
