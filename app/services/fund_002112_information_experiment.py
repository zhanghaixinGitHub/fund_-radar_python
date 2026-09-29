"""24 次新授权的独立实验执行器；不导入旧运行器，不恢复旧实验。

两个预登记假设各最多 12 次。全局账本先扣预算，工作进程凭一次性许可拟合；
保存、独立进程重新加载、另一次真实拟合逐一留证。中断不退款、不重复拟合。
"""

import hashlib
import io
import json
import os
import platform
import secrets
import subprocess
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import fund_002112_model_state_v2 as state
from app.services.fund_002112_round3_data import CLASSES, FEATURES, TIE_ORDER
from app.services.fund_002112_round3_model import LOGISTIC, compare_predictions, directions, metrics
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, quarter, read_json

PROJECT = Path(__file__).resolve().parents[2]
RUN = ROOT / "information-research/20260928-v1"
BASE = ROOT / "peer-training-ready/20260928-v2"
HYPOTHESES = ("H1_EARLY_OMISSION", "H2_EVENT_INFORMATION")
QUARTERS = ("2023Q2", "2023Q3", "2023Q4")
SLOTS = tuple(f"{q}-CANDIDATE-{r}" for q in QUARTERS for r in ("main", "replay")) + tuple(
    f"FULL-{v}-{r}" for v in ("CANDIDATE", "L20_ORIGINAL", "N7_ORIGINAL") for r in ("main", "replay")
)


def write(path, value):
    """排他创建并刷盘；半成品不覆盖，不能通过重试领取第二个拟合许可。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def once(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError("IMMUTABLE_RESULT_CHANGED:" + path.name)
    else:
        write(path, value)


def verify_files(files):
    for path, sha in files.items():
        if file_hash(path) != sha:
            raise ValueError("FROZEN_FILE_CHANGED:" + str(path))


@contextmanager
def lock():
    """整个新授权共用 Windows 进程锁；退出自动释放，不删除锁文件。"""
    import msvcrt

    with (RUN / ".execution.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def candidate_path(hypothesis):
    if hypothesis not in HYPOTHESES:
        raise ValueError("UNREGISTERED_HYPOTHESIS")
    return RUN / hypothesis


def freeze(hypothesis):
    """资料准入和独立审计都通过才能冻结候选；不改旧包的零预算。"""
    path = candidate_path(hypothesis)
    if (path / "freeze.json").exists():
        return load(hypothesis)[0]
    admission = RUN / ("early-admission" if hypothesis == HYPOTHESES[0] else "event-admission")
    audit_name = "independent-audit-v2.json" if hypothesis == HYPOTHESES[0] else "independent-audit.json"
    if not read_json(admission / "decision.json")["material_ready"] or not read_json(admission / audit_name)["passed"]:
        raise ValueError("PREPARATION_NOT_PASSED")
    auth = read_json(RUN / "protocol.json")
    if auth["maximum_real_fits"] != 24 or auth["previous_real_fits"] != 58:
        raise ValueError("AUTHORIZATION_CHANGED")
    verify_files(auth["sources"])
    once(RUN / "protocol-anchor.json", {"sha256": file_hash(RUN / "protocol.json")})
    plan = read_json(BASE / "execution-plan.json")
    expected = plan["inherited_environment"]
    state.check_environment()
    if platform.platform() != expected["platform"] or plan["recipe"] != LOGISTIC:
        raise ValueError("FIXED_ENVIRONMENT_OR_RECIPE_CHANGED")
    features = (
        list(FEATURES) if hypothesis == HYPOTHESES[0] else read_json(admission / "feature-contract.json")["features"]
    )
    controls = read_json(BASE / "frozen-controls.json")["controls"]
    old = Path(controls["2023Q2-L20-main"]["path"]).parents[1]
    files = {str(p): file_hash(p) for p in admission.rglob("*") if p.is_file()}
    files.update(read_json(admission / "sources.json")["files"])
    files.update({v["path"]: v["sha256"] for v in controls.values()})
    files.update(
        {str(p): file_hash(p) for p in (BASE / "execution-plan.json", old / "inputs.json", old / "folds.json")}
    )
    code = {
        str(Path(m.__file__).resolve()): file_hash(m.__file__)
        for m in list(sys.modules.values())
        if getattr(m, "__file__", None)
        and Path(m.__file__).suffix == ".py"
        and Path(m.__file__).resolve().is_relative_to(PROJECT)
        and not Path(m.__file__).resolve().is_relative_to(PROJECT / ".venv")
    }
    for name in (
        "scripts/fund_002112_information_research.py",
        "tests/test_fund_002112_information_experiment.py",
        "scripts/fund_002112_information_audit.py",
        "app/services/fund_002112_information_admission.py",
    ):
        code[str(PROJECT / name)] = file_hash(PROJECT / name)
    spec = {
        "hypothesis": hypothesis,
        "maximum_fits": 12,
        "global_maximum": 24,
        "features": features,
        "recipe": LOGISTIC,
        "environment": state.runtime_environment(),
        "platform": platform.platform(),
        "sources": files,
        "code": code,
        "controls": controls,
        "admission": str(admission),
        "original": str(old),
        "protocol_sha256": file_hash(RUN / "protocol.json"),
        "slots": list(SLOTS),
    }
    verify_files(files)
    write(path / "freeze.json", spec)
    write(path / "freeze-anchor.json", {"sha256": file_hash(path / "freeze.json")})
    load(hypothesis)
    return spec


def load(hypothesis):
    path = candidate_path(hypothesis)
    if file_hash(path / "freeze.json") != read_json(path / "freeze-anchor.json")["sha256"]:
        raise ValueError("FREEZE_CHANGED")
    spec = read_json(path / "freeze.json")
    if file_hash(RUN / "protocol.json") != spec["protocol_sha256"] or spec["slots"] != list(SLOTS):
        raise ValueError("PROTOCOL_CHANGED")
    verify_files(spec["sources"])
    verify_files(spec["code"])
    state.check_environment()
    if spec["environment"] != state.runtime_environment() or spec["platform"] != platform.platform():
        raise ValueError("ENVIRONMENT_CHANGED")
    admission = Path(spec["admission"])
    values = {
        "inputs": read_json(admission / "inputs.json"),
        "folds": read_json(admission / "folds.json"),
        "original_inputs": read_json(Path(spec["original"]) / "inputs.json"),
        "original_folds": read_json(Path(spec["original"]) / "folds.json"),
    }
    for kind in ("inputs", "original_inputs"):
        for part, rows in values[kind].items():
            if part not in ("train", "development"):
                raise ValueError("UNKNOWN_POOL_SPLIT")
            limit = "2023-12-31" if part == "train" else "2024-12-31"
            if any(r["target"] > limit or r["actual_direction"] not in CLASSES for r in rows):
                raise ValueError("FORBIDDEN_LABEL_SCOPE")
    index = {(r["fund_code"], r["target"]): r for r in values["inputs"]["train"]}
    if len(index) != len(values["inputs"]["train"]):
        raise ValueError("DUPLICATE_TRAIN_ROWS")
    for fold in values["folds"]:
        train = [index[tuple(k)] for k in fold["train_ids"]]
        if (
            digest(train) != fold["train_sha256"]
            or digest(fold["exam"]) != fold["exam_sha256"]
            or not fold["gate"]["passed"]
        ):
            raise ValueError("FOLD_CHANGED")
        if any(r["mature_at"] >= fold["start"] + "T00:00:00+08:00" for r in train):
            raise ValueError("IMMATURE_LABEL")
    return spec, values


def ledger():
    """全局连续哈希链；尾项删掉后仍有执行痕迹时拒绝继续。"""
    files = sorted((RUN / "fit-ledger").glob("*.json"))
    if len(files) > 24 or [p.name for p in files] != [f"{n:02d}.json" for n in range(1, len(files) + 1)]:
        raise ValueError("LEDGER_SEQUENCE_OR_BUDGET")
    previous, events = file_hash(RUN / "protocol.json"), []
    for n, p in enumerate(files, 1):
        event = read_json(p)
        if event["ordinal"] != n or event["previous_sha256"] != previous or event["hypothesis"] not in HYPOTHESES:
            raise ValueError("LEDGER_CHAIN_CHANGED")
        local = [e for e in events if e["hypothesis"] == event["hypothesis"]]
        if len(local) >= 12 or event["slot"] != SLOTS[len(local)]:
            raise ValueError("HYPOTHESIS_BUDGET_OR_ORDER")
        events.append(event)
        previous = file_hash(p)
    known = {(e["hypothesis"], e["slot"]) for e in events}
    for h in HYPOTHESES:
        for folder in ("started", "classifier-started", "classifier-completed", "completed", "restored", "failed"):
            if any((h, p.stem) not in known for p in (candidate_path(h) / folder).glob("*.json")):
                raise ValueError("ORPHAN_EXECUTION_EVIDENCE")
    return events


def reserve(hypothesis, slot, token):
    """消费必须在进程启动和 fit 前落盘，已失败槽位不再许可。"""
    events = ledger()
    local = [e for e in events if e["hypothesis"] == hypothesis]
    if len(events) >= 24 or len(local) >= 12:
        raise ValueError("BUDGET_EXHAUSTED")
    path = candidate_path(hypothesis)
    if slot != SLOTS[len(local)] or (path / "decision.json").exists():
        raise ValueError("SLOT_CONSUMED_OR_CANDIDATE_STOPPED")
    for event in local:
        completed(hypothesis, event["slot"])
    if len(local) >= 6 and not read_json(path / "historical-decision.json")["passed"]:
        raise ValueError("HISTORICAL_GATE_REQUIRED")
    n = len(events) + 1
    previous = RUN / "protocol.json" if n == 1 else RUN / "fit-ledger" / f"{n - 1:02d}.json"
    write(
        RUN / "fit-ledger" / f"{n:02d}.json",
        {
            "ordinal": n,
            "hypothesis": hypothesis,
            "slot": slot,
            "previous_sha256": file_hash(previous),
            "freeze_sha256": file_hash(path / "freeze.json"),
            "permit_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "reserved_at": datetime.now().astimezone().isoformat(),
            "budget_consumed": True,
            "refundable": False,
        },
    )


def slot_data(spec, values, slot):
    stage, variant, _ = slot.split("-")
    original = variant != "CANDIDATE"
    folds = values["original_folds" if original else "folds"]
    fold = next(f for f in folds if f["name"] == stage)
    pool = values["original_inputs" if original else "inputs"]["train"]
    index = {(r["fund_code"], r["target"]): r for r in pool}
    train = [index[tuple(k)] for k in fold["train_ids"]]
    if original and len(pool) != 4419:
        raise ValueError("ORIGINAL_POOL_CHANGED")
    features = list(FEATURES[:7]) if variant == "N7_ORIGINAL" else list(FEATURES) if original else spec["features"]
    exam = next(f["exam"] for f in values["folds"] if f["name"] == stage)
    return train, exam, fold["weights"], features


def matrix(rows, features):
    # 旧 ORIGINAL 可读取新事件行的原 20/7 列；候选必须提供完整冻结特征。
    x = np.asarray([r["x"][: len(features)] for r in rows], dtype=float)
    if x.shape != (len(rows), len(features)) or not np.isfinite(x).all():
        raise ValueError("INCOMPLETE_FINITE_INPUT_REQUIRED")
    return x


def predict(model, rows):
    with threadpool_limits(limits=1):
        scores = model["classifier"].predict_proba(model["scaler"].transform(matrix(rows, model["features"])))
    if not np.isfinite(scores).all():
        raise ValueError("NONFINITE_PREDICTIONS")
    return [
        {
            "fund_code": r["fund_code"],
            "target": r["target"],
            "actual_direction": r["actual_direction"],
            "input_hash": digest(r),
            "direction": d,
            "scores": p.tolist(),
        }
        for r, d, p in zip(rows, directions(scores), scores, strict=True)
    ]


def worker(hypothesis, slot, restore=False):
    spec, values = load(hypothesis)
    path = candidate_path(hypothesis)
    train, exam, weights, features = slot_data(spec, values, slot)
    if restore:
        receipt = completed(hypothesis, slot)
        model = joblib.load(io.BytesIO((path / "models" / (slot + ".joblib")).read_bytes()))
        if state.numeric_digest(model) != receipt["model_state"]:
            raise ValueError("RESTORED_MODEL_STATE_CHANGED")
        predictions = read_json(path / "predictions" / (slot + ".json"))
        deltas = {
            k: compare_predictions(predictions[k], predict(model, r)) for k, r in (("train", train), ("exam", exam))
        }
        once(
            path / "restored" / (slot + ".json"),
            {
                "passed": True,
                "independent_process": True,
                "deltas": deltas,
                "completed_sha256": file_hash(path / "completed" / (slot + ".json")),
                "real_fits": 0,
            },
        )
        return
    events = ledger()
    event = events[-1]
    if event["freeze_sha256"] != file_hash(path / "freeze.json"):
        raise ValueError("RESERVATION_FREEZE_CHANGED")
    token = os.environ.get("FUND_INFORMATION_FIT_PERMIT", "")
    if (
        event["hypothesis"] != hypothesis
        or event["slot"] != slot
        or not token
        or hashlib.sha256(token.encode()).hexdigest() != event["permit_sha256"]
    ):
        raise ValueError("INVALID_ONE_TIME_FIT_PERMIT")
    write(path / "started" / (slot + ".json"), {"pid": os.getpid(), "slot": slot, "ledger_ordinal": event["ordinal"]})
    x, w = matrix(train, features), np.asarray(weights, dtype=float)
    y = np.asarray([r["actual_direction"] for r in train])
    if (
        set(y) != set(CLASSES)
        or len(w) != len(train)
        or not np.isfinite(w).all()
        or min(w) <= 0
        or abs(sum(w) - len(w)) > 1e-8
    ):
        raise ValueError("WEIGHTS_OR_CLASSES_INVALID")
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaler = StandardScaler().fit(x, sample_weight=w)
        classifier = LogisticRegression(**LOGISTIC)
        write(path / "classifier-started" / (slot + ".json"), {"pid": os.getpid(), "slot": slot})
        classifier.fit(scaler.transform(x), y, sample_weight=w)
        write(path / "classifier-completed" / (slot + ".json"), {"slot": slot})
    if classifier.classes_.tolist() != list(CLASSES):
        raise ValueError("CLASS_ORDER_CHANGED")
    model = {
        "features": features,
        "recipe": LOGISTIC,
        "classes": list(CLASSES),
        "tie_order": list(TIE_ORDER),
        "training_hash": digest(train),
        "weights_hash": digest(weights),
        "classifier": classifier,
        "scaler": scaler,
    }
    file = path / "models" / (slot + ".joblib")
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("xb") as stream:
        joblib.dump(model, stream)
        stream.flush()
        os.fsync(stream.fileno())
    write(path / "predictions" / (slot + ".json"), {"train": predict(model, train), "exam": predict(model, exam)})
    files = [file, path / "predictions" / (slot + ".json")]
    files += [path / folder / (slot + ".json") for folder in ("started", "classifier-started", "classifier-completed")]
    write(
        path / "completed" / (slot + ".json"),
        {
            "files": {str(p): file_hash(p) for p in files},
            "model_state": state.numeric_digest(model),
            "ledger_ordinal": event["ordinal"],
            "ledger_sha256": file_hash(RUN / "fit-ledger" / f"{event['ordinal']:02d}.json"),
        },
    )


def completed(hypothesis, slot):
    receipt = read_json(candidate_path(hypothesis) / "completed" / (slot + ".json"))
    verify_files(receipt["files"])
    if receipt["ledger_sha256"] != file_hash(RUN / "fit-ledger" / f"{receipt['ledger_ordinal']:02d}.json"):
        raise ValueError("CHECKPOINT_LEDGER_CHANGED")
    return receipt


def child(hypothesis, slot, *, restore=False, token=""):
    env = os.environ.copy()
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["FUND_INFORMATION_FIT_PERMIT"] = token
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-m",
            "scripts.fund_002112_information_research",
            "_restore" if restore else "_fit",
            "--hypothesis",
            hypothesis,
            "--slot",
            slot,
        ],
        env=env,
        cwd=PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        raise RuntimeError("WORKER_FAILED:" + result.stderr[-4000:])


def replay(hypothesis, main):
    path = candidate_path(hypothesis)
    slots = [main, main.removesuffix("-main") + "-replay"]
    receipts = [completed(hypothesis, s) for s in slots]
    if receipts[0]["model_state"] != receipts[1]["model_state"]:
        raise ValueError("INDEPENDENT_FIT_STATE_MISMATCH")
    predictions = [read_json(path / "predictions" / (s + ".json")) for s in slots]
    for s in slots:
        if not read_json(path / "restored" / (s + ".json"))["passed"]:
            raise ValueError("RESTORE_REQUIRED")
    return {
        "passed": True,
        "main": main,
        "independent_fits": True,
        "delta": {p: compare_predictions(predictions[0][p], predictions[1][p]) for p in ("train", "exam")},
    }


def comparison(predictions, historical):
    """原性能门槛保持；按基金、日期和实际答案配对，事件特征可使输入摘要不同。"""
    candidate = predictions["CANDIDATE"]
    keys = [(r["fund_code"], r["target"], r["actual_direction"]) for r in candidate]
    if len(set(keys)) != len(keys) or any(
        [(r["fund_code"], r["target"], r["actual_direction"]) for r in rows] != keys for rows in predictions.values()
    ):
        raise ValueError("COMPARISON_IDENTITY_CHANGED")
    totals = {n: metrics(r) for n, r in predictions.items()}
    quarters = {
        q: {n: metrics([r for r in rows if quarter(r["target"]) == q]) for n, rows in predictions.items()}
        for q in sorted({quarter(r["target"]) for r in candidate})
    }
    constants = {c: sum(r["actual_direction"] == c for r in candidate) for c in CLASSES}
    controls = ("L20_ORIGINAL", "N7_ORIGINAL")
    checks = {
        "total_strictly_above_controls": all(totals["CANDIDATE"]["correct"] > totals[n]["correct"] for n in controls),
        "total_strictly_above_constants": all(totals["CANDIDATE"]["correct"] > n for n in constants.values()),
    }
    checks.update(
        {
            "class_not_worse_" + c: all(
                totals["CANDIDATE"]["class_correct"][c] >= totals[n]["class_correct"][c] for n in controls
            )
            for c in CLASSES
        }
    )
    if historical:
        checks["two_quarters_not_worse_L20"] = (
            sum(q["CANDIDATE"]["correct"] >= q["L20_ORIGINAL"]["correct"] for q in quarters.values()) >= 2
        )
    pairs = {}
    for name in controls:
        daily = [
            {
                "target": a["target"],
                "actual": a["actual_direction"],
                "candidate": a["direction"],
                "control": b["direction"],
                "delta": int(a["direction"] == a["actual_direction"]) - int(b["direction"] == b["actual_direction"]),
            }
            for a, b in zip(candidate, predictions[name], strict=True)
        ]

        def counts(rows):
            values = [r["delta"] for r in rows]
            return {"gained": values.count(1), "lost": values.count(-1), "net": sum(values)}

        pairs[name] = {
            **counts(daily),
            "daily": daily,
            "classes": {c: counts([r for r in daily if r["actual"] == c]) for c in CLASSES},
            "quarters": {q: counts([r for r in daily if quarter(r["target"]) == q]) for q in quarters},
        }
    return {
        "models": totals,
        "quarters": quarters,
        "constants": constants,
        "pairs": pairs,
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": [k for k, v in checks.items() if not v],
        "evidence_type": "DEVELOPMENT_VALIDATION_NOT_BLIND_TEST",
    }


def stage_result(hypothesis, historical):
    path = candidate_path(hypothesis)
    spec = read_json(path / "freeze.json")
    predictions = {n: [] for n in ("CANDIDATE", "L20_ORIGINAL", "N7_ORIGINAL")}
    engineering = []
    for q in QUARTERS if historical else ("FULL",):
        for variant in predictions:
            if historical and variant != "CANDIDATE":
                key = f"{q}-{variant.split('_')[0]}"
                a, b = [spec["controls"][key + "-" + suffix] for suffix in ("main", "replay")]
                if read_json(a["path"]) != read_json(b["path"]):
                    raise ValueError("FROZEN_CONTROL_REPLAY_CHANGED")
                predictions[variant].extend(read_json(a["path"])["exam"])
            else:
                slot = f"{q}-{variant}-main"
                engineering.append(replay(hypothesis, slot))
                predictions[variant].extend(read_json(path / "predictions" / (slot + ".json"))["exam"])
    result = comparison(predictions, historical)
    result.update(engineering=engineering, engineering_passed=all(e["passed"] for e in engineering))
    result["passed"] = result["passed"] and result["engineering_passed"]
    return result


def status():
    events = ledger()
    fits = sum(len(list((candidate_path(h) / "classifier-started").glob("*.json"))) for h in HYPOTHESES)
    return {
        "authorized": 24,
        "consumed": len(events),
        "actual_fits": fits,
        "cumulative_actual_fits": 58 + fits,
        "unused": 24 - len(events),
        "hypotheses": {h: sum(e["hypothesis"] == h for e in events) for h in HYPOTHESES},
    }


def run(hypothesis, interrupt_after=None):
    with lock():
        load(hypothesis)
        path = candidate_path(hypothesis)
        if (path / "decision.json").exists():
            verify(hypothesis)
            return read_json(path / "decision.json")
        slot = None
        try:
            for historical, slots in ((True, SLOTS[:6]), (False, SLOTS[6:])):
                for slot in slots:
                    event = next((e for e in ledger() if e["hypothesis"] == hypothesis and e["slot"] == slot), None)
                    if event is None:
                        token = secrets.token_hex(32)
                        reserve(hypothesis, slot, token)
                        child(hypothesis, slot, token=token)
                    elif not (path / "completed" / (slot + ".json")).exists():
                        raise ValueError("CONSUMED_SLOT_INCOMPLETE_NO_REFIT")
                    child(hypothesis, slot, restore=True)
                    if slot.endswith("-replay"):
                        replay(hypothesis, slot.removesuffix("-replay") + "-main")
                    print(json.dumps({"completed": slot, **status()}), flush=True)
                    if interrupt_after == len([e for e in ledger() if e["hypothesis"] == hypothesis]):
                        return {"status": "VALID_CHECKPOINT_SAVED", **status()}
                result = stage_result(hypothesis, historical)
                once(path / ("historical-decision.json" if historical else "full-decision.json"), result)
                if not result["passed"]:
                    decision = {
                        "status": "STOP_HISTORICAL_GATE_FAILED" if historical else "STOP_FULL_GATE_FAILED",
                        "failed_checks": result["failed_checks"],
                        "full_executed": not historical,
                        "adopted": False,
                        **status(),
                    }
                    break
            else:
                decision = {
                    "status": "PASSED_ALL_GATES_PENDING_ADOPTION",
                    "full_executed": True,
                    "adopted": False,
                    **status(),
                }
        except Exception as exc:
            decision = {
                "status": "STOP_ENGINEERING_FAILURE",
                "slot": slot,
                "reason": str(exc),
                "adopted": False,
                **status(),
            }
        once(path / "decision.json", decision)
        once(path / "decision-anchor.json", {"sha256": file_hash(path / "decision.json")})
        return decision


def verify(hypothesis):
    """零拟合验完整性、独立载入和决定；可重复调用，不消耗拟合额度。"""
    load(hypothesis)
    path = candidate_path(hypothesis)
    if (path / "decision.json").exists() and file_hash(path / "decision.json") != read_json(
        path / "decision-anchor.json"
    )["sha256"]:
        raise ValueError("DECISION_CHANGED")
    for event in ledger():
        if event["hypothesis"] == hypothesis and (path / "completed" / (event["slot"] + ".json")).exists():
            child(hypothesis, event["slot"], restore=True)
    for historical, name in ((True, "historical-decision.json"), (False, "full-decision.json")):
        if (path / name).exists() and read_json(path / name) != stage_result(hypothesis, historical):
            raise ValueError("RECOMPUTED_DECISION_CHANGED")
    return {"passed": True, **status()}
