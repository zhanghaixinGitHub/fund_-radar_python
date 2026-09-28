"""002112 补参考资料的独立执行器：固定 12 槽位，先历史门槛，后 FULL。

仅复用固定模型的纯计算及序列化函数；不导入旧实验运行器。所有真实拟合
先永久记账，再通过一次性许可在新进程执行。失败不退款，不自动重试。
"""

import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from scripts.fund_002112_peer_ready import preflight as material_preflight

from app.services import fund_002112_model_state_v2 as state
from app.services import fund_002112_round3_model as model
from app.services.fund_002112_round3_data import CLASSES, FEATURES, TIE_ORDER, digest, file_hash, now, read
from app.services.fund_002112_zero_fit_review import block_indices, quarter

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / ".local-runs/fund-exposure-002112"
PACKAGE = ROOT / "peer-training-ready/20260928-v2"
RUN = ROOT / "peer-recovered-experiment/20260928-v1"
HISTORICAL = ("2023Q2", "2023Q3", "2023Q4")
SLOTS = tuple(f"{q}-L20_RECOVERED-{r}" for q in HISTORICAL for r in ("main", "replay")) + tuple(
    f"FULL-{v}-{r}" for v in ("L20_RECOVERED", "L20_ORIGINAL", "N7_ORIGINAL") for r in ("main", "replay")
)
BUDGET = 12
BOOTSTRAP = {"block_length": 10, "replications": 10000, "seed": 20260926}


def exclusive(path, value):
    """排他创建并刷盘；损坏或半写文件不能被重试覆盖。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("IMMUTABLE_RESULT_CHANGED:" + path.name)
    else:
        exclusive(path, value)


def verify_files(files, root=None):
    for name, expected in files.items():
        path = Path(name) if root is None else root / name
        if file_hash(path) != expected:
            raise ValueError("FROZEN_FILE_CHANGED:" + str(path))


@contextmanager
def lock(path):
    """Windows 字节锁随进程退出释放，避免依靠删除残留锁文件恢复。"""
    import msvcrt

    path.mkdir(parents=True, exist_ok=True)
    with (path / ".execution.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise ValueError("EXPERIMENT_ALREADY_RUNNING") from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def environment(plan):
    expected = plan["inherited_environment"]
    actual = state.runtime_environment()
    if actual != {"python": expected["python"], **expected["libraries"]}:
        raise ValueError("ENVIRONMENT_CHANGED")
    if platform.platform() != expected["platform"] or expected["threads"] != 1:
        raise ValueError("PLATFORM_OR_THREADS_CHANGED")
    if (
        plan["recipe"] != model.LOGISTIC
        or plan["features"] != list(FEATURES)
        or plan["classes"] != list(CLASSES)
        or plan["tie_order"] != list(TIE_ORDER)
        or plan["proposed_slots_not_reserved"] != list(SLOTS)
    ):
        raise ValueError("FIXED_RECIPE_CHANGED")
    return {**expected, "executable": sys.executable}


def freeze(path=RUN):
    """将本次明确授权写入新协议；旧资料包预算及旧结果完全不变。"""
    if (path / "protocol.json").exists():
        load_frozen(path)
        return
    if (path / "attempts").exists():
        raise ValueError("CANNOT_REFREEZE_CONSUMED_RUN")
    if not material_preflight()["material_ready"]:
        raise ValueError("MATERIAL_NOT_READY")
    plan = read(PACKAGE / "execution-plan.json")
    actual = environment(plan)
    controls = read(PACKAGE / "frozen-controls.json")["controls"]
    old = Path(controls["2023Q2-L20-main"]["path"]).parents[1]
    source_files = {str(p): file_hash(p) for p in PACKAGE.iterdir() if p.is_file()}
    source_files.update(read(PACKAGE / "sources.json")["files"])
    source_files.update(read(PACKAGE / "execution-protocol.json")["files"])
    source_files.update(plan["frozen_plan"]["sources"])
    source_files.update({v["path"]: v["sha256"] for v in controls.values()})
    verify_files(source_files)
    code_files = {
        str(Path(m.__file__).resolve()): file_hash(m.__file__)
        for m in list(sys.modules.values())
        if getattr(m, "__file__", None)
        and Path(m.__file__).suffix == ".py"
        and Path(m.__file__).resolve().is_relative_to(PROJECT)
        and not Path(m.__file__).resolve().is_relative_to(PROJECT / ".venv")
    }
    for name in ("scripts/fund_002112_peer_experiment.py", "tests/test_fund_002112_peer_experiment.py"):
        code_files[str(PROJECT / name)] = file_hash(PROJECT / name)
    copies = {
        "inputs.json": PACKAGE / "inputs.json",
        "folds.json": PACKAGE / "folds.json",
        "execution-plan.json": PACKAGE / "execution-plan.json",
        "original-inputs.json": old / "inputs.json",
        "original-folds.json": old / "folds.json",
    }
    copies.update({f"controls/{k}.json": Path(v["path"]) for k, v in controls.items()})
    frozen = {}
    for name, src in copies.items():
        dest = path / "frozen" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if file_hash(dest) != file_hash(src):
                raise ValueError("PARTIAL_FREEZE_CHANGED")
        else:
            with dest.open("xb") as stream:
                stream.write(src.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
        frozen[str(dest.relative_to(path))] = file_hash(dest)
    protocol = {
        "schema": "002112-recovered-peer-independent-experiment-v1",
        "created_at": now(),
        "hypothesis": "补回参考资料，能否改善原 L20？",
        "authorization": "用户 2026-09-28 明确授权最多 12 次真实拟合，含独立复现，失败计入且不退款。",
        "fit_budget": BUDGET,
        "old_cumulative_actual_fits": 52,
        "old_package_budget_unchanged": 0,
        "slots": list(SLOTS),
        "historical_gate_before_full": True,
        "technical_failure_stops": True,
        "frozen": frozen,
        "sources": source_files,
        "code": code_files,
        "environment": actual,
        "recipe": plan["recipe"],
        "bootstrap": BOOTSTRAP,
        "historical_gates": plan["frozen_plan"]["historical_gates"],
        "full_gates": plan["frozen_plan"]["development_gates"],
        "evidence": "2023/2024 开发验证；公开时间按最终准入；数据库仅沿用 2026-09-27 固定快照。",
    }
    exclusive(path / "protocol.json", protocol)
    exclusive(path / "protocol-anchor.json", {"sha256": file_hash(path / "protocol.json")})
    load_frozen(path)


def validate_scope(rows, *, training):
    """禁止封存和回算年份进入拟合/评分，且所有输入必须有限、完整。"""
    identities = [(r["fund_code"], r["target"]) for r in rows]
    if len(set(identities)) != len(rows) or not rows:
        raise ValueError("DUPLICATE_OR_EMPTY_ROWS")
    for row in rows:
        upper = "2023-12-31" if training else "2024-12-31"
        if row["target"] > upper or row["actual_direction"] not in CLASSES:
            raise ValueError("FORBIDDEN_LABEL_SCOPE")
    model.matrix(rows, 20)


def load_frozen(path=RUN):
    verify_files({"protocol.json": read(path / "protocol-anchor.json")["sha256"]}, path)
    p = read(path / "protocol.json")
    if p["fit_budget"] != BUDGET or p["slots"] != list(SLOTS):
        raise ValueError("BUDGET_OR_SLOTS_CHANGED")
    verify_files(p["frozen"], path)
    verify_files(p["code"])
    verify_files(p["sources"])
    plan = read(path / "frozen/execution-plan.json")
    environment(plan)
    values = {
        name: read(path / "frozen" / (name + ".json"))
        for name in ("inputs", "folds", "original-inputs", "original-folds")
    }
    for name, count in (("inputs", 5137), ("original-inputs", 4419)):
        if len(values[name]["train"]) != count:
            raise ValueError("POOL_COUNT_CHANGED")
        validate_scope(values[name]["train"], training=True)
        validate_scope(values[name]["development"], training=False)
    index = {(r["fund_code"], r["target"]): r for r in values["inputs"]["train"]}
    if any(index.get((r["fund_code"], r["target"])) != r for r in values["original-inputs"]["train"]):
        raise ValueError("ORIGINAL_ROW_CHANGED")
    for fold in values["folds"]:
        train = [index[tuple(key)] for key in fold["train_ids"]]
        if not fold["gate"]["passed"] or digest(train) != fold["train_sha256"]:
            raise ValueError("FOLD_TRAIN_CHANGED")
        if digest(fold["exam"]) != fold["exam_sha256"]:
            raise ValueError("FOLD_EXAM_CHANGED")
        validate_scope(fold["exam"], training=False)
        cutoff = datetime.fromisoformat(fold["start"] + "T00:00:00+08:00")
        if any(datetime.fromisoformat(r["mature_at"]) >= cutoff for r in train):
            raise ValueError("IMMATURE_TRAIN_ROW")
        if [(r["target"], r["fund_code"]) for r in train] != sorted((r["target"], r["fund_code"]) for r in train):
            raise ValueError("ROW_ORDER_CHANGED")
        if fold["name"] in HISTORICAL:
            for variant in ("N7", "L20"):
                control = read(path / f"frozen/controls/{fold['name']}-{variant}-main.json")
                replay = read(path / f"frozen/controls/{fold['name']}-{variant}-replay.json")
                if control != replay:
                    raise ValueError("OLD_CONTROL_REPLAY_CHANGED")
                expected = [(r["target"], r["actual_direction"], digest(r)) for r in fold["exam"]]
                if [(r["target"], r["actual_direction"], r["input_hash"]) for r in control["exam"]] != expected:
                    raise ValueError("OLD_CONTROL_EXAM_IDENTITY_CHANGED")
    return values


def slot_data(values, slot):
    """RECOVERED 使用新池；ORIGINAL 从旧冻结名单、权重及 4,419 行读取。"""
    if slot not in SLOTS:
        raise ValueError("UNKNOWN_SLOT")
    stage, variant, _ = slot.split("-")
    recovered = variant == "L20_RECOVERED"
    pool = values["inputs" if recovered else "original-inputs"]["train"]
    folds = values["folds" if recovered else "original-folds"]
    fold = next(f for f in folds if f["name"] == stage)
    index = {(r["fund_code"], r["target"]): r for r in pool}
    train = [index[tuple(key)] for key in fold["train_ids"]]
    exam = next(f["exam"] for f in values["folds"] if f["name"] == stage)
    expected = fold["train_sha256" if recovered else "train_hash"]
    if digest(train) != expected:
        raise ValueError("SLOT_TRAIN_CHANGED")
    return train, exam, fold["weights"], "N7" if variant == "N7_ORIGINAL" else "L20"


def ledger(path):
    """只读顺序事件，链式摘要使删改中间消费记录无法静默续跑。"""
    files = list((path / "attempts").glob("*.json"))
    used = {f.stem for f in files}
    if used != set(SLOTS[: len(files)]) or len(files) > BUDGET:
        raise ValueError("LEDGER_SEQUENCE_OR_BUDGET_INVALID")
    previous = file_hash(path / "protocol.json")
    for i, slot in enumerate(SLOTS[: len(files)]):
        f = path / "attempts" / (slot + ".json")
        event = read(f)
        if event["slot"] != slot or event["ordinal"] != i + 1 or event["previous_sha256"] != previous:
            raise ValueError("LEDGER_CHAIN_CHANGED")
        if event["protocol_sha256"] != file_hash(path / "protocol.json"):
            raise ValueError("LEDGER_PROTOCOL_CHANGED")
        previous = file_hash(f)
    # 即使最后一条消费事件被误删，也不能把已有执行痕迹当成未开始。
    for folder in ("started", "classifier-started", "classifier-completed", "fit-complete", "checkpoints", "failures"):
        if any(f.stem not in used for f in (path / folder).glob("*.json")):
            raise ValueError("ORPHAN_EXECUTION_EVIDENCE")
    return len(files)


def checkpoint(path, slot):
    item = read(path / "checkpoints" / (slot + ".json"))
    if item["slot"] != slot or item["attempt_sha256"] != file_hash(path / "attempts" / (slot + ".json")):
        raise ValueError("CHECKPOINT_ATTEMPT_CHANGED")
    verify_files(item["files"], path)
    return item


def reserve(path, slot, token):
    """先占不可退款预算，再交付一次性许可；已消费槽位绝不再发许可。"""
    count = ledger(path)
    if count >= BUDGET:
        raise ValueError("BUDGET_EXHAUSTED")
    if slot != SLOTS[count]:
        raise ValueError("SLOT_ALREADY_CONSUMED_OR_OUT_OF_ORDER")
    for prior in SLOTS[:count]:
        checkpoint(path, prior)
    if count >= 6 and not read(path / "historical-decision.json")["passed"]:
        raise ValueError("HISTORICAL_GATE_REQUIRED")
    previous = path / "protocol.json" if not count else path / "attempts" / (SLOTS[count - 1] + ".json")
    exclusive(
        path / "attempts" / (slot + ".json"),
        {
            "slot": slot,
            "ordinal": count + 1,
            "reserved_at": now(),
            "budget_consumed": True,
            "previous_sha256": file_hash(previous),
            "protocol_sha256": file_hash(path / "protocol.json"),
            "permit_sha256": hashlib.sha256(token.encode()).hexdigest(),
        },
    )


def child(path, command, slot, token=None):
    """每次拟合和载入均为全新解释器；子进程不得继承内存模型。"""
    if path.resolve() != RUN.resolve():
        raise ValueError("WORKER_DIRECTORY_NOT_AUTHORIZED")
    env = os.environ.copy()
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["FUND_PEER_FIT_PERMIT"] = token or ""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-m", "scripts.fund_002112_peer_experiment", command, "--slot", slot],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        raise RuntimeError("CHILD_FAILED:" + command + ":" + result.stderr[-6000:])


def fit_worker(path, slot):
    values = load_frozen(path)
    count = ledger(path)
    if not count or slot != SLOTS[count - 1]:
        raise ValueError("FIT_SLOT_NOT_RESERVED")
    event = read(path / "attempts" / (slot + ".json"))
    token = os.environ.get("FUND_PEER_FIT_PERMIT", "")
    if not token or hashlib.sha256(token.encode()).hexdigest() != event["permit_sha256"]:
        raise ValueError("FIT_PERMIT_INVALID")
    if SLOTS.index(slot) >= 6 and not read(path / "historical-decision.json")["passed"]:
        raise ValueError("HISTORICAL_GATE_REQUIRED")
    train, exam, weights, variant = slot_data(values, slot)
    exclusive(path / "started" / (slot + ".json"), {"slot": slot, "pid": os.getpid(), "at": now()})
    original_fit = model.LogisticRegression.fit

    def counted_fit(estimator, *args, **kwargs):
        exclusive(path / "classifier-started" / (slot + ".json"), {"slot": slot, "pid": os.getpid(), "at": now()})
        answer = original_fit(estimator, *args, **kwargs)
        exclusive(path / "classifier-completed" / (slot + ".json"), {"slot": slot, "at": now()})
        return answer

    model.LogisticRegression.fit = counted_fit
    try:
        fitted, predictions = model.fit(train, exam, weights, variant)
    finally:
        model.LogisticRegression.fit = original_fit
    manifest = state.save_model(fitted, path / "models", slot)
    exclusive(path / "predictions" / (slot + ".json"), predictions)
    names = [f"models/{slot}.joblib", f"models/{slot}.manifest.json", f"predictions/{slot}.json"]
    names += [f"{folder}/{slot}.json" for folder in ("started", "classifier-started", "classifier-completed")]
    exclusive(
        path / "fit-complete" / (slot + ".json"),
        {
            "slot": slot,
            "manifest_sha256": manifest,
            "state_sha256": state.model_state_digest(fitted),
            "attempt_sha256": file_hash(path / "attempts" / (slot + ".json")),
            "files": {n: file_hash(path / n) for n in names},
        },
    )


def restore_worker(path, slot):
    if slot not in SLOTS:
        raise ValueError("UNKNOWN_SLOT")
    values = load_frozen(path)
    receipt = read(path / "fit-complete" / (slot + ".json"))
    verify_files(receipt["files"], path)
    if receipt["attempt_sha256"] != file_hash(path / "attempts" / (slot + ".json")):
        raise ValueError("FIT_RECEIPT_ATTEMPT_CHANGED")
    train, exam, weights, variant = slot_data(values, slot)
    fitted = state.restore_model(path / "models", slot, receipt["manifest_sha256"])
    if (
        fitted["training_hash"] != digest(train)
        or fitted["weights_hash"] != digest(weights)
        or fitted["variant"] != variant
    ):
        raise ValueError("RESTORE_TRAIN_IDENTITY_CHANGED")
    checks = state.verify_all_predictions(fitted, train, exam, read(path / "predictions" / (slot + ".json")))
    result = {
        "slot": slot,
        "checks": checks,
        "state_sha256": state.model_state_digest(fitted),
        "separate_process": True,
        "fits": 0,
    }
    dest = path / "restored" / (slot + ".json")
    if dest.exists():
        recorded = read(dest)
        recorded.pop("first_restore_process")
        if recorded != result:
            raise ValueError("RESTORE_RESULT_CHANGED")
    else:
        result["first_restore_process"] = {"pid": os.getpid(), "at": now()}
        exclusive(dest, result)


def ensure_slot(path, slot):
    """恢复完整拟合或有效检查点；对部分拟合绝不再次调用 fit。"""
    if (path / "checkpoints" / (slot + ".json")).exists():
        return checkpoint(path, slot)
    if (path / "failures" / (slot + ".json")).exists():
        raise ValueError("FAILED_SLOT_CANNOT_RETRY:" + slot)
    if not (path / "attempts" / (slot + ".json")).exists():
        token = secrets.token_hex(32)
        reserve(path, slot, token)
        child(path, "_fit", slot, token)
    elif not (path / "fit-complete" / (slot + ".json")).exists():
        raise ValueError("INCOMPLETE_CONSUMED_SLOT_NO_REFIT:" + slot)
    child(path, "_restore", slot)
    receipt = read(path / "fit-complete" / (slot + ".json"))
    files = dict(receipt["files"])
    for folder in ("fit-complete", "restored"):
        name = f"{folder}/{slot}.json"
        files[name] = file_hash(path / name)
    item = {"slot": slot, "attempt_sha256": file_hash(path / "attempts" / (slot + ".json")), "files": files}
    exclusive(path / "checkpoints" / (slot + ".json"), item)
    return item


def replay_check(path, main):
    replay = main.removesuffix("-main") + "-replay"
    for slot in (main, replay):
        checkpoint(path, slot)
    a, b = [read(path / "fit-complete" / (slot + ".json")) for slot in (main, replay)]
    if a["state_sha256"] != b["state_sha256"]:
        raise ValueError("INDEPENDENT_MODEL_STATE_MISMATCH")
    predictions = [read(path / "predictions" / (slot + ".json")) for slot in (main, replay)]
    deltas = {}
    for part in ("train", "exam"):
        for pred in predictions:
            scores = np.asarray([r["scores"] for r in pred[part]])
            if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
                raise ValueError("NONFINITE_REPLAY_SCORES")
        deltas[part] = model.compare_predictions(predictions[0][part], predictions[1][part])
    return {"main": main, "replay": replay, "state_equal": True, "max_score_delta": deltas, "passed": True}


def compare(by_variant, *, historical):
    """同日成对比较；FULL 仅去掉季度门槛，其余要求完全相同。"""
    names = ("L20_RECOVERED", "L20_ORIGINAL", "N7_ORIGINAL")
    if set(by_variant) != set(names):
        raise ValueError("COMPARISON_VARIANTS_INVALID")
    rows = by_variant[names[0]]
    identities = [(r["fund_code"], r["target"], r["actual_direction"], r["input_hash"]) for r in rows]
    if any(
        [(r["fund_code"], r["target"], r["actual_direction"], r["input_hash"]) for r in v] != identities
        for v in by_variant.values()
    ):
        raise ValueError("COMPARISON_IDENTITY_CHANGED")
    metrics = {k: model.metrics(v) for k, v in by_variant.items()}
    quarters = {
        q: {k: model.metrics([r for r in v if quarter(r["target"]) == q]) for k, v in by_variant.items()}
        for q in sorted({quarter(r["target"]) for r in rows})
    }
    constants = {c: sum(r["actual_direction"] == c for r in rows) for c in CLASSES}
    indices = block_indices([r["target"] for r in rows], BOOTSTRAP)
    pairs = {}
    for name in names[1:]:
        daily = []
        for candidate, control in zip(rows, by_variant[name], strict=True):
            delta = int(candidate["direction"] == candidate["actual_direction"]) - int(
                control["direction"] == control["actual_direction"]
            )
            daily.append(
                {
                    "target": candidate["target"],
                    "actual": candidate["actual_direction"],
                    "candidate": candidate["direction"],
                    "control": control["direction"],
                    "delta": delta,
                }
            )
        d = np.asarray([r["delta"] for r in daily])

        def counts(selected):
            values = [r["delta"] for r in selected]
            return {"gained": values.count(1), "lost": values.count(-1), "net": sum(values), "days": len(values)}

        pairs[name] = {
            **counts(daily),
            "daily": daily,
            "classes": {c: counts([r for r in daily if r["actual"] == c]) for c in CLASSES},
            "quarters": {q: counts([r for r in daily if quarter(r["target"]) == q]) for q in quarters},
            "diagnostic_interval_pp": (np.quantile(d[indices].mean(axis=1), [0.025, 0.975]) * 100).tolist(),
            "diagnostic_only": True,
        }
    candidate = metrics[names[0]]
    checks = {
        "total_strictly_above_controls": all(candidate["correct"] > metrics[n]["correct"] for n in names[1:]),
        "total_strictly_above_constants": all(candidate["correct"] > n for n in constants.values()),
    }
    checks.update(
        {
            "class_not_worse_" + c: all(
                candidate["class_correct"][c] >= metrics[n]["class_correct"][c] for n in names[1:]
            )
            for c in CLASSES
        }
    )
    if historical:
        checks["at_least_two_quarters_not_worse_L20"] = (
            sum(q[names[0]]["correct"] >= q[names[1]]["correct"] for q in quarters.values()) >= 2
        )
    return {
        "models": metrics,
        "quarters": quarters,
        "constants": constants,
        "pairs": pairs,
        "checks": checks,
        "numerical_passed": all(checks.values()),
        "historical": historical,
        "evidence_type": "DEVELOPMENT_VALIDATION_NOT_BLIND_TEST",
    }


def stage_result(path, historical):
    quarters = HISTORICAL if historical else ("FULL",)
    predictions = {n: [] for n in ("L20_RECOVERED", "L20_ORIGINAL", "N7_ORIGINAL")}
    engineering = []
    for q in quarters:
        for name in predictions:
            if historical and name != "L20_RECOVERED":
                src = path / f"frozen/controls/{q}-{name.split('_')[0]}-main.json"
            else:
                main = f"{q}-{name}-main"
                engineering.append(replay_check(path, main))
                src = path / "predictions" / (main + ".json")
            predictions[name].extend(read(src)["exam"])
    result = compare(predictions, historical=historical)
    result["engineering"] = engineering
    result["engineering_passed"] = all(x["passed"] for x in engineering)
    result["passed"] = result["numerical_passed"] and result["engineering_passed"]
    result["failed_checks"] = [k for k, v in result["checks"].items() if not v]
    return result


def counts(path):
    used = ledger(path)
    actual = len(list((path / "classifier-started").glob("*.json")))
    return {
        "budget": BUDGET,
        "consumed_slots": used,
        "remaining_slots": BUDGET - used,
        "actual_new_fits": actual,
        "cumulative_actual_fits": 52 + actual,
        "completed_classifier_calls": len(list((path / "classifier-completed").glob("*.json"))),
    }


def finish(path, status, **extra):
    decision = {"status": status, **counts(path), "adopted": False, **extra}
    once(path / "decision.json", decision)
    once(path / "decision-anchor.json", {"sha256": file_hash(path / "decision.json")})
    return decision


def run(path=RUN, *, interrupt_after=None):
    with lock(path):
        load_frozen(path)
        ledger(path)
        if (path / "decision.json").exists():
            verify_files({"decision.json": read(path / "decision-anchor.json")["sha256"]}, path)
            for slot in SLOTS[: ledger(path)]:
                if (path / "checkpoints" / (slot + ".json")).exists():
                    checkpoint(path, slot)
            return read(path / "decision.json")
        current = None
        try:
            for historical, slots in ((True, SLOTS[:6]), (False, SLOTS[6:])):
                if not historical and not read(path / "historical-decision.json")["passed"]:
                    return finish(path, "STOP_HISTORICAL_GATE_FAILED", full_executed=False)
                for current in slots:
                    ensure_slot(path, current)
                    if current.endswith("-replay"):
                        replay_check(path, current.removesuffix("-replay") + "-main")
                    print(json.dumps({"completed": current, **counts(path)}), flush=True)
                    if interrupt_after is not None and ledger(path) == interrupt_after:
                        return {"status": "CHECKPOINT_PAUSE", **counts(path)}
                result = stage_result(path, historical)
                once(path / ("historical-decision.json" if historical else "full-decision.json"), result)
                if not result["passed"]:
                    return finish(
                        path,
                        "STOP_HISTORICAL_GATE_FAILED" if historical else "STOP_FULL_GATE_FAILED",
                        failed_checks=result["failed_checks"],
                        full_executed=not historical,
                    )
            return finish(path, "PASSED_ALL_GATES_PENDING_ADOPTION", full_executed=True)
        except Exception as exc:
            failure = {"slot": current, "reason": str(exc), "traceback": traceback.format_exc()}
            if current is not None and (path / "attempts" / (current + ".json")).exists():
                once(path / "failures" / (current + ".json"), failure)
            return finish(
                path,
                "BUDGET_EXHAUSTED" if str(exc) == "BUDGET_EXHAUSTED" else "STOP_ENGINEERING_FAILURE",
                failure=failure,
                full_executed=ledger(path) > 6,
            )


def verify(path=RUN):
    """零拟合独立进程载入全部已完成模型，重算阶段决定和台账。"""
    with lock(path):
        load_frozen(path)
        used = ledger(path)
        if (path / "decision.json").exists():
            verify_files({"decision.json": read(path / "decision-anchor.json")["sha256"]}, path)
        for slot in SLOTS[:used]:
            if (path / "checkpoints" / (slot + ".json")).exists():
                checkpoint(path, slot)
                child(path, "_restore", slot)
        for historical, name in ((True, "historical-decision.json"), (False, "full-decision.json")):
            if (path / name).exists() and stage_result(path, historical) != read(path / name):
                raise ValueError("STAGE_DECISION_CHANGED")
        return {"passed": True, **counts(path)}
