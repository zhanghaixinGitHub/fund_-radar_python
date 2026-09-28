"""002112 独立修复协议：继承原 5 次，最多新增 19 次，禁止恢复旧失败运行。

仅调用原固定纯数值配方和 v2 保存校验。所有写入限于新目录；来源、标签和
日期仍来自原截至 2024 年的冻结包，不引入数据库、联网、登记或采用入口。
"""

import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from app.services import fund_002112_model_state_v2 as state
from app.services import fund_002112_round3_data as data
from app.services import fund_002112_round3_model as model
from app.services import fund_002112_round3_run as original

ROOT = data.ROOT / "round3-repair-runs"
OLD = data.RUN_ROOT / "002112-r3-0ec4bfcbef4d93b3c49470f7"
REPAIR = data.ROOT / "serialization-repair-v1"
INHERITED = tuple(original.SLOTS[:5])
NEW_SLOTS = tuple(slot for slot in original.SLOTS if slot not in INHERITED)
HISTORICAL_SLOTS = tuple(slot for slot in NEW_SLOTS if not slot.startswith("FULL-"))
MAX_NEW = 19
PROTOCOL = "FUND_002112_ROUND3_SERIALIZATION_REPAIR_V2"


@contextmanager
def lock(root=None):
    """操作系统排他锁；父进程全程持有，退出或真实中断后由操作系统释放。"""
    root = ROOT if root is None else Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".execution.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def directory(run_id):
    if not re.fullmatch(r"002112-r3r-[a-f0-9]{24}", run_id):
        raise ValueError("INVALID_REPAIR_RUN_ID")
    path = (ROOT / run_id).resolve()
    if path.parent != ROOT.resolve():
        raise ValueError("PATH_OUTSIDE_REPAIR_ROOT")
    return path


def exclusive(path, value):
    """槽位占用/开始事件永不幂等重写；即使内容相同也拒绝第二次进入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump({"hash": data.digest(value), "payload": value}, output, sort_keys=True, ensure_ascii=False)
        output.flush()
        os.fsync(output.fileno())


def verify_files(files, root=None):
    for name, expected in files.items():
        path = Path(name) if root is None else (root / name).resolve()
        if root is not None and not path.is_relative_to(root.resolve()):
            raise ValueError("EVIDENCE_PATH_ESCAPE")
        if not path.is_file() or data.file_hash(path) != expected:
            raise ValueError("EVIDENCE_CHANGED:" + path.name)


def code_manifest():
    paths = [
        data.PROJECT / "app/services/fund_002112_repaired_experiment.py",
        data.PROJECT / "app/services/fund_002112_model_state_v2.py",
        data.PROJECT / "scripts/fund_002112_repaired_experiment.py",
        data.PROJECT / "scripts/fund_002112_repaired_audit.py",
        data.PROJECT / "scripts/fund_002112_round3_audit.py",
        data.PROJECT / "tests/test_fund_002112_repaired_experiment.py",
        data.PROJECT / "tests/test_fund_002112_model_state_v2.py",
        data.PROJECT / "pyproject.toml",
    ]
    return {
        **original.code_manifest(),
        **{str(path.relative_to(data.PROJECT)): data.file_hash(path) for path in paths},
    }


def protection():
    files = {}
    for name in ("protection-before.json", "protection-extra.json"):
        files.update(data.read(ROOT / name)["files"])
    verify_files(files)
    return files


def preflight_inputs():
    """要求重建的四份载荷摘要与原包相同，并再次核验来源期限和当前用途记录。"""
    values = original.load_frozen(OLD)
    rebuilt = data.read(ROOT / "data-preflight.json")
    if not rebuilt["passed"] or rebuilt["rebuilt_hashes"] != {key: data.digest(value) for key, value in values.items()}:
        raise ValueError("REBUILT_DATA_GATE_FAILED")
    current = data.read(ROOT / "source-authorization.json")["metadata"]
    previous = values["sources"]["authorization"]["metadata"]
    if any(current[key] != value for key, value in previous.items()):
        raise ValueError("SOURCE_AUTHORIZATION_CHANGED")
    if (
        not current["enabled"]
        or current["retention_days"] <= 0
        or not current["license_scope"]
        or not {"fund_nav", "daily", "index_daily"}.issubset(current["authorized_api_names"])
    ):
        raise ValueError("SOURCE_AUTHORIZATION_INVALID")
    if not values["eligibility"]["passed"]:
        raise ValueError("DATA_GATE_FAILED")
    for stage in (*data.FOLDS, "FULL"):
        original.fold_data(values, stage)
    return values


def preflight():
    """零拟合建立唯一协议及五个模型的继承证据；原 T20 仍待新复现确认。"""
    with lock():
        pointer = ROOT / "run-binding.json"
        if pointer.exists():
            path = directory(data.read(pointer)["run_id"])
            load_frozen(path)
            return path
        if list(ROOT.glob("002112-r3r-*")):
            raise ValueError("UNBOUND_REPAIR_DIRECTORY_REQUIRES_STOP")
        protected = protection()
        values = preflight_inputs()
        proposal = data.read(REPAIR / "new-plan.json")
        authorization = data.read(ROOT / "authorization.json")
        if (
            authorization["max_new_real_fits"] != MAX_NEW
            or authorization["nonrefundable_inherited_fits"] != 5
            or proposal["budget"]["new_total_maximum"] != MAX_NEW
        ):
            raise ValueError("AUTHORIZATION_BUDGET_CHANGED")
        expected_attempts = {slot + ".json" for slot in INHERITED}
        if {p.name for p in (OLD / "attempts").glob("*.json")} != expected_attempts:
            raise ValueError("OLD_ATTEMPTS_CHANGED")
        decision = data.read(OLD / "historical-decision.json")
        if decision["status"] != "STOPPED_TECHNICAL_FAILURE":
            raise ValueError("OLD_FAILURE_MUST_REMAIN")
        for slot in INHERITED[:4]:
            original.checkpoint(OLD, slot)
        if (OLD / "checkpoints" / (INHERITED[-1] + ".json")).exists():
            raise ValueError("OLD_T20_CHECKPOINT_MUST_NOT_BE_BACKFILLED")
        proof = data.read(REPAIR / "final-zero-fit-verification.json")
        expected = {row["slot"]: row for row in proof["results"]}
        bridge = {}
        training, exams, weights = original.fold_data(values, "2023Q2")
        for slot in INHERITED:
            manifest_hash = expected[slot]["manifest_sha256"]
            fitted = state.restore_model(REPAIR / "verified-copies", slot, manifest_hash)
            predictions = data.read(OLD / "predictions" / (slot + ".json"))
            if fitted["training_hash"] != data.digest(training) or fitted["weights_hash"] != data.digest(weights):
                raise ValueError("INHERITED_TRAINING_IDENTITY_CHANGED")
            checks = state.verify_all_predictions(fitted, training, exams, predictions)
            state_hash = state.model_state_digest(fitted)
            if state_hash != expected[slot]["state_sha256"]:
                raise ValueError("INHERITED_NUMERIC_STATE_CHANGED")
            old_paths = [
                OLD / folder / (slot + suffix)
                for folder, suffix in (
                    ("attempts", ".json"),
                    ("started", ".json"),
                    ("models", ".joblib"),
                    ("model-manifests", ".json"),
                    ("predictions", ".json"),
                )
            ]
            if slot in INHERITED[:4]:
                old_paths.append(OLD / "checkpoints" / (slot + ".json"))
            bridge[slot] = {
                "slot": slot,
                "state_sha256": state_hash,
                "manifest_sha256": manifest_hash,
                "original_files": {str(p): data.file_hash(p) for p in old_paths},
                "all_rows": checks,
                "state": "PENDING_REQUIRED_REPLAY" if slot == INHERITED[-1] else "INHERITED_VERIFIED",
            }
        for variant in ("N7", "L20"):
            if bridge[f"2023Q2-{variant}-main"]["state_sha256"] != bridge[f"2023Q2-{variant}-replay"]["state_sha256"]:
                raise ValueError("INHERITED_REPLICA_CHANGED")
        # 将现有保护锚点和新准入证据一起冻结；后续报告/账本不进入此只读集合。
        anchors = {
            str(p): data.file_hash(p)
            for p in (
                ROOT / "authorization.json",
                ROOT / "source-authorization.json",
                ROOT / "data-preflight.json",
                ROOT / "engineering-acceptance.json",
                REPAIR / "new-plan.json",
                OLD / "protocol.json",
                OLD / "historical-decision.json",
            )
        }
        tests = data.read(ROOT / "engineering-acceptance.json")
        if not tests["passed"] or tests["new_real_fits"] != 0 or tests["code"] != code_manifest():
            raise ValueError("ENGINEERING_GATE_FAILED")
        spec = {
            "protocol": PROTOCOL,
            "schema": state.SCHEMA,
            "created_at": data.now(),
            "old_run": OLD.name,
            "old_fits": 34,
            "inherited_fits": 5,
            "maximum_new_fits": MAX_NEW,
            "maximum_combined_round3_fits": 24,
            "slots": list(NEW_SLOTS),
            "bridge": bridge,
            "code": code_manifest(),
            "environment": original.environment(),
            "anchors": anchors,
            "protected_files": protected,
            "frozen_data": {k: data.digest(v) for k, v in values.items()},
            "old_protocol_hash": data.digest(data.read(OLD / "protocol.json")),
            "selection": data.read(OLD / "protocol.json")["selection"],
            "no_adoption": True,
            "no_future_observation": True,
        }
        path = directory("002112-r3r-" + data.digest(spec)[:24])
        for key, value in values.items():
            data.write_once(path / (key + ".json"), value)
        for slot, item in bridge.items():
            files = {}
            for suffix in (".joblib", ".manifest.json"):
                source = REPAIR / "verified-copies" / (slot + suffix)
                dest = path / "models" / source.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as output:
                    output.write(source.read_bytes())
                    output.flush()
                    os.fsync(output.fileno())
                files[str(dest.relative_to(path))] = data.file_hash(dest)
            dest = path / "predictions" / (slot + ".json")
            data.write_once(dest, data.read(OLD / "predictions" / (slot + ".json")))
            files[str(dest.relative_to(path))] = data.file_hash(dest)
            item["files"] = files
        # 运行编号取初始化身份摘要；指针另外绑定含副本文件哈希的最终协议载荷。
        data.write_once(path / "protocol.json", spec)
        data.write_once(pointer, {"run_id": path.name, "protocol_hash": data.digest(spec)})
        for slot in INHERITED:
            child(path, "_restore", slot)
        data.write_once(
            path / "bridge-complete.json",
            {
                "at": data.now(),
                "inherited_fits": 5,
                "new_fits": 0,
                "five_independent_restores": True,
                "T20_requires_new_replay": True,
            },
        )
        return path


def load_frozen(path, *, check_sources=True):
    pointer = data.read(ROOT / "run-binding.json")
    spec = data.read(path / "protocol.json")
    if pointer != {"run_id": path.name, "protocol_hash": data.digest(spec)}:
        raise ValueError("RUN_BINDING_CHANGED")
    if (
        spec["protocol"] != PROTOCOL
        or spec["schema"] != state.SCHEMA
        or spec["slots"] != list(NEW_SLOTS)
        or spec["inherited_fits"] != 5
        or spec["maximum_new_fits"] != MAX_NEW
        or spec["code"] != code_manifest()
        or spec["environment"] != original.environment()
    ):
        raise ValueError("FROZEN_REPAIR_CODE_ENVIRONMENT_OR_BUDGET_CHANGED")
    verify_files(spec["anchors"])
    verify_files(spec["protected_files"])
    original_spec = data.read(OLD / "protocol.json")
    if original_spec["code"] != original.code_manifest() or data.digest(original_spec) != spec["old_protocol_hash"]:
        raise ValueError("ORIGINAL_PROTOCOL_CHANGED")
    values = {key: data.read(path / (key + ".json")) for key in spec["frozen_data"]}
    if {k: data.digest(v) for k, v in values.items()} != spec["frozen_data"]:
        raise ValueError("FROZEN_REPAIR_DATA_CHANGED")
    if check_sources:
        original.verify_sources(values["sources"])
    for item in spec["bridge"].values():
        verify_files(item["original_files"])
        verify_files(item["files"], path)
    return values


def attempts(path):
    return sorted((path / "attempts").glob("*.json"))


def validate_ledger(path):
    """统一继承预算和顺序；其他运行目录出现任何尝试都拒绝继续。"""
    occupied = {p.stem for p in attempts(path)}
    if len(occupied) > MAX_NEW or 5 + len(occupied) > 24:
        raise ValueError("FIT_BUDGET_EXHAUSTED")
    if occupied != set(NEW_SLOTS[: len(occupied)]):
        raise ValueError("FIT_SLOT_ORDER_OR_INHERITANCE_CHANGED")
    for other in ROOT.glob("002112-r3r-*/attempts/*.json"):
        if other.parent.parent.resolve() != path.resolve():
            raise ValueError("OTHER_RUN_ALREADY_CONSUMED_BUDGET")
    return len(occupied)


def checkpoint(path, slot):
    if slot in INHERITED:
        item = data.read(path / "protocol.json")["bridge"][slot]
        verify_files(item["files"], path)
        verify_files(item["original_files"])
        return item
    value = data.read(path / "checkpoints" / (slot + ".json"))
    if value["slot"] != slot or value["attempt_hash"] != data.file_hash(path / "attempts" / (slot + ".json")):
        raise ValueError("CHECKPOINT_ATTEMPT_CHANGED")
    verify_files(value["files"], path)
    return value


def ready_for_slot(path, slot):
    """阶段门槛独立于命令入口，FULL 或 Q3 不能由直接调用 worker 绕过。"""
    if slot not in NEW_SLOTS:
        raise ValueError("SLOT_ALREADY_INHERITED_OR_UNKNOWN")
    stage = slot.split("-")[0]
    if stage != "2023Q2":
        previous = {"2023Q3": "2023Q2", "2023Q4": "2023Q3", "FULL": "2023Q4"}[stage]
        if not data.read(path / "reproduction" / (previous + ".json"))["passed"]:
            raise ValueError("PREVIOUS_STAGE_REPLAY_NOT_PASSED")
    if stage == "FULL":
        decision = data.read(path / "historical-decision.json")
        comparison = data.read(path / "historical-comparison.json")
        if (
            not decision["passed"]
            or not comparison["numerical_passed"]
            or decision["comparison_hash"] != data.digest(comparison)
        ):
            raise ValueError("HISTORICAL_GATE_FAILED")
    if not (path / "bridge-complete.json").exists():
        raise ValueError("BRIDGE_NOT_COMPLETE")


def reserve(path, slot, token):
    ready_for_slot(path, slot)
    count = validate_ledger(path)
    if count >= MAX_NEW:
        raise ValueError("FIT_BUDGET_EXHAUSTED")
    if slot != NEW_SLOTS[count]:
        raise ValueError("FIT_SLOT_NOT_NEXT")
    for earlier in NEW_SLOTS[:count]:
        checkpoint(path, earlier)
    exclusive(
        path / "attempts" / (slot + ".json"),
        {
            "slot": slot,
            "at": data.now(),
            "permit_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "protocol_hash": data.digest(data.read(path / "protocol.json")),
            "status": "PERMANENTLY_RESERVED",
        },
    )


def child(path, command, slot, token=None):
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-m",
            "scripts.fund_002112_repaired_experiment",
            command,
            "--run-id",
            path.name,
            "--slot",
            slot,
        ],
        cwd=data.PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "FUND_002112_FIT_PERMIT": token or "",
        },
    )
    if completed.returncode:
        raise ValueError("CHILD_FAILED:" + command + ":" + completed.stderr[-1800:])


def fit_worker(path, slot):
    values = load_frozen(path)
    ready_for_slot(path, slot)
    count = validate_ledger(path)
    if not count or slot != NEW_SLOTS[count - 1]:
        raise ValueError("UNRESERVED_OR_OUT_OF_ORDER_WORKER")
    for earlier in NEW_SLOTS[: count - 1]:
        checkpoint(path, earlier)
    attempt = data.read(path / "attempts" / (slot + ".json"))
    token = os.environ.get("FUND_002112_FIT_PERMIT", "")
    if (
        not token
        or hashlib.sha256(token.encode()).hexdigest() != attempt["permit_sha256"]
        or attempt["protocol_hash"] != data.digest(data.read(path / "protocol.json"))
    ):
        raise ValueError("FIT_WORKER_PERMIT_MISSING")
    training, exams, weights = original.fold_data(values, slot.split("-")[0])
    if not values["eligibility"]["passed"]:
        raise ValueError("DATA_GATE_FAILED")
    exclusive(path / "started" / (slot + ".json"), {"slot": slot, "at": data.now()})
    variant = slot.split("-")[1]
    cls = model.HistGradientBoostingClassifier if variant == "T20" else model.LogisticRegression
    underlying = cls.fit

    def counted_fit(estimator, *args, **kwargs):
        # 直接记录真实分类器调用边界，不把预处理失败冒充已完成分类器训练。
        exclusive(path / "classifier-started" / (slot + ".json"), {"slot": slot, "at": data.now()})
        answer = underlying(estimator, *args, **kwargs)
        exclusive(path / "classifier-completed" / (slot + ".json"), {"slot": slot, "at": data.now()})
        return answer

    cls.fit = counted_fit
    try:
        fitted, predictions = model.fit(training, exams, weights, variant)
    finally:
        cls.fit = underlying
    manifest_hash = state.save_model(fitted, path / "models", slot)
    data.write_once(path / "manifest-anchors" / (slot + ".json"), {"sha256": manifest_hash})
    data.write_once(path / "predictions" / (slot + ".json"), predictions)


def restore_worker(path, slot):
    """受控自生成模型全量重算，无拟合；只接受已继承或已占用的固定槽位。"""
    values = load_frozen(path)
    if slot in INHERITED:
        expected = data.read(path / "protocol.json")["bridge"][slot]["manifest_sha256"]
    elif slot in NEW_SLOTS and (path / "started" / (slot + ".json")).exists():
        expected = data.read(path / "manifest-anchors" / (slot + ".json"))["sha256"]
    else:
        raise ValueError("UNRECOGNIZED_RESTORE_SLOT")
    fitted = state.restore_model(path / "models", slot, expected)
    training, exams, weights = original.fold_data(values, slot.split("-")[0])
    if (
        fitted["training_hash"] != data.digest(training)
        or fitted["weights_hash"] != data.digest(weights)
        or fitted["variant"] != slot.split("-")[1]
    ):
        raise ValueError("RESTORED_INPUT_IDENTITY_CHANGED")
    checks = state.verify_all_predictions(fitted, training, exams, data.read(path / "predictions" / (slot + ".json")))
    data.write_once(
        path / "restored" / (slot + ".json"),
        {
            "slot": slot,
            "all_rows": checks,
            "state_sha256": state.model_state_digest(fitted),
            "separate_process": True,
            "new_fits": 0,
        },
    )


def ensure_slot(path, slot):
    if (path / "checkpoints" / (slot + ".json")).exists():
        return checkpoint(path, slot)
    if (path / "attempts" / (slot + ".json")).exists():
        raise ValueError("INCOMPLETE_CONSUMED_SLOT_NO_RETRY:" + slot)
    token = secrets.token_hex(24)
    reserve(path, slot, token)
    try:
        child(path, "_fit", slot, token)
        child(path, "_restore", slot)
        files = [f"models/{slot}.joblib", f"models/{slot}.manifest.json"]
        files += [
            f"{folder}/{slot}.json"
            for folder in (
                "manifest-anchors",
                "predictions",
                "restored",
                "started",
                "classifier-started",
                "classifier-completed",
            )
        ]
        item = {
            "slot": slot,
            "completed_at": data.now(),
            "attempt_hash": data.file_hash(path / "attempts" / (slot + ".json")),
            "files": {name: data.file_hash(path / name) for name in files},
        }
        data.write_once(path / "checkpoints" / (slot + ".json"), item)
        return item
    except Exception as exc:
        data.write_once(path / "failures" / (slot + ".json"), {"at": data.now(), "reason": str(exc)})
        raise


def stage_result(path, stage):
    predictions, proofs = {}, {}
    for variant in model.VARIANTS:
        slots = [f"{stage}-{variant}-{role}" for role in ("main", "replay")]
        for slot in slots:
            checkpoint(path, slot)
            restored = data.read(path / "restored" / (slot + ".json"))
            if not restored["separate_process"]:
                raise ValueError("INDEPENDENT_RESTORE_REQUIRED")
        a, b = [data.read(path / "models" / (s + ".manifest.json")) for s in slots]
        if a["state_sha256"] != b["state_sha256"]:
            raise ValueError("INDEPENDENT_NUMERIC_STATE_MISMATCH:" + stage + "-" + variant)
        left, right = [data.read(path / "predictions" / (s + ".json")) for s in slots]
        proofs[variant] = {key: model.compare_predictions(left[key], right[key]) for key in ("train", "exam")}
        predictions[variant] = left["exam"]
    fold = next(f for f in data.read(path / "folds.json") if f["name"] == stage)
    if any([row["target"] for row in rows] != fold["expected_dates"] for rows in predictions.values()):
        raise ValueError("EXAM_DATES_CHANGED")
    data.write_once(path / "reproduction" / (stage + ".json"), {"passed": True, "differences": proofs})
    return predictions


def historical_result(path):
    result = {variant: [] for variant in model.VARIANTS}
    for stage in data.FOLDS:
        for variant, rows in stage_result(path, stage).items():
            result[variant].extend(rows)
    return model.comparison(result, historical=True)


def run(path, *, interrupt_after=None):
    """先历史后开发；已保存检查点复用，任何半成品或技术失败永久停止此运行。"""
    with lock():
        load_frozen(path)
        validate_ledger(path)
        if (path / "technical-stop.json").exists():
            raise ValueError("TERMINAL_TECHNICAL_STOP")
        for attempt in attempts(path):
            if not (path / "checkpoints" / attempt.name).exists():
                data.write_once(
                    path / "technical-stop.json",
                    {
                        "at": data.now(),
                        "reason": "INCOMPLETE_CONSUMED_SLOT_NO_RETRY:" + attempt.stem,
                        "new_reserved_fits": len(attempts(path)),
                    },
                )
                raise ValueError("INCOMPLETE_CONSUMED_SLOT_NO_RETRY")
        if (path / "completion.json").exists():
            for slot in NEW_SLOTS[: len(attempts(path))]:
                checkpoint(path, slot)
            return
        if interrupt_after is not None and interrupt_after not in NEW_SLOTS:
            raise ValueError("INVALID_INTERRUPT_SLOT")
        try:
            for stage in data.FOLDS:
                for slot in HISTORICAL_SLOTS:
                    if not slot.startswith(stage + "-"):
                        continue
                    ensure_slot(path, slot)
                    print(f"已保存 {slot}；新增占用 {len(attempts(path))}/19，继承 5", flush=True)
                    if interrupt_after == slot and not (path / "interruption.json").exists():
                        data.write_once(
                            path / "interruption.json",
                            {
                                "at": data.now(),
                                "slot": slot,
                                "checkpoint_complete": True,
                                "exit_code": 75,
                            },
                        )
                        os._exit(75)
                stage_result(path, stage)
            result = historical_result(path)
            data.write_once(path / "historical-comparison.json", result)
            data.write_once(
                path / "historical-decision.json",
                {
                    "passed": result["numerical_passed"],
                    "status": "PASS" if result["numerical_passed"] else "FAIL",
                    "checks": {**result["checks"], "reproduction_and_restore": True},
                    "comparison_hash": data.digest(result),
                    "inherited_fits": 5,
                    "new_fits": 13,
                    "days": 161,
                },
            )
            full = None
            if result["numerical_passed"]:
                for slot in NEW_SLOTS[len(HISTORICAL_SLOTS) :]:
                    ensure_slot(path, slot)
                    print(f"已保存 {slot}；新增占用 {len(attempts(path))}/19，继承 5", flush=True)
                full = model.comparison(stage_result(path, "FULL"), historical=False)
                data.write_once(
                    path / "development-decision.json",
                    {
                        "passed": full["numerical_passed"],
                        "checks": full["checks"],
                        "comparison_hash": data.digest(full),
                        "days": 230,
                    },
                )
            data.write_once(
                path / "comparison.json",
                {
                    "historical": result,
                    "development": full,
                    "development_status": "COMPLETED" if full is not None else "NOT_RUN_HISTORICAL_GATE_FAILED",
                },
            )
            data.write_once(
                path / "completion.json",
                {
                    "at": data.now(),
                    "new_real_fits": len(attempts(path)),
                    "inherited_fits": 5,
                    "lifetime_real_fits": 39 + len(attempts(path)),
                    "development_run": full is not None,
                    "adopted": False,
                },
            )
        except Exception as exc:
            data.write_once(
                path / "technical-stop.json",
                {
                    "at": data.now(),
                    "reason": str(exc),
                    "new_reserved_fits": len(attempts(path)),
                },
            )
            raise


def verify(path):
    with lock():
        load_frozen(path)
        before = validate_ledger(path)
        slots = (*INHERITED, *NEW_SLOTS[:before])
        for slot in slots:
            checkpoint(path, slot)
            child(path, "_restore", slot)
        if historical_result(path) != data.read(path / "historical-comparison.json"):
            raise ValueError("HISTORICAL_RECOUNT_CHANGED")
        stored = data.read(path / "comparison.json")
        if stored["development"] is not None:
            if model.comparison(stage_result(path, "FULL"), historical=False) != stored["development"]:
                raise ValueError("DEVELOPMENT_RECOUNT_CHANGED")
        if before != validate_ledger(path):
            raise ValueError("VERIFY_MUST_NOT_FIT")
        evidence = {"checked_models": len(slots), "new_fits": 0, "separate_process_restores": True, "passed": True}
        data.write_once(path / "final-verification.json", evidence)
        return evidence


def status(path):
    return {
        "run_id": path.name,
        "inherited_fits": 5,
        "new_reserved": len(attempts(path)),
        "new_classifier_calls": len(list((path / "classifier-started").glob("*.json"))),
        "new_classifier_completed": len(list((path / "classifier-completed").glob("*.json"))),
        "new_checkpoints": len(list((path / "checkpoints").glob("*.json"))),
        "maximum_new_fits": MAX_NEW,
        "historical": data.read(path / "historical-decision.json")
        if (path / "historical-decision.json").exists()
        else None,
        "technical_stop": data.read(path / "technical-stop.json") if (path / "technical-stop.json").exists() else None,
    }
