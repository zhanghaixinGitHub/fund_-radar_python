"""002112 一日离线训练编排：固定快照、六次预算、互斥及可审计恢复。

此模块只供受控本地 CLI 使用；不接同步任务，不写模型登记、关注或预测表。
一次阶段一份不可变尝试记录。进程死亡释放操作系统锁，但不退还已占用的预算。
"""

import hashlib
import importlib.metadata
import os
import platform
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from app.db.session import get_engine
from app.services import fund_002112_training_model as numerical
from app.services import fund_training_package as package
from app.services.direction_1d_protocol import digest
from app.services.direction_1d_training import MODEL_ROOT
from app.services.fund_exposure_common import PROJECT, ROOT, _publish, now, read, save
from app.services.fund_exposure_runtime import execution_lock

RUN_ROOT = ROOT / "training-runs"
MAX_FITS = 6
PROTECTED_FILES = ("plan.json", "feature-spec.json", "dataset-current.json", "study-current.json")
PROTECTED_TABLES = (
    "direction_1d_model",
    "direction_1d_training_run",
    "direction_1d_window_model",
    "analysis_model_release",
    "prediction_model_release",
    "prediction_release_pointer",
    "fund_prediction_record",
)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def run_lock():
    """系统文件锁在进程退出时自动释放；不同运行也串行，避免重复训练同批资料。"""
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    path = RUN_ROOT / ".execution.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("OFFLINE_TRAINING_BUSY") from None
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def environment():
    versions = {
        name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy", "threadpoolctl", "joblib")
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": versions,
        "threads": 1,
    }


def code_manifest():
    """记录训练、资料、时间及推理代码；测试与文档更新不改变候选身份。"""
    paths = set()
    for pattern in (
        "app/services/fund_*.py",
        "app/services/direction_1d*.py",
        "app/services/trading_calendar.py",
        "app/integrations/*fund*.py",
        "scripts/fund_002112_train.py",
        "requirements.txt",
        "app/data/calendars/*.json",
    ):
        paths.update(PROJECT.glob(pattern))
    return {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in sorted(paths)}


def protection():
    """真实只读核对正式登记和预测的摘要，不读取封存答案或个人关注内容。

    数据库端只返回行摘要，单表最大十万条；不扫描 nav_daily 或任何 LABEL payload。
    一日快照只读既有 content_hash 元数据，既有来源答案不进入训练进程。
    """
    files = {str(ROOT / name): file_hash(ROOT / name) for name in PROTECTED_FILES}
    files.update({str(p): file_hash(p) for p in sorted(MODEL_ROOT.glob("*.json"))})
    files.update({str(p): file_hash(p) for p in sorted((PROJECT / "data/prediction-models").rglob("*.json"))})
    tables = {}
    with get_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        connection.execute(text("SET LOCAL statement_timeout='15000'"))
        for table in PROTECTED_TABLES:
            # 表名来自模块固定常量，不接受 CLI 传入 SQL 标识符。
            values = (
                connection.execute(text(f"SELECT md5(t::text) FROM {table} t ORDER BY 1 LIMIT 100001")).scalars().all()
            )
            if len(values) > 100000:
                raise ValueError("OFFLINE_PROTECTION_TABLE_LIMIT")
            tables[table] = {"rows": len(values), "hash": digest(values)}
        values = connection.execute(
            text("SELECT snapshot_id::text,content_hash FROM direction_1d_snapshot ORDER BY snapshot_id LIMIT 100001")
        ).all()
        if len(values) > 100000:
            raise ValueError("OFFLINE_PROTECTION_TABLE_LIMIT")
        tables["direction_1d_snapshot_metadata_only"] = {"rows": len(values), "hash": digest([list(v) for v in values])}
    return {"files": files, "tables": tables}


def receipts_from(value):
    """收集来源回执，保持原到期时间；重复原文取更早许可期限，不给快照续期。"""
    if isinstance(value, dict):
        if "sha256" in value and ("raw_path" in value or "file" in value):
            yield value
        for child in value.values():
            yield from receipts_from(child)
    elif isinstance(value, list):
        for child in value:
            yield from receipts_from(child)


def source_manifest(data):
    snapshot = read(package.STORE / data["source_file"])
    receipts = list(receipts_from(snapshot))
    # 新年份全市场行情原文已在逐行复验中校验，此处固定其回执，恢复不跟随每日指针。
    for path in sorted((ROOT / "stock-days").glob("*.json")):
        if "2021-01-01" <= path.stem <= "2024-12-31":
            receipts.append(read(path)["receipt"])
    entries = {}
    for receipt in receipts:
        path = Path(receipt["raw_path"]) if receipt.get("raw_path") else ROOT / receipt["file"]
        identity = (str(path.resolve()), receipt["sha256"], receipt.get("expires_at"))
        entries[digest(identity)] = {"path": identity[0], "sha256": identity[1], "expires_at": identity[2]}
    return list(entries.values())


def verify_sources(entries):
    for entry in entries:
        if entry["expires_at"] and datetime.fromisoformat(entry["expires_at"]) <= now():
            raise ValueError("OFFLINE_SOURCE_EXPIRED")
        if file_hash(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("OFFLINE_SOURCE_CHANGED")


def checked_directory(run_id):
    if not re.fullmatch(r"002112-[0-9a-f]{24}", run_id):
        raise ValueError("OFFLINE_RUN_ID_INVALID")
    directory = (RUN_ROOT / run_id).resolve()
    if directory.parent != RUN_ROOT.resolve():
        raise ValueError("OFFLINE_RUN_PATH_INVALID")
    return directory


def frozen_identity(data):
    """行序、日期、类别和权重都在同一清单内；A/B/C 仅截取固定列，不重新选行。"""
    return {
        split: [
            {
                "fund_code": r["fund_code"],
                "family": r["family"],
                "base": r["base"],
                "target": r["target"],
                "actual_direction": r["actual_direction"],
                "input_hash": digest(r),
                "weight": data["weights"][i] if split == "train" else None,
            }
            for i, r in enumerate(data[split])
        ]
        for split in ("train", "development")
    }


def freeze(progress=lambda *args: None):
    """共享维护锁内只读取一次 ready 并完整复验，发布新快照不改原研究四文件。"""
    with execution_lock() as locked:
        if not locked:
            raise ValueError("EXPOSURE_BUSY")
        reference = read(package.STORE / "ready.json")
        result = package.verify(progress, publish=False, reference=reference)
        if not result["training_ready"]:
            raise ValueError("OFFLINE_NOT_READY")
        data = read(package.STORE / result["file"])
        numerical.validate_data(data)
        spec = {
            "protocol": numerical.PROTOCOL,
            "fund_code": package.FUND,
            "horizon": 1,
            "reference": {"file": result["file"], "sha256": result["sha256"]},
            "source_hash": data["source_hash"],
            "plan_hash": data["plan_hash"],
            "calendar_hash": data["calendar_hash"],
            "features": package.PLAN["features"],
            "variants": numerical.VARIANTS,
            "recipe": numerical.RECIPE,
            "class_order": list(numerical.CLASSES),
            "tie_order": list(numerical.TIE_ORDER),
            "environment": environment(),
            "code": code_manifest(),
            "maximum_fits": MAX_FITS,
            "selection": numerical.SELECTION,
            "adopted": False,
        }
        run_id = "002112-" + digest(spec)[:24]
        directory = checked_directory(run_id)
        # 修改代码不能成为重新获得六次预算的隐式开关。已有真实尝试须保留原运行处理。
        for previous in RUN_ROOT.glob("002112-*/protocol.json"):
            if previous.parent != directory and list((previous.parent / "attempts").glob("*.json")):
                if read(previous)["reference"]["sha256"] == result["sha256"]:
                    raise ValueError("OFFLINE_EXISTING_BUDGET_REQUIRES_REVIEW")
        if (directory / "protocol.json").exists():
            if read(directory / "protocol.json") != spec:
                raise ValueError("OFFLINE_PROTOCOL_COLLISION")
            return directory
        sources = source_manifest(data)
        verify_sources(sources)
        manifest = {
            "dataset_hash": digest(data),
            "source_hash": data["source_hash"],
            "rows": frozen_identity(data),
            "row_hash": digest(frozen_identity(data)),
            "summary": data["summary"],
            "sources": sources,
            "verified_at": result["verified_at"],
            "limitations": result["limitations"],
        }
        # 初始化中途断电允许核对同内容重用；协议最后发布，存在即意味着快照完整。
        for name, value in (
            ("dataset.json", data),
            ("input-manifest.json", manifest),
            ("protection-before.json", protection()),
        ):
            path = directory / name
            if path.exists():
                if name == "dataset.json" and read(path) != value:
                    raise ValueError("OFFLINE_INITIALIZATION_MISMATCH")
            else:
                save(path, value)
        save(
            directory / "state.json",
            {
                "status": "FROZEN",
                "created_at": now().isoformat(),
                "events": [],
                "fit_attempts": 0,
                "completed": [],
                "failures": {},
            },
            replace=True,
        )
        save(directory / "protocol.json", spec)
        return directory


def load_frozen(directory):
    spec = read(directory / "protocol.json")
    if spec["code"] != code_manifest() or spec["environment"] != environment():
        raise ValueError("OFFLINE_CODE_OR_ENV_CHANGED")
    if spec["recipe"] != numerical.RECIPE or spec["selection"] != numerical.SELECTION:
        raise ValueError("OFFLINE_RECIPE_CHANGED")
    data, manifest = read(directory / "dataset.json"), read(directory / "input-manifest.json")
    if digest(data) != spec["reference"]["sha256"] or digest(data) != manifest["dataset_hash"]:
        raise ValueError("OFFLINE_FROZEN_DATA_CHANGED")
    if frozen_identity(data) != manifest["rows"] or digest(manifest["rows"]) != manifest["row_hash"]:
        raise ValueError("OFFLINE_FROZEN_ROWS_CHANGED")
    if data["calendar_hash"] != package.own.sessions()[1] or read(package.STORE / "plan.json") != package.PLAN:
        raise ValueError("OFFLINE_SOURCE_PROTOCOL_CHANGED")
    # 恢复只核冻结来源，ready 即使换新也不能换本轮输入；期限仍按当前真实时间检查。
    source = read(package.STORE / data["source_file"])
    original = read(package.STORE / spec["reference"]["file"])
    if digest(source) != spec["source_hash"] or digest(original) != digest(data):
        raise ValueError("OFFLINE_SOURCE_SNAPSHOT_CHANGED")
    verify_sources(manifest["sources"])
    numerical.validate_data(data)
    return data


def put_once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("OFFLINE_ARTIFACT_CHANGED")
    else:
        save(path, value)


def record_state(directory, state, event, **details):
    state["events"].append({"at": now().isoformat(), "event": event, **details})
    state["updated_at"] = now().isoformat()
    state["fit_attempts"] = len(list((directory / "attempts").glob("*.json")))
    save(directory / "state.json", state, replace=True)


def run_stages(directory, data, *, interrupt_after_checkpoint=None):
    """每版本最多主/复现各一次；不明中断不重复拟合，已有完成文件可恢复后续步骤。

    验收参数仅在完成产物落盘后模拟进程突然退出，用于实际检验恢复，不增加拟合。
    如退出发生在 fit 内、尚无产物，预算保守视为已消耗，该阶段明确技术失败。
    """
    state = read(directory / "state.json")
    state["status"] = "RUNNING"
    record_state(directory, state, "RESUME_OR_START")
    successful = {}
    for variant in numerical.VARIANTS:
        try:
            outputs = {}
            for phase in ("main", "replay"):
                stage = f"{variant}:{phase}"
                attempt = directory / "attempts" / f"{variant}-{phase}.json"
                checkpoint = directory / "checkpoints" / f"{variant}-{phase}.json"
                if checkpoint.exists():
                    if not attempt.exists():
                        raise ValueError("OFFLINE_UNACCOUNTED_CHECKPOINT")
                    outputs[phase] = read(checkpoint)
                    record_state(directory, state, "CHECKPOINT_REUSED", stage=stage)
                    continue
                if attempt.exists():
                    raise ValueError("OFFLINE_INTERRUPTED_FIT_BUDGET_CONSUMED")
                if len(list((directory / "attempts").glob("*.json"))) >= MAX_FITS:
                    raise ValueError("OFFLINE_FIT_BUDGET_EXHAUSTED")
                # 持久化在数值调用之前；收敛异常、硬中断和重复调用都不能退还预算。
                save(attempt, {"stage": stage, "at": now().isoformat(), "dataset_hash": digest(data)})
                record_state(directory, state, "FIT_ENTERED", stage=stage)
                output = numerical.fit(data, variant)
                put_once(checkpoint, output)
                if interrupt_after_checkpoint == stage:
                    os._exit(75)
                outputs[phase] = output
                record_state(directory, state, "FIT_SAVED", stage=stage, model_hash=digest(output["model"]))
            main, replay = outputs["main"], outputs["replay"]
            if main["model"] != replay["model"]:
                raise ValueError("OFFLINE_REPLAY_MODEL_MISMATCH")
            model = main["model"]
            if model["dataset_hash"] != digest(data) or model["variant"] != variant:
                raise ValueError("OFFLINE_CHECKPOINT_IDENTITY")
            rows = numerical.predictions(model, data["development"])
            if rows != numerical.predictions(replay["model"], data["development"]):
                raise ValueError("OFFLINE_REPLAY_PREDICTIONS_MISMATCH")
            put_once(directory / "models" / f"{variant}.json", model)
            put_once(
                directory / "replay" / f"{variant}.json",
                {
                    "main_hash": digest(model),
                    "replay_hash": digest(replay["model"]),
                    "parameters_equal": True,
                    "predictions_equal": True,
                    "main_restore": main["restore"],
                    "replay_restore": replay["restore"],
                },
            )
            put_once(directory / "predictions" / f"{variant}.json", rows)
            successful[variant] = rows
            state["completed"] = list(successful)
            state["failures"].pop(variant, None)
            record_state(directory, state, "VARIANT_COMPLETED", variant=variant)
        except Exception as exc:
            # 错误只记类型及内部简短原因，不将第三方异常中的连接信息或原文写入日志。
            reason = str(exc) if isinstance(exc, ValueError) and str(exc).startswith("OFFLINE_") else type(exc).__name__
            # 重入失败阶段不把首次收敛等具体错误改写成笼统的中断错误。
            state["failures"].setdefault(variant, reason)
            record_state(directory, state, "VARIANT_FAILED", variant=variant, reason=reason)
    comparison = numerical.compare(data, successful)
    after = protection()
    unchanged = after == read(directory / "protection-before.json")
    put_once(directory / "protection-after.json", after)
    if not unchanged:
        state["failures"]["protection"] = "OFFLINE_PROTECTED_STATE_CHANGED"
    state["status"] = "COMPLETED" if comparison["complete"] and unchanged else "PARTIAL" if successful else "FAILED"
    completion = {
        "run_id": directory.name,
        "status": state["status"],
        "models": len(successful),
        "fit_attempts": len(list((directory / "attempts").glob("*.json"))),
        "failures": state["failures"],
        "protected_unchanged": unchanged,
        "comparison_hash": digest(comparison),
        "adopted": False,
        "models_registered_by_this_run": 0,
    }
    # 部分失败也留最终报告，不删成功成果；每阶段都已耗用或完成，后续重入只能复用。
    put_once(directory / "comparison.json", comparison)
    report_path = directory / "report.md"
    report_bytes = numerical.render_report(comparison).encode("utf-8")
    if report_path.exists():
        if report_path.read_bytes() != report_bytes:
            raise ValueError("OFFLINE_REPORT_CHANGED")
    else:
        _publish(report_path, report_bytes)
    put_once(directory / "completion.json", completion)
    record_state(directory, state, "RUN_FINISHED", status=state["status"])
    return completion


def execute(run_id=None, *, freeze_only=False, interrupt_after_checkpoint=None, progress=lambda *args: None):
    with run_lock():
        directory = checked_directory(run_id) if run_id else freeze(progress)
        if freeze_only:
            return {"run_id": directory.name, "status": read(directory / "state.json")["status"]}
        try:
            data = load_frozen(directory)
            return run_stages(directory, data, interrupt_after_checkpoint=interrupt_after_checkpoint)
        except Exception as exc:
            state = read(directory / "state.json")
            state["status"] = "BLOCKED"
            record_state(
                directory,
                state,
                "PRECHECK_OR_ARTIFACT_FAILED",
                reason=type(exc).__name__
                + ":"
                + (str(exc) if isinstance(exc, ValueError) and str(exc).startswith("OFFLINE_") else "CHECK_FAILED"),
            )
            raise


def status(run_id):
    directory = checked_directory(run_id)
    state = read(directory / "state.json")
    effective = state["status"]
    if effective == "RUNNING":
        try:
            with run_lock():
                # 能取得锁说明原进程已退出；保留原账本，用状态返回值明确恢复点。
                effective = "INTERRUPTED"
        except ValueError as exc:
            if str(exc) != "OFFLINE_TRAINING_BUSY":
                raise
    return {
        "run_id": run_id,
        "status": effective,
        "completed": state["completed"],
        "fit_attempts": len(list((directory / "attempts").glob("*.json"))),
        "failures": state["failures"],
        "directory": str(directory),
        "last_event": state["events"][-1] if state["events"] else None,
    }


def export_report(run_id):
    """不拟合，从已保存 JSON 参数重建逐日结果及报告，并核对已发布指标。"""
    with run_lock():
        directory = checked_directory(run_id)
        data = load_frozen(directory)
        rows = {}
        for variant in numerical.VARIANTS:
            path = directory / "models" / f"{variant}.json"
            if path.exists():
                model = read(path)
                rows[variant] = numerical.predictions(model, data["development"])
                if rows[variant] != read(directory / "predictions" / f"{variant}.json"):
                    raise ValueError("OFFLINE_RECOMPUTE_PREDICTIONS")
        comparison = numerical.compare(data, rows)
        if comparison != read(directory / "comparison.json"):
            raise ValueError("OFFLINE_RECOMPUTE_METRICS")
        if numerical.render_report(comparison) != (directory / "report.md").read_text(encoding="utf-8"):
            raise ValueError("OFFLINE_RECOMPUTE_REPORT")
        return {"run_id": run_id, "recomputed_equal": True, "report": str(directory / "report.md"), "fits": 0}
