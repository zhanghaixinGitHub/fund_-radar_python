"""002112 2024—2026 独立实验编排；只写本实验，不调用现用训练入口。

不可逆研究证据使用排他新增；预算先 fsync 消费再启动监督拟合。
同一计划中断后保留消费，完成模型直接续用。协调 GO 只属于内部验收。
"""

# 环境线程参数必须在 NumPy/scikit-learn 导入前设置。
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_info

from scripts import fund_002112_existing_data_fit_v1 as fit

PY = Path(__file__).resolve().parents[1]
WEB = Path("C:/WebStormProject/workSpace05")
JAVA = Path("C:/ideaProject/workSpace12")
RESEARCH = PY / ".local-runs/fund-exposure-002112"
ROOT = RESEARCH / "existing-data-experiment/20260930-v1"
DOC = WEB / "docs_zhx/implementation/fund-002112-existing-data-training-implementation-2026-09-30.md"
EXPERIMENT = "FUND_002112_EXISTING_DATA_2024_2026_V1"
ZONE = ZoneInfo("Asia/Shanghai")
FOLDS = {
    "V1": ("2024-12-31", "2025-01-01", "2025-06-30"),
    "V2": ("2025-06-30", "2025-07-01", "2025-09-30"),
    "V3": ("2025-09-30", "2025-10-01", "2025-12-31"),
    "T": ("2025-12-31", "2026-01-01", "2026-09-29"),
    "FINAL": ("2026-09-29", None, None),
}
BUCKETS = {"validation": 24, "replay_validation": 9, "test": 6, "final": 6, "retry": 3}
B_FILES = [
    PY / "scripts" / x
    for x in (
        "fund_002112_existing_data_v1.py",
        "fund_002112_existing_data_fit_v1.py",
        "test_fund_002112_existing_data_v1.py",
    )
]


def now():
    return datetime.now(ZONE).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    """排他创建，已存在的证据绝不覆盖；调用方显式复用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def replace_status(path, value):
    """仅用于派生预算/进度视图；消费原件在 append-only 账本中。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def lines(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


@contextmanager
def lock(root):
    """OS 持有的独占锁；进程退出自动解锁，不按年龄抢占仍活着的进程。"""
    import msvcrt

    root.mkdir(parents=True, exist_ok=True)
    path = root / ".execution.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("EXPERIMENT_ALREADY_RUNNING") from exc
        try:
            replace_status(
                root / "lock-owner.json",
                {
                    "experiment_id": EXPERIMENT,
                    "pid": os.getpid(),
                    "started_at": now(),
                    "executable": sys.executable,
                    "argv": sys.argv,
                    "ledger": str(root / "fit-ledger.jsonl"),
                },
            )
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class Budget:
    """先原子预占再 fit；失败或无结束回执的 reservation 永远计为消费。"""

    def __init__(self, root):
        self.root = Path(root)

    def state(self):
        reservations = [r for r in lines(self.root / "fit-ledger.jsonl") if r["event"] == "RESERVED"]
        counts = Counter(r["bucket"] for r in reservations)
        return {
            "experiment_id": EXPERIMENT,
            "limit": 48,
            "consumed": len(reservations),
            "remaining": 48 - len(reservations),
            "buckets": BUCKETS,
            "consumed_by_bucket": dict(counts),
            "reservations": reservations,
        }

    def reserve(self, plan, bucket, metadata):
        # 此锁也保护单独测试进程；主执行锁不能替代原子预算锁。
        with lock(self.root / "budget-lock"):
            state = self.state()
            if any(r["plan"] == plan for r in state["reservations"]):
                raise RuntimeError("PLAN_ALREADY_CONSUMED_RETRY_MUST_BE_EXPLICIT")
            if state["consumed"] >= 48 or state["consumed_by_bucket"].get(bucket, 0) >= BUCKETS[bucket]:
                raise RuntimeError("FIT_BUDGET_EXHAUSTED")
            entry = {
                "event": "RESERVED",
                "number": state["consumed"] + 1,
                "plan": plan,
                "bucket": bucket,
                "at": now(),
                **metadata,
            }
            append(self.root / "fit-ledger.jsonl", entry)
            replace_status(self.root / "budget.json", self.state())
            return entry


def progress(root, phase, status, next_action, **details):
    record = {
        "at": now(),
        "phase": phase,
        "status": status,
        "real_fits": Budget(root).state()["consumed"],
        "next_action": next_action,
        **details,
    }
    append(root / "handoff/training/progress-history.jsonl", record)
    replace_status(root / "handoff/training/progress.json", record)
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def runtime_evidence():
    """限定只读事务；仅公共模型、绑定和基金预测摘要，不查询用户个人数据。"""
    from app.core.config import get_settings
    from sqlalchemy import URL, create_engine, text

    cfg = {}
    for line in (JAVA / ".env").read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    core_url = URL.create(
        "postgresql+psycopg",
        username=cfg["FUND_CORE_DB_USERNAME"],
        password=cfg["FUND_CORE_DB_PASSWORD"],
        host=cfg.get("FUND_CORE_DB_HOST", "localhost"),
        port=int(cfg.get("FUND_CORE_DB_PORT", "54329")),
        database=cfg.get("FUND_CORE_DB_NAME", "fund_core"),
    )
    result = {"at": now(), "read_only": True}
    for name, url in (("ai", get_settings().ai_database_url), ("core", core_url)):
        engine = create_engine(
            url,
            hide_parameters=True,
            connect_args={
                "connect_timeout": 5,
                "options": "-c default_transaction_read_only=on -c statement_timeout=15000",
            },
        )
        if engine.url.host not in ("localhost", "127.0.0.1"):
            raise ValueError("LOCAL_DB_ONLY")
        with engine.connect() as c:

            def query(sql):
                return [dict(r) for r in c.execute(text(sql)).mappings()]

            data = {"locks": query("select classid,objid,mode from pg_locks where locktype='advisory' and granted")}
            if name == "ai":
                data["models"] = query(
                    "select model_id::text,content_hash,file_name,trained_at,metadata "
                    "from direction_1d_model order by model_id"
                )
                data["bindings"] = query(
                    "select cohort_id,group_id,base_nav_date,target_nav_date,branch_id,"
                    "model_id::text,activation_policy from direction_1d_window_model "
                    "order by cohort_id,group_id,target_nav_date,branch_id"
                )
                data["active_jobs"] = query(
                    "select kind,state,count(*) n from direction_1d_job "
                    "where state in ('QUEUED','RUNNING') group by kind,state"
                )
                data["model_files"] = {
                    r["model_id"]: sha(PY / ".local-runs/direction-1d-models" / r["file_name"]) for r in data["models"]
                }
            else:
                data["predictions"] = query(
                    "select forecast_id::text,content_hash,"
                    "encode(sha256(convert_to(payload_json,'UTF8')),'hex') as actual_hash,"
                    "input_hash,target_nav_date,generated_at from direction_1d_forecast "
                    "where fund_code='002112' order by forecast_id"
                )
                data["directions"] = query(
                    "select f.forecast_id::text,s.branch_id,s.model_id::text,s.model_hash,"
                    "s.predicted_direction,s.train_as_of from direction_1d_forecast f "
                    "join direction_1d_forecast_score s using(forecast_id) where f.fund_code='002112' "
                    "order by f.forecast_id,s.branch_id"
                )
            result[name] = data
        engine.dispose()
    ps = (
        "$p=Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python|java|node' -and "
        "$_.CommandLine -match 'workSpace06|workSpace05|workSpace12'} | "
        "Select-Object ProcessId,ExecutablePath,CommandLine;"
        "$n=Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object "
        "{$_.LocalPort -in 5173,8080,8000} | Select-Object LocalPort,OwningProcess;"
        "@{processes=@($p);listeners=@($n)} | ConvertTo-Json -Depth 5 -Compress"
    )
    result["processes"] = json.loads(
        subprocess.check_output(["powershell", "-NoProfile", "-Command", ps], text=True, encoding="utf-8")
    )
    return json.loads(json.dumps(result, default=str))


def protect(root):
    if (root / "protection-before.json").exists():
        return
    repositories, dirty, protected = [], {}, {}
    for repo in (PY, WEB, JAVA):

        def git(*args, repo=repo):
            return subprocess.check_output(["git", "-C", str(repo), *args])

        repositories.append(
            {
                "path": str(repo),
                "head": git("rev-parse", "HEAD").decode().strip(),
                "status": git("status", "--short", "-uall").decode("utf-8"),
            }
        )
        names = (
            set(git("diff", "--name-only", "-z").split(b"\0"))
            | set(git("diff", "--cached", "--name-only", "-z").split(b"\0"))
            | set(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"))
        )
        for name in filter(None, names):
            path = repo / name.decode("utf-8")
            if path.is_file() and path not in B_FILES:
                dirty[str(path)] = sha(path)
    # 历史研究模型、目标快照、预算及旧源码全量摘要；其他已验收原件引用近期保护清单。
    recent = read(RESEARCH / "recent-input-completion/20260930-v1/protection-before.json")
    paths = set(Path(p) for p in recent["protected"])
    for folder in (
        "information-research",
        "training-runs",
        "round3-runs",
        "round3-repair-runs",
        "peer-recovered-experiment",
        "optimization-runs",
        "datasets",
    ):
        paths.update(p for p in (RESEARCH / folder).rglob("*") if p.is_file())
    paths.update(p for p in (PY / ".local-runs/direction-1d-models").rglob("*") if p.is_file())
    paths.update(p for p in (PY / "scripts").glob("fund_002112*.py") if "existing_data" not in p.name)
    paths.update(p for p in (PY / "app/services").glob("fund_002112*.py"))
    for path in sorted(paths):
        protected[str(path)] = sha(path)
    old = read(RESEARCH / "information-research/20260928-v1/budget-authorized.json")
    fits = list((RESEARCH / "information-research/20260928-v1/fit-ledger").glob("*.json"))
    save(
        root / "protection-before.json",
        {
            "at": now(),
            "repositories": repositories,
            "dirty_files": dirty,
            "protected": protected,
            "old_cumulative_fits": old["previous_actual_fits"] + len(fits),
            "runtime": runtime_evidence(),
            "parallel_task": "01a0f020-79fd-71a0-901e-cacdd76bacfa completed",
            "allowed_parallel_changes": "C narrative plus existing alerts/materials; never revert",
        },
    )


def prepare(root):
    protect(root)
    if not (root / "authorization.json").exists():
        save(
            root / "authorization.json",
            {
                "experiment_id": EXPERIMENT,
                "at": now(),
                "new_budget": 48,
                "user_authorization": "允许新实验使用三年数据，旧实验保持不变",
                "execution_authorization": "执行文档中的独立实验，不再等待所有历史公告、政策和精确版本证据补齐；"
                "遵守八类固定候选和最多48次真实数据监督拟合，失败、复现、重试、最终重训都计数；不登记、不换模、不部署。",
                "source_document": str(DOC),
                "source_document_sha256": sha(DOC),
                "coordinator": "01a0f18e-3002-7601-ae9d-d53c4d960279",
                "old_budget_never_borrowed": True,
            },
        )
    replace_status(root / "budget.json", Budget(root).state())
    progress(root, "PREPARE", "DONE", "完成人工边界测试，等待 A ready 后冻结协议")


def inputs(root):
    api = importlib.import_module("scripts.fund_002112_existing_data_inputs_v1")
    data = api.load_inputs(root)
    names = data["group_columns"]
    rows = []
    for source in data["rows"]:
        row = dict(source)
        row.update(
            U=source["target_date"],
            T=source["base_date"],
            S=source["nav_end_date"],
            cutoff=source["as_of"],
            available={k: v["available"] for k, v in source["availability"].items()},
            reasons={k: ";".join(v["reasons"]) for k, v in source["availability"].items()},
            features={},
        )
        for group, values in source["groups"].items():
            if values is not None:
                if len(values) != len(names[group]):
                    raise ValueError("FEATURE_LENGTH_MISMATCH")
                row["features"].update(zip(names[group], values, strict=True))
        rows.append(row)
    dates = [r["U"] for r in rows]
    if len(dates) != 665 or dates != sorted(set(dates)) or dates[0] != "2024-01-02" or dates[-1] != "2026-09-29":
        raise ValueError("FROZEN_CALENDAR_MISMATCH")
    if any(any(k in r for k in ("label", "actual", "direction", "target_nav")) for r in data["rows"]):
        raise ValueError("INPUT_CONTAINS_TARGET")
    for c in fit.GROUPS:
        if fit.columns(c, names) != data["candidate_columns"][c]:
            raise ValueError("CANDIDATE_COLUMN_ORDER_MISMATCH")
    return data, rows, names


def active_data_ready(root):
    """活动版本只能由 A 的纯加载器解析；指针不存在时兼容首版。"""
    pointer = root / "handoff/data/active-revision.json"
    path = root / "handoff/data/ready.json"
    expected = None
    if pointer.exists():
        api = importlib.import_module("scripts.fund_002112_existing_data_inputs_v1")
        data = api.load_inputs(root)
        path = Path(data["data_ready_path"])
        path = path if path.is_absolute() else root / path
        expected = data["data_ready_sha256"]
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise RuntimeError("ACTIVE_READY_OUTSIDE_EXPERIMENT_OR_MISSING")
    if expected is not None and sha(path) != expected:
        raise RuntimeError("ACTIVE_READY_HASH_MISMATCH")
    return path, sha(pointer) if pointer.exists() else None


def verify_data_ready(root):
    """逐项验证 A 实际产物内容，不能仅核对清单自身；不读取目标标签。"""
    ready_path, active_sha = active_data_ready(root)
    ready = read(ready_path)
    if ready["status"] != "READY" or ready["experiment_id"] != EXPERIMENT:
        raise RuntimeError("A_HANDOFF_NOT_READY")
    for rel, expected in ready["artifacts"].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha(path) != expected:
            raise RuntimeError("A_ARTIFACT_HASH_MISMATCH: " + rel)
    for filename, expected in ready["code_sha256"].items():
        if sha(filename) != expected:
            raise RuntimeError("A_CODE_HASH_MISMATCH: " + filename)
    return {
        "passed": True,
        "artifact_count": len(ready["artifacts"]),
        "code_count": len(ready["code_sha256"]),
        "at": now(),
        "ready_path": str(ready_path),
        "ready_sha256": sha(ready_path),
        "active_revision_sha256": active_sha,
    }


def supersede_freeze(root):
    """仅在尚无真实标签/拟合时保留首冻原件并准备接收已授权的数据适配修订。"""
    if Budget(root).state()["consumed"] or lines(root / "label-access-ledger.jsonl"):
        raise RuntimeError("CANNOT_SUPERSEDE_AFTER_REAL_ACCESS")
    ready_path, active_sha = active_data_ready(root)
    if active_sha is None or ready_path == (root / "handoff/data/ready.json").resolve():
        raise RuntimeError("NEW_ACTIVE_DATA_REVISION_REQUIRED")
    old = read(root / "protocol.json")
    if old["data_ready_sha256"] == sha(ready_path):
        raise RuntimeError("ALREADY_FROZEN_CURRENT_DATA_REVISION")
    archive = root / "handoff/training/revisions/r1-superseded"
    archive.mkdir(parents=True, exist_ok=True)
    receipt = {
        "at": now(),
        "supersedes_protocol_sha256": sha(root / "protocol.json"),
        "old_data_ready_sha256": old["data_ready_sha256"],
        "new_ready_path": str(ready_path),
        "new_data_ready_sha256": sha(ready_path),
        "active_revision_sha256": active_sha,
        "reason": "协调核验发现F排除字段有原件，按原协议补适配；真实标签0/拟合0，候选预算不变",
    }
    # 移动的只是 B 自己的冻结入口，原字节留在 archive；A 首版文件不触碰。
    paths = [
        root / "protocol.json",
        root / "code-manifest.json",
        root / "validation-results.json",
        root / "handoff/training/ready-for-review.json",
    ]
    for path in paths:
        destination = archive / path.name
        if destination.exists():
            raise RuntimeError("SUPERSESSION_DESTINATION_ALREADY_EXISTS")
        expected = sha(path)
        path.rename(destination)
        if sha(destination) != expected:
            raise RuntimeError("SUPERSESSION_ARCHIVE_MISMATCH")
    save(root / "freeze-supersession.json", receipt)
    progress(root, "REVISION", "WAITING_R2_TEST_FREEZE", "完成新版本测试并冻结；旧首冻原件完整保留")


def freeze(root):
    if (root / "protocol.json").exists():
        verify_frozen(root, require_go=False)
        return
    validation = read(root / "validation-results.json")
    if not validation["passed"]:
        raise RuntimeError("BOUNDARY_TESTS_REQUIRED")
    if any(validation["source_sha256"].get(str(p)) != sha(p) for p in B_FILES):
        raise RuntimeError("TESTED_SOURCE_CHANGED")
    integrity = verify_data_ready(root)
    data, rows, names = inputs(root)
    code_paths = B_FILES + [
        PY / "scripts/fund_002112_existing_data_inputs_v1.py",
        PY / "scripts/test_fund_002112_existing_data_inputs_v1.py",
    ]
    manifest = {str(p): sha(p) for p in code_paths}
    save(root / "code-manifest.json", {"files": manifest, "at": now()})
    protocol = {
        "experiment_id": EXPERIMENT,
        "frozen_at": now(),
        "spec": str(DOC),
        "spec_sha256": sha(DOC),
        "calendar": [r["U"] for r in rows],
        "folds": FOLDS,
        "class_order": fit.CLASSES,
        "candidate_groups": fit.GROUPS,
        "columns": data["candidate_columns"],
        "optional_decisions": data["optional_decisions"],
        "lr": fit.LR_PARAMS,
        "hist": fit.H_PARAMS,
        "fallback": {c: fit.dependencies(c) for c in fit.GROUPS},
        "budget": BUCKETS,
        "max_fits": 48,
        "minimum_rows": {"logistic": 60, "hist": 120},
        "labels": "separate staged access; U vs T",
        "selection": (
            "net gain >0; improved folds>=2; UP/DOWN recall decline<=0.05; rank net gain, fewer columns, LR, letter"
        ),
        "bootstrap": {"sessions": 5, "repetitions": 2000, "seed": 20260930, "full_calendar_mask": True},
        "replay_probability_tolerance": 1e-10,
        "threads": 1,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "threadpools": threadpool_info(),
        },
        "final_label_cutoff": read(root / "handoff/data/freeze-metadata.json")["frozen_at"],
        "input_digest": data["input_digest"],
        "data_ready_path": integrity["ready_path"],
        "data_ready_sha256": integrity["ready_sha256"],
        "active_revision_sha256": integrity["active_revision_sha256"],
        "supersedes": read(root / "freeze-supersession.json") if (root / "freeze-supersession.json").exists() else None,
        "code_manifest_sha256": sha(root / "code-manifest.json"),
    }
    save(root / "protocol.json", protocol)
    save(
        root / "handoff/training/ready-for-review.json",
        {
            "status": "WAITING_INTERNAL_GO",
            "at": now(),
            "protocol_sha256": sha(root / "protocol.json"),
            "code_manifest_sha256": sha(root / "code-manifest.json"),
            "data_ready_path": integrity["ready_path"],
            "data_ready_sha256": integrity["ready_sha256"],
            "active_revision_sha256": integrity["active_revision_sha256"],
            "validation_results_sha256": sha(root / "validation-results.json"),
            "data_integrity": integrity,
            "source_integrity": {"passed": True, "files": manifest},
            "next_command": ".venv\\Scripts\\python.exe -X utf8 -B -m scripts.fund_002112_existing_data_v1 resume",
            "real_fits": 0,
            "labels_accessed": 0,
        },
    )
    progress(root, "FREEZE", "WAITING_INTERNAL_GO", "协调者核验 ready-for-review 后写入 GO")


def verify_frozen(root, *, require_go=True):
    protocol = read(root / "protocol.json")
    if sha(DOC) != protocol["spec_sha256"]:
        raise RuntimeError("SPEC_CHANGED")
    if sha(root / "code-manifest.json") != protocol["code_manifest_sha256"]:
        raise RuntimeError("CODE_MANIFEST_CHANGED")
    for path, expected in read(root / "code-manifest.json")["files"].items():
        if sha(path) != expected:
            raise RuntimeError("FROZEN_CODE_CHANGED: " + path)
    ready_path, active_sha = active_data_ready(root)
    if active_sha != protocol.get("active_revision_sha256"):
        raise RuntimeError("ACTIVE_DATA_REVISION_CHANGED")
    if sha(ready_path) != protocol["data_ready_sha256"]:
        raise RuntimeError("A_READY_CHANGED")
    verify_data_ready(root)
    if require_go:
        release = read(root / "handoff/coordinator-train-release.json")
        expected = {
            "status": "GO",
            "protocol_sha256": sha(root / "protocol.json"),
            "code_manifest_sha256": sha(root / "code-manifest.json"),
            "data_ready_sha256": sha(ready_path),
        }
        if any(str(release.get(k, "")).lower() != str(v).lower() for k, v in expected.items()):
            raise RuntimeError("COORDINATOR_GO_HASH_MISMATCH")
    return protocol


def checkpoint(root, name, paths, extra=None):
    path = root / "checkpoints" / (name + ".json")
    if path.exists():
        require_checkpoint(root, name)
        return
    save(path, {"at": now(), "artifacts": {str(p.relative_to(root)): sha(p) for p in paths}, **(extra or {})})


def require_checkpoint(root, name):
    record = read(root / "checkpoints" / (name + ".json"))
    for rel, expected in record["artifacts"].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError("CHECKPOINT_CHANGED: " + rel)
    return record


def stage_cutoff(root, stage, rows):
    if stage == "FINAL":
        return read(root / "protocol.json")["final_label_cutoff"]
    start, end = FOLDS[stage][1:]
    return min(r["cutoff"] for r in rows if start <= r["U"] <= end)


def label_access(root, stage, purpose, dates, rows):
    """先写访问事实再读标签；每个 gate 和标签版本都保持不可变。"""
    verify_frozen(root)
    key = stage.lower() + "-" + purpose
    destination = root / "labels" / (key + ".json")
    gate_path = root / "handoff/training/label-gates" / (key + ".json")
    gate = {
        "protocol_sha256": sha(root / "protocol.json"),
        "stage": stage,
        "purpose": purpose,
        "allowed_target_dates": dates,
        "fit_cutoff": stage_cutoff(root, stage, rows),
    }
    if stage == "T" and purpose == "evaluate":
        require_checkpoint(root, "predict-test")
        gate.update(
            prediction_freeze_path=str(root / "test-predictions-freeze.json"),
            prediction_freeze_sha256=sha(root / "test-predictions-freeze.json"),
        )
    if stage == "FINAL":
        require_checkpoint(root, "evaluate-test")
        gate.update(
            test_evaluation_path=str(root / "test-comparison.json"),
            test_evaluation_sha256=sha(root / "test-comparison.json"),
        )
    if gate_path.exists():
        if read(gate_path) != gate:
            raise RuntimeError("LABEL_GATE_CHANGED")
    else:
        save(gate_path, gate)
    if destination.exists():
        receipt = read(destination.with_suffix(".receipt.json"))
        if sha(destination) != receipt["sha256"] or sha(gate_path) != receipt["gate_sha256"]:
            raise ValueError("LABEL_RESULT_CHANGED")
        return read(destination)
    append(
        root / "label-access-ledger.jsonl",
        {
            "at": now(),
            "stage": stage,
            "purpose": purpose,
            "dates_digest": fit.digest(dates),
            "n": len(dates),
            "gate": str(gate_path),
            "gate_sha256": sha(gate_path),
            "protocol_sha256": gate["protocol_sha256"],
            "coordinator_go_sha256": sha(root / "handoff/coordinator-train-release.json"),
        },
    )
    api = importlib.import_module("scripts.fund_002112_existing_data_inputs_v1")
    result = api.load_labels(
        root, dates, stage=stage, purpose=purpose, protocol_sha256=gate["protocol_sha256"], gate_path=gate_path
    )
    save(destination, result)
    save(destination.with_suffix(".receipt.json"), {"sha256": sha(destination), "gate_sha256": sha(gate_path)})
    return result


def training_rows(rows, labels, stage, cutoff, candidate, names):
    """再次按拟合时点检查成熟度，防止目标日期在范围内而答案迟披露。"""
    boundary = FOLDS[stage][0]
    return [
        r
        for r in rows
        if "2024-01-02" <= r["U"] <= boundary
        and r["U"] in labels["labels"]
        and datetime.fromisoformat(labels["maturity"][r["U"]]) <= datetime.fromisoformat(cutoff)
        and fit.usable(r, candidate, names)
    ]


def worker(job_path):
    """一个预占编号只允许进程首次 claim 后 fit 一次；重跑 worker 不能绕过消费。"""
    job_path = Path(job_path).resolve()
    job = read(job_path)
    root = Path(job["root"]).resolve()
    if root != ROOT.resolve() or not job_path.is_relative_to(root / "fit-attempts"):
        raise ValueError("REAL_WORKER_ROOT_MISMATCH")
    verify_frozen(root)
    reservation = [r for r in Budget(root).state()["reservations"] if r["plan"] == job["plan"]]
    if len(reservation) != 1 or reservation[0]["job_sha256"] != sha(job_path):
        raise ValueError("WORKER_RESERVATION_REQUIRED")
    save(job_path.parent / "worker-started.json", {"at": now(), "pid": os.getpid(), "plan": job["plan"]})
    try:
        bundle = fit.fit_estimator(job["x"], job["y"], job["candidate"], retry=job["extended_iterations"])
        model_path = job_path.parent / "model.joblib"
        joblib.dump(bundle, model_path)
        loaded = joblib.load(model_path)
        save_load = fit.compare_models(bundle, loaded, job["probe_x"])
        if not save_load["passed"]:
            raise RuntimeError("MODEL_SAVE_LOAD_MISMATCH")
        save(
            job_path.parent / "result.json",
            {
                "at": now(),
                "status": "SUCCESS" if bundle["converged"] else "NON_CONVERGED",
                "model_sha256": sha(model_path),
                "structure_sha256": fit.digest(fit.model_structure(bundle)),
                "save_load": save_load,
                "observed_classes": bundle["observed_classes"],
                "unlearnable_classes": bundle["unlearnable_classes"],
                "warnings": bundle["warnings"],
                "zero_variance_columns": bundle["zero_variance_columns"],
                "params": bundle["params"],
                "python": sys.version,
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "threadpools": threadpool_info(),
                "pid": os.getpid(),
            },
        )
    except Exception:
        save(job_path.parent / "error.json", {"at": now(), "traceback": traceback.format_exc()})
        raise


def fit_one(root, stage, candidate, rows, names, labels, *, replay=False, original=None):
    tag = "replay-" if replay else "main-"
    ref_dir = (
        root / "replay" / ("validation" if stage.startswith("V") else stage.lower()) / stage
        if replay
        else root / "models" / stage.lower()
    )
    ref = ref_dir / (candidate + ".json")
    if ref.exists():
        record = read(ref)
        if record["status"] == "SUCCESS":
            if sha(record["model_path"]) != record["model_sha256"]:
                raise RuntimeError("COMPLETED_MODEL_CHANGED")
        return record
    selected = training_rows(rows, labels, stage, stage_cutoff(root, stage, rows), candidate, names)
    y = [labels["labels"][r["U"]] for r in selected]
    cols = fit.columns(candidate, names)
    minimum = 120 if candidate == "H" else 60
    reason = None
    if candidate in "FG" and not names[candidate]:
        reason = "NO_INDEPENDENT_OPTIONAL_INPUT"
    elif len(selected) < minimum:
        reason = "INSUFFICIENT_BRANCH_ROWS"
    elif len(set(y)) < 2:
        reason = "FEWER_THAN_TWO_OBSERVED_CLASSES"
    if reason:
        record = {
            "status": "SKIPPED",
            "reason": reason,
            "candidate": candidate,
            "stage": stage,
            "training_n": len(selected),
            "class_counts": dict(Counter(y)),
        }
        save(ref, record)
        return record
    x = [[r["features"][c] for c in cols] for r in selected]
    probes = [r for r in rows if fit.usable(r, candidate, names)]
    probe_x = [[r["features"][c] for c in cols] for r in probes]
    base_plan = tag + stage + "-" + candidate
    existing = [r for r in Budget(root).state()["reservations"] if r["plan"].startswith(base_plan + "-")]
    if existing:
        # 先复原已成功 worker 的回执；未结束的调用仍消费，不自动重领或重跑。
        last = existing[-1]
        job_path = root / "fit-attempts" / f"{last['number']:03d}" / "job.json"
        done = job_path.parent / "result.json"
        if not done.exists() or read(done)["status"] != "SUCCESS":
            raise RuntimeError("INTERRUPTED_PLAN_REQUIRES_EXPLICIT_TECHNICAL_REVIEW: " + base_plan)
    else:
        done = None
    extended = original is not None and original["parameters"].get("max_iter") == 4000
    for attempt in range(1, 3):
        if done is not None:
            break
        is_retry = attempt == 2
        bucket = (
            "retry"
            if is_retry
            else "replay_validation"
            if replay and stage.startswith("V")
            else "validation"
            if stage.startswith("V")
            else "test"
            if stage == "T"
            else "final"
        )
        plan = base_plan + "-" + str(attempt)
        job = {
            "root": str(root),
            "plan": plan,
            "stage": stage,
            "candidate": candidate,
            "x": x,
            "y": y,
            "probe_x": probe_x,
            "columns": cols,
            "train_dates": [r["U"] for r in selected],
            "extended_iterations": extended or is_retry,
            "protocol_sha256": sha(root / "protocol.json"),
            "snapshot_digest": read(root / "protocol.json")["input_digest"],
        }
        # 先消费再落任务文件。即使额度校验失败也不会遗留一个占着下一编号、
        # 却没有 reservation 的 job；预占后若文件写入失败仍按崩溃消费保留。
        job_content = json.dumps(job, ensure_ascii=False, indent=2, default=str, allow_nan=False).encode("utf-8")
        reservation = Budget(root).reserve(
            plan,
            bucket,
            {
                "stage": stage,
                "candidate": candidate,
                "job_sha256": hashlib.sha256(job_content).hexdigest(),
                "training_rows_sha256": fit.digest([r["U"] for r in selected]),
                "input_sha256": fit.digest(x),
                "labels_sha256": fit.digest(y),
                "columns": cols,
                "parameters": {
                    **(fit.H_PARAMS if candidate == "H" else fit.LR_PARAMS),
                    **({"max_iter": 4000} if (extended or is_retry) and candidate != "H" else {}),
                },
                "code_manifest_sha256": sha(root / "code-manifest.json"),
                "retry_reason": "NON_CONVERGED" if is_retry else None,
            },
        )
        number = reservation["number"]
        job_path = root / "fit-attempts" / f"{number:03d}" / "job.json"
        save(job_path, job)
        if sha(job_path) != reservation["job_sha256"]:
            raise RuntimeError("RESERVED_JOB_SERIALIZATION_MISMATCH")
        progress(root, stage, "FITTING", plan, candidate=candidate, replay=replay, training_n=len(selected))
        completed = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "-B",
                "-m",
                "scripts.fund_002112_existing_data_v1",
                "_fit",
                "--job",
                str(job_path),
            ],
            cwd=PY,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=os.environ.copy(),
            timeout=180,
        )
        save(
            job_path.parent / "process.json",
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "finished_at": now(),
            },
        )
        done = job_path.parent / "result.json"
        result = read(done) if done.exists() else {"status": "FAILED"}
        append(
            root / "fit-ledger.jsonl",
            {"event": "FINISHED", "plan": plan, "number": number, "at": now(), "status": result["status"]},
        )
        if result["status"] == "SUCCESS":
            break
        retry_unavailable = False
        if result["status"] == "NON_CONVERGED" and candidate != "H" and not extended and attempt == 1:
            budget = Budget(root).state()
            retry_unavailable = (
                budget["consumed_by_bucket"].get("retry", 0) >= BUCKETS["retry"] or budget["remaining"] <= 0
            )
            if not retry_unavailable:
                done = None
                continue
            # 共享重试耗尽只终止这个分支，不借验证、复现、T或FINAL的配额。
            # fit_stage 对共同 A 仍会报明确阻塞；其他分支由固定备用路径继续。
        failure = {
            "status": "FAILED",
            "candidate": candidate,
            "stage": stage,
            "plan": plan,
            "evidence": str(job_path.parent),
            "training_n": len(selected),
            "reason": "RETRY_BUDGET_EXHAUSTED" if retry_unavailable else result["status"],
            "retry_attempted": attempt == 2,
        }
        save(ref, failure)
        return failure
    result = read(done)
    model_path = done.parent / "model.joblib"
    record = {
        "status": "SUCCESS",
        "candidate": candidate,
        "stage": stage,
        "replay": replay,
        "model_path": str(model_path),
        "model_sha256": sha(model_path),
        "result_path": str(done),
        "training_n": len(selected),
        "training_dates": [r["U"] for r in selected],
        "training_range": [selected[0]["U"], selected[-1]["U"]],
        "columns": cols,
        "fit_cutoff": stage_cutoff(root, stage, rows),
        "class_counts": dict(Counter(y)),
        "unlearnable_classes": result["unlearnable_classes"],
        "parameters": result["params"],
        "zero_variance_columns": [cols[i] for i in result["zero_variance_columns"]],
        "zero_variance_policy": "保留列；仅由训练集标准化为0，scale=1",
        "code_manifest_sha256": sha(root / "code-manifest.json"),
        "input_digest": read(root / "protocol.json")["input_digest"],
        "training_rows_sha256": fit.digest(selected),
        "label_version_sha256": fit.digest(labels),
        "structure_sha256": result["structure_sha256"],
        "dependencies": fit.dependencies(candidate),
        "source_digests": sorted(set(r["source_digest"] for r in selected)),
    }
    if original:
        record["independent_replay"] = fit.compare_models(
            joblib.load(original["model_path"]), joblib.load(model_path), probe_x
        )
        record["independent_replay"]["process_ids"] = [read(original["result_path"])["pid"], result["pid"]]
        if not record["independent_replay"]["passed"]:
            save(ref, {**record, "status": "REPLAY_MISMATCH"})
            raise RuntimeError("INDEPENDENT_REPLAY_MISMATCH")
    save(ref, record)
    return record


def fit_stage(root, stage, candidates, rows, names, *, replay=False, originals=None):
    target_dates = [r["U"] for r in rows if r["U"] <= FOLDS[stage][0]]
    labels = label_access(root, stage, "train", target_dates, rows)
    refs = {}
    for c in candidates:
        if replay and originals[c]["status"] != "SUCCESS":
            continue
        refs[c] = fit_one(root, stage, c, rows, names, labels, replay=replay, original=originals[c] if replay else None)
        if c == "A" and refs[c]["status"] != "SUCCESS":
            raise RuntimeError("COMMON_BASELINE_CANNOT_FIT")
        if replay and refs[c]["status"] != "SUCCESS":
            raise RuntimeError("REPLAY_BRANCH_FAILED")
    return refs


def predictions(rows, names, refs, candidates, stage):
    models = {c: joblib.load(r["model_path"]) for c, r in refs.items() if r["status"] == "SUCCESS"}
    # 同一折同一模型一次批量推理，避免对每个日期重复遍历 BLAS 线程池。
    cached = {}
    for c, model in models.items():
        eligible = [row for row in rows if fit.usable(row, c, names)]
        if eligible:
            matrix = [[row["features"][col] for col in fit.columns(c, names)] for row in eligible]
            values = fit.probabilities(model, matrix)
            cached[c] = {row["U"]: p.tolist() for row, p in zip(eligible, values, strict=True)}
    records = []
    for row in rows:
        for candidate in candidates:
            branch, reason = fit.route(row, candidate, models, names)
            p = None if branch is None else cached[branch][row["U"]]
            records.append(
                {
                    "U": row["U"],
                    "T": row["T"],
                    "S": row["S"],
                    "cutoff": row["cutoff"],
                    "stage": stage,
                    "requested_candidate": candidate,
                    "effective_model": branch,
                    "fallback_reason": reason,
                    "probabilities": p,
                    "predicted": fit.predict_label(p) if p is not None else None,
                    "lag_sessions": row["lag_sessions"],
                    "holdings_available": row["available"]["H"],
                    "input_source_digest": row["source_digest"],
                    "model_sha256": refs[branch]["model_sha256"] if branch else None,
                    "dependency_hashes": {
                        d: refs[d]["model_sha256"] for d in fit.dependencies(candidate) if d in models
                    },
                    "label_mature_at": row["label_mature_at"],
                }
            )
    return records


def save_jsonl(path, records):
    if path.exists():
        if lines(path) != records:
            raise ValueError("IMMUTABLE_PREDICTIONS_CHANGED")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False, default=str, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def train_validation(root):
    verify_frozen(root)
    data, rows, names = inputs(root)
    for stage in ("V1", "V2", "V3"):
        if (root / "checkpoints" / (stage + ".json")).exists():
            require_checkpoint(root, stage)
            continue
        refs = fit_stage(root, stage, list(fit.GROUPS), rows, names)
        start, end = FOLDS[stage][1:]
        target = [r for r in rows if start <= r["U"] <= end]
        path = root / "predictions" / (stage + ".jsonl")
        save_jsonl(path, predictions(target, names, refs, list(fit.GROUPS), stage))
        summary = root / "models" / stage.lower() / "summary.json"
        if not summary.exists():
            save(summary, refs)
        checkpoint(root, stage, [path, summary] + [root / "models" / stage.lower() / (c + ".json") for c in fit.GROUPS])
    checkpoint(root, "train-validation", [root / "checkpoints" / (s + ".json") for s in ("V1", "V2", "V3")])
    progress(root, "VALIDATION_FITS", "DONE", "开放2025验证评价并按冻结规则选择候选")


def scored(records, labels, candidate):
    return [
        {**r, "actual": labels["labels"][r["U"]]}
        for r in records
        if r["requested_candidate"] == candidate and r["predicted"] is not None and r["U"] in labels["labels"]
    ]


def constants(records, train_labels):
    counts = Counter(train_labels["labels"].values())
    majority = max(fit.CLASSES, key=lambda c: counts[c])
    return {
        c: fit.metrics(
            [{**r, "predicted": c if c != "TRAIN_MAJORITY" else majority, "probabilities": None} for r in records]
        )
        for c in (*fit.CLASSES, "TRAIN_MAJORITY")
    }


def compare_all(records, labels, calendar, names):
    all_scored = {c: scored(records, labels, c) for c in fit.GROUPS}
    pairs = {c: fit.paired(all_scored["A"], all_scored[c], calendar) for c in fit.GROUPS}
    coverage = {
        c: {
            "target_n": len(calendar),
            "predicted_n": sum(r["predicted"] is not None for r in records if r["requested_candidate"] == c),
            "scored_n": len(all_scored[c]),
            "effective_models": dict(Counter(r["effective_model"] for r in records if r["requested_candidate"] == c)),
            "fallback_reasons": dict(
                Counter(r["fallback_reason"] for r in records if r["requested_candidate"] == c and r["fallback_reason"])
            ),
        }
        for c in fit.GROUPS
    }
    return {
        "pairs_vs_A": pairs,
        "coverage": coverage,
        "input_counts": {c: len(fit.columns(c, names)) for c in fit.GROUPS},
    }


def replay_path(root, stage, selected, rows, names):
    originals = read(root / "models" / stage.lower() / "summary.json")
    refs = fit_stage(root, stage, fit.dependencies(selected), rows, names, replay=True, originals=originals)
    start, end = FOLDS[stage][1:]
    target = rows if stage == "FINAL" else [r for r in rows if start <= r["U"] <= end]
    main = predictions(target, names, originals, list(dict.fromkeys(["A", selected])), stage)
    replayed = predictions(target, names, refs, list(dict.fromkeys(["A", selected])), stage)
    for a, b in zip(main, replayed, strict=True):
        if a["predicted"] != b["predicted"] or a["effective_model"] != b["effective_model"]:
            raise RuntimeError("PATH_REPLAY_CLASS_OR_ROUTING_MISMATCH")
        if (
            a["probabilities"] is not None
            and max(abs(x - y) for x, y in zip(a["probabilities"], b["probabilities"], strict=True)) > 1e-10
        ):
            raise RuntimeError("PATH_REPLAY_PROBABILITY_MISMATCH")
    path = root / "replay" / ("validation" if stage.startswith("V") else stage.lower()) / stage / "path-check.json"
    if not path.exists():
        save(
            path,
            {
                "passed": True,
                "rows": len(main),
                "selected": selected,
                "model_refs": refs,
                "comparison": "independent process refit, structure + probabilities + classes + route",
            },
        )
    return path


def select_and_replay(root):
    verify_frozen(root)
    require_checkpoint(root, "train-validation")
    data, rows, names = inputs(root)
    selection_path = root / "selection.json"
    if not selection_path.exists():
        combined, labels_all, folds, details = [], {}, {}, {}
        actual_models = set()
        for stage in ("V1", "V2", "V3"):
            records = lines(root / "predictions" / (stage + ".jsonl"))
            dates = sorted(set(r["U"] for r in records))
            labels = label_access(root, stage, "evaluate", dates, rows)
            combined.extend(records)
            labels_all.update(labels["labels"])
            comparison = compare_all(records, labels, dates, names)
            folds[stage] = comparison["pairs_vs_A"]
            details[stage] = comparison
            details[stage]["constants"] = constants(
                scored(records, labels, "A"), read(root / "labels" / (stage.lower() + "-train.json"))
            )
            actual_models.update(
                c
                for c, r in read(root / "models" / stage.lower() / "summary.json").items()
                if r["status"] == "SUCCESS" and c != "A"
            )
        dates = [r["U"] for r in rows if r["U"].startswith("2025")]
        comparison = compare_all(combined, {"labels": labels_all}, dates, names)
        selection = fit.choose_candidate(comparison["pairs_vs_A"], folds, comparison["input_counts"], actual_models)
        comparison["folds"] = details
        save(root / "validation-comparison.json", comparison)
        save(
            selection_path,
            {
                **selection,
                "at": now(),
                "validation_sha256": sha(root / "validation-comparison.json"),
                "protocol_sha256": sha(root / "protocol.json"),
            },
        )
    selection = read(selection_path)
    replay_files = [replay_path(root, s, selection["selected"], rows, names) for s in ("V1", "V2", "V3")]
    checkpoint(root, "select-and-replay", [selection_path, root / "validation-comparison.json", *replay_files])
    progress(
        root,
        "SELECT_AND_REPLAY",
        "DONE",
        "拟合T固定模型并冻结全部2026预测",
        selected=selection["selected"],
        selection_status=selection["status"],
    )


def predict_test(root):
    verify_frozen(root)
    require_checkpoint(root, "select-and-replay")
    if (root / "checkpoints/predict-test.json").exists():
        require_checkpoint(root, "predict-test")
        return
    data, rows, names = inputs(root)
    selected = read(root / "selection.json")["selected"]
    refs = fit_stage(root, "T", fit.dependencies(selected), rows, names)
    summary = root / "models/t/summary.json"
    if not summary.exists():
        save(summary, refs)
    replay = replay_path(root, "T", selected, rows, names)
    records = predictions(
        [r for r in rows if r["U"].startswith("2026")], names, refs, list(dict.fromkeys(["A", selected])), "T"
    )
    path = root / "test-predictions.jsonl"
    save_jsonl(path, records)
    freeze_path = root / "test-predictions-freeze.json"
    if not freeze_path.exists():
        save(
            freeze_path,
            {
                "at": now(),
                "predictions_path": str(path),
                "predictions_sha256": sha(path),
                "predictions_count": len(records),
                "target_dates": sorted(set(r["U"] for r in records)),
                "selection_sha256": sha(root / "selection.json"),
                "models_sha256": sha(summary),
                "model_hashes": {c: r.get("model_sha256") for c, r in refs.items()},
                "independent_replay_sha256": sha(replay),
                "test_labels_accessed": False,
            },
        )
    checkpoint(root, "predict-test", [path, summary, replay, freeze_path])
    progress(
        root,
        "TEST_PREDICTIONS",
        "FROZEN",
        "只在冻结后开放2026检验答案",
        target_n=len(records) // len(set(r["requested_candidate"] for r in records)),
    )


def subgroup_comparisons(base, candidate, calendar):
    """季度、净值滞后及持仓备用组全部报告，不据组内结果重选模型。"""
    groups = {}
    predicates = {
        **{f"Q{q}": (lambda r, q=q: (int(r["U"][5:7]) - 1) // 3 + 1 == q) for q in range(1, 5)},
        "lag_0": lambda r: r["lag_sessions"] == 0,
        "lag_1": lambda r: r["lag_sessions"] == 1,
        "lag_2_5": lambda r: r["lag_sessions"] is not None and 2 <= r["lag_sessions"] <= 5,
        "lag_6_20": lambda r: r["lag_sessions"] is not None and 6 <= r["lag_sessions"] <= 20,
        "holdings_available": lambda r: r["holdings_available"],
        "holdings_missing": lambda r: not r["holdings_available"],
        "own_model": lambda r: r["requested_candidate"] == r["effective_model"],
        "fallback": lambda r: r["requested_candidate"] != r["effective_model"],
    }
    for group, predicate in predicates.items():
        subset = [r for r in candidate if predicate(r)]
        dates = set(r["U"] for r in subset)
        groups[group] = fit.paired([r for r in base if r["U"] in dates], subset, calendar, bootstrap=False)
    return groups


def common_input_subsets(root, rows, names):
    """验证集输入齐全子集的解释性消融，只用于说明新增资料和备用路径。"""
    out = {}
    records = sum([lines(root / "predictions" / (s + ".jsonl")) for s in ("V1", "V2", "V3")], [])
    labels = {
        "labels": {
            k: v
            for s in ("V1", "V2", "V3")
            for k, v in read(root / "labels" / (s.lower() + "-evaluate.json"))["labels"].items()
        }
    }
    calendar = [r["U"] for r in rows if r["U"].startswith("2025")]
    for candidate, base in (("C", "A"), ("D", "B"), ("E", "C"), ("E", "D"), ("F", "E"), ("G", "E"), ("H", "E")):
        dates = {r["U"] for r in rows if fit.usable(r, candidate, names) and fit.usable(r, base, names)}
        a = [r for r in scored(records, labels, base) if r["U"] in dates and r["effective_model"] == base]
        b = [r for r in scored(records, labels, candidate) if r["U"] in dates and r["effective_model"] == candidate]
        out[candidate + "_vs_" + base] = fit.paired(a, b, calendar)
    return out


def actual_prediction_comparison(root, labels, study_records, selected):
    """只在检验开放后读取真实公共方向记录，按原策略与生成时点去重。"""
    from sqlalchemy import URL, create_engine, text

    cfg = {}
    for line in (JAVA / ".env").read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    host = cfg.get("FUND_CORE_DB_HOST", "localhost")
    if host not in ("localhost", "127.0.0.1"):
        raise ValueError("LOCAL_DB_ONLY")
    url = URL.create(
        "postgresql+psycopg",
        username=cfg["FUND_CORE_DB_USERNAME"],
        password=cfg["FUND_CORE_DB_PASSWORD"],
        host=host,
        port=int(cfg.get("FUND_CORE_DB_PORT", "54329")),
        database=cfg.get("FUND_CORE_DB_NAME", "fund_core"),
    )
    engine = create_engine(
        url,
        hide_parameters=True,
        connect_args={
            "connect_timeout": 5,
            "options": "-c default_transaction_read_only=on -c statement_timeout=15000",
        },
    )
    with engine.connect() as connection:
        result = [
            dict(r)
            for r in connection.execute(
                text("""
          select f.forecast_id::text,f.target_nav_date::text,f.base_nav_date::text,f.generated_at::text,
                 f.stored_at::text,f.window_open_at::text,f.deadline_at::text,f.input_hash,
                 s.branch_id,s.model_id::text,s.model_hash,s.predicted_direction,s.train_as_of::text,
                 (f.generated_at < f.deadline_at and f.stored_at < f.deadline_at) as on_time
          from direction_1d_forecast f join direction_1d_forecast_score s using(forecast_id)
          where f.fund_code='002112' and f.target_nav_date between '2026-01-01' and '2026-09-29'
          order by f.target_nav_date,f.generated_at,f.forecast_id,s.branch_id
        """)
            ).mappings()
        ]
    engine.dispose()
    by_strategy = {}
    study = {r["U"]: r for r in study_records if r["requested_candidate"] == selected}
    for row in result:
        if row["predicted_direction"] not in fit.CLASSES or not row["on_time"]:
            continue
        key = row["branch_id"]
        by_strategy.setdefault(key, {})
        # 只取最先真实发出的有效记录，不按事后命中择优。
        by_strategy[key].setdefault(row["target_nav_date"], row)
    comparisons = {}
    for branch, records in by_strategy.items():
        days = sorted(set(records) & set(study) & set(labels["labels"]))
        actual_records = [
            {
                "U": d,
                "predicted": records[d]["predicted_direction"],
                "actual": labels["labels"][d],
                "probabilities": None,
            }
            for d in days
        ]
        study_scored = [{**study[d], "actual": labels["labels"][d]} for d in days if study[d]["predicted"] is not None]
        comparisons[branch] = fit.paired(actual_records, study_scored, read(root / "protocol.json")["calendar"])
    return {
        "records": result,
        "first_on_time_by_strategy": by_strategy,
        "comparisons": comparisons,
        "can_prove_better_than_current_live_model": False,
        "limitation": "新基线A不是现用模型；现用模型训练时间/真实发出记录覆盖与本研究08:00重建时点不同，"
        "有限真实记录仅作单列对照，不能证明优于当前实际运行结果。没有用现用模型回算三年。",
    }


def evaluate_test(root):
    verify_frozen(root)
    require_checkpoint(root, "predict-test")
    if (root / "checkpoints/evaluate-test.json").exists():
        require_checkpoint(root, "evaluate-test")
        return
    data, rows, names = inputs(root)
    records = lines(root / "test-predictions.jsonl")
    dates = [r["U"] for r in rows if r["U"].startswith("2026")]
    labels = label_access(root, "T", "evaluate", dates, rows)
    selection = read(root / "selection.json")
    chosen = selection["selected"]
    a, b = scored(records, labels, "A"), scored(records, labels, chosen)
    pair = fit.paired(a, b, dates)
    groups = subgroup_comparisons(a, b, dates)
    own_paths = {
        c: dict(Counter(r["effective_model"] for r in records if r["requested_candidate"] == c)) for c in ("A", chosen)
    }
    comparison = {
        "at": now(),
        "selected": chosen,
        "pair": pair,
        "subgroups": groups,
        "coverage": {
            "calendar_n": len(dates),
            "mature_labels": len(labels["labels"]),
            "label_exclusions": labels["excluded"],
            "effective_paths": own_paths,
            "prediction_missing": [r for r in records if r["predicted"] is None],
        },
        "constants": constants(a, read(root / "labels/t-train.json")),
        "test_predictions_sha256": sha(root / "test-predictions.jsonl"),
        "label_version_sha256": fit.digest(labels),
        "validation_common_input_subsets": common_input_subsets(root, rows, names),
    }
    path = root / "test-comparison.json"
    if not path.exists():
        save(path, comparison)
    recall_ok = all(
        pair["candidate"]["per_class"][c]["recall"] is not None
        and pair["baseline"]["per_class"][c]["recall"] is not None
        and pair["candidate"]["per_class"][c]["recall"] >= pair["baseline"]["per_class"][c]["recall"] - 0.05
        for c in ("UP", "DOWN")
    )
    if pair["net_correct"] <= 0:
        grade = "没有观察到改善"
    elif (
        selection["status"] == "VALIDATION_GAIN"
        and pair["net_correct"] >= 3
        and pair["accuracy_difference"] >= 0.02
        and recall_ok
        and pair["block_bootstrap_95pct"][0] > 0
    ):
        grade = "历史检验中有较一致的改善"
    else:
        grade = "有数值改善，但证据不稳"
    decision = root / "decision.json"
    if not decision.exists():
        save(
            decision,
            {
                "at": now(),
                "grade": grade,
                "selected": chosen,
                "validation_status": selection["status"],
                "test_net_correct": pair["net_correct"],
                "recall_guard": recall_ok,
                "research_only": True,
                "register_or_activate": False,
            },
        )
    live_path = root / "actual-model-comparison.json"
    if not live_path.exists():
        save(live_path, actual_prediction_comparison(root, labels, records, chosen))
    checkpoint(root, "evaluate-test", [path, decision, live_path])
    progress(
        root,
        "TEST_EVALUATION",
        "DONE",
        "固定结论后拟合FINAL及独立复现",
        grade=grade,
        common_dates=pair["n"],
        baseline_correct=pair["baseline"]["correct"],
        candidate_correct=pair["candidate"]["correct"],
        net_correct=pair["net_correct"],
    )


def fit_final(root):
    verify_frozen(root)
    require_checkpoint(root, "evaluate-test")
    if (root / "checkpoints/fit-final.json").exists():
        require_checkpoint(root, "fit-final")
        return
    data, rows, names = inputs(root)
    selected = read(root / "selection.json")["selected"]
    refs = fit_stage(root, "FINAL", fit.dependencies(selected), rows, names)
    summary = root / "models/final/summary.json"
    if not summary.exists():
        save(summary, refs)
    replay = replay_path(root, "FINAL", selected, rows, names)
    portable_models = {}
    for candidate, model in refs.items():
        if model["status"] != "SUCCESS":
            continue
        destination = root / "models/final" / (candidate + ".joblib")
        if not destination.exists():
            with destination.open("xb") as stream:
                stream.write(Path(model["model_path"]).read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
        if sha(destination) != model["model_sha256"]:
            raise RuntimeError("FINAL_EXPORT_HASH_MISMATCH")
        portable_models[candidate] = {"path": str(destination), "sha256": sha(destination), "columns": model["columns"]}
    final_summary = root / "final-training-summary.json"
    if not final_summary.exists():
        save(
            final_summary,
            {
                "at": now(),
                "selected": selected,
                "dependencies": fit.dependencies(selected),
                "models": refs,
                "portable_models": portable_models,
                "replay": str(replay),
                "registered": False,
                "adopted": False,
                "training_includes_2026": True,
                "training_performance_not_used_as_evidence": True,
                "fit_consumed": Budget(root).state()["consumed"],
                "research_only": True,
            },
        )
    checkpoint(
        root, "fit-final", [summary, replay, final_summary, *[Path(p["path"]) for p in portable_models.values()]]
    )
    progress(root, "FINAL", "DONE", "核验保护、逐日账本与最终报告", selected=selected)


def date_ledger(root, rows, names):
    models = {s: read(root / "models" / s.lower() / "summary.json") for s in FOLDS}
    labels = {s: read(root / "labels" / (s.lower() + "-train.json")) for s in FOLDS}
    ledger = []
    for r in rows:
        stage = next((s for s in ("V1", "V2", "V3", "T") if FOLDS[s][1] <= r["U"] <= FOLDS[s][2]), "INITIAL_2024")
        trained = {
            s: {c: r["U"] in model.get("training_dates", []) for c, model in refs.items()} for s, refs in models.items()
        }
        ledger.append(
            {
                **r,
                "evaluation_stage": stage,
                "actual_training_inclusion": trained,
                "label_mature_by_fit": {s: r["U"] in lab["labels"] for s, lab in labels.items()},
                "candidate_input_usable": {c: fit.usable(r, c, names) for c in fit.GROUPS},
            }
        )
    save_jsonl(root / "date-ledger.jsonl", ledger)
    return ledger


def protection_after(root):
    before = read(root / "protection-before.json")
    changed = [p for p, h in before["protected"].items() if not Path(p).is_file() or sha(p) != h]
    runtime = runtime_evidence()
    same_models = runtime["ai"]["models"] == before["runtime"]["ai"]["models"]
    same_files = runtime["ai"]["model_files"] == before["runtime"]["ai"]["model_files"]
    same_bindings = runtime["ai"]["bindings"] == before["runtime"]["ai"]["bindings"]
    old = before["runtime"]["core"]
    new = runtime["core"]
    predictions_ok = all(r in new["predictions"] for r in old["predictions"])
    directions_ok = all(r in new["directions"] for r in old["directions"])
    parallel = [
        {"path": p, "before": h, "after": sha(p) if Path(p).is_file() else None}
        for p, h in before["dirty_files"].items()
        if not Path(p).is_file() or sha(p) != h
    ]
    heads = {
        r["path"]: subprocess.check_output(["git", "-C", r["path"], "rev-parse", "HEAD"], text=True).strip()
        == r["head"]
        for r in before["repositories"]
    }
    evidence = {
        "at": now(),
        "protected_checked": len(before["protected"]),
        "protected_changes": changed,
        "models_equal": same_models,
        "model_files_equal": same_files,
        "bindings_equal": same_bindings,
        "old_predictions_preserved": predictions_ok,
        "old_directions_preserved": directions_ok,
        "old_prediction_count": len(old["predictions"]),
        "current_prediction_count": len(new["predictions"]),
        "parallel_workspace_changes": parallel,
        "heads_unchanged": heads,
        "runtime": runtime,
        "passed": not changed
        and same_models
        and same_files
        and same_bindings
        and predictions_ok
        and directions_ok
        and all(heads.values()),
    }
    save(root / "protection-after.json", evidence)
    return evidence


def verify(root):
    verify_frozen(root)
    require_checkpoint(root, "fit-final")
    data, rows, names = inputs(root)
    ledger = date_ledger(root, rows, names)
    protection = (
        read(root / "protection-after.json") if (root / "protection-after.json").exists() else protection_after(root)
    )
    budget = Budget(root).state()
    completed = [r for r in lines(root / "fit-ledger.jsonl") if r["event"] == "FINISHED"]
    for stage in FOLDS:
        for ref in read(root / "models" / stage.lower() / "summary.json").values():
            if ref["status"] == "SUCCESS" and sha(ref["model_path"]) != ref["model_sha256"]:
                raise ValueError("MODEL_PROTECTION_FAILURE")
    stage_counts = {}
    for s in FOLDS:
        labels = read(root / "labels" / (s.lower() + "-train.json"))
        stage_counts[s] = {
            "target_n": sum(r["U"] <= FOLDS[s][0] for r in rows),
            "mature_labels": len(labels["labels"]),
            "classes": dict(Counter(labels["labels"].values())),
            "label_exclusions": labels["excluded"],
            "input_available": {
                c: sum(fit.usable(r, c, names) for r in rows if r["U"] <= FOLDS[s][0]) for c in fit.GROUPS
            },
            "actual_training": {
                c: r["training_n"] for c, r in read(root / "models" / s.lower() / "summary.json").items()
            },
        }
    passed = (
        len(ledger) == 665
        and protection["passed"]
        and budget["consumed"] <= 48
        and len(completed) == budget["consumed"]
    )
    acceptance = {
        "at": now(),
        "passed": passed,
        "date_ledger_rows": len(ledger),
        "stage_counts": stage_counts,
        "budget": budget,
        "protection_passed": protection["passed"],
        "research_complete": passed,
        "registered": False,
        "adopted": False,
        "services_changed_by_experiment": False,
    }
    if not (root / "acceptance.json").exists():
        save(root / "acceptance.json", acceptance)
    if not passed:
        raise RuntimeError("FINAL_ACCEPTANCE_FAILED")
    checkpoint(root, "verify", [root / "acceptance.json", root / "date-ledger.jsonl", root / "protection-after.json"])
    progress(root, "VERIFY", "DONE", "写最终中文结论", real_fit_total=budget["consumed"])


def report(root):
    require_checkpoint(root, "verify")
    result_path = root / "result.md"
    if result_path.exists():
        return
    test = read(root / "test-comparison.json")
    pair = test["pair"]
    decision = read(root / "decision.json")
    selected = read(root / "selection.json")["selected"]
    validation = read(root / "validation-comparison.json")["pairs_vs_A"][selected]
    final = read(root / "final-training-summary.json")
    accepted = read(root / "acceptance.json")
    data, rows, names = inputs(root)
    ci = pair["block_bootstrap_95pct"]
    text = [
        "# 002112 一日方向独立训练实验结果",
        "",
        f"完成时间：{now()}。实验：{EXPERIMENT}。",
        "",
        f"**{decision['grade']}。** 在 2026 年相同的 {pair['n']} 个日期中，净值基线 A 判对 "
        f"{pair['baseline']['correct']} 天（{pair['baseline']['accuracy']:.2%}），固定增强候选 {selected} 判对 "
        f"{pair['candidate']['correct']} 天（{pair['candidate']['accuracy']:.2%}），净差 {pair['net_correct']:+d} 天。",
        "",
        f"2025 三折共同 {validation['n']} 天，A 判对 {validation['baseline']['correct']} 天，"
        f"{selected} 判对 {validation['candidate']['correct']} 天，净差 {validation['net_correct']:+d} 天；"
        f"预定选择结果为 {decision['validation_status']}。没有按 2026 结果换候选。",
        "",
        f"2026 配对四格：都对 {pair['paired_cells']['both_correct']}，"
        f"只增强对 {pair['paired_cells']['candidate_only']}，"
        f"只 A 对 {pair['paired_cells']['baseline_only']}，都错 {pair['paired_cells']['both_wrong']}。"
        f"5 个交易日块、2,000 次重采样的正确率差 95% 区间为 [{ci[0]:.2%}, {ci[1]:.2%}]。",
        "",
        "| 实际类别 | 样本数 | A 判对/召回率 | 增强判对/召回率 |",
        "| --- | ---: | --- | --- |",
    ]
    for cls, label in (("UP", "上涨"), ("DOWN", "下跌"), ("FLAT", "持平")):
        a, b = pair["baseline"]["per_class"][cls], pair["candidate"]["per_class"][cls]
        ar = "无样本" if a["recall"] is None else f"{a['recall']:.2%}"
        br = "无样本" if b["recall"] is None else f"{b['recall']:.2%}"
        text.append(f"| {label} | {a['n']} | {a['correct']} / {ar} | {b['correct']} / {br} |")
    text.extend(
        [
            "",
            "持平少样本仅保留真实类别，未造样本；训练未出现的类概率槽为 0 并注明不可学习。",
            "",
            "冻结范围的 665 个目标日全部有账本。净值、沪深300/中证500和五条行业背景按当时可得时间使用；"
            "各组可用天数："
            + "，".join(f"{g}={sum(r['available'].get(g, False) for r in rows)}" for g in names)
            + "。",
            "停牌窗口只使持仓本体不可用，按固定 A/D 备用路径继续。7 个季末净值保留原披露日期，"
            "通过较早的连续窗口与滞后特征处理，没有提前净值版本时间。",
            "",
            "可选资料决定：",
            "```json",
            json.dumps(data["optional_decisions"], ensure_ascii=False, indent=2),
            "```",
            "",
            f"全部真实监督拟合 {accepted['budget']['consumed']}/48 次，含独立进程重拟合、失败及最终拟合；"
            "人工样例调用另见 synthetic-fit-ledger。独立复现检查结构、标准化参数、概率和分类，概率容差 1e-10。",
            "",
            "FINAL 研究模型及备用依赖：",
        ]
    )
    for c, model in final["models"].items():
        location = final.get("portable_models", {}).get(c, {}).get("path", model.get("model_path", "未形成"))
        text.append(f"- {c}：{model['status']}，实际训练 {model['training_n']} 天；文件 `{location}`。")
    text.extend(
        [
            "",
            "FINAL 已使用 2026 成熟标签，其训练内结果不再充当效果证据。"
            "当前历史查询版本不能完全还原当年逐时版本，所以这些结果属于按可得时间约束的历史重建研究。",
            "",
            "现用模型、绑定及旧预测保持原样。A 是本次新训练的净值基线；真实已发出记录按原 FIXED/WEEKLY "
            "等策略单列在 actual-model-comparison.json。生成时点与本研究08:00不同且共同记录有限，"
            "尚不能证明优于当前实际运行结果。没有登记、换模、服务启停、数据库写入、提交、推送或部署。",
            "",
            "逐折、逐季度、净值滞后、持仓缺失和备用使用、常量对照、完整混淆矩阵和输入齐全子集分别保存在 "
            "validation-comparison.json、test-comparison.json、date-ledger.jsonl；保护验收见 acceptance.json。",
        ]
    )
    result_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    checkpoint(root, "report", [result_path])
    progress(
        root,
        "COMPLETE",
        "DONE",
        "交协调者最终验收",
        grade=decision["grade"],
        baseline_correct=pair["baseline"]["correct"],
        candidate_correct=pair["candidate"]["correct"],
        common_dates=pair["n"],
        net_correct=pair["net_correct"],
        report=str(result_path),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "freeze",
            "supersede-freeze",
            "train-validation",
            "select-and-replay",
            "predict-test",
            "evaluate-test",
            "fit-final",
            "verify",
            "report",
            "status",
            "resume",
            "_fit",
        ],
    )
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.command == "_fit":
        worker(args.job)
        return
    if args.command == "status":
        print(
            json.dumps(
                {"budget": Budget(ROOT).state(), "progress": read(ROOT / "handoff/training/progress.json")},
                ensure_ascii=False,
            )
        )
        return
    with lock(ROOT):
        try:
            functions = {
                "prepare": prepare,
                "freeze": freeze,
                "supersede-freeze": supersede_freeze,
                "train-validation": train_validation,
                "select-and-replay": select_and_replay,
                "predict-test": predict_test,
                "evaluate-test": evaluate_test,
                "fit-final": fit_final,
                "verify": verify,
                "report": report,
            }
            if args.command == "resume":
                for command, function in functions.items():
                    if command in ("prepare", "freeze", "supersede-freeze"):
                        continue
                    if (ROOT / "checkpoints" / (command + ".json")).exists():
                        require_checkpoint(ROOT, command)
                        continue
                    function(ROOT)
            else:
                functions[args.command](ROOT)
        except Exception as exc:
            progress(ROOT, args.command, "STOPPED", str(exc), error_type=type(exc).__name__)
            raise


if __name__ == "__main__":
    main()
