"""002112 既有事件实验的只读来源边界；所有训练读取本轮内容寻址副本。"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PY = Path(__file__).resolve().parents[1]
WEB = Path("C:/WebStormProject/workSpace05")
JAVA = Path("C:/ideaProject/workSpace12")
RESEARCH = PY / ".local-runs/fund-exposure-002112"
ROOT = RESEARCH / "existing-events-experiment/20260930-v1"
COORD = WEB / ".local-runs/coordination/20260930-002112-and-narrative/event-training"
DOC = WEB / "docs_zhx/implementation/fund-002112-existing-events-training-plan-2026-09-30.md"
DOC_SHA = "fe3baee2d1a5a6a2ce79d812ce8506abf0db01c0290ff8448b21bce148f04022"
EXPERIMENT = "FUND_002112_EXISTING_EVENTS_20260930_V1"
ZONE = ZoneInfo("Asia/Shanghai")
ENTRIES = {
    "historical": RESEARCH / "data-completion/20260930-v1/historical-acceptance-final-v3.json",
    "public": RESEARCH / "data-completion/20260930-v1/public-acceptance-final-v3.json",
    "catalogs": RESEARCH / "closure/20260929-v1/public-catalogs/result.json",
    "news": RESEARCH / "closure/20260929-v1/public-news-current-snapshot/result.json",
    "materials": PY / "data/fund-materials/002112.json",
    "ready": RESEARCH / "training-ready/ready.json",
    "recent_nav": RESEARCH / "existing-data-experiment/20260930-v1/snapshot/nav-by-date.json",
}
CALENDARS = ["cn_a_share_2015_2020_research_v1.json", "cn_a_share_2021_2025_v1.json", "cn_a_share_2026_v1.json"]


def now():
    return datetime.now(ZONE).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def payload(value):
    """现成 hash/payload 包必须校验，不能只相信显示出来的字段。"""
    if "payload" in value and "hash" in value:
        if digest(value["payload"]) != value["hash"]:
            raise ValueError("PAYLOAD_HASH_MISMATCH")
        return value["payload"]
    return value


def save(path, value):
    """不可变证据相同内容可复用；不同内容不能被断点恢复静默覆盖。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if canonical(read(path)) != canonical(value):
            raise ValueError(f"IMMUTABLE_CONFLICT: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def replace(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(canonical(value) + "\n", "utf-8")
    os.replace(temp, path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text("utf-8").splitlines()] if Path(path).exists() else []


def save_lines(path, rows):
    path = Path(path)
    data = "".join(canonical(r) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text("utf-8") != data:
            raise ValueError(f"IMMUTABLE_CONFLICT: {path}")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(data)


def experiment_root(root):
    """修订共享原实验预算与内核锁；修订目录不能成为重领预算的新实验。"""
    root = Path(root).resolve()
    if (root / "revision.json").exists():
        parent = Path(read(root / "revision.json")["experiment_root"]).resolve()
        if not root.is_relative_to(parent) or root == parent:
            raise ValueError("REVISION_OUTSIDE_ORIGINAL_EXPERIMENT")
        return parent
    return root


def active_root(root):
    root = Path(root).resolve()
    if (root / "active-revision.json").exists():
        selected = (root / read(root / "active-revision.json")["path"]).resolve()
        if experiment_root(selected) != root:
            raise ValueError("ACTIVE_REVISION_IDENTITY_INVALID")
        return selected
    return root


@contextmanager
def writer_lock(root=ROOT):
    """Windows 内核字节锁；进程结束自动释放，旧 PID 文本不是删锁依据。"""
    import ctypes
    import msvcrt

    root = experiment_root(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            # PID可能复用，额外保留Windows进程创建FILETIME作为启动身份。
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            times = [ctypes.c_uint64() for _ in range(4)]
            kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
            ok = kernel.GetProcessTimes(kernel.GetCurrentProcess(), *(ctypes.byref(t) for t in times))
            if not ok:
                raise OSError("PROCESS_START_IDENTITY_UNAVAILABLE")
            replace(
                root / "lock-owner.json",
                {
                    "pid": os.getpid(),
                    "lock_acquired_at": now(),
                    "process_created_filetime": times[0].value,
                    "experiment": EXPERIMENT,
                },
            )
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def offline_guard():
    """离线阶段阻断 Python 网络入口，任何尝试都失败；数据库快照在独立明确入口执行。"""
    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append("NETWORK_ATTEMPT")
        raise RuntimeError("EXTERNAL_NETWORK_BUDGET_ZERO")

    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = blocked
    try:
        yield attempts
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo = originals


class Snapshot:
    """只遍历规定入口和实际使用的直接引用，记录每个读取时间与原始字节摘要。"""

    def __init__(self, root=ROOT):
        self.root = root
        self.records = {r["source_path"]: r for r in lines(root / "source-journal.jsonl")}

    def copy(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key in self.records:
            item = self.records[key]
            if expected and expected != item["sha256"]:
                raise ValueError("REFERENCE_HASH_CONFLICT")
            target = self.root / item["snapshot_path"]
            if sha(target) != item["sha256"]:
                raise ValueError("SNAPSHOT_CORRUPTED")
            return target
        data = path.read_bytes()
        hashed = hashlib.sha256(data).hexdigest()
        if expected and hashed != expected:
            raise ValueError(f"SOURCE_HASH_MISMATCH: {path}")
        if sha(path) != hashed:
            raise ValueError(f"SOURCE_CHANGED_DURING_FREEZE: {path}")
        target = self.root / "snapshot" / (hashed + ".bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        item = {
            "source_path": key,
            "sha256": hashed,
            "bytes": len(data),
            "read_at": now(),
            "snapshot_path": target.relative_to(self.root).as_posix(),
        }
        append(self.root / "source-journal.jsonl", item)
        self.records[key] = item
        return target

    def get(self, path, expected=None):
        return read(self.copy(path, expected))


def database_snapshot(root):
    """一次 REPEATABLE READ / READ ONLY 快照，有限字段、稳定分页与30秒超时，无 ORM 初始化。"""
    out = root / "snapshot/database.json"
    if out.exists():
        return read(out)
    from app.core.config import get_settings
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    url = get_settings().ai_database_url
    if make_url(url).host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("DATABASE_NOT_LOCAL_READONLY_SCOPE")
    queries = {
        "news_item": (
            "n.published_at, n.news_id",
            "n.news_id,n.title,n.url,n.summary,n.published_at,n.content_hash",
            "news_item n",
            "n.published_at >= '2016-01-01' AND n.published_at < '2026-09-30'",
        ),
        "news_source_reference": (
            "r.published_at,r.reference_id",
            "r.reference_id,r.news_id,r.url,r.published_at",
            "news_source_reference r",
            "r.published_at >= '2016-01-01' AND r.published_at < '2026-09-30'",
        ),
        "market_event": (
            "m.published_at,m.event_id",
            "m.event_id,m.news_id,m.event_type,m.published_at",
            "market_event m",
            "m.published_at >= '2016-01-01' AND m.published_at < '2026-09-30'",
        ),
        "event_relation": (
            "r.event_id,r.relation_id",
            "r.event_id,r.entity_type,r.entity_id",
            "event_relation r JOIN market_event m ON m.event_id=r.event_id",
            "m.published_at >= '2016-01-01' AND m.published_at < '2026-09-30'",
        ),
        "news_research_card": (
            "published_date,bundle_hash,sample_id",
            "bundle_hash,sample_id,published_date,announcement_id,stock_code,record_payload",
            "news_research_card",
            "fund_code='002112' AND published_date >= '2016-01-01' AND published_date < '2026-09-30'",
        ),
    }
    result = {"at": now(), "readonly": True, "external_requests": 0, "tables": {}, "status": "COMPLETE"}
    engine = create_engine(url, connect_args={"connect_timeout": 5})
    try:
        with engine.connect() as conn, conn.begin():
            conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
            conn.execute(text("SET LOCAL statement_timeout='30s'"))
            for name, (order, columns, table, where) in queries.items():
                rows = []
                for offset in range(0, 200001, 1000):
                    page = conn.execute(
                        text(f"SELECT {columns} FROM {table} WHERE {where} ORDER BY {order} LIMIT 1000 OFFSET :offset"),
                        {"offset": offset},
                    )
                    page = [dict(x) for x in page.mappings()]
                    if offset == 200000 and page:
                        raise ValueError("READONLY_SNAPSHOT_LIMIT_REACHED")
                    rows.extend(page)
                    if len(page) < 1000:
                        break
                result["tables"][name] = rows
    except Exception as exc:
        # 不输出可能带 URL/凭据的异常原文；连接失败也不能被解释为表中0条。
        result["status"] = "UNAVAILABLE"
        result["error_type"] = type(exc).__name__
    finally:
        engine.dispose()
    save(out, result)
    return result


def prepare(root=ROOT):
    if (root / "source-inventory.json").exists():
        return read(root / "source-inventory.json")
    if sha(DOC) != DOC_SHA:
        raise ValueError("AUTHORIZED_DOCUMENT_CHANGED")
    if not (root / "identity.json").exists():
        save(
            root / "identity.json",
            {
                "experiment": EXPERIMENT,
                "created_at": now(),
                "doc_sha256": DOC_SHA,
                "fit_limit": 28,
                "external_request_limit": 0,
            },
        )
    elif read(root / "identity.json")["experiment"] != EXPERIMENT:
        raise ValueError("EXPERIMENT_IDENTITY_CONFLICT")
    if not (root / "protection-before.json").exists():
        repositories = {}
        for repo in (PY, WEB, JAVA):
            status = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain=v1"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            ).stdout
            repositories[str(repo)] = status
        old = RESEARCH / "existing-data-experiment/20260930-v1"
        paths = [old / x for x in ["protocol.json", "fit-ledger.jsonl", "budget.json", "result.md", "acceptance.json"]]
        paths += list((old / "models").rglob("*.joblib"))
        paths += list((PY / "data/prediction-models").glob("*.json"))
        save(
            root / "protection-before.json",
            {"at": now(), "repositories": repositories, "protected": {str(p): sha(p) for p in paths}},
        )
    snap = Snapshot(root)
    entry_paths = {k: str(v) for k, v in ENTRIES.items()}
    for p in ENTRIES.values():
        snap.copy(p)
    ready = payload(snap.get(ENTRIES["ready"]))
    package = payload(snap.get(RESEARCH / "training-ready" / ready["file"]))
    # 只使用来源包中的目标份额净值和持仓；旧包的标签不传入特征适配器。
    entry_paths["historical_facts"] = str(RESEARCH / "training-ready" / package["source_file"])
    snap.copy(entry_paths["historical_facts"])
    for name in CALENDARS:
        entry_paths[name] = str(PY / "app/data/calendars" / name)
        snap.copy(entry_paths[name])
    for key, fields in [
        ("historical", ("source_record", "literal_evidence")),
        ("public", ("source", "semantic_source")),
    ]:
        for row in snap.get(ENTRIES[key])["rows"]:
            for field in fields:
                ref = row.get(field)
                if ref:
                    snap.copy(ref["path"], ref.get("sha256"))
    db = database_snapshot(root)
    inventory = {
        "experiment": EXPERIMENT,
        "frozen_at": now(),
        "entries": entry_paths,
        "sources": list(snap.records.values()),
        "database_status": db["status"],
        "database_sha256": sha(root / "snapshot/database.json"),
        "external_requests": 0,
        "scope": "指定入口及所用直接文字引用；不遍历其他资料目录",
    }
    save(root / "source-inventory.json", inventory)
    return inventory


class Frozen:
    def __init__(self, root=ROOT):
        self.root = experiment_root(root)
        self.inventory = read(root / "source-inventory.json")
        self.records = {r["source_path"]: r for r in self.inventory["sources"]}

    def get(self, path):
        key = str(Path(path).resolve())
        record = self.records[key]
        p = self.root / record["snapshot_path"]
        if sha(p) != record["sha256"]:
            raise ValueError("FROZEN_SOURCE_CORRUPTED")
        return read(p)

    def entry(self, key):
        return self.get(self.inventory["entries"][key])
