"""自训练研究包：独占创建、有界读取、摘要和可复核阶段记录。"""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / ".local-runs"
MAX_JSON = 8 * 1024**2
MAX_JSONL = 128 * 1024**2
MAX_ROWS = 10000


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def run_folder(run_id: UUID, *, prefix="direction-training") -> Path:
    if prefix not in ("direction-training", "direction-inventory") or not isinstance(run_id, UUID):
        raise ValueError("INVALID_RUN_SELECTOR")
    folder = RUN_ROOT / f"{prefix}-{run_id}"
    if folder.resolve().parent != RUN_ROOT.resolve() or folder.is_symlink():
        raise ValueError("RUN_PATH_ESCAPED")
    return folder


def new_folder(*, prefix="direction-training") -> Path:
    RUN_ROOT.mkdir(exist_ok=True)
    folder = run_folder(uuid4(), prefix=prefix)
    folder.mkdir(exist_ok=False)
    return folder


def write_json(path: Path, value) -> str:
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(content) > MAX_JSON:
        raise ValueError("JSON_TOO_LARGE")
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return file_hash(path)


def read_json(path: Path):
    with path.open("rb") as stream:
        content = stream.read(MAX_JSON + 1)
    if len(content) > MAX_JSON:
        raise ValueError("JSON_TOO_LARGE")
    return json.loads(content)


def write_jsonl(path: Path, rows) -> str:
    size = 0
    with path.open("xb") as stream:
        for i, row in enumerate(rows):
            content = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            size += len(content)
            if i >= MAX_ROWS or len(content) > 512 * 1024 or size > MAX_JSONL:
                raise ValueError("JSONL_BUDGET_EXCEEDED")
            stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return file_hash(path)


def read_jsonl(path: Path) -> list[dict]:
    if path.stat().st_size > MAX_JSONL:
        raise ValueError("JSONL_TOO_LARGE")
    rows = []
    with path.open("rb") as stream:
        while content := stream.readline(512 * 1024 + 1):
            if len(content) > 512 * 1024 or len(rows) >= MAX_ROWS:
                raise ValueError("JSONL_BUDGET_EXCEEDED")
            rows.append(json.loads(content))
    return rows


def verify_files(folder: Path, files: dict[str, str]) -> None:
    if not files or len(files) > 100:
        raise ValueError("INVALID_FILE_MANIFEST")
    for name, expected in files.items():
        target = folder / name
        if target.resolve().parent != folder.resolve() or target.is_symlink():
            raise ValueError("ARTIFACT_PATH_ESCAPED")
        if file_hash(target) != expected:
            raise ValueError("ARTIFACT_HASH_MISMATCH")


def seal(folder: Path, name: str, files: dict[str, str], **metadata) -> dict:
    verify_files(folder, files)
    result = {"created_at": now(), **metadata, "files": files}
    result["manifest_hash"] = digest(result)
    write_json(folder / name, result)
    return result


def read_seal(folder: Path, name: str) -> dict:
    result = read_json(folder / name)
    if result.get("manifest_hash") != digest({k: v for k, v in result.items() if k != "manifest_hash"}):
        raise ValueError("MANIFEST_HASH_MISMATCH")
    verify_files(folder, result["files"])
    return result
