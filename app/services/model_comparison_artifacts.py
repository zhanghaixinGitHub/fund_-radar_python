"""独占创建、冻结指纹与有界读取；研究包永不覆盖成功结果。"""

import hashlib
import json
from pathlib import Path


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def write_jsonl(path: Path, values) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("JSON artifact too large")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("JSONL artifact too large")
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= 10000 or len(line) > 512 * 1024:
                raise ValueError("JSONL row budget exceeded")
            yield json.loads(line)


def verify_files(folder: Path, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()) or file_hash(path) != expected:
            raise ValueError(f"artifact integrity mismatch: {relative}")
