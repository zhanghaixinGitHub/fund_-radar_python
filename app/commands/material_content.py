"""原公告/报告迁移：分批提取入库，独立验证后清理，失败原件原地保留。

python -m app.commands.material_content inventory|ingest|verify|delete --run PATH
迁移清单、错误和实际释放空间分别留存；每一步均可中断后恢复。
"""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.models.material_content import MaterialLocation
from app.services.material_content import PROJECT, delete_verified, repository


def inventory(directory):
    """只枚举业务资料；跳过代码、测试替身、环境、模型二进制和状态文件。"""
    rows = []
    for name in ("data", ".local-runs"):
        stack = [PROJECT / name]
        while stack:
            with os.scandir(stack.pop()) as entries:
                for item in entries:
                    path = Path(item.path)
                    if item.is_symlink():
                        continue
                    if item.is_dir(follow_symlinks=False):
                        if item.name not in {"tests", "fixtures", "node_modules", ".git", ".venv", "__pycache__"}:
                            stack.append(path)
                        continue
                    if not item.is_file(follow_symlinks=False):
                        continue
                    original_directory = any(p in {"sources", "raw", "company-sources"} for p in path.parts)
                    document = path.suffix.lower() in {".pdf", ".docx"}
                    structured = original_directory and path.suffix.lower() in {
                        ".json",
                        ".html",
                        ".htm",
                        ".txt",
                        ".js",
                        ".csv",
                        ".md",
                        ".markdown",
                    }
                    if document or structured:
                        rows.append({"path": str(path.resolve()), "bytes": item.stat().st_size})
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "inventory.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"files": len(rows), "logical_bytes": sum(r["bytes"] for r in rows)}


def ingest_one(path):
    """每个子进程独立建连接池；不在工作进程中删除或改动原件。"""
    import logging

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        value = repository().ingest(Path(path), metadata_only=True)
        return {
            "path": path,
            "source_hash": value.source_hash,
            "content_hash": value.content_hash,
            "complete": value.complete,
            "format": value.kind,
            "issue": value.issue,
        }
    except Exception as error:
        # 不输出原始响应或数据库参数；错误信息只保留类型和短原因码。
        return {"path": path, "error": type(error).__name__, "reason": (str(error).splitlines() or [""])[0][:160]}


def review_one(path):
    """独立文档内容验收，不读取评分表，也不在批量复核中删除存量原件。"""
    import logging

    from app.services.material_document import POLICY, review_source

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        review = review_source(Path(path))
        return {
            "path": path,
            "source_hash": review["source_hash"],
            "policy": POLICY,
            "ready": review["ready"],
            "document_hash": review["document_hash"],
            "issues": review["issues"],
        }
    except Exception as error:
        return {
            "path": path,
            "policy": POLICY,
            "error": type(error).__name__,
            "reason": (str(error).splitlines() or [""])[0][:160],
        }


def review_content(directory, workers, limit=None, input_path=None):
    """带检查点的内容验收；成功与待补提分别记账，不输出全文或复制原文件。"""
    from collections import Counter

    from app.services.material_document import POLICY

    journal = directory / "content-reviews.jsonl"
    completed = set()
    if journal.exists():
        for line in journal.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("policy") == POLICY and not row.get("error"):
                completed.add(row["path"])
    paths = []
    with (input_path or directory / "inventory.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            name = json.loads(line)["path"]
            if name not in completed and not any(p.lower().startswith(("test", "fixture")) for p in Path(name).parts):
                paths.append(name)
                if limit is not None and len(paths) >= limit:
                    break
    count = ready = errors = 0
    issues = Counter()
    start = last = time.monotonic()
    with journal.open("a", encoding="utf-8", buffering=1) as output, ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = iter(paths)
        pending = set()

        def submit():
            for _ in range(workers * 2 - len(pending)):
                name = next(iterator, None)
                if name is None:
                    break
                pending.add(pool.submit(review_one, name))

        submit()
        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                ready += bool(row.get("ready"))
                errors += bool(row.get("error"))
                issues.update(i["code"] for i in row.get("issues", []))
            submit()
            if time.monotonic() - last >= 20:
                print(
                    json.dumps(
                        {
                            "reviewed": count,
                            "ready": ready,
                            "pending": len(paths) - count,
                            "errors": errors,
                            "seconds": round(time.monotonic() - start),
                        }
                    ),
                    flush=True,
                )
                last = time.monotonic()
    return {
        "reviewed": count,
        "ready": ready,
        "needs_review": count - ready - errors,
        "errors": errors,
        "issues": dict(issues),
        "purpose": "GENERAL_DOCUMENT_CONTENT_REVIEW",
        "deleted_originals": 0,
    }


def ingest(directory, workers, limit=None, input_path=None):
    completed_path = directory / "ingested.jsonl"
    completed = set()
    if completed_path.exists():
        with completed_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if not row.get("error"):
                    completed.add(row["path"])
                else:
                    completed.discard(row["path"])
    with (input_path or directory / "inventory.jsonl").open(encoding="utf-8") as stream:
        candidates = (json.loads(line)["path"] for line in stream)
        paths = [
            p
            for p in candidates
            if p not in completed and not any(part.lower().startswith(("test", "fixture")) for part in Path(p).parts)
        ]
    if limit is not None:
        paths = paths[:limit]
    done, failed, incomplete = 0, 0, 0
    started, last = time.monotonic(), time.monotonic()
    with (
        completed_path.open("a", encoding="utf-8", buffering=1) as output,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        pending, iterator = {}, iter(paths)

        def submit():
            for _ in range(workers * 2 - len(pending)):
                path = next(iterator, None)
                if path is None:
                    return
                pending[pool.submit(ingest_one, path)] = path

        submit()
        while pending:
            ready, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in ready:
                path = pending.pop(future)
                try:
                    row = future.result()
                except Exception as error:
                    row = {"path": path, "error": type(error).__name__}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                done += 1
                failed += bool(row.get("error"))
                incomplete += not row.get("error") and not row.get("complete")
            submit()
            if time.monotonic() - last >= 20:
                print(
                    json.dumps(
                        {
                            "processed": done,
                            "pending_total": len(paths) - done,
                            "errors": failed,
                            "incomplete": incomplete,
                            "seconds": round(time.monotonic() - started),
                        }
                    ),
                    flush=True,
                )
                last = time.monotonic()
    return {"processed": done, "errors": failed, "incomplete": incomplete}


def delete(directory):
    from collections import Counter

    from app.services.material_content import cleanup_on_hold
    from app.services.material_migration_validation import validated_proof

    if cleanup_on_hold():
        raise ValueError("MATERIAL_DELETE_ON_USER_HOLD")

    proof = validated_proof(directory)
    if (
        proof["evidence"].get("purpose") != "EXTRACTION_QUALITY_REVIEW"
        or proof["evidence"].get("authorizes_deletion") is not True
    ):
        raise ValueError("MATERIAL_INDEPENDENT_EXTRACTION_REVIEW_REQUIRED")
    identity = proof["verification"]
    repo = repository()
    before = shutil.disk_usage(PROJECT.anchor).free
    count, total, errors = 0, 0, 0
    retained = Counter()
    started = last = time.monotonic()
    with (directory / "deleted.jsonl").open("a", encoding="utf-8", buffering=1) as output:
        with repo.engine.connect() as connection:
            rows = connection.execute(
                select(MaterialLocation.location).where(
                    MaterialLocation.verification == identity, MaterialLocation.deleted_at.is_(None)
                )
            ).yield_per(100)
            for (name,) in rows:
                path = Path(name)
                try:
                    size = delete_verified(path, root=PROJECT, verification=identity, repo=repo)
                    count += bool(size)
                    total += size
                    row = {"path": name, "bytes": size, "deleted": bool(size), "already_absent": not size}
                except Exception as error:
                    reason = (str(error).splitlines() or [""])[0][:120]
                    expected = isinstance(error, ValueError) and reason in {
                        "MATERIAL_DELETE_EXTRACTION_INCOMPLETE",
                        "MATERIAL_DELETE_DOCX_VISUALS_INCOMPLETE",
                        "MATERIAL_DELETE_UNEXTRACTED_IMAGE_OR_TEXT",
                    }
                    if expected:
                        retained[reason] += 1
                    else:
                        errors += 1
                    row = {"path": name, "error": type(error).__name__, "reason": reason, "retained": expected}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                if time.monotonic() - last >= 20:
                    print(
                        json.dumps(
                            {
                                "deleted": count,
                                "logical_bytes": total,
                                "retained": sum(retained.values()),
                                "errors": errors,
                                "seconds": round(time.monotonic() - started),
                            }
                        ),
                        flush=True,
                    )
                    last = time.monotonic()
    return {
        "deleted": count,
        "logical_bytes": total,
        "errors": errors,
        "retained": dict(retained),
        "free_before": before,
        "free_after": shutil.disk_usage(PROJECT.anchor).free,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("inventory", "ingest", "review-content", "verify", "certify", "delete", "verify-after-delete"),
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    from app.services.material_file_access import legacy_workflow_error

    legacy_workflow_error()
    directory = args.run.resolve()
    if not directory.is_relative_to(PROJECT / ".local-runs"):
        parser.error("迁移记录必须位于本项目 .local-runs 内")
    if args.action == "inventory":
        result = inventory(directory)
    elif args.action == "ingest":
        result = ingest(directory, args.workers, args.limit, args.input)
    elif args.action == "review-content":
        result = review_content(directory, args.workers, args.limit, args.input)
    elif args.action == "verify":
        from app.services.material_migration_validation import verify

        result = verify(directory)
    elif args.action in {"certify", "verify-after-delete"}:
        from app.services.material_migration_validation import certify, verify_after_delete

        result = certify(directory) if args.action == "certify" else verify_after_delete(directory)
    else:
        result = delete(directory)
    result["at"] = datetime.now(UTC).isoformat()
    (directory / (args.action + "-result.json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
