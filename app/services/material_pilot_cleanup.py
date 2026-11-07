"""008888单次授权清理：冻结清单、完整文本、独立验收和引用检查缺一不可。

不解除全局暂停，不删除基础正文、历史业务结果或共享字形。数据库旧图文
版本只在全部在用引用已核清时退出；文件删除持有实体锁，并在事务失败时
撤销删除标记。逐文件意图落盘，不能把缺失文件冒充本轮释放空间。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zlib
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from app.commands.material_text_pilot import business_baseline, space
from app.services import material_content as store
from app.services.material_document import get_review, read_document
from app.services.material_file_cleanup import LockedOriginal
from app.services.material_text_pilot import POLICY, _text_only

ROOT = Path(__file__).resolve().parents[2]
PART_FILTER = """engine NOT LIKE 'HISTORY_SVG_%' AND engine NOT LIKE 'HISTORY_PAGE_PNG_%'
 AND engine NOT LIKE '%ASSET%' AND engine NOT LIKE 'PYPDF_%'
 AND engine NOT LIKE '%SOURCE_TEXT%' AND engine NOT LIKE '%EXACT_TEXT%'
 AND engine NOT LIKE '%ARCHIVE_TEXT%' AND engine NOT LIKE '%ATTACHMENT%'"""


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _references(value, targets):
    result = []
    if isinstance(value, dict):
        if (value.get("document_source_hash"), value.get("policy")) in targets:
            result.append(value)
        for child in value.values():
            result.extend(_references(child, targets))
    elif isinstance(value, list):
        for child in value:
            result.extend(_references(child, targets))
    return result


def fingerprints(connection, *, exclude_headers=(), exclude_parts=()):
    """只流式计算元数据摘要；执行前锁表后再核一次，避免引用清单在中途变化。"""
    results = {}
    for name, sql in {
        "headers": "SELECT source_hash,policy,document_hash FROM material_document_content ORDER BY source_hash,policy",
        "parts": "SELECT source_hash,engine,page_index,content_hash FROM material_page_representation WHERE "
        + PART_FILTER
        + " ORDER BY source_hash,engine,page_index",
    }.items():
        checksum, count = hashlib.sha256(), 0
        for row in connection.execute(text(sql).execution_options(stream_results=True)).yield_per(500):
            if name == "headers" and tuple(row[:2]) in exclude_headers:
                continue
            if name == "parts" and tuple(row[:3]) in exclude_parts:
                continue
            checksum.update(store.encoded(list(row)))
            count += 1
        results[name] = {"rows": count, "sha256": checksum.hexdigest()}
    return results


def audit_references(repo, targets):
    """逐一检查所有文档目录及可能承载子文档引用的结构片段，图像字节不作引用入口。"""
    external, counts = [], {"headers": 0, "structured_parts": 0}
    needles = [h.encode("ascii") for h, _ in targets]
    with repo.engine.connect() as c:
        c.execute(text("SET TRANSACTION READ ONLY"))
        for h, policy, expected, compressed in c.execute(
            text(
                "SELECT source_hash,policy,document_hash,document_data FROM material_document_content"
            ).execution_options(stream_results=True)
        ).yield_per(50):
            raw = zlib.decompress(compressed)
            if store.checksum(raw) != expected:
                raise ValueError("CLEANUP_REFERENCE_HEADER_CORRUPTED")
            counts["headers"] += 1
            if (h, policy) not in targets and any(n in raw for n in needles):
                for reference in _references(json.loads(raw), targets):
                    external.append({"owner": [h, policy], "reference": reference})
        for h, engine, index, expected, compressed in c.execute(
            text(
                "SELECT source_hash,engine,page_index,content_hash,extracted_text "
                "FROM material_page_representation WHERE " + PART_FILTER
            ).execution_options(stream_results=True)
        ).yield_per(50):
            raw = store.decompress_page(compressed)
            if store.checksum(raw) != expected:
                raise ValueError("CLEANUP_REFERENCE_PART_CORRUPTED")
            counts["structured_parts"] += 1
            if any(n in raw for n in needles):
                for reference in _references(json.loads(raw), targets):
                    external.append({"owner": [h, engine, index], "reference": reference})
        metadata = fingerprints(c)
    return {
        "external_references": external,
        "scanned": counts,
        "fingerprints": metadata,
        "scope": "显式历史政策/文档哈希读取依赖；业务原件路径继续以来源摘要读库",
    }


def validate_proofs(run):
    run = Path(run)
    scope = load(run / "scope.json")
    complete = run / "completion-20261103"
    verification = load(run / "verification.json")
    charts = load(complete / "database-figures-verification.json")
    source = load(run / "source-verification.json")
    if (
        scope["fund_code"] != "008888"
        or len(scope["sources"]) != 11
        or not verification["complete"]
        or not verification["business_unchanged"]
        or verification["original_access_attempts"]
        or not charts["all_18_passed"]
        or len(charts["figures"]) != 18
        or not source["all_source_content_matches"]
    ):
        raise ValueError("CLEANUP_COMPLETE_CONTENT_PROOFS_REQUIRED")
    identities = {r["source_hash"] for r in scope["sources"]}
    if identities != {r["source_hash"] for r in verification["sources"]}:
        raise ValueError("CLEANUP_VERIFICATION_SCOPE_CHANGED")
    return scope, {
        str(p.relative_to(run)): digest(p)
        for p in [
            run / "scope.json",
            run / "verification.json",
            run / "source-verification.json",
            complete / "database-figures-verification.json",
            complete / "old-png-verification.json",
        ]
    }


def prepare(run):
    """形成可审阅、可重复核对的精确清理清单，不执行删除。"""
    run = Path(run)
    if (run / "completion-20261103/cleanup-result.json").exists():
        completed = execute(run)
        return {
            "already_completed": True,
            "can_cleanup": False,
            "originals_deleted": completed["originals_deleted"],
            "old_image_files_deleted": completed["old_image_files_deleted"],
        }
    scope, proofs = validate_proofs(run)
    repo, complete = store.repository(), run / "completion-20261103"
    baseline = load(complete / "baseline.json")
    sources, files = {}, []
    for source in scope["sources"]:
        h = source["source_hash"]
        review = get_review(h, repo=repo, policy=POLICY)
        if not review or not review["ready"] or review["base_content_hash"] != source["base_content_hash"]:
            raise ValueError("CLEANUP_CURRENT_TEXT_NOT_READY")
        _text_only(read_document(h, repo=repo, policy=POLICY))
        sources[h] = review["document_hash"]
        expected_paths = set(source["existing_paths"])
        if expected_paths != {r["path"] for r in baseline["originals"] if r["sha256"] == h}:
            raise ValueError("CLEANUP_FROZEN_ORIGINAL_PATHS_CHANGED")
        for row in baseline["originals"]:
            if row["path"] in expected_paths:
                files.append({**row, "source_hash": h, "file_sha256": row["sha256"], "kind": "PDF_ORIGINAL"})
    png = load(complete / "old-png-verification.json")
    if not png["pixel_identical_to_source_page"] or not png["full_text_and_table_values_verified"]:
        raise ValueError("CLEANUP_OLD_IMAGE_NOT_FULLY_REPLACED")
    if png["formal_document_hash"] != sources[png["source_hash"]]:
        raise ValueError("CLEANUP_OLD_IMAGE_TEXT_VERSION_CHANGED")
    files.append({**png, "file_sha256": png["sha256"], "kind": "DERIVED_IMAGE"})
    with repo.engine.connect() as c:
        c.execute(text("SET TRANSACTION READ ONLY"))
        if (
            c.scalar(text("SELECT source_hash FROM material_source_location WHERE location=:p"), {"p": png["path"]})
            is not None
        ):
            raise ValueError("CLEANUP_IMAGE_ALIAS_NEEDS_SEPARATE_REVIEW")
        if business_baseline(c) != scope["business_before"]:
            raise ValueError("CLEANUP_BUSINESS_RECORDS_CHANGED")
        legacy = [
            dict(r)
            for r in c.execute(
                text(
                    "SELECT source_hash,policy,document_hash,octet_length(document_data) bytes "
                    "FROM material_document_content WHERE source_hash=ANY(:hs) AND policy LIKE 'HISTORICAL_%'"
                ),
                {"hs": list(sources)},
            ).mappings()
        ]
        old_parts = [
            dict(r)
            for r in c.execute(
                text(
                    "SELECT source_hash,engine,page_index,content_hash,octet_length(extracted_text) bytes "
                    "FROM material_page_representation WHERE source_hash=ANY(:hs) AND engine LIKE 'HISTORY_%'"
                ),
                {"hs": list(sources)},
            ).mappings()
        ]
        current_aliases = [
            dict(r)
            for r in c.execute(
                text("SELECT location,source_hash FROM material_source_location WHERE source_hash=ANY(:hs)"),
                {"hs": list(sources)},
            ).mappings()
        ]
        before = space(c, list(sources))
    if {(r["location"], r["source_hash"]) for r in current_aliases} != {
        (r["location"], r["source_hash"]) for r in baseline["aliases"]
    }:
        raise ValueError("CLEANUP_SHARED_SOURCE_ALIAS_SCOPE_CHANGED")
    targets = {(r["source_hash"], r["policy"]) for r in legacy}
    prior_path = complete / "cleanup-approved-plan.json"
    prior = load(prior_path) if prior_path.exists() else None
    with repo.engine.connect() as c:
        current_fingerprints = fingerprints(c)
    if (
        prior
        and prior["reference_audit"]["fingerprints"] == current_fingerprints
        and {(r["source_hash"], r["policy"]) for r in prior["legacy_documents"]} == targets
    ):
        audit = prior["reference_audit"]
    else:
        audit = audit_references(repo, targets)
    if audit["external_references"]:
        raise ValueError("CLEANUP_LEGACY_VERSION_STILL_REFERENCED")
    for row in files:
        with LockedOriginal(
            row["path"],
            root=ROOT,
            expected_hash=row["file_sha256"],
            expected_size=row["size"],
            expected_file_id=row["file_id"],
        ) as handle:
            row["allocated_bytes"] = handle.allocated_bytes
            row["hardlinks"] = handle.hardlinks
    plan = {
        "fund_code": "008888",
        "policy": POLICY,
        "proofs": proofs,
        "sources": sources,
        "files": files,
        "legacy_documents": legacy,
        "legacy_parts": old_parts,
        "reference_audit": audit,
        "space_before": before,
        "can_cleanup": True,
        "authorization": "用户明确要求：剩余18张图表完整转文本并全部核验后，清理原件和旧图片，核对实际释放空间。",
        "preserve": "所有基础正文、全部历史评级预测、来源路径映射及全库共享字形/资产；不打开自动清理开关",
    }
    with repo.engine.connect() as c:
        plan["expected_retained_fingerprints"] = fingerprints(
            c,
            exclude_headers={(r["source_hash"], r["policy"]) for r in legacy},
            exclude_parts={(r["source_hash"], r["engine"], r["page_index"]) for r in old_parts},
        )
    save(complete / "cleanup-approved-plan.json", plan)
    return {
        "files": len(files),
        "originals": sum(r["kind"] == "PDF_ORIGINAL" for r in files),
        "legacy_parts": len(old_parts),
        "legacy_documents": len(legacy),
        "file_allocated_bytes": sum(r["allocated_bytes"] for r in files),
        "external_references": 0,
        "can_cleanup": True,
    }


def _append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute(run):
    """只消费已冻结清单；正文、图表和业务复验失效时不进入删除阶段。"""
    run = Path(run)
    complete = run / "completion-20261103"
    plan = load(complete / "cleanup-approved-plan.json")
    if (complete / "cleanup-result.json").exists():
        prior = load(complete / "cleanup-result.json")
        if any(Path(r["path"]).exists() for r in plan["files"]):
            raise ValueError("CLEANUP_FINISHED_PATH_REAPPEARED")
        for h, expected in plan["sources"].items():
            current = get_review(h, policy=POLICY)
            if not current or current["document_hash"] != expected:
                raise ValueError("CLEANUP_FINISHED_TEXT_CHANGED")
        with store.repository().engine.connect() as c:
            hs = list(plan["sources"])
            old_headers = c.scalar(
                text(
                    "SELECT count(*) FROM material_document_content "
                    "WHERE source_hash=ANY(:hs) AND policy LIKE 'HISTORICAL_%'"
                ),
                {"hs": hs},
            )
            old_parts = c.scalar(
                text(
                    "SELECT count(*) FROM material_page_representation "
                    "WHERE source_hash=ANY(:hs) AND engine LIKE 'HISTORY_%'"
                ),
                {"hs": hs},
            )
            if old_headers or old_parts:
                raise ValueError("CLEANUP_FINISHED_OLD_VERSION_REAPPEARED")
        return {**prior, "already_completed": True}
    scope, actual = validate_proofs(run)
    if plan["fund_code"] != "008888" or not plan["can_cleanup"] or actual != plan["proofs"]:
        raise ValueError("CLEANUP_APPROVED_PROOFS_CHANGED")
    repo = store.repository()
    journal = complete / "cleanup-journal.jsonl"
    done, intents = {}, {}
    if journal.exists():
        for row in map(json.loads, journal.read_text(encoding="utf-8").splitlines()):
            if row["status"] in {"DELETED", "RECOVERED_DELETED"}:
                done[row["path"]] = row
            elif row["status"] == "DELETE_INTENT":
                intents[row["path"]] = row
    released = sum(r["allocated_bytes"] for r in done.values() if r["hardlinks"] == 1)
    free_before = shutil.disk_usage(ROOT.anchor).free
    started_path = complete / "cleanup-start.json"
    if not started_path.exists():
        save(
            started_path,
            {"C_free_before_bytes": free_before, "plan_sha256": digest(complete / "cleanup-approved-plan.json")},
        )
    free_before = load(started_path)["C_free_before_bytes"]
    for row in plan["files"]:
        path = Path(row["path"])
        if str(path) in done:
            if path.exists():
                raise ValueError("CLEANUP_DELETED_PATH_REAPPEARED")
            continue
        review = get_review(row["source_hash"], repo=repo, policy=POLICY)
        if review is None or not review["ready"] or review["document_hash"] != plan["sources"][row["source_hash"]]:
            raise ValueError("CLEANUP_COMPLETE_TEXT_VERSION_CHANGED")
        if not path.exists() and str(path) in intents:
            intent = intents[str(path)]
            if (
                intent["plan_sha256"] != digest(complete / "cleanup-approved-plan.json")
                or intent["document_hash"] != review["document_hash"]
            ):
                raise ValueError("CLEANUP_INTERRUPTED_INTENT_CHANGED")
            with repo.engine.begin() as c:
                if row["kind"] == "PDF_ORIGINAL":
                    c.execute(
                        text(
                            "UPDATE material_source_location SET deleted_at=:at,verification=:v "
                            "WHERE location=:p AND source_hash=:h"
                        ),
                        {
                            "at": datetime.now(UTC),
                            "v": review["document_hash"],
                            "p": str(path),
                            "h": row["source_hash"],
                        },
                    )
            _append(journal, {**intent, "status": "RECOVERED_DELETED"})
            released += intent["allocated_bytes"] if intent["hardlinks"] == 1 else 0
            continue
        with repo.engine.connect() as c:
            with c.begin():
                # 锁住本来源的版本、正文及片段，清理期间其他执行者不能替换依据。
                for table in (
                    "material_extracted_content",
                    "material_document_content",
                    "material_page_representation",
                ):
                    c.execute(
                        text(f"SELECT source_hash FROM {table} WHERE source_hash=:h FOR SHARE"),
                        {"h": row["source_hash"]},
                    ).all()
                alias = c.execute(
                    text("SELECT source_hash FROM material_source_location WHERE location=:p FOR UPDATE"),
                    {"p": str(path)},
                ).scalar()
                if row["kind"] == "PDF_ORIGINAL" and alias != row["source_hash"]:
                    raise ValueError("CLEANUP_ORIGINAL_ALIAS_CHANGED")
                if row["kind"] == "DERIVED_IMAGE" and alias is not None:
                    raise ValueError("CLEANUP_IMAGE_ALIAS_NEEDS_SEPARATE_REVIEW")
                with LockedOriginal(
                    path,
                    root=ROOT,
                    expected_hash=row["file_sha256"],
                    expected_size=row["size"],
                    expected_file_id=row["file_id"],
                ) as handle:
                    event = {
                        "path": str(path),
                        "source_hash": row["source_hash"],
                        "kind": row["kind"],
                        "allocated_bytes": handle.allocated_bytes,
                        "hardlinks": handle.hardlinks,
                        "document_hash": review["document_hash"],
                        "plan_sha256": digest(complete / "cleanup-approved-plan.json"),
                    }
                    _append(journal, {**event, "status": "DELETE_INTENT"})
                    if alias is not None:
                        c.execute(
                            text(
                                "UPDATE material_source_location SET deleted_at=:at,verification=:v "
                                "WHERE location=:p AND source_hash=:h"
                            ),
                            {
                                "at": datetime.now(UTC),
                                "v": review["document_hash"],
                                "p": str(path),
                                "h": row["source_hash"],
                            },
                        )
                    handle.mark_delete()
                    try:
                        c.commit()
                    except BaseException:
                        c.connection.rollback()
                        raise
                if path.exists():
                    raise ValueError("CLEANUP_FILE_DELETE_NOT_OBSERVED")
                _append(journal, {**event, "status": "DELETED"})
                released += event["allocated_bytes"] if event["hardlinks"] == 1 else 0
    # 原件退出后，库内旧图文版本在短事务中单独退出；引用元数据变化时保留旧版本。
    with repo.engine.begin() as c:
        c.execute(text("SET LOCAL lock_timeout='5s'"))
        c.execute(text("LOCK TABLE material_document_content,material_page_representation IN SHARE ROW EXCLUSIVE MODE"))
        if fingerprints(c) != plan["reference_audit"]["fingerprints"]:
            raise ValueError("CLEANUP_REFERENCE_GRAPH_CHANGED_OLD_IMAGES_RETAINED")
        for row in plan["legacy_documents"]:
            n = c.execute(
                text(
                    "DELETE FROM material_document_content WHERE source_hash=:source_hash "
                    "AND policy=:policy AND document_hash=:document_hash"
                ),
                row,
            ).rowcount
            if n != 1:
                raise ValueError("CLEANUP_LEGACY_DOCUMENT_CHANGED")
        for row in plan["legacy_parts"]:
            n = c.execute(
                text(
                    "DELETE FROM material_page_representation WHERE source_hash=:source_hash "
                    "AND engine=:engine AND page_index=:page_index AND content_hash=:content_hash"
                ),
                row,
            ).rowcount
            if n != 1:
                raise ValueError("CLEANUP_LEGACY_PART_CHANGED")
    with repo.engine.connect() as c:
        c.execute(text("SET TRANSACTION READ ONLY"))
        if fingerprints(c) != plan["expected_retained_fingerprints"]:
            raise ValueError("CLEANUP_RETAINED_REFERENCE_METADATA_CHANGED")
        after = space(c, list(plan["sources"]))
        business = business_baseline(c)
    result = {
        "originals_deleted": sum(r["kind"] == "PDF_ORIGINAL" for r in plan["files"]),
        "old_image_files_deleted": sum(r["kind"] == "DERIVED_IMAGE" for r in plan["files"]),
        "legacy_parts_deleted": len(plan["legacy_parts"]),
        "legacy_documents_deleted": len(plan["legacy_documents"]),
        "legacy_payload_bytes_removed": sum(r["bytes"] for r in plan["legacy_parts"] + plan["legacy_documents"]),
        "files_allocated_bytes_reclaimed": released,
        "C_free_before_bytes": free_before,
        "C_free_after_bytes": shutil.disk_usage(ROOT.anchor).free,
        "space_after": after,
        "business_unchanged": business == scope["business_before"],
        "shared_assets_deleted": 0,
        "global_cleanup_pause_changed": False,
        "database_deleted_payload_is_not_immediate_C_reclamation": True,
    }
    save(complete / "cleanup-result.json", result)
    return result
