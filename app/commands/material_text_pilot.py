"""008888有界纯文本试点命令。提取、核验、发布与清理各自执行，不打开全局删除。

工具依赖可通过--tool-path复用既有离线包目录；正式资料读取不依赖这些目录。
本命令的scope.json冻结来源版本和路径，禁止按文件名含008888直接授予删除资格。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import event, text

from app.services import material_content as store
from app.services import material_text_pilot as pilot

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / ".local-runs/material-008888-text-pilot-20261103"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def business_baseline(connection):
    """冻结已有评级与预测的实际内容摘要，复验时不重算或覆盖任何结果。"""
    result = {}
    statements = {
        "ratings": "SELECT row_to_json(r)::text FROM fund_rating_result r WHERE fund_code='008888' ORDER BY batch_id",
        "evidence": "SELECT row_to_json(r)::text FROM fund_rating_evidence r "
        "WHERE fund_code='008888' ORDER BY evidence_hash",
        "predictions": "SELECT row_to_json(r)::text FROM prediction_feature_snapshot r "
        "WHERE fund_code='008888' ORDER BY snapshot_id",
        "tasks": "SELECT row_to_json(r)::text FROM prediction_task_input r "
        "WHERE fund_code='008888' ORDER BY generation_task_id",
    }
    for name, sql in statements.items():
        rows = list(connection.execute(text(sql)).scalars())
        result[name] = {"rows": len(rows), "sha256": store.checksum(store.encoded(rows))}
    return result


def space(connection, hashes):
    """载荷和物理关系分开计量；全库变化不能归功于本试点。"""
    tables = list(
        connection.execute(
            text(
                "SELECT relname,pg_total_relation_size(relid) bytes FROM pg_statio_user_tables "
                "WHERE relname IN ('material_extracted_content','material_page_representation',"
                "'material_document_content','material_source_location') ORDER BY relname"
            )
        ).mappings()
    )
    pages = list(
        connection.execute(
            text(
                "SELECT engine,count(*) rows,sum(octet_length(extracted_text)) bytes "
                "FROM material_page_representation WHERE source_hash=ANY(:hs) GROUP BY engine ORDER BY engine"
            ),
            {"hs": hashes},
        ).mappings()
    )
    documents = list(
        connection.execute(
            text(
                "SELECT policy,count(*) rows,sum(octet_length(document_data)) bytes "
                "FROM material_document_content WHERE source_hash=ANY(:hs) GROUP BY policy ORDER BY policy"
            ),
            {"hs": hashes},
        ).mappings()
    )
    can_read_wal = connection.scalar(text("SELECT has_function_privilege('pg_ls_waldir()', 'EXECUTE')"))
    return {
        "at": datetime.now(UTC).isoformat(),
        "scope_pages": [dict(r) for r in pages],
        "scope_documents": [dict(r) for r in documents],
        "all_fund_table_bytes": [dict(r) for r in tables],
        "windows_free_bytes": shutil.disk_usage(ROOT.anchor).free,
        "wal_bytes": (
            connection.scalar(text("SELECT coalesce(sum(size),0) FROM pg_ls_waldir()")) if can_read_wal else None
        ),
        "wal_measurement": "MEASURED" if can_read_wal else "UNAVAILABLE_ROLE_PERMISSION",
    }


def inventory(run):
    """按文档主体确认PDF范围，记录共享路径；目录与哈希中的巧合数字不是基金归属。"""
    if (run / "scope.json").exists():
        frozen = load(run / "scope.json")
        if frozen.get("fund_code") != pilot.FUND or frozen.get("policy") != pilot.POLICY:
            raise ValueError("TEXT_PILOT_FROZEN_SCOPE_IDENTITY_MISMATCH")
        return {
            "sources": len(frozen["sources"]),
            "pages": sum(s["pages"] for s in frozen["sources"]),
            "frozen_paths": sum(len(s["existing_paths"]) for s in frozen["sources"]),
            "existing_paths": sum(Path(p).is_file() for s in frozen["sources"] for p in s["existing_paths"]),
            "already_frozen": True,
        }
    repo = store.repository()
    with repo.engine.connect() as c:
        c.execute(text("SET TRANSACTION READ ONLY"))
        candidates = c.execute(
            text(
                "SELECT DISTINCT b.source_hash,b.extracted_data,b.source_size,b.content_hash "
                "FROM material_extracted_content b JOIN material_source_location l ON l.source_hash=b.source_hash "
                "WHERE b.format='PDF' AND l.location LIKE '%008888%' ORDER BY b.source_hash"
            )
        ).mappings()
        sources, excluded = [], []
        for row in candidates:
            raw = zlib.decompress(row["extracted_data"])
            if store.checksum(raw) != row["content_hash"]:
                raise ValueError("TEXT_PILOT_BASE_HASH_MISMATCH")
            pages = json.loads(raw)["pdfium"]
            title = "".join(pages[0].split())
            if "华夏国证半导体芯片" not in title or "联接基金" not in title:
                excluded.append(row["source_hash"])
                continue
            paths = list(
                c.execute(
                    text("SELECT location FROM material_source_location WHERE source_hash=:h ORDER BY location"),
                    {"h": row["source_hash"]},
                ).scalars()
            )
            current = [p for p in paths if Path(p).is_file()]
            for p in current:
                with Path(p).open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != row["source_hash"]:
                        raise ValueError("TEXT_PILOT_ORIGINAL_CHANGED")
            sources.append(
                {
                    "source_hash": row["source_hash"],
                    "base_content_hash": row["content_hash"],
                    "source_bytes": row["source_size"],
                    "pages": len(pages),
                    "title": title,
                    "paths": paths,
                    "existing_paths": current,
                }
            )
        scope = {
            "fund_code": pilot.FUND,
            "policy": pilot.POLICY,
            "sources": sources,
            "excluded_non_target_pdf_sources": excluded,
            "scope": "已明确主体的008888联接基金PDF试点；非全基金、非008888全历史穷尽声明",
            "business_before": business_baseline(c),
            "space_before": space(c, [s["source_hash"] for s in sources]),
        }
    path = run / "scope.json"
    if path.exists() and load(path) != scope:
        raise ValueError("TEXT_PILOT_SCOPE_ALREADY_FROZEN")
    save(path, scope)
    return {
        "sources": len(sources),
        "pages": sum(s["pages"] for s in sources),
        "existing_paths": sum(len(s["existing_paths"]) for s in sources),
    }


def extract(run):
    scope, repo = load(run / "scope.json"), store.repository()
    results = []
    for row in scope["sources"]:
        path = next((Path(p) for p in row["existing_paths"] if Path(p).is_file()), None)
        if path is None:
            raise ValueError("TEXT_PILOT_SOURCE_MISSING")
        started = time.monotonic()
        document = pilot.extract_pdf(path, repo=repo, temp_root=run / "temporary")
        result = {
            "source_hash": row["source_hash"],
            "pages": row["pages"],
            "pending_pages": len(document["issues"]),
            "seconds": round(time.monotonic() - started, 2),
        }
        results.append(result)
        save(run / "extraction.json", results)
        print(json.dumps(result), flush=True)
    return {
        "sources": len(results),
        "pages": sum(r["pages"] for r in results),
        "pending_pages": sum(r["pending_pages"] for r in results),
    }


def publish(run):
    repo, scope = store.repository(), load(run / "scope.json")
    results = []
    for source in scope["sources"]:
        h = source["source_hash"]
        review_file = run / "reviews" / (h + ".json")
        result = pilot.publish(h, repo=repo, reviews=load(review_file) if review_file.is_file() else {})
        results.append(
            {
                "source_hash": h,
                "ready": result["ready"],
                "document_hash": result.get("document_hash"),
                "pending": result.get("pending", []),
            }
        )
    save(run / "publication.json", results)
    return {
        "sources": len(results),
        "ready": sum(r["ready"] for r in results),
        "pending_pages": sum(len(r["pending"]) for r in results),
    }


def ocr(run):
    from app.services import material_text_ocr

    repo, scope = store.repository(), load(run / "scope.json")
    recognizer = material_text_ocr.engine()
    count = 0
    for source in scope["sources"]:
        rows = material_text_ocr.extract(source, repo=repo, recognizer=recognizer)
        save(run / "ocr" / (source["source_hash"] + ".json"), rows)
        count += len(rows)
        print(json.dumps({"source_hash": source["source_hash"], "ocr_regions": len(rows)}), flush=True)
    return {"ocr_regions": count, "verified_regions": 0, "can_delete_images": False}


def curves(run):
    """关联完整曲线区间候选，保留不确定性；不把候选标成正式图表数据。"""
    repo = store.repository()
    figures = load(run / "figure-plan.json")
    stored, unavailable = [], []
    for figure in figures:
        h, page, number = figure["source_hash"], figure["page"], figure["region"]["region"]
        name = h[:8] + f"-p{page}-r{number}.json"
        path = run / "curve-candidates" / name
        if not path.is_file():
            cached = repo.page(h, "TEXT_ONLY_CURVE_V1", (page - 1) * 10000 + number)
            if cached is None:
                unavailable.append({"source_hash": h, "page": page, "region": number})
                continue
            previous = json.loads(cached)
            if previous["region_hash"] != figure["region"]["region_hash"]:
                raise ValueError("TEXT_PILOT_CURVE_IDENTITY_CHANGED")
            data = previous["content"]
        else:
            data = load(path)
        value = {
            "source_hash": h,
            "page": page,
            "region_hash": figure["region"]["region_hash"],
            "status": "UNVERIFIED",
            "can_replace_image": False,
            "content": data,
            "remaining": ["仅为图面区间候选，尚未完成独立业务等价核验"],
        }
        reference = pilot._part(repo, h, "TEXT_ONLY_CURVE_V1", (page - 1) * 10000 + number, value)
        stored.append(
            {
                "source_hash": h,
                "page": page,
                "reference": reference,
                "series": len(data["data"]),
                "observations": sum(len(s["samples"]) for s in data["data"]),
            }
        )
    save(run / "curve-storage.json", {"stored": stored, "without_curve_candidate": unavailable})
    return {
        "stored_figures": len(stored),
        "verified_figures": 0,
        "observations": sum(s["observations"] for s in stored),
    }


def verify_sources(run):
    """独立进程重新对照原件全文/表格/图像清单，与只读库内验收分开。"""
    import pymupdf
    from pypdf import PdfReader

    from app.services.material_document import read_document

    scope, repo = load(run / "scope.json"), store.repository()
    results = []
    for source in scope["sources"]:
        h = source["source_hash"]
        path = next(Path(p) for p in source["existing_paths"] if Path(p).is_file())
        if store.checksum(path.read_bytes()) != h:
            raise ValueError("TEXT_PILOT_INDEPENDENT_SOURCE_HASH_CHANGED")
        saved = read_document(h, repo=repo, policy=pilot.CANDIDATE)
        reader = PdfReader(path)
        table_count = cells = regions = 0
        with pymupdf.open(path) as doc:
            for i, page in enumerate(doc):
                stored = saved["content"]["pages"][i]
                if (
                    page.get_text("text", sort=False) != stored["independent_text"]
                    or (reader.pages[i].extract_text() or "") != stored["third_text"]
                    or (reader.pages[i].extract_text(extraction_mode="layout") or "") != stored["layout_text"]
                ):
                    raise ValueError("TEXT_PILOT_INDEPENDENT_SOURCE_TEXT_DIFFERENCE")
                tables = page.find_tables().tables
                if [t.extract() for t in tables] != [t["cells"] for t in stored["tables"]]:
                    raise ValueError("TEXT_PILOT_INDEPENDENT_SOURCE_TABLE_DIFFERENCE")
                table_count += len(tables)
                cells += sum(t.row_count * t.col_count for t in tables)
                regions += len(stored["visual_regions"])
                if stored["preview_sha256"]:
                    preview = page.get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8), alpha=False).tobytes("png")
                    if store.checksum(preview) != stored["preview_sha256"]:
                        raise ValueError("TEXT_PILOT_INDEPENDENT_SOURCE_VISUAL_CHANGED")
        results.append(
            {
                "source_hash": h,
                "candidate_hash": saved["document_hash"],
                "source_text_and_tables_match": True,
                "pages": source["pages"],
                "tables": table_count,
                "cells": cells,
                "visual_regions": regions,
            }
        )
        print(json.dumps({"source_hash": h, "source_matches": True}), flush=True)
    save(
        run / "source-verification.json",
        {"sources": results, "all_source_content_matches": True, "semantic_completion_is_separate": True},
    )
    return {
        "sources": len(results),
        "pages": sum(r["pages"] for r in results),
        "tables": sum(r["tables"] for r in results),
        "cells": sum(r["cells"] for r in results),
    }


def verify(run):
    """新进程、只读事务及原件访问拦截同时生效；不触发任何自动补提。"""
    from app.services.material_document import get_review, read_document

    scope, repo = load(run / "scope.json"), store.repository()
    protected = {Path(p).resolve() for s in scope["sources"] for p in s["paths"]}
    blocked_attempts = []
    audit_active = [True]

    def guard_files(name, args):
        if (
            audit_active[0]
            and name == "open"
            and args
            and isinstance(args[0], (str, bytes))
            and Path(os.fsdecode(args[0])).resolve() in protected
        ):
            blocked_attempts.append(str(args[0]))
            raise ValueError("TEXT_PILOT_ORIGINAL_ACCESS_BLOCKED")

    def guard_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().split(maxsplit=1)[0].upper() not in {"SELECT", "SET"}:
            raise ValueError("TEXT_PILOT_VERIFICATION_MUST_BE_READ_ONLY")

    rows = []
    # Python审计覆盖Path.open、builtins.open及底层open事件；数据库监听禁止补提写入。
    sys.addaudithook(guard_files)
    event.listen(repo.engine, "before_cursor_execute", guard_sql)
    try:
        with store.database_only():
            for source in scope["sources"]:
                h = source["source_hash"]
                candidate = read_document(h, repo=repo, policy=pilot.CANDIDATE)
                base = repo.get(source_hash=h, fresh=True)
                if len(candidate["content"]["pages"]) != source["pages"]:
                    raise ValueError("TEXT_PILOT_DATABASE_PAGE_COUNT")
                for i, page in enumerate(candidate["content"]["pages"]):
                    if page["native_text"] != base.extracted["pdfium"][i]:
                        raise ValueError("TEXT_PILOT_DATABASE_NATIVE_TEXT")
                    for engine, expected in [
                        ("PYPDF_TEXT_V1", page["third_text"]),
                        ("PYPDF_LAYOUT_V1", page["layout_text"]),
                    ]:
                        if repo.page(h, engine, i) != expected:
                            raise ValueError("TEXT_PILOT_DATABASE_LAYOUT_TEXT")
                # 实际旧计算器入口也在原件访问拦截下读库，不只校验新服务自身。
                with store.using_repository(repo):
                    source_value = store.source_bytes(source["paths"][0])
                    with store.document_reader(source_value) as reader:
                        for i, original in enumerate(candidate["content"]["pages"]):
                            if reader.pages[i].extract_text(extraction_mode="layout") != original["layout_text"]:
                                raise ValueError("TEXT_PILOT_LEGACY_READER_CONTENT_MISMATCH")
                final = get_review(h, repo=repo, policy=pilot.POLICY)
                if final is not None and final["ready"]:
                    selected = read_document(h, repo=repo)
                    if selected["schema"] != pilot.POLICY or len(selected["content"]["pages"]) != source["pages"]:
                        raise ValueError("TEXT_PILOT_SELECTED_READER_MISMATCH")
                rows.append(
                    {
                        "source_hash": h,
                        "pages": source["pages"],
                        "database_content_verified": True,
                        "complete_text_ready": bool(final and final["ready"]),
                        "candidate_hash": candidate["document_hash"],
                    }
                )
    finally:
        audit_active[0] = False
        event.remove(repo.engine, "before_cursor_execute", guard_sql)
    with repo.engine.connect() as c:
        c.execute(text("SET TRANSACTION READ ONLY"))
        after = business_baseline(c)
        current_space = space(c, [r["source_hash"] for r in rows])
    result = {
        "sources": rows,
        "original_access_attempts": blocked_attempts,
        "business_unchanged": after == scope["business_before"],
        "business_after": after,
        "space_after": current_space,
        "complete": all(r["complete_text_ready"] for r in rows) and after == scope["business_before"],
    }
    save(run / "verification.json", result)
    return {
        "sources": len(rows),
        "pages": sum(r["pages"] for r in rows),
        "ready": sum(r["complete_text_ready"] for r in rows),
        "complete": result["complete"],
        "business_unchanged": result["business_unchanged"],
    }


def cleanup_plan(run):
    """先整批通过完整性及空间验收；失败时生成明确保留清单，绝不解除暂停。"""
    if (run / "completion-20261103/cleanup-result.json").exists():
        from app.services.material_pilot_cleanup import prepare

        completed = prepare(run)
        save(
            run / "cleanup-plan.json",
            {**completed, "status": "EXECUTED_VERIFIED", "details": "completion-20261103/cleanup-approved-plan.json"},
        )
        return completed
    scope, verification = load(run / "scope.json"), load(run / "verification.json")
    pending = [r["source_hash"] for r in verification["sources"] if not r["complete_text_ready"]]
    result = {
        "fund_code": pilot.FUND,
        "scope_sha256": store.checksum((run / "scope.json").read_bytes()),
        "verification_sha256": store.checksum((run / "verification.json").read_bytes()),
        "content_complete": verification["complete"],
        "can_cleanup": False,
        "pending_sources": pending,
        "originals_deleted": 0,
        "old_image_rows_deleted": 0,
        "reason": "完整性验收未全部通过，原件和旧图片继续保留"
        if pending
        else "待完成实际空间收益与全部共享/历史版本引用核验，不能只凭完整标记清理",
        "preserved_source_paths": [p for s in scope["sources"] for p in s["existing_paths"]],
        "global_cleanup_hold_unchanged": True,
    }
    save(run / "cleanup-plan.json", result)
    return {
        "can_cleanup": False,
        "pending_sources": len(pending),
        "originals_deleted": 0,
        "old_image_rows_deleted": 0,
        "reason": result["reason"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["inventory", "extract", "ocr", "curves", "publish", "verify", "verify-sources", "cleanup-plan"],
    )
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--tool-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    from app.services.material_file_access import legacy_workflow_error

    legacy_workflow_error()
    for path in reversed(args.tool_path):
        sys.path.insert(0, str(path.resolve()))
    result = globals()[args.action.replace("-", "_")](args.run.resolve())
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
