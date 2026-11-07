"""008888现存源文件归档；默认只盘点，--execute才迁移C盘副本，不恢复已删原件。"""

import argparse
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

from app.commands.material_text_pilot import business_baseline
from app.services import material_content as store
from app.services import material_source_archive as archive


def save(path, value):
    """D盘报告先刷盘再原子替换；兼容Windows短暂读占用，不放宽原件校验。

    独立临时名避免并发报告互相改写。仅针对共享/访问冲突有界重试；磁盘满、
    其他I/O错误或持续不可写仍明确失败，不能在没有持久化凭据时继续删除。
    """
    from app.services.material_file_cleanup import physical_path

    path = physical_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(uuid4().hex+".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(9):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 8:
                    raise
                time.sleep(0.1*(attempt+1))
    finally:
        temporary.unlink(missing_ok=True)


def inventory(repo):
    """按明确基金组件限定源文件，排除运行快照和哈希偶然命中；缺失只登记。"""
    with repo.engine.connect() as c:
        c.execute(text("SET TRANSACTION READ ONLY"))
        rows = c.execute(
            text(
                "SELECT l.location,l.source_hash,b.format,b.source_size,b.content_hash FROM material_source_location l "
                "JOIN material_extracted_content b USING(source_hash) "
                "WHERE l.location LIKE '%008888%' ORDER BY l.location"
            )
        ).mappings()
        scoped = [dict(r) for r in rows if archive.fund_for(r["location"]) == "008888"]
    return {
        "fund_code": "008888",
        "files": [r for r in scoped if Path(r["location"]).is_file()],
        "absent_aliases_not_recovered": [r["location"] for r in scoped if not Path(r["location"]).is_file()],
        "scope_note": "008888自身及其研究目录里的目标ETF/对照资料；不是全部属于008888自身的公告",
    }


def capture_text(target, repo):
    """补齐全部原生文字和兼容排版；外链图片与PDF图表单独待核验，不能冒充完成。"""
    value = repo.get(target, fresh=True)
    raw = Path(target).read_bytes()
    if store.checksum(raw) != value.source_hash:
        raise ValueError("MATERIAL_ARCHIVE_SOURCE_CHANGED")
    from app.services.material_document import get_review
    from app.services.material_important_text import release

    published = release(value.source_hash, repo)
    if published is not None:
        reviewed = get_review(value.source_hash, repo=repo, policy=published["policy"])
        if not reviewed or not reviewed["ready"] or reviewed["document_hash"] != published["document_hash"]:
            raise ValueError("MATERIAL_ARCHIVE_IMPORTANT_REVIEW_CHANGED")
        return {"status": "IMPORTANT_INFORMATION_VERIFIED", "coverage": "IMPORTANT_INFORMATION",
                "policy": published["policy"]}
    if value.kind == "PDF":
        from pypdf import PdfReader

        reader = PdfReader(target)
        if len(reader.pages) != len(value.extracted["pdfium"]):
            raise ValueError("MATERIAL_ARCHIVE_PAGE_COUNT_MISMATCH")
        for index, page in enumerate(reader.pages):
            for engine, options in (("PYPDF_TEXT_V1", {}), ("PYPDF_LAYOUT_V1", {"extraction_mode": "layout"})):
                if repo.page(value.source_hash, engine, index) is None:
                    repo.page(value.source_hash, engine, index, page.extract_text(**options) or "")
        return {
            "status": "REFERENCE_PDF_FULL_NATIVE_TEXT_SAVED_FIGURES_UNVERIFIED",
            "pages": len(reader.pages),
            "title": value.extracted["pdfium"][0].strip().splitlines()[0],
        }
    try:
        body = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        body = raw.decode("gb18030")
    if value.kind == "JSON":
        json.loads(body, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        if store.compact_json(body) != value.decode():
            raise ValueError("MATERIAL_ARCHIVE_JSON_LEXEMES_CHANGED")
    elif value.kind == "TEXT":
        if body != value.decode():
            raise ValueError("MATERIAL_ARCHIVE_FULL_TEXT_CHANGED")
    elif value.kind == "HTML":
        from bs4 import BeautifulSoup

        original, stored = BeautifulSoup(body, "html.parser"), BeautifulSoup(value.decode(), "html.parser")
        # 旧HTML曾经由DOM重排而添加空白；仅比较时容许空白差异，原始全文另存不改写。
        if re.sub(r"\s+", "", original.get_text()) != re.sub(r"\s+", "", stored.get_text()):
            raise ValueError("MATERIAL_ARCHIVE_HTML_TEXT_CHANGED")
        exact = repo.page(value.source_hash, "EXACT_HTML_TEXT_V1", 0)
        if exact is None:
            repo.page(value.source_hash, "EXACT_HTML_TEXT_V1", 0, body)
        elif exact != body:
            raise ValueError("MATERIAL_ARCHIVE_EXACT_HTML_CHANGED")
        regions = [
            {"tag": e.name, "source": e.get("src"), "label": e.get("alt")}
            for e in original.find_all(["img", "svg", "canvas", "iframe", "object", "embed"])
        ]
        result = {
            "status": "HTML_TEXT_SAVED_EXTERNAL_REGIONS_UNVERIFIED" if regions else "COMPLETE_SOURCE_TEXT",
            "text": original.get_text("\n"),
            "tables": [
                [
                    [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                    for row in table.find_all("tr")
                ]
                for table in original.find_all("table")
            ],
            "links": [{"text": e.get_text(), "target": e.get("href")} for e in original.find_all("a", href=True)],
            "unverified_external_regions": regions,
        }
        repo.page(value.source_hash, "ARCHIVE_HTML_TEXT_V1", 0, store.encoded(result).decode())
        return {
            "status": result["status"],
            "unverified_external_regions": len(regions),
            "legacy_whitespace_difference": original.get_text() != stored.get_text(),
        }
    else:
        raise ValueError("MATERIAL_ARCHIVE_SOURCE_REQUIRES_EXPLICIT_PARSER")
    return {"status": "COMPLETE_SOURCE_TEXT", "characters": len(body)}


def resumed_item(row, repo, completed):
    """仅凭已提交的迁移标记和D盘原件恢复中断进度，不将普通缺失误认作已搬迁。"""
    target = (
        archive.root_path()
        / "originals"
        / row["source_hash"][:2]
        / (row["source_hash"] + archive.SUFFIX.get(row["format"], ".bin"))
    )
    archive._verify(target, row["source_hash"], row["source_size"])
    marker = store.checksum(store.encoded({"archive": str(target), "source_hash": row["source_hash"]}))
    with repo.engine.connect() as c:
        current = (
            c.execute(
                text("SELECT source_hash,deleted_at,verification FROM material_source_location WHERE location=:p"),
                {"p": row["location"]},
            )
            .mappings()
            .one()
        )
    if (
        current["source_hash"] != row["source_hash"]
        or current["deleted_at"] is None
        or current["verification"] != marker
    ):
        raise ValueError("MATERIAL_ARCHIVE_PLAN_SOURCE_MISSING_WITHOUT_COMMITTED_MOVE")
    if row["location"] in completed:
        return completed[row["location"]]
    # 仅在数据库提交与进度文件落盘之间中断时缺少原实体分配记录；宁可少报，不估算收益。
    return {
        "source_alias": row["location"],
        "archive_path": str(target),
        "source_hash": row["source_hash"],
        "format": row["format"],
        "size": row["source_size"],
        "removed_C_copy": True,
        "reclaimed_C_allocated_bytes": 0,
        "allocation_unknown_after_interrupted_progress_write": True,
    }


def execute(run):
    repo = store.repository()
    plan = inventory(repo)
    if (run / "result.json").exists():
        prior = json.loads((run / "result.json").read_text(encoding="utf-8"))
        for item in prior["items"]:
            if Path(item["source_alias"]).exists():
                raise ValueError("MATERIAL_ARCHIVE_COMPLETED_ALIAS_REAPPEARED")
            archive._verify(Path(item["archive_path"]), item["source_hash"], item["size"])
        return {"already_completed": True, "files": len(prior["items"])}
    if (run / "plan.json").exists():
        plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    else:
        save(run / "plan.json", plan)
    with repo.engine.connect() as c:
        c.execute(text("SET TRANSACTION READ ONLY"))
        before = business_baseline(c)
    free_before = shutil.disk_usage(store.PROJECT.anchor).free
    # 第一条删除前独立落盘；即使最终汇总失败，也不能丢掉本次整盘空闲量的前值。
    save(run / "space-before.json", {"C_free_before_bytes": free_before})
    checks = {}
    moved_path = run / "moved.json"
    completed = (
        {r["source_alias"]: r for r in json.loads(moved_path.read_text(encoding="utf-8"))}
        if moved_path.exists()
        else {}
    )
    # 所有原件先归档并补齐文本表示，尚未完成这一步时不批量移除C盘文件。
    for row in plan["files"]:
        p = Path(row["location"])
        item = (
            archive.relocate(p, source_hash=row["source_hash"], repo=repo)
            if p.exists()
            else resumed_item(row, repo, completed)
        )
        if row["source_hash"] not in checks:
            checks[row["source_hash"]] = capture_text(item["archive_path"], repo)
            if row["format"] == "PDF":
                print(
                    json.dumps(
                        {"reference_pdf_text_saved": row["source_hash"], "pages": checks[row["source_hash"]]["pages"]}
                    ),
                    flush=True,
                )
        save(run / "text-checks.json", checks)
    items = []
    for row in plan["files"]:
        item = (
            archive.relocate(row["location"], source_hash=row["source_hash"], delete_local=True, repo=repo)
            if Path(row["location"]).exists()
            else resumed_item(row, repo, completed)
        )
        items.append(item)
        save(run / "moved.json", items)
    # 禁止文件读回：旧C路径和D盘路径必须分别从数据库恢复相同完整载荷。
    with store.database_only():
        for row, item in zip(plan["files"], items, strict=True):
            left, right = store.source_bytes(row["location"]), store.source_bytes(item["archive_path"])
            if left.source_hash != row["source_hash"] or left.content_hash != row["content_hash"]:
                raise ValueError("MATERIAL_ARCHIVE_DB_READBACK_CHANGED")
            if store.encoded(left.extracted) != store.encoded(right.extracted):
                raise ValueError("MATERIAL_ARCHIVE_ALIAS_CONTENT_CHANGED")
            if left.kind == "PDF":
                with store.document_reader(left) as doc:
                    for page in doc.pages:
                        page.extract_text(extraction_mode="layout")
    with repo.engine.connect() as c:
        c.execute(text("SET TRANSACTION READ ONLY"))
        after = business_baseline(c)
    if before != after:
        raise ValueError("MATERIAL_ARCHIVE_BUSINESS_CHANGED_DURING_VERIFICATION")
    result = {
        "fund_code": "008888",
        "files_moved": len(items),
        "unique_archived_sources": len({r["source_hash"] for r in items}),
        "C_source_files_remaining_in_plan": sum(Path(r["location"]).exists() for r in plan["files"]),
        "already_deleted_sources_recovered": 0,
        "items": items,
        "business_before": before,
        "business_after": after,
        "business_unchanged": before == after,
        "database_only_readback_passed": True,
        "text_statuses": dict(Counter(r["status"] for r in checks.values())),
        "C_file_allocated_bytes_reclaimed": sum(r["reclaimed_C_allocated_bytes"] for r in items),
        "C_free_before_bytes": free_before,
        "C_free_after_bytes": shutil.disk_usage(store.PROJECT.anchor).free,
        "archive_logical_bytes_unique": sum({r["source_hash"]: r["size"] for r in items}.values()),
        "unverified_reference_figures_are_not_marked_complete": True,
        "global_cleanup_pause_changed": False,
    }
    save(run / "result.json", result)
    return {k: v for k, v in result.items() if k not in {"items", "business_before", "business_after"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from app.services.material_file_access import legacy_workflow_error

    legacy_workflow_error()
    run = args.run or archive.root_path() / "runs/008888/20261105"
    if args.execute:
        result = execute(run)
    else:
        plan = inventory(store.repository())
        save(run / "inventory.json", plan)
        result = {
            "existing_files": len(plan["files"]),
            "unique_sources": len({r["source_hash"] for r in plan["files"]}),
            "formats": dict(Counter(r["format"] for r in plan["files"])),
            "absent_aliases_not_recovered": len(plan["absent_aliases_not_recovered"]),
        }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
