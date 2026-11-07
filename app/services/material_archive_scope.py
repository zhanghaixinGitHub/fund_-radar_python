"""全基金存量迁移的文件边界；原件与冻结输入/业务结果分别处理。"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

SOURCE_DIRECTORIES = {"sources", "raw", "company-sources"}
SKIP_DIRECTORIES = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
SOURCE_SUFFIXES = {
    "", ".pdf", ".json", ".html", ".htm", ".txt", ".csv", ".xml", ".js", ".raw", ".bin",
    ".doc", ".docx", ".xls", ".xlsx", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".download", ".md", ".markdown",
}
FROZEN_DIRECTORIES = {"frozen", "snapshot-candidates", "snapshots", "snapshot", "revisions", "models", "predictions",
                      "outcomes", "before-files", "after-files", "backup", "backups", "checkpoints", "checkpoint",
                      "input-snapshots", "office-output"}
DOCUMENT_SUFFIXES = {".pdf", ".doc", ".docx", ".xls", ".xlsx"}
VERIFIED_LEGACY_RESPONSES = {
    ("fxi-data-feasibility-v1", "response.xls"),
    ("cnya-data-feasibility-v1", "response.xls"),
    ("ecb-fx-feasibility-v1", "response.zip"),
}
_approved_pdf_inputs = ContextVar("explicitly_approved_pdf_inputs", default=frozenset())


@contextmanager
def approved_pdf_input_scope(items):
    """只开放用户另行授权的精确PDF清单，退出或异常即恢复默认保护。

    不从外部文件/环境变量自动推定授权。调用方提供冻结的路径、PDF类型、
    原长度与SHA256；真正迁移仍由C/D实体锁及数据库复核，不能靠分类直接删除。
    快照目录、程序依赖、非PDF及未列出的评测输入继续受原保护规则约束。
    """
    selected = set()
    for item in items:
        path = Path(item["path"]).absolute()
        if path.suffix.lower() != ".pdf" or item.get("kind") != "PDF" or item.get("size", 0) <= 0 \
                or not re.fullmatch(r"[0-9a-f]{64}", item.get("source_hash", "")):
            raise ValueError("MATERIAL_EXPLICIT_PDF_APPROVAL_INVALID")
        selected.add(os.path.normcase(str(path)))
    if not selected:
        raise ValueError("MATERIAL_EXPLICIT_PDF_APPROVAL_EMPTY")
    token = _approved_pdf_inputs.set(frozenset(selected))
    try:
        yield
    finally:
        _approved_pdf_inputs.reset(token)


def named_collection_document(parts, suffix, *, indexed):
    """旧采集器给原件使用业务文件名，不能仅因不叫哈希文件而永久留C盘。

    只接管已确认的采集用途：基金原件下载批次、公司primary-ir/new-primary、
    旧Word转换的输入副本。模型验收造出的PDF、转换输出和训练证据冻结包不在此项。
    """
    if suffix not in DOCUMENT_SUFFIXES or len(parts) < 3:
        return False
    if set(parts[:-1]) & {"new-primary", "coordinator-sources", "primary-ir", "office-input", "corrupt-sources"}:
        return True
    run = parts[1]
    downloaded_name = bool(re.match(
        r"^(?:official-|half[-_.]|prospectus[-_.]|summary[-_.]|dividend[-_]|link[-.]|target[-.]|yh-|hkex-|cifm-|wj-|bdfund-)",
        parts[-1], re.I))
    return (run.startswith(("fund-rating-", "fund-video-", "watchlist-one-day-")) and downloaded_name
            or indexed and run.startswith("quality-v2-") and parts[-1] == "original.pdf")


@lru_cache(maxsize=64)
def _scope_root(root):
    """资格筛选复用稳定项目根；真正删除仍由文件句柄重新校验物理路径和实体。"""
    return Path(root).resolve()


def classify(path, root, *, indexed=False, prefix=b""):
    """只迁确定的采集原件，保留历史快照、实验结果和程序依赖。

    即使内容摘要相同，保护目录里的冻结输入也不因原件去重而删除。应用自有的
    hash/payload封装是证据或运行记录，不能仅因位于raw目录就当作远程响应迁走。
    这里返回资格及理由；删除还须冻结精确路径、重新核对实体与D盘副本。
    """
    path, root = Path(path).absolute(), _scope_root(str(root))
    if not path.is_relative_to(root):
        return False, "OUTSIDE_PROJECT"
    parts = path.relative_to(root).parts
    if not parts or parts[0] not in {"data", ".local-runs"}:
        return False, "NOT_RUNTIME_MATERIAL_DIRECTORY"
    if any(p in SKIP_DIRECTORIES or p.startswith("deps") or p.endswith("-deps") for p in parts[:-1]):
        return False, "CODE_OR_DEPENDENCY"
    if any(p in FROZEN_DIRECTORIES for p in parts[:-1]):
        return False, "HISTORICAL_SNAPSHOT_OR_EXPERIMENT_OUTPUT"
    if path.suffix.lower() not in SOURCE_SUFFIXES:
        return False, "NOT_SOURCE_FORMAT"
    if path.name.lower().endswith((".receipt.json", ".request.json", ".manifest.json", ".metadata.json")):
        return False, "COLLECTION_RECEIPT_OR_METADATA"
    in_source_directory = bool(set(parts[:-1]) & SOURCE_DIRECTORIES)
    named_response = bool(re.fullmatch(r"(?:raw|response)(?:[-.].*)?", path.name, re.I))
    addressed_document = indexed and bool(re.fullmatch(r"[0-9a-f]{64}(?:\.[^.]+)?", path.name))
    named_document = named_collection_document(parts, path.suffix.lower(), indexed=indexed)
    approved_input = path.suffix.lower() == ".pdf" and os.path.normcase(str(path)) in _approved_pdf_inputs.get()
    # 复查采集脚本及原始SHA256回执确认的三个旧HTTP原件，未曾登记到资料库。
    # 精确限定旧批次/直接子目录/文件名；不把同名实验结果、冻结输入或其他下载推定为原件。
    legacy_response = len(parts) == 4 and parts[:2] == (".local-runs", "direction-1d-sprint-20260914") \
        and (parts[2], parts[3]) in VERIFIED_LEGACY_RESPONSES
    if not (in_source_directory or named_response and indexed or addressed_document or named_document
            or legacy_response or approved_input):
        return False, "RESULT_OR_UNCONFIRMED_SOURCE"
    if prefix:
        text = prefix.decode("utf-8-sig", errors="replace")
        if re.match(r'^\s*\{\s*"hash"\s*:', text) and '"payload"' in text:
            return False, "APPLICATION_EVIDENCE_ENVELOPE"
        if '"sha256"' in text and any(key in text for key in ('"received_at"', '"expires_at"')):
            return False, "COLLECTION_RECEIPT_OR_METADATA"
        if any(marker in text[:4096] for marker in (
            '"CODEX_FUND_INPUT_V1"', '"CODEX_002112_INPUT_V1"', '"CODEX_COMPREHENSIVE_MULTI_V1"',
            '"CODEX_FUND_RATING_', '"CODEX_FUND_PREDICTION_',
        )):
            return False, "FROZEN_BUSINESS_INPUT_OR_RESULT"
    return True, "COLLECTION_SOURCE" if in_source_directory else (
        "EXPLICITLY_APPROVED_EVALUATION_PDF" if approved_input else
        "VERIFIED_LEGACY_RESPONSE" if legacy_response else
        "NAMED_COLLECTION_DOCUMENT" if named_document else "INDEXED_ORIGINAL_RESPONSE")
