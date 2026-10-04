"""002112 本期研究公共边界：显式输出目录、不可变证据与累计预算。

不导入旧实验的执行入口，不初始化业务服务，不写数据库或现用模型。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from scripts import fund_002112_existing_data_inputs_v1 as nav_inputs
from scripts import fund_002112_existing_events_sources_v1 as io
from scripts import fund_002112_recent_bodies_v1 as bodies

OLD = io.RESEARCH / "semantic-holdings-optimization/20261001-v1"
DEFAULT_ROOT = io.RESEARCH / "signal-optimization/20261001-v1"
CLASSES = ("DOWN", "FLAT", "UP")
LIMITS = {"fit": 80, "public": 2000, "llm": 1000, "new_body": 500}
__all__ = ["nav_inputs"]


def lines(path):
    """只按物理换行解析；原文内的 Unicode 分段符不是 JSONL 换行。"""
    p = Path(path)
    with p.open(encoding="utf-8-sig") if p.exists() else _empty() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _empty():
    from io import StringIO

    return StringIO("")


def moment(value):
    point = datetime.fromisoformat(value)
    if point.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED")
    return point.astimezone(io.ZONE)


def stage(root, name, state, evidence, unresolved=None, next_step=None):
    io.append(
        Path(root) / "stage-status.jsonl",
        {
            "at": io.now(),
            "stage": name,
            "state": state,
            "evidence": evidence,
            "unresolved": unresolved or [],
            "next": next_step,
        },
    )


def reserve(root, kind, identifier, metadata=None):
    """先记账再请求/拟合；中断也占额度。外层调用者持有整个批次写锁。"""
    ledger = Path(root) / (kind + "-ledger.jsonl")
    previous = lines(ledger)
    if any(r["id"] == identifier for r in previous):
        raise ValueError("ATTEMPT_ALREADY_RESERVED:" + identifier)
    if len(previous) >= LIMITS[kind]:
        raise ValueError("BUDGET_EXHAUSTED:" + kind)
    row = {"at": io.now(), "id": identifier, "attempt": len(previous) + 1, **(metadata or {})}
    io.append(ledger, row)
    return row


def register_sources(root, paths, filename="source-manifest.json"):
    files = {str(Path(p).resolve()): io.sha(p) for p in sorted(set(map(str, paths)))}
    io.save(Path(root) / filename, {"files": files})
    return files


def verify_files(manifest):
    for p, expected in manifest["files"].items():
        if io.sha(p) != expected:
            raise ValueError("SOURCE_CHANGED:" + p)


def choose_report(reports, cutoff):
    return bodies.choose_report(reports, cutoff)


def original_rows():
    return lines(OLD / "feature-revision-r3/inputs.jsonl")


def bundle():
    return io.read(bodies.BUNDLE)
