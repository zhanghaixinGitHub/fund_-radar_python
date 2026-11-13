"""按正式批次保存评价事实；页面只读，生成命令只写D盘展示文件。"""

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from app.services.fund_rating_inputs import canonical, digest
from app.services.fund_rating_narrative import REVISION, render

MAX_BYTES = 128 * 1024


def _binding(result, snapshot):
    return {
        "fund_code": result.fund_code,
        "batch_id": result.batch_id,
        "snapshot_hash": digest(snapshot),
        "result_hash": digest(
            {
                "status": result.status,
                "score": str(result.score),
                "grade": result.grade,
                "dimensions": result.dimension_values,
                "public": result.public_evidence,
            }
        ),
    }


def _path(root, result):
    if not re.fullmatch(r"[a-f0-9]{64}", result.batch_id) or not re.fullmatch(r"[0-9]{6}", result.fund_code):
        raise ValueError("NARRATIVE_IDENTITY")
    return Path(root).resolve() / "narratives" / f"{result.batch_id}-{result.fund_code}.json"


def load(result, snapshot, *, root):
    path = _path(root, result)
    try:
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("NARRATIVE_SIZE")
        packet = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if packet.get("revision") != REVISION:
        return None
    if packet.get("binding") != _binding(result, snapshot) or packet.get("facts_hash") != digest(packet.get("facts")):
        raise ValueError("NARRATIVE_BINDING_OR_CONTENT")
    return packet["facts"]


def prepare(result, snapshot, *, root, force=False):
    """只编译已发布/待同事务发布结果的解释，不修改数据库对象或评分快照。"""
    if not force and load(result, snapshot, root=root) is not None:
        return {"fund_code": result.fund_code, "state": "REUSED"}
    matches = [m for m in snapshot.get("members", []) if m.get("fund_code") == result.fund_code]
    if len(matches) != 1:
        raise ValueError("NARRATIVE_MEMBER_IDENTITY")
    member = matches[0]
    if member.get("evidence_hash") != digest(member["bundle"]):
        raise ValueError("NARRATIVE_MEMBER_HASH")
    bundle = member["bundle"]
    from app.services.fund_rating_narrative_facts import compile_facts

    facts = compile_facts(bundle, result.dimension_values or {}, root=Path(root))
    from app.services.fund_rating_narrative_methods import attach

    facts = attach(bundle, result.dimension_values or {}, facts, root=Path(root))
    packet = {
        "revision": REVISION,
        "binding": _binding(result, snapshot),
        "facts": facts,
        "facts_hash": digest(facts),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    content = canonical(packet).encode("utf-8")
    if len(content) > MAX_BYTES:
        raise ValueError("NARRATIVE_SIZE")
    path = _path(root, result)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件与目标在同一D盘目录；原子替换避免页面读到半份JSON。
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + "-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if path.read_bytes() != content:
        raise OSError("NARRATIVE_WRITE_READBACK_FAILED")
    return {"fund_code": result.fund_code, "state": "GENERATED", "facts_hash": packet["facts_hash"]}


def enrich(item, result, snapshot, *, root):
    """已通过现有时效/撤回/事项门禁之后才允许叠加评价。"""
    if item.get("status") not in {"RATED", "SCORED", "PARTIAL"}:
        return {**item, "dimensions": [{**d, "assessment": None} for d in item.get("dimensions", [])]}
    if not getattr(result, "batch_id", None):
        return item
    facts = load(result, snapshot, root=root)
    bundle = next(m["bundle"] for m in snapshot["members"] if m["fund_code"] == result.fund_code)
    if facts is None:
        # 新增基金尚未生成展示文件时，也按同批正式说明组织八维；GET不读取原件。
        from app.services.fund_rating_narrative_adapters import base_facts

        if not all(bundle.get(key) for key in ("as_of_date", "holding_date", "share_class", "family")):
            return item
        facts = base_facts(bundle)
    return render(item, facts, bundle["family"])
