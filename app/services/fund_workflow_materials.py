"""本机补充资料的交接凭据；保留采集缺口，不以路径存在代替资料完整。"""

from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save


def workflow_root(root=None) -> Path:
    return Path(root) if root is not None else Path(get_settings().fund_insight_directory) / "market-workflow"


def receipt_path(code: str, root=None) -> Path:
    from app.services.fund_market_preparation import validate_codes

    validate_codes([code])
    return workflow_root(root) / "comprehensive-inputs" / (code + ".json")


def registered_materials(code: str, *, root=None, at=None):
    """只复用当前来源版本的交接；路径、正文、基金身份或取得时间变化都重新审阅。"""
    path = receipt_path(code, root)
    if not path.exists():
        return None
    receipt = read(path)
    material = read(workflow_root(root) / "materials" / (code + ".json"))
    candidate = Path(receipt["materials_path"])
    payload = read(candidate)
    reviewed = datetime.fromisoformat(receipt["checked_at"])
    if (
        receipt.get("fund_code") != code
        or payload.get("fund_code") != code
        or receipt.get("workflow_source_hash") != material["source_hash"]
        or receipt.get("workflow_materials_hash") != digest(material)
        or receipt.get("materials_hash") != digest(payload)
        or reviewed.tzinfo is None
        or (at is not None and reviewed > at)
    ):
        raise ValueError("MARKET_COMPREHENSIVE_HANDOFF_CHANGED")
    return receipt


def input_gaps(packet: dict) -> list[str]:
    """复核原综合政策的必需维度和实际基准日，不把 prepare 成功当成可以预测。"""
    from app.services.comprehensive_prediction_policy import policy

    gaps = []
    source_ids = {d["id"] for d in packet["documents"]} | set(packet["facts"])
    source_ids.add(packet["report"].get("sourceUrl"))
    documents = {d["id"]: d for d in packet["documents"]}
    for dimension in policy()["required_dimensions"]:
        row = packet.get("comprehensive_coverage", {}).get(dimension, {})
        allowed = {"AVAILABLE", "VERIFIED_EMPTY"} if dimension == "DISCLOSURES" else {"AVAILABLE"}
        refs = row.get("source_refs", [])
        if row.get("status") not in allowed or not refs or not set(refs) <= source_ids:
            gaps.append(f"综合必需资料待补充：{dimension}")
        elif dimension == "DISCLOSURES":
            kind = "ANNOUNCEMENT" if row["status"] == "AVAILABLE" else "DISCLOSURE_SEARCH"
            if any(ref not in documents or documents[ref].get("kind") != kind for ref in refs):
                gaps.append("本次公告原文或检索回执仍待核验")
    if not packet["time_context"]["baseline_complete"]:
        gaps.append("本基金实际基准日净值或所列行情仍未齐备")
    return gaps


def unresolved_gaps(material: dict, packet: dict, resolutions: list[dict]) -> list[str]:
    """逐项核对人工审阅的补充证据；只影响预测用途，原采集和评级缺口不改写。

    每条原缺口必须有明确的用途判断、连续原文引用及足够的解释。字节匹配只能
    证明引用真实，证据是否充分仍由实际阅读的聊天负责，不能批量自动填就绪标记。
    """
    original = set(material.get("gaps", []))
    documents = {d["id"]: d for d in packet["documents"]}
    resolved = set()
    for row in resolutions:
        gap = row.get("gap")
        evidence = row.get("evidence", [])
        if (
            gap not in original
            or gap in resolved
            or row.get("decision") not in {"RESOLVED", "NOT_APPLICABLE_TO_PREDICTION"}
            or len(row.get("explanation", "").strip()) < 80
            or not evidence
        ):
            raise ValueError("MARKET_COMPREHENSIVE_RESOLUTION_INVALID")
        for proof in evidence:
            document = documents.get(proof.get("document_id"))
            quote = proof.get("quote")
            if not document or not isinstance(quote, str) or len(quote.strip()) < 20 or quote not in document["body"]:
                raise ValueError("MARKET_COMPREHENSIVE_RESOLUTION_SOURCE_INVALID")
        resolved.add(gap)
    return [gap for gap in material.get("gaps", []) if gap not in resolved]


def inspect_materials(code: str, materials_path, *, root=None, clock=None, resolutions=None) -> dict:
    """登记本基金补充包并读回缺项；不批准审阅、不生成结论、不覆盖采集原件。"""
    from app.services.fund_exposure_common import now
    from app.services.fund_information_snapshot import prepare
    from app.services.fund_market_review import pending_prediction_events

    current = (clock or now)()
    root = workflow_root(root)
    material = read(root / "materials" / (code + ".json"))
    path = Path(materials_path).resolve()
    payload = read(path)
    if payload.get("fund_code") != code:
        raise ValueError("COMPREHENSIVE_FUND_MISMATCH")
    resolutions = resolutions or []
    try:
        packet = prepare(current, code, path)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        # 日历等契约不适配也留存真实路径和失败，后续不能又误报成默认目录没文件。
        # 整体未通过时仍不批准；仅可独立交接下方严格核验过的本基金公告来源。
        failure = str(error)
        gaps = list(material.get("gaps", []))
        if failure == "COMPREHENSIVE_CALENDAR_POLICY_REQUIRED" and resolutions:
            from app.services.fund_comprehensive_materials import checked_documents

            # 只允许日历失败时单独交接公告原文。仍使用正式冻结的来源校验，
            # 并要求证据属于本基金的公告或目录；不能据此清除净值、行情、
            # 持仓和经营缺口，更不能消除下方保留的日历失败或批准综合审阅。
            try:
                documents = {d["id"]: d for d in checked_documents(current, code, payload)}
            except (ValueError, KeyError, TypeError, IndexError):
                documents = {}
            source_only = [
                row
                for row in resolutions
                if row.get("gap") == "本次公告目录或原文尚未核验完整"
                and row.get("evidence")
                and all(
                    proof.get("document_id") in documents
                    and documents[proof["document_id"]].get("kind") in {"ANNOUNCEMENT", "DISCLOSURE_SEARCH"}
                    and code in documents[proof["document_id"]].get("codes", [])
                    for proof in row["evidence"]
                )
            ]
            if source_only:
                gaps = unresolved_gaps(material, {"documents": list(documents.values())}, source_only)
        gaps += ["综合资料校验未通过：" + failure]
    else:
        failure = None
        gaps = unresolved_gaps(material, packet, resolutions) + input_gaps(packet)
    if pending_prediction_events(material.get("candidate_events", []), material["source_hash"], root=root):
        gaps.append("本期新增事项尚需结合原文复核")
    receipt = {
        "fund_code": code,
        "materials_path": str(path),
        "materials_hash": digest(payload),
        "workflow_source_hash": material["source_hash"],
        "workflow_materials_hash": digest(material),
        "checked_at": current.isoformat(),
        "resolutions": resolutions,
        "gaps": list(dict.fromkeys(gaps)),
        "reviewed": False,
        "prepare_failure": failure,
    }
    versioned_save(receipt_path(code, root), receipt)
    return receipt
