"""主动表现组件的prepare/snapshot适配候选；不采集、不导入外部路径、不提交数据库。"""

from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from app.services.material_content import content_digest, source_bytes, source_exists

SCOPE = {"return", "risk", "stability"}
TOLERANCE = Decimal("1e-55")


def _plain(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def _number(value):
    if value is None or isinstance(value, bool):
        raise ValueError("R5_ACTIVE_PERFORMANCE_POINT_REQUIRED")
    point = Decimal(str(value))
    if not point.is_finite() or not 0 <= point <= 100:
        raise ValueError("R5_ACTIVE_PERFORMANCE_POINT_RANGE")
    return point


def _bytes(ref, root):
    """父级已canonical_source；此处仅允许正式root/sources内的原件。"""
    path = (root / ref["file"]).resolve()
    if not path.is_relative_to(root / "sources") or not source_exists(path):
        raise ValueError("R5_ACTIVE_PERFORMANCE_CANONICAL_SOURCE_REQUIRED")
    raw = source_bytes(path)
    if content_digest(raw) != ref["sha256"]:
        raise ValueError("R5_ACTIVE_PERFORMANCE_SOURCE_HASH")
    return raw


def _json(ref, root):
    return json.loads(_bytes(ref, root).decode("utf-8-sig"))


def _scope(inputs):
    selected = inputs.get("selected_dimensions")
    if (
        not isinstance(selected, list)
        or not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= SCOPE
    ):
        raise ValueError("R5_ACTIVE_PERFORMANCE_SELECTED_DIMENSIONS")
    return frozenset(selected)


def _packets(bundle):
    result = bundle.get("active_performance_calculations", [])
    if not isinstance(result, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("inputs"), dict) for row in result
    ):
        raise ValueError("R5_ACTIVE_PERFORMANCE_PACKET_LIST_REQUIRED")
    return result


def check_binding(bundle, inputs, *, root, material_reviewer=None, archive_reader=None, period_reader=None):
    """绑定fresh底稿、正式PDF与同报告转载锚点，不能把跨格式哈希当同一字节。

    target的精确新增键为manager_name、holding_date、primary_report_ref、
    material_primary_report_ref。主PDF仍是当前底稿原件；转载锚必须同法律产品、
    管理人、期间、原公开日且关键引文在已封存的主PDF解析文本中出现。
    """
    from app.services.fund_rating_reviewed_snapshot import BASIS

    root = Path(root).resolve()
    selected = _scope(inputs)
    target = inputs.get("target", {})
    if (
        bundle.get("basis") != BASIS
        or bundle.get("family") != "ACTIVE_EQUITY"
        or bundle.get("share_class") != "C"
        or inputs.get("fund_code") != bundle.get("fund_code")
        or inputs.get("as_of_date") != str(bundle.get("as_of_date"))
        or inputs.get("share_class") != bundle.get("share_class")
        or target.get("fund_code") != bundle.get("fund_code")
        or target.get("share_class") != "C"
        or target.get("legal_name") != bundle.get("identity", {}).get("legal_name")
        or target.get("manager_name") != bundle.get("identity", {}).get("manager_name")
        or target.get("holding_date") != str(bundle.get("holding_date"))
        or target.get("primary_report_ref") != bundle.get("original_report_archive", {}).get("sha256")
        or any(
            bundle.get(key)
            for key in ("r5_calculation", "passive_components_calculation", "passive_performance_calculation")
        )
    ):
        raise ValueError("R5_ACTIVE_PERFORMANCE_INPUT_SCOPE")
    if bundle.get("stability_calculation") and "stability" in selected:
        raise ValueError("R5_ACTIVE_PERFORMANCE_LEGACY_STABILITY_SCOPE_CONFLICT")
    for digest, ref in inputs["sources"].items():
        path = (root / ref["file"]).resolve()
        if ref.get("sha256") != digest or not path.is_relative_to(root / "sources"):
            raise ValueError("R5_ACTIVE_PERFORMANCE_CANONICAL_SOURCE_REQUIRED")
    if material_reviewer is None:
        from app.services.fund_rating_active_performance import _require_material_review as material_reviewer
    if archive_reader is None:
        from app.services.fund_rating_reviewed_snapshot import _read as archive_reader
    if period_reader is None:
        from app.services.fund_quality_v2_disclosures import report_period as period_reader
    material = material_reviewer(inputs, root)
    if material.get("pending") is not False or material.get("review_closed") is not True:
        raise ValueError("R5_ACTIVE_PERFORMANCE_MATERIAL_REVIEW_REQUIRED")
    primary = target["primary_report_ref"]
    if primary not in inputs["sources"]:
        raise ValueError("R5_ACTIVE_PERFORMANCE_PRIMARY_SOURCE_REQUIRED")
    # 每次核主PDF字节；沿用当前正式底稿已封存、来源哈希相同的全文解析证据。
    _bytes(bundle["original_report_archive"], root)
    _bytes(inputs["sources"][primary], root)
    identity = archive_reader(bundle["identity_archive"], root)
    if (
        not identity
        or identity.get("identity_verified") is not True
        or identity.get("code") != bundle["fund_code"]
        or identity.get("source", {}).get("sha256") != primary
        or not isinstance(identity.get("pages"), list)
    ):
        raise ValueError("R5_ACTIVE_PERFORMANCE_PRIMARY_IDENTITY_ARCHIVE")
    primary_text = "\n".join(identity["pages"])
    identity_published = identity.get("published_on")
    if (
        not identity_published
        or identity_published > inputs["as_of_date"]
        or identity_published != bundle.get("report_published_on")
        or str(period_reader(primary_text)[1]) != target["holding_date"]
        or any(
            _plain(value) not in _plain(primary_text)
            for value in [target["fund_code"], target["legal_name"], target["manager_name"]]
        )
    ):
        raise ValueError("R5_ACTIVE_PERFORMANCE_PRIMARY_PERIOD_OR_IDENTITY")
    review_ref = inputs.get("material_event_review_ref")
    if review_ref not in inputs["sources"]:
        raise ValueError("R5_ACTIVE_PERFORMANCE_MATERIAL_REVIEW_SOURCE_REQUIRED")
    review = _json(inputs["sources"][review_ref], root)
    material_primary = target.get("material_primary_report_ref")
    selected_anchors = target.get("material_primary_anchor_ids")
    if (
        not isinstance(selected_anchors, list)
        or not selected_anchors
        or len(set(selected_anchors)) != len(selected_anchors)
        or not set(selected_anchors) <= set(review.get("anchors", []))
    ):
        raise ValueError("R5_ACTIVE_PERFORMANCE_PRIMARY_ANCHOR_IDS_REQUIRED")
    anchors = [
        row
        for row in review.get("evidence", [])
        if row.get("id") in selected_anchors and row.get("source_ref") == material_primary
    ]
    if len(anchors) != len(selected_anchors) or material_primary not in inputs["sources"]:
        raise ValueError("R5_ACTIVE_PERFORMANCE_CURRENT_REPORT_ANCHOR_REQUIRED")
    report = _json(inputs["sources"][material_primary], root)
    data = report.get("data") or {}
    codes = {str(row.get("stock")) for row in data.get("security", [])}
    text = data.get("notice_content", "")
    published = str(data.get("notice_date", ""))[:10]
    if (
        report.get("success") != 1
        or not codes.intersection(inputs["target_aliases"])
        or published != identity_published
        or published > inputs["as_of_date"]
        or str(period_reader(text)[1]) != target["holding_date"]
        or any(_plain(value) not in _plain(text) for value in [target["legal_name"], target["manager_name"]])
        or any(
            not row.get("quote")
            or _plain(row["quote"]) not in _plain(text)
            or _plain(row["quote"]) not in _plain(primary_text)
            for row in anchors
        )
    ):
        raise ValueError("R5_ACTIVE_PERFORMANCE_MATERIAL_PRIMARY_REPORT_CONFLICT")
    return {
        "fund_code": bundle["fund_code"],
        "selected_dimensions": sorted(selected),
        "primary_report_ref": primary,
        "material_primary_report_ref": material_primary,
        "holding_date": target["holding_date"],
        "published_on": published,
        "cross_format_anchor_count": len(anchors),
    }


def validate_input_archives(bundle, *, root, archive_reader=None):
    """来源校验阶段逐包对比input_archive与inline输入，防止替换计算参数。"""
    if archive_reader is None:
        from app.services.fund_rating_reviewed_snapshot import _read as archive_reader
    for packet in _packets(bundle):
        if not packet.get("input_archive") or archive_reader(packet["input_archive"], root) != packet["inputs"]:
            raise ValueError("R5_ACTIVE_PERFORMANCE_INPUT_ARCHIVE_MISMATCH")


def replay_components(bundle, dimensions, *, root, compute=None, binding_checker=None):
    """snapshot逐包重放原件，比较所选维的lower/upper/candidate，拒绝重叠作用域。"""
    if compute is None:
        from app.services.fund_rating_active_performance import calculate as compute
    if binding_checker is None:
        binding_checker = check_binding
    seen = set()
    points = {}
    for packet in _packets(bundle):
        raw = packet["inputs"]
        selected = _scope(raw)
        if seen.intersection(selected):
            raise ValueError("R5_ACTIVE_PERFORMANCE_DUPLICATE_SCOPE")
        seen.update(selected)
        binding_checker(bundle, raw, root=root)
        calculated = compute(raw, root=root)
        if (
            set(calculated.get("dimensions", {})) != selected
            or calculated.get("complete_score") is not None
            or calculated.get("grade") is not None
        ):
            raise ValueError("R5_ACTIVE_PERFORMANCE_RECALCULATION_SCOPE")
        for key in selected:
            point = _number(calculated["dimensions"][key])
            if any(
                dimensions[key].get(bound) is None or abs(_number(dimensions[key][bound]) - point) > TOLERANCE
                for bound in ("lower", "upper", "candidate")
            ):
                raise ValueError("R5_ACTIVE_PERFORMANCE_DIMENSION_RECALCULATION:" + key)
            points[key] = str(point)
    return points


def prepare_active_performance(
    session,
    request,
    *,
    root,
    current_reader=None,
    compute=None,
    archiver=None,
    source_validator=None,
    snapshot_calculator=None,
    binding_checker=None,
):
    """fresh读取后原件计算并追加组件，保留其余维、verified_at、有效期和原历史。

    依赖注入仅用于隔离集成测试；正式调用不传入这些参数。事务提交与登记由
    既有发布层负责，本函数不提交。两包以selected_dimensions的精确集合作为scope。
    """
    from app.services.fund_rating_reviewed_snapshot import BASIS

    if current_reader is None:
        from app.services.fund_rating_pipeline import current_inputs as current_reader
    if compute is None:
        from app.services.fund_rating_active_performance import calculate as compute
    if archiver is None:
        from app.services.fund_rating_pipeline import archive_json as archiver
    if source_validator is None:
        from app.services.fund_rating_reviewed_snapshot import validate_sources as source_validator
    if snapshot_calculator is None:
        from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculator
    if binding_checker is None:
        binding_checker = check_binding
    root = Path(root).resolve()
    code = request.get("fund_code")
    if not isinstance(code, str) or not re.fullmatch(r"\d{6}", code):
        raise ValueError("R5_ACTIVE_PERFORMANCE_FUND_CODE")
    previous = current_reader(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request.get("as_of_date"):
        raise ValueError("R5_ACTIVE_PERFORMANCE_CURRENT_PERIOD_REQUIRED")
    members = [row["bundle"] for row in previous["input_snapshot"]["members"] if row["fund_code"] == code]
    if len(members) != 1:
        raise ValueError("R5_ACTIVE_PERFORMANCE_CURRENT_MEMBER_REQUIRED")
    bundle = deepcopy(members[0])
    raw = deepcopy(request["inputs"])
    selected = _scope(raw)
    if raw.get("as_of_date") != request["as_of_date"]:
        raise ValueError("R5_ACTIVE_PERFORMANCE_REQUEST_PERIOD_CONFLICT")
    binding_checker(bundle, raw, root=root)
    explanation = request.get("explanation", "").strip()
    if len(explanation) < 40:
        raise ValueError("R5_ACTIVE_PERFORMANCE_EXPLANATION_REQUIRED")
    value = compute(raw, root=root)
    if (
        set(value.get("dimensions", {})) != selected
        or value.get("complete_score") is not None
        or value.get("grade") is not None
    ):
        raise ValueError("R5_ACTIVE_PERFORMANCE_RECALCULATION_SCOPE")
    document = deepcopy(bundle["reviewed_document"])
    for key in selected:
        point = str(_number(value["dimensions"][key]))
        document["dimensions"][key] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
        bundle.setdefault("public_explanations", {})[key] = explanation
        bundle.get("public_metrics", {}).pop(key, None)
    packages = []
    for packet in _packets(bundle):
        existing = _scope(packet["inputs"])
        if existing == selected:
            continue
        if existing.intersection(selected):
            raise ValueError("R5_ACTIVE_PERFORMANCE_OVERLAPPING_SCOPE_REQUIRES_REVIEW")
        packages.append(deepcopy(packet))
    packages.append({"inputs": raw, "input_archive": archiver(root, raw)})
    bundle.update(
        active_performance_calculations=packages,
        research_format="single",
        reviewed_document=document,
        research_archive=archiver(root, document),
    )
    extras = {row["sha256"]: row for row in bundle.get("extra_archives", [])}
    for digest, ref in raw["sources"].items():
        extras[digest] = deepcopy(ref)
    bundle["extra_archives"] = list(extras.values())
    source_validator(bundle, datetime.now(UTC), root=root)
    checked = snapshot_calculator(bundle, root=root)
    if checked.get("status") != "PARTIAL" or checked.get("score") is not None or checked.get("grade") is not None:
        raise ValueError("R5_ACTIVE_PERFORMANCE_COMPONENT_CANNOT_CLAIM_COMPLETE")
    return {
        "basis": BASIS,
        "entries": [{key: value for key, value in bundle.items() if key not in {"basis", "reviewed_document"}}],
    }
