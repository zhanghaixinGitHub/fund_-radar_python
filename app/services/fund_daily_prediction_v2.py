"""每日第二版适配：新一期等待真实审阅，旧研究不能冒充今日预测。"""

from pathlib import Path

from app.services.direction_1d_protocol import digest
from app.services.direction_1d_protocol import window as planning_window
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save
from app.services.prediction_daily_guidance import review_guidance
from app.services.prediction_review_v2 import policy_v2


def adoption():
    from app.services.prediction_review_v2_publication import adoption_status

    return adoption_status()


def saved_records(owner, codes, target):
    from app.services.prediction_review_v2_publication import daily_records

    return daily_records(owner, codes, target)


def matches(row, code, horizon, window, materials_input_hash, nav_input_hash=None):
    """身份和资料版本必须全部相同；缺任一证明就继续等待。"""
    return (
        row.get("analysis_version") == "COMPREHENSIVE_REVIEW_V2"
        and row.get("policy_hash") == digest(policy_v2())
        and row.get("fund_code") == code
        and row.get("horizon") == horizon
        and row.get("base_date") == window["base_nav_date"]
        and row.get("target_date") == window["target_nav_date"]
        and row.get("owner_verified") is True
        and row.get("snapshot_verified") is True
        and materials_input_hash is not None
        and row.get("materials_input_hash") == materials_input_hash
        and (nav_input_hash is None or row.get("nav_input_hash") == nav_input_hash)
    )


def verify_outputs(code, window, records, *, materials_input_hash=None, materials_ready=False, nav_input_hash=None):
    rows = []
    for horizon in ("T1", "T5_V1", "T20_V1", "M6_V1"):
        candidates = [r for r in records if matches(r, code, horizon, window, materials_input_hash, nav_input_hash)]
        dated = [
            r
            for r in candidates
            if r.get("record_kind") == "DATED_FORECAST"
            and r.get("ex_ante_eligible") is True
            and r.get("end_date")
            and r.get("decision") in {"UP", "DOWN"}
        ]
        conditional = [
            r
            for r in candidates
            if horizon == "M6_V1" and r.get("record_kind") == "CONDITIONAL_OUTLOOK" and not r.get("end_date")
            and r.get("decision") in {"UP", "DOWN"}
        ]
        abstain = [r for r in candidates if r.get("decision") == "ABSTAIN"
                   and (r.get("review_complete") is True or r.get("ex_ante_eligible") is True)]
        if dated:
            status, selected = "SAVED_AND_READ_BACK", dated
        elif conditional:
            status, selected = "CONDITIONAL_SAVED", conditional
        elif abstain:
            status, selected = "REVIEWED_ABSTAIN", abstain
        else:
            status, selected = "ACTION_REQUIRED", []
        data_blocked = bool(selected) and all(r.get("source_decision") == "BLOCKED" for r in selected)
        rows.append(
            {
                "horizon": horizon,
                "status": status,
                "record_ids": [r["id"] for r in selected],
                "generationStatus": "SUCCEEDED"
                if dated
                else "CONDITIONAL"
                if conditional
                else "BLOCKED" if data_blocked else "ABSTAIN"
                if abstain
                else "WAITING_ANALYSIS"
                if materials_ready
                else "WAITING_MATERIALS",
                "counts_as_dated_forecast": bool(dated),
                "review_complete": bool(selected),
                "data_blocked": data_blocked,
                "source_reasons": [r["source_reason"] for r in selected if r.get("source_reason")],
                "direction_available": bool(dated or conditional),
                "review_status": (
                    "REVIEWED_BLOCKED" if data_blocked else "REVIEWED_ABSTAIN" if abstain
                    else "REVIEWED_DIRECTION" if selected else "NOT_REVIEWED"
                ),
                # 正式第二版尚未接入旧准确率计算，保存方向不等于已验证准确率。
                "eligible_for_accuracy": False,
            }
        )
    pending = any(r["status"] == "ACTION_REQUIRED" for r in rows)
    return {
        "fund_code": code,
        "analysis_version": "COMPREHENSIVE_REVIEW_V2",
        "status": "ACTION_REQUIRED" if pending else "CLOSED",
        "actions": [],
        "horizons": rows,
    }


def followthrough_report(owner, codes, window, records, *, at):
    from app.services.fund_prediction_snapshot_compat import materials_input_identity
    from app.services.fund_workflow_materials import registered_materials

    results = []
    for code in codes:
        failure, handoff, identity = None, None, None
        try:
            handoff = registered_materials(code, at=at)
            if handoff:
                material = read(Path(handoff["materials_path"]))
                if material.get("fund_code") != code:
                    raise ValueError("V2_DAILY_MATERIAL_FUND_MISMATCH")
                identity = materials_input_identity(material)
        except (OSError, ValueError, KeyError) as error:
            failure = type(error).__name__ + ":" + str(error)
        ready = bool(handoff and not failure)
        from app.services.fund_prediction_nav_identity import current_nav_identity

        result = verify_outputs(code, window, records, materials_input_hash=identity, materials_ready=ready,
                                nav_input_hash=current_nav_identity(code, window["base_nav_date"]))
        result["prepare_failure"] = failure
        if not ready:
            result["actions"] = [
                {
                    "fund_code": code,
                    "action": "PREPARE_V2_CURRENT_MATERIALS",
                    "reason": failure or "本期真实资料尚未登记。",
                }
            ]
        results.append(result)
    pending = [row["fund_code"] for row in results if row["status"] == "ACTION_REQUIRED"]
    periods = [period for row in results for period in row["horizons"]]
    from app.services.prediction_review_watch import watch_report

    watch = watch_report(owner, codes, at=at)
    return {
        "checked_at": at.isoformat(),
        "owner_user_id": owner,
        "window": window,
        "analysis_version": "COMPREHENSIVE_REVIEW_V2",
        "fund_count": len(codes),
        "status": "ACTION_REQUIRED" if pending else "CLOSED",
        "funds": results,
        "all_predictions_generated": all(p["counts_as_dated_forecast"] for p in periods),
        "all_reviews_saved": not pending,
        "dated_prediction_count": sum(p["counts_as_dated_forecast"] for p in periods),
        "conditional_saved_count": sum(p["status"] == "CONDITIONAL_SAVED" for p in periods),
        "abstain_saved_count": sum(p["status"] == "REVIEWED_ABSTAIN" and not p["data_blocked"] for p in periods),
        "blocked_saved_count": sum(p["data_blocked"] for p in periods),
        "reviewed_period_count": sum(p["review_complete"] for p in periods),
        "unreviewed_period_count": sum(not p["review_complete"] for p in periods),
        "reassessment_watch": watch,
        "reassessment_pending_count": watch["reassessment_pending_count"],
        "reassessment_check_failure_count": watch["failure_count"],
        "pending_funds": pending,
    }


def freeze_request(folder, owner, code, horizon, window, handoff, *, nav_input_hash=None):
    """只登记分析请求和真实资料身份，不把请求说明伪装为已审阅输入。"""
    from app.services.fund_prediction_snapshot_compat import materials_input_identity
    from app.services.prediction_review_v2_requests import request_batch

    live = planning_window(now())
    if any(live[k] != window[k] for k in ("base_nav_date", "target_nav_date")):
        raise ValueError("V2_DAILY_REQUEST_WINDOW_CHANGED")
    current = adoption()
    if current is None:
        raise ValueError("V2_NOT_FORMALLY_ADOPTED")
    material = read(Path(handoff["materials_path"]))
    if material.get("fund_code") != code:
        raise ValueError("V2_DAILY_MATERIAL_FUND_MISMATCH")
    if digest(material) != handoff.get("materials_hash"):
        raise ValueError("V2_DAILY_MATERIAL_HASH_MISMATCH")
    rules = policy_v2()
    identity = {
        "owner_hash": digest(owner),
        "fund_code": code,
        "horizon": horizon,
        "window": window,
        "policy_hash": digest(rules),
        "materials_input_hash": materials_input_identity(material),
        "materials_hash": handoff.get("materials_hash"),
        "nav_input_hash": nav_input_hash,
    }
    if current["policyHash"] != identity["policy_hash"]:
        raise ValueError("V2_DAILY_RULES_DIFFER_FROM_ADOPTED_POLICY")
    folder = Path(folder) / "v2" / digest(identity)
    # 指引单独留版本，不改变旧请求身份、已审阅原文或正式政策摘要。
    guidance = review_guidance()
    guidance_path = folder / ("review-guidance-" + digest(guidance) + ".json")
    if not guidance_path.exists():
        versioned_save(guidance_path, guidance)
    path = folder / "analysis-request.json"
    if not path.exists():
        request = {
            "format": "FUND_PREDICTION_V2_DAILY_ANALYSIS_REQUEST",
            **identity,
            "requested_at": now().isoformat(),
            "generationStatus": "WAITING_ANALYSIS",
            "materials_path": handoff["materials_path"],
            "period_rule": rules["horizons"][horizon],
            "legacy_material_gaps": handoff.get("gaps", []),
            "review_guidance": guidance,
            "review_guidance_hash": digest(guidance),
            "instructions": "阅读本期原文，按本周期规则比较主次。完成严格 prepare/evaluate 后，"
            "在 reviewed-trial-ref.json 填 trial_directory；不能复制旧方向或使用旧 V1 审阅回执。",
        }
        versioned_save(path, request)
    else:
        request = read(path)
        if any(request.get(k) != v for k, v in identity.items()):
            raise ValueError("V2_DAILY_REQUEST_IDENTITY_CHANGED")
    task = request_batch([code], horizons=[horizon], materials_input_hash=identity["materials_input_hash"])
    if task.get("planningWindow") and (
        task["planningWindow"]["base_nav_date"] != window["base_nav_date"]
        or task["planningWindow"]["target_nav_date"] != window["target_nav_date"]
    ):
        raise ValueError("V2_DAILY_REQUEST_WINDOW_CHANGED")
    versioned_save(folder / "request-receipt.json", task)
    return {
        "path": str(path),
        "directory": str(folder),
        "input_hash": digest(request),
        "generationStatus": "WAITING_ANALYSIS",
        "review_path": str(folder / "reviewed-trial-ref.json"),
        "review_guidance_path": str(guidance_path),
        "review_guidance_hash": digest(guidance),
        "task": task,
    }


def import_reviewed_trial(folder, owner, code, horizon, window, materials_input_hash):
    """发布已有的完整第二版严格审阅；普通 review.json 不构成发布凭据。"""
    from app.services.fund_prediction_snapshot_compat import materials_input_identity
    from app.services.prediction_review_v2_publication import publish_trial
    from app.services.prediction_trial_v2 import checked_input, local_path

    reference = Path(folder) / "reviewed-trial-ref.json"
    if not reference.exists():
        raise ValueError("V2_STRICT_REVIEWED_TRIAL_REFERENCE_REQUIRED")
    directory = local_path(read(reference)["trial_directory"])
    manifest = read(directory / "manifest.json")
    if manifest["inventory"]["owner_hash"] != digest(owner):
        raise ValueError("V2_DAILY_TRIAL_OWNER_MISMATCH")
    if code not in {r["fund_code"] for r in manifest["selection"]}:
        raise ValueError("V2_DAILY_TRIAL_FUND_MISSING")
    for entry in manifest["selection"]:
        packet, data = checked_input(directory, entry)
        period = packet["periods"][horizon]
        if period["base_date"] != window["base_nav_date"] or period["start_date"] != window["target_nav_date"]:
            raise ValueError("V2_DAILY_TRIAL_WINDOW_MISMATCH")
        if entry["fund_code"] == code:
            material = read(Path(data["materials_path"])) if data.get("materials_path") else data
            if data.get("materials_path") and digest(material) != data.get("materials_hash"):
                raise ValueError("V2_DAILY_TRIAL_MATERIALS_CHANGED")
            if materials_input_identity(material) != materials_input_hash:
                raise ValueError("V2_DAILY_TRIAL_MATERIALS_CHANGED")
    result = publish_trial(directory, adopt=True)
    versioned_save(Path(folder) / "v2-import-receipt.json", result)
    return result
