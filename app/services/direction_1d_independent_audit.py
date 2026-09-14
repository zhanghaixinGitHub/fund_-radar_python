"""未来对照P0：本人范围、已登记原模型、ACTIVITY12真实训练成员及日程审计。"""

import importlib.metadata
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from app.db.session import get_nav_preview_engine
from app.repositories import direction_1d as repo
from app.services import direction_1d_activity_study as activity
from app.services import direction_1d_independent as protocol
from app.services.direction_1d_data import classify
from app.services.direction_1d_inference import load_model
from app.services.direction_1d_protocol import calendar, canonical, digest, score, window
from app.services.direction_1d_selection import available_at_prediction
from app.services.direction_1d_training import select_fit, weights
from app.services.direction_training_artifacts import file_hash

PROJECT = Path(__file__).resolve().parents[2]


def code_files() -> dict:
    """只绑定本链路实际执行文件和依赖版本；不改变既有研究或SPX任务的冻结代码。"""
    names = (
        "app/services/direction_1d_independent.py",
        "app/services/direction_1d_independent_audit.py",
        "app/services/direction_1d_independent_data.py",
        "app/services/direction_1d_independent_report.py",
        "app/services/direction_1d_independent_runtime.py",
        "scripts/direction_1d_independent.py",
        "scripts/direction_1d_independent_task.ps1",
        "app/services/direction_1d_protocol.py",
        "app/services/direction_1d_training.py",
        "app/services/direction_1d_inference.py",
        "app/services/direction_1d_selection.py",
        "app/services/direction_1d_data.py",
        "app/services/direction_1d_activity_study.py",
        "app/services/direction_1d_tree_study.py",
        "app/services/direction_1d_sector_study.py",
        "app/repositories/direction_1d.py",
        "app/services/trading_calendar.py",
        "app/data/calendars/cn_a_share_2021_2025_v1.json",
        "app/data/calendars/cn_a_share_2026_v1.json",
    )
    return {
        "files": {n: file_hash(PROJECT / n) for n in names},
        "dependencies": {n: importlib.metadata.version(n) for n in ("numpy", "scikit-learn", "sqlalchemy", "httpx")},
    }


def live_context(identity_evidence: Path, core_env: Path) -> dict:
    """从上轮本人已确认的预测ID找回唯一账户，只查询该账户关注及公开输入。

    连接强制只读并限制查询时间；不允许调用 observe/infer，它们会登记业务快照。
    凭据来自既有.env，仅交给连接器；结果不包含密码、Token或其他账户资料。
    """
    prior = protocol.read(identity_evidence)
    ids = [r["forecast_id"] for r in prior["forecasts"]]
    if not 1 <= len(ids) <= 100 or len(ids) != len(set(ids)):
        raise ValueError("KNOWN_OWNER_EVIDENCE_INVALID")
    settings = dotenv_values(core_env)
    engine = create_engine(
        URL.create(
            "postgresql+psycopg",
            username=settings["FUND_CORE_DB_USERNAME"],
            password=settings["FUND_CORE_DB_PASSWORD"],
            host=settings.get("FUND_CORE_DB_HOST", "localhost"),
            port=int(settings.get("FUND_CORE_DB_PORT", "54329")),
            database=settings.get("FUND_CORE_DB_NAME", "fund_core"),
        ),
        hide_parameters=True,
        connect_args={
            "connect_timeout": 5,
            "options": "-c statement_timeout=10000 -c default_transaction_read_only=on",
        },
    )
    try:
        with engine.connect() as c:
            owners = (
                c.execute(
                    text(
                        "SELECT DISTINCT user_id FROM direction_1d_user_forecast "
                        "WHERE forecast_id=ANY(CAST(:ids AS uuid[]))"
                    ),
                    {"ids": ids},
                )
                .scalars()
                .all()
            )
            if len(owners) != 1:
                raise ValueError("KNOWN_OWNER_NOT_UNIQUE")
            codes = list(
                c.execute(
                    text("SELECT fund_code FROM watchlist_item WHERE user_id=:owner ORDER BY fund_code"),
                    {"owner": owners[0]},
                ).scalars()
            )
            core_at = c.execute(text("SELECT clock_timestamp()")).scalar_one()
            receipts = [
                dict(r)
                for r in c.execute(
                    text(
                        "SELECT f.fund_code,f.target_nav_date,r.status FROM direction_1d_forecast f "
                        "LEFT JOIN direction_1d_forecast_receipt r USING(forecast_id) "
                        "WHERE f.forecast_id=ANY(CAST(:ids AS uuid[])) ORDER BY f.fund_code"
                    ),
                    {"ids": ids},
                ).mappings()
            ]
            outcomes = c.execute(
                text("SELECT count(*) FROM direction_1d_outcome WHERE forecast_id=ANY(CAST(:ids AS uuid[]))"),
                {"ids": ids},
            ).scalar_one()
    finally:
        engine.dispose()
    if not 1 <= len(codes) <= 100 or len(codes) != len(set(codes)):
        raise ValueError("OWNER_SCOPE_EMPTY_OR_UNBOUNDED")
    with get_nav_preview_engine().connect() as c, c.begin():
        c.execute(text("SET TRANSACTION READ ONLY"))
        at = c.execute(text("SELECT clock_timestamp()")).scalar_one()
        if abs((protocol.now() - at).total_seconds()) > 5 or abs((at - core_at).total_seconds()) > 5:
            raise ValueError("CLOCK_SKEW")
        source = dict(repo.source(c))
        # 只保存研究实际需要的来源字段，避免把数据库连接或供应商配置带出。
        source = {
            k: source[k]
            for k in (
                "source_id",
                "source_code",
                "enabled",
                "authorized_api_names",
                "authorization_verified_at",
                "retention_days",
                "rate_limit_per_minute",
            )
        }
        profiles = repo.profiles(c, codes)
        registry = repo.models(c)
        released = c.execute(text("SELECT count(*) FROM analysis_model_release")).scalar_one()
    result = {
        "at": at,
        "codes": codes,
        "scope_hash": digest(codes),
        "owner_binding_hash": digest(str(owners[0])),
        "identity_evidence_hash": file_hash(identity_evidence),
        "source": source,
        "profiles": profiles,
        "registry": registry,
        "existing_receipts": receipts,
        "existing_outcome_count": outcomes,
        "formal_release_count": released,
        "database_writes": 0,
    }
    return json.loads(canonical(result))


def audit_candidate(root: Path) -> tuple[dict, dict, dict]:
    """校验旧包而非重训；用真实成员日期及原尺度来源纠正仅看train_as_of的误解。"""
    activity.verify(root)
    spec = protocol.read(root / "study.json")
    model = protocol.read(root / "main/FINAL.json")
    restored = protocol.read(root / "replay/FINAL.json")
    # 五轮成员的旧包超过未来单份快照的64MiB限制；经旧校验器核验文件指纹后，
    # 沿用研究读取器。这里只处理已封存开发资料，不放宽未来快照大小限制。
    fits = activity.read(root / "fits.json")["FINAL"]
    if model != restored or model["fit_hash"] != digest(fits):
        raise ValueError("FINAL_OR_MEMBERS_CHANGED")
    if any(r["u"] > "2024-12-31" for r in fits):
        raise ValueError("PROTECTED_TRAINING_MEMBER")
    w = weights(fits)
    if digest(w.tolist()) != model["weight_hash"]:
        raise ValueError("FINAL_WEIGHT_CHANGED")
    control = protocol.read(root / "base/main/FINAL.json")
    if (
        digest(control) != model["control_model_hash"]
        or model["mean"][:11] != control["mean"]
        or model["scale"][:11] != control["scale"]
    ):
        raise ValueError("FINAL_SCALE_LINEAGE_CHANGED")
    extra = np.asarray([r["activity_input"]["x"][0] for r in fits])
    mean = float(np.average(extra, weights=w))
    std = float(np.sqrt(np.average((extra - mean) ** 2, weights=w)))
    if not np.allclose([mean, std], [model["mean"][-1], model["scale"][-1]], rtol=1e-12, atol=1e-12):
        raise ValueError("FINAL_ACTIVITY_SCALE_CHANGED")
    # 全部真实拟合输入逐值恢复，零拟合。分数恢复证据不代表独立预测效果。
    expected = activity.predict(model, fits)
    actual = np.asarray([protocol.model_score(model, activity.vector(r)) for r in fits])
    difference = float(np.max(np.abs(expected - actual)))
    if difference > 1e-12:
        raise ValueError("FINAL_FORWARD_RESTORE_MISMATCH")
    groups, families = {}, {}
    for row in fits:
        code = row["fund_code"]
        if (code in groups and groups[code] != row["group"]) or (code in families and families[code] != row["family"]):
            raise ValueError("TRAINING_MEMBER_MAPPING_INCONSISTENT")
        groups[code], families[code] = row["group"], row["family"]
    mapping = protocol.read(root / "base/base/evidence/mapping-reviewed.json")["funds"]
    proof = {
        "model_hash": digest(model),
        "model_file_hash": file_hash(root / "main/FINAL.json"),
        "study_hash": digest(spec),
        "fit_hash": digest(fits),
        "actual_fit_count": len(fits),
        "actual_target_start": min(r["u"] for r in fits),
        "actual_target_end": max(r["u"] for r in fits),
        "latest_label_mature_at": max(r["mature_at"] for r in fits),
        "train_as_of_metadata": model["train_as_of"],
        "archived_finished_at": protocol.read(root / "main/completion.json")["finished_at"],
        "source_expires_at": spec["source"]["source_expires_at"],
        "scale_source": {
            "first_11_control_hash": digest(control),
            "extra_weighted_mean": mean,
            "extra_weighted_std": std,
        },
        "restore_max_score_diff": difference,
        "fund_codes": sorted(groups),
        "groups": groups,
        "families": families,
        "new_fits": 0,
        "historical_first_versions_verified": False,
    }
    return model, proof, mapping


def fixed_models(context: dict, initial: Path) -> tuple[dict, dict]:
    dataset = protocol.read(initial / "dataset.json")
    manifest = protocol.read(initial / "dataset-manifest.json")
    if digest(dataset) != manifest["hash"] or any(r["u"] > "2024-12-31" for r in dataset):
        raise ValueError("FIXED_DATASET_CHANGED_OR_PROTECTED")
    at = protocol.instant(context["at"])
    models, evidence = {}, {}
    for original in sorted(context["registry"], key=lambda r: (r["registered_at"], r["model_id"])):
        row = {**original, **{k: protocol.instant(original[k]) for k in ("trained_at", "registered_at", "expires_at")}}
        group = row["group_id"]
        if group in evidence or not available_at_prediction(row, at):
            continue
        model = load_model(row)
        members = select_fit([r for r in dataset if r["group"] == group], protocol.instant(model["train_as_of"]))
        if (
            not members
            or digest(members) != model["fit_hash"]
            or digest(weights(members).tolist()) != model["weight_hash"]
        ):
            raise ValueError("FIXED_TRAINING_MEMBERS_CHANGED")
        score(model, members[0]["x"])
        models[row["model_id"]] = model
        evidence[group] = {
            "model_id": row["model_id"],
            "model_hash": digest(model),
            "model_file_hash": row["content_hash"],
            "expires_at": row["expires_at"].isoformat(),
            "registered_at": row["registered_at"].isoformat(),
            "trained_at": row["trained_at"].isoformat(),
            "fund_codes": sorted({r["fund_code"] for r in members}),
            "actual_target_start": min(r["u"] for r in members),
            "actual_target_end": max(r["u"] for r in members),
            "fit_hash": digest(members),
        }
    return models, evidence


def coverage(context: dict, candidate: dict, fixed: dict, mapping: dict) -> list[dict]:
    profiles = {p["fund_code"]: p for p in context["profiles"]}
    result = []
    for code in context["codes"]:
        p = profiles.get(code)
        group = (
            classify(p)
            if p
            else {"group_id": None, "classification_reason": "PROFILE_MISSING", "product_family_id": None}
        )
        asset, family = group["group_id"], group["product_family_id"]
        link = mapping.get(code, {})
        reasons = []
        if not asset:
            reasons.append(group["classification_reason"])
        if code not in candidate["fund_codes"]:
            reasons.append("NOT_IN_CANDIDATE_TRAINING_SCOPE")
        elif candidate["groups"][code] != asset or candidate["families"][code] != family:
            reasons.append("CANDIDATE_GROUP_OR_FAMILY_CHANGED")
        if asset not in fixed or code not in fixed[asset]["fund_codes"]:
            reasons.append("FIXED_TRAINING_SCOPE_UNAVAILABLE")
        if link.get("status") != "RESEARCH_MAPPING_SUPPORTED" or not link.get("index_code"):
            reasons.append("INDEX_MAPPING_UNVERIFIED")
        result.append(
            {
                "fund_code": code,
                "fund_name": p["fund_name"] if p else code,
                "group": asset,
                "family": family,
                "source_id": context["source"]["source_id"],
                "fixed_model_id": fixed.get(asset, {}).get("model_id"),
                "index_code": link.get("index_code"),
                "mapping_hash": digest(link),
                "profile_hash": p.get("profile_hash") if p else None,
                "eligible": not reasons,
                "reasons": reasons,
            }
        )
    return result


def prepare(out: Path, candidate_root: Path, initial: Path, identity_evidence: Path, core_env: Path) -> dict:
    if out.exists():
        raise ValueError("AUDIT_OUTPUT_ALREADY_EXISTS")
    context = live_context(identity_evidence, core_env)
    candidate, proof, mapping = audit_candidate(candidate_root)
    models, fixed = fixed_models(context, initial)
    items = coverage(context, proof, fixed, mapping)
    days, calendar_hash = calendar()
    sessions = [str(d) for d in days]
    current_window = window(protocol.now())
    first = (
        current_window["target_nav_date"]
        if current_window["status"] == "OPEN"
        else next(d for d in sessions if d > current_window["target_nav_date"])
    )
    planned = protocol.schedule(sessions, first)
    blockers = []
    if not planned["complete"]:
        blockers.append("OFFICIAL_CALENDAR_120_PLUS_10_INCOMPLETE")
    if not any(r["eligible"] for r in items):
        blockers.append("NO_COMMON_ELIGIBLE_FUNDS")
    if protocol.now() >= protocol.instant(proof["source_expires_at"]):
        blockers.append("CANDIDATE_SOURCE_EXPIRED")
    if not {"fund_nav", "fund_div", "index_daily"}.issubset(context["source"]["authorized_api_names"]):
        blockers.append("SOURCE_CAPABILITY_MISSING")
    draft = {
        "protocol": protocol.PROTOCOL,
        "policy": protocol.POLICY,
        "created_at": protocol.now().isoformat(),
        "status": "P0_BLOCKED" if blockers else "P0_AUDITED_PENDING_DATA_ACCEPTANCE",
        "blockers": blockers,
        "schedule": planned,
        "sessions": sessions,
        "calendar_hash": calendar_hash,
        "official_calendar_last_date": "2026-12-31",
        "members": {r["fund_code"]: r for r in items if r["eligible"]},
        "scope_hash": context["scope_hash"],
        "owner_binding_hash": context["owner_binding_hash"],
        "coverage": items,
        "candidate_expires_at": proof["source_expires_at"],
        "fixed_expires_at": {v["model_id"]: v["expires_at"] for v in fixed.values()},
        "candidate_evidence": proof,
        "fixed_evidence": fixed,
        "source": context["source"],
        "model_hash": digest({"candidate": candidate, "fixed": models}),
        "identity_evidence_path": str(identity_evidence.resolve()),
        "core_env_path": str(core_env.resolve()),
        "code": code_files(),
        "model_released": False,
        "new_fits": 0,
    }
    out.mkdir(parents=True, exist_ok=False)
    protocol.seal(out / "draft.json", draft)
    protocol.seal(out / "models.json", {"candidate": candidate, "fixed": models})
    protocol.seal(out / "runtime.json", context)
    return {
        "status": draft["status"],
        "blockers": blockers,
        "owner_count": len(items),
        "eligible_count": len(draft["members"]),
        "known_target_days": len(planned["targets"]),
        "missing_target_days": planned["missing_target_days"],
        "actual_training_end": proof["actual_target_end"],
        "restore_max_score_diff": proof["restore_max_score_diff"],
    }


def verify(root: Path) -> dict:
    draft = protocol.unseal(root / "draft.json")
    models = protocol.unseal(root / "models.json")
    runtime = protocol.unseal(root / "runtime.json")
    if draft["protocol"] != protocol.PROTOCOL or draft["policy"] != protocol.POLICY or draft["code"] != code_files():
        raise ValueError("FROZEN_PROTOCOL_OR_CODE_CHANGED")
    if draft["model_hash"] != digest(models) or draft["scope_hash"] != runtime["scope_hash"]:
        raise ValueError("FROZEN_MODELS_OR_SCOPE_CHANGED")
    if draft["schedule"] != protocol.schedule(draft["sessions"], draft["schedule"]["first_target"]):
        raise ValueError("FROZEN_SCHEDULE_CHANGED")
    activity.validate_model(models["candidate"])
    return {"draft": draft, "models": models, "runtime": runtime}


def require_startable(root: Path) -> dict:
    """完整日程是硬门槛。P0草稿不能作为已启动合同，不能把旧29份预测拼进来。"""
    state = verify(root)
    draft = state["draft"]
    require_calendar_ready(draft)
    acceptance = protocol.unseal(root / "input-acceptance/result.json")
    if (
        acceptance["status"] != "ALL_INPUTS_RESTORED_DIAGNOSTIC_ONLY"
        or acceptance["draft_hash"] != digest(draft)
        or acceptance["model_hash"] != draft["model_hash"]
        or set(acceptance["inputs"]) != set(draft["members"])
    ):
        raise ValueError("LIVE_INPUT_ACCEPTANCE_INVALID")
    for code, expected in acceptance["inputs"].items():
        if digest(protocol.unseal(root / "input-acceptance" / (code + ".json"))) != expected:
            raise ValueError("LIVE_INPUT_ACCEPTANCE_CHANGED")
    end = protocol.instant(draft["schedule"]["grace_targets"][-1] + "T23:59:59+08:00")
    if any(protocol.instant(v) <= end for v in [draft["candidate_expires_at"], *draft["fixed_expires_at"].values()]):
        raise ValueError("MODEL_EXPIRY_BEFORE_STAGE_END")
    return state


def require_calendar_ready(draft: dict) -> None:
    if draft["blockers"] or not draft["schedule"]["complete"]:
        raise ValueError("START_BLOCKED:" + ",".join(draft["blockers"]))
    if protocol.now() - protocol.instant(draft["created_at"]) > timedelta(days=1):
        raise ValueError("P0_AUDIT_STALE_RECHECK_BEFORE_START")
