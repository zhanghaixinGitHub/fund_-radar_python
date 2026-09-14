"""独立未来研究运行器：零训练、本地留档、只读公共库及有界失败记录。"""

import traceback
from datetime import date, datetime, time, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_nav_preview_engine
from app.repositories import direction_1d as repo
from app.services import direction_1d_independent as p
from app.services import direction_1d_independent_audit as audit
from app.services import direction_1d_independent_data as data
from app.services import direction_1d_independent_report as reporting


def failure(exc: Exception) -> dict:
    """保留调用栈位置及稳定业务错误码，不保存含凭据的SQL/HTTP异常正文。"""
    message = str(exc)
    code = (
        message
        if isinstance(exc, ValueError)
        and message
        and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789:," for c in message)
        else type(exc).__name__
    )
    return {
        "code": code,
        "frames": [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in traceback.extract_tb(exc.__traceback__)],
    }


def fresh_scope(draft: dict) -> dict:
    context = audit.live_context(Path(draft["identity_evidence_path"]), Path(draft["core_env_path"]))
    if context["owner_binding_hash"] != draft["owner_binding_hash"]:
        raise ValueError("OWNER_IDENTITY_CHANGED")
    return context


def input_acceptance(root: Path) -> dict:
    """正式启动前对全部共同基金做一次当前输入验收；日历不完整时不发这些请求。"""
    state = audit.verify(root)
    draft = state["draft"]
    audit.require_calendar_ready(draft)
    if fresh_scope(draft)["scope_hash"] != draft["scope_hash"]:
        raise ValueError("OWNER_SCOPE_CHANGED_REAUDIT_REQUIRED")
    out = root / "input-acceptance"
    if out.exists():
        raise ValueError("ACCEPTANCE_ALREADY_ATTEMPTED")
    out.mkdir()
    target = draft["schedule"]["first_target"]
    market = data.collect_market(out / "market", draft, list(draft["members"]), target)
    results = {}
    for code, member in draft["members"].items():
        snapshot = data.collect_nav(member, draft, target)
        snapshot["market"] = {**market, "mapping_hash": member["mapping_hash"]}
        expected = p.answers(snapshot, draft["sessions"], member, state["models"], p.now())
        p.seal(out / (code + ".json"), snapshot)
        if p.answers(p.unseal(out / (code + ".json")), draft["sessions"], member, state["models"], p.now()) != expected:
            raise ValueError("LIVE_INPUT_RESTORE_FAILED")
        results[code] = p.digest(snapshot)
    result = {
        "status": "ALL_INPUTS_RESTORED_DIAGNOSTIC_ONLY",
        "draft_hash": p.digest(draft),
        "at": p.now().isoformat(),
        "inputs": results,
        "model_hash": p.digest(state["models"]),
        "new_fits": 0,
    }
    p.seal(out / "result.json", result)
    return result


def activate(root: Path) -> dict:
    state = audit.require_startable(root)
    draft = state["draft"]
    context = fresh_scope(draft)
    if context["scope_hash"] != draft["scope_hash"]:
        raise ValueError("OWNER_SCOPE_CHANGED_REAUDIT_REQUIRED")
    at = p.now()
    if at >= p.bounds(draft["sessions"], draft["schedule"]["first_target"])[2]:
        raise ValueError("FIRST_TARGET_ALREADY_MISSED")
    contract = {**draft, "status": "ACTIVE_FORWARD_RESEARCH", "activated_at": at.isoformat()}
    p.seal(root / "contract.json", contract)
    return {
        "status": contract["status"],
        "first_target": contract["schedule"]["first_target"],
        "last_target": contract["schedule"]["targets"][-1],
        "model_released": False,
    }


def load(root: Path) -> tuple[dict, dict]:
    state = audit.verify(root)
    if state["draft"]["blockers"] or not state["draft"]["schedule"]["complete"]:
        raise ValueError("START_BLOCKED:" + ",".join(state["draft"]["blockers"]))
    contract = p.unseal(root / "contract.json")
    expected = {**state["draft"], "status": "ACTIVE_FORWARD_RESEARCH", "activated_at": contract["activated_at"]}
    if contract != expected or not contract["schedule"]["complete"]:
        raise ValueError("ACTIVE_CONTRACT_CHANGED")
    return contract, state["models"]


def slot_for(contract: dict, at: datetime) -> tuple[str, str] | None:
    """触发器允许5分钟启动延迟；其他开机时点不追补旧槽，截止后统一记遗漏。"""
    for target in contract["schedule"]["targets"]:
        base, opened, deadline = p.bounds(contract["sessions"], target)
        if opened <= at < deadline:
            candidates = []
            for slot in p.POLICY["prediction_slots"]:
                day = base if slot >= "18:00" else target
                trigger = p.instant(day + "T" + slot + ":00+08:00")
                if 0 <= (at - trigger).total_seconds() <= p.POLICY["slot_lateness_seconds"]:
                    candidates.append((trigger, slot))
            return (target, max(candidates)[1]) if candidates else None
    return None


def remember_nav_versions(root: Path, snapshot: dict) -> dict:
    """同一来源/内容首次本地观察不续期，所有基金共用版本索引且只增不改。"""
    expires = p.instant(snapshot["source"]["expires_at"])
    for version in snapshot["nav_source_versions"]:
        key = p.digest([snapshot["source"]["source_id"], snapshot["fund_code"], version["content_hash"]])
        path = root / "nav-versions" / (key + ".json")
        try:
            p.seal(path, version)
        except FileExistsError:
            pass
        original = p.unseal(path)
        expires = min(expires, p.instant(original["expires_at"]))
    if expires <= p.now():
        raise ValueError("LOCAL_NAV_VERSION_EXPIRED")
    snapshot["source"]["expires_at"] = expires.isoformat()
    return snapshot


def forecast_slot(root: Path, contract: dict, models: dict, context: dict, target: str, slot: str) -> dict:
    out = root / "attempts" / target / slot.replace(":", "")
    try:
        p.seal(out / "reservation.json", {"at": p.now().isoformat(), "target": target, "slot": slot})
    except FileExistsError:
        return {"status": "SLOT_ALREADY_ATTEMPTED", "target": target, "slot": slot}
    active = set(context["codes"]) & set(contract["members"])
    need = [c for c in sorted(active) if not (root / "questions" / target / c / "readback.json").exists()]
    market, market_error = None, None
    if need:
        try:
            market = data.collect_market(out / "market", contract, need, target)
        except Exception as exc:
            market_error = failure(exc)
    results, errors = {}, {}
    for code, member in contract["members"].items():
        folder = root / "questions" / target / code
        if (folder / "readback.json").exists():
            try:
                results[code] = p.verify_question(folder, contract, models)["status"]
            except (OSError, ValueError, KeyError):
                results[code] = "INTEGRITY_FAILURE"
            continue
        if code not in active:
            results[code] = "NOT_IN_CURRENT_OWNER_SCOPE"
            continue
        if market is None and slot != p.POLICY["prediction_slots"][-1]:
            results[code] = "MARKET_PENDING_NEXT_PLANNED_SLOT"
            continue
        try:
            snapshot = remember_nav_versions(root, data.collect_nav(member, contract, target))
            snapshot["market"] = {**market, "mapping_hash": member["mapping_hash"]} if market else None
            if market is None:
                snapshot["candidate_missing_reason"] = "MARKET_UNAVAILABLE_AT_FINAL_SLOT"
            results[code] = p.archive(root, contract, snapshot, models)["status"]
        except Exception as exc:
            errors[code] = failure(exc)
            results[code] = "FORECAST_FAILED:" + errors[code]["code"]
    result = {"at": p.now().isoformat(), "results": results, "errors": errors, "market_error": market_error}
    p.seal(out / "result.json", result)
    return result


def read_outcome(member: dict, contract: dict, target: str) -> dict | None:
    """仅查询本期计划T/U两日；不会读取2025保护答案或用历史题补预测。"""
    base = p.bounds(contract["sessions"], target)[0]
    with get_nav_preview_engine().connect() as c, c.begin():
        c.execute(text("SET TRANSACTION READ ONLY"))
        source = dict(repo.source(c))
        if str(source["source_id"]) != member["source_id"]:
            raise ValueError("OUTCOME_SOURCE_CHANGED")
        rows = repo.navs(
            c, member["fund_code"], source["source_id"], date.fromisoformat(base), date.fromisoformat(target)
        )
        events = [
            dict(r)
            for r in c.execute(
                text(
                    "SELECT ex_date,nav_ex_date,content_hash FROM fund_dividend WHERE fund_code=:code "
                    "AND source_id=:source AND (ex_date=:target OR nav_ex_date=:target) ORDER BY content_hash LIMIT 101"
                ),
                {"code": member["fund_code"], "source": source["source_id"], "target": date.fromisoformat(target)},
            ).mappings()
        ]
        if len(events) > 100 or "fund_div" not in source["authorized_api_names"]:
            raise ValueError("OUTCOME_EVENT_SOURCE_OR_LIMIT_INVALID")
        at = c.execute(text("SELECT clock_timestamp()")).scalar_one()
        if abs((p.now() - at).total_seconds()) > 5:
            raise ValueError("CLOCK_SKEW")
    found = {str(r["nav_date"]): r for r in rows}
    if base not in found or target not in found:
        return None
    observed = []
    for day in (base, target):
        r = found[day]
        if r["ann_date"] and r["ann_date"] > at.astimezone(p.ZONE).date():
            return None
        if r["updated_at"] > at:
            raise ValueError("FUTURE_OUTCOME_VERSION")
        observed.append(
            {
                "fund_code": member["fund_code"],
                "date": day,
                "unit_nav": str(r["unit_nav"]),
                "content_hash": r["content_hash"],
                "source_id": str(source["source_id"]),
                "observed_at": at.isoformat(),
                "updated_at": r["updated_at"].isoformat(),
                "ann_date": str(r["ann_date"]) if r["ann_date"] else None,
                "expires_at": (
                    at + timedelta(days=min(source["retention_days"], contract["source"]["retention_days"]))
                ).isoformat(),
            }
        )
    return {
        **observed[1],
        "base_revision": observed[0],
        "events": p.json.loads(p.canonical(events)),
        "events_status": "KNOWN_EVENT" if events else "UNKNOWN",
    }


def outcomes(root: Path, contract: dict, models: dict, context: dict, at: datetime) -> dict:
    completed, pending, failed, queried = 0, 0, 0, 0
    errors = []
    end = datetime.combine(date.fromisoformat(contract["schedule"]["grace_targets"][-1]), time(23, 59, 59), p.ZONE)
    for target in reversed(contract["schedule"]["targets"]):
        if at < datetime.combine(date.fromisoformat(target), time(18), p.ZONE):
            continue
        for code, member in contract["members"].items():
            if p.now() >= end:
                break
            folder = root / "questions" / target / code
            # 完整历史已保存即可停止反复查；较新5日仍读取以发现修订。
            old = (at.date() - date.fromisoformat(target)).days > 7
            if old and ((folder / "first-outcome.json").exists() or (folder / "missing-forecast-answer.json").exists()):
                continue
            if code not in context["codes"] or queried >= p.POLICY["max_outcome_queries_per_tick"]:
                continue
            queried += 1
            try:
                observation = read_outcome(member, contract, target)
                if observation is None:
                    pending += 1
                    continue
                try:
                    valid = p.verify_question(folder, contract, models)["status"] == "VERIFIED"
                except (ValueError, OSError, KeyError):
                    valid = False
                if valid:
                    p.record_outcome(folder, contract, models, observation)
                else:
                    payload = {
                        "kind": "ANSWER_FOR_EXPECTED_MISSING_FORECAST",
                        "fund_code": code,
                        "target": target,
                        "observation": observation,
                        "answer": p.label(observation["base_revision"]["unit_nav"], observation["unit_nav"]),
                    }
                    try:
                        p.seal(folder / "missing-forecast-answer.json", payload)
                    except FileExistsError:
                        pass
                completed += 1
            except Exception as exc:
                failed += 1
                errors.append({"fund_code": code, "target": target, **failure(exc)})
    return {"queried": queried, "completed": completed, "pending": pending, "failed": failed, "errors": errors}


def report(root: Path, *, at=None) -> dict:
    if (root / "final-report.json").exists():
        return p.unseal(root / "final-report.json")
    contract, models = load(root)
    at = at or p.now()
    return reporting.evaluate(contract, reporting.collect_records(root, contract, models), at)


def tick(root: Path) -> dict:
    contract, models = load(root)
    # 固定截止后结束，之后不再请求或把迟到答案拼入最终报告。
    terminal = root / "final-report.json"
    if terminal.exists():
        return {"status": "STAGE_FINISHED", "verdict": p.unseal(terminal)["verdict"]}
    at = p.now()
    end = datetime.combine(date.fromisoformat(contract["schedule"]["grace_targets"][-1]), time(23, 59, 59), p.ZONE)
    if at >= end:
        summary = report(root, at=at)
        p.seal(terminal, summary)
        return {"status": "STAGE_FINISHED", "verdict": summary["verdict"]}
    context = fresh_scope(contract)
    current_slot = slot_for(contract, at)
    result = {
        "at": at.isoformat(),
        "new_follow_queue": sorted(set(context["codes"]) - {r["fund_code"] for r in contract["coverage"]}),
    }
    if current_slot:
        result["forecast"] = forecast_slot(root, contract, models, context, *current_slot)
    for target in contract["schedule"]["targets"]:
        if at < p.bounds(contract["sessions"], target)[2]:
            continue
        for code in contract["members"]:
            folder = root / "questions" / target / code
            if not (folder / "readback.json").exists():
                try:
                    p.seal(
                        folder / "missed.json",
                        {
                            "status": "EXPECTED_BUT_NOT_COMPLETED",
                            "observed_at": at.isoformat(),
                            "target": target,
                            "fund_code": code,
                        },
                    )
                except FileExistsError:
                    pass
    end = datetime.combine(date.fromisoformat(contract["schedule"]["grace_targets"][-1]), time(23, 59, 59), p.ZONE)
    # 所有期限采用计划日；错过的日子照样累计。到总截止后先报告，不再补读答案。
    if at < end:
        for slot in p.POLICY["outcome_slots"]:
            trigger = p.instant(at.date().isoformat() + "T" + slot + ":00+08:00")
            if 0 <= (at - trigger).total_seconds() <= p.POLICY["slot_lateness_seconds"]:
                reservation = root / "outcome-attempts" / at.date().isoformat() / slot.replace(":", "")
                try:
                    p.seal(reservation / "reserved.json", {"at": at.isoformat()})
                except FileExistsError:
                    break
                result["outcomes"] = outcomes(root, contract, models, context, at)
                p.seal(reservation / "result.json", result["outcomes"])
                break
    summary = report(root, at=p.now())
    if summary["stage_finished"]:
        try:
            p.seal(terminal, summary)
        except FileExistsError:
            pass
    for checkpoint in p.POLICY["checkpoints"]:
        if summary["due_target_days"] >= checkpoint:
            path = root / "checkpoints" / (str(checkpoint) + ".json")
            if not path.exists():
                # 20日仅保留运行质量；60日为描述报告；120日可能仍等待10日宽限。
                payload = (
                    {"at": summary["as_of"], "checkpoint": checkpoint, "coverage": summary["coverage"]}
                    if checkpoint == 20
                    else summary
                )
                p.seal(path, payload)
    result.update(status="STAGE_FINISHED" if summary["stage_finished"] else "RUNNING", coverage=summary["coverage"])
    if result["status"] == "RUNNING" and (
        result.get("outcomes", {}).get("failed", 0)
        or any(
            v.startswith("FORECAST_FAILED:") or v in ("INTEGRITY_FAILURE", "LATE_ARCHIVE")
            for v in result.get("forecast", {}).get("results", {}).values()
        )
    ):
        result["status"] = "DEGRADED"
    p.seal(root / "health" / (at.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex + ".json"), result)
    return result
