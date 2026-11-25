"""总指挥的持久派工与结束门控；状态来自正式回执，不用聊天自述代替验收。"""

from collections import Counter
from datetime import datetime
from functools import cache
from hashlib import sha256
from pathlib import Path

from app.services.direction_1d_protocol import digest
from app.services.fund_daily_prediction import supplement_run_id
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from app.services.prediction_window_v2 import is_midnight

HORIZONS = {"T1", "T5_V1", "T20_V1", "M6_V1"}


def publish_material_rows(directory, state, catalog, rows):
    """一次锁内统一更新逐基金、总回执、待办和统计，保留全持仓范围。"""
    from app.services.fund_daily_profiles import VERIFIED_SOURCE_RESULTS
    from app.services.fund_daily_supplements import record_supplement_summary, task_key

    keys = {task_key(row) for row in catalog["tasks"]}
    by_key = {task_key(row): row for row in rows}
    if len(rows) != len(by_key) or set(by_key) != keys:
        raise ValueError("COMMANDER_RECEIPT_ITEMS_MISMATCH")
    day, window, job = state["checked_at"][:10], state["window"], supplement_run_id(state)
    for code in state["scope"]:
        versioned_save(directory / "supplement-funds" / f"{code}.json", {
            "fund_code": code, "day": day, "window": window, "job_id": job,
            "items": [row for row in rows if row["fund_code"] == code],
        })
    pending = [{**task, "source_review": by_key[task_key(task)],
                "next_source": by_key[task_key(task)].get("next_source") or task.get("next_source")}
               for task in catalog["tasks"] if by_key[task_key(task)]["status"] not in VERIFIED_SOURCE_RESULTS]
    counts = dict(Counter(row["status"] for row in rows))
    result = {"day": day, "base_nav_date": window["base_nav_date"], "target_nav_date": window["target_nav_date"],
              "job_id": job, "checked_at": state["checked_at"], "status": "EXECUTION_INCOMPLETE" if counts.get(
                  "NOT_ATTEMPTED") else "ACTION_REQUIRED" if pending else "SUCCEEDED",
              "counts": counts, "item_count": len(rows), "pending_item_count": len(pending),
              "unattempted_item_count": counts.get("NOT_ATTEMPTED", 0),
              "pending_funds": sorted({row["fund_code"] for row in pending}),
              "funds": {code: dict(Counter(row["status"] for row in rows if row["fund_code"] == code))
                        for code in state["scope"]}}
    versioned_save(directory / "source-rechecks.json", {k: v for k, v in result.items() if k in {
        "day", "base_nav_date", "target_nav_date", "job_id"}} | {"items": rows})
    versioned_save(directory / "supplement-tasks.json", {
        "job_id": job, "window": window, "checked_at": state["checked_at"], "tasks": pending,
    })
    versioned_save(directory / "supplement-execution.json", result)
    record_supplement_summary(directory, state, result)
    return result


def initialize_material_ledger(directory, state, supplements):
    """只展开清单，首次未取件就明确记未执行；同日恢复保留已有逐项证据。"""
    supplements.prepare(state)
    catalog = read(directory / "supplement-catalog.json")
    rows = []
    for task in catalog["tasks"]:
        from app.services.fund_daily_supplements import task_key

        row = supplements.previous.get(task_key(task))
        rows.append(row or {"fund_code": task["fund_code"], "field": task["field"],
                            "required_date": task.get("required_date"), "status": "NOT_ATTEMPTED",
                            "attempted_at": "", "attempted_source": "尚未派工取件", "summary": "本轮尚未逐项执行。",
                            "next_source": task.get("next_source") or "按固定清单取证",
                            "evidence_ref": str(directory / "supplement-funds" / (task["fund_code"] + ".json"))})
    return publish_material_rows(directory, state, catalog, rows)


def merge_worker_receipt(directory, state, receipt, *, project):
    """正式合并明确执行，拒绝串期、漏条和缺少独立原件摘要的成功自述。"""
    from app.services.fund_daily_profiles import VERIFIED_SOURCE_RESULTS
    from app.services.fund_daily_supplements import task_key
    from app.services.material_file_access import original_bytes

    code = receipt.get("fund_code")
    if (receipt.get("identity") != identity(state) or code not in state["scope"]
            or not receipt.get("reviewed_by")):
        raise ValueError("COMMANDER_WORKER_RECEIPT_IDENTITY")
    catalog = read(directory / "supplement-catalog.json")
    if catalog.get("job_id") != supplement_run_id(state) or catalog.get("window") != state["window"]:
        raise ValueError("COMMANDER_CATALOG_CHANGED")
    expected = {task_key(row) for row in catalog["tasks"] if row["fund_code"] == code}
    rows = receipt.get("items", [])
    if len(rows) != len(expected) or {task_key(row) for row in rows} != expected:
        raise ValueError("COMMANDER_WORKER_RECEIPT_COVERAGE")
    allowed = set(VERIFIED_SOURCE_RESULTS) | {"NOT_ATTEMPTED", "REVIEW_REQUIRED", "NOT_PUBLISHED",
                                        "ACCESS_REQUIRED", "TEMPORARY_FAILURE", "MISMATCH", "ADDRESS_CHANGED"}

    @cache
    def original_hash(path):
        # 回执绑定原始字节。一般资料读取可能返回已提取的正文，二者的摘要不同。
        return sha256(original_bytes(path)).hexdigest()

    for row in rows:
        if row.get("status") not in allowed or not all(row.get(k) for k in (
            "summary", "attempted_source", "next_source", "evidence_ref",
        )):
            raise ValueError("COMMANDER_WORKER_ITEM_FIELDS")
        if row["status"] == "NOT_ATTEMPTED":
            continue
        at = datetime.fromisoformat(row.get("attempted_at", ""))
        if (at.tzinfo is None or at.date().isoformat() != state["checked_at"][:10]
                or at > datetime.fromisoformat(state["checked_at"])):
            raise ValueError("COMMANDER_WORKER_ITEM_DATE")
        proof = (project / row["evidence_ref"]).resolve()
        if (not proof.is_relative_to(project.parent) or not proof.is_file()
                or proof == (directory / "supplement-funds" / f"{code}.json").resolve()):
            raise ValueError("COMMANDER_WORKER_EVIDENCE_PATH")
        if (not row.get("observations") or not row.get("evidence_sha256")
                or original_hash(proof) != row["evidence_sha256"]):
            raise ValueError("COMMANDER_WORKER_EVIDENCE_HASH")
        row["evidence_ref"] = str(proof)
    source = read(directory / "source-rechecks.json")
    if source.get("job_id") != supplement_run_id(state):
        raise ValueError("COMMANDER_SOURCE_JOB_CHANGED")
    existing = [row for row in source["items"] if row["fund_code"] != code]
    return publish_material_rows(directory, state, catalog, existing + rows)


def identity(state):
    return {
        "job_id": supplement_run_id(state), "window": state["window"],
        "scope": state["scope"], "owner_hash": digest(state["owner_user_id"]),
        "day": state["checked_at"][:10],
    }


def aggregate_business(directory, state, *, fund_code=None):
    """从绑定原轮次的单基金回执汇总，缺少的基金仍留在原全名单里。"""
    scope = set(state["scope"])

    def preserve(rows, updated):
        return [item for item in rows
                if (code := item.get("fund_code") or item.get("fundCode")) in scope and code not in updated]

    receipts = []
    for code in (state["scope"] if fund_code is None else [fund_code]):
        path = directory / "fund-business" / code / "business-status.json"
        row = read(path) if path.is_file() else {}
        if row.get("identity") == identity(state) and row.get("fund_code") == code and not row.get("inputs_changed"):
            receipts.append(row)
    for section in ("ratings", "advice", "monthly_compilation"):
        previous = state.get(section) or {}
        updated = {row["fund_code"]: row[section] for row in receipts if section in row}

        # 先保留本轮尚未逐只重跑的正式回执；旧结果不能代替绑定的新验收。
        result = {"scope": state["scope"], "status": "PARTIAL_SUCCESS",
                  "items": preserve(previous.get("items", []), updated) + [
                      item for current in updated.values() for item in current.get("items", [])
                  ]}
        if section == "ratings":
            result["readback"] = preserve(previous.get("readback", []), updated) + [
                item for current in updated.values() for item in current.get("readback", [])
            ]
            covered = set(previous.get("checklist_month_covered", [])) - set(updated)
            covered.update(code for current in updated.values() for code in current.get("checklist_month_covered", []))
            result["checklist_month_covered"] = sorted(covered & scope)
            result["newRule"] = {"items": preserve(previous.get("newRule", {}).get("items", []), updated) + [
                item for current in updated.values() for item in current.get("newRule", {}).get("items", [])
            ]}
        key = {"ratings": "rating_complete", "advice": "advice_complete"}.get(section)
        complete = len(receipts) == len(state["scope"]) and all(
            business_result(row, state, row["fund_code"])[key] if key else
            row.get(section, {}).get("status") == "SUCCEEDED" for row in receipts
        )
        result["status"] = "SUCCEEDED" if complete else "PARTIAL_SUCCESS"
        state[section] = result
        versioned_save(directory / (section.replace("_", "-") + "-status.json"), result)
    versioned_save(directory / "run.json", state)
    return state


def business_result(receipt, state, code):
    """缺回执、旧窗口、只准备未审阅、部分评分均不算正式完整结果。"""
    bound = bool(receipt and receipt.get("identity") == identity(state) and receipt.get("fund_code") == code
                 and not receipt.get("inputs_changed"))
    periods = receipt.get("periods", []) if bound else []
    four = len(periods) == 4 and {row.get("horizon") for row in periods} == HORIZONS
    rating = receipt.get("ratings", {}) if bound else {}
    rows = rating.get("readback", [])
    checked = (rating.get("newRule") or {}).get("items", [])
    scored = bool(
        len(rows) == 1 and rows[0].get("fundCode") == code and rows[0].get("status") == "SCORED"
        and rows[0].get("score") is not None and code in rating.get("checklist_month_covered", [])
        and len(checked) == 1 and checked[0].get("fundCode") == code
        and checked[0].get("status") in {"SCORED", "RATED"}
    )
    advice = receipt.get("advice", {}) if bound else {}
    items = advice.get("items", [])
    advice_complete = bool(
        len(items) == 1 and items[0].get("fund_code") == code and items[0].get("read_back")
        and items[0].get("status") == "READY"
        and not items[0].get("pending_dependencies") and not items[0].get("missing")
        and items[0].get("checklist_current_predictions") and items[0].get("inputs_current") is True
    )
    reviewed_no_direction = bool(
        scored and four and all(row.get("review_complete") and row.get("status") == "REVIEWED_ABSTAIN"
                                 and row.get("source_reasons") and not row.get("data_blocked") for row in periods)
        and len(items) == 1 and items[0].get("fund_code") == code
        and items[0].get("read_back") and items[0].get("inputs_current") is True
        and items[0].get("status") in {"INSUFFICIENT", "QUALIFIED"} and items[0].get("missing")
    )
    return {
        "receipt_bound": bound, "rating_complete": scored,
        "reviews_saved": four and all(row.get("review_complete") for row in periods),
        "predictions_complete": four and all(row.get("counts_as_dated_forecast") for row in periods),
        "advice_complete": advice_complete,
        "reviewed_no_direction": reviewed_no_direction,
        "missing_horizons": [h for h in sorted(HORIZONS) if not any(
            row.get("horizon") == h and row.get("counts_as_dated_forecast") for row in periods
        )],
    }


class Commander:
    def __init__(self, directory, state, at, *, audit=None):
        self.directory, self.state, self.at = Path(directory), state, at
        self.path = self.directory / "commander.json"
        self.binding = identity(state)
        if not self.binding["job_id"] or len(set(state["scope"])) != len(state["scope"]):
            raise ValueError("COMMANDER_RUN_IDENTITY_INVALID")
        self.control = read(self.path) if self.path.is_file() else {
            "identity": self.binding, "assignments": {}, "acceptances": {}, "events": [], "status": "OPEN",
        }
        if self.control["identity"] != self.binding:
            raise ValueError("COMMANDER_RUN_IDENTITY_CHANGED")
        if audit is None:
            from app.commands.fund_daily_check import read_supplement_work

            audit = read_supplement_work(self.directory, state, at)
        self.audit = audit

    def business(self, code):
        path = self.directory / "fund-business" / code / "business-status.json"
        return read(path) if path.is_file() else {}

    def fingerprint(self, code):
        # 逐项状态、证据引用、重查条件和业务结果变化后，旧验收自动失效。
        path = self.directory / "supplement-funds" / f"{code}.json"
        return digest({"ledger": read(path) if path.is_file() else None,
                       "audit": self.audit.get("funds", {}).get(code), "business": self.business(code)})

    def snapshot(self):
        funds = []
        broken = self.audit.get("status") == "RECEIPTS_INCOMPLETE"
        expired = bool(self.state["window"].get("deadline_at") and self.at >= datetime.fromisoformat(
            self.state["window"]["deadline_at"]
        ))
        for code in self.state["scope"]:
            audit = self.audit.get("funds", {}).get(code, {})
            counts = audit.get("counts", {})
            business = business_result(self.business(code), self.state, code)
            assigned = self.control["assignments"].get(code)
            accepted = self.control["acceptances"].get(code, {})
            current_acceptance = accepted.get("fingerprint") == self.fingerprint(code)
            if broken or not counts:
                action = "REPAIR_RECEIPTS"
            elif assigned:
                action = "COLLECT_WORKER"
            elif not current_acceptance:
                action = "VERIFY_RETURN" if accepted else "DISPATCH"
            elif is_midnight(self.state["window"]) and not business["receipt_bound"]:
                # 先用可见资料推进本基金预测，然后继续补清单上的资料缺口。
                action = "RUN_FUND_BUSINESS"
            elif counts.get("CONTINUE", 0) and any(
                row.get("outcome", "CONTINUE") == "CONTINUE" and row.get("field") not in {
                    "advice_predictions", "advice_rating", "advice_ledger", "advice_funding", "advice_conditions",
                } for row in audit.get("remaining", [])
            ):
                action = "RETURN_TO_WORKER"
            elif not business["receipt_bound"] or not business["reviews_saved"]:
                action = "RUN_FUND_BUSINESS"
            elif counts.get("WAITING", 0):
                action = "WAITING"
            elif business["reviewed_no_direction"] and all(
                row.get("field") == "advice_predictions" for row in audit.get("remaining", [])
            ):
                action = "REVIEWED_NO_DIRECTION"
            elif counts.get("CONTINUE", 0):
                action = "RETURN_BUSINESS_GAPS"
            elif not all(business[k] for k in ("rating_complete", "predictions_complete", "advice_complete")):
                action = "RETURN_BUSINESS_GAPS"
            else:
                action = "COMPLETE"
            funds.append({"fund_code": code, "action": action, "worker": assigned,
                          "counts": counts, "remaining": audit.get("remaining", []), **business})
        pending = [row for row in funds if row["action"] not in {"COMPLETE", "WAITING", "REVIEWED_NO_DIRECTION"}]
        status = "MISSED_DEADLINE" if expired else "CONTINUE" if pending else (
            "WAITING" if any(row["action"] == "WAITING" for row in funds) else "DONE_WITH_GAPS"
            if any(row["action"] == "REVIEWED_NO_DIRECTION" for row in funds) else "SUCCEEDED"
        )
        fingerprint = digest({code: self.fingerprint(code) for code in self.state["scope"]})
        check_path = self.directory / "commander-live-check.json"
        verified = read(check_path) if check_path.is_file() else {}
        live_bound = (verified.get("commander_identity") == self.binding
                      and verified.get("commander_fingerprint") == fingerprint)
        # 隔离队列通过不等于真实业务验收：正式完整还必须有本轮独立实时检查成功。
        live_ok = live_bound and (verified.get("status") == "SUCCEEDED" if status == "SUCCEEDED" else
                                 verified.get("status") == "INCOMPLETE" and not any(
                                     "MISMATCH" in issue or "CHANGED" in issue or "MISSING" in issue
                                     or "REVIEWS_INCOMPLETE" in issue or "CHECK_FAILED" in issue
                                     or issue == "REASSESSMENT_REQUIRED" for issue in verified.get("issues", [])))
        from app.services.fund_daily_budget import run_budget

        budget = run_budget(self.directory, self.at)
        return {
            "identity": self.binding, "checked_at": self.at.isoformat(), "status": status,
            "execution_budget": budget,
            "ready_for_final_check": not pending and not expired,
            "can_finish": not pending and not expired and live_ok, "fund_count": len(funds),
            "live_verified": bool(live_ok), "verification_fingerprint": fingerprint,
            "next_action": "FINISH" if not pending and not expired and live_ok and (
                not budget or budget["pause_reason"] != "TIME_LIMIT") else
            "PAUSE_LIMIT" if budget and budget["limit_reached"] else
            "VERIFY_FINAL" if not pending and not expired else "KEEP_ORIGINAL_PENDING" if expired else "CONTINUE",
            "counts": dict(Counter(row["action"] for row in funds)),
            "rating_complete": sum(row["rating_complete"] for row in funds),
            "predictions_complete": sum(row["predictions_complete"] for row in funds),
            "advice_complete": sum(row["advice_complete"] for row in funds), "funds": funds,
        }

    def change(self, action, *, code=None, worker=None, reason=None, limit=5):
        if code is not None and code not in self.state["scope"]:
            raise ValueError("COMMANDER_FUND_OUTSIDE_RUN_SCOPE")
        if action not in {"dispatch", "accept", "release", "checkpoint", "finish"}:
            raise ValueError("COMMANDER_ACTION_INVALID")
        before = self.snapshot()
        if action == "dispatch":
            if before["status"] == "MISSED_DEADLINE":
                raise ValueError("COMMANDER_ORIGINAL_WINDOW_EXPIRED")
            active = self.control["assignments"]
            if not code or not worker or not 1 <= limit <= 5:
                raise ValueError("COMMANDER_DISPATCH_ARGUMENTS_REQUIRED")
            if code in active or worker in active.values():
                raise ValueError("COMMANDER_DUPLICATE_ASSIGNMENT")
            if len(active) >= limit:
                raise ValueError("COMMANDER_SUPPLEMENT_LIMIT")
            active[code] = worker
            self.control["status"] = "OPEN"
        elif action == "accept":
            if not code:
                raise ValueError("COMMANDER_FUND_REQUIRED")
            self.control["acceptances"][code] = {
                "fingerprint": self.fingerprint(code), "checked_at": self.at.isoformat(),
            }
            self.control["assignments"].pop(code, None)
            self.control["status"] = "OPEN"
            # 验收不合格立即生成原基金退回清单；accept 的含义是执行检查，绝非强制通过。
        elif action == "release":
            if not code or not reason:
                raise ValueError("COMMANDER_RELEASE_REASON_REQUIRED")
            self.control["assignments"].pop(code, None)
            self.control["acceptances"].pop(code, None)
            self.control["status"] = "OPEN"
        elif action == "checkpoint":
            if not reason:
                raise ValueError("COMMANDER_INTERRUPTION_REASON_REQUIRED")
            self.control["status"] = "INTERRUPTED"
        elif not before["can_finish"]:
            # 不写完成标志，完整保留仍可行动的清单。
            self.control["events"].append({"action": "FINISH_REJECTED", "at": self.at.isoformat(),
                                           "counts": before["counts"]})
            versioned_save(self.path, self.control)
            versioned_save(self.directory / "commander-next.json", before)
            return before
        else:
            self.control["status"] = before["status"]
            self.state["commander_status"] = before["status"]
            self.state["status"] = "CLOSED" if before["status"] == "SUCCEEDED" else "COMMANDER_" + before["status"]
            versioned_save(self.directory / "run.json", self.state)
        self.control["events"].append({"action": action.upper(), "fund_code": code, "worker": worker,
                                       "reason": reason, "at": self.at.isoformat()})
        versioned_save(self.path, self.control)
        snapshot = self.snapshot()
        versioned_save(self.directory / "commander-next.json", snapshot)
        return snapshot
