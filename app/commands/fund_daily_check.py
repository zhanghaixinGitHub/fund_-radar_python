"""每日只读核对资料、持仓评分及四周期正式归档，漏跑和缺回执一律不算成功。"""

import argparse
import json
from collections import Counter
from datetime import datetime
from functools import cache
from pathlib import Path

from app.services.direction_1d_protocol import digest
from app.services.fund_daily_prediction import supplement_run_id
from app.services.fund_daily_runtime import PROJECT, LocalDailyBackend
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save


def assess(state, followthrough, scope, ratings_readback, checked_at, live_window=None, one_day_readback=None,
           advice_readback=None, supplements=None):
    issues = []
    if state.get("owner_user_id") != followthrough.get("owner_user_id"):
        issues.append("OWNER_MISMATCH")
    if set(state.get("scope", [])) != set(scope):
        issues.append("HOLDINGS_CHANGED")
    if state.get("scope_hash") != digest(scope):
        issues.append("SCOPE_RECEIPT_MISMATCH")
    if state.get("checked_at", "")[:10] != checked_at[:10]:
        issues.append("RUN_NOT_CHECKED_TODAY")
    if state.get("window") != followthrough.get("window"):
        issues.append("PREDICTION_WINDOW_CHANGED")
    if live_window and state.get("window", {}).get("target_nav_date") != live_window["target_nav_date"]:
        issues.append("LIVE_PREDICTION_WINDOW_CHANGED")
    if supplements is not None:
        window = state.get("window", {})
        if (not supplement_run_id(state) or supplements.get("job_id") != supplement_run_id(state)
                or supplements.get("day") != checked_at[:10]
                or not supplements.get("checked_at") or supplements["checked_at"] > checked_at
                or any(supplements.get(k) != window.get(k) for k in ("base_nav_date", "target_nav_date"))
                or set(supplements.get("funds", {})) != set(scope)
                or supplements.get("item_audit_complete") is False):
            issues.append("SUPPLEMENT_RECEIPT_MISMATCH")
        elif supplements.get("status") != "SUCCEEDED" or supplements.get("pending_item_count") != 0:
            issues.append("SUPPLEMENT_ITEMS_PENDING")
            if supplements.get("unattempted_item_count"):
                issues.append("SUPPLEMENT_ITEMS_NOT_ATTEMPTED")
    if followthrough.get("analysis_version") == "COMPREHENSIVE_REVIEW_V2":
        if followthrough.get("status") != "CLOSED" or not followthrough.get("all_reviews_saved"):
            issues.append("PREDICTION_REVIEWS_INCOMPLETE")
        if followthrough.get("reassessment_pending_count", 0):
            issues.append("REASSESSMENT_REQUIRED")
        if followthrough.get("reassessment_check_failure_count", 0):
            issues.append("REASSESSMENT_CHECK_FAILED")
    elif followthrough.get("status") != "CLOSED" or not followthrough.get("all_predictions_generated"):
        issues.append("PREDICTIONS_INCOMPLETE")
    materials = state.get("materials_terminal") or {}
    if not materials.get("terminal") or (
        state.get("supplement_run_id") and
        (materials.get("mode") != "FUND_CHECKLIST" or materials.get("job_id") != supplement_run_id(state))
    ):
        issues.append("MATERIALS_NOT_FINISHED")
    elif any(
        c.get("status") not in {"SUCCEEDED", "PARTIAL_SUCCESS"}
        for c in materials.get("children", [])
        if c.get("job_type") not in {"FUND_RATINGS", "DIRECTION_1D_PREDICTIONS"}
    ):
        issues.append("MATERIAL_STAGE_FAILED")
    outcome = state.get("prediction_outcomes") or {}
    summary = outcome.get("summary") or {}
    counts = {"total", "checked", "completed", "created", "pending", "failed", "remaining"}
    if not counts <= summary.keys():
        issues.append("OUTCOME_RECEIPT_MISSING")
    elif any(summary[k] for k in ("pending", "failed", "remaining")):
        issues.append("OLD_PREDICTION_OUTCOMES_INCOMPLETE")
    daily = state.get("one_day_outcomes") or {}
    daily_summary = daily.get("summary") or {}
    if not counts <= daily_summary.keys():
        issues.append("ONE_DAY_OUTCOME_RECEIPT_MISSING")
    elif (
        daily.get("status") != "SUCCEEDED"
        or daily.get("owner_user_id") != state.get("owner_user_id")
        or set(daily.get("scope", [])) != set(scope)
        or daily.get("dueBefore") != checked_at[:10]
        or daily.get("checked_at", "")[:10] != checked_at[:10]
        or any(daily_summary[k] for k in ("pending", "failed", "remaining"))
        or daily_summary["total"] != daily_summary["completed"]
    ):
        issues.append("ONE_DAY_OUTCOMES_INCOMPLETE")
    if one_day_readback is None:
        issues.append("ONE_DAY_OUTCOME_READBACK_MISSING")
    elif (
        any(one_day_readback[k] for k in ("pending", "failed", "remaining"))
        or one_day_readback["total"] != one_day_readback["completed"]
        or one_day_readback["total"] != daily_summary.get("total")
    ):
        issues.append("ONE_DAY_OUTCOME_READBACK_INCOMPLETE")
    rating = state.get("ratings") or {}
    if rating.get("status") != "SUCCEEDED" or set(rating.get("scope", [])) != set(scope):
        issues.append("RATINGS_INCOMPLETE")
    if len(ratings_readback) != len(scope):
        issues.append("RATING_READBACK_INCOMPLETE")
    elif {r.get("fundCode") for r in ratings_readback} != set(scope):
        issues.append("RATING_READBACK_SCOPE_MISMATCH")
    elif any(r.get("status") not in {"RATED", "SCORED"} or r.get("score") is None for r in ratings_readback):
        issues.append("RATING_RESULTS_NOT_COMPLETE")
    if state.get("workflow_version") == "HOLDINGS_END_TO_END_V1":
        from datetime import date

        from app.services.fund_rating_period import last_completed_month

        target = str(last_completed_month(date.fromisoformat(checked_at[:10])))[:7]
        if any((r.get("asOfDate") or "")[:7] < target for r in ratings_readback):
            issues.append("MONTHLY_RATINGS_OUTDATED")
        advice = state.get("advice", {})
        if advice.get("status") != "SUCCEEDED" or set(advice.get("scope", [])) != set(scope):
            issues.append("ADVICE_INPUTS_PENDING")
        if advice_readback is None or {r["fund_code"] for r in advice_readback} != set(scope):
            issues.append("ADVICE_READBACK_INCOMPLETE")
        elif any(not r["read_back"] for r in advice_readback):
            issues.append("ADVICE_REPORTS_NOT_COMPLETE")
        if advice_readback is not None and any(
            r.get("read_back") and not r.get("inputs_current") for r in advice_readback
        ):
            issues.append("ADVICE_INPUTS_CHANGED_OR_UNVERIFIED")
    return {
        "status": "SUCCEEDED" if not issues else "INCOMPLETE",
        "checked_at": checked_at,
        "scope_kind": "SIMULATED_HOLDINGS",
        "scope": scope,
        "fund_count": len(scope),
        "issues": issues,
        "predictions": followthrough,
        "ratings_readback": ratings_readback,
        "prediction_outcomes": outcome,
        "one_day_outcomes": daily,
        "one_day_outcome_readback": one_day_readback,
        "advice": state.get("advice"),
        "advice_readback": advice_readback,
        "supplements": supplements,
    }


def read_supplement_check(directory, state):
    """只读原范围的逐项回执及待办，核对本轮统计有没有漏项或串期。"""
    from app.services.fund_daily_profiles import VERIFIED_SOURCE_RESULTS
    from app.services.fund_daily_supplements import task_key

    summary_path = directory / "supplement-execution.json"
    if not summary_path.is_file():
        return {}
    summary = read(summary_path)
    paths = [directory / name for name in ("supplement-catalog.json", "supplement-tasks.json", "source-rechecks.json")]
    if not all(path.is_file() for path in paths):
        return {**summary, "item_audit_complete": False}
    catalog, queue, source = (read(path) for path in paths)
    expected = {task_key(row) for row in catalog["tasks"]}
    rows = [row for row in source["items"] if task_key(row) in expected]
    awaiting = {task_key(row) for row in rows if row["status"] not in VERIFIED_SOURCE_RESULTS}
    summary["item_audit_complete"] = (
        catalog.get("job_id") == queue.get("job_id") == source.get("job_id", summary["job_id"]) == summary["job_id"]
        and catalog.get("window") == queue.get("window") == state["window"]
        and all(source.get(field) == summary.get(field) for field in ("day", "base_nav_date", "target_nav_date"))
        and {task_key(row) for row in rows} == expected
        and len(rows) == len(expected) == summary.get("item_count")
        and Counter(row["status"] for row in rows) == summary.get("counts")
        and awaiting == {task_key(row) for row in queue["tasks"]}
        and len(queue["tasks"]) == len(awaiting) == summary.get("pending_item_count")
        and all(row.get("summary") and row.get("evidence_ref") and row.get("next_source") for row in rows)
    )
    return summary


def read_supplement_work(directory, state, at, code=None):
    """按原清单派发未通过验收的条目。只读本地记录，不生成上午检查回执。"""
    from app.services.fund_daily_profiles import VERIFIED_SOURCE_RESULTS
    from app.services.fund_daily_supplements import task_key

    window = state["window"]
    job = supplement_run_id(state)
    scope = state["scope"]
    if code and code not in scope:
        raise ValueError("SUPPLEMENT_FUND_OUTSIDE_RUN_SCOPE")
    funds = {fund: {"counts": {}, "remaining": []} for fund in ([code] if code else scope)}
    result = {"job_id": job, "window": window, "funds": funds}
    catalog_path = directory / "supplement-catalog.json"
    if not catalog_path.is_file():
        return {**result, "status": "RECEIPTS_INCOMPLETE", "reason": "SUPPLEMENT_CATALOG_MISSING"}

    catalog = read(catalog_path)
    source_path = directory / "source-rechecks.json"
    source = read(source_path) if source_path.is_file() else {}
    tasks = catalog["tasks"]
    keys = [task_key(task) for task in tasks]
    day = state["checked_at"][:10]
    if (not job or catalog.get("job_id") != job or catalog.get("window") != window
            or set(k[0] for k in keys) != set(scope) or len(keys) != len(set(keys))):
        return {**result, "status": "RECEIPTS_INCOMPLETE", "reason": "SUPPLEMENT_CATALOG_MISMATCH"}

    source_rows = {}
    if (source.get("job_id") == job and source.get("day") == day
            and all(source.get(k) == window[k] for k in ("base_nav_date", "target_nav_date"))):
        source_rows = {task_key(row): row for row in source["items"]}

    @cache
    def evidence_file(reference):
        if not reference:
            return None
        path = (PROJECT / reference).resolve()
        return path if path.is_relative_to(PROJECT.parent) and path.is_file() else None

    @cache
    def evidence_hash(path):
        from hashlib import sha256

        from app.services.material_file_access import original_bytes

        return sha256(original_bytes(path)).hexdigest()

    fund_rows = {}
    for fund in funds:
        path = directory / "supplement-funds" / f"{fund}.json"
        saved = read(path) if path.is_file() else {}
        if (saved.get("fund_code") == fund and saved.get("day") == day
                and saved.get("job_id") == job and saved.get("window") == window):
            fund_rows.update({task_key(row): row for row in saved["items"] if row.get("fund_code") == fund})

    for key, task in zip(keys, tasks, strict=True):
        fund, field, required_date = key
        if fund not in funds:
            continue
        row = source_rows.get(key, {})
        local = fund_rows.get(key, {})
        status = row.get("status", "NOT_ATTEMPTED")
        reference = evidence_file(row.get("evidence_ref"))
        have_evidence = bool(reference and row.get("summary") and row.get("attempted_source")
                             and row.get("attempted_at", "")[:10] == day
                             and all(row.get(k) == local.get(k) for k in ("status", "evidence_ref", "attempted_at")))
        if have_evidence and row.get("evidence_sha256"):
            have_evidence = row["evidence_sha256"] == evidence_hash(reference)
        substantive_evidence = have_evidence and (
            reference != (directory / "supplement-funds" / f"{fund}.json").resolve() or row.get("observations")
        )
        retry = row.get("retry_at") or task.get("retry_at")
        try:
            retry_at = datetime.fromisoformat(retry) if retry else None
        except (ValueError, TypeError):
            retry_at = None
        later = retry_at is not None and retry_at.tzinfo is not None and retry_at > at
        if status in VERIFIED_SOURCE_RESULTS and substantive_evidence:
            outcome = status
        elif ((status in {"NOT_PUBLISHED", "TEMPORARY_FAILURE"} and have_evidence and later)
              or (status == "ACCESS_REQUIRED" and substantive_evidence)) and row.get("next_source"):
            outcome = "WAITING"
        else:
            outcome = "CONTINUE"

        record = funds[fund]
        counts = record["counts"]
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome in {"CONTINUE", "WAITING"}:
            record["remaining"].append({
                "field": field, "required_date": required_date, "outcome": outcome, "receipt_status": status,
                "summary": row.get("summary"), "evidence_ref": row.get("evidence_ref"),
                "next_source": row.get("next_source") or task.get("next_source"), "retry_at": retry,
            })

    audit = read_supplement_check(directory, state)
    counts = Counter()
    for record in funds.values():
        counts.update(record["counts"])
    if not audit.get("item_audit_complete"):
        status = "RECEIPTS_INCOMPLETE"
    elif counts["CONTINUE"]:
        status = "CONTINUE"
    elif counts["WAITING"]:
        status = "WAITING"
    else:
        status = "SUCCEEDED"
    return {**result, "status": status, "counts": dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path, help="人工复验另存，不覆盖当日09:00检查记录")
    parser.add_argument("--supplement-work", nargs="?", const="", metavar="基金代码",
                        help="只读逐项补件验收和续办清单；省略代码时检查本轮全部持仓")
    args = parser.parse_args()
    checked_at = now()
    at = checked_at.isoformat()
    directory = args.run_dir or PROJECT / ".local-runs" / ("daily-prediction-" + checked_at.strftime("%Y%m%d"))
    path = directory / "run.json"
    if args.supplement_work is not None:
        result = (read_supplement_work(directory, read(path), checked_at, args.supplement_work) if path.is_file()
                  else {"status": "NOT_STARTED", "reason": "RUN_MISSING"})
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(0 if result["status"] == "SUCCEEDED" else 2)
    output = args.output or PROJECT / ".local-runs" / "daily-checks" / (now().strftime("%Y%m%d") + ".json")
    if not output.resolve().is_relative_to(PROJECT / ".local-runs"):
        raise ValueError("DAILY_CHECK_OUTPUT_OUTSIDE_LOCAL_RUNS")
    from app.services.prediction_trial_v2 import storage_check

    storage_check(now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
    if not path.exists():
        result = {"status": "NOT_STARTED", "checked_at": at, "issues": ["TODAYS_RUN_MISSING"]}
    else:
        backend = LocalDailyBackend(directory)
        try:
            _, _, scope, live_window = backend.context()
            # 重新交叉核对两端原预测、当前资料摘要和本人归档，不只读昨天的成功标志。
            guard_output = output.with_name(output.stem + "-followthrough.json")
            followthrough = backend.guard(guard_output, None)
            ratings = [backend.get(f"http://127.0.0.1:8000/internal/v1/funds/{c}/rating") for c in scope]
            state = read(path)
            from app.services.fund_daily_advice import read_saved_advice

            advice = read_saved_advice(state["owner_user_id"], state.get("advice", {}), backend=backend)
            supplements = read_supplement_check(directory, state)
            result = assess(
                state, followthrough, scope, ratings, at, live_window, backend.one_day_outcome_status(), advice,
                supplements,
            )
        except Exception as error:
            result = {
                "status": "FAILED",
                "checked_at": at,
                "issues": [type(error).__name__],
                "reason": str(error)[:300],
            }
        finally:
            backend.close()
    versioned_save(output, result)
    print(
        json.dumps({"status": result["status"], "issues": result["issues"], "output": str(output)}, ensure_ascii=False)
    )
    raise SystemExit(0 if result["status"] == "SUCCEEDED" else 2)


if __name__ == "__main__":
    main()
