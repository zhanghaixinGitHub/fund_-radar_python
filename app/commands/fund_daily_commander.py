"""总指挥派工、逐只验收及收尾检查；检查失败返回2，保留原队列继续执行。"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from app.services.fund_daily_commander import Commander
from app.services.fund_exposure_common import PROJECT, now, read


def verify_live(directory, state, backend):
    """独立重读本人归档、当前评分和建议；不把隔离回执的通过当作线上验收。"""
    from app.commands.fund_daily_check import assess, read_supplement_check
    from app.services.fund_daily_advice import read_saved_advice
    from app.services.fund_daily_commander import aggregate_business, identity
    from app.services.fund_materials_store import versioned_save

    at, owner, scope, window = backend.context()
    if (owner != state["owner_user_id"] or set(scope) != set(state["scope"])
            or window != state["window"] or at[:10] != state["checked_at"][:10]):
        raise ValueError("COMMANDER_LIVE_CONTEXT_CHANGED")
    ratings = [backend.get(f"http://127.0.0.1:8000/internal/v1/funds/{code}/rating") for code in scope]
    current_ratings = {row["fundCode"]: row for row in ratings}
    binding = identity(state)
    for code in state["scope"]:
        path = directory / "fund-business" / code / "business-status.json"
        if not path.is_file():
            continue
        row = read(path)
        if row.get("identity") != binding:
            raise ValueError("COMMANDER_BUSINESS_RECEIPT_CHANGED")
        row["periods"] = backend.periods(code, window)
        if row.get("ratings", {}).get("readback") != [current_ratings[code]]:
            row.setdefault("ratings", {})["readback"] = [current_ratings[code]]
            row["ratings"]["newRule"] = {"items": [{"fundCode": code, "status": "CHANGED_RECHECK_REQUIRED"}]}
        saved = {item["fund_code"]: item for item in read_saved_advice(owner, row.get("advice", {}), backend=backend)}
        for item in row.get("advice", {}).get("items", []):
            current = saved.get(code, {})
            item["inputs_current"] = bool(current.get("read_back") and current.get("inputs_current"))
            item["read_back"] = bool(current.get("read_back"))
        versioned_save(path, row)
    aggregate_business(directory, state)
    state["prediction_outcomes"] = backend.prediction_outcomes(scope)
    versioned_save(directory / "run.json", state)
    state["one_day_outcomes"] = backend.one_day_outcomes()
    versioned_save(directory / "run.json", state)
    if state.get("supplement_run_id"):
        state["materials_terminal"] = {"mode": "FUND_CHECKLIST", "job_id": state["supplement_run_id"],
                                       "terminal": True, "status": state.get("supplements", {}).get("status")}
    state["checked_at"] = now().isoformat()
    versioned_save(directory / "run.json", state)
    followthrough = backend.guard(directory / "commander-followthrough.json", None)
    advice = read_saved_advice(owner, state.get("advice", {}), backend=backend)
    result = assess(state, followthrough, scope, ratings, now().isoformat(), window,
                    backend.one_day_outcome_status(), advice, read_supplement_check(directory, state))
    c = Commander(directory, state, now())
    result.update(commander_identity=c.binding, commander_fingerprint=c.snapshot()["verification_fingerprint"])
    versioned_save(directory / "commander-live-check.json", result)
    return Commander(directory, state, now()).snapshot()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--action", default="status",
                        choices=("status", "template", "dispatch", "merge", "accept", "release", "checkpoint",
                                 "verify", "finish"))
    parser.add_argument("--fund-code")
    parser.add_argument("--worker")
    parser.add_argument("--reason")
    parser.add_argument("--receipt", type=Path, help="合并总指挥实际复核后的单基金完整条目回执")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--details", action="store_true", help="输出逐项退回清单，默认只输出下一动作摘要")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if directory.parent != (PROJECT / ".local-runs").resolve():
        parser.error("总指挥只使用本项目 .local-runs 下的原运行目录")
    at = now()
    from app.commands.fund_daily_prediction import fund_execution_lock, run_lock
    from app.services.prediction_trial_v2 import storage_check

    def execute():
        state = read(directory / "run.json")
        if args.action == "status":
            return Commander(directory, state, at).snapshot()
        if args.action == "template":
            from app.services.fund_daily_commander import identity

            if args.fund_code not in state["scope"]:
                raise ValueError("COMMANDER_FUND_REQUIRED")
            return {"identity": identity(state), "fund_code": args.fund_code, "reviewed_by": "",
                    "items": read(directory / "supplement-funds" / (args.fund_code + ".json"))["items"]}
        storage_check(at.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
        if args.action in {"dispatch", "verify"}:
            from app.services.fund_daily_budget import assert_daily_budget

            assert_daily_budget(directory, at)
        if args.action == "verify":
            from app.services.fund_daily_runtime import LocalDailyBackend

            backend = LocalDailyBackend(directory)
            try:
                return verify_live(directory, state, backend)
            finally:
                backend.close()
        if args.action == "merge":
            from app.services.fund_daily_commander import merge_worker_receipt

            if not args.receipt or not args.receipt.resolve().is_relative_to(directory):
                raise ValueError("COMMANDER_RECEIPT_OUTSIDE_RUN")
            receipt = read(args.receipt)
            if receipt.get("fund_code") != args.fund_code:
                raise ValueError("COMMANDER_MERGE_FUND_MISMATCH")
            state["checked_at"] = at.isoformat()
            merge_worker_receipt(directory, state, receipt, project=PROJECT)
        commander = Commander(directory, state, at)
        if args.action == "merge":
            return commander.change("accept", code=args.fund_code)
        return commander.change(args.action, code=args.fund_code, worker=args.worker,
                                reason=args.reason, limit=args.limit)

    if args.action in {"status", "template"}:
        result = execute()
    else:
        with run_lock(directory / "commander-control.lock", wait=args.action in {"dispatch", "release", "merge"}):
            if args.action in {"dispatch", "release"}:
                # 派工只改自己的队列。单基金评分重算占用业务锁时仍可派工和收回窗口。
                result = execute()
            else:
                with ExitStack() as locks:
                    if args.action == "merge":
                        locks.enter_context(fund_execution_lock(directory, args.fund_code))
                    locks.enter_context(run_lock(wait=args.action == "merge"))
                    if args.action == "verify":
                        for code in read(directory / "run.json")["scope"]:
                            locks.enter_context(fund_execution_lock(directory, code))
                    result = execute()
    if args.action != "template" and not args.details:
        result["funds"] = [{k: v for k, v in row.items() if k != "remaining"} for row in result["funds"]]
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if args.action == "template" or result["can_finish"] else 2)


if __name__ == "__main__":
    main()
