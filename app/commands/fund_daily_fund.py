"""总指挥逐基金调用正式评分、严格审阅发布和建议；不缩小原运行的持仓名单。"""

import argparse
import json
from pathlib import Path

from app.services.direction_1d_protocol import digest
from app.services.fund_daily_budget import assert_daily_budget
from app.services.fund_daily_commander import aggregate_business, identity
from app.services.fund_daily_prediction import FUND_INPUT_ERRORS
from app.services.fund_exposure_common import PROJECT, now, read
from app.services.fund_materials_store import versioned_save
from app.services.material_content import material_read_cache


def advance_fund(directory, state, code, backend, *, publish=False):
    assert_daily_budget(directory, now())
    at, owner, scope, window = backend.context()
    if (owner != state["owner_user_id"] or scope != state["scope"] or window != state["window"]
            or at[:10] != state["checked_at"][:10] or code not in scope):
        raise ValueError("DAILY_FUND_ORIGINAL_CONTEXT_CHANGED")
    scoped = {**state, "scope": [code]}
    output = directory / "fund-business" / code
    result = {"identity": identity(state), "fund_code": code, "checked_at": at, "periods": [], "prepared": []}
    # 每次只处理一只，后排基金不再被整批十分钟时限跳过；完整原名单始终留在 run.json。
    result["monthly_compilation"] = backend.prepare_monthly([code])
    assert_daily_budget(directory, now())
    result["ratings"] = backend.ratings(state.get("materials_terminal") or {}, scoped)
    versioned_save(output / "business-status.json", result)
    periods = backend.periods(code, window)
    for period in periods:
        assert_daily_budget(directory, now())
        try:
            if backend.read_back(code, period["horizon"], window) or period["status"] != "READY":
                continue
            packet = backend.freeze(code, period["horizon"], directory / "funds" / code / period["horizon"])
            result["prepared"].append(packet)
            versioned_save(output / "business-status.json", result)
            # 没有真实审阅文件及严格evaluate结果，绝不自动写方向或暂不判断。
            if publish and packet.get("evaluation") and Path(packet["review_path"]).is_file():
                backend.import_and_archive(code, period["horizon"], Path(packet["directory"]), owner, window)
        except FUND_INPUT_ERRORS as error:
            result.setdefault("failures", []).append({"horizon": period["horizon"], "type": type(error).__name__,
                                                       "reason": str(error)[:250]})
            versioned_save(output / "business-status.json", result)
    result["periods"] = backend.periods(code, window)
    assert_daily_budget(directory, now())
    result["advice"] = backend.advice(
        {**scoped, "ratings": result["ratings"]}, {"funds": [{"fund_code": code, "horizons": result["periods"]}]},
    )
    from app.services.fund_daily_advice import read_saved_advice

    saved = {row["fund_code"]: row for row in read_saved_advice(owner, result["advice"], backend=backend)}
    for row in result["advice"]["items"]:
        current = saved.get(row["fund_code"], {})
        row["inputs_current"] = bool(current.get("read_back") and current.get("inputs_current"))
    result["checked_at"] = now().isoformat()
    versioned_save(output / "business-status.json", result)
    return result


def supplement_input(directory, code):
    path = directory / "supplement-funds" / (code + ".json")
    return digest(read(path)) if path.is_file() else None


def merge_fund_result(directory, result, original_input, *, live_input_matches=True):
    """调用方持全局锁；合入本基金回执时禁止覆盖计算期间其他基金的更新。"""
    latest = read(directory / "run.json")
    code = result["fund_code"]
    if (not live_input_matches or identity(latest) != result["identity"]
            or supplement_input(directory, code) != original_input):
        rejected = {**result, "inputs_changed": True}
        versioned_save(directory / "fund-business" / code / "business-status.json", rejected)
        raise ValueError("DAILY_FUND_INPUT_CHANGED_BEFORE_MERGE")
    return aggregate_business(directory, latest, fund_code=code)


@material_read_cache
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fund-code", required=True)
    parser.add_argument("--publish-prepared", action="store_true")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if directory.parent != (PROJECT / ".local-runs").resolve():
        parser.error("逐基金业务只使用本项目原运行目录")
    from app.commands.fund_daily_prediction import fund_execution_lock, run_lock
    from app.services.fund_daily_runtime import LocalDailyBackend
    from app.services.prediction_trial_v2 import storage_check

    code = args.fund_code
    at = now()
    storage_check(at.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
    with fund_execution_lock(directory, code):
        with run_lock(wait=True):
            state = read(directory / "run.json")
            if code not in state["scope"]:
                parser.error("基金不在本轮真实持仓范围")
            original_input = supplement_input(directory, code)
        backend = LocalDailyBackend(directory)
        try:
            result = advance_fund(directory, state, code, backend, publish=args.publish_prepared)
            live_at, live_owner, live_scope, live_window = backend.context()
            live_input_matches = (
                live_at[:10] == result["identity"]["day"] and live_owner == state["owner_user_id"]
                and live_scope == state["scope"] and live_window == state["window"]
                and backend.periods(code, live_window) == result["periods"]
            )
        finally:
            backend.close()
        storage_check(at.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
        with run_lock(wait=True):
            merge_fund_result(directory, result, original_input, live_input_matches=live_input_matches)
    print(json.dumps({"fund_code": result["fund_code"], "receipt": str(
        directory / "fund-business" / args.fund_code / "business-status.json"
    ), "periods": result["periods"], "prepared": result["prepared"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
