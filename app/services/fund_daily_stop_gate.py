"""原生Stop结束检查：只有原轮真实验收成功，才允许总指挥正常收尾。"""

from datetime import datetime
from pathlib import Path

from app.services.fund_daily_budget import budget_status, limits
from app.services.fund_daily_commander import Commander, identity
from app.services.fund_exposure_common import read


def armed_ticket(directory, state, thread_id, at):
    if directory.name != "daily-prediction-" + at.strftime("%Y%m%d"):
        raise ValueError("STOP_GATE_WRONG_DAY_DIRECTORY")
    if state and state.get("checked_at", "")[:10] != at.date().isoformat():
        raise ValueError("STOP_GATE_RUN_NOT_CURRENT_DAY")
    return {"enabled": True, "thread_id": thread_id, "day": at.date().isoformat(),
            "run_dir": str(directory), "armed_at": at.isoformat(), "mode": "FULL_ACCEPTANCE_OR_LIMITS",
            "limits": limits(), "completed_rounds": 0, "round_started_at": at.isoformat(),
            "identity": identity(state) if state else None}


def block(reason):
    return {"decision": "block", "reason": reason}


def pause_for_budget(budget):
    reason = (f"总指挥达到本轮上限：已运行{budget['elapsed_minutes']}分钟，"
              f"已完成{budget['completed_rounds']}轮复核；上限{budget['max_runtime_minutes']}分钟或"
              f"{budget['max_rounds']}轮，任一先到就暂停。保留原缺口，本轮未验收完成。")
    return {"continue": False, "stopReason": reason, "systemMessage": reason}, {
        "status": "PAUSED_" + budget["pause_reason"], "accepted": False, "budget": budget}


def evaluate_stop(payload, ticket, at, *, snapshot=None, state=None, control=None):
    """只判断原总指挥，不接管普通聊天、子窗口或用户主动中断。"""
    if (payload.get("hook_event_name") != "Stop" or not ticket.get("enabled")
            or payload.get("session_id") != ticket.get("thread_id")):
        return {}, {"status": "NOT_APPLICABLE"}
    directory = Path(ticket["run_dir"])
    if ticket.get("day") != at.date().isoformat():
        # 日期、市场窗口属于业务边界；错过后暂停，绝不把当时未发布的预测记成成功。
        return {"continue": False, "stopReason": "原每日任务已跨日，保留未验收状态，请按原窗口报告未完成。"}, {
            "status": "PAUSED_DATE_CHANGED", "accepted": False}
    budget = budget_status(ticket, at)
    state_path = directory / "run.json"
    if state is None:
        state = read(state_path) if state_path.is_file() else {}
    if not state:
        if budget["limit_reached"]:
            return pause_for_budget(budget)
        return block("本轮总指挥尚未完成初始化，不能收尾。先在原工作区载入dev-env.ps1，"
                     "使用fund_daily_prediction --initialize-only读取本人的真实持仓并生成今天完整清单，"
                     "随后按总指挥持久队列实际逐基金执行、复核、打回和续办。"), {
                         "status": "BLOCKED_NOT_STARTED", "accepted": False}
    if (state.get("checked_at", "")[:10] != ticket["day"]
            or ticket.get("identity") and ticket["identity"] != identity(state)):
        if budget["limit_reached"]:
            return pause_for_budget(budget)
        return block("原运行的账号、完整持仓、日期或估值窗口与启动记录不符。核实并恢复原轮次，"
                     "不能缩小名单或换日期来通过结束检查。"), {
                         "status": "BLOCKED_IDENTITY_CHANGED", "accepted": False}
    deadline = state["window"].get("deadline_at")
    if deadline and at >= datetime.fromisoformat(deadline):
        # 回调晚到时保留先触发的时间上限，不把14:52已暂停的业务改写为15:00才过窗。
        if (budget["limit_reached"] and budget["pause_reason"] == "TIME_LIMIT"
                and datetime.fromisoformat(budget["deadline_at"]) <= datetime.fromisoformat(deadline)):
            return pause_for_budget(budget)
        return {"continue": False, "stopReason": "已超过原预测窗口，任务未验收；保留原待办并报告错过窗口。"}, {
            "status": "PAUSED_WINDOW_EXPIRED", "accepted": False}
    if snapshot is None:
        snapshot = Commander(directory, state, at).snapshot()
    if control is None:
        path = directory / "commander.json"
        control = read(path) if path.is_file() else {}
    # 不用回答文本、只审阅或外部等待代替完整业务；必须实时检查成功且finish已正式通过。
    accepted = bool(snapshot.get("status") == "SUCCEEDED" and snapshot.get("can_finish")
                    and snapshot.get("live_verified") and control.get("status") == "SUCCEEDED"
                    and control.get("identity") == identity(state))
    # 第5轮确已合格可以正常完成；晚执行的Stop也不能否认上限前已正式完成的记录。
    if accepted and budget["pause_reason"] == "TIME_LIMIT":
        completed_before_limit = any(
            event.get("action") == "FINISH" and event.get("at")
            and datetime.fromisoformat(event["at"]) <= datetime.fromisoformat(budget["deadline_at"])
            for event in control.get("events", [])
        )
        accepted = completed_before_limit
    if accepted:
        return {}, {"status": "ACCEPTED", "accepted": True, "identity": identity(state)}
    if budget["limit_reached"]:
        return pause_for_budget(budget)
    counts = snapshot.get("counts", {})
    first = [{"code": row["fund_code"], "action": row["action"]}
             for row in snapshot.get("funds", []) if row.get("action") != "COMPLETE"][:5]
    reason = (
        f"总指挥结束验收未通过，原任务仍未完成。原目录：{directory}。"
        f"下一步：{snapshot.get('next_action')}；基金状态：{counts}；首批续办：{first}。"
        "请实际执行fund_daily_commander --run-dir <原目录> --action status --details，"
        "按原队列逐项补件、读证据、复核、退回并再次验收；窗口结束不等于验收通过。"
        "已具备条件的基金立即逐只评分、真实审阅四周期并正式发布读回，再更新建议。"
        "全部条件满足后执行--action verify，再按结果续办，最后--action finish。"
        "不得只解释缺项或写一份未完成总结后再次结束，不得套用暂不判断或旧评分充数。"
        f"本轮最多{budget['max_runtime_minutes']}分钟或{budget['max_rounds']}轮，先到先暂停且不算完成。"
        f"当前已完成{budget['completed_rounds']}轮；初次全持仓处理算第1轮。"
        "逐只实际处理并验收全部持仓后，使用根入口dev.ps1 DailyRoundComplete登记一轮；"
        "未覆盖全部持仓或只有未审阅的准备记录不能登记一轮，Stop次数不是复核轮数。"
    )
    # stop_hook_active=true仍按真实结果检查，不能把第二次想结束当成自动豁免。
    return block(reason), {"status": "BLOCKED_INCOMPLETE", "accepted": False, "counts": counts,
                           "next_action": snapshot.get("next_action"), "first": first}
