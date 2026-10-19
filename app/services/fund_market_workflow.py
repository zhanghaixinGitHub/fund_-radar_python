"""一键同步的市场资料编排：按本次真实结果放行，月度评级不随日常同步重复执行。"""

from datetime import timedelta

from app.core.logging import get_logger
from app.services.fund_exposure_common import now
from app.services.fund_market_preparation import MarketFundPreparation, market_scope, validate_codes

logger = get_logger(__name__)


class MarketSyncWorkflow:
    """生命周期限定为一个批次；来源失败不影响其他基金采集，也不复用上次就绪标志。"""

    def __init__(self, *, preparation=None, scope_provider=market_scope, owner_user_id=None):
        self.preparation = preparation or MarketFundPreparation()
        self.scope_provider = scope_provider
        self.owner_user_id = owner_user_id
        self.funds = None
        self.news_results = {}
        self.material_results = {}
        self.input_results = {}

    def scope(self):
        if self.funds is None:
            rows = self.scope_provider()
            codes = validate_codes([r["fund_code"] for r in rows])
            if not codes or len(codes) != len(rows):
                raise ValueError("MARKET_WORKFLOW_SCOPE_EMPTY_OR_DUPLICATE")
            self.funds = rows
        return self.funds

    def close(self):
        self.preparation.close()

    def news(self, progress):
        items = []
        funds = self.scope()
        for index, fund in enumerate(funds, 1):
            code = fund["fund_code"]
            try:
                result = self.preparation.sync_news(code, progress=progress)
            except Exception:
                logger.exception("fund_market_workflow.news >>> 公告核验失败, fund_code=%s", code)
                result = {"status": "FAILED", "catalog_complete": False, "candidate_events": []}
            self.news_results[code] = result
            items.append({"fund_code": code, **result})
            progress(index, len(funds), code, "已核验基金公告" if result["status"] == "SUCCEEDED" else "基金公告待补充")
        return self.summary(items, "基金公告核验")

    @staticmethod
    def summary(items, title):
        ready = sum(item["status"] == "SUCCEEDED" for item in items)
        return {
            "status": "SUCCEEDED" if ready == len(items) else "PARTIAL_SUCCESS",
            "created": ready,
            "skipped": len(items) - ready,
            "items": items,
            "message": f"{title}：完成{ready}只，待补充{len(items) - ready}只",
        }

    def materials(self, progress):
        funds = self.scope()
        # 共用持仓维护会话新增的采集入口、来源缓存和排他锁；不另建后台循环。
        checked = self.preparation.sync_holdings(funds, progress=progress)
        checked_codes = {f["fund_code"] for f in checked.get("funds", []) if f["status"] == "SUCCEEDED"}
        items = []
        for index, fund in enumerate(funds, 1):
            code = fund["fund_code"]
            try:
                item = self.preparation.sync_material(fund, self.news_results.get(code), progress=progress)
                # 只接受本次检查或本周期有逐基金证据的复用；失败/互斥不能以页面快照代替。
                if code not in checked_codes:
                    item["gaps"].append("本次持仓披露检查尚未完成")
                    item["status"] = "PARTIAL_SUCCESS"
            except Exception:
                logger.exception("fund_market_workflow.materials >>> 资料采集失败, fund_code=%s", code)
                item = {"fund_code": code, "status": "FAILED", "gaps": ["本次基金资料采集未完成"]}
            self.material_results[code] = item
            items.append(item)
            progress(index, len(funds), code, "基金资料已检查")
        result = self.summary(items, "基金持仓与公司资料更新")
        result["holdings_check"] = checked.get("status")
        result["holdings_reused"] = checked.get("reused", False)
        result["holdings_checked_at"] = checked.get("source_checked_at") or checked.get("finished_at")
        return result

    def inputs(self, progress):
        result = self.preparation.prepare_inputs(self.scope(), progress=progress)
        for item in result["items"]:
            material = self.material_results.get(item["fund_code"])
            if (
                not material
                or material["status"] != "SUCCEEDED"
                or self.news_results.get(item["fund_code"], {}).get("status") != "SUCCEEDED"
            ):
                item.update(status="WAITING_DATA", prediction_ready=False)
                item["gaps"].append("本次资料检查未全部完成")
            self.input_results[item["fund_code"]] = item
        count = sum(i["prediction_ready"] for i in result["items"])
        result.update(
            status="SUCCEEDED" if count == len(result["items"]) else "PARTIAL_SUCCESS",
            created=count,
            skipped=len(result["items"]) - count,
            message=f"检查{len(result['items'])}只基金，{count}只分析资料齐备",
        )
        return result

    def prediction_codes(self):
        return sorted(code for code, item in self.input_results.items() if item["prediction_ready"])

    def ratings(self, progress, *, updater=None):
        """评级自有八维资料与预测资料分别核验；不要求公司财务包代替评级原件。"""
        from sqlalchemy.orm import Session

        from app.db.session import get_engine
        from app.services.fund_rating import update_ratings
        from app.services.fund_rating_inputs import prepare

        frozen = now()
        month_end = frozen.date().replace(day=1) - timedelta(days=1)
        # prepare 会再次检查分类、完整八维原件、估值日及来源有效性。仅有页面持仓不算齐备。
        with Session(get_engine()) as session:
            groups = prepare(session, self.scope(), month_end, frozen)
        ready = {
            m["fund_code"]
            for members in groups.values()
            for m in members
            if m["admitted"]
            and self.news_results.get(m["fund_code"], {}).get("status") == "SUCCEEDED"
            and not self.news_results[m["fund_code"]].get("candidate_events")
            and m["fund_code"] in self.material_results
            and "本次持仓披露检查尚未完成" not in self.material_results[m["fund_code"]].get("gaps", [])
            and self.material_results[m["fund_code"]]["status"] != "FAILED"
        }
        result = (updater or update_ratings)(
            None, as_of=month_end, due_only=True, ready_codes=ready, progress_reporter=progress
        )
        pending_events = [code for code, value in self.news_results.items() if value.get("candidate_events")]
        deferred = result.get("deferred", [])
        return {
            **result,
            "status": "PARTIAL_SUCCESS" if deferred or result["failures"] or pending_events else "SUCCEEDED",
            "created": result["rated"],
            "skipped": result["notRated"],
            "review_required": pending_events,
            "message": (
                f"月度评级：更新{result['rated']}只，{len(result.get('alreadyCurrent', []))}类本月已有结果；"
                f"{len(deferred)}类资料待补充，{len(pending_events)}只基金有事项待复核"
            ),
        }
