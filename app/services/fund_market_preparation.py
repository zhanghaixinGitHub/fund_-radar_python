"""市场资料准备与逐基金准入；与持仓维护共用采集器，不另建定时循环。

公开资料同步可覆盖整个市场，预测范围仍由Java的有效关注关系决定。
这里保存采集成果和具体缺口；缺少综合审阅、境外来源或八维原件时不假装就绪。
"""

import re
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_engine
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read
from app.services.fund_materials import snapshot
from app.services.fund_materials_store import versioned_save
from app.services.fund_rating_inputs import catalog

logger = get_logger(__name__)


def market_scope() -> list[dict]:
    """动态读取与评级一致的真实有效目录，超过既有限额就明确失败而非截断。"""
    with Session(get_engine()) as session:
        return catalog(session)


def validate_codes(codes) -> tuple[str, ...]:
    """代码仅是公开数据范围；不接受路径、来源后缀或任意账号字段。"""
    if len(codes) > 5000 or any(not isinstance(c, str) or not re.fullmatch(r"[0-9]{6}", c) for c in codes):
        raise ValueError("MARKET_PREPARATION_SCOPE_INVALID")
    return tuple(sorted(set(codes)))


class MarketFundPreparation:
    """为任务中心提供按基金执行的资料采集和实际完整性检查。"""

    def __init__(self, *, root: Path | None = None, clock=now):
        self.root = root or Path(get_settings().fund_insight_directory) / "market-workflow"
        self.clock = clock
        self._company_collector = None
        self._news_client = None

    def close(self):
        if self._company_collector is not None:
            self._company_collector.close()
        if self._news_client is not None:
            self._news_client.close()

    def sync_news(self, code: str, *, progress=lambda *args: None) -> dict:
        from app.services.fund_market_news import MarketNewsClient, synchronize

        news_root = self.root / "news"
        self._news_client = self._news_client or MarketNewsClient(news_root)
        value = synchronize(code, progress=progress, client=self._news_client, root=news_root)
        from app.services.fund_market_review import pending_events

        return {
            "status": value["status"],
            "items": len(value["items"]),
            "errors": len(value["errors"]),
            "candidate_events": pending_events(value["candidate_events"]),
            "catalog_complete": value["catalog_complete"],
            "message": "基金公告已核验" if value["status"] == "SUCCEEDED" else "部分基金公告仍待核验",
            "source_hash": digest({key: item["source"]["sha256"] for key, item in value["items"].items()}),
        }

    def sync_holdings(self, funds: list[dict], *, progress=lambda *args: None) -> dict:
        """统一遵守披露期每日、平时每周的周期，不因每日综合任务强制重复抓取。

        未到检查周期时，仅复用本周期已完成、日期一致的逐基金回执，保留原检查
        时间和失败状态；新增基金没有回执时仍待补，不推断成已经检查成功。
        """
        from app.services.market_holdings_maintenance import MarketHoldingsMaintenance, due

        progress(0, len(funds), None, "正在检查基金市场披露报告与持仓")
        maintenance = MarketHoldingsMaintenance()
        result = maintenance.check(funds=funds)
        if result.get("status") == "NOT_DUE":
            path = maintenance.root / "state.json"
            previous = read(path) if path.exists() else {}
            checked_on = previous.get("last_checked_on")
            finished_at = previous.get("finished_at")
            current = self.clock()
            # 运行中的半份回执、未来日期或旧周期的成功都不能解除当前缺口。
            valid = (
                previous.get("status") in {"SUCCEEDED", "PARTIAL_SUCCESS"}
                and checked_on == result.get("last_checked_on")
                and checked_on is not None
                and checked_on <= current.date().isoformat()
                and not due(current.date(), checked_on)
                and finished_at is not None
                and datetime.fromisoformat(finished_at).tzinfo is not None
                and datetime.fromisoformat(finished_at) <= current
            )
            requested = {f["fund_code"] for f in funds}
            result = {
                **result,
                "reused": bool(valid),
                "performed_now": False,
                "source_checked_at": finished_at if valid else None,
                "funds": [f for f in previous.get("funds", []) if f["fund_code"] in requested] if valid else [],
            }
        else:
            result = {**result, "reused": False, "performed_now": result.get("status") != "BUSY"}
        message = "沿用本周期已核验的持仓资料" if result.get("reused") else "基金市场持仓检查已结束"
        progress(len(result.get("funds", [])), len(funds), None, message)
        return result

    def sync_material(self, fund: dict, news: dict | None, *, progress=lambda *args: None) -> dict:
        """留存本基金的报告、来源和已公开公司资料；不把页面持仓等同于八维/综合输入完整。"""
        code = fund["fund_code"]
        validate_codes([code])
        checked = self.clock()
        material = snapshot(code)
        gaps = []
        reports = material.get("reports", []) if material else []
        if not reports:
            gaps.append("缺少本基金已核验的披露持仓")
        latest = reports[0] if reports else None
        if latest and not latest.get("fullDisclosure"):
            gaps.append("最新报告只有部分持仓披露，不能当作完整组合")
        if latest and datetime.fromisoformat(latest["publishedDate"]).date() > checked.date():
            raise ValueError("MARKET_PREPARATION_FUTURE_REPORT")
        if not news or news.get("status") != "SUCCEEDED" or not news.get("catalog_complete"):
            gaps.append("本次公告目录或原文尚未核验完整")
        companies = material.get("companies", []) if material else []
        held = {p["stockCode"] for p in (latest or {}).get("holdings", [])}
        if held:
            from app.services.market_company_materials import MarketCompanyCollector

            self._company_collector = self._company_collector or MarketCompanyCollector(self.root / "company-sources")
            # 采集结果单独留存，不改写另一会话维护的持仓页面、历史报告或公司快照。
            collected = {c["stockCode"]: c for c in companies}
            for company_code in sorted(held):
                collected[company_code] = self._company_collector.company(company_code)
            companies = list(collected.values())
        covered = {c.get("stockCode") for c in companies if c.get("quote") and not c.get("gaps")}
        if held - covered:
            gaps.append(f"{len(held - covered)}家披露持仓公司的行情与经营资料仍需补充")
        fingerprint = digest(
            {
                "fund_code": code,
                "profile_hash": fund.get("profile_hash"),
                "last_nav_date": str(fund.get("last_nav_date")),
                "nav_updated_at": str(fund.get("nav_updated_at")),
                "reports": reports,
                # 取得时间不属于业务内容；相同正文再次核验不应成为虚构的新事项。
                "companies": [
                    {k: v for k, v in c.items() if k != "sources"}
                    | {"source_hashes": sorted(s["sha256"] for s in c.get("sources", []))}
                    for c in companies
                ],
                "news_hash": news.get("source_hash") if news else None,
            }
        )
        result = {
            "fund_code": code,
            "checked_at": checked.isoformat(),
            "status": "PARTIAL_SUCCESS" if gaps else "SUCCEEDED",
            "source_hash": fingerprint,
            "report_date": latest.get("endDate") if latest else None,
            "published_date": latest.get("publishedDate") if latest else None,
            "reports": reports,
            "companies": companies,
            "documents": material.get("documents", []) if material else [],
            "gaps": gaps,
            "candidate_events": news.get("candidate_events", []) if news else [],
            "prediction_ready": False,
            "rating_ready": False,
        }
        path = self.root / "materials" / (code + ".json")
        versioned_save(path, result)
        progress(1, 1, code, "资料已留存，必要缺项已单独记录" if gaps else "基金资料已留存")
        return result

    def prepare_inputs(self, funds: list[dict], *, progress=lambda *args: None) -> dict:
        """冻结实际可用输入；沿用原综合资料校验，不用净值或另一基金来源顶替审阅。"""
        from app.services.fund_information_snapshot import prepare
        from app.services.fund_workflow_materials import input_gaps, registered_materials, unresolved_gaps

        items = []
        for index, fund in enumerate(funds, 1):
            code = fund["fund_code"]
            path = self.root / "materials" / (code + ".json")
            prior = read(path) if path.exists() else None
            gaps = list(prior.get("gaps", [])) if prior else ["本次基金资料准备尚未完成"]
            packet = None
            current = self.clock()
            material_path = None
            try:
                handoff = registered_materials(code, root=self.root, at=current)
                material_path = handoff["materials_path"] if handoff else None
                packet = prepare(current, code, material_path)
                if handoff is not None:
                    # 每次按当前原件重新核验，交接回执不是可手工放行的布尔开关。
                    gaps = unresolved_gaps(prior, packet, handoff["resolutions"])
                    if not handoff.get("reviewed"):
                        gaps.append("本次更新资料尚未完成综合核验与审阅")
                gaps.extend(input_gaps(packet))
                # 旧综合包即便字段完整，也不能跳过这次更新后来源变化的审阅。
                material_path = Path(packet["materials_path"])
                approved = read(material_path)
                if not prior or approved.get("workflow_source_hash") != prior["source_hash"]:
                    gaps.append("本次更新资料尚未完成综合核验与审阅")
                from app.services.fund_market_review import pending_events

                if not prior or pending_events(prior.get("candidate_events", [])):
                    gaps.append("本期新增事项尚需结合原文复核")
            except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
                logger.info(
                    "fund_market_preparation.prepare_inputs >>> 输入待补充, fund_code=%s, reason=%s",
                    code,
                    type(error).__name__,
                )
                gaps.append("本期综合资料或有效估值依据尚未齐备")
                prepare_failure = str(error)
            else:
                prepare_failure = None
            item = {
                "fund_code": code,
                "status": "WAITING_DATA" if gaps else "READY",
                "prediction_ready": not gaps,
                "source_hash": prior.get("source_hash") if prior else None,
                "input_hash": digest(packet) if packet and not gaps else None,
                "checked_at": current.isoformat(),
                "gaps": list(dict.fromkeys(gaps)),
                "materials_path": str(material_path) if material_path else None,
                "prepare_failure": prepare_failure,
            }
            versioned_save(self.root / "inputs" / (code + ".json"), item)
            items.append(item)
            progress(index, len(funds), code, "分析资料已核验" if not gaps else "分析资料仍有缺项")
        ready = sum(i["prediction_ready"] for i in items)
        return {
            # 已留存真实缺项是部分完成，不能把尚无综合判断包装成执行故障或全部成功。
            "status": "SUCCEEDED" if ready == len(items) else "PARTIAL_SUCCESS",
            "created": ready,
            "skipped": len(items) - ready,
            "items": items,
            "message": f"检查{len(items)}只基金，{ready}只分析资料齐备，{len(items) - ready}只仍待补充",
        }

    def readiness(self, code: str) -> dict:
        """只读本次逐基金回执；没有回执时不从目录或旧预测推断本次已经就绪。"""
        validate_codes([code])
        path = self.root / "inputs" / (code + ".json")
        return read(path) if path.exists() else {"fund_code": code, "prediction_ready": False, "gaps": ["资料尚未核验"]}
