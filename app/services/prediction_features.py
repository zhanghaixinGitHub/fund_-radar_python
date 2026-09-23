"""公共时点特征：当前资料留版本，回放明确区分实收证据与历史可得性假设。"""

from bisect import bisect_right
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories.prediction_store import encode, one, rows, snapshot
from app.services.historical_nav_samples import _build_metrics
from app.services.prediction_contract import PredictionFailure, ValuationCalendar, fingerprint, reinvested_series
from app.services.trading_calendar import load_calendar, load_current_calendar

ZONE = ZoneInfo("Asia/Shanghai")


def public_calendar(fund):
    """当前国内开放式估值按沪深日历作明确实验假设，跨境产品缺少自身政策时单项失败。"""
    if "QDII" in (fund["fund_name"] + str(fund.get("fund_type", ""))).upper():
        raise PredictionFailure(
            "CALENDAR_POLICY_MISSING",
            "CALENDAR",
            "该跨境基金的估值及交易截止日历尚未核验",
            details={"fundCode": fund["fund_code"], "fundName": fund["fund_name"]},
            next_action="补充该基金公布的估值及开放日政策后重试；不使用境内日历猜测",
        )
    past, current = load_calendar(), load_current_calendar()
    return ValuationCalendar(
        "CN_FUND_SSE_ASSUMPTION_V1",
        past.sessions + current.sessions,
        past.definition.coverage_start,
        current.definition.coverage_end,
        fingerprint([past.content_hash, current.content_hash]),
    )


def archive_fact(connection, fund_code, kind, source_key, effective, published, observed, payload):
    """first_observed_at 使用数据库真实接收/修订时刻；不会改成历史公告日期。"""
    digest = fingerprint(payload)
    connection.execute(
        text("""INSERT INTO prediction_fact_version
      (fact_id,fund_code,kind,source_key,payload_hash,effective_at,published_at,first_observed_at,quality,payload)
      VALUES(:id,:fund,:kind,:key,:hash,:effective,:published,:observed,'OBSERVED_AT_TIME',CAST(:payload AS jsonb))
      ON CONFLICT(fund_code,kind,source_key,payload_hash) DO NOTHING"""),
        {
            "id": uuid4(),
            "fund": fund_code,
            "kind": kind,
            "key": source_key,
            "hash": digest,
            "effective": effective,
            "published": published,
            "observed": observed,
            "payload": encode(payload),
        },
    )
    return digest


def at_midnight(value):
    return datetime.combine(value, time(), ZONE) if value else None


def select_fact_version(versions, cutoff, *, replay=False):
    """未来公开、晚修订版本永远不替代已知历史版本；缺证据交给显式重建研究组。"""
    eligible = [
        v
        for v in versions
        if v["first_observed_at"] <= cutoff and (v["published_at"] is None or v["published_at"] <= cutoff)
    ]
    return max(eligible, key=lambda v: v["first_observed_at"], default=None)


def known_nav_version(nav, versions, cutoff, replay):
    """已留存版本优先。历史重建只允许无修订证据的初始版本，不能把已知晚修订倒灌。"""
    selected = select_fact_version(versions, cutoff)
    if selected:
        result = dict(selected["payload"])
        result["nav_date"] = date.fromisoformat(str(result["nav_date"]))
        result["unit_nav"] = Decimal(str(result["unit_nav"]))
        for name in ("updated_at", "source_published_at"):
            result[name] = datetime.fromisoformat(str(result[name])) if result.get(name) else None
        return result
    if replay and len({str(v["payload"].get("unit_nav")) for v in versions}) > 1:
        raise PredictionFailure(
            "LATE_REVISION_UNAVAILABLE",
            "FEATURE_BUILD",
            "该历史时点没有可证实的净值版本，已知晚修订不得倒灌",
            details={"navDate": str(nav["nav_date"]), "knowledgeCutoff": cutoff.isoformat()},
        )
    return nav


def known_dividend_version(event, versions, cutoff, replay):
    """分红修订与净值采用同一时点边界；公告早不代表今天修订的金额当时已知。"""
    selected = select_fact_version(versions, cutoff)
    if selected:
        result = dict(selected["payload"])
        for field, value in result.items():
            if value is not None and field.endswith("_date"):
                result[field] = date.fromisoformat(str(value))
        result["cash_dividend"] = (
            Decimal(str(result["cash_dividend"])) if result.get("cash_dividend") is not None else None
        )
        result["updated_at"] = datetime.fromisoformat(str(result["updated_at"]))
        return result
    economic_fields = ("cash_dividend", "ex_date", "nav_ex_date", "process_status", "implementation_ann_date")
    revisions = {tuple(str(v["payload"].get(key)) for key in economic_fields) for v in versions}
    if replay and len(revisions) > 1:
        raise PredictionFailure(
            "LATE_REVISION_UNAVAILABLE",
            "FEATURE_BUILD",
            "该历史时点缺少可证实的分红版本，已知晚修订不得倒灌",
            details={"eventKey": event["source_event_key"], "knowledgeCutoff": cutoff.isoformat()},
        )
    return event


def feature_values(series, lookback):
    if len(series) < 21:
        raise PredictionFailure(
            "NAV_HISTORY_INSUFFICIENT",
            "FEATURE_BUILD",
            "历史净值不足，无法计算本期预测",
            details={"requiredPoints": 21, "actualPoints": len(series)},
        )
    length = min(lookback, len(series) - 1)
    result = {
        "momentum": float(series[-1] / series[-length - 1] - 1),
        "actualLookbackReturns": length,
        "degradedLookback": length < lookback,
    }
    # 训练模型必须有61点且位置有定义；不足时仍可用基线，不为新算法伪造60日特征。
    metrics = _build_metrics(tuple(series[-61:])) if len(series) >= 61 else None
    if metrics:
        result.update({k: float(v) for k, v in metrics.items()})
    recent_return = float(series[-1] / series[-21] - 1)
    peak = max(series[-61:])
    drawdown = float(series[-1] / peak - 1)
    previous = float(series[-21] / max(series[max(0, len(series) - 81) : -20]) - 1) if len(series) >= 41 else None
    signals = [1 if recent_return > 0 else -1 if recent_return < 0 else 0]
    if previous is not None:
        signals.append(1 if drawdown > previous else -1 if drawdown < previous else 0)
    result.update(
        trendRiskFactor=sum(signals) / len(signals),
        return20=recent_return,
        currentDrawdown=drawdown,
        previousDrawdown=previous,
    )
    return result


def read_fund_data(fund_code, cutoff, *, replay=False, start=None):
    """一次事务批量读取净值、分红和公共资料；不读取账号、仓位或金额。"""
    with get_engine().connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
        fund = one(
            c,
            """SELECT f.*,s.source_id,s.enabled,s.authorized_api_names
            FROM fund_share_class f JOIN source_registry s ON s.source_code=f.source_code
            WHERE f.fund_code=:code""",
            code=fund_code,
        )
        if not fund or not fund["enabled"]:
            raise PredictionFailure("FUND_SOURCE_UNAVAILABLE", "SOURCE", "基金不存在或登记来源未启用")
        calendar = public_calendar(fund)
        navs = rows(
            c,
            """SELECT * FROM nav_daily WHERE fund_code=:code AND source_id=:source
              AND nav_date BETWEEN :start AND :end ORDER BY nav_date LIMIT 3001""",
            code=fund_code,
            source=fund["source_id"],
            start=start or cutoff.date() - timedelta(days=160),
            end=cutoff.date(),
        )
        if len(navs) > 3000:
            raise PredictionFailure("RESEARCH_ROW_BUDGET", "SOURCE", "单基金数据超过批次预算，请缩短时间段")
        dividends = rows(
            c,
            """SELECT * FROM fund_dividend WHERE fund_code=:code AND source_id=:source
              ORDER BY ann_date LIMIT 1001""",
            code=fund_code,
            source=fund["source_id"],
        )
        if len(dividends) > 1000:
            raise PredictionFailure("EVENT_ROW_BUDGET", "SOURCE", "分红记录超过单基金核验预算")
        state = one(c, "SELECT * FROM simulation_market_refresh WHERE fund_code=:code", code=fund_code)
        if not state or not state["dividends_verified_at"]:
            raise PredictionFailure("EVENT_ADJUSTMENT_UNRESOLVED", "SOURCE", "尚无该基金的分红同步核验水位")
        profile = one(
            c,
            "SELECT * FROM fund_profile WHERE fund_code=:code AND source_id=:source",
            code=fund_code,
            source=fund["source_id"],
        )
        managers = rows(
            c,
            """SELECT source_id,source_record_key,manager_name,ann_date,begin_date,end_date,content_hash,updated_at
              FROM fund_manager_assignment WHERE fund_code=:code AND source_id=:source
              ORDER BY begin_date DESC LIMIT 50""",
            code=fund_code,
            source=fund["source_id"],
        )
        facts = []
        # 历史读入也保存实际接收版本；不把今天接收的记录标成过去实收。
        for row in navs:
            archive_fact(
                c,
                fund_code,
                "NAV",
                str(row["nav_date"]),
                at_midnight(row["nav_date"]),
                row["source_published_at"],
                row["updated_at"],
                row,
            )
        for row in dividends:
            archive_fact(
                c,
                fund_code,
                "DIVIDEND",
                row["source_event_key"],
                at_midnight(row["ex_date"] or row["ann_date"]) or cutoff,
                at_midnight(row["ann_date"]),
                row["updated_at"],
                row,
            )
            if row["ann_date"] and at_midnight(row["ann_date"]) <= cutoff and row["updated_at"] <= cutoff:
                facts.append(
                    {
                        "kind": "DIVIDEND_ANNOUNCEMENT",
                        "title": "分红公告",
                        "value": (f"公告 {row['ann_date']}，除息 {row['ex_date']}，每份分红 {row['cash_dividend']} 元"),
                        "source": fund["source_code"],
                        "sourceKey": row["source_event_key"],
                        "publishedAt": str(row["ann_date"]),
                        "firstObservedAt": str(row["updated_at"]),
                        "ruleVersion": "PUBLIC_FACT_NEUTRAL_V1",
                        "factor": 0,
                        "explanation": "分红为收益分配，不直接等同利好；总回报已按事件复权，事件分数不重复加权",
                    }
                )
        if not replay:
            for row in managers:
                archive_fact(
                    c,
                    fund_code,
                    "MANAGER",
                    row["source_record_key"],
                    at_midnight(row["begin_date"]) or cutoff,
                    at_midnight(row["ann_date"]),
                    row["updated_at"],
                    row,
                )
                if (
                    row["updated_at"] <= cutoff
                    and not row["end_date"]
                    and (row["ann_date"] is None or at_midnight(row["ann_date"]) <= cutoff)
                    and (row["begin_date"] is None or at_midnight(row["begin_date"]) <= cutoff)
                ):
                    facts.append(
                        {
                            "kind": "MANAGER",
                            "title": "在任经理",
                            "value": row["manager_name"],
                            "effectiveDate": str(row["begin_date"]),
                            "source": fund["source_code"],
                            "sourceKey": row["source_record_key"],
                            "publishedAt": str(row["ann_date"]) if row["ann_date"] else None,
                            "firstObservedAt": str(row["updated_at"]),
                            "ruleVersion": "PUBLIC_FACT_NEUTRAL_V1",
                            "factor": 0,
                            "explanation": "记录任职事实；经理姓名或变更本身不预设利好利空",
                        }
                    )
            if profile and profile["updated_at"] <= cutoff:
                archive_fact(
                    c, fund_code, "PROFILE", "current", profile["updated_at"], None, profile["updated_at"], profile
                )
                for field, label in (("management_fee", "年管理费率"), ("custodian_fee", "年托管费率")):
                    facts.append(
                        {
                            "kind": "FEE",
                            "title": label,
                            "value": str(profile.get(field)),
                            "source": fund["source_code"],
                            "sourceKey": "fund_basic:" + field,
                            "factor": 0,
                            "explanation": "来源单位为百分比/年，已体现在净值中，不重复扣费",
                        }
                    )
        versions = rows(
            c,
            """SELECT kind,source_key,payload_hash,first_observed_at,published_at,payload
                            FROM prediction_fact_version WHERE fund_code=:code AND kind IN ('NAV','DIVIDEND')
                            AND (kind='DIVIDEND' OR effective_at BETWEEN :start AND :end)
                            ORDER BY first_observed_at""",
            code=fund_code,
            start=at_midnight(start or cutoff.date() - timedelta(days=160)),
            end=cutoff,
        )
        version_map, dividend_versions = {}, {}
        for version in versions:
            target = version_map if version["kind"] == "NAV" else dividend_versions
            target.setdefault(version["source_key"], []).append(version)
        return {
            "fund": fund,
            "calendar": calendar,
            "navs": navs,
            "dividends": dividends,
            "facts": facts,
            "dividendWatermark": state["dividends_verified_at"],
            "navVersions": version_map,
            "dividendVersions": dividend_versions,
        }


def cash_events(data, dates, cutoff, *, replay=False):
    events = {}
    for row in data["dividends"]:
        versions = data.get("dividendVersions", {}).get(row["source_event_key"], [])
        effective_days = [row["nav_ex_date"] or row["ex_date"]] + [
            date.fromisoformat(str(v["payload"].get("nav_ex_date") or v["payload"].get("ex_date")))
            for v in versions
            if v["payload"].get("nav_ex_date") or v["payload"].get("ex_date")
        ]
        if not any(day and dates[0] < day <= dates[-1] for day in effective_days):
            continue
        row = known_dividend_version(
            row,
            versions,
            cutoff,
            replay,
        )
        day = row["nav_ex_date"] or row["ex_date"]
        if day is None or not dates[0] < day <= dates[-1]:
            continue
        available = at_midnight(row["implementation_ann_date"] or row["ann_date"])
        if (
            available is None
            or available > cutoff
            or (not replay and row["updated_at"] > cutoff)
            or row["process_status"] != "实施"
            or (row["nav_ex_date"] and row["ex_date"] != row["nav_ex_date"])
            or day not in dates
            or day in events
            or row["cash_dividend"] is None
        ):
            raise PredictionFailure(
                "EVENT_ADJUSTMENT_UNRESOLVED",
                "FEATURE_BUILD",
                "相关分红的日期、版本或金额无法核验",
                details={"eventKey": row["source_event_key"], "effectiveDate": str(day)},
            )
        events[day] = row["cash_dividend"]
    return events


def build_features(data, cutoff, lookback, *, replay=False, persist=True):
    calendar = data["calendar"]
    known = []
    for nav in data["navs"]:
        if nav["nav_date"] >= cutoff.astimezone(ZONE).date():
            continue
        nav = known_nav_version(nav, data.get("navVersions", {}).get(str(nav["nav_date"]), []), cutoff, replay)
        # LIVE只用真实接收版本；历史研究单列“下一估值日可用且采用现有历史版本”的假设。
        if replay:
            index = bisect_right(calendar.sessions, nav["nav_date"])
            available = (
                at_midnight(calendar.sessions[index]) if index < len(calendar.sessions) else cutoff + timedelta(days=1)
            )
            if nav["source_published_at"]:
                available = max(available, nav["source_published_at"])
        else:
            available = max(nav["updated_at"], nav["source_published_at"] or nav["updated_at"])
        if available <= cutoff and nav["nav_date"] < cutoff.astimezone(ZONE).date():
            known.append(nav)
    if len(known) < 21:
        raise PredictionFailure(
            "NAV_HISTORY_INSUFFICIENT",
            "FEATURE_BUILD",
            "必要历史净值不足",
            details={"requiredPoints": 21, "actualPoints": len(known)},
        )
    latest = known[-1]["nav_date"]
    expected_index = bisect_right(calendar.sessions, cutoff.astimezone(ZONE).date() - timedelta(days=1)) - 1
    if expected_index >= 0 and latest < calendar.sessions[expected_index]:
        raise PredictionFailure(
            "NAV_LATEST_NOT_READY",
            "FEATURE_BUILD",
            "最近应有的净值尚未取得",
            details={"expectedDate": str(calendar.sessions[expected_index]), "actualDate": str(latest)},
        )
    index = bisect_right(calendar.sessions, latest)
    wanted = list(calendar.sessions[max(0, index - 61) : index])
    # 历史不足61点可降阶，但窗口内断档不能用压缩日期或前值补齐。
    wanted = [d for d in wanted if d >= known[0]["nav_date"]]
    values = {r["nav_date"]: r["unit_nav"] for r in known}
    series = reinvested_series(wanted, values, cash_events(data, wanted, cutoff, replay=replay))
    features = feature_values(series, lookback)
    source_hash = fingerprint([{k: str(v) for k, v in r.items()} for r in known if r["nav_date"] in wanted])
    result = {
        "fundCode": data["fund"]["fund_code"],
        "family": str(data["fund"]["fund_master_id"]),
        "fundType": data["fund"]["fund_type"],
        "knowledgeCutoff": cutoff.isoformat(),
        "dataAsOf": str(latest),
        "featureSchemaVersion": "NAV_TOTAL_RETURN_V1",
        "features": features,
        "dataQuality": "ASSUMED_AVAILABILITY" if replay else "OBSERVED_AT_TIME",
        "availabilityAssumptions": ["历史净值按下一估值日可用；原始历史版本无法完整恢复，不属于严格前向证据"]
        if replay
        else [],
        "facts": data["facts"] if not replay else [],
        "sourceHash": source_hash,
        "dividendWatermark": str(data["dividendWatermark"]),
        "calendarAssumption": "境内基金以沪深估值日及15:00为实验起点；未确认特殊开放限制单独披露",
        "missingOptionalFactors": ["新闻正文未接入", "经理职业履历未取得", "完整底层持仓及行业暴露未取得"],
        "series": [str(v) for v in series],
        "dates": list(map(str, wanted)),
    }
    result["featureHash"] = fingerprint(result)
    if persist:
        with get_engine().begin() as connection:
            result["snapshotId"] = str(snapshot(connection, result["fundCode"], cutoff, result))
    return result
