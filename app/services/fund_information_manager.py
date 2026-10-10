"""基金经理的持续背景与任期表现；只消费截止时已保存资料，不猜测离任隐情或能力分数。"""

import re
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation

from sqlalchemy import text

from app.services.direction_1d_protocol import calendar, digest
from app.services.fund_exposure_common import ROOT, read
from app.services.fund_materials_store import source_path

CONTRACT = "MANAGER_CONTEXT_V1"
NAV_SOURCE_URL = "https://tushare.pro/document/2?doc_id=119"


def _date(value: str) -> date | None:
    match = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", value)
    if not match:
        return None
    try:
        return date(*map(int, match.groups()))
    except ValueError:
        return None


def _field(body: str, label: str, following: str) -> str:
    match = re.search(re.escape(label) + r"(.*?)" + re.escape(following), body)
    return match[1].strip("：:") if match else ""


def _names(value: str) -> list[str]:
    names = re.split(r"[、,，/；;]", value)
    return names if names and all(re.fullmatch(r"[\u4e00-\u9fff·]{2,12}", n) for n in names) else []


def parse_notice(document: dict, original: dict, now: datetime) -> dict | None:
    """从公告固定表项读取公开原因、变更日期和经理姓名，并保留完整原文。

    公告送出日期取正文，目录日期仅用于定位。接收时刻晚于截止、基金关联未经核对、
    表项无法识别或正文超过完整事实的展示上限时，不猜填；由调用方留下具体缺口。
    """
    receipt = original.get("receipt", {})
    if not original.get("fund_name_or_code_mentioned") or not receipt.get("received_at"):
        return None
    try:
        received = datetime.fromisoformat(receipt["received_at"])
    except (ValueError, TypeError):
        return None
    if received.tzinfo is None or received > now:
        return None
    body = "\n".join(original.get("pages", []))
    compact = re.sub(r"\s+", "", body)
    published = _date(compact.partition("公告送出日期")[2])
    if not published or published > now.date() or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("sha256", "")):
        return None
    remaining = _names(_field(compact, "共同管理本基金的其他基金经理姓名", "离任基金经理姓名"))
    leaving = _names(_field(compact, "离任基金经理姓名", "2.离任基金经理的相关信息"))
    reason = _field(compact, "离任原因", "离任日期")
    effective = _date(_field(compact, "离任日期", "转任本公司其他工作岗位"))
    valid = bool(remaining and leaving and reason and effective and len(body) <= 2200)
    if "增聘基金经理" in compact and not leaving:
        added = _names(_field(compact, "新任基金经理姓名", "共同管理本基金的其他基金经理姓名"))
        previous = _names(_field(compact, "共同管理本基金的其他基金经理姓名", "2.新任基金经理的相关信息"))
        effective = _date(compact.partition("任职日期")[2])
        remaining = sorted(set(added + previous))
        reason = "增聘基金经理，具体任职信息见公告原文。"
        valid = bool(added and previous and effective and len(body) <= 2200)
    return {
        "id": document["id"], "published_date": str(published),
        "effective_date": str(effective) if effective else None,
        "remaining": remaining, "leaving": leaving, "reason": reason,
        "parsed": valid, "body": body, "source_hash": receipt["sha256"],
        "received_at": received.isoformat(),
        "source": {"title": document["title"], "url": original.get("url") or document["sourceUrl"],
                   "publishedDate": str(published)},
    }


def load_notices(material: dict, now: datetime) -> tuple[list[dict], list[str]]:
    """经理变更不受新闻的两周窗口和一百份公司公告上限影响；最多检查最近六十四份原件。"""
    candidates = sorted(
        (d for d in material["documents"] if d["kind"] == "fund" and "基金经理变更" in d["title"]
         and (d.get("publishedDate") or "9999") <= str(now.date())),
        key=lambda d: (d.get("publishedDate", ""), d["id"]), reverse=True,
    )
    notices, gaps, seen = [], [], set()
    if len(candidates) > 64:
        gaps.append("经理变更仅核对最近六十四份已保存公告，更早记录未逐份核对。")
    for d in candidates[:64]:
        path = source_path(ROOT, "supplement/documents/" + d["id"].removeprefix("fund-") + ".json")
        if not path.exists():
            gaps.append("部分经理变更公告尚未取得原文。")
            continue
        notice = parse_notice(d, read(path), now)
        if notice and notice["source_hash"] not in seen:
            notices.append(notice)
            seen.add(notice["source_hash"])
    return sorted(notices, key=lambda n: (n["published_date"], n["id"])), list(dict.fromkeys(gaps))


def performance(rows: list[dict], start: date, end: date, sessions: tuple[date, ...]) -> dict:
    """按完整交易日复权净值计算，不用累计净值比值或缺值填充。

    起点使用任职/独立管理开始后首个交易日收盘，避免把交接当日全部收益归属个人。
    任期收益为末值/首值-1；最大回撤为区间峰值到后续低点的最大跌幅（正数）。
    二十日正收益窗口占比仅描述历史重叠窗口，不是次日胜率，也不充当独立样本数。
    """
    expected = [d for d in sessions if start <= d <= end]
    empty = {"status": "MISSING", "reason": "任期内可核对的复权净值不足。"}
    if not sessions or start < sessions[0] or end > sessions[-1] or len(expected) < 2:
        return empty
    actual = {r["nav_date"]: r for r in rows if start <= r["nav_date"] <= end}
    if len(actual) != len([r for r in rows if start <= r["nav_date"] <= end]):
        return {**empty, "reason": "任期净值存在重复日期，暂不计算经理表现。"}
    if any(d not in actual for d in expected):
        return {**empty, "reason": "任期交易日净值存在缺口，暂不计算完整区间表现。"}
    try:
        values = [Decimal(str(actual[d]["adjusted_nav"])) for d in expected]
    except (InvalidOperation, KeyError):
        return {**empty, "reason": "任期复权净值缺失，不能用未复权净值替代。"}
    if any(not v.is_finite() or v <= 0 for v in values):
        return {**empty, "reason": "任期复权净值无效，暂不计算经理表现。"}
    peak, drawdown = values[0], Decimal(0)
    for v in values:
        peak = max(peak, v)
        drawdown = max(drawdown, 1 - v / peak)
    rolling = [values[i] / values[i - 20] - 1 for i in range(20, len(values))]
    return {
        "status": "AVAILABLE", "start": str(expected[0]), "end": str(expected[-1]),
        "observations": len(values), "return_pct": round(float(values[-1] / values[0] - 1) * 100, 4),
        "max_drawdown_pct": round(float(drawdown) * 100, 4),
        "positive_20d_pct": round(sum(v > 0 for v in rolling) / len(rolling) * 100, 4) if rolling else None,
        "rolling_windows": len(rolling), "short_sample": len(values) - 1 < 252,
    }


def build_context(assignments: list[dict], notices: list[dict], navs: list[dict], now: datetime,
                  base: date, sessions: tuple[date, ...], benchmark: str, gaps: list[str]) -> dict:
    """当前经理按同一截止时刻识别，公告与任职表冲突时降为待核实，不沿用已离任人员。"""
    active = [a for a in assignments if a.get("begin_date") and a["begin_date"] <= now.date()
              and (not a.get("end_date") or a["end_date"] > now.date())]
    active = sorted(active, key=lambda a: (a["manager_name"], a["begin_date"]))
    current_names = {a["manager_name"] for a in active}
    eligible = [n for n in notices if not n["effective_date"] or n["effective_date"] <= str(now.date())]
    latest = eligible[-1] if eligible else None
    verified = bool(active) and len(current_names) == len(active) and len(active) <= 4
    if latest and (not latest["parsed"] or set(latest["remaining"]) != current_names):
        verified = False
        gaps = [*gaps, "最新经理变更公告与任职记录未完成一致性核对，当前团队和能力评价待核实。"]
    if not latest:
        gaps = [*gaps, "尚未取得可核对的经理变更原文及公开原因。"]
    manager_results = []
    cutoff = min(base, max((r["nav_date"] for r in navs), default=base))
    if verified:
        for a in active:
            begin = a["begin_date"]
            others = [r for r in assignments if r["manager_name"] != a["manager_name"]
                      and r.get("begin_date") and r["begin_date"] <= cutoff
                      and (not r.get("end_date") or r["end_date"] >= begin)]
            solo_start = None
            if len(active) == 1 and all(r.get("end_date") for r in others):
                # 离任日是交接边界，独立管理表现从之后的第一个可核对收盘开始计算。
                solo_start = max([begin, *[r["end_date"] + timedelta(days=1) for r in others]])
            manager_results.append({
                "name": a["manager_name"], "begin_date": str(begin),
                "co_managed": bool(others),
                "tenure": performance(navs, begin, cutoff, sessions),
                "solo_start": str(solo_start) if solo_start else None,
                "solo": performance(navs, solo_start, cutoff, sessions) if solo_start else None,
            })
    return {
        "contract": CONTRACT, "as_of": now.isoformat(), "latest_nav_date": str(cutoff),
        "current_verified": verified, "managers": manager_results, "latest_change": latest,
        "assignments": assignments, "navs": navs, "notices": notices, "gaps": list(dict.fromkeys(gaps)),
        "benchmark": benchmark, "benchmark_comparison": "MISSING", "peer_comparison": "MISSING",
        "evaluation": "任期表现可作观察，但缺少同期基准、同类比较及个人贡献证据，暂不能确认超额管理能力。",
    }


def prepare(connection, source_id, material: dict, now: datetime, base: date, benchmark: str) -> dict:
    """单基金有界读取：任职记录与近五年净值冻结入原输入快照；无远端采集或额外写入。"""
    params = {"source": source_id, "now": now, "today": now.date(), "base": base}
    assignments = [dict(r) for r in connection.execute(text(
        "SELECT manager_name,ann_date,begin_date,end_date,content_hash,updated_at "
        "FROM fund_manager_assignment WHERE fund_code='002112' AND source_id=:source "
        "AND ann_date<=:today AND updated_at<=:now ORDER BY begin_date,manager_name LIMIT 65"
    ), params).mappings()]
    notices, gaps = load_notices(material, now)
    sessions, _ = calendar()
    floor = max(sessions[0], base - timedelta(days=5 * 366))
    params["start"] = floor
    navs = [dict(r) for r in connection.execute(text(
        "SELECT nav_date,adjusted_nav,content_hash,updated_at,ann_date FROM nav_daily "
        "WHERE fund_code='002112' AND source_id=:source AND nav_date BETWEEN :start AND :base "
        "AND updated_at<=:now AND ann_date<=:today ORDER BY nav_date LIMIT 2000"
    ), params).mappings()]
    if len(assignments) > 64:
        assignments = []
        gaps.append("任职记录超出本次核对范围，暂不评价经理能力。")
    return build_context(assignments, notices, navs, now, base, sessions, benchmark, gaps)


def _metric_text(label: str, result: dict) -> str:
    if result["status"] != "AVAILABLE":
        return label + "：" + result["reason"]
    value = (f"{label}（{result['start']}至{result['end']}，{result['observations']}个净值观测）："
             f"复权净值区间收益{result['return_pct']:+.2f}%，最大回撤{result['max_drawdown_pct']:.2f}%。")
    if result["positive_20d_pct"] is not None:
        value += (f"滚动20日正收益窗口占比{result['positive_20d_pct']:.2f}%"
                  f"（{result['rolling_windows']}个重叠窗口，不是预测胜率）。")
    if result["short_sample"]:
        value += "不足252个交易日收益观测，样本较短。"
    return value


def facts(context: dict) -> dict:
    """将经理事实送入同一次综合判断；必须引用，并作为背景或双向因素参与取舍。

    不预置涨跌权重：离任原因、历史收益或高回撤均不能机械转成次日方向。
    原始净值和任职行保留在输入快照，页面只展示指标口径、日期和必要限制。
    """
    common = {"category": "基金经理", "source": None, "direction_eligible": False,
              "required_in_analysis": True, "relation": "本基金的管理连续性与经理任期表现。"}
    change = context["latest_change"]
    result = {}
    if change and change["parsed"]:
        result["manager:change"] = {
            **common, "source": change["source"], "source_hash": change["source_hash"],
            "text": f"经理变更公告原文：\n{change['body']}\n"
                    "此为仍需关注的管理背景，不是今日新发生的事件；离任原因只按公开原文理解。",
        }
    else:
        result["manager:change"] = {**common, "text": "经理变更及公开原因暂未完整核实，不猜测原因或市场影响。"}
    profiles, metrics = [], []
    for manager in context["managers"]:
        profiles.append(f"{manager['name']}自{manager['begin_date']}起参与管理本基金。"
                        + ("任期包含共同管理，不能将全部业绩归于个人。" if manager['co_managed'] else ""))
        metrics.append(_metric_text(manager["name"] + "参与管理期间", manager["tenure"]))
        if manager["solo"]:
            metrics.append(_metric_text(manager["name"] + "当前独立管理阶段", manager["solo"]))
        else:
            metrics.append("未识别出可单独评价的独立管理区间。")
    profile_text = "\n".join(profiles) or "当前基金经理身份与任职期间待核实。"
    metric_text = "\n".join(metrics) or "当前经理任期表现暂不能可靠计算，不能视作能力差或零收益。"
    result["manager:profile"] = {**common, "text": profile_text, "required_in_analysis": False}
    result["manager:performance"] = {
        **common, "text": metric_text + "\n" + context["evaluation"]
        + "\n收益和回撤使用已保存的复权单位净值，起点为任职或交接后的首个交易日收盘。"
        + "\n基金合同业绩比较基准：" + (context["benchmark"] or "暂未核实")
        + "。尚未取得可比基准序列，不以大盘指数替代。历史表现不保证下一交易日涨跌。",
        "metrics": context["managers"],
        # 检查时刻变化不应触发同一资料重复付费生成；实际使用的数据版本另行参与身份摘要。
        "input_hash": digest({
            "assignments": [
                {k: r.get(k) for k in ("manager_name", "ann_date", "begin_date", "end_date", "content_hash")}
                for r in context["assignments"]
            ],
            "navs": [{k: r.get(k) for k in ("nav_date", "adjusted_nav", "content_hash")} for r in context["navs"]],
            "contract": context["contract"],
        }),
    }
    return result
