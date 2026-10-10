"""002112 的事项和时间边界：从已冻结原文及交易日历计算，不让生成模型猜测。"""

import re
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime

from app.services.direction_1d_protocol import ZONE, calendar, digest


def time_context(data: dict, sessions: list[str] | None = None) -> dict:
    """分别核对目标、比较基准、净值和每只股票/指数；目标日未收盘不算缺数。

    sessions 供离线回放使用冻结日历；线上默认使用项目已验哈希的日历。
    不用工作日减一天，不用最新净值或行情日期替代比较基准日。
    """
    stamp = datetime.fromisoformat(data["as_of"])
    if stamp.tzinfo is None:
        raise ValueError("ANALYSIS_TIMEZONE_REQUIRED")
    stamp = stamp.astimezone(ZONE)
    if sessions is None:
        days, version = calendar()
        sessions = [str(d) for d in days]
    else:
        version = digest(sessions)
    if sessions != sorted(set(sessions)):
        raise ValueError("ANALYSIS_CALENDAR_INVALID")
    today = str(stamp.date())
    index = bisect_left(sessions, today)
    if index < len(sessions) and sessions[index] == today and stamp.hour >= 15:
        index += 1
    if not 0 < index < len(sessions):
        raise ValueError("ANALYSIS_CALENDAR_UNAVAILABLE")
    base, target = sessions[index - 1 : index + 1]
    if (data["window"]["base_nav_date"], data["window"]["target_nav_date"]) != (base, target):
        raise ValueError("ANALYSIS_TIME_WINDOW_MISMATCH")
    latest_nav = max((str(r["nav_date"]) for r in data["nav"]), default=None)
    if latest_nav != data["latest_nav_date"]:
        raise ValueError("ANALYSIS_NAV_DATE_MISMATCH")

    def state(day):
        if day and (day > base or day not in sessions):
            raise ValueError("ANALYSIS_QUOTE_DATE_INVALID")
        return "MISSING" if not day else "READY" if day == base else "STALE"

    stocks = {c["code"]: {"date": (c.get("quote") or {}).get("date")} for c in data["companies"]}
    markets = {code: {"date": q["date"]} for code, q in data["market"].items()}
    # 指数是项目已固定的两个对照，完全未取得也必须留下缺口。
    for code in ("000300.SH", "000905.SH"):
        markets.setdefault(code, {"date": None})
    for value in [*stocks.values(), *markets.values()]:
        value["status"] = state(value["date"])
    for company in data["companies"]:
        if stocks[company["code"]]["status"] == "READY" and (company.get("quote") or {}).get("change_pct") is None:
            stocks[company["code"]]["status"] = "MISSING_VALUE"
    for code, quote in data["market"].items():
        if markets[code]["status"] == "READY" and quote.get("change_pct") is None:
            markets[code]["status"] = "MISSING_VALUE"
    nav_state = state(latest_nav)
    latest_quote = max((q["date"] for q in stocks.values() if q["date"]), default=None)
    missing_stocks = [code for code, q in stocks.items() if q["status"] != "READY"]
    missing_markets = [code for code, q in markets.items() if q["status"] != "READY"]
    ready = nav_state == "READY" and bool(stocks) and not missing_stocks and not missing_markets
    statements = [f"目标日{target}，比较基准日{base}；资料截止北京时间{stamp:%Y-%m-%d %H:%M}。"]
    statements.append(f"已取得净值截至{latest_nav or '暂缺'}，持仓行情最新日期{latest_quote or '暂缺'}。")
    if ready:
        statements.append("比较基准日净值、所列持仓股票及对照指数行情已齐备。")
    else:
        if nav_state != "READY":
            statements.append("比较基准日净值尚未取得。")
        if missing_stocks or not stocks:
            statements.append("部分所列持仓股票的比较基准日行情尚未取得。")
        if missing_markets:
            statements.append("部分对照指数的比较基准日行情尚未取得。")
    statements.append("目标日尚未收盘，目标日收盘行情尚未形成，不属于资料缺失。")
    return {
        "target_date": target,
        "baseline_date": base,
        "cutoff": stamp.isoformat(),
        "calendar_hash": version,
        "latest_nav_date": latest_nav,
        "nav_status": nav_state,
        "latest_stock_quote_date": latest_quote,
        "stocks": stocks,
        "markets": markets,
        "baseline_complete": ready,
        "target_close_status": "NOT_YET_OCCURRED",
        "statements": statements,
    }


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def matter_anchor(event: dict) -> tuple[str, str]:
    """只用原文命名的具体事项识别，不用模型的正负面、类型或标题相似度。

    计划须含年度/期次且明确名称；会议和中介文件可从实际引用中找到同一名称。
    无明确身份时保留独立事件并标记待核实，不以类型相同强行归并。
    多事项引用不强行合并，也不让某个计划的身份吞掉另一笔交易。
    """
    quotes = _compact("\n".join(event["quotes"]))
    title = _compact(event["document"]["title"])
    patterns = [
        r"20\d{2}年(?:度)?(?:第[一二三四五六七八九十\d]+期)?(?:限制性股票|股票期权|股票期权与限制性股票|员工持股)(?:激励)?计划",
        r"20\d{2}年(?:度)?向特定对象发行[AH]?股股票",
        r"20\d{2}年(?:半年度|年度|第一季度|第三季度)(?:利润分配|权益分派)",
    ]
    anchors = {m.group() for pattern in patterns for m in re.finditer(pattern, quotes)}
    if not anchors:
        anchors = {m.group() for pattern in patterns for m in re.finditer(pattern, title)}
    if len(anchors) == 1:
        return "NAMED_MATTER", next(iter(anchors))
    # 报告及摘要共用同一报告期；非报告文件不因提到同一期经营数据而并入报告。
    report = re.fullmatch(r".*?(20\d{2}年(?:半年度|年度|第一季度|第三季度)报告)(?:摘要)?", title)
    if report and not anchors:
        return "NAMED_REPORT", report[1]
    return "UNRESOLVED", event["id"]


def group_events(events: list[dict]) -> list[dict]:
    """完整分区：每个抽取事件恰好出现一次，公司身份为不可跨越的分区键。

    同具体名称合并；完全相同引文只在同公司内去重。无法证明相同的事项单列，
    公司层面再统一形成一份综合依据，防止未识别的同一事项被多份文件反复加权。
    """
    if len({e["id"] for e in events}) != len(events):
        raise ValueError("ANALYSIS_EVENT_ID_DUPLICATE")
    buckets = defaultdict(list)
    exact = {}
    for event in sorted(events, key=lambda e: e["id"]):
        doc = event["document"]
        company = tuple(sorted(set(doc["codes"])))
        # 多公司关联没有证明唯一发行人，不允许凭关联列表猜主体或跨文件合并。
        status, anchor = matter_anchor(event) if len(company) == 1 else ("UNRESOLVED", event["id"])
        quote_key = (company, doc["kind"], tuple(sorted(event["quotes"])))
        key = (company, status, anchor, doc["kind"])
        if len(company) == 1 and quote_key in exact:
            key = exact[quote_key]
        exact[quote_key] = key
        buckets[key].append(event)
    result = []
    for (company, status, anchor, kind), members in sorted(buckets.items()):
        result.append(
            {
                "id": "matter:" + digest([company, status, anchor] + ([] if kind == "ANNOUNCEMENT" else [kind]))[:20],
                "company_codes": list(company),
                "kind": kind,
                "anchor": anchor,
                "identity_status": status,
                "members": members,
                "event_ids": [m["id"] for m in members],
            }
        )
    actual = [i for group in result for i in group["event_ids"]]
    if Counter(actual) != Counter(e["id"] for e in events):
        raise ValueError("ANALYSIS_EVENT_PARTITION_INVALID")
    return result


def event_facts(groups: list[dict], companies: list[dict]) -> dict:
    """每家公司一份综合依据，内部保留各事项身份和全量原文，不把事项混为一件。

    具体事项已确认时保留其多文件分组；无法证明同一事项时仍逐项保留。它们仅
    通过同一个发行人资料入口影响持仓判断，不能因更多文件产生多个方向权重。
    解释可以分别说明不同事项，但程序去重后的依据集合对同一公司资料只保留一次。
    """
    issuers = defaultdict(list)
    for group in groups:
        codes = tuple(group["company_codes"])
        key = codes if len(codes) == 1 else (*codes, group["id"])
        if group["kind"] != "ANNOUNCEMENT":
            key = (group["kind"], *key)
        issuers[key].append(group)
    result = {}
    for key, matters in sorted(issuers.items()):
        company_codes = matters[0]["company_codes"]
        members = [e for group in matters for e in group["members"]]
        sources, lines, seen, receipts = [], [], set(), []
        for event in members:
            doc = event["document"]
            for index, quote in enumerate(event["quotes"]):
                if quote not in seen:
                    lines.append(f"{doc['date']}《{doc['title']}》：{quote}")
                    seen.add(quote)
                receipts.append(
                    {
                        "event_id": event["id"],
                        "document_id": doc["id"],
                        "quote": quote,
                        "source_hash": doc.get("source_hash"),
                        **(event.get("quote_spans") or [{} for _ in event["quotes"]])[index],
                    }
                )
            source = {"title": doc["title"], "url": doc.get("url"), "publishedDate": doc["date"]}
            if source not in sources:
                sources.append(source)
        related = [c for c in companies if c["code"] in company_codes]
        first = members[0]["document"]
        identity = "issuer-evidence:" + digest(key)[:20]
        result[identity] = {
            "category": {"ANNOUNCEMENT": "公告", "NEWS": "新闻", "POLICY": "政策"}[first["kind"]],
            "text": "\n".join(lines),
            "sources": sources,
            "quote_receipts": receipts,
            "source": sources[0],
            "relation": first.get("relation")
            or "；".join(f"{c['name']}披露仓位{c['weight_pct']:.2f}%" for c in related),
            "company_codes": company_codes,
            "event_group": identity,
            "identity_status": (
                "VERIFIED_HOLDING_LINK"
                if first["kind"] != "ANNOUNCEMENT"
                else "VERIFIED_ISSUER"
                if len(company_codes) == 1
                else "UNRESOLVED_ISSUER"
            ),
            "direction_eligible": bool(company_codes) if first["kind"] != "ANNOUNCEMENT" else len(company_codes) == 1,
            "weight_policy": "ONE_ISSUER_ONE_JOINT_ASSESSMENT_NO_DOCUMENT_COUNT_WEIGHT",
            "matters": [{k: v for k, v in group.items() if k != "members"} for group in matters],
            "member_event_ids": [e["id"] for e in members],
            "body_scope": "FULL_TEXT"
            if all(e["document"]["body_scope"] == "FULL_TEXT" for e in members)
            else "VERIFIED_EXCERPT",
            "body_truncated": any(e["document"]["body_truncated"] for e in members),
        }
        if first["kind"] != "ANNOUNCEMENT":
            result[identity]["relation"] = (
                "关联披露持仓："
                + result[identity]["relation"]
                + (
                    "。仅为应用领域关联，不代表原文直接点名这些公司或公司已经取得订单、收入；"
                    "必须区分资料保存日期、原文所述计划期限及当前适用性。"
                )
            )
    return result


def reference_catalog(facts: dict) -> dict:
    """短且确定的原样引用编号，保留完整身份供追溯；绝不猜测补前缀或近似匹配。"""
    result = {}
    for identity, fact in sorted(facts.items()):
        ref = "F" + digest(identity)[:12]
        if ref in result:
            raise ValueError("ANALYSIS_REFERENCE_COLLISION")
        result[ref] = {**fact, "fact_id": identity}
    return result
