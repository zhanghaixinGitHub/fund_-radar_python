"""将原预测中的具体事实转为业务解释；不改预测、不训练、不补入事后消息。

净值及指数数字取不可变预测正文。个股只按原预测登记的原件摘要恢复；公告只
取当时已入模的正文片段。可能影响是条件性的业务解释，不能冒充已证实的价格因果。
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from urllib.parse import urlsplit

from app.services import direction_1d_information as info

VERSION = "PREDICTION_INFORMATION_ZH_V2"


def movement(value: float) -> str:
    return f"{'上涨' if value > 0 else '下跌' if value < 0 else '持平'}{abs(value) * 100:.2f}%"


def original_holdings(body: dict) -> list[dict]:
    """恢复当时同版本的披露权重与个股行情；原件改变或缺失时不使用最新版本兜底。"""
    from app.services.fund_exposure_quotes import reports

    full = body["input"]["information"]
    cutoff = datetime.fromisoformat(body["input"]["feature_as_of"])
    versions = full["source_versions"]
    candidates = []
    try:
        for report in reports():
            if report["raw"]["sha256"] not in versions["reports"]:
                continue
            info.receipt_available(report["raw"], cutoff)
            candidates.append(report)
        report = info.recipe.legacy.choose_report(candidates, cutoff.astimezone(info.ZONE).isoformat())
        if not report or report["report_end"] != full["holding_report_date"]:
            return []
        saved = info.read(info.ROOT / "stock-days" / (full["market_date"] + ".json"))
        if saved["receipt"]["sha256"] != versions["quotes"].get(full["market_date"]):
            return []
        info.receipt_available(saved["receipt"], cutoff)
        rows = []
        for holding in report["holdings"]:
            quote = saved["rows"].get(holding["stock_code"])
            if not quote:
                continue
            weight = float(holding["nav_weight_pct"]) / 100
            change = float(quote["pct_chg"]) / 100
            if not math.isfinite(weight + change) or weight < 0:
                return []
            rows.append(
                {"code": holding["stock_code"], "name": holding["stock_name"], "weight": weight, "change": change}
            )
        # 与原始组合输入逐项对账，防止选中同报告期的另一版持仓。
        if (
            abs(sum(r["weight"] for r in rows) - full["holding_coverage"]) > 1e-10
            or abs(sum(r["weight"] * r["change"] for r in rows) - full["numeric"][7]) > 1e-10
        ):
            return []
        return rows
    except (ValueError, KeyError, TypeError, OSError):
        return []


def nav_driver(body: dict) -> dict:
    """低位只相对于本条预测的历史价格区间，不等于估值便宜或必然反弹。"""
    features = body["input"]["features"]
    values = [float(row["unit_nav"]) for row in body["input"]["values"]]
    position = features[5]
    if not 0 <= position <= 1:
        raise ValueError("INVALID_NAV_POSITION")
    location = "下半部" if position < 0.5 else "上半部" if position > 0.5 else "中部"
    falling = features[0] < 0 and features[1] < 0
    rising = features[0] > 0 and features[1] > 0
    implication = (
        "近期仍在下跌，说明短期走势偏弱；处于区间下半部并不等于估值便宜，也不能据此确认反弹。"
        if falling and position < 0.5
        else "近期仍在下跌，短期走势偏弱；仅凭历史位置不能确认何时止跌。"
        if falling
        else "近期走势向上，可作为关注延续性的线索；但已经上涨不保证目标日继续上涨。"
        if rising
        else "短期和较长区间的涨跌不一致，尚不能把其中一段走势当作明确的延续信号。"
    )
    if max(values) == min(values):
        location_text = "这一回看窗口的净值相同，无法比较高低位置"
    else:
        location_text = f"在这一高低区间内位于{position * 100:.1f}%的位置（0%为最低点），处于{location}"
    return {
        "category": "净值",
        "title": "净值在近期什么位置",
        "assessment": "近期偏弱" if falling else "近期回升" if rising else "走势分歧",
        "observation": (
            f"截至{body['base_nav_date']}，单位净值{values[-1]:.4f}，近60个交易日回看区间为"
            f"{min(values):.4f}—{max(values):.4f}；{location_text}。"
            f"近5个交易日{movement(features[0])}，近20个交易日{movement(features[1])}。"
        ),
        "implication": implication,
    }


def market_driver(body: dict, holdings: list[dict]) -> dict:
    full = body["input"]["information"]
    values = dict(zip(info.recipe.NAMES["C_EVENTS"], full["numeric"], strict=True))
    combined = values["holding_return_1"]
    weak = combined < 0
    selected = sorted(holdings, key=lambda row: abs(row["weight"] * row["change"]), reverse=True)[:3]
    examples = "；".join(f"{r['name']}{movement(r['change'])}（披露仓位{r['weight'] * 100:.2f}%）" for r in selected)
    observation = f"{full['market_date']}，" + (examples + "。" if examples else "")
    observation += (
        f"按披露仓位加权，可取得行情的股票合计{'拖累' if weak else '拉动'}约"
        f"{abs(combined) * 100:.2f}个百分点。沪深300{movement(values['000300.SH_return_1'])}，"
        f"中证500{movement(values['000905.SH_return_1'])}。"
    )
    return {
        "category": "持仓行情",
        "title": "哪些持仓在拉动或拖累",
        "assessment": "存在下行压力" if weak else "已有上行表现",
        "observation": observation,
        "implication": (
            f"若这种{'弱势' if weak else '强势'}延续，可能继续{'拖累' if weak else '拉动'}基金。"
            f"这是上一交易日按{full['holding_report_date']}披露仓位的估算，并非目标日已经发生的涨跌。"
        ),
    }


# 规则只解释事项的作用路径；审批、拟议、授予、落地等阶段仍由标题和原文体现。
# 不把资金用途、回购、激励或人事变动一律判成确定利好或利空。
EVENT_RULES = (
    (
        r"配售|可转换债券|定向增发|向特定对象发行",
        "融资与潜在摊薄",
        "双向影响",
        "融资可能补充资金；若新增股份或后续转股落地，也可能带来每股权益摊薄。实际影响取决于融资条件和资金用途，不能仅凭公告判定目标日涨跌。",
        (r"新增发行[^。；]{1,140}[。；]", r"(?:拟|计划)[^。；]{0,100}(?:发行|募集)[^。；]{0,60}[。；]"),
    ),
    (
        r"受让|收购|重大资产重组",
        "股权交易进展",
        "收益待验证",
        "股权交易可能影响资金占用及未来投资收益；过户或审批进展不等于已经增厚利润。尚需结合后续经营效果，不能直接当作目标日上涨的理由。",
        (r"(?:转让价款|价款)合计为人民币[\d,，.]+元", r"占[^。；]{0,30}(?:总股本|股份总数)的?[\d.]+%"),
    ),
    (
        r"授予.*限制性股票|限制性股票.*授予",
        "股权激励授予",
        "影响待观察",
        "股权激励可能影响员工激励、后续考核和股份支付成本；授予价格不是股价的合理估值，也不构成目标日上涨的直接证据。",
        (r"授予数量[：:][\d,.，]+万?股", r"授予价格[：:][\d,.，]+元/股", r"授予日[：:]\d{4}年\d{1,2}月\d{1,2}日"),
    ),
    (
        r"回购",
        "股份回购进展",
        "影响待观察",
        "实际回购可能影响股份供求，但计划不等于已买入。需要结合公告中的执行数量和进度判断，不能单凭回购标题确认短期上涨。",
        (r"(?:累计|尚未|未实施)[^。；]{0,100}回购[^。；]{0,90}[。；]",),
    ),
    (
        r"减持",
        "股东减持事项",
        "潜在卖压",
        "若减持计划实际执行，可能增加股份供给；是否实施、时间和规模会影响实际卖压，不能把计划提前当成已经卖出。",
        (r"(?:拟|计划|累计|尚未)[^。；]{0,100}减持[^。；]{0,90}[。；]",),
    ),
    (
        r"业绩预告|业绩快报|年度报告|半年度报告",
        "经营业绩变化",
        "需结合预期",
        "经营业绩变化可能影响盈利预期，但报告所述是对应期间的经营情况；还需判断是否超出此前预期，不能直接等同于目标日涨跌。",
        (r"(?:营业收入|净利润)[^。；]{0,150}[。；]",),
    ),
    (
        r"共同投资|对外投资",
        "对外投资进展",
        "收益待验证",
        "出资会占用资金，投资回报通常需要后续经营兑现；认缴金额不能视为新增收入，也不能直接当作短期利好。",
        (r"以自有资金认缴出资人民币[\d,.，]+万元", r"占比约[\d.]+%"),
    ),
    (
        r"核心技术人员",
        "技术人员调整",
        "需看实际变化",
        "技术团队变化可能影响研发连续性，需要区分离职和岗位调整；仍在公司任职不能写成核心人员流失。",
        (r"原核心技术人员[^。；]{1,220}[。；]",),
    ),
)


def event_drivers(body: dict, holdings: list[dict]) -> list[dict]:
    """从当时入模片段选出具体事项，常规会议和核查意见不冒充主要涨跌原因。"""
    full = body["input"]["information"]
    cutoff = datetime.fromisoformat(body["input"]["feature_as_of"])
    segments = full["text"].splitlines()
    company = {r["code"][:6]: r for r in holdings}
    candidates = []
    seen = set()
    for source in sorted(full["sources"], key=lambda row: (row["available_at"], row["id"]), reverse=True):
        if datetime.fromisoformat(source["available_at"]) > cutoff:
            continue
        parsed = urlsplit(source.get("source_url") or "")
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
            continue
        matching = [s for s in segments if s.startswith(source["title"] + " ")]
        if len(matching) != 1:
            continue
        content = re.sub(r"\s+", "", matching[0][len(source["title"]) :])
        if source["kind"] == "ANNOUNCEMENT":
            if re.search(r"核查意见|审核意见|会议决议|股东会决议|召开.*通知", source["title"]):
                continue
            selected = next(
                ((index, rule) for index, rule in enumerate(EVENT_RULES) if re.search(rule[0], source["title"])), None
            )
            if not selected:
                continue
            priority, (_, subject, assessment, implication, patterns) = selected
            quotes = [m.group(0).strip("。；") for pattern in patterns if (m := re.search(pattern, content))]
            if not quotes:
                continue
            if priority == 0:
                # 保留正文中的“拟”阶段，不把融资方案改写成已完成发行；删除重复法律背景。
                quotes = [("拟" if "公司拟" in content and not quotes[0].startswith("拟") else "") + quotes[0]]
            if priority == 1:
                quotes.insert(0, re.sub(r"^关于|的公告$|公告$", "", source["title"]))
            code = re.search(r"证券代码[：:](\d{6})", content)
            held = company.get(code.group(1)) if code else None
            stock_name = re.search(r"证券简称[：:]([^：:]{1,16}?)(?:公告编号|证券代码)", content)
            name = held["name"] if held else stock_name.group(1) if stock_name else "持仓公司"
            key = (code.group(1) if code else name, subject)
            if key in seen:
                continue
            seen.add(key)
            relation = f"该公司披露仓位为{held['weight'] * 100:.2f}%。" if held else ""
            observation = f"{source['published_date']}公告披露：{'；'.join(quotes[:3])}。{relation}"
            title = f"{name}：{subject}"
            category = "公告"
        else:
            # 政策/新闻只引用原预测已保存的正文，不从外部再补一段事后理由。
            priority = -1
            category = "政策" if source["kind"] == "POLICY" else "新闻"
            title, assessment = source["title"], "方向待确认"
            observation = f"{source['published_date']}原文片段：{content[:220]}{'…' if len(content) > 220 else ''}"
            implication = (
                "这条消息与当时披露持仓的业务有关，但仅凭保存的片段尚不能确认受益或受损方向，不据此强行判断目标日涨跌。"
            )
        candidates.append(
            (
                priority,
                source["published_date"],
                {
                    "category": category,
                    "title": title,
                    "assessment": assessment,
                    "observation": observation,
                    "implication": implication,
                    "source": {
                        "title": source["title"],
                        "url": source["source_url"],
                        "publishedDate": source["published_date"],
                    },
                },
            )
        )
    # 每类单独限额，避免公告挤掉少量但相关的新闻与政策。
    selected_rows = []
    for category, limit in (("公告", 3), ("政策", 1), ("新闻", 1)):
        rows = sorted(
            (row for row in candidates if row[2]["category"] == category),
            key=lambda row: (row[0], -int(row[1].replace("-", ""))),
        )
        selected_rows.extend(row[2] for row in rows[:limit])
    return selected_rows


def narrative(body: dict, restored: dict) -> dict:
    """说明已知信息与原预测的关系；证据矛盾时直说，不反向拼出支持原方向的故事。"""
    directions = {row["direction"] for row in restored["branches"]}
    if len(directions) != 1 or body["fund_code"] != "002112":
        raise ValueError("DIRECTION_DISAGREEMENT")
    direction = next(iter(directions))
    full = body["input"]["information"]
    holdings = original_holdings(body)
    drivers = [nav_driver(body), market_driver(body, holdings), *event_drivers(body, holdings)]
    falling = body["input"]["features"][0] < 0 and full["numeric"][7] < 0
    rising = body["input"]["features"][0] > 0 and full["numeric"][7] > 0
    label = "上涨" if direction == "UP" else "下跌"
    summary = f"对{body['target_nav_date']}的判断偏向{label}。"
    if direction == "UP" and falling:
        summary += "但近期净值和持仓行情仍偏弱，以下已知事项尚不足以确认反弹。"
    elif direction == "DOWN" and rising:
        summary += "但近期净值和持仓行情仍在走强，现有事实与预测方向存在分歧。"
    else:
        summary += "下面列出预测时已知、值得关注的具体变化及其可能影响。"
    missing = [name for key, name in (("POLICY", "政策"), ("NEWS", "新闻")) if not full["counts"][key]]
    context = (
        "没有可核验的相关" + "、".join(missing) + "可作为本次判断的理由。"
        if missing
        else "政策和新闻需结合所涉及的公司、实施阶段及实际影响判断。"
    )
    return {
        "styleVersion": VERSION,
        "summary": summary,
        "context": context,
        "supporting": "",
        "opposing": "",
        "drivers": drivers,
        "limitations": [
            "这些是预测时已知的事实及可能影响，并非已经证实的当日涨跌原因；消息也可能已被价格反映。",
            f"持仓以{full['holding_report_date']}披露报告为准，可能不同于当前实际仓位。",
            "现有资料和历史验证仍有局限，不能把预测当作确定结论。",
        ],
    }
