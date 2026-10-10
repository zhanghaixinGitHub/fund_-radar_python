"""仅解释事件版原记录；暂不判断也保留当时观察和明确的证据缺口。"""

from app.services import direction_1d_business_explanation as business
from app.services.direction_1d_events import FEATURES, REASONS

MESSAGES = {**REASONS, "VALIDATION_INSUFFICIENT": "现有历史检验尚未证明判断可靠，暂不输出涨跌结论。"}


def narrative(body: dict, restored: dict) -> dict:
    branch = restored["branches"][0]
    full = body["input"]["information"]
    evidence = full["event_evidence"]
    abstained = branch["status"] == "ABSTAINED"
    label = "上涨" if branch["direction"] == "UP" else "下跌"
    reasons = branch["decision"]["reason_codes"]
    summary = (
        f"对{body['target_nav_date']}暂不判断涨跌。" + MESSAGES[reasons[0]]
        if abstained
        else f"对{body['target_nav_date']}的判断偏向{label}，具体事件与已知行情提供了同方向支持。"
    )
    nav = business.nav_driver(body)
    nav.update(
        assessment="背景观察", implication="净值用于了解过去的走势和风险；连续下跌或处于近期低位，不单独作为反弹依据。"
    )
    drivers = [nav, business.market_driver(body, business.original_holdings(body))]
    effects = {f["feature"]: f["contribution"] for f in branch["factors"]}
    items = sorted(evidence["events"], key=lambda e: -abs(effects[e["field"]]))[:5]
    for event in items:
        positive = event["direction"] == "BENEFIT"
        forecast = event["field"] == FEATURES[5]
        drivers.append(
            {
                "category": {"ANNOUNCEMENT": "公告", "POLICY": "政策", "NEWS": "新闻"}[event["kind"]],
                "title": event["title"],
                "assessment": "潜在支持" if positive else "潜在拖累",
                "observation": event["quote"][:550],
                "implication": f"涉及持仓公司{event['code']}，披露仓位{event['weight'] * 100:.2f}%；"
                f"公开于{event['published_date']}。"
                + ("这是业绩预告，尚非最终实现结果。" if forecast else "该事项按已核验的公开阶段参与判断。")
                + "其影响不能直接等同于基金当天涨跌。",
                **(
                    {
                        "source": {
                            "url": event["source_url"],
                            "title": event["title"],
                            "publishedDate": event["published_date"],
                        }
                    }
                    if event.get("source_url")
                    else {}
                ),
            }
        )
    context = " ".join(MESSAGES[reason] for reason in reasons[1:]) if reasons else "以上只引用本次判断时已知的资料。"
    if not context:
        context = "目前已见资料不足以形成可支持涨跌方向的具体事件依据。"
    missing = [name for key, name in (("NEWS", "新闻"), ("POLICY", "政策")) if full["counts"][key] == 0]
    if missing:
        context += " 本次没有可核验的相关" + "、".join(missing) + "，不能视为没有风险。"
    return {
        "styleVersion": "PREDICTION_INFORMATION_ZH_V3",
        "summary": summary,
        "context": context,
        "supporting": "",
        "opposing": "",
        "drivers": drivers,
        "limitations": [
            "暂不判断表示依据不足或存在冲突，不等于预计持平。",
            "持仓来自已披露报告，可能不同于当日真实持仓；行情日期以所列日期为准。",
            "事件对经营的潜在影响不等于下一交易日价格反应，结果仍需后续检验。",
        ],
    }
