"""公开原文的评级字段适配。只识别已核对的报告版式，遇到缺行或歧义即阻断。"""

import re
from datetime import date
from decimal import Decimal

from app.services.fund_rating_metrics import reconcile_equity_assets, standard_cost


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def statement_amount(text: str, label: str) -> Decimal:
    """取本期列，不能取附注编号、上年度列或把未识别值当作零。"""
    lines = [line for line in text.splitlines() if re.match(r"^\s*" + re.escape(label) + r"\s", line)]
    if len(lines) != 1:
        raise ValueError("RATING_STATEMENT_ROW_AMBIGUOUS:" + label)
    values = re.findall(r"(?<![\d.])\d[\d,]*\.\d{2}(?!\d)", lines[0])
    if not values:
        raise ValueError("RATING_STATEMENT_AMOUNT_MISSING:" + label)
    return Decimal(values[0].replace(",", ""))


def equity_balance(pages: list[str], positions: list[dict]) -> dict:
    """提取半年度/年度报表的股票加五项非股票资产；其他投资出现时拒绝套用。"""
    text = "\n".join(pages)
    # 目录中也有同名标题，选择正文的最后一次出现。
    start = text.rfind("6.1 资产负债表")
    end = text.find("6.2 利润表", start)
    if start < 0 or end < 0:
        raise ValueError("RATING_BALANCE_SECTION_MISSING")
    section = text[start:end]
    if "单位：人民币元" not in compact(section):
        raise ValueError("RATING_BALANCE_CURRENCY_UNIT")
    names = ("货币资金", "结算备付金", "存出保证金", "应收清算款", "应收申购款")
    balance = {
        "other_assets": {name: str(statement_amount(section, name)) for name in names},
        "liabilities": str(statement_amount(section, "负债合计")),
        "total_assets": str(statement_amount(section, "资产总计")),
    }
    net = statement_amount(section, "净资产合计")
    reconcile_equity_assets(positions, net, balance)
    # “无达到20%”只能转为严格上界，绝不转成最大持有人为0或20%的事实。
    no_large_holder = "本报告期内无单一投资者持有基金份额比例达到或超过20%的情况" in compact(text)
    restrictions_absent = all(
        phrase in compact(text) for phrase in ("本基金本报告期末未持有因认购新发/增发证券而流通受限的证券",)
    )
    return {
        "net_assets": str(net),
        "balance_sheet": balance,
        "largest_holder": None,
        "largest_holder_upper_exclusive": "0.20" if no_large_holder else None,
        "new_issue_restrictions_absent": restrictions_absent,
    }


def key_information(pages: list[str]) -> dict:
    """适配德邦鑫星价值C官方资料概要；费率采用综合运作口径，保留披露的精度。"""
    text = compact("\n".join(pages))
    if not all(v in text for v in ("德邦鑫星价值", "001412", "002112", "交易币种人民币")):
        raise ValueError("RATING_PRODUCT_DOCUMENT_IDENTITY")
    published = re.search(r"送出日期：(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    current = re.search(r"基金经理([\u4e00-\u9fa5]{2,5})开始担任", text)
    start = re.search(r"开始担任本基金基金经理的日期(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not published or not current or not start:
        raise ValueError("RATING_PRODUCT_DOCUMENT_FIELDS")

    def date_text(match):
        return "-".join(f"{int(v):0{4 if i == 0 else 2}d}" for i, v in enumerate(match.groups()))

    def percent(pattern):
        match = re.search(pattern, text)
        if not match:
            raise ValueError("RATING_FEE_DISCLOSURE_MISSING")
        return Decimal(match[1]) / 100

    fees = {
        "management": percent(r"管理费([\d.]+)%"),
        "custody": percent(r"托管费([\d.]+)%"),
        "sales": percent(r"销售服务费([\d.]+)%"),
        "operating_cost": percent(r"基金运作综合费率（年化）持有期间([\d.]+)%"),
    }
    if "不收取申购费用" not in text or not re.search(r"N[≥≧]30日-?0(?:\.0+)?%?-?(?![\d.])", text):
        raise ValueError("RATING_FEE_SCENARIO_UNVERIFIED")
    fees.update(subscription=Decimal(0), redemption=Decimal(0), subscription_mode="EXTERNAL")
    cost = standard_cost(
        *(fees[k] for k in ("management", "custody", "sales", "subscription", "redemption")),
        mode=fees["subscription_mode"],
        operating_cost=fees["operating_cost"],
    )
    return {
        "product_id": "DBFUND:001412",
        "fund_code": "002112",
        "currency": "CNY",
        "published_on": date_text(published),
        "current_manager": current[1],
        "manager_start": date_text(start),
        "fees": {k: str(v) for k, v in fees.items()},
        "cost_1y": str(cost),
        "operating_cost_basis": "官方资料概要现行费率与最近年报其他运作费用测算；非实际账户费用",
    }


def appointed_teams(pages: list[str], info: dict, start: date, end: date) -> dict:
    """从招募说明书的正式任职记录重建团队，临时代履职责不等于任免。

    返回正式任职口径的真实试算事实；是否采用此口径属于方法验证，不能自动激活规则。
    """
    text = compact("\n".join(pages))
    marker = text.find("本基金历任基金经理：")
    if marker < 0:
        raise ValueError("RATING_MANAGER_DOCUMENT_HISTORY")
    section = text[marker : marker + 1500]
    pattern = (
        r"([\u4e00-\u9fa5]{2,4})(?:先生|女士)，(\d{4})年(\d{1,2})月(\d{1,2})日至"
        r"(\d{4})年(\d{1,2})月(\d{1,2})日管理本基金"
    )
    records = []
    for match in re.finditer(pattern, section):
        groups = match.groups()
        a, b = date(*map(int, groups[1:4])), date(*map(int, groups[4:7]))
        if b < a:
            raise ValueError("RATING_MANAGER_DOCUMENT_DATES")
        records.append({"name": groups[0], "from": str(a), "to": str(b)})
    if not records:
        raise ValueError("RATING_MANAGER_DOCUMENT_HISTORY")
    records.append({"name": info["current_manager"], "from": info["manager_start"], "to": None})
    dates = sorted({r["from"] for r in records} | {r["to"] for r in records if r["to"]})
    teams = []
    for day in dates:
        if day > str(end):
            continue
        # 公告离任日期作为变更生效日；不凭空追加一天，也不将临时履职事件计为任免。
        members = sorted(r["name"] for r in records if r["from"] <= day and (r["to"] is None or r["to"] > day))
        if not members:
            raise ValueError("RATING_MANAGER_DOCUMENT_GAP")
        if not teams or teams[-1]["members"] != members:
            teams.append({"effective_date": day, "members": members})
    if not teams or teams[0]["effective_date"] > str(start):
        raise ValueError("RATING_MANAGER_DOCUMENT_GAP")
    changes = [t for t in teams[1:] if str(start) < t["effective_date"] <= str(end)]
    last_change = date.fromisoformat(teams[-1]["effective_date"])
    months = min(Decimal(36), Decimal((end - last_change).days) / Decimal("30.4375"))
    return {
        "records": records,
        "teams": teams,
        "team_changes": len(changes),
        "team_months": str(months),
        "basis": "正式任免生效日口径；临时代履职不计为正式任职变化",
    }
