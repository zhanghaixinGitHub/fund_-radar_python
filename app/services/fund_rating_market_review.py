"""全市场已有资料的部分评价计算；不借旧评分、不采集、不改变八维权重。"""

import re
import unicodedata
from decimal import Decimal, localcontext


def normalized_text(value: str) -> str:
    """只规范字符形态和空白，保留原文的业务数值及比较符。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def annual_fee_score(fees: dict) -> dict:
    """按万元、365天、无涨跌场景重算成本；费用率为比例，金额为人民币元。

    综合运作费率已包含管理、托管等项目，不再叠加。申购费按内扣/外扣
    确定实际投入，运作费及赎回费使用对应本金；不复用旧口径的相加结果。
    """
    with localcontext() as context:
        context.prec = 60
        buy, sell, operating = (Decimal(str(fees[k])) for k in ("subscription", "redemption", "operating_cost"))
        if any(not v.is_finite() or not 0 <= v <= 1 for v in (buy, sell, operating)):
            raise ValueError("MARKET_REVIEW_FEE_RANGE")
        mode = fees.get("subscription_mode")
        if mode not in {"EXTERNAL", "INTERNAL"}:
            raise ValueError("MARKET_REVIEW_FEE_MODE")
        principal = Decimal(10000)
        invested = principal / (1 + buy) if mode == "EXTERNAL" else principal * (1 - buy)
        operation = invested * operating
        redemption = (invested - operation) * sell
        cost = principal - invested + operation + redemption
        score = max(Decimal(0), min(Decimal(100), 100 - cost / 4))
        return {
            "cost_cny": str(cost),
            "score": str(score),
            "invested_cny": str(invested),
            "operating_cny": str(operation),
            "redemption_cny": str(redemption),
        }


def fee_profile(
    text: str, *, code: str, share_class: str, legal_name: str, supporting_text: str = "",
    published_on: str | None = None,
) -> dict:
    """从已核身份的本份额概要读取公开费率；缺明文场景即留缺项。

    同一概要并列多份额时，以紧邻综合费率的份额标签定位，不能取第一项。
    本适配器核明确的一年赎回费及外扣申购费。普通份额读取本代码概要中
    覆盖365天的完整末档；A类按“1年”列示的非零档须另核同产品招募书的
    365天定义和场外分表。锁定期或缺失费档不猜测、不填零。
    """
    from app.services.fund_rating_public_market import a_share_subscription

    value = normalized_text(text)
    combined = value + normalized_text(supporting_text)
    if code not in value or normalized_text(legal_name) not in value:
        raise ValueError("本份额概要身份尚未核准")
    share = share_class.upper().split("-", 1)[0]
    if share not in {"A", "C", "普通"}:
        raise ValueError("当前份额交易费用场景尚未适配")
    if share == '普通' and (
        '下属基金代码'+code not in value[:2000] or '运作方式普通开放式' not in value[:2000]
        or re.search(r'[AC]类(?:基金)?份额', value[:1600])
    ):
        # 保持真实普通份额身份，本代码概要须明确下属代码和普通开放式；
        # 不能把其他份额或封闭REIT默认当作A类。
        raise ValueError('普通份额本代码销售场景尚未核准')
    start = value.find("三、投资本基金涉及的费用")
    if start < 0:
        start = value.find("投资本基金涉及的费用")
    if start < 0:
        raise ValueError("概要费用表尚未识别")
    section = value[start:]
    # 单列概要可能把“年化”表头和下一行同名费用标签分开打印。
    # 只允许该精确标签重复，不跳过A/C标签或任意文字，仍保留并列份额唯一定位。
    pattern = r"基金运作综合费率\(年化\)(?:基金运作综合费率)?(?:持有期间|持有期)?[-—]?(\d+(?:\.\d+)?)%"
    rates = list(re.finditer(pattern, section))
    selected = [
        m for m in rates if re.search(rf"{share}(?:类)?(?:份额)?$", section[max(0, m.start() - 100) : m.start()])
    ]
    if len(rates) == 1:
        selected = rates
    if len(selected) != 1:
        raise ValueError("并列份额的综合运作费率尚未唯一定位")
    operating = Decimal(selected[0][1]) / 100
    selling = section.split("(二)", 1)[0]
    if share == "C":
        currency = re.search(r'C类(人民币|美元)份额', value[:160])
        # 多币种产品标题也会写作C(人民币份额)；以本代码概要“交易币种”
        # 的单一明确币种补核，遇并列币种不选择第一项。
        currency = currency or re.search(r'交易币种(人民币|美元)(?![、，/,和或])', value[:3000])
        qualified = rf'C类{currency[1]}(?:基金)?份额|' if currency else ''
        explicit = re.search(
            rf"(?:{qualified}C类(?:基金)?份额)(?:(?!A类|C类|。).){{0,130}}"
            r"(?:不收取|无)(?:认购[、/]?)?(?:申购费|认/申购费)", combined
        )
        # 部分定报先定义收费方式，再以“称为C类基金份额”收束。
        # 仅接受这一完整定义，不能由概要表未列申购费推断免收。
        explicit = explicit or re.search(
            r"在投资者认购或申购时不收取认购费或申购费[，,]?"
            r"而从本类别基金资产中计提销售服务费的[，,]?称为C类基金份额", combined
        )
        individual = re.search(r"\(C类(?:人民币|美元)?份额\)|\(C份额\)", value[:160])
        foreign_currency = currency and any(v != currency[1] for v in re.findall(r'C类(人民币|美元)份额', selling))
        if not explicit and not (individual and not foreign_currency and re.search(r"不收取申购费|无申购费", selling)):
            raise ValueError("本份额免申购费的明文依据尚未核准")
        buy = Decimal(0)
    else:
        try:
            buy = Decimal(a_share_subscription(section, value))
        except ValueError as error:
            # 本口径直接使用本份额综合运作费，不把销售服务费另加一次；
            # 因而无需为综合费中的项目再设“必须明确为零”的重复门槛。
            tiers = re.findall(r"(?<![<≤=])M<(\d[\d,]*)万(?:元)?(\d+(?:\.\d+)?)%", selling)
            if len(tiers) != 1 or Decimal(tiers[0][0].replace(",", "")) < 1 or "前收费" not in selling:
                raise ValueError("万元普通投资者申购费尚未核准") from error
            buy = Decimal(tiers[0][1]) / 100
    # 只在本份额销售表内核赎回区间；A/C并列表禁止借用另一类零费率。
    labels = list(re.finditer(r"[AC](?:类)?(?:份额)?费用类型", selling))
    if labels:
        own = next((m for m in labels if m[0].startswith(share)), None)
        if own:
            end = next((m.start() for m in labels if m.start() > own.start()), len(selling))
            selling = selling[own.start() : end]
    # 原费用表可将真正的零写作“0”，零值不依赖百分号；必须在表格结束或
    # 明确注释边界结束，不能把0.5%或后续另一份额数字截成0。
    zero_rate = r"0(?:\.0+)?(?:%|(?=注[:：]|$|[，；。]))"
    zero = re.search(r"[NYT](?:≥|>=)(?:7|30|180|365)(?:天|日)[-—]?" + zero_rate, selling)
    zero = zero or re.search(r"(?:7|30|180|365)(?:天|日)≤[NYT][-—]?" + zero_rate, selling)
    redemption = Decimal(0)
    flat = re.findall(
        r'[NYT](?:≥|>=)(7|30|180|365)(?:天|日)(\d+(?:\.\d+)?)%[-－—]?(?=注[:：]|$)', selling
    ) if share == '普通' and '场内' not in selling else []
    if not zero and len(flat) == 1:
        # 普通份额唯一无上界末档覆盖365天，直接取公开费率，不能改成0。
        redemption = Decimal(flat[0][1]) / 100
    elif not zero:
        # 场内/场外并列表不能借费档。只有同一招募书明确“1年指365天”，
        # 且A类场外表在A类场内表前闭合，才读取365天实际落入的非零一年档。
        support = normalized_text(supporting_text)
        table = re.search(r"A类份额的场外赎回费(.{1,600}?)本基金A类份额的场内赎回费", support)
        rates_365 = re.findall(r"1年≤[NT]<2年(\d+(?:\.\d+)?)%", table[1] if table else '')
        summary_rates = re.findall(r"1年≤[NT]<2年(\d+(?:\.\d+)?)%场外", selling)
        if (share != 'A' or '1年指365天' not in support or len(rates_365) != 1
                or len(summary_rates) != 1 or Decimal(rates_365[0]) != Decimal(summary_rates[0])
                or not re.search(r'7(?:日|天)≤[NT]<1年', table[1])
                or not re.search(r'7(?:日|天)≤[NT]<365(?:日|天)', selling)):
            raise ValueError("持有365天的本份额赎回费或可赎回条件尚未核准")
        redemption = Decimal(rates_365[0]) / 100
    stamp = re.search(r"送出日期[:：]?(\d{4})(?:年|-)(\d{1,2})(?:月|-)(\d{1,2})", value)
    if not stamp and not published_on:
        raise ValueError("概要公开日期尚未核准")
    published = (
        "-".join(f"{int(v):0{4 if i == 0 else 2}d}" for i, v in enumerate(stamp.groups()))
        if stamp else published_on
    )
    if published_on:
        from datetime import date

        date.fromisoformat(published_on)
        if published > published_on:
            raise ValueError("概要正文日期晚于已核公开日期")
        # 编制日不冒充公开日；外层只有核过原公告目录后才传入该元数据。
        published = published_on
    return {
        "fund_code": code,
        "published_on": published,
        "fees": {
            "subscription": str(buy),
            "redemption": str(redemption),
            "operating_cost": str(operating),
            "subscription_mode": "EXTERNAL",
        },
    }


def weighted_interval(parts: dict[str, tuple[str, str] | None], weights: dict[str, int]) -> dict:
    """用原内部预算合成有据区间；None是未知域，不是实际零分或100分。"""
    if set(parts) != set(weights) or sum(weights.values()) != 100:
        raise ValueError("MARKET_REVIEW_INNER_WEIGHTS")
    lower = upper = Decimal(0)
    for key, weight in weights.items():
        lo, hi = (Decimal(str(v)) for v in parts[key]) if parts[key] is not None else (Decimal(0), Decimal(100))
        if not 0 <= lo <= hi <= 100:
            raise ValueError("MARKET_REVIEW_INTERVAL_RANGE")
        lower += lo * weight / 100
        upper += hi * weight / 100
    return {"lower": str(lower), "upper": str(upper), "candidate": str(lower) if lower == upper else None}
