"""以本份额分红原件、单位净值和发行人日收益重建独立再投资序列。

供应商复权值有冲突时不覆盖原值、不扩大容差。只有全部窗口内事件由
原件绑定且逐估值日单位净值/现金再投资与发行人曲线相符才形成派生序列。
当前只适配已实际核对的万家006730公告版式，其他产品明确拒收。
"""

import hashlib
import json
import re
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path
from urllib.parse import urlsplit

from app.services.fund_rating_public_market import compact, public_document, publication
from app.services.fund_rating_rules import decimal

CASH_RETURN_SOURCE_POLICY = {
    "scope": "WJ_006730_ORIGINAL_CASH_NOTICES_OWN_UNIT_NAV_ISSUER_WEALTH",
    "event_unit": "公告明确元/10份，按原文A/C代码列绑定本份额后除以10；不能用基金总份额作分母",
    "formula": "W_t=W_(t-1)*(unit_nav_t+cash_per_share_t)/unit_nav_(t-1)",
    "timing": "逐笔公告送出日与公开日不晚于考察日，除息日落在实际三年估值窗口内",
    "coverage": "窗口内事件清单逐笔核原件；逐日单位净值和再投资收益与发行人曲线共同核对，不用成功标志证明完整",
    "rounding": "再投资与发行人归一化曲线差异<=0.000001/min(发行人累计收益因子)，不放宽既有严格上界",
    "preservation": "供应商复权值原样保留；新派生序列及来源哈希单独登记，不改写供应商原始字段或正式旧包",
}


def wj_cash_notice(pages: list[str], *, fund_code: str, as_of: date) -> dict:
    """读取实际公告中的C类每十份分红、除息日及再投资采用净值日。

    A/C顺序须由基金简称与交易代码两行共同证明；仅接纳已核实的三位小数
    公告格式。数字位数或单位变化须新增真实适配，不从预期分红金额反推。
    """
    text = compact("\n".join(pages))
    if fund_code != "006730" or "万家中证500指数增强" not in text[:1000] or "分红公告" not in text[:150]:
        raise ValueError("QUALITY_CASH_NOTICE_PRODUCT_FORMAT_UNSUPPORTED")
    binding = (
        "下属分级基金的基金简称万家中证500指数增强发起式A"
        "万家中证500指数增强发起式C下属分级基金的交易代码006729006730"
    )
    if binding not in text:
        raise ValueError("QUALITY_CASH_NOTICE_SHARE_COLUMNS_UNBOUND")
    amount = re.search(
        r"本次下属分级基金分红方案[（(]单位[:：]元/10份基金份额[）)]"
        r"(\d+\.\d{3})(\d+\.\d{3})(?=注[:：])", text
    )
    ex = re.search(r"除息日(20\d{2})年(\d{1,2})月(\d{1,2})日", text)
    register = re.search(r"权益登记日(20\d{2})年(\d{1,2})月(\d{1,2})日", text)
    reinvest = re.search(r"将以(20\d{2})年(\d{1,2})月(\d{1,2})日除息后的基金份额净值为基准计算", text)
    if not amount or not ex or not register or not reinvest:
        raise ValueError("QUALITY_CASH_NOTICE_AMOUNT_UNIT_OR_REINVESTMENT_UNPARSED")
    day = date(*map(int, ex.groups()))
    sent = publication(pages)
    if sent > str(as_of) or day > as_of or sent > str(day):
        raise ValueError("QUALITY_CASH_NOTICE_PUBLICATION_OR_EVENT_AFTER_CUTOFF")
    if register.groups() != ex.groups() or reinvest.groups() != ex.groups():
        raise ValueError("QUALITY_CASH_NOTICE_EX_DATE_REINVESTMENT_CONFLICT")
    cash = decimal(amount[2]) / 10
    if cash <= 0 or "本次分红免收分红手续费和红利再投资费用" not in text:
        raise ValueError("QUALITY_CASH_NOTICE_POSITIVE_NET_CASH_OR_FEE_UNVERIFIED")
    return {
        "fund_code": fund_code, "ex_date": str(day), "cash_per_share_cny": str(cash),
        "cash_per_ten_shares_cny": amount[2], "published_on": sent,
        "reinvestment_nav_date": str(day), "reinvestment_fee": "0",
        "share_binding_excerpt": binding, "amount_excerpt": amount[0], "reinvestment_excerpt": reinvest[0],
    }


def reconcile_cash_history(
    curve: dict, own_snapshot: dict, benchmark_raw: bytes, benchmark_ref: dict,
    cash_snapshot: dict, notice_refs: list[dict], root: Path, *, fund_code: str, as_of: date,
) -> dict:
    """实际现金事件全部核原件，并逐日复算，不由供应商复权值选择分红参数。

    curve及own_snapshot由数据库净值适配器验证授权和完整估值日历。这里再核
    份额、原公告单位、每个事件和每日单位净值，保存原供应商序列及冲突幅度。
    """
    address = urlsplit(benchmark_ref["url"])
    if (
        fund_code != "006730" or own_snapshot.get("fund_code") != fund_code
        or cash_snapshot.get("fund_code") != fund_code
        or address.scheme != "https" or address.hostname != "www.wjasset.com"
        or address.path != "/common-web/ets40/yieldListQueryAPI"
        or benchmark_ref.get("params", {}).get("fundCode") != fund_code
        or benchmark_ref.get("params", {}).get("incomeType") != "8"
        or benchmark_ref.get("format") != "ISSUER_DAILY_BENCHMARK_WJ"
    ):
        raise ValueError("QUALITY_CASH_RECONCILIATION_PRODUCT_OR_SOURCE_UNSUPPORTED")
    raw = json.loads(benchmark_raw, parse_float=Decimal)
    if raw.get("code") != "ETS-5BP0000" or not isinstance(raw.get("results"), list):
        raise ValueError("QUALITY_CASH_RECONCILIATION_ISSUER_SCHEMA")
    records = raw["results"]
    by_day = {str(date.fromisoformat(row["tradeDate"])): row for row in records}
    if not records or len(records) > 10000 or len(by_day) != len(records):
        raise ValueError("QUALITY_CASH_RECONCILIATION_ISSUER_DATES")
    days = curve["dates"]
    expected = {}
    for row in cash_snapshot.get("dividend_records", []):
        day = row["ex_date"]
        if not days[0] < day <= days[-1]:
            continue
        if day in expected or day not in days or row.get("process_status") != "实施" or not row.get("content_hash"):
            raise ValueError("QUALITY_CASH_EVENT_DUPLICATE_CALENDAR_OR_LINEAGE")
        expected[day] = decimal(row["cash_dividend"])
    if not expected:
        raise ValueError("QUALITY_CASH_EVENTS_NOT_REGISTERED_FOR_THIS_ADAPTER")
    events = {}
    for ref in notice_refs:
        if not ref.get("published_on") or ref["published_on"] > str(as_of):
            continue
        origin = urlsplit(ref["url"])
        if (origin.scheme != "https" or origin.hostname != "www.wjasset.com"
                or not origin.path.startswith("/upload/pdf/")):
            raise ValueError("QUALITY_CASH_NOTICE_ISSUER_HOST_REQUIRED")
        path = (root / ref["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or path.stat().st_size > 8_000_000:
            raise ValueError("QUALITY_CASH_NOTICE_PATH_OR_SIZE")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != ref["sha256"] or not body.startswith(b"%PDF"):
            raise ValueError("QUALITY_CASH_NOTICE_ORIGINAL_HASH_OR_PDF")
        pages, _ = public_document(body, ref, fund_code, "分红公告")
        parsed = wj_cash_notice(pages, fund_code=fund_code, as_of=as_of)
        if parsed["published_on"] != ref["published_on"]:
            raise ValueError("QUALITY_CASH_NOTICE_REGISTERED_PUBLICATION_CONFLICT")
        day = parsed["ex_date"]
        if not days[0] < day <= days[-1]:
            continue
        if day in events and events[day]["cash_per_share_cny"] != parsed["cash_per_share_cny"]:
            raise ValueError("QUALITY_CASH_NOTICE_EVENT_AMOUNT_CONFLICT")
        events[day] = {**parsed, "source_hashes": [ref["sha256"]], "source_url": ref["url"]}
    if set(events) != set(expected):
        raise ValueError("QUALITY_CASH_EVENT_ORIGINALS_MISSING:" + ",".join(sorted(set(expected) - set(events))))
    if any(decimal(events[day]["cash_per_share_cny"]) != amount for day, amount in expected.items()):
        raise ValueError("QUALITY_CASH_SNAPSHOT_AND_ORIGINAL_AMOUNT_CONFLICT")
    rows = {row["nav_date"]: row for row in own_snapshot["history"]["rows"]}
    if len(rows) != len(own_snapshot["history"]["rows"]):
        raise ValueError("QUALITY_CASH_OWN_UNIT_NAV_DUPLICATE")
    with localcontext() as context:
        context.prec = 40
        wealth, units = [], []
        for day in days:
            if day not in by_day or day not in rows or by_day[day].get("unitNv") in (None, ""):
                raise ValueError("QUALITY_CASH_DAILY_UNIT_OR_WEALTH_MISSING:" + day)
            value = decimal(by_day[day]["unitNv"])
            if value != decimal(rows[day]["unit_nav"]) or value <= 0:
                raise ValueError("QUALITY_CASH_DAILY_UNIT_NAV_CONFLICT:" + day)
            units.append(value)
            factor = 1 + decimal(by_day[day]["returnRatio"]) / 100
            if factor <= 0:
                raise ValueError("QUALITY_CASH_ISSUER_WEALTH_NONPOSITIVE")
            wealth.append(factor)
        derived = [Decimal(1)]
        for i, day in enumerate(days[1:], 1):
            # 事件是否缺失另由完整日收益恒等式检验；这里的0是无登记现金事件日
            # 的再投资递推定义，不能用它宣称缺少公告的事件已经核验。
            cash = decimal(events[day]["cash_per_share_cny"]) if day in events else Decimal(0)
            derived.append(derived[-1] * (units[i] + cash) / units[i - 1])
        bound = Decimal(".000001") / min(wealth)
        discrepancy = max(abs(value / wealth[0] - expected) for value, expected in zip(wealth, derived, strict=True))
        if discrepancy > bound:
            raise ValueError("QUALITY_CASH_DAILY_REINVESTMENT_IDENTITY_CONFLICT")
        original = list(map(decimal, curve["adjusted_nav"]))
        conflict = max(abs(a / original[0] - b / wealth[0]) for a, b in zip(original, wealth, strict=True))
        return {
            "curve": {**curve, "adjusted_nav": list(map(str, wealth))},
            "provider_adjusted_nav": list(curve["adjusted_nav"]),
            "derived_source": "ISSUER_WEALTH_RECONCILED_WITH_OWN_UNIT_NAV_AND_ORIGINAL_CASH_NOTICES",
            "events": events, "unit_days_checked": len(days),
            "maximum_reinvestment_difference": str(discrepancy), "reinvestment_rounding_bound": str(bound),
            "maximum_original_provider_difference": str(conflict),
            "event_source_hashes": sorted({sha for value in events.values() for sha in value["source_hashes"]}),
            "scope": "SEPARATE_DERIVED_HISTORY_ORIGINAL_PROVIDER_BYTES_RETAINED_NOT_SOURCE_CORRECTION",
        }
