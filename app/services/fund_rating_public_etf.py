"""ETF 联接公开资料适配：穿透、合同跟踪目标及八维完整性。

首案 016008（招商消费电子联接 C）使用管理人官方接口回执，即 CMFCHINA_OFFICIAL 来源
剖面；其后基金一律走 PUBLIC_REPRINT_PROVIDER 剖面：法定披露公开转载、已授权净值来源
及管理人官方逐日基准，基金代码、目标 ETF、标的指数和业绩基准表述都从证据包 scope 与
原件读取并逐项核验，不再依赖模块级写死的单一基金。资料准备不采集、不出级；导入重读
原件，手动任务才发布。原文未披露的最大个人持有人比例保持未知。
"""

import hashlib
import io
import json
import math
import re
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from pypdf import PdfReader

from app.core.config import get_settings
from app.services.fund_rating_disclosures import compact
from app.services.fund_rating_metrics import standard_cost, total_return
from app.services.fund_rating_public_policy import ANCHORS, ETF_BASIS, TERMS
from app.services.fund_rating_rules import DIMENSIONS, decimal

FUND_CODE = "016008"
TARGET_CODE = "159779"
INDEX_NAME = "中证消费电子主题指数"
BENCHMARK_NAME = INDEX_NAME + "收益率×95%+中国人民银行人民币活期存款利率（税后）×5%"
SCOPE = {
    "basis": ETF_BASIS,
    "asset": "DOMESTIC_EQUITY_ETF_FEEDER",
    "currency": "CNY",
    "target_fund": TARGET_CODE,
    "target_index": INDEX_NAME,
    "share_class": "C",
}
# 首案 016008 的来源剖面：管理人官方接口回执。新增基金使用公开转载剖面。
OFFICIAL_PROFILE = "CMFCHINA_OFFICIAL_V1"
REPRINT_PROFILE = "PUBLIC_REPRINT_PROVIDER_V1"


def etf_scope(scope: dict) -> dict:
    """适用范围逐键校验；目标 ETF 和标的指数只接受可与原件核对的公开标识。"""
    required = {"basis", "asset", "currency", "target_fund", "target_index", "share_class"}
    if (
        not isinstance(scope, dict)
        or set(scope) != required
        or scope["basis"] != ETF_BASIS
        or scope["asset"] != "DOMESTIC_EQUITY_ETF_FEEDER"
        or scope["currency"] != "CNY"
        or scope["share_class"] != "C"
        or not re.fullmatch(r"\d{6}", str(scope["target_fund"]))
        or not isinstance(scope["target_index"], str)
        or not compact(scope["target_index"])
    ):
        raise ValueError("RATING_ETF_SCOPE")
    return deepcopy(scope)


def tracking_targets(bundle: dict) -> dict:
    """合同跟踪目标来自证据包并经概要原文核验；首案包未单列该字段，其 4%/0.35% 已核验。"""
    value = bundle.get("tracking_targets") or {"tracking_error": "0.04", "tracking_bias": "0.0035"}
    if any(decimal(value[k]) <= 0 for k in ("tracking_error", "tracking_bias")):
        raise ValueError("RATING_ETF_TRACKING_TARGETS")
    return value


def etf_policy() -> dict:
    """首次真实总分前固定的指数权重和公开指标；不会迁就个案结果改变参数。"""
    anchors = deepcopy(ANCHORS)
    tracking_points = [(0, 100), (0.5, 90), (1, 70), (1.5, 50), (2, 20), (3, 0)]
    anchors.update(tracking_error_ratio=tracking_points, tracking_bias_ratio=tracking_points)
    terms = deepcopy(TERMS)
    terms["management"] = [
        ("tracking_error_ratio", 50),
        ("tracking_bias_ratio", 30),
        ("strategy_continuity", 10),
        ("governance_state", 10),
    ]
    # 本类型事前采用公开资产结构；不把未公开持有人比例补零，也不运行时调权重。
    terms["liquidity"] = [("restricted_ratio", 40), ("cash_ratio", 30), ("net_assets", 30)]
    return {
        "basis": ETF_BASIS,
        "family": "INDEX",
        "revision": "public-etf-link-2026-10-14-v1",
        "public_evidence_version": 1,
        "weights": dict(zip(DIMENSIONS, (10, 15, 10, 10, 25, 10, 10, 10), strict=True)),
        "terms": terms,
        "anchors": {k: [[str(x), y] for x, y in v] for k, v in anchors.items()},
        "minimum_products": 1,
        "comparison": "ABSOLUTE_NO_PEER_RANK",
        "local_only": True,
        "holdings_age_days": 210,
        "verification_age_days": 31,
        "history_months": 36,
        "status": "REQUIRES_REAL_VALIDATION",
        "rounding": "ROUND_HALF_EVEN",
        "scope": deepcopy(SCOPE),
        "tracking_benchmark": BENCHMARK_NAME,
        "tracking_error_target": "0.04",
        "daily_bias_target": "0.0035",
        "benchmark_public_precision": "0.00005",
    }


def _section(text: str, start: str, end: str) -> str:
    """正文取最后一个标题，排除目录；缺少边界或顺序错误就停止解析。"""
    a = text.rfind(start)
    b = text.find(end, a + len(start))
    if a < 0 or b < a:
        raise ValueError("RATING_ETF_SECTION_MISSING:" + start)
    return text[a:b]


def _amount(section: str, label: str) -> str:
    """资产负债表取本期第一列；横杠仅在原文明确出现时解析为零。"""
    lines = [x.strip() for x in section.splitlines() if re.match(r"^\s*" + re.escape(label) + r"\s", x)]
    if len(lines) != 1:
        raise ValueError("RATING_ETF_BALANCE_ROW:" + label)
    rest = lines[0][len(label) :].strip()
    rest = re.sub(r"^6\.4\.\d+(?:\.\d+)*\s+", "", rest)
    m = re.match(r"(-|[\d,]+\.\d{2})(?:\s|$)", rest)
    if not m:
        raise ValueError("RATING_ETF_BALANCE_VALUE:" + label)
    return "0" if m[1] == "-" else m[1].replace(",", "")


def balance(pages: list[str]) -> dict:
    """完整资产加总和净资产对账，基金/债券投资与现金分别保留，不算作现金。"""
    section = _section("\n".join(pages), "6.1 资产负债表", "6.2 利润表")
    if "单位：人民币元" not in compact(section):
        raise ValueError("RATING_ETF_BALANCE_UNIT")
    names = (
        "货币资金",
        "结算备付金",
        "存出保证金",
        "其中：股票投资",
        "基金投资",
        "债券投资",
        "资产支持证券投资",
        "贵金属投资",
        "其他投资",
        "衍生金融资产",
        "买入返售金融资产",
        "债权投资",
        "其他债权投资",
        "其他权益工具投资",
        "应收清算款",
        "应收股利",
        "应收申购款",
        "递延所得税资产",
        "其他资产",
    )
    # “其中”重复的债券子行不属于资产一级项目，避免重复相加。
    cleaned = re.sub(
        r"(?m)^(债权投资[^\n]*\n)其中：债券投资[^\n]*\n资产支持证券投资[^\n]*\n其他投资[^\n]*\n",
        r"\1",
        section,
    )
    assets = {name: _amount(cleaned, name) for name in names}
    result = {
        "assets": assets,
        "total_assets": _amount(section, "资产总计"),
        "liabilities": _amount(section, "负债合计"),
        "net_assets": _amount(section, "净资产合计"),
    }
    if min(map(decimal, assets.values())) < 0 or decimal(result["net_assets"]) <= 0:
        raise ValueError("RATING_ETF_BALANCE_RANGE")
    if abs(sum(map(decimal, assets.values())) - decimal(result["total_assets"])) > Decimal(".01") or abs(
        decimal(result["total_assets"]) - decimal(result["liabilities"]) - decimal(result["net_assets"])
    ) > Decimal(".01"):
        raise ValueError("RATING_ETF_BALANCE_MISMATCH")
    return result


def target_holdings(pages: list[str]) -> dict:
    """解析全部指数和积极股票，而非前十名；报告行业汇总独立核对，拒绝编造逐股行业。"""
    text = "\n".join(pages)
    body = _section(text, "7.3.1 期末指数投资", "7.4 报告期内股票投资组合")
    parts = body.split("7.3.2 期末积极投资", 1)
    if len(parts) != 2:
        raise ValueError("RATING_ETF_ALL_STOCK_SECTIONS")
    pattern = r"^\s*(\d+)\s+(\d{6})\s+(.+?)\s+([\d,]+)\s+([\d,]+\.\d{2})\s+([\d.]+)\s*$"
    positions = []
    subtotals = {}
    for kind, section in zip(("INDEX", "ACTIVE"), parts, strict=True):
        rows = list(re.finditer(pattern, section, re.M))
        if not rows or [int(m[1]) for m in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("RATING_ETF_STOCK_ROWS_INCOMPLETE")
        for m in rows:
            positions.append(
                {
                    "code": m[2],
                    "issuer": m[2],
                    "name": compact(m[3]),
                    "shares": m[4].replace(",", ""),
                    "amount": m[5].replace(",", ""),
                    "kind": kind,
                }
            )
        subtotals[kind] = sum(decimal(m[5].replace(",", "")) for m in rows)
    if len({p["code"] for p in positions}) != len(positions):
        raise ValueError("RATING_ETF_DUPLICATE_STOCK")
    report_balance = balance(pages)
    if abs(sum(subtotals.values()) - decimal(report_balance["assets"]["其中：股票投资"])) > Decimal(".01"):
        raise ValueError("RATING_ETF_STOCK_SUM_MISMATCH")
    industries = []
    for kind, start, end in (
        ("INDEX", "7.2.1 期末指数投资", "7.2.2 期末积极投资"),
        ("ACTIVE", "7.2.2 期末积极投资", "7.2.3 报告期末按行业"),
    ):
        s = compact(
            re.sub(r"招商中证消费电子主题.*?中期报告\s*第\s*\d+\s*页\s*共\s*\d+\s*页", "", _section(text, start, end))
        )
        rows = re.findall(r"([A-S])([\u4e00-\u9fff、]+?)([\d,]+\.\d{2}|-)([\d.]+|-)(?=[A-S]|合计)", s)
        if len(rows) != 19 or {r[0] for r in rows} != set("ABCDEFGHIJKLMNOPQRS"):
            raise ValueError("RATING_ETF_INDUSTRY_ROWS_INCOMPLETE")
        total = Decimal(0)
        for code, name, amount, _ in rows:
            value = Decimal(0) if amount == "-" else decimal(amount.replace(",", ""))
            total += value
            industries.append({"code": code, "name": name, "amount": str(value), "kind": kind})
        if abs(total - subtotals[kind]) > Decimal(".01"):
            raise ValueError("RATING_ETF_INDUSTRY_SUM_MISMATCH")
    restricted = _section(text, "6.4.12.1.1 受限证券类别：股票", "6.4.12.1.2 受限证券类别：债券")
    restricted = re.sub(r"招商中证消费电子主题.*?中期报告\s*第\s*\d+\s*页\s*共\s*\d+\s*页", "", restricted)
    blocks = re.split(r"(?m)(?=^[ \t]*\d{6}[ \t]+)", restricted)[1:]
    known = {p["code"]: p for p in positions}
    restricted_amounts, restricted_shares = {}, {}
    for block in blocks:
        code = re.match(r"\s*(\d{6})", block)[1]
        matches = re.findall(r"([\d,]+)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*-(?:\s|$)", block)
        if code not in known or len(matches) != 1:
            raise ValueError("RATING_ETF_RESTRICTED_ROW")
        quantity, _, amount = matches[0]
        restricted_amounts[code] = restricted_amounts.get(code, Decimal(0)) + decimal(amount.replace(",", ""))
        restricted_shares[code] = restricted_shares.get(code, Decimal(0)) + decimal(quantity.replace(",", ""))
    if not blocks:
        raise ValueError("RATING_ETF_RESTRICTED_MISSING")
    for p in positions:
        value = restricted_amounts.get(p["code"], Decimal(0))
        if value > decimal(p["amount"]) or restricted_shares.get(p["code"], 0) > decimal(p["shares"]):
            raise ValueError("RATING_ETF_RESTRICTED_RANGE")
        p["restricted_amount"] = str(value)
    required = ("本基金本报告期末未持有暂时停牌等流通受限股票", "本基金本报告期末无参与转融通证券出借业务的证券")
    if not all(q in compact(text) for q in required):
        raise ValueError("RATING_ETF_OTHER_RESTRICTIONS_UNVERIFIED")
    return {
        "positions": positions,
        "industries": industries,
        "balance": report_balance,
        "restricted_stock_amount": str(sum(restricted_amounts.values())),
    }


def key_information(pages: list[str]) -> dict:
    """对应 C 份额的身份、费用豁免、标准一年场景及跟踪目标均来自资料概要。"""
    text = compact("\n".join(pages))
    required = (
        "招商中证消费电子",
        "下属基金代码016008",
        "基金代码016007",
        "基金类型股票型交易币种人民币",
        "申购本基金C类基金份额不收取申购费用",
        "7天≤N0%",
        "年化跟踪误差控制在4%以内",
    )
    if not all(x in text for x in required):
        raise ValueError("RATING_ETF_KEY_DOCUMENT_IDENTITY_OR_SCENARIO")

    def percentage(pattern):
        m = re.search(pattern, text)
        if not m:
            raise ValueError("RATING_ETF_FEE_MISSING")
        return str(decimal(m[1]) / 100)

    fees = {
        "management": percentage(r"管理费年费率：([\d.]+)%"),
        "custody": percentage(r"托管费年费率：([\d.]+)%"),
        "sales": percentage(r"销售服务费年费率：([\d.]+)%"),
        "operating_cost": percentage(r"基金运作综合费率（年化）-([\d.]+)%"),
        "subscription": "0",
        "redemption": "0",
        "subscription_mode": "EXTERNAL",
    }
    if "净值扣除基金财产中目标ETF份额所对应资产净值后的剩余部分" not in text:
        raise ValueError("RATING_ETF_FEE_EXEMPTION_MISSING")
    pub = re.search(r"送出日期：(\d{4})年(\d+)月(\d+)日", text)
    manager = re.search(r"基金经理([\u4e00-\u9fff]{2,4})开始担任", text)
    start = re.search(r"开始担任本基金基金经理的日期(\d{4})年(\d+)月(\d+)日", text)
    if not pub or not manager or not start:
        raise ValueError("RATING_ETF_KEY_DOCUMENT_FIELDS")
    cost = standard_cost(
        *(fees[k] for k in ("management", "custody", "sales", "subscription", "redemption")),
        mode="EXTERNAL",
        operating_cost=fees["operating_cost"],
    )
    return {
        "fund_code": FUND_CODE,
        "product_id": "CMF:016007",
        "currency": "CNY",
        "published_on": str(date(*map(int, pub.groups()))),
        "current_manager": manager[1],
        "manager_start": str(date(*map(int, start.groups()))),
        "fees": fees,
        "cost_1y": str(cost),
        "operating_cost_basis": "官方资料概要综合测算；目标ETF资产部分管理和托管费豁免，不重复加基本费率",
    }


def report_facts(link_pages: list[str], target_pages: list[str]) -> dict:
    """共同报告期、目标身份及不利披露逐项核对；没有大额个人比例字段就保留未知。"""
    link, target = compact("\n".join(link_pages)), compact("\n".join(target_pages))
    if not all(
        x in link
        for x in (
            "下属分级基金的交易代码016007016008",
            "基金主代码159779",
            "本基金本报告期末未持有股票",
            "本报告期基金投资策略无改变",
            "本报告期内，基金管理人无受调查或处罚等情况",
        )
    ):
        raise ValueError("RATING_ETF_LINK_REPORT_FACTS")
    if not all(
        x in target
        for x in ("基金主代码159779", "本报告期基金投资策略无改变", "本报告期内，基金管理人无受调查或处罚等情况")
    ):
        raise ValueError("RATING_ETF_TARGET_REPORT_FACTS")
    dates = []
    for pages in (link_pages, target_pages):
        m = re.search(r"(\d{4})年(\d{2})月(\d{2})日", compact(pages[0]))
        if not m:
            raise ValueError("RATING_ETF_REPORT_DATE")
        dates.append(str(date(*map(int, m.groups()))))
    if dates[0] != dates[1]:
        raise ValueError("RATING_ETF_LOOKTHROUGH_DATE_MISMATCH")
    # 穿透持仓的最大持有人包括联接基金自己，不能解释为另一个外部大额持有人。
    owner = re.search(r"74,264,100\.0052\.35%", target)
    if not owner:
        raise ValueError("RATING_ETF_TARGET_HOLDER_DISCLOSURE")
    return {
        "report_date": dates[0],
        "strategy_continuity": 1,
        "governance_state": 2,
        "largest_personal_holder_ratio": None,
        "target_disclosed_holder_percent": "52.35",
        "target_disclosed_holder_name": "招商中证消费电子主题ETF联接基金",
    }


def appointed_teams(pages: list[str], info: dict, end: date) -> dict:
    """核对招募书正式经理历史；指数评价使用跟踪表现，任期仅展示事实。"""
    text = compact("\n".join(pages))
    m = re.search(
        r"本基金历任基金经理包括：([\u4e00-\u9fff]{2,4})女士，管理时间为(\d{4})年(\d+)月(\d+)日至(\d{4})年(\d+)月(\d+)日",
        text,
    )
    if not m or str(date(*map(int, m.groups()[4:]))) != info["manager_start"]:
        raise ValueError("RATING_ETF_MANAGER_HISTORY")
    return {
        "teams": [
            {"effective_date": str(date(*map(int, m.groups()[1:4]))), "members": [m[1]]},
            {"effective_date": info["manager_start"], "members": [info["current_manager"]]},
        ],
        "team_changes": 1,
        "team_months": str(Decimal((end - date.fromisoformat(info["manager_start"])).days) / Decimal("30.4375")),
    }


def _read_archive(ref: dict, root: Path) -> bytes:
    p = (root / ref["file"]).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file() or p.stat().st_size > 20_000_000:
        raise ValueError("RATING_ETF_ARCHIVE_PATH")
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("RATING_ETF_ARCHIVE_HASH")
    return data


def _pages(raw: bytes) -> list[str]:
    return [p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages]


def original_facts(refs: dict, root: Path, as_of: date) -> dict:
    """从归档原件重建事实；准备与导入均执行，无法用手填字段绕过原文核验。"""
    from app.services.direction_1d_protocol import calendar

    lp, tp = (_pages(_read_archive(refs[k], root)) for k in ("link-half", "target-half"))
    for pages in (lp, tp):
        published = re.search(r"送出日期：(\d{4})年(\d+)月(\d+)日", compact(pages[0]))
        if not published or date(*map(int, published.groups())) > as_of:
            raise ValueError("RATING_ETF_REPORT_AFTER_CUTOFF")
    summary = key_information(_pages(_read_archive(refs["link-summary"], root)))
    summary["checked_on"] = (
        datetime.fromisoformat(refs["link-summary"]["acquired_at"])
        .astimezone(ZoneInfo("Asia/Shanghai"))
        .date()
        .isoformat()
    )
    if date.fromisoformat(summary["published_on"]) > as_of:
        raise ValueError("RATING_ETF_DOCUMENT_AFTER_CUTOFF")
    disclosure = report_facts(lp, tp)
    holding = target_holdings(tp)
    holding["report_date"] = disclosure["report_date"]
    lb = balance(lp)
    if decimal(lb["assets"]["其中：股票投资"]) != 0:
        raise ValueError("RATING_ETF_DIRECT_STOCKS_UNSUPPORTED")
    holding["link_balance"] = lb
    holding["target_fund_amount"] = lb["assets"]["基金投资"]
    holding["net_assets"] = lb["net_assets"]
    # 目标基金明细必须与基金投资总额一致，避免遗漏第二只底层基金。
    s = compact(_section("\n".join(lp), "7.2 期末投资目标基金明细", "7.3 期末按行业分类"))
    if "招商中证消费电子主题ETF" not in s or compact(format(decimal(holding["target_fund_amount"]), ",.2f")) not in s:
        raise ValueError("RATING_ETF_TARGET_FUND_VALUE_MISMATCH")
    manager = appointed_teams(_pages(_read_archive(refs["link-prospectus"], root)), summary, as_of)
    sessions, calendar_hash = calendar()
    month = as_of.replace(year=as_of.year - 3, day=1)
    first = max(d for d in sessions if (d.year, d.month) == (month.year, month.month))
    days = [d for d in sessions if first <= d <= as_of]
    nav_rows = []
    for label in sorted(k for k in refs if k.startswith("official-nav-page")):
        result = json.loads(_read_archive(refs[label], root))
        if result.get("code") != 0:
            raise ValueError("RATING_ETF_SOURCE_REJECTED")
        nav_rows.extend(result["data"]["list"])
    by_day = {}
    for row in nav_rows:
        if row["productCode"] != FUND_CODE or row["navDate"] in by_day:
            raise ValueError("RATING_ETF_NAV_IDENTITY_OR_DUPLICATE")
        by_day[row["navDate"]] = row
    if any(str(d) not in by_day for d in days):
        raise ValueError("RATING_ETF_NAV_CALENDAR_GAP")
    dividend = json.loads(_read_archive(refs["getFundMelonmd-1"], root))
    split = json.loads(_read_archive(refs["getFundSplit-1"], root))
    if any(v.get("code") != 0 or v.get("data") != [] for v in (dividend, split)):
        raise ValueError("RATING_ETF_EVENT_PATTERN_UNSUPPORTED")
    rows = [by_day[str(d)] for d in days]
    if any(
        decimal(v["relatePrice"]) != decimal(v["cumulativeNet"]) or decimal(v["relatePrice"]) != decimal(v["adjustNav"])
        for v in rows
    ):
        raise ValueError("RATING_ETF_NAV_EVENT_CONFLICT")
    chart_response = json.loads(_read_archive(refs["official-chart"], root))
    if chart_response.get("code") != 0:
        raise ValueError("RATING_ETF_BENCHMARK_MISSING")
    chart = chart_response["data"]
    if len({c["date"] for c in chart}) != len(chart) or compact(chart[0]["indexName"]) != BENCHMARK_NAME:
        raise ValueError("RATING_ETF_BENCHMARK_IDENTITY")
    cdays = {c["date"]: c for c in chart}
    values = [decimal(v["relatePrice"]) for v in rows]
    for day, value in zip(days, values, strict=True):
        if str(day) not in cdays or abs(value / values[0] - 1 - decimal(cdays[str(day)]["navYield"]) / 100) > Decimal(
            ".000051"
        ):
            raise ValueError("RATING_ETF_OFFICIAL_RETURN_CONFLICT")
    months = []
    for d in days:
        if not months or (months[-1].year, months[-1].month) != (d.year, d.month):
            months.append(d)
        else:
            months[-1] = d
    history = {
        "dates": list(map(str, days)),
        "unit_nav": list(map(str, values)),
        "cash_dividend_per_share": ["0"] * len(days),
        "split_factor": ["1"] * len(days),
        "month_ends": list(map(str, months)),
        "calendar_hash": calendar_hash,
        "events_complete": True,
        "missing_dates": [],
        "event_coverage": {
            "dates_checked": len(days),
            "official_event_endpoints_empty": True,
            "official_unit_cumulative_adjusted_equal": True,
        },
        "benchmark_index": [str(1 + decimal(cdays[str(d)]["hs300Yield"]) / 100) for d in days],
        "benchmark_name": BENCHMARK_NAME,
        "supplementary_non_trading_dates": sorted(set(cdays) - set(map(str, days))),
    }
    return {
        "history": history,
        "holdings": holding,
        "key_information": summary,
        "disclosure": disclosure,
        "manager": manager,
    }


def prepare_etf_bundle(source_root: Path, as_of: date, frozen: datetime) -> dict:
    """归档已取得资料并生成待导入包；不会改变数据库评级、方法激活或其他基金输入。"""
    from app.services.fund_rating_inputs import canonical

    refs = {}
    for p in source_root.glob("*-receipt.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        label = r.get("label") or p.name.removesuffix("-receipt.json")
        if label == "getFundNavPage-2":
            label = "official-nav-page2"
        if label in {
            "link-half",
            "target-half",
            "link-summary",
            "link-prospectus",
            "official-chart",
            "getFundMelonmd-1",
            "getFundSplit-1",
            "productAnn-0",
            "productAnn-2",
        } or label.startswith("official-nav-page"):
            r["label"] = label
            refs[label] = r
    required = {
        "link-half",
        "target-half",
        "link-summary",
        "link-prospectus",
        "official-chart",
        "getFundMelonmd-1",
        "getFundSplit-1",
        "productAnn-0",
        "productAnn-2",
    }
    if not required <= refs.keys():
        raise ValueError("RATING_ETF_REQUIRED_ARCHIVES")
    facts = original_facts(refs, source_root, as_of)
    # 不把新概要或新经理公告遗漏在手选资料之外：核对完整官方目录的最新可用条目。
    sales = json.loads(_read_archive(refs["productAnn-2"], source_root))["data"]
    announcements = json.loads(_read_archive(refs["productAnn-0"], source_root))["data"]
    if len(sales["list"]) != sales["total"] or len(announcements["list"]) != announcements["total"]:
        raise ValueError("RATING_ETF_CATALOG_INCOMPLETE")
    candidates = [
        x
        for x in sales["list"]
        if "C类份额" in compact(x["title"]) and "资料概要" in x["title"] and x["pubDate"] <= str(as_of)
    ]
    latest = max(candidates, key=lambda x: x["pubDate"])
    if latest["attachmentUrl"] != refs["link-summary"]["url"]:
        raise ValueError("RATING_ETF_NEWER_SUMMARY_EXISTS")
    archive_root = Path(get_settings().fund_insight_directory) / "ratings"
    archives = {}
    for r in refs.values():
        raw = _read_archive(r, source_root)
        acquired = datetime.fromisoformat(r["acquired_at"])
        if acquired.tzinfo is None or acquired > frozen:
            raise ValueError("RATING_ETF_SOURCE_FUTURE")
        extension = ".pdf" if raw.startswith(b"%PDF") else ".json"
        relative = "sources/" + r["sha256"] + extension
        out = archive_root / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if hashlib.sha256(out.read_bytes()).hexdigest() != r["sha256"]:
                raise ValueError("RATING_ETF_EXISTING_ARCHIVE_CORRUPT")
        else:
            with out.open("xb") as stream:
                stream.write(raw)
        if r["sha256"] in archives:
            # 分红和折算的真实空响应可以同字节；保留两个请求回执，共用一份不可变原文。
            archives[r["sha256"]].setdefault("aliases", []).append({**r, "file": relative})
        else:
            archives[r["sha256"]] = {**r, "file": relative}
    sources = [
        {"title": title, "url": refs[label]["url"], "published_on": published}
        for label, title, published in (
            ("link-half", "联接基金2026年中期报告", "2026-08-31"),
            ("target-half", "目标ETF2026年中期报告", "2026-08-31"),
            ("link-summary", "016008产品资料概要", facts["key_information"]["published_on"]),
            ("link-prospectus", "联接基金招募说明书", "2026-02-13"),
        )
    ]
    result = {
        "basis": ETF_BASIS,
        "fund_code": FUND_CODE,
        "family": "INDEX",
        "currency": "CNY",
        "as_of_date": str(as_of),
        "found_date": "2023-02-14",
        "source_scope": "PUBLIC_DISCLOSURE_LOCAL_ANALYSIS",
        "category_label": "境内消费电子ETF联接C（公开资料评价）",
        "sources": sources,
        "raw_archive": list(archives.values()),
        "report_hash": refs["link-half"]["sha256"],
        "summary_hash": refs["link-summary"]["sha256"],
        "documents_checked_at": min(refs[x]["acquired_at"] for x in ("link-summary", "link-prospectus")),
        "classification": {
            "product_id": "CMF:016007",
            "family": "INDEX",
            "relationship_verified": True,
            "effective_from": str(as_of),
            "source_refs": [refs["link-summary"]["sha256"], refs["link-half"]["sha256"]],
            "scope": SCOPE,
        },
        **facts,
    }
    return json.loads(canonical(result))


def verify_etf_originals(bundle: dict) -> None:
    """导入和读取有效性时重建原文事实；任一字段手填修改或原件冲突都拒绝。"""
    root = Path(get_settings().fund_insight_directory) / "ratings"
    refs = {v["label"]: v for r in bundle["raw_archive"] for v in (r, *r.get("aliases", []))}
    if bundle.get("source_profile") == REPRINT_PROFILE:
        actual = reprint_facts(
            refs,
            root,
            bundle["fund_code"],
            bundle["fund_name_stem"],
            date.fromisoformat(bundle["as_of_date"]),
            bundle["classification"]["scope"],
        )
        if any(actual[k] != bundle[k] for k in actual):
            raise ValueError("RATING_ETF_ORIGINAL_FACT_CONFLICT")
        if (
            refs["link-half"]["sha256"] != bundle["report_hash"]
            or refs["link-summary"]["sha256"] != bundle["summary_hash"]
        ):
            raise ValueError("RATING_ETF_DOCUMENT_REFERENCE_CONFLICT")
        return
    actual = original_facts(refs, root, date.fromisoformat(bundle["as_of_date"]))
    if any(actual[k] != bundle[k] for k in actual):
        raise ValueError("RATING_ETF_ORIGINAL_FACT_CONFLICT")
    if refs["link-half"]["sha256"] != bundle["report_hash"] or refs["link-summary"]["sha256"] != bundle["summary_hash"]:
        raise ValueError("RATING_ETF_DOCUMENT_REFERENCE_CONFLICT")


def build_etf_metrics(bundle: dict, as_of: date, frozen: datetime) -> tuple[dict, dict, datetime]:
    """八维计算的真实入口；共用历史、有效期规则，穿透结构及跟踪使用专用指标。"""
    from app.services.fund_rating_public_inputs import public_history, public_valid_until

    code = bundle.get("fund_code")
    if (
        bundle.get("basis") != ETF_BASIS
        or not (isinstance(code, str) and re.fullmatch(r"\d{6}", code))
        or bundle.get("family") != "INDEX"
        or bundle.get("currency") != "CNY"
        or bundle.get("as_of_date") != str(as_of)
    ):
        raise ValueError("RATING_ETF_SCOPE")
    etf_scope(bundle["classification"]["scope"])
    holding = bundle["holdings"]
    target = holding["balance"]
    for b in (target, holding["link_balance"]):
        if abs(sum(map(decimal, b["assets"].values())) - decimal(b["total_assets"])) > Decimal(".01") or abs(
            decimal(b["total_assets"]) - decimal(b["liabilities"]) - decimal(b["net_assets"])
        ) > Decimal(".01"):
            raise ValueError("RATING_ETF_BALANCE_MISMATCH")
    if (
        holding["net_assets"] != holding["link_balance"]["net_assets"]
        or holding["target_fund_amount"] != holding["link_balance"]["assets"]["基金投资"]
    ):
        raise ValueError("RATING_ETF_LINK_BALANCE_CONFLICT")
    info, disclosure = bundle["key_information"], bundle["disclosure"]
    if info["fund_code"] != code or info["currency"] != "CNY" or date.fromisoformat(info["published_on"]) > as_of:
        raise ValueError("RATING_ETF_KEY_DOCUMENT_SCOPE")
    if disclosure["strategy_continuity"] not in (0, 1) or disclosure["governance_state"] not in (0, 1, 2):
        raise ValueError("RATING_ETF_DISCLOSURE_STATE")
    metrics = public_history(bundle["history"], as_of)
    metrics.update({k: decimal(disclosure[k]) for k in ("strategy_continuity", "governance_state")})
    # 正式经理历史是核验事实，不用于替代本类型的跟踪指标。
    metrics.update({k: decimal(bundle["manager"][k]) for k in ("team_months", "team_changes")})
    stock_total = sum(decimal(p["amount"]) for p in holding["positions"])
    # 资产负债表股票投资可能含原文注明的不在明细合计内的可退替代款估值增值；无注明时增量为零。
    stock_expected = decimal(target["assets"]["其中：股票投资"]) - decimal(holding.get("substitution_increment", "0"))
    if (
        stock_total <= 0
        or any(decimal(p["amount"]) <= 0 for p in holding["positions"])
        or abs(stock_total - stock_expected) > Decimal(".01")
    ):
        raise ValueError("RATING_ETF_STOCK_SUM_MISMATCH")
    metrics["issuer_hhi"] = sum((decimal(p["amount"]) / stock_total) ** 2 for p in holding["positions"])
    totals = {}
    for p in holding["industries"]:
        totals[p["code"]] = totals.get(p["code"], Decimal(0)) + decimal(p["amount"])
    if abs(sum(totals.values()) - stock_total) > Decimal(".01"):
        raise ValueError("RATING_ETF_INDUSTRY_SUM_MISMATCH")
    metrics["industry_hhi"] = sum((v / stock_total) ** 2 for v in totals.values())
    net = decimal(holding["net_assets"])
    target_net = decimal(target["net_assets"])
    metrics.update(
        net_assets=net,
        cash_ratio=decimal(holding["link_balance"]["assets"]["货币资金"]) / net,
        restricted_ratio=decimal(holding["target_fund_amount"])
        / net
        * decimal(holding["restricted_stock_amount"])
        / target_net,
    )
    benchmark = list(map(decimal, bundle["history"]["benchmark_index"]))
    index = total_return(*(bundle["history"][k] for k in ("unit_nav", "cash_dividend_per_share", "split_factor")))
    if len(benchmark) != len(index) or any(x <= 0 for x in benchmark):
        raise ValueError("RATING_ETF_TRACKING_ALIGNMENT")
    diff = [(index[i] / index[i - 1] - 1) - (benchmark[i] / benchmark[i - 1] - 1) for i in range(1, len(index))]
    mean = sum(diff) / len(diff)
    metrics["tracking_error"] = (252 * sum((v - mean) ** 2 for v in diff) / (len(diff) - 1)).sqrt()
    metrics["tracking_bias"] = sum(abs(v) for v in diff) / len(diff)
    targets = tracking_targets(bundle)
    metrics["tracking_error_ratio"] = metrics["tracking_error"] / decimal(targets["tracking_error"])
    metrics["tracking_bias_ratio"] = metrics["tracking_bias"] / decimal(targets["tracking_bias"])
    if any(not 0 <= metrics[k] <= 1 for k in ("restricted_ratio", "cash_ratio")):
        raise ValueError("RATING_ETF_ASSET_RATIO")
    fees = info["fees"]
    metrics["cost_1y"] = standard_cost(
        *(fees[k] for k in ("management", "custody", "sales", "subscription", "redemption")),
        mode=fees["subscription_mode"],
        operating_cost=fees["operating_cost"],
    )
    report = date.fromisoformat(holding["report_date"])
    expiry = public_valid_until(as_of, report, datetime.fromisoformat(bundle["documents_checked_at"]), frozen)
    facts = {"nav": str(as_of), "holdings": str(report), "manager": info["checked_on"], "fees": info["checked_on"]}
    return metrics, facts, expiry


def independent_etf_history(bundle: dict, metrics: dict) -> dict:
    """独立浮点复算真实股票/行业/现金及每日跟踪，不调用生产结构和跟踪计算。"""
    from app.services.fund_rating_public_validation import independent_history

    common = independent_history(bundle, metrics, history_only=True)
    h = bundle["holdings"]
    nav = list(map(float, bundle["history"]["unit_nav"]))
    bm = list(map(float, bundle["history"]["benchmark_index"]))
    diff = [(nav[i] / nav[i - 1] - 1) - (bm[i] / bm[i - 1] - 1) for i in range(1, len(nav))]
    avg = sum(diff) / len(diff)
    tracking_error = math.sqrt(252 * sum((x - avg) ** 2 for x in diff) / (len(diff) - 1))
    tracking_bias = sum(abs(x) for x in diff) / len(diff)
    industry = {}
    for p in h["industries"]:
        industry[p["code"]] = industry.get(p["code"], 0.0) + float(p["amount"])
    total = sum(float(p["amount"]) for p in h["positions"])
    targets = tracking_targets(bundle)
    error_target = float(targets["tracking_error"])
    bias_target = float(targets["tracking_bias"])
    actual = {
        "issuer_hhi": sum((float(p["amount"]) / total) ** 2 for p in h["positions"]),
        "industry_hhi": sum((x / total) ** 2 for x in industry.values()),
        "restricted_ratio": float(h["target_fund_amount"])
        / float(h["net_assets"])
        * float(h["restricted_stock_amount"])
        / float(h["balance"]["net_assets"]),
        "tracking_error": tracking_error,
        "tracking_bias": tracking_bias,
        "tracking_error_ratio": tracking_error / error_target,
        "tracking_bias_ratio": tracking_bias / bias_target,
        "net_assets": float(h["net_assets"]),
        "cash_ratio": float(h["link_balance"]["assets"]["货币资金"]) / float(h["net_assets"]),
        "strategy_continuity": float(bundle["disclosure"]["strategy_continuity"]),
        "governance_state": float(bundle["disclosure"]["governance_state"]),
    }
    fees = bundle["key_information"]["fees"]
    buy = float(fees["subscription"])
    purchase = buy / (1 + buy) if fees["subscription_mode"] == "EXTERNAL" else buy
    actual["cost_1y"] = purchase + (1 - purchase) * float(fees["redemption"]) + float(fees["operating_cost"])
    if any(not math.isclose(v, float(metrics[k]), rel_tol=1e-11, abs_tol=1e-12) for k, v in actual.items()):
        raise ValueError("RATING_ETF_INDEPENDENT_METRIC_MISMATCH")
    common["metrics"].update(actual)
    return common


def reprint_scope(target_fund: str, target_index: str) -> dict:
    """逐只基金的适用范围；目标 ETF 与标的指数随后必须在原件中逐一核对。"""
    return etf_scope(
        {
            "basis": ETF_BASIS,
            "asset": "DOMESTIC_EQUITY_ETF_FEEDER",
            "currency": "CNY",
            "target_fund": target_fund,
            "target_index": target_index,
            "share_class": "C",
        }
    )


def _reprint_section(text: str, start: str, end: str) -> str:
    """公开转载文本的章节边界；与首案相同的“取正文最后一次标题”语义。"""
    matches = list(re.finditer(start, text))
    if not matches:
        raise ValueError("RATING_ETF_SECTION_MISSING:" + start)
    stop = re.search(end, text[matches[-1].end() :])
    if not stop:
        raise ValueError("RATING_ETF_SECTION_MISSING:" + end)
    return text[matches[-1].start() : matches[-1].end() + stop.start()]


def reprint_balance(pages: list[str]) -> dict:
    """转载文本资产负债表；科目行与对账规则同首案，仅标题空白按转载排版适配。"""
    text = _reprint_section("\n".join(pages), r"6\.1\s*资产负债表", r"6\.2\s*利润表")
    if "单位：人民币元" not in compact(text):
        raise ValueError("RATING_ETF_BALANCE_UNIT")
    names = (
        "货币资金",
        "结算备付金",
        "存出保证金",
        "其中：股票投资",
        "基金投资",
        "债券投资",
        "资产支持证券投资",
        "贵金属投资",
        "其他投资",
        "衍生金融资产",
        "买入返售金融资产",
        "债权投资",
        "其他债权投资",
        "其他权益工具投资",
        "应收清算款",
        "应收股利",
        "应收申购款",
        "递延所得税资产",
        "其他资产",
    )
    # “其中”重复的债权子行不属于资产一级项目，避免重复相加；转载排版的行首空白照原文适配。
    cleaned = re.sub(
        r"(?m)^(\s*债权投资[^\n]*\n)\s*其中：债券投资[^\n]*\n\s*资产支持证券投资[^\n]*\n\s*其他投资[^\n]*\n",
        r"\1",
        text,
    )
    # 管理人对零值科目可整行不列示；缺失行取零并以资产合计勾稽兜底，重复行仍然拒绝。
    def row_amount(label):
        lines = [x.strip() for x in cleaned.splitlines() if re.match(r"^\s*" + re.escape(label) + r"\s", x)]
        if not lines:
            return "0"
        if len(lines) != 1:
            raise ValueError("RATING_ETF_BALANCE_ROW:" + label)
        rest = re.sub(r"^6\.4\.\d+(?:\.\d+)*\s+", "", lines[0][len(label) :].strip())
        m = re.match(r"(-|[\d,]+\.\d{2})(?:\s|$)", rest)
        if not m:
            raise ValueError("RATING_ETF_BALANCE_VALUE:" + label)
        return "0" if m[1] == "-" else m[1].replace(",", "")

    assets = {name: row_amount(name) for name in names}
    result = {
        "assets": assets,
        "total_assets": _amount(text, "资产总计"),
        "liabilities": _amount(text, "负债合计"),
        "net_assets": _amount(text, "净资产合计"),
    }
    if min(map(decimal, assets.values())) < 0 or decimal(result["net_assets"]) <= 0:
        raise ValueError("RATING_ETF_BALANCE_RANGE")
    if abs(sum(map(decimal, assets.values())) - decimal(result["total_assets"])) > Decimal(".01") or abs(
        decimal(result["total_assets"]) - decimal(result["liabilities"]) - decimal(result["net_assets"])
    ) > Decimal(".01"):
        raise ValueError("RATING_ETF_BALANCE_MISMATCH")
    return result


_BENCHMARK_EXPRESSION_NORMALIZE = str.maketrans({"×": "*", "＋": "+", "（": "(", "）": ")", "％": "%"})


def _benchmark_expression(text: str) -> str:
    """业绩比较基准表述以概要正文为准；只规范化全半角符号，不改写任何成分。"""
    m = re.search(r"业绩比较基准([^。；;]{4,140}?95[％%][^。；;]{0,60}?5[％%])", text)
    if not m:
        raise ValueError("RATING_ETF_BENCHMARK_EXPRESSION_MISSING")
    return compact(m[1]).translate(_BENCHMARK_EXPRESSION_NORMALIZE)


def _benchmark_series(raw: dict, ref: dict, code: str, expression: str) -> dict:
    """管理人官方逐日基准归一化为同日基金收益与基准收益；身份不符即拒绝。"""

    def iso(day):
        if re.fullmatch(r"\d{8}", str(day)):
            return date(*map(int, (str(day)[:4], str(day)[4:6], str(day)[6:8]))).isoformat()
        return str(day)

    fmt = ref.get("format")
    if fmt == "ISSUER_DAILY_BENCHMARK_TH":
        payload = raw.get("payload") or {}
        if str(raw.get("retCode")) != "0" or str(payload.get("retCode")) != "0":
            raise ValueError("RATING_ETF_BENCHMARK_RESPONSE")
        if compact(str(payload.get("benchMark") or "")).translate(_BENCHMARK_EXPRESSION_NORMALIZE) != expression:
            raise ValueError("RATING_ETF_BENCHMARK_IDENTITY")
        fund, index = payload.get("fundHistoryRecord") or {}, payload.get("indexHistoryRecord") or {}
        if fund.get("fundCode") != code:
            raise ValueError("RATING_ETF_BENCHMARK_PRODUCT")
        fund_rows, index_rows = fund.get("fundDayRecordList"), index.get("fundDayRecordList")
        if not isinstance(fund_rows, list) or not isinstance(index_rows, list) or not fund_rows:
            raise ValueError("RATING_ETF_BENCHMARK_RESPONSE")
        by_day = {r["fundDate"]: r for r in index_rows}
        if len(by_day) != len(index_rows) or len({r["fundDate"] for r in fund_rows}) != len(fund_rows):
            raise ValueError("RATING_ETF_BENCHMARK_DUPLICATE")
        rows = []
        for r in fund_rows:
            day = r["fundDate"]
            if day not in by_day:
                raise ValueError("RATING_ETF_BENCHMARK_DAY_MISMATCH")
            rows.append(
                {
                    "navDate": iso(day),
                    "productCode": code,
                    "unit_nav": decimal(r["fundNetValue"]),
                    "publisher_nav": decimal(r["fundAdjustedNav"]),
                    "benchmark_level": decimal(by_day[day]["fundNetValue"]),
                }
            )
        return {"kind": "LEVELS", "rows": rows}
    if fmt == "ISSUER_DAILY_BENCHMARK_FS":
        if raw.get("code") != "0000" or not isinstance(raw.get("data"), list) or not raw["data"]:
            raise ValueError("RATING_ETF_BENCHMARK_RESPONSE")
        if any(r.get("fundCode") != code for r in raw["data"]):
            raise ValueError("RATING_ETF_BENCHMARK_PRODUCT")
        if len({r["navDate"] for r in raw["data"]}) != len(raw["data"]):
            raise ValueError("RATING_ETF_BENCHMARK_DUPLICATE")
        rows = [
            {
                "navDate": iso(r["navDate"]),
                "productCode": code,
                "unit_nav": decimal(r["nav"]) if r.get("nav") is not None else None,
                "publisher_cumulative": 1 + decimal(r["dayInc"]) if r.get("dayInc") is not None else None,
                "benchmark_cumulative": 1 + decimal(r["baseInc"]) if r.get("baseInc") is not None else None,
            }
            for r in sorted(raw["data"], key=lambda r: r["navDate"])
        ]
        return {"kind": "CUMULATIVE", "rows": rows}
    if fmt == "ISSUER_DAILY_BENCHMARK_GF":
        rows_raw = raw.get("CREATE")
        if not isinstance(rows_raw, list) or not rows_raw:
            raise ValueError("RATING_ETF_BENCHMARK_RESPONSE")
        if any(r.get("FUNDCODE") != code or r.get("IDXTYPE") != "yield_his" for r in rows_raw):
            raise ValueError("RATING_ETF_BENCHMARK_PRODUCT")
        if len({r["NAVDATE"] for r in rows_raw}) != len(rows_raw):
            raise ValueError("RATING_ETF_BENCHMARK_DUPLICATE")

        def percent(value):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                # 窗口起点的零值官方写作裸 0，其余一律为百分数字符串。
                if value != 0:
                    raise ValueError("RATING_ETF_BENCHMARK_RESPONSE")
                return Decimal(0)
            m = re.fullmatch(r"(-?[\d.]+)%", str(value).strip())
            if not m:
                raise ValueError("RATING_ETF_BENCHMARK_RESPONSE")
            return decimal(m[1]) / 100

        rows = [
            {
                "navDate": iso(r["NAVDATE"]),
                "productCode": code,
                "publisher_cumulative": 1 + v
                if (v := percent(r.get("YIELD"))) is not None
                else None,
                "benchmark_cumulative": 1 + w
                if (w := percent(r.get("YIELD_BENCHMARK"))) is not None
                else None,
            }
            for r in sorted(rows_raw, key=lambda r: r["NAVDATE"])
        ]
        return {"kind": "CUMULATIVE", "rows": rows}
    raise ValueError("RATING_ETF_BENCHMARK_FORMAT")


def _reprint_key_information(
    summary_pages: list[str],
    prospectus_pages: list[str],
    code: str,
    name: str,
    *,
    index_name: str,
) -> dict:
    """对应 C 份额的身份、费用豁免、跟踪目标及标准一年场景；费率不从零占位。"""
    from app.services.fund_rating_public_market import publication

    text = compact("\n".join(summary_pages))
    if (
        code not in text
        or name not in text
        or "交易币种人民币" not in text
        or not re.search(r"基金类型(?:股票型|指数型|基金中基金)", text)
        or "目标ETF" not in text
        or "指数证券投资基金" not in text
        or index_name not in text
    ):
        raise ValueError("RATING_ETF_KEY_DOCUMENT_IDENTITY_OR_SCENARIO")
    share_identity = re.search(
        r"(?:下属基金简称|基金简称C)((?:(?!下属基金|基金代码).){1,80}?C)(?:下属基金(?:交易)?代码|基金代码C)" + code,
        text,
    )
    individual_share = re.search(r"基金简称((?:(?!基金代码).){1,80}?C)基金代码" + code, text[:800])
    # 两栏排版把份额名称尾字母拆到代码单元格之后；仍限定在原表格标签与短距离内。
    interleaved_share = re.search(r"下属基金代码" + code + r"[\u4e00-\u9fffA-Za-z（）()]{0,12}?C(?:基金|）)", text)
    share_ok = (
        share_identity
        or individual_share
        or interleaved_share
        or re.search(r"C(?:类)?(?:基金)?份额", text[:600])
    )
    if not share_ok:
        raise ValueError("RATING_ETF_SHARE_CLASS_IDENTITY")
    master = re.search(r"(?<!属)基金(?:主)?代码(\d{6})", text)
    prospectus = compact("\n".join(prospectus_pages))
    # 招募书正文不强制出现份额代码；其身份由转载公告记录与全文基金名称绑定。
    if name not in prospectus or "C类基金份额" not in prospectus:
        raise ValueError("RATING_ETF_PROSPECTUS_IDENTITY")

    def percentage(pattern, source):
        m = re.search(pattern, source)
        if not m:
            raise ValueError("RATING_ETF_FEE_MISSING:" + pattern)
        return str(decimal(m[1]) / 100)

    fees = {
        "management": percentage(r"管理费(?:年费率|费率)?[：:]?\s*([\d.]+)%", text),
        "custody": percentage(r"托管费(?:年费率|费率)?[：:]?\s*([\d.]+)%", text),
        "sales": percentage(r"销售服务费(?:年费率|费率)?[：:]?\s*([\d.]+)%", text),
        "operating_cost": percentage(r"基金运作综合费率（年化）(?:持有期间|持有期)?[-－—]?\s*([\d.]+)%", text),
        "subscription": "0",
        "redemption": "0",
        "subscription_mode": "EXTERNAL",
    }
    fee_text = text[text.find("三、投资本基金涉及的费用") :] if "三、投资本基金涉及的费用" in text else text
    explicit_c = re.search(
        r"C类(?:基金)?份额(?:不收取认购[、/]申购费用|不收取认购费/申购费|不收取认/申购费|不收取申购费|不收取认购费和申购费)",
        text + prospectus,
    ) or re.search(r"不收取申购费", fee_text)
    if not explicit_c:
        raise ValueError("RATING_ETF_SUBSCRIPTION_SCENARIO")
    if not re.search(
        r"(?:[NYT][≥≧](?:7|30|180|365)日|(?:7|30|180|365)天≤[NYT]|(?:7|30|180|365)日≤[NYT]|[NYT][≥≧](?:7|30|180|365)天)[-－—]?0(?:\.0+)?%?(?![\d.])",
        fee_text,
    ) and not re.search(r"(?:持有期|持有期限|持有时间)(?:在)?(?:7|30)日(?:以上|及以上).*?不收取赎回费", fee_text):
        raise ValueError("RATING_ETF_REDEMPTION_SCENARIO")
    # 目标 ETF 对应资产的管理费/托管费豁免是联接费用口径前提，措辞按管理人变体核对。
    combined = text + prospectus
    exemption = (
        "净值扣除基金财产中目标ETF份额所对应资产净值后的剩余部分" in combined
        or "投资于目标ETF的部分不收取管理费和托管费" in combined
        or re.search(r"投资于目标ETF的[^。]{0,40}不收取管理费", combined)
    )
    if not exemption:
        raise ValueError("RATING_ETF_FEE_EXEMPTION_MISSING")
    current = re.findall(r"基金经理([\u4e00-\u9fff]{2,5})(?:开始担任|任职日期)", text)
    if not current:
        # 两栏排版把“开始担任本基金……基金经理姓名”拆到同一行，措辞变体仍限定原表格标签。
        current = re.findall(
            r"开始担任本基金(?:\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{2}-\d{2})基金经理([\u4e00-\u9fff]{2,4}?)(?=基金经理的日期|证券从业)",
            text,
        )
    if not current:
        raise ValueError("RATING_ETF_CURRENT_TEAM_MISSING")
    starts = re.findall(r"开始担任本基金基金经理的日期(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not starts:
        starts = re.findall(r"开始担任本基金(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not starts:
        starts = re.findall(r"开始担任本基金(\d{4})-(\d{2})-(\d{2})基金经理", text)
    if not starts:
        raise ValueError("RATING_ETF_MANAGER_START_MISSING")
    cost = standard_cost(
        *(fees[k] for k in ("management", "custody", "sales", "subscription", "redemption")),
        mode="EXTERNAL",
        operating_cost=fees["operating_cost"],
    )
    return {
        "fund_code": code,
        "product_id": "PUBLIC:" + master[1] if master else None,
        "currency": "CNY",
        "published_on": publication(summary_pages),
        "current_manager": "、".join(current),
        "current_managers": current,
        "manager_start": str(min(date(*map(int, s)) for s in starts)),
        "fees": fees,
        "cost_1y": str(cost),
        "operating_cost_basis": "官方资料概要综合测算；目标ETF资产部分管理和托管费豁免，不重复加基本费率",
    }


def _tracking_targets(text: str) -> dict:
    error = re.search(r"年(?:化)?跟踪误差(?:不超过|控制在|控制)([\d.]+)%", text)
    bias = re.search(r"日均跟踪偏离度(?:的)?绝对值(?:不超过|控制在)([\d.]+)%", text)
    if not error or not bias:
        raise ValueError("RATING_ETF_TRACKING_TARGETS_MISSING")
    targets = {"tracking_error": str(decimal(error[1]) / 100), "tracking_bias": str(decimal(bias[1]) / 100)}
    if any(decimal(v) <= 0 for v in targets.values()):
        raise ValueError("RATING_ETF_TRACKING_TARGETS")
    return targets


def reprint_facts(refs: dict, root: Path, code: str, name: str, as_of: date, scope: dict) -> dict:
    """公开转载剖面从归档原件重建事实；任何字段不能用无原件依据的手填值。"""
    from app.services.fund_rating_public_market import (
        disclosures,
        provider_history,
        provider_team,
        public_document,
        publication,
    )
    from app.services.fund_rating_public_stock_tables import parse

    scope = etf_scope(scope)
    target_code = scope["target_fund"]
    index_name = compact(scope["target_index"])

    def read(label):
        ref = refs[label]
        raw = (root / ref["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("RATING_ETF_ORIGINAL_HASH")
        return raw

    def pages(label, default_title):
        # 转载公告标题必须含基金全称与文种；dfcfw 附件再按公告目录逐件绑定身份。
        from urllib.parse import urlsplit

        from app.services.fund_rating_public_market import ATTACHMENT_HOST, attachment_identity

        ref = refs[label]
        code_for_doc = target_code if label == "target-half" else code
        title = ref.get("document_title") or default_title
        if urlsplit(ref["url"]).hostname == ATTACHMENT_HOST:
            attachment_identity(refs, root, label, code_for_doc, title)
        return public_document(read(label), ref, code_for_doc, title)[0]

    link_pages = pages("link-half", "中期报告")
    target_pages = pages("target-half", "中期报告")
    summary_pages = pages("link-summary", "产品资料概要")
    prospectus_pages = pages("link-prospectus", "招募说明书")
    link_text = compact("\n".join(link_pages))
    target_text = compact("\n".join(target_pages))
    if name not in link_text or code not in link_text:
        raise ValueError("RATING_ETF_LINK_REPORT_IDENTITY")
    if target_code not in link_text or target_code not in target_text:
        raise ValueError("RATING_ETF_TARGET_IDENTITY")
    for pages in (link_pages, target_pages, summary_pages):
        if publication(pages) > str(as_of):
            raise ValueError("RATING_ETF_DOCUMENT_AFTER_CUTOFF")
    if "2026年6月30日" not in link_text or "2026年6月30日" not in target_text:
        raise ValueError("RATING_ETF_REPORT_PERIOD")
    snapshot = json.loads(read("provider"))
    if (
        snapshot.get("fund_code") != code
        or snapshot.get("source", {}).get("enabled") is not True
        or not snapshot.get("source", {}).get("authorization_verified_at")
    ):
        raise ValueError("RATING_ETF_PROVIDER_IDENTITY_OR_AUTHORIZATION")
    current_names = sorted(
        {
            r["manager_name"]
            for r in snapshot["manager_records"]
            if r["begin_date"] and r["begin_date"] <= str(as_of) and (not r["end_date"] or r["end_date"] > str(as_of))
        }
    )
    info = _reprint_key_information(summary_pages, prospectus_pages, code, name, index_name=index_name)
    if not info["product_id"] or date.fromisoformat(info["published_on"]) > as_of:
        raise ValueError("RATING_ETF_KEY_DOCUMENT_FIELDS")
    if sorted(info["current_managers"]) != current_names:
        raise ValueError("RATING_ETF_TEAM_ORIGINAL_CONFLICT")
    combined = compact("\n".join([*summary_pages, *prospectus_pages]))
    expression = _benchmark_expression(combined)
    if index_name not in expression:
        raise ValueError("RATING_ETF_BENCHMARK_INDEX_MISMATCH")
    targets = _tracking_targets(combined)

    link_balance = reprint_balance(link_pages)
    if decimal(link_balance["assets"]["其中：股票投资"]) != 0 or decimal(link_balance["assets"]["基金投资"]) <= 0:
        raise ValueError("RATING_ETF_DIRECT_STOCKS_UNSUPPORTED")
    # 目标基金明细必须与基金投资总额一致，避免遗漏第二只底层基金；明细章节号按准则版本适配。
    detail = None
    for start, end in (
        (r"7\.2\s*期末投资目标基金明细", r"7\.3\s*(?:报告)?期末按行业分类"),
        (r"7\.10\s*期末按公允价值占基金资产净值比例大小排序的前十名基金投资明细", r"7\.11\s"),
    ):
        try:
            detail = compact(_reprint_section("\n".join(link_pages), start, end))
            break
        except ValueError:
            continue
    if detail is None:
        raise ValueError("RATING_ETF_TARGET_FUND_DETAIL_MISSING")
    if compact(format(decimal(link_balance["assets"]["基金投资"]), ",.2f")) not in detail:
        raise ValueError("RATING_ETF_TARGET_FUND_VALUE_MISMATCH")
    if target_code not in detail:
        # 明细表只列基金名称时，以标的指数核心词与“金额相等即唯一底层”共同绑定身份。
        fragment = re.sub(r"指数$", "", re.sub(r"^(中证|国证|深证|上证|标普)", "", scope["target_index"]))
        if fragment not in detail:
            raise ValueError("RATING_ETF_TARGET_FUND_IDENTITY")
    target_balance = reprint_balance(target_pages)
    # 转载可能把编号标题粘到附注行尾；只恢复换行，不改动任何数值与措辞。
    target_pages = [re.sub(r"(?<=[。；;])(?=7\.2\s+报告期末按行业分类)", "\n", "\n".join(target_pages))]
    target_text = compact("\n".join(target_pages))
    stock_amount = target_balance["assets"]["其中：股票投资"]
    substitution_increment = "0"
    try:
        tables = parse(target_pages, stock_amount, target_balance["net_assets"])
    except ValueError as error:
        if not str(error).startswith("RATING_STOCK_FULL_AMOUNT_MISMATCH"):
            raise
        # 资产负债表股票投资含可退替代款估值增值、明细与行业合计不含的列报差异：
        # 须有原文注释、7.1 权益项与资产负债表一致、明细合计与行业合计独立一致三点核验。
        if "权益投资项含可退替代款估值增值" not in target_text or "不含可退替代款估值增值" not in target_text:
            raise
        industry_total = re.findall(r"(?m)^\s*合计\s+([\d,]+\.\d{2})\s+[\d.]+\s*$", "\n".join(target_pages))
        equity = re.findall(r"(?m)^\s*其中：股票\s+([\d,]+\.\d{2})\s+[\d.]+\s*$", "\n".join(target_pages))
        if len(industry_total) != 1 or len(equity) != 1:
            raise ValueError("RATING_ETF_SUBSTITUTION_INCREMENT_UNVERIFIED") from error
        t2 = decimal(industry_total[0].replace(",", ""))
        if abs(decimal(equity[0].replace(",", "")) - decimal(stock_amount)) > Decimal(".02"):
            raise ValueError("RATING_ETF_SUBSTITUTION_INCREMENT_UNVERIFIED") from error
        tables = parse(target_pages, str(t2), target_balance["net_assets"])
        substitution_increment = str(decimal(stock_amount) - t2)
        if decimal(substitution_increment) <= 0:
            raise ValueError("RATING_ETF_SUBSTITUTION_INCREMENT_UNVERIFIED") from error
    target_disclosure = disclosures(target_pages, tables["holdings"], include_holder=False)
    link_disclosure = disclosures(link_pages, [], include_holder=False)
    positions = []
    restricted_by_code = {}
    for row in target_disclosure["restricted_rows"]:
        restricted_by_code[row["code"]] = restricted_by_code.get(row["code"], Decimal(0)) + decimal(row["amount"])
    for p in tables["holdings"]:
        bare = p["stock_code"].split(".")[0]
        positions.append(
            {
                "code": bare,
                "issuer": bare,
                "name": p["stock_name"],
                "shares": p["quantity_shares"],
                "amount": p["market_value_cny"],
                "kind": "INDEX_ACTIVE_COMBINED",
                "restricted_amount": str(restricted_by_code.get(bare, Decimal(0))),
            }
        )
    industries = [
        {"code": r["industry_code"], "name": r["name"], "amount": r["value_cny"], "kind": "INDEX_ACTIVE_COMBINED"}
        for r in tables["reported_industries"]
    ]
    # 穿透持仓中的大额持有人包括联接基金自己；不存在时保持未知，不解释为零。
    holder_name = None
    holder_percent = None
    holder_section = re.search(r"报告期内单一投资者持有基金份额比例达到或超过20%的情况(.{0,2500})", target_text)
    if holder_section:
        m = re.search(
            r"([\u4e00-\u9fff（）()]{2,40}?(?:联接|ETF联接)[\u4e00-\u9fff（）()]{0,40}?).{0,200}?(\d+\.\d+)%",
            holder_section[1],
        )
        if m:
            holder_name, holder_percent = m[1], str(decimal(m[2]))
    disclosure = {
        "report_date": "2026-06-30",
        "strategy_continuity": 1,
        "governance_state": min(link_disclosure["governance_state"], target_disclosure["governance_state"]),
        "largest_personal_holder_ratio": None,
        "target_disclosed_holder_percent": holder_percent,
        "target_disclosed_holder_name": holder_name,
    }

    _, h = provider_history(snapshot, as_of)
    days = h["dates"]
    rows = {r["nav_date"]: r for r in h["rows"]}
    events = {}
    for r in snapshot.get("dividend_records") or []:
        if r.get("process_status") != "实施":
            raise ValueError("RATING_ETF_EVENT_PATTERN_UNSUPPORTED")
        if days[0] <= r["ex_date"] <= days[-1]:
            events[r["ex_date"]] = decimal(r["cash_dividend"])
    unit_nav = [rows[d]["unit_nav"] for d in days]
    dividends = [str(events.get(d, Decimal(0))) for d in days]
    splits = ["1"] * len(days)
    index = total_return(unit_nav, dividends, splits)
    adjusted = list(map(decimal, h["adjusted_nav"]))
    difference = max(abs(v / adjusted[0] - i) for v, i in zip(adjusted, index, strict=True))
    bound = Decimal(".0001") / min(adjusted) * 2
    if difference > bound:
        raise ValueError("RATING_ETF_PROVIDER_EVENT_CONFLICT")
    series = _benchmark_series(json.loads(read("benchmark")), refs["benchmark"], code, expression)
    if refs["benchmark"].get("format") in ("ISSUER_DAILY_BENCHMARK_FS", "ISSUER_DAILY_BENCHMARK_GF"):
        # 响应本身不含基准表述；管理人官网基金页明文发布同一合同基准，逐字核对后绑定。
        page_text = compact(read("benchmark-page").decode("utf-8")).translate(_BENCHMARK_EXPRESSION_NORMALIZE)
        if code not in page_text or expression not in page_text:
            raise ValueError("RATING_ETF_BENCHMARK_PAGE_IDENTITY")
    by_day = {r["navDate"]: r for r in series["rows"]}
    if any(d not in by_day for d in days):
        raise ValueError("RATING_ETF_BENCHMARK_HISTORY_INCOMPLETE")
    picked = [by_day[d] for d in days]
    if any(v is None for r in picked for k, v in r.items() if k not in ("navDate", "productCode")):
        raise ValueError("RATING_ETF_BENCHMARK_HISTORY_INCOMPLETE")
    if series["kind"] == "LEVELS":
        benchmark_index = [r["benchmark_level"] / picked[0]["benchmark_level"] for r in picked]
        publisher = [r["publisher_nav"] / picked[0]["publisher_nav"] for r in picked]
        if any(decimal(rows[d]["unit_nav"]) != r["unit_nav"] for d, r in zip(days, picked, strict=True)):
            raise ValueError("RATING_ETF_PUBLISHER_UNIT_NAV_CONFLICT")
        publisher_bound = Decimal(".0001") / min(publisher) + Decimal(".0001") / min(adjusted) * 2
    else:
        benchmark_index = [r["benchmark_cumulative"] / picked[0]["benchmark_cumulative"] for r in picked]
        publisher = [r["publisher_cumulative"] / picked[0]["publisher_cumulative"] for r in picked]
        publisher_bound = Decimal(".0001") / min(publisher) + Decimal(".0001") / min(adjusted) * 2
    if any(v <= 0 for v in benchmark_index):
        raise ValueError("RATING_ETF_BENCHMARK_NONPOSITIVE")
    discrepancy = max(abs(a / adjusted[0] - p / publisher[0]) for a, p in zip(adjusted, publisher, strict=True))
    if discrepancy > publisher_bound:
        raise ValueError("RATING_ETF_PROVIDER_PUBLISHER_NAV_CONFLICT")
    entries = []
    total_declared = None
    for label in (k for k in refs if k == "catalog" or k.startswith("catalog-")):
        catalog = json.loads(read(label))
        page = catalog.get("Data")
        if not isinstance(page, list) or any(e.get("FUNDCODE") != code for e in page):
            raise ValueError("RATING_ETF_CATALOG_INCOMPLETE")
        if total_declared is not None and total_declared != catalog.get("TotalCount"):
            raise ValueError("RATING_ETF_CATALOG_INCOMPLETE")
        total_declared = catalog.get("TotalCount")
        entries.extend(page)
    if not entries or total_declared is None or len(entries) != total_declared:
        raise ValueError("RATING_ETF_CATALOG_INCOMPLETE")
    if len({e.get("ID") for e in entries}) != len(entries):
        raise ValueError("RATING_ETF_CATALOG_INCOMPLETE")
    summaries = [
        e
        for e in entries
        if "资料概要" in e.get("TITLE", "") and "C" in e.get("TITLE", "") and e.get("PUBLISHDATEDesc", "") <= str(as_of)
    ]
    if not summaries:
        raise ValueError("RATING_ETF_CATALOG_SUMMARY_MISSING")
    latest = max(summaries, key=lambda e: e["PUBLISHDATEDesc"])
    if refs["link-summary"].get("article_id") != latest["ID"]:
        raise ValueError("RATING_ETF_NEWER_SUMMARY_EXISTS")
    history = {
        "dates": days,
        "unit_nav": unit_nav,
        "cash_dividend_per_share": dividends,
        "split_factor": splits,
        "month_ends": h["month_ends"],
        "calendar_hash": h["calendar_hash"],
        "events_complete": True,
        "missing_dates": [],
        "event_coverage": {
            "dates_checked": len(days),
            "provider_history": "TUSHARE_PRO_FUND",
            "provider_events_verified_at": (snapshot.get("event_refresh") or {}).get("dividends_verified_at"),
            "cash_events": {k: str(v) for k, v in sorted(events.items())},
            "provider_adjusted_max_difference": str(difference),
            "rounding_bound": str(bound),
        },
        "benchmark_index": [str(v) for v in benchmark_index],
        "benchmark_name": expression,
        "publisher_cross_check": {
            "kind": series["kind"],
            "maximum_normalized_difference": str(discrepancy),
            "rounding_bound": str(publisher_bound),
        },
        "supplementary_non_trading_dates": sorted(set(by_day) - set(days)),
    }
    manager = provider_team(snapshot, {"current_managers": current_names}, date.fromisoformat(days[0]), as_of)
    info["checked_on"] = (
        datetime.fromisoformat(refs["link-summary"]["acquired_at"])
        .astimezone(ZoneInfo("Asia/Shanghai"))
        .date()
        .isoformat()
    )
    return {
        "history": history,
        "holdings": {
            "positions": positions,
            "industries": industries,
            "balance": target_balance,
            "link_balance": link_balance,
            "report_date": disclosure["report_date"],
            "net_assets": link_balance["net_assets"],
            "target_fund_amount": link_balance["assets"]["基金投资"],
            "restricted_stock_amount": target_disclosure["restricted_stock_amount"],
            "substitution_increment": substitution_increment,
        },
        "key_information": info,
        "disclosure": disclosure,
        "manager": manager,
        "tracking_targets": targets,
    }
