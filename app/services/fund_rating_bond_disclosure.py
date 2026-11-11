"""债券指数、可转债、混合配置的一年公开证据候选；不联网、不写库。

只从绑定来源重建观察，未知项独立留空。完整债券账簿等仍是补件，
不因新规则能出分关闭；明确数值不达标也不冒充法律违规定性。
"""

import json
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from zoneinfo import ZoneInfo

from app.services.direction_1d_protocol import calendar
from app.services.fund_quality_v2_balance_proofs import strict_balance_values
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_stock_book import periodic_financial_view
from app.services.fund_rating_direct_index import Originals, _flows, holder_execution_evidence
from app.services.fund_rating_evidence_preview import performance
from app.services.fund_rating_management_inputs import calculate as management_calculate
from app.services.fund_rating_market_review import annual_fee_score, fee_profile, normalized_text
from app.services.fund_rating_public_stock_tables import cells
from app.services.fund_rating_reviewed_quality import STATES
from app.services.material_content import archive_member, content_digest

VERSION = "FIXED_INCOME_AND_ALLOCATION_DISCLOSURE_ONE_YEAR_20261111_V1_CANDIDATE"
PROFILES = {"BOND_INDEX", "CONVERTIBLE", "MIXED"}
KEYS = ("return", "risk", "efficiency", "stability", "management", "holdings", "liquidity", "cost")


def require(condition, code):
    if not condition:
        raise ValueError("BOND_DISCLOSURE_" + code)


def _row_before_folio(lines):
    """数据主行之后一遇页码/页眉/页脚就截断，不能跨页找数字补字段。"""
    result = []
    for line in lines:
        value = line.strip()
        if result and (
            re.fullmatch(r"[1-9]\d{0,2}", value) or re.search(r"第\s*\d+\s*页|共\s*\d+\s*页|20\d{2}\s*年.*报告$", value)
        ):
            break
        result.append(line)
    return result


def _adjacent_zero(lines, fragment):
    """仅接已核零续位；1—9单数字有页码歧义，一律不接。"""
    block = _row_before_folio(lines)
    if len(block) < 2 or not re.fullmatch(r" +0 *", block[1]):
        return None
    # 必须紧邻主行，不能跳空行或从后面的行找；多个纯数字候选也拒绝。
    digits = [value.strip() for value in block[1:] if re.fullmatch(r"\d+", value.strip())]
    positions = list(re.finditer(re.escape(fragment), block[0]))
    if digits != ["0"] or len(positions) != 1:
        return None
    edge = sum(2 if unicodedata.east_asian_width(c) in {"W", "F"} else 1 for c in block[0][: positions[0].end()])
    column = len(block[1]) - len(block[1].lstrip())
    if column < 20 or not edge <= column <= edge + 2:
        return None
    return "0"


def interval_predicate(lower, upper, minimum=None, maximum=None):
    """上下界足以唯一判定满足/不满足才给布尔值，跨边界保持未知。"""
    lo, hi = D(str(lower)), D(str(upper))
    require(lo <= hi, "INTERVAL_ORDER")
    if minimum is not None:
        minimum = D(str(minimum))
        if hi < minimum:
            return False
    if maximum is not None:
        maximum = D(str(maximum))
        if lo > maximum:
            return False
    if (minimum is None or lo >= minimum) and (maximum is None or hi <= maximum):
        return True
    return None


def execution_points(profile, observations):
    weights = {
        "MIXED": {"stock": 40, "cash": 40, "leverage": 20},
        "CONVERTIBLE": {"bond": 30, "convertible": 30, "cash": 20, "leverage": 20},
    }[profile]
    require(set(observations) == set(weights), "CORE_CONSTRAINT_SET")
    if any(v is None for v in observations.values()):
        return None
    require(all(type(v) is bool for v in observations.values()), "CORE_CONSTRAINT_KIND")
    return D(75) * sum(D(w) for key, w in weights.items() if observations[key]) / 100


def _json(source, sha):
    return source.json(sha)  # 先核真实原件/同版本库内表示及摘要。


def _manager_claim(source, claim):
    """复用历史ZIP成员同版本提取，不伪造或重建已删除的ZIP/原件二进制。"""
    if not claim.get("archive_member"):
        source.claim(claim)
        return source.plain(claim["source_hash"])
    raw = source.load(claim["source_hash"])
    member = archive_member(raw, claim["archive_member"])
    require(content_digest(member) == claim["member_sha256"], "ARCHIVE_MEMBER_HASH")
    document = json.loads(member.decode("utf-8-sig"))
    text = normalized_text(document["data"]["notice_content"])
    require(normalized_text(claim["quote"]) in text, "ARCHIVE_MEMBER_QUOTE")
    return text


def _nav(inputs, source):
    provider = _json(source, inputs["refs"]["provider"])
    require(
        provider["fund_code"] == inputs["fund_code"]
        and provider["source"]["enabled"]
        and provider["source"]["authorization_verified_at"],
        "PROVIDER_IDENTITY",
    )
    h = provider["history"]
    ends = h["month_ends"][-13:]
    require(len(ends) == 13 and ends[-1] == inputs["as_of_date"], "YEAR_ENDS")
    months = [int(d[:4]) * 12 + int(d[5:7]) for d in ends]
    require(months == list(range(months[0], months[0] + 13)), "CONTIGUOUS_MONTHS")
    sessions, calendar_hash = calendar()
    expected = [str(d) for d in sessions if ends[0] <= str(d) <= ends[-1]]
    days = [d for d in h["dates"] if ends[0] <= d <= ends[-1]]
    require(days == expected and h["calendar_hash"] == calendar_hash, "VALUATION_CALENDAR")
    by_day = {r["nav_date"]: r for r in h["rows"]}
    require(
        len(by_day) == len(h["rows"]) and all(d in by_day and by_day[d].get("content_hash") for d in days),
        "NAV_GAP_OR_PROVENANCE",
    )
    adjusted = [D(by_day[d]["adjusted_nav"]) for d in days]
    require(all(v > 0 for v in adjusted), "NAV_POSITIVE")
    require(all(end == max(str(d) for d in sessions if str(d)[:7] == end[:7]) for end in ends), "ACTUAL_MONTH_ENDS")
    return days, ends, adjusted


def _history(inputs, source):
    days, ends, adjusted = _nav(inputs, source)
    components = []
    for item in inputs["benchmark_components"]:
        # 合同名称不能作为可任填的别名桥梁；只允许已按原件核准的代码/简称/合同名。
        # 否则可把中证转债数据挂到沪深300或中信标普名称下，摘要校验仍会误通过。
        exact_names = {
            "000300.SH": {("沪深300", "沪深300指数")},
            "000012.SH": {("国债指数", "上证国债指数")},
            "000832.CSI": {("中证转债", "中证可转换债券指数")},
            "000906.SH": {("中证800", "中证800指数")},
            "H11001.CSI": {("中证全债", "中证全债指数")},
            "CBA_QJ_00": {
                ("中债综合全价指数", "中债综合全价指数"),
                ("中债综合指数", "中债综合指数"),
            },
        }
        require(
            (item["name"], item.get("contract_name", item["name"])) in exact_names.get(item["code"], set()),
            "CONTRACT_REFERENCE_CODE_NAME_BINDING",
        )
        raw = _json(source, item["source_hash"])
        if item.get("kind") == "CBA":
            ref = inputs["sources"][item["source_hash"]]
            require(
                ref.get("url") == "https://indices.chinabond.com.cn/cbweb-mn/indices/singleIndexQueryResult"
                and ref["params"]["indexid"] == "2c90818811afed8d0111c0c672b31578"
                and item["metric"] == "QJZS_00",
                "CBA_IDENTITY",
            )
            points = {
                datetime.fromtimestamp(int(k) / 1000, ZoneInfo("Asia/Shanghai")).date().isoformat(): D(str(v))
                for k, v in raw[item["metric"]].items()
                if v not in [None, "", "N/A"]
            }
            expected_count = len(points)
        else:
            basic = raw["basic"]
            require(
                basic and basic["index_code"] == item["code"] and basic["display_name"] == item["name"],
                "BENCHMARK_IDENTITY",
            )
            points = {r["trade_date"]: D(r["close_price"]) for r in raw["daily"]}
            expected_count = len(raw["daily"])
        require(len(points) == expected_count and all(d in points and points[d] > 0 for d in days), "BENCHMARK_GAP")
        components.append((D(item["weight"]), points))
    require(sum(w for w, _ in components) == 1, "BENCHMARK_WEIGHT_SUM")
    contract = inputs["benchmark_contract"]
    source.claim(contract)
    require(
        all(
            normalized_text(item.get("contract_name", item["name"])) in normalized_text(contract["quote"])
            for item in inputs["benchmark_components"]
        ),
        "CONTRACT_REFERENCE_NAMES",
    )
    for item in inputs["benchmark_components"]:
        name = re.escape(normalized_text(item.get("contract_name", item["name"])))
        text = normalized_text(contract["quote"])
        patterns = [name + r"(?:收益率)?[×*](\d+(?:\.\d+)?)%", r"(\d+(?:\.\d+)?)%[×*]" + name]
        matches = [D(m[1]) / 100 for pattern in patterns for m in re.finditer(pattern, text)]
        require(matches and set(matches) == {D(item["weight"])}, "CONTRACT_REFERENCE_WEIGHT")
    benchmark = [D(1)]
    for before, after in zip(days, days[1:], strict=False):
        change = sum(w * (points[after] / points[before] - 1) for w, points in components)
        benchmark.append(benchmark[-1] * (1 + change))
    return {
        "dates": days,
        "month_ends": ends,
        "adjusted_nav": list(map(str, adjusted)),
        "benchmark_index": list(map(str, benchmark)),
    }


def _major_stocks(pages, stock_total):
    """只核已披露主要股票的原行，不将全仓可得作为前十大指标前提。"""
    text = "\n".join(pages)
    headings = list(re.finditer(r"([578])\.3\s*[^\n]*股票\s*投\s*资\s*明\s*细", text))
    if not headings:
        require(D(stock_total) == 0 and ("未持有股票" in normalized_text(text)), "MAJOR_STOCK_SECTION")
        return []
    hit = headings[-1]
    tail = text[hit.end() :]
    end = re.search(r"(?m)^\s*" + hit[1] + r"\.4\s*(?!\d)", tail)
    require(end is not None, "MAJOR_STOCK_END")
    lines = tail[: end.start()].splitlines()
    pattern = re.compile(r"^\s*(\d{1,4})\s+(\d{5,6})\s*(.*)$")
    starts = [i for i, line in enumerate(lines) if pattern.match(line)]
    result = []
    for i, start in enumerate(starts):
        rank, code, first = pattern.match(lines[start]).groups()
        if int(rank) > 10:
            break
        stop = starts[i + 1] if i + 1 < len(starts) else len(lines)
        block = _row_before_folio(lines[start:stop])
        rest = []
        for line in block[1:]:
            line = line.strip()
            if not line or re.fullmatch(r"\d{1,3}", line) or re.search(r"20\d{2}\s*年.*报告$", line):
                continue
            if re.match(r"(?:注[:：]?|金额单位|序\s*号|股票代码|占基金|产净值|例[（(])", line):
                break
            rest.append(line)
        # 个别原表把数量最后一位排到下一行，金额和净值比例仍在本行。
        # 只在原数量逗号分组明确少一位、且续行恰有一个数字时接回，随后仍由cells校验。
        wrapped = re.fullmatch(
            r"(.+?)\s+(\d{1,3}(?:,\d{3})*,\d{2})\s+([\d,]+\.\d{2})\s+(\d+\.\d{2})",
            first,
        )
        carry = _adjacent_zero(block, wrapped[2]) if wrapped else None
        if wrapped and carry is not None:
            quantity = wrapped[2] + carry
            require(re.fullmatch(r"\d{1,3}(?:,\d{3})+", quantity), "MAJOR_STOCK_WRAPPED_QUANTITY")
            first = f"{wrapped[1]} {quantity} {wrapped[3]} {wrapped[4]}"
        name, values = cells([first, *rest])
        result.append({"rank": int(rank), "code": code, "name": name, "amount": values[1].replace(",", "")})
    require(
        [r["rank"] for r in result] == list(range(1, len(result) + 1)) and 0 < len(result) <= 10, "MAJOR_STOCK_RANKS"
    )
    require(len({r["code"] for r in result}) == len(result), "MAJOR_STOCK_DUPLICATE")
    total = sum(D(r["amount"]) for r in result)
    require(total <= D(stock_total), "MAJOR_STOCK_EXCEEDS_TOTAL")
    if len(result) < 10:
        require(total == D(stock_total), "SMALL_STOCK_BOOK_NOT_RECONCILED")
    return result


def _major_bonds(pages):
    text = "\n".join(pages)
    hits = list(
        re.finditer(
            r"([578])\.([56])\s*(?:期末|报告期末)按公允价值占基金资产净值比例大小排序的前五名债券投资明细", text
        )
    )
    require(hits, "MAJOR_BOND_SECTION")
    hit = hits[-1]
    tail = text[hit.end() :]
    end = re.search(r"(?m)^\s*" + re.escape(hit[1] + "." + str(int(hit[2]) + 1)) + r"\s*(?:期末|报告期末)", tail)
    require(end is not None, "MAJOR_BOND_END")
    body = tail[: end.start()]
    if "未持有债券" in normalized_text(body):
        return []
    # 名称可跨行；先按明确“序号+代码”分组，再从每组右侧读取数量/金额/比例。
    starts = list(re.finditer(r"(?m)^\s*(\d)\s+(\d{6,9})(?=\s)", body))
    rows = []
    for index, match in enumerate(starts):
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(body)
        lines = body[match.end() : stop].splitlines()
        raw_block = body[match.start() : stop].splitlines()
        while raw_block and not raw_block[0].strip():
            raw_block.pop(0)
        raw_block = _row_before_folio(raw_block)
        # 第一行的序号/代码已经由start匹配，不重复纳入名称与数值。
        lines = [lines[0], *raw_block[1:]]
        clean = [
            line.strip()
            for line in lines
            if line.strip()
            and not re.fullmatch(r"\d{1,3}", line.strip())
            and not re.search(r"20\d{2}\s*年.*报告$", line.strip())
        ]
        line = " ".join(clean)
        # 公允价值小数第二位可能独占下一行；只接明确一位小数后的唯一数字续行。
        wrapped = re.fullmatch(r"(.+?)\s+([\d,]+)\s+([\d,]+\.\d)\s+(\d+\.\d{2})\s*", line)
        carry = _adjacent_zero(raw_block, wrapped[3]) if wrapped else None
        if wrapped and carry is not None:
            line = f"{wrapped[1]} {wrapped[2]} {wrapped[3]}{carry} {wrapped[4]}"
        value = re.fullmatch(r"(.+?)\s+([\d,]+)\s+([\d,]+\.\d{2})\s+(\d+\.\d{2})\s*", line)
        require(value is not None, "MAJOR_BOND_ROW_CELLS")
        rows.append(
            {
                "rank": int(match[1]),
                "code": match[2],
                "name": normalized_text(value[1]),
                "amount": value[3].replace(",", ""),
            }
        )
    require(0 < len(rows) <= 5 and [r["rank"] for r in rows] == list(range(1, len(rows) + 1)), "MAJOR_BOND_RANKS")
    return rows


def _active_execution(inputs, source, balance, bonds):
    rule = inputs["allocation_contract"]
    source.claim(rule["claim"])
    clause = normalized_text(rule["claim"]["quote"])
    require(D(rule["cash_min"]) == D(".05") and re.search(r"基金资产净值(?:的)?5%", clause), "CONTRACT_CASH_FLOOR")
    require(
        D(rule["leverage_max"]) == D("1.4")
        and re.search(r"基金(?:总资产|资产总值)不得超过基金(?:资产净值|净资产)的140%", clause),
        "CONTRACT_LEVERAGE_LIMIT",
    )
    if inputs["profile"] == "MIXED":
        pattern = r"股票[^。；]{0,65}?(\d+(?:\.\d+)?)%?[-—~～至](\d+(?:\.\d+)?)%"
        limits = [(D(m[1]) / 100, D(m[2]) / 100) for m in re.finditer(pattern, clause)]
        require((D(rule["stock_min"]), D(rule["stock_max"])) in limits, "CONTRACT_STOCK_RANGE")
    else:
        # 可转债核心分母尚未独立验证，禁止候选包凭任意定义摘录启用。
        require(rule.get("noncash_basis") is None, "NONCASH_DEFINITION_NOT_VALIDATED")
        require(
            D(rule["bond_min"]) == D(".8") and re.search(r"债券[^。；]{0,60}不低于基金资产的80%", clause),
            "CONTRACT_BOND_MINIMUM",
        )
        require(
            D(rule["convertible_min"]) == D(".8")
            and re.search(r"可转[换]?债[^。；]{0,100}不低于非现金基金资产的80%", clause),
            "CONTRACT_CONVERTIBLE_MINIMUM",
        )
    net = D(balance["net_assets"])
    gross = D(balance["total_assets"])
    stock = D(balance["stock_amount"])
    money = D(balance["assets"]["货币资金"])
    # 以已公开的本金作为下界；不得将结算备付金、应收款混入合格现金。
    text = "\n".join(source.pages(inputs["primary_ref"]))
    begin = list(re.finditer(r"6\.4\.7\.1\s*货币资金", text))
    principal = []
    if begin:
        body = text[begin[-1].end() :]
        stop = re.search(r"6\.4\.7\.2(?!\d)", body)
        body = body[: stop.start()] if stop else body[:3000]
        # 只取活期存款本金作保守下界；定期/其他存款与应计利息不混入。
        demand = re.search(r"活期存款(.*?)(?:定期存款|其他存款|合计)", body, re.S)
        if demand:
            principal = [D(m[1].replace(",", "")) for m in re.finditer(r"本金\s+([\d,]+\.\d{2})", demand[1])]
    lower = sum(principal) if principal else D(0)
    upper = money
    require(lower <= money, "BANK_PRINCIPAL_RECONCILIATION")
    gov_total = D(rule["government_total_cny"])
    gov_rows = re.findall(r"(?m)^\s*1\s+国家债券\s+([\d,]+\.\d{2}|[-－—])", text)
    require(
        (D(balance["bond_amount"]) == 0 and not bonds and not gov_rows and gov_total == 0)
        or (
            len(gov_rows) == 1
            and (D(0) if gov_rows[0] in {"-", "－", "—"} else D(gov_rows[0].replace(",", ""))) == gov_total
        ),
        "GOVERNMENT_TOTAL_SOURCE",
    )
    known_long = D(0)
    for term in inputs.get("government_long_terms", []):
        source.claim(term["claim"])
        maturity = date.fromisoformat(term["maturity"])
        require(
            f"{maturity.year}年{maturity.month}月{maturity.day}日偿还本金" in normalized_text(term["claim"]["quote"]),
            "LONG_BOND_MATURITY_QUOTE",
        )
        require(term["code"] in {r["code"] for r in bonds}, "LONG_BOND_HOLDING")
        row = next(r for r in bonds if r["code"] == term["code"])
        require("国债" in row["name"], "LONG_GOVERNMENT_HOLDING_IDENTITY")
        require(
            date.fromisoformat(term["maturity"])
            > (date.fromisoformat(inputs["holding_date"]).replace(year=int(inputs["holding_date"][:4]) + 1)),
            "LONG_BOND_MATURITY",
        )
        known_long += D(row["amount"])
    require(known_long <= gov_total, "LONG_BOND_EXCEEDS_GOVERNMENT")
    # 对未知国债采取最宽上界仍不能达标，可唯一判“不满足”；跨界保持未知。
    upper += gov_total - known_long
    for term in inputs.get("government_short_terms", []):
        source.claim(term["claim"])
        row = next(r for r in bonds if r["code"] == term["code"])
        maturity = date.fromisoformat(term["maturity"])
        holding = date.fromisoformat(inputs["holding_date"])
        require(
            "国债" in row["name"]
            and f"{maturity.year}年{maturity.month}月{maturity.day}日偿还本金"
            in normalized_text(term["claim"]["quote"]),
            "SHORT_GOVERNMENT_MATURITY_QUOTE",
        )
        require(holding < maturity <= holding.replace(year=holding.year + 1), "SHORT_BOND_MATURITY")
        lower += D(row["amount"])
    require(lower <= upper, "CASH_BOUND_ORDER")
    cash = interval_predicate(lower / net, upper / net, minimum=rule["cash_min"])
    # 期货每日无负债结算后，衍生品公允价值零不能证明交易保证金零。
    # 满足现金下限必须另有明确未投资/未交易原文；否则现金仍未知。
    report_text = normalized_text(text)
    margin_proofs = []
    for instrument in ["股指期货", "国债期货"]:
        if instrument not in clause:
            continue
        no_position = re.search(
            r"本基金本报告期(?:末)?(?:无"
            + instrument
            + r"投资|未(?:持有|投资)"
            + instrument
            + r"|未(?:进行|参与)"
            + instrument
            + r"交易)",
            report_text,
        )
        if no_position:
            margin_proofs.append(
                {"instrument": instrument, "source_hash": inputs["primary_ref"], "quote": no_position[0]}
            )
        elif cash is not False:
            cash = None
    leverage = interval_predicate(gross / net, gross / net, maximum=rule["leverage_max"])
    if inputs["profile"] == "MIXED":
        # 合同未明说“总/净”且两种解释均同侧时才能判；不选有利解释。
        ratios = sorted([stock / gross, stock / net])
        observations = {
            "stock": interval_predicate(*ratios, minimum=rule["stock_min"], maximum=rule["stock_max"]),
            "cash": cash,
            "leverage": leverage,
        }
    else:
        bond = D(balance["bond_amount"])
        cb = D(rule["convertible_total_cny"])
        cb_rows = re.findall(r"(?m)^\s*7\s+可转债[（(]可交换债[）)]\s+([\d,]+\.\d{2})", text)
        require(len(cb_rows) == 1 and D(cb_rows[0].replace(",", "")) == cb, "CONVERTIBLE_CLASSIFICATION_ROW")
        bond_ratios = sorted([bond / gross, bond / net])
        # 非现金定义尚未核准时显式返回未知，不能据一般“合规”自述挑分母。
        noncash = rule.get("noncash_basis")
        cb_state = None
        if noncash is not None:
            raise ValueError("BOND_DISCLOSURE_NONCASH_BASIS_UNVERIFIED")
        observations = {
            "bond": interval_predicate(*bond_ratios, minimum=rule["bond_min"]),
            "convertible": cb_state,
            "cash": cash,
            "leverage": leverage,
        }
    return execution_points(inputs["profile"], observations), {
        "core_constraints": observations,
        "futures_no_position_originals": margin_proofs,
        "unverified_margin_not_assumed_zero": True,
        "bank_principal_from_note": str(sum(principal)) if principal else None,
        "qualified_cash_lower_cny": str(lower),
        "qualified_cash_upper_cny": str(upper),
        "qualified_cash_upper_nav_ratio": str(upper / net),
        "interpretation": "按固定公开约束得分，不是法律违规定性；数值短缺的原因及日常调整继续补件。",
    }


def _management(inputs, source, provider, report_text):
    if inputs.get("management_inputs"):
        result = management_calculate(
            inputs["management_inputs"],
            root=source.root,
            manager_name=inputs["manager_name"],
            as_of=inputs["as_of_date"],
            fund_code=inputs["fund_code"],
            legal_name=inputs["legal_name"],
            primary_report_hash=inputs["primary_ref"],
        )
        return None if result["score"] is None else D(result["score"])
    review = inputs["manager_review"]
    require(
        len(review["claims"]) == 2 and review["claims"][0]["source_hash"] == inputs["primary_ref"],
        "MANAGER_CLAIM_ROLES",
    )
    texts = [_manager_claim(source, claim) for claim in review["claims"]]
    require(review["experience_state"] == "BASIC_ROLE_EVIDENCED", "EXPERIENCE_SCOPE")
    require("基金经理" in normalized_text(review["claims"][0]["quote"]), "ACTUAL_MANAGER_ROLE")
    # 当前候选仅支持已核整改观察档；不能用可填写state把整改变为正常。
    governance = normalized_text(review["claims"][1]["quote"])
    require(review["governance_state"] == "REMEDIATION_DISCLOSED", "GOVERNANCE_STATE_NOT_REPLAYED")
    require(
        normalized_text(inputs["manager_name"]) in texts[1][:16000]
        and "管理人" in governance
        and re.search("警示函|责令改正|罚款", governance),
        "GOVERNANCE_MANAGER_EVENT",
    )
    completed = re.search(
        r"已完成全部整改工作|完成整改|整改成果(?:已经|已).{0,60}验收通过|管理人高度重视[，,]已按法律法规和监管要求对相关问题进行了全面整改",
        governance,
    )
    # 长安两列表格把左栏标签尾“出整改意见)”插进“相关”之间。
    # 只接受这一个已核版式的完整过去时声明，不删任意文字或接受未来计划。
    completed = completed or (
        inputs["manager_name"] == "长安基金管理有限公司"
        and re.search(r"管理人高度重视,已按法律法规和监管要求对相出整改意见\)关问题进行了全面整改[。.]", governance)
    )
    require(completed is not None, "GOVERNANCE_COMPLETED_REMEDIATION")
    event = re.search(r"受到(?:调查|稽查)或处罚等措施的时间(20\d{2})年(\d{1,2})月(\d{1,2})日", governance)
    require(
        event is not None and date(*map(int, event.groups())) <= date.fromisoformat(inputs["as_of_date"]),
        "GOVERNANCE_EVENT_DATE",
    )
    names = review["current_names"]
    require(names and all(n in report_text for n in names), "MANAGER_REPORT_NAMES")
    start = provider["history"]["month_ends"][-13]
    end = inputs["as_of_date"]
    intervals = []
    for row in provider["manager_records"]:
        if row.get("begin_date"):
            intervals.append((row["begin_date"], row.get("end_date") or end))
    cursor = start
    for begin, stop in sorted(intervals):
        if begin <= cursor <= stop:
            cursor = max(cursor, stop)
    require(cursor >= end, "RESPONSIBILITY_WINDOW")
    return (D(STATES[review["experience_state"]]) * 40 + D(75) * 20 + D(STATES[review["governance_state"]]) * 40) / 100


def calculate(inputs: dict, *, root: Path):
    require(inputs.get("version") == VERSION and inputs.get("profile") in PROFILES, "VERSION_PROFILE")
    require(inputs.get("currency") == "CNY", "CURRENCY")
    require(
        inputs["primary_ref"] in inputs["sources"] and inputs["refs"]["half"] == inputs["primary_ref"],
        "PRIMARY_BINDING",
    )
    root = Path(root).resolve()
    source = Originals(inputs, root, lazy=True)
    scores = {key: None for key in KEYS}
    facts = {}
    errors = {}
    reviews_valid = False
    try:
        allocation = inputs["allocation_exception_review"]
        require(
            allocation["complete"] is True
            and allocation["conclusion"] == "NUMERIC_COMPARISON_ONLY_NO_EXCEPTION_RELIED_ON",
            "ALLOCATION_REVIEW_NOT_COMPLETE",
        )
        require(
            {c["source_hash"] for c in allocation["claims"]} == {inputs["primary_ref"], inputs["refs"]["prospectus"]},
            "ALLOCATION_REVIEW_SOURCES",
        )
        for claim in allocation["claims"]:
            source.claim(claim)
        event = inputs["event_review"]
        require(event["complete"] is True, "EVENT_REVIEW_NOT_COMPLETE")
        audit = source.json(event["source_hash"])
        require(
            audit["fund_code"] == inputs["fund_code"]
            and audit["as_of_date"] == inputs["as_of_date"]
            and audit["complete"] is True
            and audit["pending_events"] == []
            and audit["conclusion"] == "NO_UNRESOLVED_MATERIAL_EVENT_FOR_AS_OF",
            "EVENT_REVIEW_AUDIT_BINDING",
        )
        require(
            all(
                r["event"]["fund_code"] == inputs["fund_code"]
                and r["decision"] in {"NO_RATING_IMPACT", "RATING_REVIEW_COMPLETED"}
                and len(r["note"]) >= 80
                for r in audit["reviews"]
            ),
            "EVENT_REVIEW_RECEIPTS",
        )
        reviews_valid = True
        facts["review_completion"] = {
            "allocation": allocation,
            "event_audit_source_hash": event["source_hash"],
            "conclusion": audit["conclusion"],
        }
    except (KeyError, ValueError) as exc:
        errors["review_completion"] = str(exc)
    pages = source.pages(inputs["primary_ref"])
    report = normalized_text("\n".join(pages))
    require(inputs["fund_code"] in report and normalized_text(inputs["legal_name"]) in report, "REPORT_LEGAL_IDENTITY")
    require(str(report_period("\n".join(pages))[1]) == inputs["holding_date"], "REPORT_PERIOD")
    view, _ = periodic_financial_view(pages)
    balance = strict_balance_values(view, observed_on=inputs["holding_date"])["point_book"]
    require(balance is not None, "BALANCE_NOT_IDENTIFIED")
    extracts = []
    for term in ["信用风险", "利率风险敞口", "利率风险的敏感性分析", "前十名证券的发行主体"]:
        position = report.rfind(term)
        if position >= 0:
            extracts.append(
                {"topic": term, "source_hash": inputs["primary_ref"], "quote": report[position : position + 1400]}
            )
    facts["disclosed_risk_facts"] = {
        "holding_date": inputs["holding_date"],
        "gross_assets": balance["total_assets"],
        "net_assets": balance["net_assets"],
        "gross_to_net": str(D(balance["total_assets"]) / D(balance["net_assets"])),
        "report_extracts": extracts,
        "scope": (
            "风险分仅衡量同一年的已实现最大回撤，不代表未来低风险、全部信用审查或全仓压力测试通过。"
            "信用、期限、利率敏感性、转债股性及杠杆继续按已披露事实解释。"
        ),
    }
    with localcontext() as ctx:
        ctx.prec = 65
        for stage in ["performance", "holdings", "execution", "management", "liquidity", "cost"]:
            try:
                if stage == "performance":
                    days, ends, values = _nav(inputs, source)
                    peak = values[0]
                    drawdown = D(0)
                    for value in values:
                        peak = max(peak, value)
                        drawdown = max(drawdown, 1 - value / peak)
                    scores["risk"] = 100 * (1 - drawdown)
                    facts["window"] = {
                        "from": ends[0],
                        "through": ends[-1],
                        "month_ends": ends,
                        "valuation_points": len(days),
                    }
                    history = _history(inputs, source)
                    points, pfacts = performance(history)
                    for key in ["return", "risk", "stability"]:
                        scores[key] = points[key]
                    if inputs["profile"] == "BOND_INDEX":
                        scores["efficiency"] = points["efficiency"]
                    facts["performance"] = pfacts
                    facts["window"] = {
                        "from": history["month_ends"][0],
                        "through": history["month_ends"][-1],
                        "month_ends": history["month_ends"],
                        "valuation_points": len(history["dates"]),
                    }
                elif stage == "holdings":
                    bonds = _major_bonds(view)
                    require(
                        sum(D(row["amount"]) for row in bonds) <= D(balance["bond_amount"]),
                        "MAJOR_BONDS_EXCEED_TOTAL",
                    )
                    if inputs["profile"] == "BOND_INDEX":
                        require(False, "BOND_INDEX_HOLDING_CLASSIFICATION_NOT_VALIDATED")
                    else:
                        stocks = _major_stocks(view, balance["stock_amount"])
                        concentration = sum(D(r["amount"]) for r in stocks) + sum(
                            D(r["amount"]) for r in bonds if "国债" not in r["name"]
                        )
                        scores["holdings"] = 100 * max(D(0), 1 - concentration / D(balance["net_assets"]))
                        facts["holdings"] = {
                            "stocks": stocks,
                            "bonds": bonds,
                            "major_risk_concentration_cny": str(concentration),
                            "net_assets": balance["net_assets"],
                            "unpublished_full_book_still_supplement": True,
                        }
                elif stage == "execution" and inputs["profile"] != "BOND_INDEX":
                    scores["efficiency"], facts["execution"] = _active_execution(
                        inputs, source, balance, _major_bonds(view)
                    )
                elif stage == "management":
                    provider = _json(source, inputs["refs"]["provider"])
                    scores["management"] = _management(inputs, source, provider, report)
                elif stage == "liquidity":
                    review = inputs["holder_review"]
                    source.claim(review["mechanism"])
                    mechanism = normalized_text(review["mechanism"]["quote"])
                    require("赎回" in mechanism and "巨额" in mechanism, "REDEMPTION_MECHANISM")
                    if review.get("execution_profile") == "TRUSTEE_REDEMPTION_PRICING_AND_REPORT_REVIEW":
                        source.claim(review["execution"])
                        source.claim(review["actual_operation"])
                        trustee = normalized_text(review["execution"]["quote"])
                        operation = normalized_text(review["actual_operation"]["quote"])
                        require(
                            all(
                                t in trustee
                                for t in [
                                    "托管人",
                                    "申购赎回价格",
                                    "进行了核查",
                                    "投资组合报告",
                                    "真实",
                                    "准确",
                                    "完整",
                                ]
                            )
                            and ("未发现损害基金" in trustee),
                            "TRUSTEE_ACTUAL_REVIEW",
                        )
                        require(
                            "遵守" in operation
                            and "基金合同" in operation
                            and any(
                                t in operation for t in ["没有损害基金", "无损害基金", "不存在损害基金", "无重大违法"]
                            ),
                            "ACTUAL_OPERATION",
                        )
                    else:
                        holder_execution_evidence(source, review, inputs["primary_ref"])
                    flow_pages = pages
                    if inputs["fund_code"] == "160323":
                        # 同产品正式A简称带LOF、C简称不带；只在申赎表的已核别名视图中统一。
                        flow_pages = [
                            p.replace("华夏磐泰混合（LOF）A", "华夏磐泰混合A").replace(
                                "华夏磐泰混合(LOF)A", "华夏磐泰混合A"
                            )
                            for p in pages
                        ]
                    flows, _ = _flows(
                        flow_pages,
                        allow_reversed=True,
                        allow_interleaved=True,
                        allow_period_variant=True,
                        additional_shares=inputs.get("additional_shares", ()),
                    )
                    own = next(r for r in flows if r["share"] == inputs["share_class"])
                    require(review["actual_effect_review_complete"] is True, "HOLDER_REVIEW_OPEN")
                    scores["liquidity"] = None if review["confirmed_pressure_effects"] else D(75)
                    facts["reported_redemption_shares"] = own["redeemed"]
                elif stage == "cost":
                    fee = fee_profile(
                        "\n".join(source.pages(inputs["refs"]["summary"])),
                        code=inputs["fund_code"],
                        share_class=inputs["share_class"],
                        legal_name=inputs["legal_name"],
                        supporting_text="\n".join(source.pages(inputs["refs"]["prospectus"])),
                        published_on=inputs["sources"][inputs["refs"]["summary"]]["published_on"],
                    )
                    cost = annual_fee_score(fee["fees"])
                    scores["cost"] = D(cost["score"])
                    facts["cost"] = cost
            except (KeyError, ValueError, StopIteration) as exc:
                errors[stage] = str(exc) or type(exc).__name__
        blocked = (
            not reviews_valid
            or any(
                e.get("status") == "UNRESOLVED" and e.get("severity") in {"MATERIAL", "SEVERE"}
                for e in inputs.get("reviewed_events", [])
            )
            or bool(inputs.get("holder_review", {}).get("confirmed_pressure_effects"))
            or inputs.get("manager_review", {}).get("governance_state") == "MATERIAL_UNRESOLVED"
            or inputs.get("allocation_exception_review", {}).get("state") == "UNRESOLVED"
        )
        facts["evidence_errors"] = errors
        facts["source_sha256"] = sorted(source.raw)
        return {
            "dimensions": {key: str(value) if value is not None else None for key, value in scores.items()},
            "facts": facts,
            "blocked": blocked,
        }
