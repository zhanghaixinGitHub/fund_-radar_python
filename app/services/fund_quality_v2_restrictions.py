"""复用既有受限表逐行核对算法，独立于治理、策略和持有人字段解析。

只提供公开披露受限证券金额，调用方还须证明全资产范围及净资产对账。
本模块不改变现行正式评级适配，也不将未识别的表格当作零。
"""

import hashlib
import re
from collections import Counter
from decimal import Decimal

from app.services.fund_rating_public_market import compact, section
from app.services.fund_rating_rules import decimal


def restriction_page_header_view(pages: list[str]) -> tuple[str, dict]:
    """只移除重复报告页眉、明确页码及连续页序验证的首行页码，恢复分页原句。

    不删除一般正文或按含“无”猜结论。每条删除保留页号、行号及原文；
    证券表内数字不恢复或改写，此视图与原字节分别记录摘要。
    """
    page_lines = [page.splitlines() for page in pages]
    titles = Counter(line.strip() for lines in page_lines for line in lines)
    repeated = {
        line
        for line, count in titles.items()
        if count >= 2 and re.fullmatch(r"[^\d\n]{2,100}20\d{2}\s*年\s*(?:中期|年度)报告", line)
    }
    # 某些发行人把页码以裸整数写在每页首个抽取行。只有报告身份存在，且
    # 同一数字=实际页序的规则在相邻三页连续成立，才将对应位置认作页码。
    # 页面内部股数/金额、首行但不满足连续页序的整数均原样保留。
    first_lines = {i: next(((n, line.strip()) for n, line in enumerate(lines, 1) if line.strip()), (0, ""))
                   for i, lines in enumerate(page_lines, 1)}
    page_numbers = {i for i, (_, value) in first_lines.items() if value == str(i)}
    report_identity = re.search(r"(?:20\d{2}|[〇零一二三四五六七八九0]{4})年(?:年度|中期)报告",
                                compact("\n".join(pages[:2])))
    sequential = {i for i in page_numbers if report_identity and any(
        {start, start + 1, start + 2} <= page_numbers for start in (i - 2, i - 1, i)
    )}
    view, edits = [], []
    for page_number, page in enumerate(pages, 1):
        lines = []
        for number, line in enumerate(page.splitlines(), 1):
            stripped = line.strip()
            is_sequential_page = page_number in sequential and number == first_lines[page_number][0]
            if stripped in repeated or re.fullmatch(r"第\s*\d+\s*页\s*共\s*\d+\s*页", stripped) or is_sequential_page:
                edits.append({"page": page_number, "line": number, "removed_header": line,
                              **({"position_rule": "FIRST_NONEMPTY_LINE_EQUALS_PHYSICAL_PAGE_IN_CONSECUTIVE_RUN"}
                                 if is_sequential_page else {})})
            else:
                lines.append(line)
        view.append("\n".join(lines))
    original, text = "\n".join(pages), "\n".join(view)
    return text, {
        "removed_headers": edits,
        "original_text_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "parser_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scope": "REPEATED_REPORT_HEADERS_EXPLICIT_OR_VERIFIED_SEQUENTIAL_PAGE_NUMBERS_ONLY",
        "sequential_bare_page_numbers": sorted(sequential),
    }


def full_asset_restriction_scope(text: str) -> re.Match[str] | None:
    """只接纳明确覆盖其余资产的原文，不把一般风险管理描述当作范围证明。

    有些原件按附注标题或“附注中列示”引用受限表，并未给出数字章号；
    调用者仍必须逐项核完同报告全部受限表，不能由这句声明推导受限金额。
    """
    return re.search(
        r"除[^。]{0,180}[567]\.4\.12[^。]{0,180}(?:其余|其他)[^。]{0,40}变现|"
        r"本期末本基金的其他资产均能及时变现|"
        r"除附注[567]\.4\.12所披露的流通受限不能自由转让的基金资产外，"
        r"本基金未持有其他有[^。]{0,90}重大流动性风险的投资品种|"
        r"除在附注(?:[“\"]期末本基金持有的流通受限证券[”\"])?中列示的"
        r"部分基金资产流通暂时受限制外[（(]如有[）)][,，]其余均能及时变现|"
        r"除本报告[“\"]?期末本基金持有的流通受限证券[”\"]?章节中所列示券种"
        r"流通暂时受限制不能自由转让外[,，](?:其余|本基金本报告期末的其他资产)均能及时变现",
        compact(text),
    )


def restricted_numeric_view(block: str) -> str:
    """恢复原文已存在的金额小数尾数及千分位股数折行，绝不补数字。

    例如2,919,70换行8只能拼成2,919,708；若尾数组合不能满足完整千分位
    格式则直接拒收。不能只把最后一行8当成完整股数。
    """
    numeric = re.sub(r"(?<=\d)(?=[\r\n])\s+(?=\.\d{2}\b)", "", block)
    numeric = re.sub(r"(\d+\.\d)\s*\r?\n\s*(\d)(?=\s|$)", r"\1\2", numeric)
    numeric = re.sub(r"(\d+\.)\s*\r?\n\s*(\d{2})(?=\s|$)", r"\1\2", numeric)

    def join_money(match):
        candidate = match[1] + match[2]
        # 只有未完整的千分位组能唯一接回金额；普通整数换行接金额可能是
        # “数量 / 成本”两列，不能拼成新金额后把单价误认成股数。
        if not re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d{2}", candidate):
            raise ValueError("QUALITY_RESTRICTED_MONEY_WRAP_AMBIGUOUS")
        return candidate

    # 跨三行的大金额须从末行向前恢复，例如125,0 / 40,00 / 0.00。
    # 每轮只连接原有片段；循环上限拒绝异常长碎片，不截断或补齐数字。
    for _ in range(8):
        joined = re.sub(
            r"(?<![\d,.])(\d{1,3}(?:,\d{3})*,\d{0,2})[ \t]*\r?\n[ \t]*(\d{1,3}(?:,\d{3})*\.\d{2})(?=\s|$)",
            join_money,
            numeric,
        )
        if joined == numeric:
            break
        numeric = joined
    else:
        raise ValueError("QUALITY_RESTRICTED_MONEY_FRAGMENT_LIMIT")

    def join_integer(match):
        candidate = match[1] + match[2]
        if not re.fullmatch(r"\d{1,3}(?:,\d{3})+", candidate):
            raise ValueError("QUALITY_RESTRICTED_QUANTITY_WRAP_AMBIGUOUS")
        return candidate

    numeric = re.sub(
        r"(?<![\d,.])(\d{1,3}(?:,\d{3})*,\d{1,2})[ \t]*\r?\n[ \t]*(\d{1,2})(?=\s|$)",
        join_integer,
        numeric,
    )
    if compact(numeric) != compact(block):
        raise ValueError("QUALITY_RESTRICTED_VIEW_CHANGED_ORIGINAL_DIGITS")
    return numeric


def _serial_stock_rows(stock_part: str, positions: list[dict]) -> list[dict]:
    """先读顺序明确的股票行；缺行、含糊金额或越界持仓均不能补值。"""
    blocks = re.split(r"(?m)(?=^[ \t]*\d{5,6}(?:[ \t]+|\r?$))", stock_part)[1:]
    rows = []
    by_code = {p["stock_code"].split(".")[0]: p for p in positions}
    for block in blocks:
        prefix = re.match(r"\s*(\d{5,6})", block)[1]
        known = [k for k in by_code if k.startswith(prefix)]
        if len(prefix) == 5:
            immediate = re.match(r"\s*\d{5}[ \t]*\r?\n[ \t]*([0-9])(?:\s|$)", block)
            if immediate and prefix + immediate[1] in known:
                known = [prefix + immediate[1]]
            tails = re.findall(r"(?m)^\s*([0-9])(?:[ \t]+[\u4e00-\u9fff]|[ \t]*$)", block)
            candidates = [prefix + digit for digit in tails if prefix + digit in known]
            if len(known) > 1 and len(set(candidates)) == 1:
                known = list(set(candidates))
            if len(known) > 1:
                # 跨页末尾一位代码不能从年份取值；再用该行已披露的股票名称独立定位。
                names = [
                    k
                    for k in known
                    if re.search(".*?".join(map(re.escape, compact(by_code[k]["stock_name"]))), block, re.S)
                ]
                if len(names) == 1:
                    known = names
        if len(known) != 1:
            raise ValueError("RATING_MARKET_RESTRICTED_IDENTITY:" + prefix)
        # 仅恢复数值小数点后明确缺位的折行，保持原行与恢复后的文本用于核验。
        numeric = restricted_numeric_view(block)
        # 备注整列为空时仍可读已经明确的数量/成本/估值；仅接受表行末尾三元组，
        # 不从空备注推算金额。章号及“注：”之后是下一表或脚注，不能串入该行。
        numeric = re.split(r"(?m)^\s*(?:注[：:]|[56]\.4\.12\.\d+(?:\.\d+)*(?:\s|$))", numeric, maxsplit=1)[0]
        # 部分转载把两位单价和千分位股数粘连；数量×单价须同时与成本、估值金额一致。
        glued = re.search(
            r"([\d,]+\.\d{2})\s+([\d,]+\.\d{2})([1-9]\d{0,2}(?:,\d{3})+)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*[-－—]",
            numeric,
        )
        if glued:
            issue, price, shares, cost_amount, fair_amount = (decimal(v.replace(",", "")) for v in glued.groups())
            if issue * shares != cost_amount or price * shares != fair_amount:
                raise ValueError("RATING_MARKET_GLUED_PRICE_QUANTITY_CONFLICT")
            numeric = (
                numeric[: glued.start()]
                + glued[0].replace(glued[2] + glued[3], glued[2] + " " + glued[3], 1)
                + numeric[glued.end() :]
            )
        values = re.findall(
            r"(?<![\d,.])([\d,]+(?:\.\d{2})?)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*[-－—](?:\s|$)", numeric
        )
        if not values:
            values = re.findall(r"(?<![\d,.])([\d,]+(?:\.\d{2})?)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*\Z", numeric)
        if len(values) != 1:
            raise ValueError("RATING_MARKET_RESTRICTED_VALUE:" + prefix)
        qty, cost, fair = (decimal(v.replace(",", "")) for v in values[0])
        if qty <= 0 or min(cost, fair) < 0:
            raise ValueError("RATING_MARKET_RESTRICTED_VALUE_DOMAIN")
        if qty > decimal(by_code[known[0]]["quantity_shares"]) or fair > decimal(by_code[known[0]]["market_value_cny"]):
            raise ValueError(
                "RATING_MARKET_RESTRICTED_RANGE:"
                + known[0]
                + ":"
                + str(qty)
                + ":"
                + by_code[known[0]]["quantity_shares"]
                + ":"
                + str(fair)
                + ":"
                + by_code[known[0]]["market_value_cny"]
            )
        rows.append({"code": known[0], "quantity": str(qty), "cost": str(cost), "amount": str(fair)})
    if not blocks and not re.search(
        r"(?:未持有|无).*?(?:认购新发|流通受限证券)|流通受限证券(?:注[：:])?无[。.]?$", compact(stock_part)
    ):
        raise ValueError("RATING_MARKET_RESTRICTED_TABLE_MISSING")
    return rows


def restricted_securities(
    pages: list[str], positions: list[dict], bond_amount: str | None = None, *, scope: str = "ALL_ASSETS"
) -> dict:
    """默认核对全部受限资产；显式股票研究范围仅核股票，不把未核债券记为零。

    ALL_ASSETS沿用现行V2受限比例的严格边界。STOCKS_ONLY仅供独立股票
    成交候选，返回明确范围、未评估项目及原文，不能生成全资产受限比例。
    """
    if scope not in {"ALL_ASSETS", "STOCKS_ONLY"}:
        raise ValueError("QUALITY_RESTRICTION_SCOPE_UNSUPPORTED")
    text, header_view = restriction_page_header_view(pages)
    suspended = section(text, r"[56]\.4\.12\.2\s*期末持有的暂时停牌", r"[56]\.4\.12\.3")
    lending = section(text, r"[56]\.4\.12\.4\s*期末参与转融通", r"[56]\.4\.13")
    if not re.search(
        r"(?:未持有(?:期末)?参与|无参与).*?转融通证券出借业务的(?:融出)?证券|无因参与转融通证券出借业务而出借的证券|^.*?期末参与转融通证券出借业务的证券(?:注[：:])?无[。.]?$",
        compact(lending),
    ):
        raise ValueError("RATING_MARKET_LENDING_UNVERIFIED")
    pledged = section(text, r"[56]\.4\.12\.3\s*期末债券正回购", r"[56]\.4\.12\.4")
    pledge = None
    if scope == "ALL_ASSETS":
        from app.services.fund_quality_v2_pledged_bonds import pledged_bond_table

        pledge = pledged_bond_table(restricted_numeric_view(pledged), bond_amount=bond_amount)
    restricted = section(text, r"[56]\.4\.12\.1\s*因认购", r"[56]\.4\.12\.2")
    stock_part, separator, bond_part = restricted.partition("受限证券类别：债券")
    stock_layout = None
    try:
        rows = _serial_stock_rows(stock_part, positions)
    except ValueError as serial_error:
        recoverable = (
            "RATING_MARKET_RESTRICTED_VALUE:",
            "RATING_MARKET_RESTRICTED_IDENTITY:",
            "RATING_MARKET_RESTRICTED_TABLE_MISSING",
            "QUALITY_RESTRICTED_MONEY_WRAP_AMBIGUOUS",
            "QUALITY_RESTRICTED_QUANTITY_WRAP_AMBIGUOUS",
        )
        if not str(serial_error).startswith(recoverable) or not any(
            all(word in line for word in ("代码", "名称", "价格")) for line in stock_part.splitlines()
        ):
            raise
        from app.services.fund_quality_v2_restricted_layout import fixed_width_stock_restrictions

        try:
            layout = fixed_width_stock_restrictions(stock_part, positions)
        except ValueError as layout_error:
            raise ValueError(str(serial_error) + "; FIXED_LAYOUT:" + str(layout_error)) from layout_error
        # 替换的是整张股票表；不把顺序解析的半表与恢复表相加。
        rows = layout["rows"]
        stock_layout = {k: value for k, value in layout.items() if k != "rows"}
        stock_layout["serial_rejection"] = str(serial_error)
    total = sum((decimal(row["amount"]) for row in rows), Decimal(0))
    by_code = {p["stock_code"].split(".")[0]: p for p in positions}
    if not re.search(
        r"(?:未持有|无)暂时停牌等流通受限(?:的)?股票|期末持有的暂时停牌等流通受限股票(?:注[：:])?无[。.]?$",
        compact(suspended),
    ):
        halt_rows = re.findall(r"(?m)^\s*(\d{6})\s+(.+?)(?=^\s*\d{6}\s|\Z)", suspended, re.S)
        if not halt_rows:
            raise ValueError("RATING_MARKET_SUSPENDED_TABLE_UNVERIFIED")
        for code, block in halt_rows:
            value = re.findall(r"([\d,]+)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*[-－—](?:\s|$)", block)
            if code not in by_code or len(value) != 1:
                raise ValueError("RATING_MARKET_SUSPENDED_ROW_UNVERIFIED")
            qty, cost, fair = (decimal(v.replace(",", "")) for v in value[0])
            rows.append({"code": code, "quantity": str(qty), "cost": str(cost), "amount": str(fair)})
            total += fair
    # 同一发行人可能有多批限售股份；逐批合法仍不代表合计不会超过完整持仓。
    for code, position in by_code.items():
        issuer_rows = [r for r in rows if r["code"] == code]
        if sum(decimal(r["quantity"]) for r in issuer_rows) > decimal(position["quantity_shares"]) or sum(
            decimal(r["amount"]) for r in issuer_rows
        ) > decimal(position["market_value_cny"]):
            raise ValueError("RATING_MARKET_RESTRICTED_ISSUER_SUM")
    # 新债未上市也属于披露的受限资产。逐行读取股/债各自的表格，不把债券代码
    # 去股票名单中匹配；债券合计还须被同报告资产负债表的全部债券金额覆盖。
    empty_bond_table = None
    if separator and scope == "ALL_ASSETS":
        bond_blocks = re.split(r"(?m)(?=^[ \t]*\d{6}(?:[ \t]+|\r?$))", bond_part)[1:]
        bond_rows = []
        for block in bond_blocks:
            code = re.match(r"\s*(\d{6})", block)[1]
            values = re.findall(
                r"(?<![\d,.])([\d,]+(?:\.\d{2})?)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*[-－—](?:\s|$)",
                restricted_numeric_view(block),
            )
            if len(values) != 1:
                raise ValueError("QUALITY_RESTRICTED_BOND_ROW_UNRESOLVED:" + code)
            qty, cost, fair = (decimal(v.replace(",", "")) for v in values[0])
            if qty <= 0 or min(cost, fair) < 0:
                raise ValueError("QUALITY_RESTRICTED_BOND_VALUE_INVALID")
            bond_rows.append(
                {
                    "code": code,
                    "quantity": str(qty),
                    "cost": str(cost),
                    "amount": str(fair),
                    "asset_type": "BOND",
                    "source_row": block[:1800],
                }
            )
        if not bond_rows and not re.search(r"(?:未持有|无).*?(?:债券|证券)", compact(bond_part)):
            # 完整空表与缺表不同：必须保留所有列标题、无任何证券条目，且五个
            # 数值列都明确以横杠表示无项目。只有几个横杠或少一列仍不能认作0。
            pattern = (
                r"证券代码证券名称成功认购日受限期流通受限类型认购价格期末估值单价"
                r"数量[（(]单位[:：]张[）)]期末成本总额期末估值总额备注[-－—]{5}"
            )
            if not re.fullmatch(pattern, compact(bond_part)):
                raise ValueError("QUALITY_RESTRICTED_BOND_TABLE_UNVERIFIED")
            empty_bond_table = {
                "basis": "COMPLETE_EMPTY_BOND_TABLE_FIVE_NUMERIC_DASHES_NO_SECURITY_ROW",
                "numeric_columns": ["认购价格", "期末估值单价", "数量（张）", "期末成本总额", "期末估值总额"],
                "excerpt": bond_part,
            }
        bond_total = sum(decimal(r["amount"]) for r in bond_rows)
        if bond_rows and (bond_amount is None or bond_total > decimal(bond_amount)):
            raise ValueError("QUALITY_RESTRICTED_BOND_BALANCE_CONFLICT")
        rows.extend(bond_rows)
        total += bond_total
    if pledge:
        restricted_bonds = [r for r in rows if r.get("asset_type") == "BOND"]
        if {r["code"] for r in restricted_bonds} & {r["code"] for r in pledge["rows"]}:
            raise ValueError("QUALITY_RESTRICTED_BOND_PLEDGE_OVERLAP_NOT_RESOLVED")
        bond_total = sum(decimal(r["amount"]) for r in restricted_bonds) + decimal(pledge["amount_cny"])
        if bond_total and (bond_amount is None or bond_total > decimal(bond_amount)):
            raise ValueError("QUALITY_RESTRICTED_ALL_BONDS_EXCEED_FULL_BOND_BALANCE")
        rows.extend(pledge["rows"])
        total += decimal(pledge["amount_cny"])
    result = {"restricted_amount": str(total), "rows": rows, "parser_header_view": header_view}
    if stock_layout:
        result["stock_layout_evidence"] = stock_layout
    if pledge:
        result["pledged_bond_evidence"] = pledge
    if empty_bond_table:
        result["empty_bond_table_evidence"] = empty_bond_table
    if scope == "STOCKS_ONLY":
        result.update(
            scope="DISCLOSED_STOCK_RESTRICTIONS_ONLY_NOT_ALL_ASSETS",
            not_assessed=["受限债券", "债券正回购质押", "其他非股票资产的受限金额"],
            excluded_bond_table_excerpt=bond_part[:1800] if separator else None,
            excluded_pledged_bonds_excerpt=pledged[:1800],
        )
    return result
