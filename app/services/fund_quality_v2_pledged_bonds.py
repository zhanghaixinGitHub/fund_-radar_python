"""只从完整质押债券明细重建受限金额，融资款余额不能替代抵押资产估值。"""

import re
from datetime import date
from decimal import Decimal

from app.services.fund_rating_public_market import compact
from app.services.fund_rating_rules import decimal


class PledgedBondIncomplete(ValueError):
    """保留已经重建的市场明细；异常绝不携带可当作全资产指标的总值。"""

    def __init__(self, reason: str, *, rows: list[dict], markets: list[dict], unresolved_market: str):
        super().__init__(reason)
        self.evidence = {
            "known_rows": list(rows),
            "known_markets": list(markets),
            "unresolved_market": unresolved_market,
            "scope": "PARTIAL_PLEDGE_FACTS_NOT_COMPLETE_RESTRICTED_METRIC",
        }


def pledged_bond_table(section: str, *, bond_amount: str | None) -> dict:
    """核对银行间及交易所两个市场，返回债券数量、估值、到期日与原文。

    参数section是同一报告6/5.4.12.3的完整小节，bond_amount为同日财报债券
    余额（人民币元）。明确未质押时为0；正质押必须逐行及合计守恒。融资款、
    标准券折算要求及未来还款日期均不能推算实际质押数量或资产金额。
    """
    starts = list(re.finditer(
        r"[56]\.4\.12\.3\.(?P<number>[12])\s*(?P<numbered_title>银行间市场债券正回购|交易所市场债券正回购)"
        r"|(?m:^[ \t]*(?P<standalone_title>银行间市场债券正回购|交易所市场债券正回购)[ \t]*$)",
        section,
    ))
    headings = []
    for match in starts:
        title = match["numbered_title"] or match["standalone_title"]
        expected = "1" if title == "银行间市场债券正回购" else "2"
        if match["number"] and match["number"] != expected:
            raise ValueError("QUALITY_PLEDGED_BONDS_NUMBER_AND_MARKET_CONFLICT")
        if match["standalone_title"] and not re.search(
            r"[56]\.4\.12\.3\s*期末债券正回购交易中作为抵押的债券", section[:match.start()]
        ):
            raise ValueError("QUALITY_PLEDGED_BONDS_PARENT_SECTION_UNVERIFIED")
        headings.append(expected)
    if headings != ["1", "2"]:
        raise ValueError("QUALITY_PLEDGED_BONDS_MARKET_SECTIONS_UNVERIFIED")
    markets, all_rows = [], []
    for number, start in enumerate(starts):
        body = section[start.end() : starts[number + 1].start() if number + 1 < len(starts) else len(section)]
        normalized = compact(body)
        market = "INTERBANK" if headings[number] == "1" else "EXCHANGE"
        heading_evidence = {"heading_excerpt": start[0], "heading_kind": "NUMBERED"
                            if start["number"] else "STANDALONE_TITLE_WITHIN_PARENT_PLEDGE_SECTION"}
        negative = re.search(
            r"(?:未持有(?:因)?.*?作为(?:抵押|质押)的债券|无因.*?债券正回购交易而抵押的债券|"
            r"无(?:因)?从事.*?形成的卖出回购(?:金融资产|证券)款余额|"
            r"无(?:因)?从事.*?形成的卖出回购(?:金融资产|证券)款，无抵押债券|"
            r"卖出回购(?:金融资产|证券)款余额为0(?:\.0+)?(?:元)?，无抵押债券|"
            r"无(?:银行间|交易所)市场债券正回购(?:余额)?，因此没有.*?作为抵押的债券|^无[。.]?$)",
            normalized,
        )
        positive_amounts = [decimal(v.replace(",", "")) for v in re.findall(r"(?<![\d,.])[\d,]+\.\d{2}(?!\d)", body)]
        if negative:
            integer_balances = [decimal(value.replace(",", "")) for value in re.findall(
                r"余额(?:为|[:：])?([\d,]+)元", normalized
            )]
            security_rows = re.search(r"(?m)^[ \t]*(?:\d{6}|\d{9})\s+", body)
            if any(v > 0 for v in positive_amounts + integer_balances) or security_rows:
                raise ValueError("QUALITY_PLEDGED_BONDS_NEGATIVE_AND_POSITIVE_CONFLICT:" + market)
            markets.append({"market": market, "amount_cny": "0", "basis": "EXPLICIT_NO_PLEDGED_BONDS",
                            "excerpt": body, **heading_evidence})
            continue
        if "债券代码" not in normalized or "期末估值总额" not in normalized:
            # 正回购融资余额和标准券保证条件只说明有负债，不说明抵押资产价值。
            reason = (
                "COLLATERAL_AMOUNT_NOT_DISCLOSED"
                if any(v > 0 for v in positive_amounts)
                else "MARKET_STATEMENT_UNPARSED"
            )
            raise PledgedBondIncomplete(
                "QUALITY_PLEDGED_BONDS_" + reason + ":" + market,
                rows=all_rows,
                markets=markets,
                unresolved_market=market,
            )
        if "单位：人民币元" not in normalized or not re.search(r"数量[（(]张[）)]", normalized):
            raise ValueError("QUALITY_PLEDGED_BONDS_UNIT_UNVERIFIED:" + market)
        money = r"[\d,]+\.\d{2}"
        pattern = (
            r"(?m)^[ \t]*(\d{6}|\d{9})\s+(.{1,140}?)\s+(20\d{2}-\d{2}-\d{2})\s+"
            rf"({money})\s+({money})\s+({money})(?=\s|$)"
        )
        rows = []
        for match in re.finditer(pattern, body, re.S):
            code, name, maturity, price, quantity, amount = match.groups()
            date.fromisoformat(maturity)
            values = [decimal(v.replace(",", "")) for v in (price, quantity, amount)]
            if min(values) <= 0:
                raise ValueError("QUALITY_PLEDGED_BONDS_VALUE_INVALID:" + market)
            rows.append(
                {
                    "code": code,
                    "name": compact(name),
                    "market": market,
                    "maturity_date": maturity,
                    "unit_price_cny": str(values[0]),
                    "quantity": str(values[1]),
                    "quantity_unit": "bond",
                    "amount": str(values[2]),
                    "asset_type": "PLEDGED_BOND",
                    "source_row": match[0].strip(),
                }
            )
        totals = re.findall(rf"(?m)^[ \t]*合计\s+({money})\s+({money})(?=\s|$)", body)
        if not rows or len(rows) > 5000 or len(totals) != 1:
            raise ValueError("QUALITY_PLEDGED_BONDS_ROWS_OR_TOTAL_UNPARSED:" + market)
        if len({(row["code"], row["maturity_date"]) for row in rows}) != len(rows):
            raise ValueError("QUALITY_PLEDGED_BONDS_DUPLICATE_ROW:" + market)
        quantity, amount = (decimal(v.replace(",", "")) for v in totals[0])
        if (
            sum(decimal(row["quantity"]) for row in rows) != quantity
            or sum(decimal(row["amount"]) for row in rows) != amount
        ):
            raise ValueError("QUALITY_PLEDGED_BONDS_TOTAL_CONFLICT:" + market)
        markets.append(
            {
                "market": market,
                "amount_cny": str(amount),
                "quantity_bonds": str(quantity),
                "basis": "FULL_COLLATERAL_ROWS_AND_TOTAL",
                "excerpt": body,
                **heading_evidence,
            }
        )
        all_rows.extend(rows)
    total = sum((decimal(row["amount"]) for row in all_rows), Decimal(0))
    if total and (bond_amount is None or total > decimal(bond_amount)):
        raise ValueError("QUALITY_PLEDGED_BONDS_EXCEED_FULL_BOND_BALANCE")
    return {
        "amount_cny": str(total),
        "rows": all_rows,
        "markets": markets,
        "scope": "ACTUAL_PLEDGED_BOND_FAIR_VALUES_NOT_FINANCING_BALANCES",
    }
