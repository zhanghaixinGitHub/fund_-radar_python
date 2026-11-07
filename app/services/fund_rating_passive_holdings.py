"""按R5既有绝对复制匹配公式编译持仓维度，同行缺项不阻塞独立可算的持仓。

指数复制的发行人/细行业匹配均为100×(1-TV)，各占50%；这些子项本身
不使用同行中位数。本模块仍要求同日完整股票、实际ETF投资金额、目标权重
和逐证券身份/细行业原件，不能把其他维度未闭合误写成已有完整综合分。
"""

import re
from copy import deepcopy
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path

from app.services.fund_rating_direct_index import Originals, _mappings, _table
from app.services.fund_rating_direct_index_math import replication
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_passive_portfolio import lookthrough, portfolio
from app.services.fund_rating_r5 import number
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy

VERSION = "R5_PASSIVE_HOLDINGS_ORIGINALS_20261028_1"


def _identity(source: Originals, digest: str, code: str, legal_name: str) -> list[str]:
    """代码和法律全名必须出现在当期原件身份区，不能由目录名称代替。"""
    pages = source.pages(digest)
    header = normalized_text("\n".join(pages))[:15000]
    if code not in header or normalized_text(legal_name) not in header:
        raise ValueError("R5_HOLDINGS_REPORT_IDENTITY:" + code)
    return pages


def _ordered_name(name: str, block: str) -> bool:
    """名称/简称原字符可被列间文字打断，但不能补字、重排或无界匹配。"""
    if name in block:
        return True
    cursor = block.find(name[:4])
    if cursor < 0:
        return False
    cursor += 4
    for char in name[4:]:
        found = block.find(char, cursor, cursor + 81)
        if found < 0:
            return False
        cursor = found + 1
    return True


def _fund_holding(source: Originals, own_ref: str, target: dict, amount: str) -> dict:
    """核联接原件唯一目标ETF明细与全部基金投资余额精确相等。

    目标ETF身份在2.1.1独立核准，金额来自7.2/8.2完整目标基金表；不能用
    名称相似、前十大不完整覆盖或输入金额自行声称全基金投资已经穿透。
    """
    original_text = "\n".join(source.pages(own_ref))
    text = normalized_text(original_text)
    identity = re.search(r"2\.1\.1目标基金基本情况(.{1,2500}?)2\.2基金产品说明", text)
    if (not identity or target["fund_code"] not in identity[1]
            or normalized_text(target["legal_name"]) not in identity[1]):
        raise ValueError("R5_HOLDINGS_TARGET_RELATIONSHIP")
    patterns = [r"([78])\.2\s*期末投资目标基金明细(.*?)(?=\1\.3\s*)",
                # 部分联接中报把基金投资列在7.10.1，7.10只是父标题。
                # 仅允许这一个真实子节，仍以7.11封闭表格，不能吞入后续表。
                r"([78])\.10(?:\.1)?\s*期末按公允价值[^\n]*基金投资明细(.*?)(?=\1\.11\s*)",
                r"([78])\.12\.2\s*报告期末按公允价值[^\n]*所有基金投资明细(.*?)(?=\1\.13\s*)"]
    blocks = sorted([m for pattern in patterns for m in re.finditer(pattern, original_text, re.S)],
                    key=lambda m: m.start(), reverse=True)
    raw_table = next((m[2] for m in blocks if "人民币元" in normalized_text(m[2])
                      or "公允价值" in m[2] and "(元)" in normalized_text(m[2])), None)
    block = normalized_text(raw_table) if raw_table else None
    ranks = re.findall(r"(?m)^\s*(10|[1-9])\s+\S", raw_table or "")
    if not block or ranks != ["1"]:
        raise ValueError("R5_HOLDINGS_COMPLETE_TARGET_FUND_TABLE")
    # 报告表内名称可能折行，但去空白后的法律全名及原金额必须都实际存在。
    money = format(number(amount), ",.2f")
    legal = normalized_text(target["legal_name"])
    matched_name = _ordered_name(legal, block)
    abbreviation = None
    if not matched_name and target.get("source_hash"):
        # 明细表可只写真实基金简称。简称只能从同一代码/法律全名已核的
        # 目标ETF同期原报告字段提取，不能由调用者自行编造或按名称截断。
        target_pages = _identity(source, target["source_hash"], target["fund_code"], target["legal_name"])
        target_header = normalized_text("\n".join(target_pages))[:15000]
        short = re.search(r"基金简称(.{2,100}?)(?:场内简称|基金主代码|交易代码|基金扩位简称)", target_header)
        if short:
            abbreviation = short[1]
            matched_name = _ordered_name(abbreviation, block)
    fragments = [money]
    matched_money = bool(re.search(r"(?<![\d.,])" + re.escape(money), block))
    if not matched_money:
        # 仅拼接完整账簿金额已唯一识别的原数字片段；不填小数或借用比例。
        # 金额小数或最后一个千分位组可能换行；枚举该组内的真实断点，
        # 尾段至少保留两位，且必须是独立数字段，不能借比例小数或序号。
        alternatives = [(money[:-2], money[-2:])]
        if "," in money:
            prefix, tail = money.rsplit(",", 1)
            alternatives.extend((money[:split], money[split:])
                                for split in range(len(prefix) + 1, len(money) - 1))
        for prefix, tail in alternatives:
            if re.search(re.escape(prefix) + r".{0,160}?(?<![\d.,])" + re.escape(tail)
                         + r"(?![\d.,])", raw_table, re.S):
                fragments, matched_money = [prefix, tail], True
                break
    if not matched_name or not matched_money:
        raise ValueError("R5_HOLDINGS_FUND_AMOUNT_OR_NAME")
    return {"amount_cny": amount, "target_code": target["fund_code"], "excerpt": block,
            "table_abbreviation_verified_from_target_report": abbreviation,
            "reported_amount_fragments": fragments, "table_rank_rows": ranks,
            "scope": "SOLE_DISCLOSED_FUND_FAIR_VALUE_EQUALS_ALL_FUND_INVESTMENTS_EXACTLY"}


def calculate(inputs: dict, *, root: Path) -> dict:
    """每次核原字节，复用相同不可变输入的解析；不更新时间或评分其他维度。"""
    Originals(inputs, root)
    return deepcopy(_calculate_cached(canonical(inputs), str(root.resolve())))


@lru_cache(maxsize=24)
def _calculate_cached(serialized: str, root_name: str) -> dict:
    import json

    inputs, root = json.loads(serialized), Path(root_name)
    if inputs.get("version") != VERSION or inputs.get("basis") != PASSIVE_BASIS:
        raise ValueError("R5_HOLDINGS_COMPILER_SCOPE")
    if policy(PASSIVE_BASIS)["terms"]["holdings"] != [["issuer_replication", 50], ["industry_replication", 50]]:
        raise ValueError("R5_HOLDINGS_FROZEN_FORMULA_CHANGED")
    source = Originals(inputs, root)
    from app.services.fund_rating_security_disclosures import verified_securities

    identities = verified_securities(source, inputs)
    financial_refs = [inputs["primary_ref"], inputs["index"]["weights"]]
    if inputs.get("target_etf"):
        financial_refs.append(inputs["target_etf"]["source_hash"])
    if any(source.sources[digest].get("archive_kind") == "DERIVED" for digest in financial_refs):
        raise ValueError("R5_HOLDINGS_FINANCIAL_ORIGINAL_REQUIRED")
    own_pages = _identity(source, inputs["primary_ref"], inputs["fund_code"], inputs["legal_name"])
    if "增强" in inputs["legal_name"] or inputs["holding_date"] > inputs["as_of_date"]:
        raise ValueError("R5_HOLDINGS_PASSIVE_SCOPE_OR_FUTURE_HOLDINGS")
    if not inputs.get("target_etf") and not re.search(
        r"(?:本基金(?:为|属)被动|本基金采用完全复制|本基金主要采用完全复制)", normalized_text("\n".join(own_pages))
    ):
        raise ValueError("R5_HOLDINGS_PASSIVE_MANDATE_REQUIRED")
    own = portfolio(own_pages, holding_date=inputs["holding_date"], verified_securities=identities)
    relation = None
    if inputs.get("target_etf"):
        target = inputs["target_etf"]
        pages = _identity(source, target["source_hash"], target["fund_code"], target["legal_name"])
        underlying = portfolio(pages, holding_date=inputs["holding_date"], verified_securities=identities)
        relation = _fund_holding(source, inputs["primary_ref"], target, own["balance"]["fund_amount"])
        stocks = lookthrough(own, underlying, fund_holding_cny=relation["amount_cny"])
    else:
        if number(own["balance"]["fund_amount"]) != 0:
            raise ValueError("R5_HOLDINGS_UNKNOWN_FUND_LOOKTHROUGH")
        stocks = own
    index = inputs["index"]
    metadata = [row for row in _table(source.json(index["metadata"])) if row["ts_code"] == index["code"]]
    if (len(metadata) != 1 or metadata[0]["ts_code"] != index["code"]
            or metadata[0]["name"] != index["name"]
            or metadata[0]["publisher"] not in {"深圳证券信息有限公司", "中证指数有限公司",
                                                "中证指数有限公司、上海证券交易所"}):
        raise ValueError("R5_HOLDINGS_INDEX_IDENTITY")
    # 目标在本法律产品当期合同中明确出现，不能只拿当前指数目录来绑定。
    contract_text = normalized_text("\n".join(own_pages))
    contract_name = index.get("contract_name", index["name"])
    if contract_name + "指数" not in contract_text:
        raise ValueError("R5_HOLDINGS_CONTRACT_INDEX")
    identity_claim = index.get("identity_claim")
    if contract_name != index["name"]:
        if not identity_claim:
            raise ValueError("R5_HOLDINGS_INDEX_ALIAS_PROOF_REQUIRED")
        source.claim(identity_claim)
        if not all(value in normalized_text(identity_claim["quote"]) for value in (
            contract_name + "指数", index["code"].split(".")[0], index["name"]
        )):
            raise ValueError("R5_HOLDINGS_INDEX_ALIAS_CONFLICT")
    weights = {}
    for row in _table(source.json(index["weights"])):
        day = row["trade_date"]
        day = day if "-" in day else day[:4] + "-" + day[4:6] + "-" + day[6:]
        if row["index_code"] != index["code"] or day != inputs["holding_date"] or row["con_code"] in weights:
            raise ValueError("R5_HOLDINGS_INDEX_WEIGHT_IDENTITY_DATE_OR_DUPLICATE")
        weight = number(row["weight"]) / 100
        if not 0 <= weight <= 1:
            raise ValueError("R5_HOLDINGS_INDEX_WEIGHT_DOMAIN")
        weights[row["con_code"]] = str(weight)
    if not weights or abs(sum(map(number, weights.values())) - 1) > Decimal(".000005") * len(weights):
        raise ValueError("R5_HOLDINGS_INDEX_WEIGHT_TOTAL")
    securities = set(weights) | {row["stock_code"] for row in stocks["positions"]}
    issuers, industries = _mappings(source, inputs, securities)
    values = replication(stocks["positions"], stocks["stock_total"], weights, issuers, industries)
    with localcontext() as context:
        context.prec = 60
        sub = {key: 100 * (1-number(values[key])) for key in ("issuer_replication", "industry_replication")}
        score = sum(sub.values()) / 2
    return {"score": str(score), "subscores": {key: str(value) for key, value in sub.items()},
            "raw_total_variation": values, "source_refs": sorted(source.sources),
            "fund_code": inputs["fund_code"], "as_of_date": inputs["as_of_date"],
            "holding_date": inputs["holding_date"], "stock_count": len(stocks["positions"]),
            "stock_risk_budget_cny": stocks["stock_total"], "target_fund_proof": relation,
            "nonsecurity_substitution_increment_cny": own["nonsecurity_substitution_increment"],
            "scope": "R5_STOCK_INTERNAL_REPLICATION_ONLY_NO_LIQUIDITY_OR_FULL_SCORE",
            "security_disclosures": identities}
