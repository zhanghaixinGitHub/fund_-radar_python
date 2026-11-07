"""R5指数股票与联接预算的原件账簿，不把非证券替代款分摊给股票。"""

import re
from collections import defaultdict
from decimal import Decimal, localcontext

from app.services.fund_quality_v2_asset_structure import gross_asset_book
from app.services.fund_quality_v2_balance_proofs import strict_balance_values
from app.services.fund_quality_v2_restrictions import full_asset_restriction_scope, restricted_securities
from app.services.fund_quality_v2_stock_book import _stock_parser_view, periodic_financial_view
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_public_stock_tables import parse, parse_disclosed
from app.services.fund_rating_r5 import number


def portfolio(pages: list[str], *, holding_date: str, include_restrictions: bool = False,
              verified_securities: dict | None = None) -> dict:
    """同日完整股票、账面股票及替代款独立对账；零股票仅取严格账簿唯一值。

    全资产受限另按原附注核算。同行复制和成交只取完整股票表，不强求与这些
    原始观察无关的其他维度解析也先成功；目标完整流动性必须单独闭合受限。
    """
    view, trace = periodic_financial_view(pages)
    proof = strict_balance_values(view, observed_on=holding_date)
    book = proof["point_book"]
    if book is None:
        raise ValueError("R5_PASSIVE_BALANCE_NOT_IDENTIFIED")
    if number(book["stock_amount"]) == 0:
        positions, stock_total, increment = [], "0", "0"
    else:
        disclosed = parse_disclosed(view, book["stock_amount"], book["net_assets"],
                                    verified_securities=verified_securities)
        stock_total = disclosed["stock_value_cny"]
        positions = parse(view, stock_total, book["net_assets"], require_industry=True,
                          verified_securities=verified_securities)["holdings"]
        increment = disclosed.get("substitution_increment_cny", "0")
    result = {"balance": book, "balance_proof": proof, "positions": positions,
              "stock_total": stock_total, "nonsecurity_substitution_increment": increment, "chapter_view": trace}
    if include_restrictions:
        scope = full_asset_restriction_scope("\n".join(pages))
        if not scope:
            # 有的报告明确将全部流通受限事项指向6.4.12，而不保证其他资产
            # “及时变现”。受限比例评价法定限制，不能把立即变现保证误加为
            # 必要措辞；此适配额外要求非股票资产完整闭合为已核现金/国债。
            scope = re.search(
                r"本基金所持部分证券在证券交易所上市[，,]其余亦可在银行间同业市场交易[，,]"
                r"部分基金资产流通暂时受限制不能自由转让的情况参见附注[567]\.4\.12",
                normalized_text("\n".join(pages)),
            )
            if not scope:
                raise ValueError("R5_PASSIVE_FULL_ASSET_RESTRICTION_SCOPE_REQUIRED")
            # 先核原文是否真的覆盖全部资产，再解析作为附加条件的非股票账簿。
            # 否则零股票联接或含非证券替代款的ETF会先报股票表错误，掩盖
            # 实际尚缺的全资产范围证明，误导后续补件方向。
            closed_nonstock = gross_asset_book(view)
            result["closed_nonstock_scope"] = closed_nonstock
        restriction_view, edits = _stock_parser_view(view, positions)
        result["restricted_assets"] = restricted_securities(
            restriction_view, positions, book["bond_amount"], scope="ALL_ASSETS"
        )
        result["restriction_scope_excerpt"] = scope[0]
        result["restriction_layout_edits"] = edits
    return result


def lookthrough(link: dict, target_etf: dict, *, fund_holding_cny: str) -> dict:
    """以本产品实际持有ETF公允价值/ETF净资产估算底层股票风险预算。

    先核本联接全部基金投资恰等于唯一目标ETF，ETF自身不能再含未知基金；
    证券明细排除有原文依据的非证券替代款。结果是风险预算，不伪称股票实持。
    """
    with localcontext() as context:
        context.prec = 70
        holding, net = number(fund_holding_cny), number(target_etf["balance"]["net_assets"])
        if holding <= 0 or net <= 0 or holding != number(link["balance"]["fund_amount"]):
            raise ValueError("R5_ETF_FULL_FUND_HOLDING_NOT_RECONCILED")
        if number(target_etf["balance"]["fund_amount"]) != 0:
            raise ValueError("R5_ETF_FURTHER_LOOKTHROUGH_REQUIRED")
        scale, amounts = holding / net, defaultdict(Decimal)
        for row in link["positions"]:
            amounts[row["stock_code"]] += number(row["market_value_cny"])
        for row in target_etf["positions"]:
            amounts[row["stock_code"]] += scale * number(row["market_value_cny"])
        positions = [{"stock_code": key, "market_value_cny": str(amounts[key])} for key in sorted(amounts)]
        # 合计是已核证券预算逐项之和，尾数保留；不把替代款差额补到任一证券。
        total = sum(number(row["market_value_cny"]) for row in positions)
        return {"positions": positions, "stock_total": str(total), "etf_scale": str(scale),
                "scope": "DISCLOSED_ETF_FAIR_VALUE_STOCK_RISK_BUDGET_NOT_DIRECT_OWNERSHIP"}
