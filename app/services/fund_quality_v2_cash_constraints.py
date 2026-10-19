"""公开现金约束的可测量性研究；不替换RAW4纯现金目标，不进入评分。

只识别同一完整财务报告中明示的现金加短政府债下限，并重读银行存款附注。
未披露政府债剩余期限、实际期货保证金或现金受限分布时保留区间；区间端点
是数学界限，不是给未知原始数据填零。披露日结果也不能证明逐日合同合规。
"""

import re
import unicodedata
from datetime import date
from decimal import Decimal, localcontext

from app.services.fund_quality_v2_accounting import accounting_observations
from app.services.fund_quality_v2_cash_clauses import cash_basket_clause, cash_clause_page_view
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_margin_evidence import margin_absence_evidence
from app.services.fund_quality_v2_restrictions import full_asset_restriction_scope
from app.services.fund_rating_public_market import compact
from app.services.fund_rating_rules import decimal

VERSION = "QUALITY_CASH_BASKET_MEASURABILITY_20261017_8"
MARGIN_ABSENCE = (
    r"本基金(?:在)?(?:本报告期末|本报告期(?:内)?|报告期末)"
    r"(?:未投资|未持有)股指期货(?:合约)?[。.]?"
)


def deposit_noninterest_book_bound(*, total: str, principal: str, interest: str,
                                  provision: str | None) -> dict:
    """由明示账面总额减明示应计利息得到非利息余额，不反填未知坏账准备。

    原表总额T=本金P+利息I-减值E，故T-I=P-E。在原表未出现其他加项的
    前提下，扣除全部利息所得余额是本金的保守账面边界；即使部分减值实际
    属于利息，也不会夸大本金。E的横杠仍为未知，不能把算术差额写作原披露E。
    本函数不证明余额可以即时支付赎回；受限金额和期货保证金须继续核对。
    """
    with localcontext() as context:
        context.prec = 40
        book, gross, accrued = map(decimal, (total, principal, interest))
        loss = decimal(provision) if provision is not None else None
        if min(book, gross, accrued) < 0 or loss is not None and loss < 0:
            raise ValueError("QUALITY_CASH_DEPOSIT_NEGATIVE_DOMAIN")
        if book > gross + accrued + Decimal(".01"):
            raise ValueError("QUALITY_CASH_DEPOSIT_UNEXPLAINED_POSITIVE_ADDITION")
        if loss is not None and abs(gross + accrued - loss - book) > Decimal(".01"):
            raise ValueError("QUALITY_CASH_DEPOSIT_EXPLICIT_LOSS_NOT_RECONCILED")
        return {
            "noninterest_book_lower_bound": str(max(book - accrued, Decimal(0))),
            "formula": "max(明示活期账面总额-明示应计利息,0); 不反填减值金额",
            "provision_observed": str(loss) if loss is not None else None,
            "provision_state": "EXPLICIT_AMOUNT" if loss is not None else "UNASSIGNED_NOT_ZERO",
        }


def explicit_margin_absence(text: str) -> str | None:
    """只接纳原文的期末或全期未投资/未持有陈述；未交易及未来政策均不够。"""
    body = compact(text)
    statements = [match[0] for match in re.finditer(MARGIN_ABSENCE, body)]
    positive = re.search(r"本基金(?:在)?(?:本报告期末|报告期末)(?:投资|持有)股指期货(?:合约)?[。.]", body)
    if statements and positive:
        raise ValueError("QUALITY_CASH_FUTURES_ABSENCE_SOURCE_CONFLICT")
    return statements[0] if statements else None


def validate_deposit_additions(active_text: str) -> None:
    """非利息余额推导前拒收未识别的金额加项，防止其他收入掩盖本金减值。"""
    for line in unicodedata.normalize("NFKC", active_text).splitlines():
        monetary_row = (re.search(r"\d[\d,]*\.\d{2}", line)
                        or re.search(r"(?:^|\s)[-－—](?:\s|$)", line)
                        or re.match(r"^\s*(?:加:|减:|等于:|其他)", line))
        if monetary_row and not re.match(
            r"^\s*(?:活期存款|等于:本金|加:应计利息|减:坏账准备)\s", line
        ):
            raise ValueError("QUALITY_CASH_DEPOSIT_UNREVIEWED_AMOUNT_ROW")


def _basket_rule(text: str) -> tuple[Decimal, str]:
    """同报告明示的合计下限及现金排除项；不由普遍法规给单个产品套入5%。"""
    body = compact(text)
    prefix = (r"每个交易日日终[，,]?(?:在)?扣除股指期货(?:合约)?需缴纳的交易保证金后[，,]?"
              r"(?:应当)?(?:保持)?")
    basket = r"现金(?:或者|或)到期日在一年以内的政府债券"
    # 原报告会用“和”连接最后两个排除项；三个项目本身仍须完整明示，
    # 不扩展为任意排除项，也不把“现金加政府债”改成纯现金要求。
    suffix = r"[，,。]?其中[，,]?现金不包括结算备付金、存出保证金(?:、|和)应收申购款等"
    matches = [match for pattern in (
        prefix + r"(?:投资于)?" + basket + r"(?:的比例)?不低于基金资产净值的?(\d+(?:\.\d+)?)[%％]" + suffix,
        prefix + r"不低于基金资产净值的?(\d+(?:\.\d+)?)[%％]的" + basket + suffix,
    ) for match in re.finditer(pattern, body)]
    # 条款可出现在同报告的基金基本情况、产品说明或价格风险说明中，
    # 但须有紧邻的本基金语境。目标基金的条款不能替代联接基金自己的要求。
    scoped = []
    for match in matches:
        preceding = body[max(0, match.start()-700):match.start()]
        if "本基金" not in preceding:
            continue
        if any(word in preceding[-350:] for word in ("目标基金", "目标ETF")):
            # 联接基金可在同一原句先明示“本基金投资于目标ETF的比例……；”，
            # 随后又以“本基金每个交易日……”说明自身现金义务。该紧邻且完整
            # 的主语链能区分本基金与目标ETF；仅看到孤立“本基金”仍不足以
            # 推翻目标基金范围，不能接纳目标ETF投资范围中的自称语句。
            own_sibling = re.search(
                r"本基金投资于目标ETF的比例不低于基金资产净值的?"
                r"(?:100|\d{1,2})(?:\.\d+)?[%％][;；]本基金$", preceding
            )
            if not own_sibling:
                continue
        scoped.append(match)
    matches = scoped
    floors = {decimal(match[1]) / 100 for match in matches}
    if len(floors) != 1:
        raise ValueError("QUALITY_CASH_BASKET_COMPLETE_SAME_REPORT_CONSTRAINT_MISSING_OR_CONFLICTING")
    return next(iter(floors)), matches[0][0]


def shortfall_bounds(*, floor: str, cash_lower: str, cash_upper: str,
                     government_upper: str, net_assets: str, margin_absence_excerpt: str | None) -> dict:
    """返回披露日缺口的数学区间，单位为净资产比例，不估计缺失的保证金。

    cash_lower/upper是有来源的可用现金边界；政府债上界可保守取全部债券余额，
    也可取全部非现金资产作更宽的上界，政府债实际值仍保持未知。只有原文
    明确未投资股指期货时才允许保证金为0。
    扣减保证金后的净缓冲允许为负；未知保证金且没有有限上界时，净缓冲下界
    和缺口上界都不能写成有限数，更不能截为0或把缺口封顶在合同下限。
    """
    if margin_absence_excerpt is not None and not re.fullmatch(MARGIN_ABSENCE, compact(margin_absence_excerpt)):
        raise ValueError("QUALITY_CASH_BASKET_MARGIN_ABSENCE_NOT_EXPLICIT")
    return _shortfall_arithmetic(floor=floor, cash_lower=cash_lower, cash_upper=cash_upper,
                                 government_upper=government_upper, net_assets=net_assets,
                                 zero_margin_identified=margin_absence_excerpt is not None)


def _shortfall_arithmetic(*, floor: str, cash_lower: str, cash_upper: str,
                          government_upper: str, net_assets: str, zero_margin_identified: bool) -> dict:
    """只供完成原句核对的两个入口调用；内部结论不充当来源核验证据。"""
    with localcontext() as context:
        context.prec = 40
        target, lower, upper, bonds, net = map(decimal, (floor, cash_lower, cash_upper, government_upper, net_assets))
        if not 0 < target <= 1 or net <= 0 or min(lower, upper, bonds) < 0 or lower > upper:
            raise ValueError("QUALITY_CASH_BASKET_BOUND_DOMAIN")
        available_lower = lower if zero_margin_identified else None
        available_upper = upper + bonds
        gap_lower = max(target - available_upper / net, Decimal(0))
        gap_upper = max(target - available_lower / net, Decimal(0)) if available_lower is not None else None
        exact = gap_upper is not None and gap_lower == gap_upper
        return {
            "shortfall_ratio_lower": str(gap_lower),
            "shortfall_ratio_upper": str(gap_upper) if gap_upper is not None else None,
            "identified_shortfall_ratio": str(gap_lower) if exact else None,
            "state": "POINT_IDENTIFIED_FOR_REPORT_DATE_NOT_DAILY"
            if exact else "INTERVAL_ONLY_NOT_ADMISSIBLE_FOR_TOTAL_SCORE",
            "unknown_margin_requirement": not zero_margin_identified,
            "government_amount_point_identified_by_bound": None if bonds else "0",
            "government_amount_lower_bound": "0", "government_amount_upper_bound": str(bonds),
            "available_amount_lower_bound": str(available_lower) if available_lower is not None else None,
            "available_amount_upper_bound": str(available_upper),
            "unknown_margin_bound": "NO_FINITE_LOWER_NET_BUFFER_OR_UPPER_SHORTFALL"
            if available_lower is None else None,
            "buffer_formula": "cash + eligible_government_bonds - required_margin; signed, not clipped at zero",
            "scope": "DISCLOSED_DATE_CONSTRAINT_BOUND_NOT_PURE_CASH_TARGET_OR_DAILY_COMPLIANCE",
        }


def scoped_shortfall_bounds(*, report_text: str, cash_lower: str, cash_upper: str,
                            government_upper: str, net_assets: str) -> dict:
    """重读义务原句并逐工具核缺位，避免把股指期货缺位用于其他保证金范围。

    本入口不接受无保证金布尔标志，也不让调用方给义务默认下限。条款缺
    排除项、未明示扣减范围时拒收；实际金额未知时保留无限/有限数学边界。
    """
    clause = cash_basket_clause(report_text)
    if clause["state"] != "REPORTED_CLAUSE_BASIS_IDENTIFIED":
        raise ValueError("QUALITY_CASH_BASKET_CLAUSE_BASIS_INCOMPLETE:" + ",".join(clause["missing_basis"]))
    evidence = margin_absence_evidence(report_text,
        requirement_scope=clause["margin_requirement_scope"],
        required_instruments=clause["required_margin_instruments"])
    result = _shortfall_arithmetic(floor=clause["reported_floor_ratio"], cash_lower=cash_lower,
        cash_upper=cash_upper, government_upper=government_upper, net_assets=net_assets,
        zero_margin_identified=evidence["zero_required_margin_identified"])
    return result | {"reported_clause": clause, "margin_scope_evidence": evidence}


def rebuild_cash_constraint(pages: list[str], *, observed_on: str, source_hash: str,
                            restricted_metric: dict | None = None) -> dict:
    """从实际报告重建合同下限与现金附注，返回原文、金额和精确的可测量性缺口。

    调用方先校验基金身份、原字节哈希、公开日与考察日。本函数再核经济期、
    完整资产负债守恒、附注明示的一列/两列当前日期和本金/利息/总额。
    坏账准备横杠保持未知；账面总额减已披露应计利息只形成非利息账面边界。
    受限资料仅在同源且全资产原句存在时用于保守下界；未知受限情况、保证金
    不作为确切0。附注及金额均从原始页读取；金额断行的文本列对齐轨迹
    不能冒充PDF像素坐标，不能反推未披露的金额或章号。
    """
    date.fromisoformat(observed_on)
    if not re.fullmatch(r"[a-f0-9]{64}", source_hash) or report_period("\n".join(pages))[1].isoformat() != observed_on:
        raise ValueError("QUALITY_CASH_BASKET_SOURCE_OR_ECONOMIC_PERIOD")
    raw_text = "\n".join(pages)
    clause_text, clause_view = cash_clause_page_view(pages)
    clause = cash_basket_clause(clause_text)
    if clause["state"] != "REPORTED_CLAUSE_BASIS_IDENTIFIED":
        raise ValueError("QUALITY_CASH_BASKET_CLAUSE_BASIS_INCOMPLETE:" + ",".join(clause["missing_basis"]))
    # 原件财务读取器已能处理无自动章号、交错日期和单列附注；附注位置、
    # 金额及其原文行直接来自原始页，不再使用补章号的内部视图。
    accounting = accounting_observations(pages, date.fromisoformat(observed_on))
    observations = accounting["observations"]
    note_id = accounting.get("cash_note_reference")
    note_heads = list(re.finditer(
        rf"(?m)^\s*{re.escape(note_id or 'NO_NOTE')}\s*(?:货币资金|银行存款)[ \t]*$", raw_text
    ))
    if len(note_heads) != 1:
        raise ValueError("QUALITY_CASH_BASKET_DEPOSIT_NOTE_SCOPE_UNRESOLVED")
    note_text = raw_text[note_heads[0].end():]
    next_note = (note_id or "").rsplit(".", 1)[0] + ".2"
    note_text = re.split(rf"(?m)^\s*{re.escape(next_note)}(?:\s|$)", note_text, maxsplit=1)[0]
    active_start = re.search(r"(?m)^\s*活期存款\s", note_text)
    if not active_start:
        raise ValueError("QUALITY_CASH_BASKET_DEMAND_DEPOSIT_ROW_MISSING")
    active_text = re.split(r"(?m)^\s*定期存款\s", note_text[active_start.start():], maxsplit=1)[0]
    validate_deposit_additions(active_text)
    needed = {"money_funds", "net_assets", "total_assets", "demand_deposits_total",
              "demand_deposits_principal", "demand_deposits_accrued_interest"}
    if not needed <= observations.keys():
        raise ValueError("QUALITY_CASH_BASKET_EXPLICIT_ACCOUNTING_VALUES_UNRESOLVED:" +
                         ",".join(sorted(needed - observations.keys())))
    def value(name: str) -> Decimal:
        return decimal(observations[name]["value"])
    total, principal, interest = map(value, ("demand_deposits_total", "demand_deposits_principal",
                                          "demand_deposits_accrued_interest"))
    provision = observations.get("demand_deposits_credit_loss", {}).get("value")
    deposit = deposit_noninterest_book_bound(total=str(total), principal=str(principal), interest=str(interest),
                                            provision=provision)
    cash, net, assets = map(value, ("money_funds", "net_assets", "total_assets"))
    if cash > assets or net <= 0:
        raise ValueError("QUALITY_CASH_BASKET_BALANCE_DOMAIN")
    # 未知的债券横杠不读取为0。全部非货币资产提供宽松但可靠的政府债上界，
    # 不宣称债券或合格政府债实际金额。即使这个上界很宽，也不能伪造精确值。
    government_upper = assets - cash
    government_basis = "ALL_NON_MONEY_ASSETS_CONSERVATIVE_UPPER_NOT_ACTUAL_BOND_AMOUNT"
    known_restricted, restriction_basis = None, None
    if restricted_metric is not None:
        if (restricted_metric.get('source_hashes') != [source_hash]
                or decimal(restricted_metric['net_assets']) != net
                or restricted_metric['observed_on'] != observed_on or not full_asset_restriction_scope(raw_text)):
            raise ValueError("QUALITY_CASH_BASKET_RESTRICTED_BINDING_OR_SCOPE")
        known_restricted = decimal(restricted_metric['restricted_amount'])
        if known_restricted < 0 or known_restricted > assets:
            raise ValueError("QUALITY_CASH_BASKET_RESTRICTED_AMOUNT_DOMAIN")
        restriction_basis = 'CONSERVATIVELY_SUBTRACT_ALL_VERIFIED_RESTRICTIONS_NOT_ASSUME_CASH_SHARE_ZERO'
    # 即使已知受限资产全是股票，也不按资产种类免减；保守扣除全额不夸大现金。
    cash_lower = (max(decimal(deposit["noninterest_book_lower_bound"]) - known_restricted, Decimal(0))
                  if known_restricted is not None else Decimal(0))
    result = scoped_shortfall_bounds(report_text=clause_text, cash_lower=str(cash_lower), cash_upper=str(cash),
                                     government_upper=str(government_upper), net_assets=str(net))
    result["reported_clause"]["report_page_view"] = clause_view
    absence = explicit_margin_absence(raw_text)
    return result | {
        'version': VERSION, 'source_hash': source_hash, 'observed_on': observed_on,
        'contract_basket_floor_ratio': clause['reported_floor_ratio'],
        'contract_clause': clause['original_clause_normalized'],
        'contract_version_verified': False,
        'cash_clause_page_view': clause_view,
        'currency': 'CNY', 'net_assets': str(net), 'cash_book_amount': str(cash),
        'demand_deposit_principal': str(principal), 'demand_deposit_interest': str(interest),
        'demand_deposit_provision': provision, 'deposit_book_bound': deposit,
        'parser_view_accounting': accounting, 'government_upper_basis': government_basis,
        'accounting_source_view': 'ORIGINAL_PAGES_NOT_RENUMBERED_STATEMENT',
        'restricted_amount': None if known_restricted is None else str(known_restricted),
        'restriction_bound_basis': restriction_basis, 'margin_absence_excerpt': absence,
        'chapter_view': {'scope': 'ORIGINAL_HEADINGS_AND_AMOUNT_ROWS_NO_SYNTHETIC_RENUMBERING'},
        'score_or_grade_created': False, 'raw4_metric_replaced': False,
    }
