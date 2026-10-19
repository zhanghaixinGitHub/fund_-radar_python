"""逐期配置约束的真实证据及精确投影，不把约束子集冒充完整合同允许集合。

金额和比例全程转为有理数。当前原件适配支持招商股票/混合合同明确的
股票及港股比例；同份合同的现金、衍生品等未完成解释时，只输出条件性
子集距离。它不能写入 allocation_deviation，更不能证明日常投资合规。
"""

import hashlib
import re
import unicodedata
from calendar import monthrange
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from itertools import combinations
from math import comb
from pathlib import Path
from urllib.parse import urlsplit

VERSION = "QUALITY_ALLOCATION_EVIDENCE_20261017_3"


def _number(value) -> Fraction:
    """只接收十进制原值或精确分数，拒绝浮点、布尔及非有限值。"""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("ALLOCATION_EXACT_NUMBER_REQUIRED")
    text = str(value)
    if len(text) > 150:
        raise ValueError("ALLOCATION_NUMBER_TOO_LONG")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("ALLOCATION_EXACT_NUMBER_REQUIRED") from exc
    if not number.is_finite():
        raise ValueError("ALLOCATION_FINITE_NUMBER_REQUIRED")
    if abs(number.as_tuple().exponent) > 100 or abs(number.adjusted()) > 100:
        raise ValueError("ALLOCATION_NUMBER_SCALE_EXCEEDED")
    return Fraction(number)


def _stored(value: Fraction) -> dict:
    """显示值不作为证明；分子/分母保留投影及均值的精确结果。"""
    with localcontext() as ctx:
        ctx.prec = 60
        shown = str(Decimal(value.numerator) / Decimal(value.denominator))
    return {"value": shown, "numerator": str(value.numerator), "denominator": str(value.denominator)}


def _solve(rows: tuple, rhs: tuple) -> tuple | None:
    """有理数高斯消元；退化交点跳过，不依赖数值容差或成功标志。"""
    size = len(rows)
    matrix = [list(row) + [value] for row, value in zip(rows, rhs, strict=True)]
    for column in range(size):
        pivot = next((i for i in range(column, size) if matrix[i][column]), None)
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [value / scale for value in matrix[column]]
        for i in range(size):
            if i != column:
                scale = matrix[i][column]
                matrix[i] = [a - scale * b for a, b in zip(matrix[i], matrix[column], strict=True)]
    return tuple(row[-1] for row in matrix)


def project_half_l1(weights: list, constraints: list[dict]) -> dict:
    """求 simplex 内线性允许集合上的精确最小半L1距离，比例单位为0—1。

    constraints 每行含 coefficients、upper，表示 a*x <= upper。预算和
    非负约束由本函数加入。将所有 |x_i-w_i| 折点平面加入有限划分后，
    每个单元上目标线性，故枚举所有满秩顶点可取得全局最小值。仅支持
    2—4类、最多16个约束及10万个交点；超界报缺口，不降级为近似解。
    """
    point = tuple(map(_number, weights))
    size = len(point)
    if not 2 <= size <= 4 or len(constraints) > 16:
        raise ValueError("ALLOCATION_PROJECTION_SCOPE_EXCEEDED")
    if any(value < 0 for value in point) or sum(point) != 1:
        raise ValueError("ALLOCATION_WEIGHTS_NOT_A_COMPLETE_POINT")
    limits = []
    for item in constraints:
        row = tuple(map(_number, item["coefficients"]))
        if len(row) != size or not any(row):
            raise ValueError("ALLOCATION_CONSTRAINT_SHAPE")
        limits.append((row, _number(item["upper"])))
    for i in range(size):
        limits.append((tuple(Fraction(-int(i == j)) for j in range(size)), Fraction(0)))
    # 目标折点和预算平面也参与顶点划分。重复平面只影响效率，不影响证明。
    planes = list(dict.fromkeys(limits + [
        (tuple(Fraction(int(i == j)) for j in range(size)), point[i]) for i in range(size)
    ]))
    candidates = comb(len(planes), size - 1)
    if candidates > 100_000:
        raise ValueError("ALLOCATION_EXACT_VERTEX_BUDGET_EXCEEDED")
    best, witness, feasible = None, None, 0
    for selected in combinations(planes, size - 1):
        vertex = _solve((tuple(Fraction(1) for _ in point),) + tuple(row for row, _ in selected),
                        (Fraction(1),) + tuple(value for _, value in selected))
        if vertex is None or any(sum(a * b for a, b in zip(row, vertex, strict=True)) > upper
                                 for row, upper in limits):
            continue
        feasible += 1
        distance = sum(abs(a - b) for a, b in zip(vertex, point, strict=True)) / 2
        if best is None or distance < best:
            best, witness = distance, vertex
    if best is None:
        raise ValueError("ALLOCATION_ALLOWED_SET_EMPTY")
    return {"method": "EXACT_RATIONAL_PIECEWISE_LINEAR_VERTEX_ENUMERATION", "distance": _stored(best),
            "projected_weights": [_stored(value) for value in witness],
            "enumerated_intersections": candidates, "feasible_intersections": feasible}


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def contract_allocation_clauses(pages: list[str]) -> dict:
    """读合同实际投资范围和完整组合限制，保留每一尚未解释的原句。

    只接纳成对正文标题，目录、招募书摘要或一般风险提示不充作合同条款。
    股票/存托凭证和港股比例在范围、组合限制两处须相同，不猜分母。
    """
    text = _compact("\n".join(pages))
    scope_starts = list(re.finditer(r"二、投资范围", text))
    sections = []
    for start in scope_starts:
        stop = text.find("三、投资策略", start.end())
        if stop > start.end() and stop - start.end() < 12000:
            sections.append(text[start.end():stop])
    starts = list(re.finditer(r"四、投资限制1、组合限制基金的投资组合应遵循以下限制[:：]", text))
    if len(sections) != 1 or len(starts) != 1:
        raise ValueError("ALLOCATION_CONTRACT_COMPLETE_SECTION_UNPARSED")
    stop = text.find("2、禁止行为", starts[0].end())
    if stop < 0:
        raise ValueError("ALLOCATION_CONTRACT_RESTRICTION_END_UNPARSED")
    restrictions = text[starts[0].end():stop]
    adjustment = re.search(r"除上述", restrictions)
    if adjustment is None:
        raise ValueError("ALLOCATION_CONTRACT_ADJUSTMENT_SECTION_UNPARSED")
    adjustment_excerpt = restrictions[adjustment.start():]
    restrictions = restrictions[:adjustment.start()]
    markers = list(re.finditer(r"\((\d{1,2})\)", restrictions))
    # 最后一个编号之后的调整期条文不能丢失，单列保留，而非默认为免责。
    if [int(m[1]) for m in markers] != list(range(1, len(markers) + 1)) or not markers:
        raise ValueError("ALLOCATION_CONTRACT_CLAUSE_SEQUENCE_UNVERIFIED")
    clauses = [{"clause_id": int(m[1]), "excerpt": restrictions[m.end():
                markers[i+1].start() if i+1 < len(markers) else len(restrictions)]}
               for i, m in enumerate(markers)]
    pattern = (r"本基金(?:投资组合中)?股票、存托凭证(?:资产)?(?:的)?投资"
               r"(?:比例为(?P<basis>基金资产(?:总值|净值)?)的|占基金资产的比例为)(?P<low>\d+(?:\.\d+)?)%"
               r"[-—至～](?P<high>\d+(?:\.\d+)?)%[,，]其中港股通标的股票(?:的)?"
               r"投资比例不超过本基金股票资产的(?P<hk>\d+(?:\.\d+)?)%")
    matches = [re.search(pattern, part) for part in (sections[0], clauses[0]["excerpt"])]
    if any(m is None for m in matches) or any(matches[0][k] != matches[1][k] for k in ("low", "high", "hk")):
        raise ValueError("ALLOCATION_STOCK_HK_CONSTRAINT_MISSING_OR_CONFLICTING")
    bases = {m["basis"] or "基金资产" for m in matches}
    if len(bases) != 1:
        raise ValueError("ALLOCATION_CONTRACT_RATIO_DENOMINATOR_CONFLICT")
    basis = next(iter(bases))
    identified_basis = {"基金资产总值": "TOTAL_ASSETS", "基金资产净值": "NET_ASSETS"}.get(basis)
    low, high, hk = (_number(matches[0][k]) / 100 for k in ("low", "high", "hk"))
    if not 0 <= low <= high <= 1 or not 0 <= hk <= 1:
        raise ValueError("ALLOCATION_CONTRACT_RATIO_RANGE")
    return {"stock_lower": _stored(low), "stock_upper": _stored(high), "hk_of_stock_upper": _stored(hk),
            "scope_excerpt": sections[0], "stock_constraint_excerpt": matches[0][0],
            "ratio_denominator_evidence": {"literal": basis, "economic_basis": identified_basis,
                "state": ("EXPLICIT_CONTRACT_DENOMINATOR" if identified_basis
                          else "CONTRACT_RATIO_DENOMINATOR_UNRESOLVED"),
                "scope": "NOT_INFERRED_FROM_REPORT_COLUMN_OR_FUND_NAME", "excerpt": matches[0][0]},
            "combination_clauses": clauses, "adjustment_and_transition_excerpt": adjustment_excerpt,
            "full_allowed_set_remaining": [
                {"reason": "CASH_AND_DERIVATIVE_SCOPE_FROM_INVESTMENT_RANGE", "excerpt": sections[0]},
                *[{"reason": "CONTRACT_CLAUSE_NOT_TRANSLATED_TO_COMPLETE_ALLOWED_SET", **item}
                  for item in clauses if item["clause_id"] != 1]],
            "source_pages": [i+1 for i, page in enumerate(pages)
                             if "二、投资范围" in _compact(page) or "四、投资限制" in _compact(page)]}


def _industry_hk_amount(text: str) -> dict:
    """由成对境内/港股行业正文取精确金额；仅具名未持有原句可证明0。"""
    domestic = list(re.finditer(r"[5678]\.2\.1\s*(?:报告期末)?按行业分类的境内股票投资组合", text))
    foreign = list(re.finditer(r"[5678]\.2\.2\s*(?:报告期末)?按行业分类的港股通投资股票投资组合", text))
    if len(domestic) != 1 or len(foreign) != 1 or domestic[0].end() >= foreign[0].start():
        raise ValueError("ALLOCATION_DOMESTIC_HK_SECTION_PAIR_UNPARSED")
    end = re.search(r"[5678]\.3\s*(?:报告期末|期末)", text[foreign[0].end():])
    if end is None:
        raise ValueError("ALLOCATION_HK_SECTION_END_UNPARSED")
    own = text[domestic[0].end():foreign[0].start()]
    abroad = text[foreign[0].end():foreign[0].end()+end.start()]
    total_pattern = r"(?m)^\s*合\s*计\s+([\d,]+\.\d{2})\s+[\d.]+"
    own_totals = re.findall(total_pattern, own)
    hk_totals = re.findall(total_pattern, abroad)
    absence = re.search(r"本基金本报告期末未持有港股通投资股票[。.]", _compact(abroad))
    if len(own_totals) != 1 or (absence is not None and hk_totals):
        raise ValueError("ALLOCATION_DOMESTIC_HK_TOTAL_CONFLICT_OR_MISSING")
    if absence is None and len(hk_totals) != 1:
        raise ValueError("ALLOCATION_HK_EXACT_TOTAL_OR_ABSENCE_MISSING")
    return {"domestic_amount": own_totals[0].replace(",", ""),
            "hk_amount": "0.00" if absence else hk_totals[0].replace(",", ""),
            "basis": "EXPLICIT_PERIOD_END_HK_ABSENCE" if absence else "EXACT_HK_INDUSTRY_TOTAL",
            "excerpt": _compact(abroad), "domestic_excerpt": _compact(own)}


def report_allocation_point(block: str, amounts: dict, *, industry_text: str = "") -> dict:
    """从已定位的资产表原句取股票/港股金额，绝不以四舍五入占比反推金额。

    其余资产取总资产减股票这一可唯一确定的补集；不把其他行的横杠改为0。
    港股只有原句金额才可用，栏目缺席或“可不投资”的策略均不能产生零值。
    """
    body = _compact(block)
    stocks = re.findall(r"其中[:：]股票([\d,]+\.\d{2})(?:\d|\.|-)", body)
    hk = re.findall(r"上表权益投资中通过港股通交易机制投资的港股金额(?:为)?人民币([\d,]+\.\d{2})元", body)
    industry = _industry_hk_amount(industry_text) if industry_text else None
    if len(stocks) != 1 or len(hk) > 1 or (not hk and industry is None):
        raise ValueError("ALLOCATION_STOCK_OR_HK_EXACT_AMOUNT_UNPARSED")
    foreign_text = hk[0].replace(",", "") if hk else industry["hk_amount"]
    stock, foreign = _number(stocks[0].replace(",", "")), _number(foreign_text)
    total, equity = _number(amounts["合计"]), _number(amounts["权益投资"])
    if total <= 0 or not 0 <= foreign <= stock <= total or stock != equity:
        raise ValueError("ALLOCATION_STOCK_EQUITY_OR_TOTAL_NOT_RECONCILED")
    if industry and (foreign != _number(industry["hk_amount"]) or
                     _number(industry["domestic_amount"]) + foreign != stock):
        raise ValueError("ALLOCATION_DOMESTIC_HK_AND_STOCK_AMOUNTS_NOT_RECONCILED")
    hk_excerpt = re.search(
        r"上表权益投资中通过港股通交易机制投资的港股金额(?:为)?人民币[\d,]+\.\d{2}元", body)
    return {"amounts_cny": {"stock": stocks[0].replace(",", ""), "hk_stock": foreign_text,
                            "total_assets": amounts["合计"]},
            "weights": [_stored((stock-foreign)/total), _stored(foreign/total), _stored((total-stock)/total)],
            "categories": ["DOMESTIC_STOCK_AND_DR", "HK_CONNECT_STOCK", "OTHER_ASSET_COMPLEMENT"],
            "hk_amount_excerpt": hk_excerpt[0] if hk_excerpt else industry["excerpt"],
            "industry_reconciliation": industry,
            "denominator": "TOTAL_ASSETS_NOT_NAV",
            "denominator_scope": "ASSET_COORDINATE_SYSTEM_NOT_CONTRACT_RATIO_EVIDENCE",
            "unobserved_asset_rows_imputed": []}


def report_product_net_assets(pages: list[str], *, observed_on: str, total_assets: str) -> dict:
    """从原文补产品层净资产；不把本份额净资产当产品总额，也不倒用比较列。

    季报只适配明示A/C两列、人民币元、当期起止日及两代码/简称互相绑定
    的3.1表；其他份额结构留缺口。年/中期直接复用已核日期与列序的资产
    负债表读取器。净资产仅用于条件演算，不能自行确定合同分母。
    """
    text = unicodedata.normalize("NFKC", "\n".join(pages))
    compact = _compact(text)
    observed = date.fromisoformat(observed_on)
    quarter = re.search(r"(20\d{2})年第([1-4])季度报告", compact[:2500])
    if quarter:
        year, q = map(int, quarter.groups())
        end = date(year, q*3, monthrange(year, q*3)[1])
        if end != observed:
            raise ValueError("ALLOCATION_NAV_QUARTER_PERIOD_MISMATCH")
        blocks = list(re.finditer(r"3\.1\s*主要财务指标", text))
        if len(blocks) != 1:
            raise ValueError("ALLOCATION_NAV_QUARTER_FINANCIAL_SECTION_UNPARSED")
        tail = text[blocks[0].end():]
        stop = re.search(r"3\.2\s*基金净值表现", tail)
        if stop is None:
            raise ValueError("ALLOCATION_NAV_FINANCIAL_SECTION_END_UNPARSED")
        body = tail[:stop.start()]
        period = (rf"报告期[（(]{year}年{(q-1)*3+1}月1日[-－—至～]+"
                  rf"{year}年{q*3}月{observed.day}日[）)]")
        if "人民币元" not in _compact(body) or not re.search(period, _compact(body)):
            raise ValueError("ALLOCATION_NAV_FINANCIAL_PERIOD_OR_CNY_UNVERIFIED")
        basic = compact[:compact.find("3.1主要财务指标")]
        names = re.search(r"下属分级基金的基金简称(.{2,70}?A)(.{2,70}?C)下属分级基金的交易代码", basic)
        codes = re.search(r"下属分级基金的交易代码(\d{6})(\d{6})(?!\d)", basic)
        if names is None or codes is None or codes[1] == codes[2]:
            raise ValueError("ALLOCATION_NAV_EXPLICIT_TWO_SHARE_CLASS_HEADER_REQUIRED")
        header = _compact(body).split("1.本期已实现收益")[0]
        if names[1]+names[2] not in header:
            raise ValueError("ALLOCATION_NAV_FINANCIAL_SHARE_CLASS_ORDER_UNBOUND")
        money = r"(\d[\d,]*\.\d{2})"
        rows = list(re.finditer(rf"(?m)^\s*4\.\s*期末基金资产净值\s+{money}\s+{money}[ \t]*$", body))
        if len(rows) != 1:
            raise ValueError("ALLOCATION_NAV_CURRENT_SHARE_CLASS_VALUES_UNPARSED")
        parts = [value.replace(",", "") for value in rows[0].groups()]
        nav = sum(map(_number, parts))
        proof = {"basis": "EXPLICIT_CURRENT_A_C_NET_ASSETS_SUM", "share_codes": list(codes.groups()),
                 "share_names": list(names.groups()), "share_net_assets_cny": parts,
                 "original_row": rows[0][0], "period_excerpt": re.search(period, _compact(body))[0]}
    else:
        from app.services.fund_quality_v2_accounting import accounting_observations
        from app.services.fund_quality_v2_stock_book import periodic_financial_view

        view, edits = periodic_financial_view(pages)
        try:
            facts = accounting_observations(view, observed)
        except ValueError as exc:
            if str(exc) != "ACCOUNTING_BALANCE_DATE_CNY_UNIT_OR_UNIQUE_BODY_UNVERIFIED":
                raise
            # 首年报告确实只有本期末一列时，使用独立的严格版式适配；不
            # 修改成熟的两列读取器，也不把第二列缺席理解成比较期为0。
            return _first_year_single_column_net_assets(view, observed=observed, total_assets=total_assets) | {
                "financial_parser_view": edits}
        rows = facts["observations"]
        if not {"net_assets", "total_assets", "total_liabilities"} <= rows.keys():
            raise ValueError("ALLOCATION_NAV_CURRENT_BALANCE_TOTALS_UNPARSED")
        nav = _number(rows["net_assets"]["value"])
        if (_number(rows["total_assets"]["value"]) != _number(total_assets) or
                nav + _number(rows["total_liabilities"]["value"]) != _number(total_assets)):
            raise ValueError("ALLOCATION_NAV_EXACT_ASSETS_LIABILITIES_EQUATION_CONFLICT")
        proof = {"basis": "EXPLICIT_CURRENT_CNY_BALANCE_NET_ASSETS", "net_asset_row": rows["net_assets"],
                 "total_assets_row": rows["total_assets"], "liabilities_row": rows["total_liabilities"],
                 "financial_parser_view": edits}
    if not 0 < nav <= _number(total_assets):
        raise ValueError("ALLOCATION_NAV_NOT_POSITIVE_OR_EXCEEDS_TOTAL_ASSETS")
    return {"net_assets_cny": _stored(nav)["value"], "observed_on": observed_on, "unit": "CNY_YUAN",
            "proof": proof, "scope": "PRODUCT_NET_ASSETS_NOT_CONTRACT_DENOMINATOR_ACCEPTANCE"}


def _first_year_single_column_net_assets(pages: list[str], *, observed: date, total_assets: str) -> dict:
    """仅接首年、人民币元、唯一当期一列的完整资产负债表，逐项保留原行。"""
    text = unicodedata.normalize("NFKC", "\n".join(pages))
    inception = re.search(r"基金合同生效日(20\d{2})年(\d{1,2})月(\d{1,2})日", _compact(text)[:16000])
    if inception is None or date(*map(int, inception.groups())).year != observed.year:
        raise ValueError("ALLOCATION_NAV_ONE_COLUMN_NOT_EXPLICIT_FIRST_YEAR")
    headers = list(re.finditer(r"(?m)^[ \t]*([567])\.1[ \t]+资产负债表[ \t]*$", text))
    if len(headers) != 1:
        raise ValueError("ALLOCATION_NAV_ONE_COLUMN_UNIQUE_BALANCE_BODY_REQUIRED")
    tail = text[headers[0].end():]
    end = re.search(rf"(?m)^[ \t]*{headers[0][1]}\.2[ \t]+利润表", tail)
    if end is None:
        raise ValueError("ALLOCATION_NAV_ONE_COLUMN_BALANCE_END_MISSING")
    body = tail[:end.start()]
    first_row = re.search(r"(?m)^[ \t]*资产[:：][ \t]*$", body)
    if first_row is None:
        raise ValueError("ALLOCATION_NAV_ONE_COLUMN_ASSET_BODY_MISSING")
    header = body[:first_row.start()]
    day = f"{observed.year}年{observed.month}月{observed.day}日"
    if "单位:人民币元" not in _compact(header) or not _compact(header).endswith("资产附注号本期末" + day):
        raise ValueError("ALLOCATION_NAV_ONE_COLUMN_CNY_DATE_LAYOUT_UNVERIFIED")
    rows = {}
    for label in ("资产总计", "负债合计", "净资产合计", "负债和净资产总计"):
        matches = list(re.finditer(rf"(?m)^[ \t]*{label}[ \t]+(\d[\d,]*\.\d{{2}})[ \t]*$", body))
        if len(matches) != 1:
            raise ValueError("ALLOCATION_NAV_ONE_COLUMN_EXPLICIT_ROW_REQUIRED:" + label)
        rows[label] = {"value": matches[0][1].replace(",", ""), "original_row": matches[0][0]}
    nav = _number(rows["净资产合计"]["value"])
    total = _number(total_assets)
    if (not 0 < nav <= total or _number(rows["资产总计"]["value"]) != total or
            _number(rows["负债和净资产总计"]["value"]) != total or
            nav + _number(rows["负债合计"]["value"]) != total):
        raise ValueError("ALLOCATION_NAV_ONE_COLUMN_EXACT_BALANCE_CONFLICT")
    return {"net_assets_cny": _stored(nav)["value"], "observed_on": str(observed), "unit": "CNY_YUAN",
            "proof": {"basis": "EXPLICIT_FIRST_YEAR_SINGLE_CURRENT_CNY_COLUMN", "rows": rows,
                      "column_header_excerpt": header, "inception_excerpt": inception[0]},
            "scope": "PRODUCT_NET_ASSETS_NOT_CONTRACT_DENOMINATOR_ACCEPTANCE"}


def allocation_history(*, refs: list[dict], root: Path, reports: list[dict],
                       required_dates: list[str], legal_name: str, as_of: date, code: str) -> dict:
    """以实际历史合同和报告计算约束子集距离，按经济日去重并列清精确阻塞。

    发布日仅用于信息可得性筛选。合同修订标题未证明法律生效日，所以保留
    候选版本绑定缺口；没有已核生效链时，条件性距离不成为评分原始值。
    """
    from app.services.fund_rating_public_market import public_document

    contracts, gaps, amendment_notices = [], [], []
    from app.services.fund_quality_v2_contract_amendments import verify_cmf_fee_amendment

    for ref in refs:
        if ref.get("label") != "contract-amendment-notice":
            continue
        try:
            file = (root/ref["file"]).resolve()
            if not file.is_relative_to((root/"sources").resolve()) or not file.is_file():
                raise ValueError("CONTRACT_NOTICE_ARCHIVE_PATH")
            amendment_notices.append(verify_cmf_fee_amendment(
                file.read_bytes(), ref, refs=refs, root=root, legal_name=legal_name, as_of=as_of))
        except (ValueError, KeyError, OSError) as exc:
            gaps.append({"source_hashes": [ref["sha256"]], "source_url": ref["url"], "reason": str(exc)})
    for ref in refs:
        if ref.get("label") != "historical-contract" or legal_name + "基金合同" not in ref.get("title", ""):
            continue
        item = {"source_hashes": [ref["sha256"]], "source_url": ref["url"],
                "published_on": ref.get("published_on"), "title": ref.get("title")}
        try:
            address = urlsplit(ref["url"])
            if (address.scheme != "https" or address.hostname != "static.cmfchina.com" or address.query
                    or address.fragment or address.username or address.password or address.port not in {None, 443}
                    or not re.fullmatch(r"/fundarticle/\d{8}/\d+\.pdf", address.path)):
                raise ValueError("ALLOCATION_CONTRACT_ISSUER_ROUTE_UNSUPPORTED")
            published = date.fromisoformat(ref["published_on"])
            if published > as_of:
                raise ValueError("ALLOCATION_CONTRACT_PUBLISHED_AFTER_AS_OF")
            file = (root/ref["file"]).resolve()
            if not file.is_relative_to((root/"sources").resolve()):
                raise ValueError("ALLOCATION_CONTRACT_OUTSIDE_SOURCE_ARCHIVE")
            raw = file.read_bytes()
            if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                raise ValueError("ALLOCATION_CONTRACT_ACTUAL_HASH_CHANGED")
            pages, _ = public_document(raw, ref, code, "基金合同")
            if legal_name not in _compact(pages[0])[:1000]:
                raise ValueError("ALLOCATION_CONTRACT_LEGAL_NAME_NOT_ON_COVER")
            clauses = contract_allocation_clauses(pages)
            item.update(clauses=clauses, version_binding_gap="REVISION_OR_PUBLICATION_DATE_IS_NOT_EFFECTIVE_DATE")
            registered = re.search(r"基金合同[（(](20\d{2})年(\d{1,2})月(\d{1,2})日修订[）)]", ref["title"])
            if registered:
                revision = str(date(*map(int, registered.groups())))
                notices = [notice for notice in amendment_notices if notice["effective_on"] == revision]
                if notices:
                    item.update(registered_revision_notice_evidence=notices,
                        version_binding_gap="REGISTERED_REVISION_HAS_NOTICE_EFFECTIVE_DATE_BUT_COMPLETE_VERSION_CHAIN_UNPROVED")
            contracts.append(item)
        except (ValueError, KeyError, OSError) as exc:
            gaps.append(item | {"reason": str(exc)})
    observations = []
    for day in sorted(set(required_dates)):
        rows = [r for r in reports if r.get("observed_on") == day]
        points = [r for r in rows if "allocation_asset_point" in r]
        item = {"observed_on": day, "source_hashes": sorted({h for r in rows for h in r["source_hashes"]})}
        known = {tuple((w["numerator"], w["denominator"]) for w in r["allocation_asset_point"]["weights"])
                 for r in points}
        nav_points = [r for r in points if r["allocation_asset_point"].get("net_asset_evidence")]
        nav_values = {_number(r["allocation_asset_point"]["net_asset_evidence"]["net_assets_cny"]) for r in nav_points}
        if not points or len(points) != len(rows) or len(known) != 1 or len(nav_values) > 1:
            item.update(reason="ALLOCATION_REPORT_POINT_MISSING_OR_COUNTERPART_CONFLICT",
                        report_gaps=[r.get("allocation_point_gap", r.get("error", "COUNTERPART_VALUE_CONFLICT"))
                                     for r in rows],
                        net_asset_counterpart_values=[_stored(v) for v in sorted(nav_values)])
        else:
            # 同一经济日的年报/四季报不重复入均值。不同版本条款只输出各自
            # 条件距离；不能按距离最好/最差挑合同，更不能把发布日期当生效日。
            point = dict(points[0]["allocation_asset_point"])
            if nav_points:
                # 已识别的净资产必须逐源相等；缺位来源保留未知，不让来源
                # 顺序决定是否有净资产场景，也不把旧版布尔核验当作净资产。
                chosen_nav = min(nav_points, key=lambda row: row["source_hashes"][0])
                point["net_asset_evidence"] = chosen_nav["allocation_asset_point"]["net_asset_evidence"]
                point["net_asset_supporting_source_hashes"] = sorted({
                    h for row in nav_points for h in row["source_hashes"]})
                point.pop("net_assets_gap", None)
            eligible = [c for c in contracts if c["published_on"] <= day]
            projections = []
            for contract in eligible:
                clauses = contract["clauses"]
                low, high, hk = (Fraction(int(clauses[k]["numerator"]), int(clauses[k]["denominator"]))
                                 for k in ("stock_lower", "stock_upper", "hk_of_stock_upper"))
                denominator = clauses["ratio_denominator_evidence"]
                basis = denominator["economic_basis"]
                scenarios = (["TOTAL_ASSETS", "NET_ASSETS"] if basis is None else [basis])
                for scenario in scenarios:
                    net = point.get("net_asset_evidence")
                    if scenario == "NET_ASSETS" and not net:
                        continue
                    # 坐标权重仍按总资产求和为1。若场景采用净资产为合同
                    # 分母，缩放合同阈值，不能将按净资产权重误塞入simplex。
                    scale = (_number(net["net_assets_cny"]) / _number(point["amounts_cny"]["total_assets"])
                             if scenario == "NET_ASSETS" else Fraction(1))
                    result = project_half_l1(
                        [Fraction(int(w["numerator"]), int(w["denominator"])) for w in point["weights"]],
                        [{"coefficients": [1, 1, 0], "upper": high*scale},
                         {"coefficients": [-1, -1, 0], "upper": -low*scale},
                         {"coefficients": [-hk, 1-hk, 0], "upper": 0}])
                    projections.append({"contract_source_hashes": contract["source_hashes"], **result,
                        "asset_coordinate_basis": "TOTAL_ASSETS", "contract_denominator_scenario": scenario,
                        "denominator_evidence": denominator,
                        "denominator_assumption": ("HYPOTHETICAL_FUND_ASSETS_EQUALS_" + scenario
                                                   if basis is None else None),
                        "conditions": ["EFFECTIVE_CONTRACT_VERSION_NOT_BOUND", "FULL_ALLOWED_SET_UNRESOLVED"] +
                                      (["CONTRACT_RATIO_DENOMINATOR_UNRESOLVED"] if basis is None else []),
                        "source_threshold_rescaling": _stored(scale)})
            item.update(asset_point=point, conditional_subset_projections=projections,
                        reason="FULL_ALLOWED_SET_EFFECTIVE_VERSION_AND_DENOMINATOR_NOT_RESOLVED" if projections
                               else "CONTRACT_DENOMINATOR_NET_ASSET_POINT_REQUIRED" if eligible
                               else "NO_PRIOR_PARSED_CONTRACT_CANDIDATE",
                        full_metric_value=None,
                        missing_constraints=["CASH_AND_REQUIRED_MARGIN", "OTHER_CONTRACT_PORTFOLIO_LIMITS",
                                             "EFFECTIVE_CONTRACT_VERSION_BINDING"] +
                            (["CONTRACT_RATIO_DENOMINATOR_UNRESOLVED"] if any(
                                c["clauses"]["ratio_denominator_evidence"]["economic_basis"] is None for c in eligible)
                             else []),
                        net_asset_point_gap=point.get("net_assets_gap"))
        observations.append(item)
    return {"version": VERSION, "window_required_report_dates": sorted(set(required_dates)),
            "contract_candidates": contracts, "contract_parse_gaps": gaps,
            "amendment_notice_evidence": amendment_notices,
            "observations": observations,
            "observation_count": len(observations), "independent_products_added": 0,
            "scope": "CONDITIONAL_CONTRACT_SUBSET_DISTANCE_NOT_FULL_ALLOCATION_METRIC",
            "scoring_metric": None}
