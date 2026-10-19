"""读取合同基准的组成、生效变更及日收益计算方法；不生成替代基准曲线。"""

import hashlib
import re
import unicodedata
from datetime import date
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

from app.services.fund_rating_public_market import public_document

VERSION = "QUALITY_BENCHMARK_CONTRACT_20261017_1"
CMF_TITLE = "招商基金管理有限公司关于旗下部分基金调整业绩比较基准并修订基金合同等法律文件的公告"
# 当前适配范围只覆盖已实际检视的两只招商产品的旧/新三要素。
# 未知文字不能被吞进“指数名称”；扩展其他名称须先检视其原件及身份。
INDEX_NAMES = ('沪深300指数', '中证800指数', '恒生综合指数',
               '中债综合(全价)指数', '中债-综合全价(1-3年)指数')


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def weighted_index_expression(expression: str) -> list[dict]:
    """逐项读取后置百分比公式，权重以比例存储；不得猜指数代码或币种。

    此适配器只接受完整的1至6项指数收益率表达式。加号、括号内的汇率
    说明及全部字节都须被消费，权重合计必须恰为100%，重复成分拒绝。
    """
    text = _compact(expression)
    if not text or len(text) > 1500:
        raise ValueError("BENCHMARK_EXPRESSION_SIZE")
    term = re.compile(r"(?P<name>" + '|'.join(map(re.escape, INDEX_NAMES)) + r")收益率"
                      r"(?P<note>\([^()]{1,30}\))?[*×](?P<percent>\d+(?:\.\d+)?)%")
    components, cursor = [], 0
    while cursor < len(text) and len(components) < 6:
        match = term.match(text, cursor)
        if match is None:
            raise ValueError("BENCHMARK_EXPRESSION_UNCONSUMED_TEXT")
        weight = Decimal(match['percent'])/100
        if not 0 < weight <= 1:
            raise ValueError("BENCHMARK_EXPRESSION_WEIGHT_RANGE")
        components.append({'index_name': match['name'], 'weight': str(weight),
                           'fx_literal': match['note'] or None})
        cursor = match.end()
        if cursor < len(text):
            if text[cursor] not in '+＋':
                raise ValueError("BENCHMARK_EXPRESSION_TERM_SEPARATOR")
            cursor += 1
            if cursor == len(text):
                raise ValueError("BENCHMARK_EXPRESSION_TRAILING_SEPARATOR")
    if (cursor != len(text) or sum(Decimal(c['weight']) for c in components) != 1
            or len({c['index_name'] for c in components}) != len(components)):
        raise ValueError("BENCHMARK_EXPRESSION_WEIGHT_SUM_OR_DUPLICATE")
    return components


def cmf_benchmark_amendment(pages: list[str], *, legal_name: str, catalog_published_on: str,
                            as_of: date) -> dict:
    """从公告主表读取具名行旧/新三要素和两处一致的实际生效日期。

    不读取附件里的解释性重复公式。目录日期与正文落款不一致时保留两者，
    以较晚日期作保守可用日下界；它仍不证明首次HTTP公开时间或完整版本链。
    """
    text = _compact('\n'.join(pages))
    if (not pages or CMF_TITLE not in _compact(pages[0])[:200] or not legal_name
            or len(pages) > 100):
        raise ValueError("BENCHMARK_CHANGE_ISSUER_TITLE_OR_LEGAL_NAME")
    table_heads = list(re.finditer("序号基金全称原基金合同业绩比较基准调整后的新业绩比较基准", text))
    table_end = text.find("上述基金调整业绩比较基准要素的原因")
    if len(table_heads) != 1 or table_end <= table_heads[0].end():
        raise ValueError("BENCHMARK_CHANGE_MAIN_TABLE_BOUNDARY")
    table = text[table_heads[0].end():table_end]
    rows = list(re.finditer(r"(?<!\d)(\d{1,2})" + re.escape(legal_name)
                           + r"(?P<body>.+?)(?=\d{1,2}招商|$)", table))
    if len(rows) != 1:
        raise ValueError("BENCHMARK_CHANGE_NAMED_ROW_NOT_UNIQUE")
    # 本表适配的是两组三要素的后置权重公式。先按第3个百分号分列，
    # 再各自完整解析，不能把六个权重当成一个公式或按当前新名称回填旧列。
    body = rows[0]['body']
    stops = [m.end() for m in re.finditer('%', body)]
    if len(stops) != 6:
        raise ValueError("BENCHMARK_CHANGE_THREE_COMPONENT_COLUMNS_REQUIRED")
    old, new = body[:stops[2]], body[stops[2]:]
    old_components, new_components = weighted_index_expression(old), weighted_index_expression(new)
    if len(old_components) != 3 or len(new_components) != 3 or old == new:
        raise ValueError("BENCHMARK_CHANGE_OLD_NEW_FORMULAS_UNBOUND")
    initial = list(re.finditer(r"决定自(20\d{2})年(\d{1,2})月(\d{1,2})日起调整旗下部分基金的业绩比较基准", text))
    effective = list(re.finditer(r"三、上述基金修订后的基金合同、托管协议\(如有\)自"
                                r"(20\d{2})年(\d{1,2})月(\d{1,2})日起生效", text))
    dated = list(re.finditer(r"特此公告[。.]招商基金管理有限公司(20\d{2})年(\d{1,2})月(\d{1,2})日", text))
    if len(initial) != 1 or len(effective) != 1 or len(dated) != 1:
        raise ValueError("BENCHMARK_CHANGE_EFFECTIVE_AND_DOCUMENT_DATE_UNPARSED")
    day, first, document_date = [date(*map(int, m.groups())) for m in (effective[0], initial[0], dated[0])]
    catalog_date = date.fromisoformat(catalog_published_on)
    available = max(catalog_date, document_date)
    if day != first or day < available:
        raise ValueError("BENCHMARK_CHANGE_EFFECTIVE_DATE_CONFLICT")
    if available > as_of:
        raise ValueError("BENCHMARK_CHANGE_DOCUMENT_NOT_AVAILABLE_BY_CUTOFF")
    return {'version': VERSION, 'legal_name': legal_name, 'table_row': int(rows[0][1]),
            'catalog_published_on': str(catalog_date), 'document_dated_on': str(document_date),
            'availability_date_floor': str(available),
            'publication_date_state': 'CATALOG_AND_DOCUMENT_DATES_DIFFER' if catalog_date != document_date
                else 'CATALOG_AND_DOCUMENT_DATES_AGREE',
            'effective_on': str(day), 'state_at_as_of': 'EFFECTIVE' if day <= as_of else 'ANNOUNCED_FUTURE_CHANGE',
            'old_expression': old, 'new_expression': new, 'old_components': old_components,
            'new_components': new_components, 'named_row_excerpt': rows[0][0], 'effective_excerpt': effective[0][0],
            'named_row_pages': [i+1 for i, p in enumerate(pages) if rows[0][0] in _compact(p)],
            'scope': 'NAMED_EFFECTIVE_TRANSITION_NOT_COMPLETE_PRIOR_OR_LATER_VERSION_CHAIN',
            'new_formula_backcast_allowed': False}


def contract_calculation(pages: list[str], expression: str) -> dict:
    """读取招募书给出的指数代码和日加权连乘、估值汇率方法，代码不加市场后缀。"""
    text = _compact('\n'.join(pages))
    if _compact(expression) not in text:
        raise ValueError("BENCHMARK_CALCULATION_FORMULA_NOT_PRESENT_IN_DOCUMENT")
    components = weighted_index_expression(expression)
    heading = '3、业绩比较基准的计算方法'
    if text.count(heading) != 1 or text.count('2、业绩比较基准要素的基本信息') != 1:
        raise ValueError("BENCHMARK_CALCULATION_NUMBERED_SECTIONS_UNPARSED")
    before, method = text.split(heading)
    method = method.split('4、基金管理人管理投资偏离业绩比较基准', 1)
    if len(method) != 2:
        raise ValueError("BENCHMARK_CALCULATION_SECTION_END")
    method = method[0]
    required = ['每日收益率为基础', '时间加权为计算原则', '按照预设权重比例', '连乘每日收益率',
                '恒生综合指数收益率采用估值汇率进行调整', '由港元计价折算为人民币计价后']
    if not all(clause in method for clause in required):
        raise ValueError("BENCHMARK_DAILY_CHAIN_AND_FX_METHOD_UNRESOLVED")
    info = before.split('2、业绩比较基准要素的基本信息')[1]
    for component in components:
        pattern = re.escape(component['index_name']) + r"由([^。]{3,100}?)编制发布,指数代码为([A-Za-z0-9]{3,20}),"
        matches = list(re.finditer(pattern, info))
        if len(matches) != 1:
            raise ValueError("BENCHMARK_LITERAL_INDEX_CODE_UNBOUND:" + component['index_name'])
        component.update(publisher=matches[0][1], literal_code=matches[0][2],
                         identity_excerpt=matches[0][0])
    return {'version': VERSION, 'components': components, 'method_excerpt': method,
            'calculation': 'DAILY_WEIGHTED_COMPONENT_RETURNS_CHAINED',
            'foreign_component_conversion': 'HKD_TO_CNY_USING_FUND_VALUATION_FX',
            'raw_series_available': False,
            'scope': 'CONTRACT_METHOD_NOT_COMPONENT_PRICES_OR_COMPLETE_EFFECTIVE_HISTORY'}


def benchmark_transitions(*, refs: list[dict], root: Path, legal_name: str, as_of: date, code: str) -> dict:
    """逐件核公告实际原字节、官网地址、正文日期与具名行，保留未解析原件。"""
    events, gaps = [], []
    for ref in refs:
        if ref.get('label') != 'historical-contract' or _compact(ref.get('title', '')) != CMF_TITLE:
            continue
        try:
            address = urlsplit(ref['url'])
            if (address.scheme != 'https' or address.hostname != 'static.cmfchina.com'
                    or address.username or address.password or address.port not in {None, 443}
                    or address.query or address.fragment
                    or not re.fullmatch(r'/fundarticle/20\d{6}/\d+\.pdf', address.path)):
                raise ValueError('BENCHMARK_CHANGE_ISSUER_ENDPOINT')
            file = (root/ref['file']).resolve()
            if (not file.is_relative_to((root/'sources').resolve()) or not file.is_file()
                    or file.stat().st_size > 25_000_000):
                raise ValueError('BENCHMARK_CHANGE_SOURCE_PATH_OR_SIZE')
            raw = file.read_bytes()
            if hashlib.sha256(raw).hexdigest() != ref['sha256']:
                raise ValueError('BENCHMARK_CHANGE_ACTUAL_SOURCE_HASH')
            pages, _ = public_document(raw, ref, code, '公告')
            event = cmf_benchmark_amendment(pages, legal_name=legal_name,
                                          catalog_published_on=ref['published_on'], as_of=as_of)
            events.append(event | {'source_hashes': [ref['sha256']], 'source_url': ref['url'],
                                   'acquired_at': ref['acquired_at']})
        except (ValueError, KeyError, OSError) as exc:
            gaps.append({'reason': str(exc), 'source_hashes': [ref.get('sha256')], 'source_url': ref.get('url')})
    return {'version': VERSION, 'events': events, 'unresolved_originals': gaps,
            'effective_history_complete': False, 'scoring_metric': None,
            'remaining': 'BOUND_INITIAL_CONTRACT_AND_ALL_EFFECTIVE_CHANGES_REQUIRED_NOT_LATEST_FORMULA_BACKCAST'}
