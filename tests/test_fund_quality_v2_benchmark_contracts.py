"""合同基准解析的反例测试；虚构文本仅检验工程边界，不进入真实校准。"""

from datetime import date

import pytest
from app.services.fund_quality_v2_benchmark_contracts import (
    CMF_TITLE,
    cmf_benchmark_amendment,
    contract_calculation,
    weighted_index_expression,
)

NAME = '测试招商股票型证券投资基金'
OLD = '沪深300指数收益率*80%+恒生综合指数收益率(经汇率调整后)*10%+中债综合(全价)指数收益率*10%'
NEW = '中证800指数收益率×80%+恒生综合指数收益率×10%+中债-综合全价(1-3年)指数收益率×10%'


def notice():
    return [CMF_TITLE + '\n决定自2026年6月1日起调整旗下部分基金的业绩比较基准。\n'
            '序号 基金全称 原基金合同业绩比较基准 调整后的新业绩比较基准\n'
            '19 ' + NAME + '\n' + OLD + '\n' + NEW + '\n'
            '上述基金调整业绩比较基准要素的原因详见附件。',
            '三、上述基金修订后的基金合同、托管协议(如有)自2026年6月1日起生效。\n'
            '特此公告。招商基金管理有限公司2026年5月1日',
            '附件：' + NAME + '其他解释公式不能当主表列：' + OLD]


def test_actual_body_date_and_future_effectivity_are_separate_from_catalog():
    result = cmf_benchmark_amendment(notice(), legal_name=NAME,
                                     catalog_published_on='2026-04-30', as_of=date(2026, 5, 1))
    assert result['state_at_as_of'] == 'ANNOUNCED_FUTURE_CHANGE'
    assert result['publication_date_state'] == 'CATALOG_AND_DOCUMENT_DATES_DIFFER'
    assert result['availability_date_floor'] == '2026-05-01'
    assert result['effective_on'] == '2026-06-01'
    assert result['old_components'][1]['fx_literal'] == '(经汇率调整后)'
    assert result['new_components'][0]['index_name'] == '中证800指数'
    assert result['named_row_pages'] == [1] and not result['new_formula_backcast_allowed']
    with pytest.raises(ValueError, match='NOT_AVAILABLE_BY_CUTOFF'):
        cmf_benchmark_amendment(notice(), legal_name=NAME,
                               catalog_published_on='2026-04-30', as_of=date(2026, 4, 30))


@pytest.mark.parametrize('old,new', [
    ('自2026年6月1日起生效', '自2026年6月2日起生效'),
    (NEW, NEW.replace('80%', '75%')),
    (NAME, '别的证券投资基金'),
    (OLD, OLD + '另有未解释条件'),
    ('调整后的新业绩比较基准', '变更情况'),
    ('特此公告。', '无落款'),
])
def test_ambiguous_or_incomplete_original_cannot_bind_transition(old, new):
    with pytest.raises(ValueError):
        cmf_benchmark_amendment([p.replace(old, new) for p in notice()], legal_name=NAME,
                               catalog_published_on='2026-04-30', as_of=date(2026, 9, 30))


@pytest.mark.parametrize('expression', [NEW+'+', NEW+'忽略这句', NEW.replace('80%', '0%'),
                                       NEW.replace('80%', '81%'), NEW.replace('中证800', '恒生综合')])
def test_formula_is_fully_consumed_and_exact_weight_sum_is_required(expression):
    with pytest.raises(ValueError):
        weighted_index_expression(expression)


def calculation():
    return ['本基金的业绩比较基准为：' + NEW + '\n2、业绩比较基准要素的基本信息\n'
            '中证800指数由中证指数有限公司编制发布，指数代码为000906，具体信息见官网。\n'
            '恒生综合指数由恒生指数有限公司编制发布，指数代码为HSCI，具体信息见官网。\n'
            '中债-综合全价（1-3年）指数由中债金融估值中心有限公司编制发布，指数代码为CBA00223，具体信息见官网。\n'
            '3、业绩比较基准的计算方法\n'
            '本基金业绩比较基准收益率的计算方法以每日收益率为基础，以时间加权为计算原则。\n'
            '先分别计算每日收益率，再按照预设权重比例计算当日组合要素基准的日收益率，并连乘每日收益率。\n'
            '其中，恒生综合指数收益率采用估值汇率进行调整，由港元计价折算为人民币计价后，再纳入计算。\n'
            '4、基金管理人管理投资偏离业绩比较基准的定性或定量方法']


def test_literal_codes_and_valuation_fx_are_read_without_guessing_market_suffixes():
    result = contract_calculation(calculation(), NEW)
    assert [c['literal_code'] for c in result['components']] == ['000906', 'HSCI', 'CBA00223']
    assert result['calculation'] == 'DAILY_WEIGHTED_COMPONENT_RETURNS_CHAINED'
    assert result['raw_series_available'] is False
    with pytest.raises(ValueError, match='METHOD_UNRESOLVED'):
        contract_calculation([p.replace('估值汇率', '其他汇率') for p in calculation()], NEW)
    with pytest.raises(ValueError, match='INDEX_CODE_UNBOUND'):
        contract_calculation([p.replace('指数代码为HSCI', '代码另行确认') for p in calculation()], NEW)
    with pytest.raises(ValueError, match='FORMULA_NOT_PRESENT'):
        contract_calculation(calculation(), NEW.replace('80%', '75%').replace('10%', '15%', 1))
