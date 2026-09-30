"""回读预告原表的完整行名、期间及修正列，旧抽取结果保留供差异追溯。"""
import calendar
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pdfplumber
from scripts.fund_002112_data_completion_v1 import OUT, read, save, sha, now


def compact(text):
    return re.sub(r'\s+', '', text or '')


def metric_from_complete_label(label):
    """以金额单元格覆盖的完整行名区分扣非，不能只读包含“归属”的半行。"""
    label = compact(label)
    if '扣除非经常性损益' in label or '扣非' in label:
        return 'NET_PROFIT_EXCLUDING_NON_RECURRING_ITEMS'
    if '净亏损' in label:
        return 'NET_LOSS_ATTRIBUTABLE_TO_ISSUER_SHAREHOLDERS'
    if '归属' in label and '净利润' in label:
        return 'NET_PROFIT_ATTRIBUTABLE_TO_PARENT'
    return None


def exact_periods(text):
    """返回原文中的明确起止日期；双连字符是日期分隔符而非负数。"""
    result = []
    for m in re.finditer(r'(20\d{2})年(\d{1,2})月(\d{1,2})日[至—－~～–-]+(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})日', compact(text)):
        try:
            lo = date(int(m[1]), int(m[2]), int(m[3]))
            hi = date(int(m[4] or m[1]), int(m[5]), int(m[6]))
        except ValueError:
            continue
        if lo <= hi:
            result.append({'start': str(lo), 'end': str(hi), 'quote': m[0]})
    if not result:
        for m in re.finditer(r'(20\d{2})年(\d{1,2})月?[至—－~～–-]+(?:(20\d{2})年)?(\d{1,2})月', compact(text)):
            year, first, finalyear, last = int(m[1]), int(m[2]), int(m[3] or m[1]), int(m[4])
            if 1 <= first <= 12 and 1 <= last <= 12:
                lo = date(year, first, 1); hi = date(finalyear, last, calendar.monthrange(finalyear, last)[1])
                if lo <= hi:
                    result.append({'start': str(lo), 'end': str(hi), 'quote': m[0]})
    return result


def named_periods(text):
    """仅解释明确年和期间；第三季度独立为七至九月。"""
    spans = {'前三季度': (1, 9), '第三季度': (7, 9), '第一季度': (1, 3), '第四季度': (10, 12),
             '半年度': (1, 6), '上半年': (1, 6), '年度': (1, 12), '年报': (1, 12)}
    result = []
    for m in re.finditer(r'(20\d{2})年?(前三季度|第三季度|第一季度|第四季度|半年度|上半年|年度|年报)', compact(text)):
        year = int(m[1]); first, last = spans[m[2]]
        result.append({'start': str(date(year, first, 1)),
                       'end': str(date(year, last, calendar.monthrange(year, last)[1])), 'quote': m[0]})
    return result


def column_role(header):
    text = compact(header)
    if any(w in text for w in ['修正前', '更正前', '前次预测', '前次预告']):
        return 'PREVIOUS_FORECAST_QUOTED'
    if any(w in text for w in ['修正后', '更正后']):
        return 'REVISED_FORECAST'
    return 'CURRENT_REPORT_COLUMN'


def run():
    version = sys.argv[1] if len(sys.argv) > 1 else 'v5'
    totals = []; changes = []
    for source in sorted((OUT/'forecast-final-v4').glob('*.json')):
        record = read(source); dest = OUT/('forecast-final-'+version)/source.name
        if dest.exists():
            totals.append(read(dest)); continue
        doc = read(record['source']); raw = Path(doc['receipt']['path'])
        assert sha(raw) == record['raw_sha256']
        facts = []
        with pdfplumber.open(raw) as pdf:
            for original in record['facts']:
                f = dict(original); page = pdf.pages[f['page']-1]
                current = f.get('current_cell_rectangle') or f['cell_rectangles'][f['current_column']]
                labels = f.get('label_cell_rectangles') or [f['cell_rectangles'][i] for i, cell in enumerate(f['row_cells'])
                                                          if cell and '归属' in compact(cell) and f['cell_rectangles'][i]]
                # 包含金额单元格的完整高度，补回在“归属”前一行的扣非限定词。
                rect = (min(x[0] for x in labels), min(current[1], min(x[1] for x in labels)),
                        max(x[2] for x in labels), max(current[3], max(x[3] for x in labels)))
                label = page.crop(rect).extract_text() or ''
                header = page.crop((current[0], f['table_bbox'][1], current[2], current[1])).extract_text() or ''
                context = page.crop((0, 0, page.width, f['table_bbox'][1])).extract_text() or ''
                if f['page'] > 1:
                    # 上一页表格之后的章节标题可能属于下一页的续表，记录原页来源。
                    prev = pdf.pages[f['page']-2]
                    prevtables = [x['table_bbox'][3] for x in record['facts'] if x['page'] == f['page']-1]
                    if prevtables:
                        tail = prev.crop((0, max(prevtables), prev.width, prev.height)).extract_text() or ''
                        context = tail + '\n' + context
                metric = metric_from_complete_label(label)
                if metric is None:
                    # 数值格只占合并行下半部时，旧完整行名仍保留，未凭邻行补猜。
                    metric = f['metric']
                role = column_role(header)
                period = f['accounting_period']; basis = f['effective_period_basis']
                candidates = exact_periods(header)
                if len(candidates) == 1:
                    period = candidates[0]; basis = 'EXACT_CURRENT_COLUMN_PERIOD'
                elif len(named_periods(header)) == 1:
                    period = named_periods(header)[0]; basis = 'EXPLICIT_CURRENT_COLUMN_NAMED_PERIOD'
                elif period is None and exact_periods(context):
                    period = exact_periods(context)[-1]; basis = 'NEAREST_EXPLICIT_DATE_RANGE_BEFORE_TABLE'
                elif period is None and named_periods(context):
                    period = named_periods(context)[-1]; basis = 'NEAREST_EXPLICIT_NAMED_PERIOD_BEFORE_TABLE'
                # 本报告期只对应当表的章节；不把整篇累计标题套在每个季度表上。
                units = list(re.finditer(r'单位[：:]?(人民币)?(百万元|千元|万元|亿元|元)', compact(context)))
                unit = units[-1][2] if units else None
                currency = 'CNY' if units and units[-1][1] else None
                stage = f['stage']
                if role == 'PREVIOUS_FORECAST_QUOTED':
                    stage = 'PREVIOUS_FORECAST_QUOTED'
                elif '快报' in record['title'] and '主要财务数据和指标' in compact(context):
                    stage = 'PRELIMINARY_EARNINGS'
                f.update({'original_metric_before_full_label_replay': f['metric'], 'metric': metric,
                          'complete_metric_label': label, 'complete_metric_label_rectangle': rect,
                          'current_column_context': header, 'current_column_reference_role': role,
                          'table_preceding_context': context, 'accounting_period': period,
                          'period_verified': bool(period) or f['period_verified'], 'effective_period_basis': basis,
                          'explicit_table_unit_replayed': unit, 'explicit_table_currency_replayed': currency,
                          'original_stage': f['stage'], 'stage': stage,
                          'full_label_and_column_context_replayed': True})
                if metric != original['metric'] or role == 'PREVIOUS_FORECAST_QUOTED' or stage != original['stage']:
                    changes.append({'id': record['id'], 'page': f['page'], 'old_metric': original['metric'],
                                    'metric': metric, 'role': role, 'stage': stage})
                facts.append(f)
        record.update({'at': now(), 'superseded_extraction': str(source), 'superseded_sha256': sha(source), 'facts': facts,
                       'legacy_numeric_extraction_preserved': True, 'training_eligible': False})
        save(dest, record); totals.append(record)
    # 续跑只补未完成文件；汇总从所有已保存记录重建，包含中断前的修正。
    changes = [{'id': r['id'], 'page': f['page'], 'old_metric': f['original_metric_before_full_label_replay'],
                'metric': f['metric'], 'role': f['current_column_reference_role'], 'stage': f['stage']}
               for r in totals for f in r['facts'] if f['metric'] != f['original_metric_before_full_label_replay']
               or f['current_column_reference_role'] == 'PREVIOUS_FORECAST_QUOTED' or f['stage'] != f['original_stage']]
    save(OUT/('forecast-context-result-'+version+'.json'), {'at': now(), 'scope': len(totals),
         'with_grid_facts': sum(bool(r['facts']) for r in totals), 'grid_facts': sum(len(r['facts']) for r in totals),
         'layout_verified_facts': sum(f['independent_layout_verified'] for r in totals for f in r['facts']),
         'period_verified_facts': sum(f['period_verified'] for r in totals for f in r['facts']),
         'with_table_or_literal': sum(bool(r['facts']) or r['literal_fact_count'] > 0 for r in totals),
         'metric_counts': dict(Counter(f['metric'] for r in totals for f in r['facts'])),
         'column_roles': dict(Counter(f['current_column_reference_role'] for r in totals for f in r['facts'])),
         'corrections': changes, 'new_fits': 0})


if __name__ == '__main__':
    run()
