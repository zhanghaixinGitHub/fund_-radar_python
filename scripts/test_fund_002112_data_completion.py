"""补数校验的关键反例，不访问网络、数据库或封存标签。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.fund_002112_data_completion_financial_v1 import parse
from scripts.fund_002112_data_completion_facts_v1 import parse_buyback,purpose
from scripts.fund_002112_data_completion_v1 import save,sha,fetch
from scripts.fund_002112_data_completion_public_audit_v2 import decode,norm,day
from scripts.fund_002112_data_completion_forecast_literals import clauses

class CompletionBoundaries(unittest.TestCase):
    def test_two_column_summary_never_invents_prior_amount(self):
        rows,_=parse(['主要会计数据 单位：元 本报告期 本报告期比上年同期增减(%)\n归属于上市公司股东的净利润 100 10.00'],'2023年半年度报告摘要')
        self.assertEqual(len(rows[0]['cells']),2)
        self.assertEqual(rows[0]['cells'][1]['column']['kind'],'YOY_PERCENT')
        self.assertIsNone(rows[0]['currency']);self.assertIsNone(rows[0]['arithmetic_check'])

    def test_quarter_and_ytd_adjusted_comparatives_both_reconcile(self):
        rows,_=parse(['主要会计数据 单位：人民币元 本报告期 上年同期 调整前 调整后 增减(%) 年初至报告期末 上年初至上年报告期末 调整前 调整后 增减(%)\n归属于上市公司股东的净利润 100 90 80 25% 200 180 160 25%'],'2023年第三季度报告')
        self.assertEqual(rows[0]['cells'][0]['column']['period'],'2023-Q3')
        self.assertEqual(rows[0]['cells'][4]['column']['period'],'2023-YTD9')
        self.assertTrue(rows[0]['arithmetic_check']['passed']);self.assertTrue(rows[0]['cumulative_arithmetic_check']['passed'])

    def test_bad_ytd_percentage_not_hidden_by_good_quarter(self):
        rows,gaps=parse(['主要会计数据 单位：元 本报告期 上年同期 调整前 调整后 增减(%) 年初至报告期末 上年初至上年报告期末 调整前 调整后 增减(%)\n归属于上市公司股东的净利润 100 90 80 25% 200 180 160 40%'],'2023年第三季度报告')
        self.assertFalse(rows);self.assertEqual(gaps[0]['reason'],'CUMULATIVE_PERCENTAGE_RECONCILIATION_FAILED')

    def test_previous_and_revised_forecasts_remain_separate(self):
        rows=clauses(['前次业绩预告情况。预计归属于上市公司股东的净利润22亿元到24亿元。更正后的业绩预告情况。预计归属于上市公司股东的净利润18.6亿元左右。'],'2023年业绩预告更正公告')
        self.assertEqual([r['reference_role'] for r in rows],['PREVIOUS_FORECAST_QUOTED','REVISED_FORECAST'])
        self.assertEqual(rows[0]['amount']['second'],'24');self.assertTrue(rows[1]['approximate'])

    def test_profit_decline_does_not_become_current_loss_amount(self):
        rows=clauses(['本期业绩预告情况。归属于上市公司股东的净利润与上年同期相比，减少50%左右。'],'2023年年度业绩预减公告')
        self.assertIsNone(rows[0]['amount']);self.assertEqual(rows[0]['yoy']['direction'],'减少')

    def test_explicit_net_loss_preserves_negative_meaning(self):
        rows=clauses(['本期业绩预告情况。归属于上市公司股东的净亏损约为32亿元到39亿元。'],'2023年半年度业绩预亏公告')
        self.assertEqual(rows[0]['metric'],'NET_LOSS_ATTRIBUTABLE_TO_ISSUER_SHAREHOLDERS')
        self.assertEqual(rows[0]['amount']['profit_loss_role'],'LOSS_EXPLICIT')
        self.assertEqual(rows[0]['amount']['second'],'39')

    def test_currency_after_profit_word_is_retained(self):
        rows=clauses(['本期业绩预告情况。归属于上市公司股东的净利润盈利：人民币2,050,000万元–2,250,000万元。'],'2023年前三季度业绩预告')
        self.assertEqual(rows[0]['amount']['currency'],'CNY');self.assertEqual(rows[0]['amount']['second'],'2,250,000')

    def test_percentage_unit_in_header(self):
        rows,gaps=parse(['主要会计数据 单位：元 币种：人民币 2022年 2021年 本年比上年增减(%) 2020年\n归属于上市公司股东的净利润 400 350 14.29 300'],'2022年年度报告')
        self.assertEqual(len(rows),1);self.assertTrue(rows[0]['arithmetic_check']['passed'])
        self.assertEqual(rows[0]['cells'][2]['original'],'14.29')

    def test_quarter_three_cumulative_is_not_single_quarter(self):
        rows,_=parse(['主要财务数据 单位：万元 币种：人民币 年初至报告期末 上年初至上年报告期末 比上年同期增减(%)\n归属于上市公司股东的净利润 110 100 10.00'],'2023年第三季度报告')
        self.assertEqual(rows[0]['cells'][0]['column']['period'],'2023-YTD9')

    def test_quarter_breakout_not_annual_comparison(self):
        rows,gaps=parse(['主要会计数据 单位：元 币种：人民币 2022年 2021年 增减(%) 2020年 报告期分季度的主要会计数据 第一季度 第二季度 第三季度 第四季度\n归属于上市公司股东的净利润 100 120 130 140'],'2022年年度报告')
        self.assertFalse(rows);self.assertEqual(gaps[0]['reason'],'QUARTER_BREAKOUT_NOT_ANNUAL_COMPARISON')

    def test_adjustment_columns_not_silently_chosen(self):
        rows,gaps=parse(['主要会计数据 单位：元 2022年 2021年 调整前 调整后 本年比上年增减(%)\n归属于上市公司股东的净利润 100 90 80 25'],'2022年年度报告')
        self.assertFalse(rows);self.assertEqual(gaps[0]['reason'],'ADJUSTMENT_COLUMNS_REQUIRE_EXPLICIT_LAYOUT')

    def test_explicit_adjusted_columns_are_both_preserved(self):
        rows,_=parse(['主要会计数据 单位：人民币元 2022年 2021年 本年比上年增减(%) 2020年 调整后 调整前 调整后 调整前\n归属于上市公司股东的净利润 110 100 90 10.00 80 70'],'二〇二二年年度报告')
        self.assertEqual([c['column'].get('adjustment') for c in rows[0]['cells']],[None,'调整后','调整前',None,'调整后','调整前'])
        self.assertEqual(rows[0]['arithmetic_check']['prior_column'],1)
        self.assertTrue(rows[0]['arithmetic_check']['passed'])

    def test_no_repurchase_does_not_impute_amount(self):
        rows=parse_buyback(['截至2023年9月30日，公司尚未实施回购股份。'])
        self.assertEqual(rows[0]['stage'],'NOT_IMPLEMENTED');self.assertIsNone(rows[0]['amount'])

    def test_customer_obligation_and_forecast_types(self):
        self.assertEqual(purpose('关于为购车客户提供回购责任的公告','BUYBACK'),'CUSTOMER_FINANCING_REPURCHASE_LIABILITY')
        self.assertEqual(purpose('2023年半年度业绩预亏公告','PERFORMANCE'),'EARNINGS_FORECAST')

    def test_gb_charset_and_catalog_media_prefix(self):
        text,enc=decode('<meta charset="gb2312"><title>医保公告</title>'.encode('gb2312'))
        self.assertIn('医保公告',text);self.assertEqual(norm('【人民网】医保公告'),norm('医保公告'))
        self.assertEqual(day('2021年09月17日09:56'),'2021-09-17')

    def test_duplicate_receipt_does_not_issue_request(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);url='https://example.test/evidence';key=hashlib.sha256(url.encode()).hexdigest()
            save(root/'receipts'/f'{key}.json',{'ok':True,'url':url,'cached_proof':'retained'})
            with patch('scripts.fund_002112_data_completion_v1.OUT',root),patch('httpx.Client',side_effect=AssertionError('network must not be called')):
                self.assertEqual(fetch(url,'historical')['cached_proof'],'retained')
            self.assertFalse((root/'requests').exists())

if __name__=='__main__':unittest.main()
