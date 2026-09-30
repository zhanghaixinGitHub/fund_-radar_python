"""原页复核发现的跨行扣非、修正前列和季度期间反例。"""
import unittest
from scripts.fund_002112_data_completion_forecast_context import metric_from_complete_label, exact_periods, named_periods, column_role


class ForecastContextBoundaries(unittest.TestCase):
    def test_deducted_profit_label_does_not_become_parent_profit(self):
        self.assertEqual(metric_from_complete_label('扣除非经常性损益后的\n归属于上市公司股东的\n净利润'),
                         'NET_PROFIT_EXCLUDING_NON_RECURRING_ITEMS')

    def test_previous_column_is_not_revised_forecast(self):
        self.assertEqual(column_role('2020年1月1日~2020年\n修正前'), 'PREVIOUS_FORECAST_QUOTED')
        self.assertEqual(column_role('修正后'), 'REVISED_FORECAST')

    def test_double_dash_is_valid_explicit_date_separator(self):
        self.assertEqual(exact_periods('2021年07月01日--2021年09月30日')[0]['start'], '2021-07-01')

    def test_nearest_quarter_is_separate_from_year_to_date(self):
        rows = named_periods('2023年前三季度主要财务数据。（二）2023年第三季度主要财务数据和指标')
        self.assertEqual(rows[0]['start'], '2023-01-01')
        self.assertEqual(rows[-1]['start'], '2023-07-01')

    def test_explicit_month_range_does_not_require_day_text(self):
        row = exact_periods('2022年1月-2022 年3月')[0]
        self.assertEqual((row['start'], row['end']), ('2022-01-01', '2022-03-31'))


if __name__ == '__main__':
    unittest.main()
