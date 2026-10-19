"""分红原件、份额与日收益恒等式反例；测试算例不能进入真实校准。"""

import copy
import hashlib
import json
from datetime import date

import pytest
from app.services import fund_quality_v2_cash_reconciliation as module

NOTICE = (
    "万家中证500指数增强型发起式证券投资基金分红公告 公告送出日期2023年6月20日 "
    "下属分级基金的基金简称万家中证500指数增强发起式A万家中证500指数增强发起式C "
    "下属分级基金的交易代码006729006730 "
    "本次下属分级基金分红方案（单位：元/10份基金份额）2.000 1.000 注： "
    "权益登记日2023年6月21日 除息日2023年6月21日 "
    "注册登记机构将以2023年6月21日除息后的基金份额净值为基准计算 "
    "本次分红免收分红手续费和红利再投资费用"
)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("006729006730", "006730006729"),
        ("元/10份基金份额", "元/份基金份额"),
        ("将以2023年6月21日", "将以2023年6月22日"),
        ("免收分红手续费和红利再投资费用", "收取红利再投资费用"),
    ],
)
def test_share_order_units_reinvestment_date_and_net_fee_must_be_explicit(before, after):
    with pytest.raises(ValueError):
        module.wj_cash_notice([NOTICE.replace(before, after)], fund_code="006730", as_of=date(2023, 6, 22))


def test_original_per_ten_shares_is_divided_by_ten_and_as_of_is_not_receipt_date():
    event = module.wj_cash_notice([NOTICE], fund_code="006730", as_of=date(2023, 6, 22))
    assert event["cash_per_share_cny"] == "0.100" and event["ex_date"] == "2023-06-21"
    with pytest.raises(ValueError, match="AFTER_CUTOFF"):
        module.wj_cash_notice([NOTICE], fund_code="006730", as_of=date(2023, 6, 19))


def inputs(tmp_path, monkeypatch):
    days = ["2023-06-20", "2023-06-21", "2023-06-22"]
    curve = {"dates": days, "month_ends": days, "adjusted_nav": ["1", "1.001", "1.102"]}
    own = {"fund_code": "006730", "history": {"rows": [
        {"nav_date": day, "unit_nav": value} for day, value in zip(days, ["1", ".9", ".99"], strict=True)
    ]}}
    benchmark = {"code": "ETS-5BP0000", "results": [
        {"tradeDate": day, "unitNv": unit, "returnRatio": value}
        for day, unit, value in zip(days, ["1", ".9", ".99"], ["0", "0", "10"], strict=True)
    ]}
    ref = {
        "url": "https://www.wjasset.com/common-web/ets40/yieldListQueryAPI",
        "params": {"fundCode": "006730", "incomeType": "8"}, "format": "ISSUER_DAILY_BENCHMARK_WJ",
    }
    snapshot = {"fund_code": "006730", "dividend_records": [{
        "ex_date": days[1], "cash_dividend": ".1", "content_hash": "TEST_ONLY", "process_status": "实施",
    }]}
    original = b"%PDF-TEST-ONLY-NOT-CALIBRATION"
    (tmp_path / "notice.pdf").write_bytes(original)
    notices = [{"file": "notice.pdf", "sha256": hashlib.sha256(original).hexdigest(),
                "url": "https://www.wjasset.com/upload/pdf/test.pdf", "published_on": "2023-06-20"}]
    monkeypatch.setattr(module, "public_document", lambda *args: ([NOTICE], None))
    return curve, own, benchmark, ref, snapshot, notices


def run(args, tmp_path):
    curve, own, benchmark, ref, snapshot, notices = args
    return module.reconcile_cash_history(curve, own, json.dumps(benchmark).encode(), ref, snapshot,
                                         notices, tmp_path, fund_code="006730", as_of=date(2023, 6, 22))


def test_derived_history_is_separate_and_source_arrays_are_never_overwritten(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch)
    before = copy.deepcopy(args)
    result = run(args, tmp_path)
    assert args == before
    assert result["provider_adjusted_nav"] == ["1", "1.001", "1.102"]
    assert result["curve"]["adjusted_nav"] == ["1", "1", "1.1"]
    assert result["maximum_reinvestment_difference"] == "0"


@pytest.mark.parametrize("failure", ["missing_notice", "wrong_cash", "wrong_unit", "missing_event"])
def test_status_flags_cannot_replace_notice_and_daily_numeric_identity(tmp_path, monkeypatch, failure):
    args = list(inputs(tmp_path, monkeypatch))
    if failure == "missing_notice":
        args[-1] = []
        args[-2]["verified"] = True
    elif failure == "wrong_cash":
        args[-2]["dividend_records"][0]["cash_dividend"] = ".2"
    elif failure == "wrong_unit":
        args[1]["history"]["rows"][1]["unit_nav"] = ".91"
    else:
        # 仅修改公告和事件快照也无法制造收益：真实日净值/发行人收益不符合即拒收。
        args[-2]["dividend_records"][0]["cash_dividend"] = ".09"
        monkeypatch.setattr(module, "public_document", lambda *a: ([NOTICE.replace("1.000 注", "0.900 注")], None))
    with pytest.raises(ValueError):
        run(args, tmp_path)
