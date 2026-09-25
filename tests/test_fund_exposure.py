"""002112 数据链条的边界验证；合成值只验证口径，不能作为真实预测成绩。"""

from copy import deepcopy
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pytest
from app.integrations.dbfund_reports import official_url, parse_text
from app.services import fund_exposure_features as f
from app.services import fund_exposure_quotes as q
from app.services import fund_exposure_runtime as runtime
from app.services.direction_1d_protocol import ZONE, calendar, label
from app.services.fund_exposure_common import read, save


def report(end="2021-06-30", available="2021-07-22T08:00:00+08:00", full=False):
    return {
        "report_end": end,
        "available_at": available,
        "raw": {"sha256": end + available, "received_at": "2026-09-24T10:00:00+08:00"},
        "parsed_at": "2026-09-24T10:01:00+08:00",
        "full_stock_disclosure": full,
        "stock_nav_pct": "80",
        "disclosed_nav_pct": "15",
        "reported_industries": [],
        "holdings": [
            {"stock_code": "000001.SZ", "nav_weight_pct": "10"},
            {"stock_code": "600000.SH", "nav_weight_pct": "5"},
        ],
    }


def test_report_publication_and_receipt_are_different_gates():
    quarter = report()
    half = report(available="2021-09-01T08:00:00+08:00", full=True)
    before = datetime(2021, 9, 1, 7, 59, tzinfo=ZONE)
    assert f.select_report([quarter, half], before) is quarter
    assert f.select_report([quarter, half], before + timedelta(minutes=1)) is half
    with pytest.raises(ValueError, match="NO_AVAILABLE_REPORT"):
        f.select_report([quarter, half], before, live=True)


def test_late_old_report_does_not_replace_new_period():
    newer = report(end="2021-03-31", available="2021-04-20T08:00:00+08:00")
    older = report(end="2020-12-31", available="2021-04-25T08:00:00+08:00", full=True)
    assert f.select_report([newer, older], datetime(2021, 5, 1, tzinfo=ZONE)) is newer


def bundle():
    sessions, _ = calendar()
    base = date(2021, 8, 20)
    i = sessions.index(base)
    days = {}
    for d in sessions[i - 20 : i + 1]:
        days[str(d)] = {
            "rows": {"000001.SZ": {"pct_chg": 2.0, "amount": 100.0}, "600000.SH": {"pct_chg": -1.0, "amount": 50.0}},
            "receipt": {"sha256": str(d), "received_at": "2021-08-21T01:00:00+08:00"},
        }
    return {"reports": [report()], "days": days, "indices": {}}, base, datetime.combine(sessions[i + 1], time(8), ZONE)


def test_partial_holdings_use_nav_weights_without_renormalizing(monkeypatch):
    monkeypatch.setattr(f, "initialize", lambda: {"max_report_age_days": 210})
    b, base, at = bundle()
    result = f.calculate(b, base, at)
    assert result["holdings_features"][0] == pytest.approx(0.0015)
    assert result["disclosed_nav_weight"] == pytest.approx(0.15)
    assert result["unexplained_nav_weight"] == pytest.approx(0.85)
    assert result["quote_coverage_of_disclosed_weight"] == pytest.approx(1)
    assert result == f.calculate(deepcopy(b), base, at)


def test_missing_quote_invalidates_features_instead_of_becoming_zero(monkeypatch):
    monkeypatch.setattr(f, "initialize", lambda: {"max_report_age_days": 210})
    b, base, at = bundle()
    del b["days"][str(base)]["rows"]["600000.SH"]
    result = f.calculate(b, base, at)
    assert result["holdings_features"] is None
    assert result["quote_coverage_of_disclosed_weight"] == pytest.approx(2 / 3)
    assert result["missing"][0]["stock_code"] == "600000.SH"


def test_historical_next_day_0800_boundary():
    b, base, at = bundle()
    with pytest.raises(ValueError, match="QUOTE_NOT_YET_AVAILABLE"):
        f.calculate(b, base, at - timedelta(seconds=1))


def test_exact_flat_label_and_minimum_class_count():
    assert label("1.00000", "1.00001")["actual_direction"] == "UP"
    assert label("1.00000", "1.00000")["actual_direction"] == "FLAT"
    rows = [
        {
            "target": str(date(2021, 1, 1) + timedelta(days=i)),
            "actual_direction": "FLAT" if i < 16 else "UP" if i < 200 else "DOWN",
        }
        for i in range(400)
    ]
    result = f.training_gate(rows)
    assert not result["eligible"] and result["class_counts"]["FLAT"] == 16


def test_hash_evidence_and_create_once(tmp_path):
    path = tmp_path / "evidence.json"
    save(path, {"value": 1})
    with pytest.raises(FileExistsError):
        save(path, {"value": 2})
    path.write_text(path.read_text().replace('"value":1', '"value":2'))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        read(path)


def report_text():
    return [
        """德邦鑫星价值 002112 报告送出日期：2021年8月31日
7.1 期末基金资产组合情况
1 权益投资 40.00 18.00
2 固定收益投资 - -
7 银行存款和结算备付金合计 60.00 27.00
7.2 期末按行业分类的股票投资组合
C 制造业 4 0.00 20.00
合计 40.00 20.00
7.3 期末按公允价值占基金资产净值比例大小排序的所有股票投资明细
1 000001 公司甲 10 20.00 10.00
1 600000 公司乙 10 20.00 10.00
7.4 报告期内股票投资组合的重大变动
"""
    ]


def test_pdf_tied_ranks_split_amount_and_denominators():
    r = parse_text(report_text(), "德邦鑫星价值2021年中期报告")
    assert r["holding_count"] == 2
    assert r["stock_nav_pct"] == "20.00"
    assert r["assets"]["权益投资"]["total_asset_pct"] == "18.00"
    assert [h["reported_rank"] for h in r["holdings"]] == [1, 1]


@pytest.mark.parametrize("bad", ["2 600000 公司乙 10 19.00 10.00", "1 000001 公司乙 10 20.00 10.00"])
def test_pdf_missing_amount_or_duplicate_security_rejected(bad):
    pages = [report_text()[0].replace("1 600000 公司乙 10 20.00 10.00", bad)]
    with pytest.raises(ValueError, match="REPORT_"):
        parse_text(pages, "德邦鑫星价值2021年中期报告")


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/upload/pdf/a.pdf",
        "http://www.dbfund.com.cn/upload/pdf/a.pdf",
        "https://www.dbfund.com.cn/private",
    ],
)
def test_report_url_scope(url):
    with pytest.raises(ValueError):
        official_url(url)


def test_provider_budget_stops_before_network(monkeypatch):
    monkeypatch.setattr(q, "permission", lambda: {"authorized_api_names": ["daily"]})
    monkeypatch.setattr(q, "initialize", lambda: {"maximum_provider_requests_per_command": 0})
    with pytest.raises(ValueError, match="BUDGET_REACHED"):
        q.Provider().query("daily", {"trade_date": "19000101"}, q.FIELDS)


def test_forward_existing_input_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "specification", lambda: None)
    monkeypatch.setattr(runtime, "window", lambda _: {"target_nav_date": "2026-09-25"})
    path = tmp_path / "forward-inputs/2026-09-25.json"
    save(path, {"kind": "LIVE_INPUT_ONLY_NOT_A_PREDICTION"})
    before = path.read_bytes()
    assert runtime.capture()["status"] == "ALREADY_CAPTURED"
    assert path.read_bytes() == before


def test_runtime_does_not_retry_before_persisted_time(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    at = datetime(2026, 9, 24, 22, tzinfo=ZONE)
    monkeypatch.setattr(runtime, "now", lambda: at)
    save(tmp_path / "runtime-control.json", {"enabled": True, "until": "2026-12-31T15:00:00+08:00"})
    save(tmp_path / "runtime-state.json", {"next_attempt_at": (at + timedelta(minutes=30)).isoformat()})
    monkeypatch.setattr(runtime, "acquire", lambda: pytest.fail("must not call source"))
    assert runtime.tick()["status"] == "NOT_DUE"


def test_capture_complete_input_is_still_not_a_prediction(tmp_path, monkeypatch):
    """训练门槛未过时，即使输入完整，也只能追加输入记录，不能伪造方向。"""
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    at = datetime(2026, 9, 24, 22, tzinfo=ZONE)
    monkeypatch.setattr(runtime, "now", lambda: at)
    monkeypatch.setattr(runtime, "specification", lambda: None)
    required = runtime.input_days(date(2026, 9, 24))
    nav = [
        {"nav_date": d, "unit_nav": str(1 + i / 1000), "ann_date": d, "content_hash": str(d)}
        for i, d in enumerate(required)
    ]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(runtime, "get_engine", lambda: SimpleNamespace(connect=Connection))
    monkeypatch.setattr(runtime.repo, "source", lambda _: {"source_id": "test", "retention_days": 365})
    monkeypatch.setattr(runtime.repo, "navs", lambda *_: nav)
    monkeypatch.setattr(runtime, "load_bundle", lambda: {"days": {}, "indices": {}})
    monkeypatch.setattr(
        runtime, "calculate", lambda *_a, **_k: {"status": "AVAILABLE", "market_features": [0] * 4, "quote_hashes": {}}
    )
    save(tmp_path / "study.json", {"status": "INSUFFICIENT_EVIDENCE_KEEP_EXISTING"})
    save(tmp_path / "study-current.json", {"file": "study.json"})
    result = runtime.capture()
    payload = read(tmp_path / "forward-inputs" / (result["target"] + ".json"))
    assert payload["candidate_direction"] is None and payload["training_eligible"] is False
    assert payload["kind"] == "LIVE_INPUT_ONLY_NOT_A_PREDICTION"
    assert runtime.capture()["status"] == "ALREADY_CAPTURED"


def test_disable_preserves_input_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    save(tmp_path / "runtime-control.json", {"enabled": True})
    save(tmp_path / "forward-inputs/sample.json", {"proof": 1})
    assert runtime.disable()["status"] == "DISABLED"
    assert read(tmp_path / "runtime-control.json")["enabled"] is False
    assert read(tmp_path / "forward-inputs/sample.json") == {"proof": 1}
