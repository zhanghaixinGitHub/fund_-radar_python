"""风险事实验收：缺值、日期穿越和披露分母不能被合计计算掩盖。"""

from copy import deepcopy
from datetime import date

import pytest
from app.services.fund_risk_summary import calculate


@pytest.fixture
def material():
    return {
        "fundCode": "002112",
        "asOfDate": "2026-09-28",
        "reports": [
            {
                "endDate": "2026-06-30",
                "publishedDate": "2026-08-31",
                "fullDisclosure": True,
                "sourceUrl": "https://www.dbfund.com.cn/report.pdf",
                "disclosedWeightPct": 60,
                "holdings": [
                    {"stockCode": "300308.SZ", "stockName": "公司甲", "weightPct": 40},
                    {"stockCode": "300502.SZ", "stockName": "公司乙", "weightPct": 20},
                ],
                "industries": [{"name": "制造业", "weightPct": 60}],
                "assets": [{"name": "股票", "weightPct": 55}],
            }
        ],
        "companies": [
            {"stockCode": "300308.SZ", "quote": {"date": "2026-09-28", "changePct": -5}},
            {"stockCode": "300502.SZ", "quote": {"date": "2026-09-28", "changePct": None}},
        ],
    }


def summary(material):
    return calculate("002112", material, today=date(2026, 9, 29))


def test_partial_quotes_keep_unknown_and_original_denominator(material):
    result = summary(material)
    assert result["crossCheck"]["coverageWeightPct"] == 40
    assert result["crossCheck"]["staticContributionPctPoints"] == -2
    assert result["disclosedWeightPct"] == 60
    assert result["assets"][0]["denominator"] == "基金总资产"
    assert result["industries"][0]["denominator"] == "基金净资产"


@pytest.mark.parametrize(
    "field,value",
    [("weightPct", None), ("weightPct", "NaN"), ("weightPct", True), ("weightPct", -1), ("stockCode", "300502.SZ")],
)
def test_invalid_or_duplicate_holding_fails_closed(material, field, value):
    material["reports"][0]["holdings"][0][field] = value
    assert not summary(material)["available"]


def test_report_published_after_snapshot_is_not_backfilled(material):
    material["reports"][0]["publishedDate"] = "2026-09-29"
    assert not summary(material)["available"]


def test_mixed_quote_dates_are_not_added(material):
    material["companies"][1]["quote"] = {"date": "2026-09-25", "changePct": 2}
    assert summary(material)["crossCheck"] is None


def test_future_quote_and_duplicate_company_are_excluded(material):
    material["companies"][1]["quote"] = {"date": "2026-09-29", "changePct": 99}
    material["companies"].append(deepcopy(material["companies"][0]))
    assert summary(material)["crossCheck"] is None


def test_scope_mismatch_rejected(material):
    material["fundCode"] = "000001"
    with pytest.raises(ValueError, match="SCOPE_MISMATCH"):
        summary(material)


def test_total_mismatch_does_not_invent_missing_exposure(material):
    material["reports"][0]["disclosedWeightPct"] = 90
    assert not summary(material)["available"]


def test_publish_preserves_previous_inputs_and_detects_archive_tamper(material, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from app.services import fund_risk_summary as risk
    from app.services.fund_exposure_common import read

    monkeypatch.setattr(risk, "snapshot", lambda code: deepcopy(material))
    monkeypatch.setattr(risk, "get_settings", lambda: SimpleNamespace(fund_insight_directory=tmp_path))
    first = risk._publish("002112", root=tmp_path)
    material["companies"][0]["quote"]["changePct"] = 3
    second = risk._publish("002112", root=tmp_path)
    assert first["version"] != second["version"]
    old = tmp_path / "002112" / "risk" / (first["version"] + ".json")
    assert read(old)["result"]["crossCheck"]["staticContributionPctPoints"] == -2
    assert risk.current("002112")["crossCheck"]["staticContributionPctPoints"] == 1.2
    current = tmp_path / "002112" / "risk" / (second["version"] + ".json")
    current.write_text(current.read_text(encoding="utf-8").replace('"changePct":3', '"changePct":99'), encoding="utf-8")
    # 直接破坏校验包，任何被修改的内容均不能继续作为可展示事实。
    current.write_text('{"hash":"wrong","payload":{}}', encoding="utf-8")
    with pytest.raises(ValueError):
        risk.current("002112")
