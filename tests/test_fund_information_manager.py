"""经理变更持续纳入、任期归属、复权与缺口、综合引用及历史不可变边界。"""

from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from app.schemas.fund_information_analysis import validate_analysis
from app.services import fund_information_manager as manager
from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_exposure_common import save
from app.services.fund_information_analysis import display_narrative, narrative
from app.services.fund_information_contracts import reference_catalog
from app.services.fund_information_snapshot import inventory

NOW = datetime(2026, 10, 10, 11, tzinfo=ZONE)
NOTICE = """公告送出日期：2026年5月30日
基金名称 德邦鑫星价值灵活配置混合型证券投资基金
基金经理变更类型 解聘基金经理
共同管理本基金的其他基金经理姓名 陆阳
离任基金经理姓名 雷涛
2. 离任基金经理的相关信息
离任基金经理姓名 雷涛
离任原因 个人原因
离任日期 2026年05月29日
转任本公司其他工作岗位的说明 -"""


def document(body=NOTICE, received="2026-09-24T12:00:00+08:00"):
    d = {"kind": "fund", "id": "fund-example", "title": "基金经理变更公告",
         "publishedDate": "2026-05-30", "sourceUrl": "https://example.com/change.pdf"}
    raw = {"pages": [body], "fund_name_or_code_mentioned": True,
           "receipt": {"received_at": received, "sha256": digest(body)}}
    return d, raw


def test_old_change_is_loaded_with_exact_reason_and_source(tmp_path, monkeypatch):
    d, raw = document()
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    save(tmp_path / "supplement/documents/example.json", raw)
    notices, gaps = manager.load_notices({"documents": [d]}, NOW)
    assert gaps == [] and len(notices) == 1
    n = notices[0]
    assert n["parsed"] and n["reason"] == "个人原因" and n["remaining"] == ["陆阳"]
    assert n["effective_date"] == "2026-05-29" and n["body"] == NOTICE
    assert n["source"]["publishedDate"] == "2026-05-30"


@pytest.mark.parametrize("change", [
    {"received_at": "2026-10-11T00:00:00+08:00"},
    {"received_at": "2026-09-24T00:00:00"},
])
def test_future_or_naive_receipt_is_not_admitted(change):
    d, raw = document()
    raw["receipt"].update(change)
    assert manager.parse_notice(d, raw, NOW) is None


def test_future_publication_unrelated_and_unrecognized_reason_do_not_get_invented():
    d, raw = document(NOTICE.replace("2026年5月30日", "2026年11月30日"))
    assert manager.parse_notice(d, raw, NOW) is None
    d, raw = document()
    raw["fund_name_or_code_mentioned"] = False
    assert manager.parse_notice(d, raw, NOW) is None
    d, raw = document(NOTICE.replace("离任原因 个人原因", "说明 不提供此表项"))
    assert manager.parse_notice(d, raw, NOW)["parsed"] is False


def test_new_appointment_and_upcoming_change_keep_effective_date_boundary():
    body = """公告送出日期：2026年10月09日
基金经理变更类型 增聘基金经理
新任基金经理姓名 张三
共同管理本基金的其他基金经理姓名 陆阳
2.新任基金经理的相关信息
新任基金经理姓名 张三 任职日期 2026年10月12日"""
    d, raw = document(body, "2026-10-09T12:00:00+08:00")
    notice = manager.parse_notice(d, raw, NOW)
    assert notice["parsed"] and set(notice["remaining"]) == {"张三", "陆阳"}
    c = context()
    days, rows = series([100, 120, 90, 110])
    result = manager.build_context(c["assignments"], [c["latest_change"], notice], rows,
                                   NOW, days[-1], days, "", [])
    assert result["latest_change"]["id"] == c["latest_change"]["id"]
    assert result["current_verified"] and len(result["managers"]) == 1


def series(values):
    days = tuple(date(2026, 6, 1) + timedelta(days=i) for i in range(len(values)))
    rows = [{"nav_date": d, "adjusted_nav": v, "unit_nav": 999} for d, v in zip(days, values, strict=True)]
    return days, rows


def test_adjusted_return_drawdown_and_overlapping_window_denominator():
    days, rows = series([100, 120, 90] + [110] * 19)
    result = manager.performance(rows, days[0], days[-1], days)
    assert result["return_pct"] == 10 and result["max_drawdown_pct"] == 25
    assert result["rolling_windows"] == 2 and result["positive_20d_pct"] == 50
    assert result["short_sample"] is True


@pytest.mark.parametrize("value", [None, 0, -1, Decimal("NaN")])
def test_missing_or_invalid_adjusted_nav_never_falls_back_to_unit_nav(value):
    days, rows = series([100, value, 110])
    assert manager.performance(rows, days[0], days[-1], days)["status"] == "MISSING"


def test_missing_trading_day_and_duplicate_day_rejected():
    days, rows = series([100, 90, 110])
    assert manager.performance(rows[::2], days[0], days[-1], days)["status"] == "MISSING"
    assert manager.performance(rows + rows[:1], days[0], days[-1], days)["status"] == "MISSING"


def context():
    days, rows = series([100, 120, 90, 110])
    people = [{"manager_name": "陆阳", "begin_date": days[0], "end_date": None},
              {"manager_name": "雷涛", "begin_date": days[0], "end_date": days[1]}]
    d, raw = document(NOTICE.replace("2026年05月29日", "2026年06月02日"))
    notice = manager.parse_notice(d, raw, NOW)
    return manager.build_context(people, [notice], rows, NOW, days[-1], days, "合同基准", [])


def test_co_managed_history_and_solo_period_separated():
    c = context()
    assert c["current_verified"] is True
    m = c["managers"][0]
    assert m["co_managed"] and m["solo_start"] == "2026-06-03"
    assert m["tenure"]["return_pct"] == 10
    assert m["solo"]["return_pct"] == pytest.approx(22.2222)
    assert c["benchmark_comparison"] == "MISSING" and c["peer_comparison"] == "MISSING"
    text = manager.facts(c)["manager:performance"]["text"]
    assert "暂不能确认超额管理能力" in text and "不以大盘指数替代" in text


def test_assignment_conflict_blocks_personal_metrics_and_future_exit_keeps_team():
    c = context()
    n = deepcopy(c["latest_change"])
    n["remaining"] = ["其他人"]
    days, rows = series([100, 120, 90, 110])
    out = manager.build_context(c["assignments"], [n], rows, NOW, days[-1], days, "", [])
    assert not out["current_verified"] and out["managers"] == []
    assert out["gaps"]
    people = deepcopy(c["assignments"])
    people[1]["end_date"] = date(2026, 11, 1)
    out = manager.build_context(people, [], rows, NOW, days[-1], days, "", [])
    assert len(out["managers"]) == 2 and all(m["solo"] is None for m in out["managers"])


def candidate(refs):
    return {"direction": "DOWN", "confidence": "LOW", "summary": "持仓偏弱，倾向下跌。",
            "reasons": [{"title": "持仓", "refs": ["market"], "role": "支持下跌",
                         "meaning": "走势偏弱。", "implication": "延续是有条件的假设。"},
                        {"title": "经理变更与任期表现", "refs": refs, "role": "背景观察",
                         "meaning": "公开原因为个人原因，当前经理任期包含共同管理。",
                         "implication": "独立管理样本较短，只能作背景，不能直接预测次日涨跌。"}],
            "counterpoints": [], "synthesis": "经理背景限制历史业绩参考性，行情倾向仍有不确定性。",
            "change_conditions": [], "limitations": ["基准和同类比较暂缺。"]}


def test_manager_evidence_is_required_and_cannot_be_mechanical_direction():
    facts = reference_catalog(manager.facts(context()))
    required = [ref for ref, fact in facts.items() if fact["required_in_analysis"]]
    facts["market"] = {"category": "净值", "text": "行情事实", "source": None}
    value = candidate(required)
    assert validate_analysis(value, facts)["direction"] == "DOWN"
    value["reasons"][1]["refs"] = required[:1]
    with pytest.raises(ValueError, match="MANAGER_EVIDENCE_MISSING"):
        validate_analysis(value, facts)
    value = candidate(required)
    value["reasons"][1]["role"] = "支持下跌"
    with pytest.raises(ValueError, match="MATTER_UNRESOLVED"):
        validate_analysis(value, facts)


def test_saved_narrative_and_input_identity_remain_stable():
    c = context()
    original = manager.facts(c)
    c["as_of"] = NOW.replace(hour=12).isoformat()
    assert manager.facts(c) == original
    facts = reference_catalog(original)
    refs = [r for r, f in facts.items() if f["required_in_analysis"]]
    facts["market"] = {"category": "净值", "text": "行情事实", "source": None}
    analysis = candidate(refs)
    body = {"analysis": analysis, "evidence": facts, "narrative": narrative(analysis, facts, [], [])}
    before = deepcopy(body)
    shown = display_narrative(body)
    assert body == before and "个人原因" in shown["drivers"][1]["observation"]
    assert shown["drivers"][1]["category"] == "基金经理"
    assert shown["drivers"][1]["sources"][0]["url"] == "https://example.com/change.pdf"


def test_inventory_reports_manager_coverage_without_falsely_marking_complete():
    rows = inventory({"manager_context": context(), "nav": [], "latest_nav_date": None,
                      "report": {"endDate": "2026-06-30"}, "market": {}, "documents": []})
    assert len(rows) == 16 and rows[0]["status"] == "PARTIAL"
    assert "陆阳" in rows[0]["detail"] and "不受两周限制" in rows[11]["detail"]
