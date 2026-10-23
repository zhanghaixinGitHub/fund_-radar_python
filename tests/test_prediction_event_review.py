"""预测事项审阅不能冒充评级完成，也不能跨基金、正文版本或目标日复用。"""

from datetime import datetime, timedelta

import pytest
from app.services import fund_market_review as service
from app.services.fund_exposure_common import read, save

CODE = "018156"
ARTICLE = "AN202609241829820518"
SHA = "a" * 64
SOURCE = "b" * 64
NOW = datetime.fromisoformat("2026-10-20T19:00:00+08:00")
BODY = "本基金基金经理毛丁丁因个人原因离任，皮劲松继续担任本基金基金经理，相关手续已经按规定办理。"
NOTE = (
    "已阅读本基金该正文版本：经理离任属于管理团队变化，不能由此直接断言目标日上涨或下跌。"
    "短期判断仍应核对本基金实际医药投向、已闭市证券行情及本期净值；中期保留管理持续性与策略执行的不确定性。"
    "另一位经理继续管理是相反证据，但不能证明没有调整风险；如后续有新策略、组合调整或新的任免公告，应重新审阅。"
    "本回执只记录预测用途的事项影响，正式评级尚待资料齐备后独立复核，不把它标为评级完成。"
)
EVENT = {"fund_code": CODE, "article_id": ARTICLE, "source_hash": SHA}


@pytest.fixture
def sources(tmp_path, monkeypatch):
    from app.services import direction_1d_protocol as protocol
    from app.services import fund_market_news

    root, news = tmp_path / "workflow", tmp_path / "news"
    save(root / "materials" / (CODE + ".json"), {"fund_code": CODE, "source_hash": SOURCE, "candidate_events": [EVENT]})
    item = {
        "source": {"sha256": SHA},
        "candidate_material_event": True,
        "body": BODY,
        "entry": {"published": "2026-09-24"},
        "received_at": "2026-10-20T18:00:00+08:00",
    }
    save(news / "funds" / (CODE + ".json"), {"items": {ARTICLE: item}})
    monkeypatch.setattr(service, "now", lambda: NOW)
    monkeypatch.setattr(service, "review_root", lambda: root / "reviews")
    monkeypatch.setattr(fund_market_news, "directory", lambda: news)
    monkeypatch.setattr(protocol, "window", lambda current: {"target_nav_date": "2026-10-21"})
    service.record_event_review(
        CODE, ARTICLE, SHA, "REQUIRES_RATING_REVIEW", NOTE, root=root / "reviews", news_root=news
    )
    return root, news, item


def record(root, news, *, sha=SHA, quotes=None, note=NOTE):
    return service.record_prediction_event_review(
        CODE, ARTICLE, sha, note, [BODY] if quotes is None else quotes, root=root, news_root=news
    )


def test_prediction_review_keeps_formal_rating_pending_and_preserves_its_receipt(sources):
    root, news, _ = sources
    rating_path = root / "reviews" / CODE / (SHA + ".json")
    original = read(rating_path)
    assert service.pending_prediction_events([EVENT], SOURCE, root=root) == [EVENT]
    result = record(root, news)
    assert result["decision"] == "REVIEWED_FOR_PREDICTION"
    assert service.pending_prediction_events([EVENT], SOURCE, root=root) == []
    assert service.pending_events([EVENT], root=root / "reviews") == [EVENT]
    assert read(rating_path) == original


@pytest.mark.parametrize(
    "problem", ["body_hash", "other_fund", "quote", "future_received", "naive_received", "future_publication"]
)
def test_unverified_originals_cannot_create_prediction_review(sources, problem):
    root, news, item = sources
    if problem == "body_hash":
        item["source"]["sha256"] = "c" * 64
    elif problem == "other_fund":
        save(
            root / "materials" / (CODE + ".json"),
            {"fund_code": "014978", "source_hash": SOURCE, "candidate_events": [EVENT]},
            replace=True,
        )
    elif problem == "quote":
        with pytest.raises(ValueError, match="SOURCE_MISMATCH"):
            record(root, news, quotes=["这一段并非原文中的连续内容，不能因为知道标题就虚构事件引用。"])
        return
    elif problem == "future_received":
        item["received_at"] = (NOW + timedelta(minutes=1)).isoformat()
    elif problem == "naive_received":
        item["received_at"] = "2026-10-20T18:00:00"
    else:
        item["entry"]["published"] = "2026-10-21"
    save(news / "funds" / (CODE + ".json"), {"items": {ARTICLE: item}}, replace=True)
    with pytest.raises(ValueError, match="SOURCE_MISMATCH"):
        record(root, news)
    assert not (root / "prediction-event-reviews" / CODE / (SHA + ".json")).exists()


def test_different_workflow_or_target_cannot_reuse_review(sources, monkeypatch):
    from app.services import direction_1d_protocol as protocol

    root, news, _ = sources
    record(root, news)
    assert service.pending_prediction_events([EVENT], "c" * 64, root=root) == [EVENT]
    monkeypatch.setattr(protocol, "window", lambda current: {"target_nav_date": "2026-10-22"})
    assert service.pending_prediction_events([EVENT], SOURCE, root=root) == [EVENT]


@pytest.mark.parametrize("quotes,note", [([], NOTE), (["标题"], NOTE), ([BODY, BODY], NOTE), ([BODY], "已阅读")])
def test_missing_original_quotes_or_analysis_is_rejected(sources, quotes, note):
    root, news, _ = sources
    with pytest.raises(ValueError, match="REVIEW_INVALID"):
        record(root, news, quotes=quotes, note=note)


def test_review_does_not_clear_missing_baseline_or_other_material_gaps(sources, monkeypatch):
    from app.services import fund_information_snapshot
    from app.services.fund_workflow_materials import inspect_materials

    root, news, _ = sources
    path = root / "supplement.json"
    save(path, {"fund_code": CODE})
    collected = read(root / "materials" / (CODE + ".json"))
    save(root / "materials" / (CODE + ".json"), {**collected, "gaps": ["持仓尚未核齐"]}, replace=True)
    packet = {
        "report": {},
        "documents": [{"id": "proof", "kind": "ANNOUNCEMENT", "body": BODY}],
        "facts": {},
        "comprehensive_coverage": {
            dimension: {"status": "AVAILABLE", "source_refs": ["proof"]}
            for dimension in ["EXPOSURE", "MARKET", "DISCLOSURES"]
        },
        "time_context": {"baseline_complete": False},
    }
    monkeypatch.setattr(fund_information_snapshot, "prepare", lambda *args: packet)
    record(root, news)
    result = inspect_materials(CODE, path, root=root, clock=lambda: NOW)
    assert "本期新增事项尚需结合原文复核" not in result["gaps"]
    assert "持仓尚未核齐" in result["gaps"]
    assert "本基金实际基准日净值或所列行情仍未齐备" in result["gaps"]
    with pytest.raises(ValueError, match="GAPS_REMAIN"):
        service.approve_comprehensive(CODE, NOTE, root=root, materials_path=path)
    assert service.pending_events([EVENT], root=root / "reviews") == [EVENT]
