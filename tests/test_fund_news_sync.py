"""公告事实测试只使用固定替身，验证时间资格、失败保留、版本与内部权限。"""

import json
from contextlib import nullcontext
from datetime import datetime

import pytest
from app.services import fund_news_sync as news
from app.services.fund_exposure_common import read, save

AT = datetime.fromisoformat("2026-09-29T16:20:00+08:00")
DAY = datetime.fromisoformat("2026-08-31T00:00:00+08:00")
TITLE = "德邦鑫星价值灵活配置混合型证券投资基金2026年中期报告"


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "collect_company_news", lambda *args: {
        "facts": [], "errors": [], "limited": False, "coverage": {"companies": 0},
    })
    monkeypatch.setattr(news, "directory", lambda: tmp_path / "news")
    monkeypatch.setattr(news, "SUPPLEMENT", tmp_path / "prior")
    monkeypatch.setattr(news, "now", lambda: AT)
    registry = tmp_path / "registry.json"
    save(registry, {"category_counts": {"a" * 32: 1}})
    monkeypatch.setattr(news, "source_path", lambda *args: registry)
    monkeypatch.setattr(news, "client", lambda: nullcontext(object()))
    state = {
        "calls": [],
        "body": TITLE + "报告送出日期：2026年08月31日" + "基金信息" * 40,
        "item": {
            "contentId": "report",
            "title": TITLE,
            "activationDate": int(DAY.timestamp() * 1000),
            "publishDate": int(AT.timestamp() * 1000),
            "url": "/report.pdf",
        },
        "fail": False,
    }

    def fetch(connection, url, suffix="json", params=None, **kwargs):
        state["calls"].append(url)
        if state["fail"]:
            raise ValueError("simulated source failure")
        raw = (
            json.dumps({"contents": [state["item"]], "totalCount": 1, "totalPage": 1}).encode()
            if params
            else state["body"].encode()
        )
        return raw, {"url": url, "received_at": AT.isoformat(), "sha256": news.digest(raw.hex())}

    monkeypatch.setattr(news, "fetch", fetch)
    monkeypatch.setattr(news, "article_text", lambda raw, pdf: ([raw.decode()], "TEXT_EXTRACTED"))
    return state


def test_valid_publication_is_only_a_fact_and_repeat_preserves_first_receipt(pipeline):
    result = news._collect({}, lambda *args: None)
    assert result["status"] == "SUCCEEDED" and result["verified"] == 1
    event = next((news.directory() / "events").glob("*.json"))
    original = event.read_bytes()
    saved = read(event)
    assert saved["business_eligible"]
    assert not saved["training_eligible"] and not saved["prediction_eligible"]
    assert saved["effective_at"] is None and saved["publication_precision"] == "DAY"
    news._collect(result, lambda *args: None)
    assert len(list((news.directory() / "events").glob("*.json"))) == 1
    assert event.read_bytes() == original
    public = news.current("002112")
    assert public["items"][0]["stage"] == "已披露文件"
    assert "evidence" not in public["items"][0] and "modelId" not in public["items"][0]


@pytest.mark.parametrize(
    "body",
    [
        TITLE + "截至2026年08月31日" + "基金信息" * 40,
        "另一只基金报告送出日期：2026年08月31日" + "基金信息" * 40,
        TITLE + "报告送出日期：2026年09月01日" + "基金信息" * 40,
    ],
)
def test_interior_date_wrong_title_or_wrong_date_cannot_be_verified(pipeline, body):
    pipeline["body"] = body
    assert news._collect({}, lambda *args: None)["verified"] == 0
    assert not news.current("002112")["complete"]


def test_failure_preserves_prior_fact_without_claiming_complete(pipeline):
    previous = news._collect({}, lambda *args: None)
    prior = news.current("002112")["items"]
    event = next((news.directory() / "events").glob("*.json"))
    original = event.read_bytes()
    pipeline["fail"] = True
    result = news._collect(previous, lambda *args: None)
    assert result["status"] == "PARTIAL_SUCCESS" and result["errors"] == 1
    assert result["last_success_at"] == previous["last_success_at"]
    retained = news.current("002112")["items"][0]
    assert retained["eventId"] == prior[0]["eventId"]
    assert retained["evidenceHash"] == prior[0]["evidenceHash"]
    assert retained["stage"] == "此前已披露，本次待复核"
    assert event.read_bytes() == original
    assert not news.current("002112")["complete"]
    pipeline["fail"] = False
    news._collect(result, lambda *args: None)
    assert news.current("002112")["items"] == prior
    assert event.read_bytes() == original


def test_first_received_uses_earliest_matching_original_not_latest_fetch(pipeline):
    old = {
        "url": "https://www.dbfund.com.cn/report.pdf",
        "sha256": news.digest(pipeline["body"].encode().hex()),
        "received_at": "2026-09-01T08:00:00+08:00",
    }
    save(news.SUPPLEMENT / "public-receipt-versions/old.json", old)
    # 更早时间属于另一个原文，不能抢占此版本的首次取得时间。
    save(
        news.SUPPLEMENT / "public-receipt-versions/other.json",
        {**old, "sha256": "a" * 64, "received_at": "2026-08-31T08:00:00+08:00"},
    )
    news._collect({}, lambda *args: None)
    assert news.current("002112")["items"][0]["firstReceivedAt"] == old["received_at"]


def test_revised_body_has_new_version_and_old_bytes_are_preserved(pipeline):
    news._collect({}, lambda *args: None)
    old = next((news.directory() / "events").glob("*.json"))
    original = old.read_bytes()
    pipeline["body"] += "更正说明"
    news._collect({}, lambda *args: None)
    assert len(list((news.directory() / "events").glob("*.json"))) == 2
    assert old.read_bytes() == original


def test_unknown_scope_and_tampered_snapshot_never_become_zero_events(pipeline):
    assert not news.current("008888")["complete"]
    with pytest.raises(ValueError):
        news.current("../002112")
    with pytest.raises(ValueError):
        news.synchronize("")
    news._collect({}, lambda *args: None)
    ref = read(news.directory() / "current.json")
    path = news.directory() / "snapshots" / (ref["hash"] + ".json")
    save(path, {**read(path), "items": []}, replace=True)
    with pytest.raises(ValueError, match="SNAPSHOT_INVALID"):
        news.current("002112")
