from datetime import datetime

from app.services import fund_prediction_news_recovery as news
from app.services.fund_materials_store import versioned_save
from app.services.fund_workflow_materials import input_gaps


def test_same_day_date_only_notice_remains_a_gap_instead_of_invented_time(tmp_path, monkeypatch):
    source = {"file": "sources/notice.json", "sha256": "a" * 64}
    versioned_save(
        tmp_path / "news/funds/123456.json",
        {
            "status": "SUCCEEDED",
            "catalog_complete": True,
            "coverage": {"start": "2026-10-12", "end": "2026-11-11"},
            "items": {
                "AN202611111831605448": {
                    "entry": {"published": "2026-11-11", "title": "渠道公告"},
                    "source": source,
                    "body": "真实取得的当日公告正文，只有日期，没有确切公开时刻。",
                    "received_at": "2026-11-11T10:00:00+08:00",
                }
            },
        },
    )
    checked = []
    monkeypatch.setattr(news, "read_source", lambda root, src: checked.append(src))
    material = {
        "fund_code": "123456",
        "documents": [],
        "facts": {},
        "source_manifest": {},
        "comprehensive_coverage": {},
    }
    result = news.refresh(material, root=tmp_path, checked_at=datetime.fromisoformat("2026-11-11T11:00:00+08:00"))
    assert result["publication_time_required"] == ["AN202611111831605448"]
    assert material["documents"] == [] and checked == [source]
    packet = {**material, "report": {}, "time_context": {"baseline_complete": True}}
    assert "本次同日公告的公开时间证据仍待核验" in input_gaps(packet)
