from app.services.fund_prediction_report_recovery import bind_latest_partial


def test_adjacent_class_codes_are_checked_before_whitespace_is_removed():
    latest = {
        "id": "quarter",
        "sourceArticleId": "AN202610271830980599",
        "publishedDate": "2026-10-27",
        "endDate": "2026-09-30",
        "fullDisclosure": False,
        "holdings": [],
    }
    material = {
        "fund_code": "019449",
        "report": {"endDate": "2026-06-30"},
        "source_manifest": {},
        "documents": [
            {
                "id": "own:AN202610271830980599",
                "kind": "REPORT",
                "codes": ["019449"],
                "published_date": "2026-10-27",
                "body": "2026年9月30日\n下属份额代码 007280    019449\n",
            }
        ],
    }
    assert bind_latest_partial(material, latest)
    assert material["report"]["endDate"] == "2026-06-30"
    assert material["source_manifest"]["latest_partial_report"]["report"]["fullDisclosure"] is False
    material["source_manifest"] = {}
    material["documents"][0]["body"] = "2026年9月30日 007280019449"
    assert not bind_latest_partial(material, latest)
