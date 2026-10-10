"""页面投影保留引文与来源，审计材料仍存在独立快照，不影响实验输入和方向。"""

from copy import deepcopy

from app.services.fund_information_archive import public_evidence


def test_public_projection_keeps_exact_quotes_and_every_source_without_mutating_archive():
    facts = {
        "Fexample": {
            "text": "甲公司尚未完成回购，金额为 100 万元。",
            "category": "公告",
            "sources": [{"title": "回购计划"}, {"title": "董事会决议"}],
            "relation": "甲公司披露仓位1.00%",
            "fact_id": "issuer-evidence:example",
            "quote_receipts": [{"start": 0, "end": 24}],
            "matters": [{"id": "first"}],
            "member_event_ids": ["a:0", "b:0"],
        }
    }
    before = deepcopy(facts)
    display = public_evidence(facts)
    assert display["Fexample"] == {
        k: v for k, v in facts["Fexample"].items() if k not in {"quote_receipts", "matters", "member_event_ids"}
    }
    assert facts == before
    assert display["Fexample"]["text"] == before["Fexample"]["text"]
    assert len(display["Fexample"]["sources"]) == 2


def test_infer_archives_large_receipts_without_bloating_or_changing_page_result(monkeypatch):
    """通过整个保存调用链的替身检查，不连接真实数据库，也不覆盖任何预测。"""
    import json
    from contextlib import nullcontext
    from datetime import datetime
    from types import SimpleNamespace

    from app.services import fund_information_analysis as service
    from app.services.direction_1d_protocol import ZONE

    at = datetime(2026, 8, 26, 8, 30, tzinfo=ZONE)
    facts = {
        "Fexample": {
            "category": "持仓行情",
            "text": "基准日股票下跌。",
            "source": None,
            "quote_receipts": [{"quote": "原始审计字符" * 40000}],
        }
    }
    data = {
        "fund_code": "002112",
        "fund_name": "测试基金002112",
        "facts": {},
        "documents": [],
        "report": {"endDate": "2026-06-30"},
        "inventory": [],
        "time_context": {"statements": []},
        "window": {
            "target_nav_date": "2026-08-26",
            "base_nav_date": "2026-08-25",
            "deadline_at": "2026-08-26T15:00:00+08:00",
        },
        "as_of": at.isoformat(),
        "latest_nav_date": "2026-08-25",
        "retention_days": 30,
        "product_family_id": "002112",
        "source_manifest": [],
        "source_id": "source",
        "nav": [],
        "companies": [],
    }
    analysis = {
        "direction": "DOWN",
        "summary": "条件性偏弱。",
        "synthesis": "若弱势延续，可能拖累。",
        "reasons": [
            {
                "title": "行情偏弱",
                "refs": ["Fexample"],
                "role": "支持下跌",
                "meaning": "所列行情偏弱。",
                "implication": "仅为延续假设。",
            }
        ],
        "counterpoints": [],
        "change_conditions": [],
        "limitations": [],
    }
    saved, completed = [], []
    monkeypatch.setattr(service.repo, "clock", lambda: at)
    monkeypatch.setattr(service, "prepare", lambda now: data)
    monkeypatch.setattr(service, "get_settings", lambda: SimpleNamespace(deepseek_model="test"))
    monkeypatch.setattr(service, "scope_lock", lambda *args: nullcontext())
    monkeypatch.setattr(service, "get_engine", lambda: SimpleNamespace(begin=lambda: nullcontext(object())))
    monkeypatch.setattr(service, "select_revision", lambda *args: {"revision_sequence": 1, "input_identity": "test"})
    monkeypatch.setattr(service, "analyze", lambda *args: (analysis, facts, []))

    def save_snapshot(connection, kind, key, payload, now, expires):
        saved.append((kind, deepcopy(payload)))
        return kind + "-id", kind + "-hash"

    monkeypatch.setattr(service.repo, "save_snapshot", save_snapshot)
    monkeypatch.setattr(service, "complete", lambda *args: completed.append(args))
    result = service.infer("2026-08-26")
    body = json.loads(result["payload_json"])
    assert [kind for kind, _ in saved] == ["ANALYSIS", "ANALYSIS_FACTS"]
    assert saved[1][1]["facts"] == facts
    assert "quote_receipts" not in body["evidence"]["Fexample"]
    assert body["analysis"] == analysis
    assert body["evidence_snapshot_id"] == "ANALYSIS_FACTS-id"
    assert body["evidence_snapshot_hash"] == "ANALYSIS_FACTS-hash"
    assert len(result["payload_json"]) < 150000
    assert len(completed) == 1
