"""回看导入、独立留档和读取边界；只使用本机隔离schema，不生成真实预测。"""

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.services import analysis_reviews as service
from app.services.direction_1d_protocol import digest
from sqlalchemy import text
from tests.test_prediction_database import database  # noqa: F401


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    (root / "sources").mkdir(parents=True)
    raw = b"isolated source bytes"
    (root / "sources" / "notice.pdf").write_bytes(raw)
    generated = "2026-10-14T16:12:58+08:00"
    body = "公告明确说明尚未完成。"
    inputs = {
        "fund_code": "016008",
        "target_nav_date": "2026-10-14",
        "base_nav_date": "2026-10-13",
        "information_cutoff": "2026-10-14T09:35:00+08:00",
        "retrieved_at": generated,
        "metrics": {"quote_date": "2026-10-13"},
        "documents": [
            {
                "id": "notice",
                "published_date": "2026-10-13",
                "body": body,
                "quote": "尚未完成",
                "read_ranges": [{"start": 0, "end": len(body)}],
                "review_scope": "FULL_TEXT",
                "receipt": {
                    "file": "notice.pdf",
                    "url": "https://example.com/notice.pdf",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
            }
        ],
        "overnight_news": {"published_at": "2026-10-14T06:20:00+08:00", "url": "https://example.com/news"},
        "apple_primary": {"url": "https://example.com/release"},
    }
    result = {
        "kind": "CODEX_COMPREHENSIVE_ASOF_REPLAY",
        "fund_code": "016008",
        "target_nav_date": "2026-10-14",
        "base_nav_date": "2026-10-13",
        "information_cutoff": inputs["information_cutoff"],
        "actual_generated_at": generated,
        "input_hash": digest(inputs),
        "direction": "DOWN",
        "confidence": "LOW",
        "up_probability": None,
        "validation_status": "UNVALIDATED",
        "summary": "偏下跌，把握低。",
        "supporting_evidence": [{"sources": ["notice"], "fact": body, "interpretation": "事项尚不确定。"}],
        "counter_evidence": [],
        "change_conditions": ["事项落地后重新判断。"],
        "limitations": ["事后回看。"],
    }
    verification = {
        "checked_at": generated,
        "all_quotes_recomputed": True,
        "all_document_quotes_exact": True,
        "frozen_input_hash_matches": True,
        "target_day_quotes_not_used": True,
    }
    for filename, value in [
        ("analysis-input.json", {"input_hash": digest(inputs), "input": inputs}),
        ("analysis-result.json", result),
        ("verification.json", verification),
    ]:
        (root / filename).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(service.repo, "clock", lambda: datetime.fromisoformat("2026-10-14T18:00:00+08:00"))
    return root


@pytest.mark.parametrize("problem", ["fund", "hash", "early", "timezone", "source", "verification"])
def test_rejects_wrong_identity_changed_original_and_fake_time(archive, problem):
    path = archive / "analysis-result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if problem == "fund":
        result["fund_code"] = "002112"
    elif problem == "hash":
        result["input_hash"] = "0" * 64
    elif problem == "early":
        result["actual_generated_at"] = "2026-10-14T09:35:00+08:00"
    elif problem == "timezone":
        result["information_cutoff"] = "2026-10-14T09:35:00"
    elif problem == "source":
        (archive / "sources" / "notice.pdf").write_bytes(b"changed")
    else:
        verification = archive / "verification.json"
        value = json.loads(verification.read_text(encoding="utf-8"))
        value["target_day_quotes_not_used"] = False
        verification.write_text(json.dumps(value), encoding="utf-8")
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="ANALYSIS_REVIEW_"):
        service.prepare_archive(archive, "016008", "2026-10-14")


def test_preview_preserves_original_actual_time_and_excludes_accuracy_without_db(archive, monkeypatch):
    monkeypatch.setattr(service, "get_engine", lambda: pytest.fail("预览不得写数据库"))
    original = {path.name: path.read_bytes() for path in archive.glob("*.json")}
    result = service.import_archive(archive, "016008", "2026-10-14")
    assert result["status"] == "PREVIEW"
    assert result["review"]["actualGeneratedAt"] == "2026-10-14T16:12:58+08:00"
    assert result["review"]["excludedFromAccuracy"] is True
    assert original == {path.name: path.read_bytes() for path in archive.glob("*.json")}


def test_real_isolated_import_idempotence_scoped_read_and_immutable_original(database, archive, monkeypatch):  # noqa: F811
    path = Path(__file__).parents[1] / "alembic/versions/20261014_36_analysis_review.py"
    spec = importlib.util.spec_from_file_location("analysis_review_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with database.begin() as connection:
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.upgrade()
    monkeypatch.setattr(service, "get_engine", lambda: database)
    monkeypatch.setattr(service.repo, "profiles", lambda connection, codes: [{"fund_code": codes[0]}])
    first = service.import_archive(archive, "016008", "2026-10-14", commit=True)
    second = service.import_archive(archive, "016008", "2026-10-14", commit=True)
    assert first["reviewId"] == second["reviewId"] and second["reused"]
    page = service.list_reviews("016008")
    assert len(page["items"]) == 1 and page["items"][0]["excludedFromAccuracy"] is True
    assert service.list_reviews("002112")["items"] == []
    assert (
        service.list_reviews(
            "016008", before=datetime.fromisoformat(first["actualGeneratedAt"]), before_id=UUID(first["reviewId"])
        )["items"]
        == []
    )
    with database.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM fund_prediction_record")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM prediction_outcome")).scalar_one() == 0
    with pytest.raises(Exception, match="immutable analysis review"), database.begin() as connection:
        connection.execute(text("UPDATE fund_analysis_review SET target_nav_date='2026-10-15'"))


def test_internal_route_requires_service_token_and_validates_cursor(monkeypatch):
    from app.core.config import get_settings
    from app.main import app
    from fastapi.testclient import TestClient

    calls = []
    monkeypatch.setattr(
        service, "list_reviews", lambda *args, **kwargs: calls.append(args) or {"items": [], "nextCursor": None}
    )
    client = TestClient(app)
    path = "/internal/v1/multi-predictions/funds/016008/analysis-reviews"
    assert client.get(path).status_code == 403
    response = client.get(path, headers={"X-Service-Token": get_settings().ai_service_token.get_secret_value()})
    assert response.status_code == 200 and calls == [("016008",)]
    assert (
        client.get(
            path.replace("016008", "bad"),
            headers={"X-Service-Token": get_settings().ai_service_token.get_secret_value()},
        ).status_code
        == 422
    )


def test_incomplete_cursor_fails_before_database(monkeypatch):
    monkeypatch.setattr(service, "get_engine", lambda: pytest.fail("无效游标不能读取数据库"))
    with pytest.raises(ValueError, match="ANALYSIS_REVIEW_CURSOR_INVALID"):
        service.list_reviews("016008", before_id=UUID("00000000-0000-0000-0000-000000000001"))
