"""用内存仓储验证真实闭环异常顺序；合成记录不连接数据库、不计入真实观察。"""

import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.services import direction_1d_inference as inference
from app.services.direction_1d_protocol import ZONE, canonical, digest


def test_registered_json_hash_and_registry_identity(tmp_path, monkeypatch):
    model_id = uuid4()
    model = {"group_id": "CN_EQUITY", "cohort_id": "synthetic"}
    path = tmp_path / f"{model_id}.json"
    path.write_text(canonical(model), encoding="utf-8")
    row = {"file_name": path.name, "model_id": model_id, "content_hash": digest(model), "metadata": model, **model}
    monkeypatch.setattr(inference, "MODEL_ROOT", tmp_path)
    assert inference.load_model(row) == model
    with pytest.raises(ValueError, match="REGISTRY"):
        inference.load_model({**row, "cohort_id": "other"})
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="HASH"):
        inference.load_model(row)
    with pytest.raises(ValueError, match="PATH"):
        inference.load_model({**row, "file_name": "../outside.json"})


def test_first_answer_frozen_t_then_append_revisions_without_retraining_inputs(monkeypatch):
    now = datetime(2026, 9, 16, 20, tzinfo=ZONE)

    def source(day, nav):
        return {
            "nav_date": day,
            "unit_nav": nav,
            "content_hash": digest([day, nav]),
            "expires_at": (now + timedelta(days=30)).isoformat(),
        }

    frozen = source("2026-09-14", "1")
    body = {
        "task_key": "SYNTHETIC_ONLY",
        "input_snapshot_id": str(uuid4()),
        "fund_code": "123456",
        "base_nav_date": "2026-09-14",
        "target_nav_date": "2026-09-15",
        "expires_at": frozen["expires_at"],
        "input": {"values": [frozen], "event_status": "UNKNOWN"},
    }
    raw = canonical(body)
    monkeypatch.setattr(inference.repo, "clock", lambda: now)
    monkeypatch.setattr(
        inference.repo,
        "get_job",
        lambda _: {
            "state": "SUCCEEDED",
            "kind": "FORECAST",
            "result": {"payload_json": raw, "content_hash": digest(body)},
        },
    )
    snapshots = []
    current = {"base": "1.1", "target": "1.2"}

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execution_options(self, **_):
            return self

        def begin(self):
            return self

        def execute(self, statement, params):
            sql = str(statement)
            if "fund_dividend" in sql:
                return SimpleNamespace(mappings=lambda: [])
            row = next(
                (s for s in snapshots if "hash" not in params or s["payload"]["revision_key"] == params["hash"]), None
            )
            return SimpleNamespace(first=lambda: row, mappings=lambda: SimpleNamespace(first=lambda: row))

    monkeypatch.setattr(inference, "get_engine", lambda: SimpleNamespace(connect=Connection))
    monkeypatch.setattr(inference.repo, "source", lambda _: {"source_id": "synthetic"})
    monkeypatch.setattr(
        inference.repo,
        "navs",
        lambda _c, _code, _s, start, _end: [
            source(str(start), current["base" if start == date(2026, 9, 14) else "target"])
        ],
    )
    monkeypatch.setattr(inference.repo, "observe", lambda _c, _code, _s, rows, _now: rows)

    def save(_c, _kind, _key, payload, _now, _expires):
        sid, h = str(uuid4()), digest(payload)
        snapshots.append({"snapshot_id": sid, "payload": json.loads(canonical(payload)), "content_hash": h})
        return sid, h

    monkeypatch.setattr(inference.repo, "save_snapshot", save)
    first = inference.labels(uuid4())
    assert first["payload"]["base_unit_nav"] == "1"
    assert first["payload"]["training_eligible"] is True
    revised = inference.labels(uuid4())
    assert revised["payload"]["base_unit_nav"] == "1.1"
    assert revised["payload"]["base_revised"] is True
    assert revised["payload"]["training_eligible"] is False
    assert inference.labels(uuid4())["snapshot_id"] == revised["snapshot_id"]
    current["target"] = "0.9"
    third = inference.labels(uuid4())
    assert third["payload"]["actual_direction"] == "DOWN"
    assert first["payload"]["actual_direction"] == "UP"
    assert len(snapshots) == 3
    assert digest(json.loads(third["payload_json"])) == third["content_hash"]


def test_future_target_does_not_fetch_or_make_answer(monkeypatch):
    body = {"target_nav_date": "2026-09-15"}
    monkeypatch.setattr(
        inference.repo,
        "get_job",
        lambda _: {
            "state": "SUCCEEDED",
            "kind": "FORECAST",
            "result": {"payload_json": canonical(body), "content_hash": digest(body)},
        },
    )
    monkeypatch.setattr(inference.repo, "clock", lambda: datetime(2026, 9, 14, tzinfo=ZONE))
    monkeypatch.setattr(inference, "get_engine", lambda: pytest.fail("未到期不应查询答案"))
    assert inference.labels(uuid4()) == {"status": "PENDING_TARGET"}
