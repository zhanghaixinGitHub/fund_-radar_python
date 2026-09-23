"""本机隔离 PostgreSQL schema 验收；不改变真实路由，结束时删除本测试拥有的schema。"""

import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.db.session import get_engine
from app.services import prediction_models as models
from app.services import prediction_selection as selection
from app.services.prediction_contract import PredictionFailure, fingerprint
from sqlalchemy import create_engine, text
from tests.test_prediction_models import candidate, package


@pytest.fixture
def database(monkeypatch, tmp_path):
    original = get_engine()
    if original.url.host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("仅对本机隔离schema运行")
    schema = "prediction_test_" + uuid4().hex
    with original.begin() as c:
        c.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(original.url, connect_args={"options": f"-c search_path={schema}"}, hide_parameters=True)
    try:
        for filename in (
            "20260922_24_prediction_platform.py",
            "20260922_25_prediction_recovery.py",
            "20260922_26_prediction_check_state.py",
        ):
            path = Path(__file__).parents[1] / "alembic/versions" / filename
            spec = importlib.util.spec_from_file_location("prediction_migration", path)
            migration = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(migration)
            with engine.begin() as c:
                monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(c)))
                migration.upgrade()
        monkeypatch.setattr(models, "get_engine", lambda: engine)
        monkeypatch.setattr(selection, "get_engine", lambda: engine)
        monkeypatch.setattr(models, "model_directory", lambda: tmp_path)
        models.bootstrap_models()
        yield engine
    finally:
        engine.dispose()
        with original.begin() as c:
            c.execute(text(f"DROP SCHEMA {schema} CASCADE"))


def test_atomic_next_batch_and_real_package_fallback(database):
    before = models.freeze_routes()
    key = models.route_key("T5_V1")
    p = package()
    registered = models.register_model(p)
    activation = models.activate(registered["modelId"], reason={"test": "A10"}, expected_revision=1)
    after = models.freeze_routes()
    assert before[key]["revision"] == 1 and after[key]["revision"] == activation["revision"] == 2
    features = dict.fromkeys(models.FEATURES, 0.0) | {"momentum": 0.1}
    assert models.infer_route(before[key], features, "T5_V1")["modelId"] == before[key]["model_id"]
    assert models.infer_route(after[key], features, "T5_V1")["modelId"] == registered["modelId"]
    # 仅损坏隔离目录中的测试包；回退回执不能继续显示请求的新模型。
    (models.model_directory() / (registered["modelId"] + ".json")).write_text("{}", encoding="utf-8")
    fallback = models.infer_route(after[key], features, "T5_V1")
    assert fallback["modelId"] == before[key]["model_id"]
    assert fallback["requestedModelId"] == registered["modelId"]
    assert fallback["fallbackReason"][0]["code"] == "MODEL_HASH_MISMATCH"
    with database.connect() as c:
        assert c.execute(text("SELECT count(*) FROM model_activation_event WHERE action='FALLBACK'")).scalar() == 1


def test_selection_persists_and_activates(database):
    key = models.route_key("T5_V1")
    route = models.freeze_routes()[key]
    registered = models.register_model(package())
    protocol = {"version": "EXPERIMENT_SELECTION_V1", "fixture": "A07"}
    old, new = candidate(route["model_id"]), candidate(registered["modelId"], 0.58)
    old["protocolHash"] = new["protocolHash"] = fingerprint(protocol)
    result = selection.compare_and_activate("fixture-a07", old, [new], protocol)
    assert result["decision"] == "ACTIVATE"
    assert models.freeze_routes()[key]["model_id"] == registered["modelId"]
    with pytest.raises(PredictionFailure, match="其他任务"):
        models.activate(registered["modelId"], reason={}, expected_revision=1)
    with database.begin() as c, pytest.raises(Exception, match="append only"):
        c.execute(text("UPDATE model_evaluation SET payload='{}'"))


def test_original_prediction_and_revised_outcomes(database, monkeypatch):
    from datetime import UTC, date, datetime
    from decimal import Decimal

    from app.repositories import prediction_store as store
    from app.services import prediction_generation as generation
    from tests.test_prediction_contract import calendar

    monkeypatch.setattr(store, "get_engine", lambda: database)
    monkeypatch.setattr(generation, "get_engine", lambda: database)
    model = models.freeze_routes()[models.route_key("T5_V1")]["model_id"]
    now = datetime.now(UTC)
    original = {
        "predictionId": str(uuid4()),
        "fundCode": "006730",
        "horizonId": "T5_V1",
        "mode": "LIVE",
        "modelId": model,
        "activationRevision": 1,
        "generatedAt": "2026-09-15T01:00:00+00:00",
        "startDate": "2026-09-15",
        "endDate": "2026-09-16",
        "direction": "UP",
        "role": "PRIMARY",
        "targetDefinitionId": "FIXTURE_TOTAL_RETURN",
        "reason": "不可改写的人工隔离样例",
    }
    with database.begin() as c:
        first, created = store.save_prediction(c, original, "fixture-original")
        repeated, second_created = store.save_prediction(
            c, original | {"predictionId": str(uuid4()), "direction": "NON_UP"}, "fixture-original"
        )
    assert created and not second_created and first == repeated == original
    values = {date(2026, 9, 15): Decimal("1"), date(2026, 9, 16): Decimal("1.1")}
    monkeypatch.setattr(
        generation,
        "read_fund_data",
        lambda *a, **kw: {
            "calendar": calendar(),
            "navs": [{"nav_date": d, "unit_nav": v, "updated_at": now} for d, v in values.items()],
            "dividends": [],
            "dividendWatermark": now,
        },
    )
    generation.verify_outcomes()
    generation.verify_outcomes()
    with database.connect() as c:
        assert c.execute(text("SELECT count(*) FROM prediction_outcome")).scalar() == 1
    values[date(2026, 9, 16)] = Decimal("0.9")
    generation.verify_outcomes()
    with database.connect() as c:
        assert c.execute(text("SELECT count(*) FROM prediction_outcome")).scalar() == 2
        assert c.execute(text("SELECT payload->>'direction' FROM fund_prediction_record")).scalar() == "UP"
    assert len(store.history("006730")[0]["outcomes"]) == 2
    # 新增两份独立夹具；limit=1也必须轮转到每份，而不是一直重查已有结果的第一份。
    with database.begin() as c:
        for suffix in ("rotation-a", "rotation-b"):
            store.save_prediction(c, original | {"predictionId": str(uuid4())}, suffix)
    for _ in range(3):
        generation.verify_outcomes(limit=1)
    with database.connect() as c:
        assert c.execute(text("SELECT count(*) FROM prediction_check_state")).scalar() == 3
    first_page = store.history("006730", limit=1)[0]["payload"]
    second_page = store.history(
        "006730", limit=1, before=first_page["generatedAt"], before_id=first_page["predictionId"]
    )[0]["payload"]
    assert first_page["predictionId"] != second_page["predictionId"]
    with database.begin() as c, pytest.raises(Exception, match="append only"):
        c.execute(text("UPDATE fund_prediction_record SET payload='{}'"))


def test_committed_checkpoint_recovers_after_worker_is_killed(database, monkeypatch, tmp_path):
    """真实子进程写入一项后强制中断；恢复租约和原始期次，不假装为生产故障演练。"""
    import subprocess
    import sys
    import time

    from app.services import prediction_generation as generation

    monkeypatch.setattr(generation, "get_engine", lambda: database)

    class Deferred:
        def __init__(self):
            self.calls = []

        def submit(self, *args):
            self.calls.append(args)

    executor = Deferred()
    monkeypatch.setattr(generation, "_executor", executor)
    task = generation.create_task(["006730"], request_key="kill-test")
    with database.connect() as c:
        schema = c.execute(text("SELECT current_schema()")).scalar()
    worker = tmp_path / "checkpoint_worker.py"
    worker.write_text(
        """import sys,time
from sqlalchemy import create_engine,text
from app.db.session import get_engine
engine=create_engine(get_engine().url,connect_args={"options":"-c search_path="+sys.argv[1]},hide_parameters=True)
with engine.begin() as c:
 c.execute(text("UPDATE prediction_generation_task SET status='RUNNING',"
 "lease_until=clock_timestamp()+interval '2 seconds' WHERE task_id=:id"),{"id":sys.argv[2]})
 c.execute(text("UPDATE prediction_generation_item SET status='FAILED',attempts=1,"
 "result=CAST(:payload AS jsonb) WHERE task_id=:id AND horizon_id='T5_V1'"),
 {"id":sys.argv[2],"payload":'{"generationStatus":"FAILED"}'})
print("COMMITTED",flush=True)
time.sleep(120)
""",
        encoding="utf-8",
    )
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    process = subprocess.Popen(
        [sys.executable, str(worker), schema, task["taskId"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        assert process.stdout.readline().strip() == "COMMITTED"
        process.kill()
        process.wait(timeout=10)
        time.sleep(2.1)
        assert generation.recover_tasks() == 1
        restored = generation.task_status(task["taskId"])
        assert restored["failedItems"] == 1 and restored["pendingItems"] == 2
        assert restored["plannedItems"] == 3

        def unavailable(*args, **kwargs):
            raise PredictionFailure("FIXTURE_INPUT_MISSING", "SOURCE", "隔离测试缺少输入")

        monkeypatch.setattr(generation, "read_fund_data", unavailable)
        generation.run_task(task["taskId"])
        completed = generation.task_status(task["taskId"])
        assert completed["status"] == "FAILED" and completed["pendingItems"] == 0
        assert completed["failedItems"] == 3
        assert all(item["attempts"] == 1 for item in completed["items"])
        assert generation.create_task(["006730"], request_key="kill-test")["taskId"] == task["taskId"]
        with database.connect() as c:
            assert c.execute(text("SELECT count(*) FROM prediction_task_event WHERE kind='RECOVERED'")).scalar() == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
