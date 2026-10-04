"""真实 PostgreSQL 独立 schema 验证互斥与请求边界；不生成正式预测、不登记模型。"""

from datetime import date, datetime, timedelta
from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from app.db.session import get_engine
from app.repositories.prediction_store import encode
from app.services import multi_prediction_sync as sync
from app.services import prediction_generation as generation
from app.services.prediction_contract import PredictionFailure, ValuationCalendar, prediction_policy, target_dates
from sqlalchemy import create_engine, text
from tests.test_multi_prediction_sync import setup_worker


@pytest.fixture
def isolated(monkeypatch):
    original = get_engine()
    if original.url.host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("仅连接本机的独立测试 schema")
    schema = "sync_isolation_" + uuid4().hex
    with original.begin() as c:
        c.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(original.url, connect_args={"options": f"-c search_path={schema}"}, hide_parameters=True)
    with engine.begin() as c:
        c.execute(
            text("""CREATE TABLE prediction_generation_task (
            task_id uuid PRIMARY KEY, request_key varchar(200) UNIQUE, mode text, status text,
            payload jsonb, result jsonb, lease_until timestamptz, created_at timestamptz DEFAULT clock_timestamp(),
            finished_at timestamptz);
            CREATE TABLE prediction_generation_item (task_id uuid, fund_code text, horizon_id text,
            status text DEFAULT 'PENDING', attempts int DEFAULT 0, result jsonb, prediction_id uuid,
            PRIMARY KEY(task_id,fund_code,horizon_id));
            CREATE TABLE prediction_task_event (event_id uuid, task_id uuid, kind text, payload jsonb);""")
        )
    clock = [datetime.fromisoformat("2026-09-30T09:00:00+08:00")]
    queued = []
    monkeypatch.setattr(generation, "get_engine", lambda: engine)
    monkeypatch.setattr(generation, "freeze_routes", lambda: {"test": {"revision": 1}})
    monkeypatch.setattr(generation, "_creation_time", lambda c: clock[0])
    monkeypatch.setattr(generation, "_executor", SimpleNamespace(submit=lambda fn, task: queued.append(task)))
    try:
        yield SimpleNamespace(engine=engine, clock=clock, queued=queued)
    finally:
        engine.dispose()
        # schema 由本测试生成且不接收外部输入，只删除本次隔离数据。
        with original.begin() as c:
            c.execute(text(f"DROP SCHEMA {schema} CASCADE"))


@pytest.mark.parametrize("change", ["none", "close", "date", "policy", "subset", "legacy_order"])
def test_active_task_reuse_requires_same_window_policy_and_actual_items(isolated, change):
    first = generation.create_task(["000001"])
    horizons = [h["horizon_id"] for h in prediction_policy()["horizons"]]
    retry_items = None
    if change in {"close", "date"}:
        isolated.clock[0] = datetime.fromisoformat(
            "2026-09-30T15:00:00+08:00" if change == "close" else "2026-10-01T09:00:00+08:00"
        )
    if change in {"policy", "legacy_order"}:
        with isolated.engine.begin() as c:
            field, value = (
                ("policyHash", "old-policy") if change == "policy" else ("horizonIds", list(reversed(horizons)))
            )
            c.execute(
                text(
                    "UPDATE prediction_generation_task "
                    "SET payload=jsonb_set(payload,CAST(:p AS text[]),CAST(:v AS jsonb))"
                ),
                {"p": "{" + field + "}", "v": encode(value)},
            )
    if change == "subset":
        retry_items = {("000001", horizons[0])}
    second = generation.create_task(["000001"], retry_items=retry_items)
    assert second["taskId"] == (first["taskId"] if change in {"none", "legacy_order"} else None)
    assert second["blockedByPreviousTask"] == (change not in {"none", "legacy_order"})
    assert len(isolated.queued) == 1


def test_old_partial_result_is_not_counted_as_current_sync_and_other_batch_can_create(isolated, monkeypatch):
    old = generation.create_task(["000001"])
    with isolated.engine.begin() as c:
        c.execute(
            text("""UPDATE prediction_generation_item SET status='CREATED',result=CAST(:r AS jsonb)
            WHERE task_id=:id AND horizon_id='T5_V1'"""),
            {"id": old["taskId"], "r": encode({"startDate": "2026-09-30"})},
        )
    isolated.clock[0] = datetime.fromisoformat("2026-09-30T15:01:00+08:00")
    worker = setup_worker(monkeypatch, count=1)
    monkeypatch.setattr(sync, "create_task", generation.create_task)
    result = worker.service.sync(progress_reporter=lambda *args: None)
    multi = [i for i in result.items if i["horizonId"] != "T1"]
    assert len(multi) == 3 and all(i["state"] == "WAITING" and i["targetDate"] is None for i in multi)
    assert result.created == 0 and result.existing == 1  # 只计本次一日预测
    assert not any(path.endswith("/finalize") for _, path in worker.calls)
    assert generation.create_task(["000002"])["taskId"] != old["taskId"]
    assert len(isolated.queued) == 2


def test_retry_after_old_task_finishes_gets_current_date(isolated):
    first = generation.create_task(["000001"])
    with isolated.engine.begin() as c:
        c.execute(
            text("UPDATE prediction_generation_task SET status='FAILED' WHERE task_id=:id"), {"id": first["taskId"]}
        )
    isolated.clock[0] = datetime.fromisoformat("2026-10-09T09:00:00+08:00")
    second = generation.create_task(["000001"])
    assert second["taskId"] != first["taskId"]
    with isolated.engine.connect() as c:
        assert (
            c.execute(
                text("SELECT payload->>'generatedAt' FROM prediction_generation_task WHERE task_id=:id"),
                {"id": second["taskId"]},
            )
            .scalar()
            .startswith("2026-10-09")
        )


def test_private_explicit_request_cannot_reuse_other_owners_task(isolated):
    first = generation.create_task(["000001", "000002"], request_key="owner-one")
    with pytest.raises(PredictionFailure) as failure:
        generation.create_task(["000001"], request_key="owner-two")
    assert failure.value.payload["code"] == "TASK_SCOPE_BUSY"
    assert failure.value.payload["details"] == {}
    assert generation.create_task(["000001", "000002"], request_key="owner-one")["taskId"] == first["taskId"]
    assert len(isolated.queued) == 1


def test_dispatch_failure_has_terminal_state_and_next_request_can_retry(isolated, monkeypatch):
    def fail(*args):
        raise RuntimeError("injected executor shutdown")

    monkeypatch.setattr(generation, "_executor", SimpleNamespace(submit=fail))
    failed = generation.create_task(["000001"])
    assert failed["status"] == "FAILED" and failed["pendingItems"] == 0
    assert all(i["result"]["error"]["code"] == "TASK_DISPATCH_FAILED" for i in failed["items"])
    monkeypatch.setattr(generation, "_executor", SimpleNamespace(submit=lambda fn, task: isolated.queued.append(task)))
    retry = generation.create_task(["000001"])
    assert retry["taskId"] != failed["taskId"] and retry["status"] == "QUEUED"
    assert len(isolated.queued) == 1


def test_recovery_dispatch_failure_keeps_success_and_continues_other_tasks(isolated, monkeypatch):
    tasks = [generation.create_task([f"{code:06d}"]) for code in (1, 2, 3)]
    with isolated.engine.begin() as c:
        c.execute(
            text(
                "UPDATE prediction_generation_item SET status='CREATED',result='{}' "
                "WHERE task_id=:id AND horizon_id='T5_V1'"
            ),
            {"id": tasks[0]["taskId"]},
        )
    dispatched = []

    def dispatch(fn, task):
        if str(task) == tasks[0]["taskId"]:
            raise RuntimeError("injected first recovery dispatch failure")
        dispatched.append(str(task))

    monkeypatch.setattr(generation, "_executor", SimpleNamespace(submit=dispatch))
    assert generation.recover_tasks() == 2
    failed = generation.task_status(tasks[0]["taskId"])
    assert failed["status"] == "FAILED" and failed["createdItems"] == 1 and failed["pendingItems"] == 0
    assert dispatched == [t["taskId"] for t in tasks[1:]]


@pytest.mark.parametrize("partial_horizons", [False, True])
def test_one_sync_isolates_all_overlapping_old_tasks_from_runnable_items(isolated, monkeypatch, partial_horizons):
    """在同一次sync中复现局部占用；同时覆盖多个旧任务、周期子集和无关基金。"""
    horizons = {h["horizon_id"] for h in prediction_policy()["horizons"]}
    if partial_horizons:
        occupied = {("000001", "T5_V1"), ("000001", "M6_V1"), ("000002", "T20_V1")}
        generation.create_task(["000001"], retry_items={("000001", "T5_V1")})
        generation.create_task(["000001", "000002"], retry_items=occupied - {("000001", "T5_V1")})
    else:
        occupied = {("000001", horizon) for horizon in horizons}
        generation.create_task(["000001"])
    previous = list(isolated.queued)
    isolated.clock[0] = datetime.fromisoformat("2026-10-01T09:00:00+08:00")

    def finish_fixture(_fn, task_id):
        isolated.queued.append(task_id)
        with isolated.engine.begin() as c:
            c.execute(
                text(
                    "UPDATE prediction_generation_item SET status='CREATED',result=CAST(:r AS jsonb) WHERE task_id=:id"
                ),
                {"id": task_id, "r": encode({"startDate": "2026-10-01"})},
            )
            c.execute(
                text("UPDATE prediction_generation_task SET status='SUCCEEDED' WHERE task_id=:id"), {"id": task_id}
            )

    monkeypatch.setattr(generation, "_executor", SimpleNamespace(submit=finish_fixture))
    worker = setup_worker(monkeypatch, count=3)
    monkeypatch.setattr(sync, "create_task", generation.create_task)
    original_post = worker.service._client.post
    monkeypatch.setattr(
        worker.service._client,
        "post",
        lambda url, **kw: (
            httpx.Response(200, json={"failed": 0}, request=httpx.Request("POST", "http://test/finalize"))
            if url.endswith("/finalize")
            else original_post(url, **kw)
        ),
    )
    result = worker.service.sync(progress_reporter=lambda *args: None)
    multi = {(i["fundCode"], i["horizonId"]): i for i in result.items if i["horizonId"] != "T1"}
    assert len(multi) == 9 and len(result.items) == 12
    assert {key for key, value in multi.items() if value["state"] == "WAITING"} == occupied
    assert all(
        value["state"] == "COMPLETED" and value["targetDate"] == "2026-10-01"
        for key, value in multi.items()
        if key not in occupied
    )
    assert len(isolated.queued) == len(previous) + 1
    for task in previous:
        assert generation.task_status(task)["status"] == "QUEUED"
    fresh = generation.task_status(isolated.queued[-1])
    assert {(i["fundCode"], i["horizonId"]) for i in fresh["items"]} == set(multi) - occupied


def test_expired_lease_does_not_recover_or_duplicate_live_worker(isolated, monkeypatch):
    task = generation.create_task(["000001"])
    entered, release = Event(), Event()
    runs = []

    def execute(task_id):
        runs.append(task_id)
        with isolated.engine.begin() as c:
            c.execute(
                text(
                    "UPDATE prediction_generation_task SET status='RUNNING',"
                    "lease_until=clock_timestamp()-interval '1 second' WHERE task_id=:id"
                ),
                {"id": task_id},
            )
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(generation, "_run_task", execute)
    thread = Thread(target=generation.run_task, args=(task["taskId"],))
    thread.start()
    try:
        assert entered.wait(5)
        worker = setup_worker(monkeypatch, count=1)
        monkeypatch.setattr(sync, "create_task", generation.create_task)
        monkeypatch.setattr(sync, "WAIT_SECONDS", 0)
        result = worker.service.sync(progress_reporter=lambda *args: None)
        assert sum(i["state"] == "WAITING" for i in result.items) == 3
        generation.run_task(task["taskId"])
        assert generation.recover_tasks() == 0
        assert len(runs) == 1 and generation.task_status(task["taskId"])["status"] == "RUNNING"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert generation.recover_tasks() == 1
    assert generation.task_status(task["taskId"])["status"] == "INTERRUPTED"


def test_real_create_run_store_sync_reuses_exact_dates_with_inference_substitute(isolated, monkeypatch):
    """任务/日期/保存/统计用真实代码和PG；只有输入及推理被替换，无训练和模型注册。"""
    from app.services import prediction_task_inputs
    from app.services.prediction_direction import direction_fields

    with isolated.engine.begin() as c:
        c.execute(
            text("""CREATE TABLE fund_prediction_record (
            prediction_id uuid PRIMARY KEY,period_key text UNIQUE,fund_code text,horizon_id text,mode text,
            model_id text,activation_revision int,generated_at timestamptz,content_hash text,payload jsonb);
            CREATE TABLE prediction_attempt(
                attempt_id uuid,task_id uuid,fund_code text,horizon_id text,payload jsonb);""")
        )
    days = tuple(date(2030, 1, 1) + timedelta(days=i) for i in range(730))
    calendar = ValuationCalendar(
        "ISOLATED_WEEKDAYS", tuple(d for d in days if d.weekday() < 5), days[0], days[-1], "test"
    )
    monkeypatch.setattr(prediction_task_inputs, "task_input", lambda _mode, _task, _code, read: read())
    monkeypatch.setattr(
        generation,
        "read_fund_data",
        lambda code, now: {"calendar": calendar, "fund": {"fund_code": code, "fund_type": "MIXED"}},
    )
    monkeypatch.setattr(generation, "build_features", lambda *args: {})
    # 路由非空占位，不读取/更换实际模型；推理返回的是明确的隔离夹具值。
    monkeypatch.setattr(
        generation,
        "freeze_routes",
        lambda: {generation.route_key(h["horizon_id"]): {"fixture": True} for h in prediction_policy()["horizons"]},
    )

    def payload(data, features, horizon, instant, route, *, task_id):
        return {
            "predictionId": str(uuid4()),
            "fundCode": data["fund"]["fund_code"],
            **target_dates(calendar, instant, horizon["horizon_id"]),
            **direction_fields(horizon["horizon_id"]),
            "mode": "LIVE",
            "modelId": "isolated-fixture",
            "activationRevision": 1,
            "generatedAt": instant.isoformat(),
            "role": "PRIMARY",
        }

    monkeypatch.setattr(generation, "prediction_payload", payload)
    monkeypatch.setattr(generation, "_executor", SimpleNamespace(submit=lambda fn, task: fn(task)))
    worker = setup_worker(monkeypatch, count=1)
    monkeypatch.setattr(sync, "create_task", generation.create_task)
    original_post = worker.service._client.post
    monkeypatch.setattr(
        worker.service._client,
        "post",
        lambda url, **kw: (
            httpx.Response(200, json={"failed": 0}, request=httpx.Request("POST", "http://test/finalize"))
            if url.endswith("/finalize")
            else original_post(url, **kw)
        ),
    )
    for instant, created, existing, start in [
        ("2030-09-30T09:00:00+08:00", 3, 1, "2030-09-30"),
        ("2030-09-30T10:00:00+08:00", 0, 4, "2030-09-30"),
        ("2030-09-30T15:00:00+08:00", 3, 1, "2030-10-01"),
        ("2030-10-01T09:00:00+08:00", 0, 4, "2030-10-01"),
    ]:
        isolated.clock[0] = datetime.fromisoformat(instant)
        result = worker.service.sync(progress_reporter=lambda *args: None)
        assert (result.total, result.created, result.existing) == (4, created, existing)
        assert {i["targetDate"] for i in result.items if i["horizonId"] != "T1"} == {start}
    with isolated.engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM fund_prediction_record")).scalar() == 6
