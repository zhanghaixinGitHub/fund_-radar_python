"""真实窗口推理及到期答案；只接纳已登记模型，原始成功结果不重算。"""

import hashlib
from datetime import date, datetime
from uuid import UUID

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.services.direction_1d_data import classify
from app.services.direction_1d_protocol import (
    FEATURE_VERSION,
    PROTOCOL,
    TARGET,
    ZONE,
    canonical,
    digest,
    features,
    input_days,
    label,
    score,
    window,
)
from app.services.direction_1d_selection import ACTIVATION_POLICY, available_at_prediction
from app.services.direction_1d_training import MODEL_ROOT


def load_model(row):
    import json

    name = row["file_name"]
    if name != str(row["model_id"]) + ".json":
        raise ValueError("MODEL_PATH_INVALID")
    path = MODEL_ROOT / name
    if path.is_symlink() or path.resolve().parent != MODEL_ROOT.resolve():
        raise ValueError("MODEL_PATH_INVALID")
    with path.open("rb") as f:
        raw = f.read(65537)
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != row["content_hash"]:
        raise ValueError("MODEL_HASH_MISMATCH")
    model = json.loads(raw)
    if (
        model != row["metadata"]
        or model.get("group_id") != row["group_id"]
        or model.get("cohort_id") != row["cohort_id"]
    ):
        raise ValueError("MODEL_REGISTRY_MISMATCH")
    return model


def infer(code: str) -> dict:
    now = repo.clock()
    if abs((datetime.now(ZONE) - now).total_seconds()) > 5:
        raise ValueError("CLOCK_SKEW")
    w = window(now)
    if w["status"] != "OPEN":
        raise ValueError("MISSED_DEADLINE")
    wanted = input_days(date.fromisoformat(w["base_nav_date"]))
    with get_engine().connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
        source = repo.source(c)
        ps = repo.profiles(c, [code])
        if not ps:
            raise ValueError("SOURCE_UNAVAILABLE")
        mapping = classify(ps[0])
        if not mapping["group_id"]:
            raise ValueError(mapping["classification_reason"])
        all_models = [m for m in repo.models(c) if m["group_id"] == mapping["group_id"]]
        if not all_models:
            raise ValueError("MODEL_PENDING")
        # 初始cohort稳定；新增组/名单需要受控新cohort，不能按最新文件修改时间猜测。
        cohort = all_models[0]["cohort_id"]
        eligible = [m for m in all_models if m["cohort_id"] == cohort and available_at_prediction(m, now)]
        if not eligible:
            # 无模型只记失败作业；不能锁死一个空映射，阻止随后真实完成的模型。
            raise ValueError("MODEL_UNAVAILABLE")
        for branch in ("FIXED", "WEEKLY"):
            selected = (eligible[0] if branch == "FIXED" else eligible[-1]) if eligible else None
            c.execute(
                text("""INSERT INTO
          direction_1d_window_model(cohort_id,group_id,base_nav_date,target_nav_date,branch_id,model_id,activation_policy)
              VALUES(:cohort,:group,:base,:target,:branch,:model,:policy) ON CONFLICT DO NOTHING"""),
                {
                    "cohort": cohort,
                    "group": mapping["group_id"],
                    "base": w["base_nav_date"],
                    "target": w["target_nav_date"],
                    "branch": branch,
                    "policy": ACTIVATION_POLICY,
                    "model": selected["model_id"] if selected else None,
                },
            )
        locked = (
            c.execute(
                text("""SELECT w.branch_id,w.locked_at,m.* FROM direction_1d_window_model w
          LEFT JOIN direction_1d_model m USING(model_id)
          WHERE w.cohort_id=:cohort AND w.group_id=:group AND w.target_nav_date=:target
            AND w.activation_policy=:policy ORDER BY w.branch_id"""),
                {
                    "cohort": cohort,
                    "group": mapping["group_id"],
                    "target": w["target_nav_date"],
                    "policy": ACTIVATION_POLICY,
                },
            )
            .mappings()
            .all()
        )
        if not any(r["model_id"] for r in locked):
            # 新策略与旧空映射分开留存，已存在的真实预测仍由作业幂等键复用。
            unavailable = True
        else:
            unavailable = False
            rows = repo.navs(c, code, source["source_id"], wanted[0], wanted[-1])
            by_date = {r["nav_date"]: r for r in rows}
            if any(d not in by_date for d in wanted):
                raise ValueError("DATA_PENDING")
            x = features([by_date[d]["unit_nav"] for d in wanted])
            observed = repo.observe(c, code, source, [by_date[d] for d in wanted], now)
            events = [
                dict(r)
                for r in c.execute(
                    text("""SELECT ex_date,nav_ex_date,content_hash FROM fund_dividend
              WHERE fund_code=:code AND source_id=:source AND (ex_date=:target OR nav_ex_date=:target) LIMIT 100"""),
                    {"code": code, "source": source["source_id"], "target": w["target_nav_date"]},
                ).mappings()
            ]
            key = f"{PROTOCOL}:{cohort}:{code}:{w['target_nav_date']}"
            snapshot = {
                "fund_code": code,
                "target_definition": TARGET,
                "feature_version": FEATURE_VERSION,
                "base_nav_date": w["base_nav_date"],
                "target_nav_date": w["target_nav_date"],
                "values": observed,
                "features": x,
                "feature_as_of": now.isoformat(),
                "max_input_observed_at": now.isoformat(),
                "event_status": "KNOWN_EVENT" if events else "UNKNOWN",
                "events": events,
                "event_limitations": "已查本地现金分红；未授权拆分完整源，无记录不能证明无所有事件。",
            }
            expires = min(datetime.fromisoformat(v["expires_at"]) for v in observed)
            sid, ih = repo.save_snapshot(c, "INPUT", key, snapshot, now, expires)
    if unavailable:
        raise ValueError("MODEL_NOT_ACTIVE_FOR_WINDOW")
    # 输入写入提交后重新读取，防止未提交快照被当成可恢复证据。
    with get_engine().connect() as c:
        saved = (
            c.execute(text("SELECT payload,content_hash FROM direction_1d_snapshot WHERE snapshot_id=:id"), {"id": sid})
            .mappings()
            .one()
        )
        if digest(saved["payload"]) != ih or saved["content_hash"] != ih:
            raise ValueError("INPUT_HASH_MISMATCH")
    branches = []
    majority = None
    for row in locked:
        b = {
            "branch_id": row["branch_id"],
            "model_id": str(row["model_id"]) if row["model_id"] else None,
            "model_hash": row["content_hash"],
            "score": None,
            "predicted_direction": None,
            "status": "MODEL_UNAVAILABLE",
        }
        try:
            if not row["model_id"] or row["expires_at"] <= now:
                raise ValueError("MODEL_UNAVAILABLE")
            model = load_model(row)
            s = score(model, x)
            b.update(
                score=s,
                predicted_direction="UP" if s > 0.5 else "NON_UP",
                status="AVAILABLE",
                train_as_of=model["train_as_of"],
                trained_at=row["trained_at"].isoformat(),
                registered_at=row["registered_at"].isoformat(),
                model_selected_at=row["locked_at"].isoformat(),
            )
            if row["branch_id"] == "FIXED":
                majority = model["majority"]
        except (ValueError, OSError, KeyError) as error:
            b["reason"] = str(error) if isinstance(error, ValueError) else "MODEL_UNAVAILABLE"
        branches.append(b)
    if not any(b["status"] == "AVAILABLE" for b in branches):
        raise ValueError("MODEL_UNAVAILABLE")
    generated = repo.clock()
    if generated >= datetime.fromisoformat(w["deadline_at"]):
        raise ValueError("MISSED_DEADLINE")
    body = {
        "schema_version": "DIRECTION_1D_EXPERIMENT_V1",
        "protocol": PROTOCOL,
        "activation_policy": ACTIVATION_POLICY,
        "task_key": key,
        "cohort_id": cohort,
        "fund_code": code,
        "fund_name": ps[0]["fund_name"],
        "group_id": mapping["group_id"],
        "product_family_id": mapping["product_family_id"],
        "status": "PREDICTED",
        "experiment_status": "EXPERIMENTAL",
        "horizon_trading_days": 1,
        "target_definition": TARGET,
        "model_released": False,
        "up_probability": None,
        **{k: v for k, v in w.items() if k != "status"},
        "input_snapshot_id": sid,
        "input_hash": ih,
        "input_json": canonical(saved["payload"]),
        "input": saved["payload"],
        "generated_at": generated.isoformat(),
        "expires_at": expires.isoformat(),
        "latest_nav_date": w["base_nav_date"],
        "branches": branches,
        "baselines": [
            {"branch_id": k, "predicted_direction": None if v is None else "UP" if v else "NON_UP"}
            for k, v in (
                ("ALWAYS_UP", 1),
                ("ALWAYS_NON_UP", 0),
                ("INITIAL_MAJORITY", majority),
                ("MOMENTUM", int(float(observed[-1]["unit_nav"]) > float(observed[-2]["unit_nav"]))),
            )
        ],
        "kind": "FORWARD_ORIGINAL",
        "limitations": ["模型分数尚未校准，效果未验证。", "单位净值涨跌不等于分红后投资回报。"],
    }
    payload = canonical(body)
    return {"payload_json": payload, "content_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def labels(job_id: UUID):
    import json

    job = repo.get_job(job_id)
    if not job or job["state"] != "SUCCEEDED" or job["kind"] != "FORECAST":
        raise ValueError("FORECAST_NOT_FOUND")
    original_raw = job["result"]["payload_json"]
    if hashlib.sha256(original_raw.encode("utf-8")).hexdigest() != job["result"]["content_hash"]:
        raise ValueError("FORECAST_HASH_MISMATCH")
    body = json.loads(original_raw)
    now = repo.clock()
    if date.fromisoformat(body["target_nav_date"]) > now.astimezone(ZONE).date():
        return {"status": "PENDING_TARGET"}
    if now >= datetime.fromisoformat(body["expires_at"]):
        return {"status": "EVIDENCE_EXPIRED"}
    with get_engine().connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
        source = repo.source(c)
        day = date.fromisoformat(body["target_nav_date"])
        rows = repo.navs(c, body["fund_code"], source["source_id"], day, day)
        if not rows:
            # 仅到期且公布时段已开始才请求公共答案；异步作业不会阻塞Java核对线程。
            if now.astimezone(ZONE) >= datetime.combine(day, datetime.min.time().replace(hour=18), ZONE):
                from app.services.direction_1d_jobs import submit

                block = int(now.timestamp()) // 1800
                submit(
                    f"LABEL_SYNC:{body['fund_code']}:{day}:{block}",
                    "LABEL_SYNC",
                    {
                        "fund_code": body["fund_code"],
                        "base_nav_date": body["base_nav_date"],
                        "target_nav_date": str(day),
                    },
                )
            return {"status": "PENDING_NAV"}
        observed = repo.observe(c, body["fund_code"], source, rows, now)[0]
        base = body["input"]["values"][-1]
        first_exists = c.execute(
            text("SELECT 1 FROM direction_1d_snapshot WHERE kind='LABEL' AND task_key=:key LIMIT 1"),
            {"key": body["task_key"]},
        ).first()
        # 首次一律按预测时冻结T；以后读取到T修订时追加展示，不偷偷改变首次结论。
        if first_exists:
            t = date.fromisoformat(body["base_nav_date"])
            revised = repo.navs(c, body["fund_code"], source["source_id"], t, t)
            if revised:
                base = repo.observe(c, body["fund_code"], source, revised, now)[0]
        base_revised = base["content_hash"] != body["input"]["values"][-1]["content_hash"]
        events = [
            dict(r)
            for r in c.execute(
                text("""SELECT ex_date,nav_ex_date,content_hash FROM fund_dividend
          WHERE fund_code=:code AND source_id=:source AND (ex_date=:target OR nav_ex_date=:target)
          ORDER BY content_hash LIMIT 100"""),
                {"code": body["fund_code"], "source": source["source_id"], "target": day},
            ).mappings()
        ]
        answer = label(base["unit_nav"], observed["unit_nav"])
        payload = {
            **answer,
            "task_key": body["task_key"],
            "input_snapshot_id": body["input_snapshot_id"],
            "label_observed_at": now.isoformat(),
            "base_source": base,
            "target_source": observed,
            "target_nav_date": body["target_nav_date"],
            "event_status": "KNOWN_EVENT" if events else "UNKNOWN",
            "events": events,
            "base_revised": base_revised,
            # T已修订意味着原始七特征与新T不再一致；该修订仅供核对，不冒充完整训练输入。
            "training_eligible": not base_revised,
            "status": "AVAILABLE",
            "kind": "FORWARD_ORIGINAL",
            "target_definition": TARGET,
        }
        # 同来源版本重复读取返回原答案快照；新版本才追加，日期/首次结果不漂移。
        key_hash = digest({"base": base["content_hash"], "target": observed["content_hash"], "events": events})
        previous = (
            c.execute(
                text("""SELECT snapshot_id,payload,content_hash FROM direction_1d_snapshot
          WHERE kind='LABEL' AND task_key=:key AND payload->>'revision_key'=:hash ORDER BY as_of LIMIT 1"""),
                {"key": body["task_key"], "hash": key_hash},
            )
            .mappings()
            .first()
        )
        if previous:
            return {
                "snapshot_id": str(previous["snapshot_id"]),
                "content_hash": previous["content_hash"],
                "payload": previous["payload"],
                "payload_json": canonical(previous["payload"]),
            }
        payload["revision_key"] = key_hash
        sid, h = repo.save_snapshot(
            c, "LABEL", body["task_key"], payload, now, datetime.fromisoformat(body["expires_at"])
        )
    return {"snapshot_id": sid, "content_hash": h, "payload": payload, "payload_json": canonical(payload)}
