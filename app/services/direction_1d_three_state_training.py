"""一日三分类重新拟合、时间切分检查和精确复现；不改旧研究数据及模型文件。"""

import hashlib
import warnings
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from threadpoolctl import threadpool_limits

from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.services import direction_1d_training as old
from app.services.direction_1d_protocol import FEATURE_VERSION, FEATURES, RECIPE, ZONE, canonical, digest
from app.services.direction_1d_three_state import CLASSES, POLICY, PROTOCOL, TARGET, TIE_ORDER, predict


def fit(rows, group, cohort, as_of):
    """各类至少 30 个家族日期、总计至少 252 个日期；标准化只使用训练集。"""
    rows = sorted(rows, key=lambda row: (row["t"], row["family"], row["fund_code"]))
    counts = Counter(label for _, _, label in {(r["family"], r["t"], r["actual_direction"]) for r in rows})
    if len({r["t"] for r in rows}) < 252 or any(counts[key] < 30 for key in CLASSES):
        raise ValueError("INSUFFICIENT_THREE_STATE_SAMPLES")
    if any(datetime.fromisoformat(r["mature_at"]) > as_of for r in rows):
        raise ValueError("IMMATURE_TRAINING_SAMPLE")
    x, y, weights = (
        np.asarray([r["x"] for r in rows]),
        np.asarray([r["actual_direction"] for r in rows]),
        old.weights(rows),
    )
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaler = StandardScaler().fit(x, sample_weight=weights)
        classifier = LogisticRegression(**RECIPE).fit(scaler.transform(x), y, sample_weight=weights)
    if classifier.classes_.tolist() != list(CLASSES):
        raise ValueError("THREE_CLASSES_REQUIRED")
    weighted_counts = {key: float(weights[y == key].sum()) for key in CLASSES}
    model = {
        "protocol": PROTOCOL,
        "target_definition": TARGET,
        "horizon": 1,
        "feature_version": FEATURE_VERSION,
        "features": list(FEATURES),
        "recipe": RECIPE,
        "direction_policy": POLICY,
        "class_order": list(CLASSES),
        "tie_order": list(TIE_ORDER),
        "group_id": group,
        "cohort_id": cohort,
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "train_as_of": as_of.isoformat(),
        "fit_start": rows[0]["t"],
        "fit_end": rows[-1]["t"],
        "fit_count": len(rows),
        "distinct_dates": len({r["t"] for r in rows}),
        "classes": dict(counts),
        "fit_hash": digest(rows),
        "weight_hash": digest(weights.tolist()),
        "majority": max(TIE_ORDER, key=lambda key: weighted_counts[key]),
    }
    restored = np.asarray([[predict(model, r["x"])["class_scores"][key] for key in CLASSES] for r in rows])
    difference = float(np.max(np.abs(restored - classifier.predict_proba(scaler.transform(x)))))
    if difference > 1e-12:
        raise ValueError("MODEL_RESTORE_MISMATCH")
    model["restore_max_score_diff"] = difference
    return model


def metrics(model, rows):
    """按三类逐项核对；没有持平样本时持平召回率为空，不能冒充 100%。"""
    actual = [r["actual_direction"] for r in rows]
    predicted = [predict(model, r["x"])["direction"] for r in rows]
    recalls = {
        key: sum(a == p == key for a, p in zip(actual, predicted, strict=True)) / actual.count(key)
        if key in actual
        else None
        for key in CLASSES
    }
    count = len(rows)
    return {
        "count": count,
        "correct": sum(a == p for a, p in zip(actual, predicted, strict=True)),
        "recall": recalls,
        "predicted_counts": dict(Counter(predicted)),
        "actual_counts": dict(Counter(actual)),
        "kind": "HISTORICAL_DEVELOPMENT_CHECK",
        "majority_correct": actual.count(model["majority"]),
        "independent_validation": False,
    }


def training_input(connection, now):
    """固定旧训练名单；仅加入 Java 已核对的首次、未修订、未过期前向样本。"""
    frozen = connection.execute(
        text("""SELECT spec FROM direction_1d_training_run
      WHERE spec->>'protocol'='DIRECTION_1D_V1' ORDER BY created_at LIMIT 1""")
    ).scalar_one()
    directory = frozen["artifact_directory"]
    root = (old.RUN_ROOT / directory).resolve()
    if root.parent != old.RUN_ROOT.resolve() or not directory.startswith("direction-1d-"):
        raise ValueError("TRAINING_PATH_INVALID")
    history = old.read(root / "history.json")
    if digest(history) != frozen["data_hash"]:
        raise ValueError("FROZEN_INPUT_CHANGED")
    known = {fund["fund_code"]: fund for fund in history["funds"]}
    found = (
        connection.execute(
            text("""SELECT s.payload,s.task_key,s.content_hash,a.assessed_at,i.payload AS input,
      i.content_hash AS input_hash FROM direction_1d_snapshot s
      JOIN direction_1d_assessment_ack a ON a.task_key=s.task_key AND a.label_hash=s.content_hash
      JOIN direction_1d_snapshot i ON i.snapshot_id=(s.payload->>'input_snapshot_id')::uuid
      WHERE s.kind='LABEL' AND s.expires_at>:now AND i.expires_at>:now AND a.assessed_at<=:now
        AND s.payload->>'training_eligible'='true' ORDER BY s.as_of LIMIT 50001"""),
            {"now": now},
        )
        .mappings()
        .all()
    )
    if len(found) > 50000:
        raise ValueError("ASSESSED_SAMPLE_LIMIT_REACHED")
    samples = {}
    for row in found:
        inp, answer = row["input"], row["payload"]
        code = inp["fund_code"]
        if code not in known or answer["actual_direction"] not in CLASSES:
            continue
        if digest(inp) != row["input_hash"] or digest(answer) != row["content_hash"]:
            raise ValueError("TRAINING_EVIDENCE_HASH_MISMATCH")
        # 同基金同一目标日新旧协议可各留档一次，训练时只能算同一份真实答案。
        key = (code, inp["target_nav_date"])
        if key in samples:
            continue
        samples[key] = {
            "fund_code": code,
            "family": known[code]["product_family_id"],
            "group": known[code]["group_id"],
            "t": inp["base_nav_date"],
            "u": inp["target_nav_date"],
            "x": inp["features"],
            "y": answer["y"],
            "actual_direction": answer["actual_direction"],
            "mature_at": row["assessed_at"].isoformat(),
            "input_hash": row["input_hash"],
            "label_hash": row["content_hash"],
            "kind": "FORWARD_ORIGINAL",
        }
    history["forward_samples"] = list(samples.values())
    return history, "D1V2-" + digest([frozen["cohort_id"], POLICY, RECIPE])[:24]


def train_registered():
    """持数据库锁防并发重复训练；相同输入零拟合，新增模型与旧档案独立登记。"""
    with get_engine().connect() as lock:
        if not lock.execute(text("SELECT pg_try_advisory_lock(721124)")).scalar_one():
            return {"status": "SKIPPED", "reason": "TRAINING_IN_PROGRESS", "fits": 0}
        try:
            return _train_registered()
        finally:
            lock.execute(text("SELECT pg_advisory_unlock(721124)"))


def _train_registered():
    now = repo.clock()
    with get_engine().connect() as connection:
        history, cohort = training_input(connection, now)
        previous = connection.execute(
            text("""SELECT spec FROM direction_1d_training_run
          WHERE cohort_id=:cohort ORDER BY created_at DESC LIMIT 1"""),
            {"cohort": cohort},
        ).scalar_one_or_none()
    data_hash = digest(history)
    if previous and previous["data_hash"] == data_hash:
        return {"status": "SKIPPED", "reason": "NO_NEW_ASSESSED_SAMPLES", "fits": 0}
    rows = old.build_samples(history)
    # 首次三分类沿用原首模的开发历史；后续更新才按成熟的前向样本推进窗口。
    # 2025 被保护、2026 前向样本很少时不能拼接缺口或放宽 252 日门槛。
    training_rows = rows if previous else [row for row in rows if row["kind"] == "HISTORICAL_RECONSTRUCTION"]
    directory = old.RUN_ROOT / ("direction-1d-three-state-" + str(uuid4()))
    directory.mkdir(parents=True)
    groups = sorted({r["group"] for r in rows})
    spec = {
        "run_id": str(uuid4()),
        "protocol": PROTOCOL,
        "cohort_id": cohort,
        "train_as_of": now.isoformat(),
        "data_hash": data_hash,
        "artifact_directory": directory.name,
        "retention_days": history["retention_days"],
        "groups": groups,
        "recipe": RECIPE,
        "direction_policy": POLICY,
        "history_usage_manifest": history["history_usage_manifest"],
        "code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    old.write_new(directory / "protocol.json", spec)
    old.write_new(directory / "history.json", history)
    old.write_new(directory / "dataset.json", rows)
    models, reports, failures = {}, {}, {}
    for group in groups:
        selected = old.select_fit([r for r in training_rows if r["group"] == group], now)
        try:
            model = fit(selected, group, cohort, now)
            replay = fit(selected, group, cohort, now)
            if model != replay:
                raise ValueError("REPLAY_MODEL_MISMATCH")
            models[group] = model
        except ValueError as error:
            failures[group] = str(error)
            continue
        # 2024 是既有开发历史，不把这一检查包装为全新独立验证。
        cutoff = datetime(2024, 1, 1, tzinfo=ZONE)
        before = old.select_fit([r for r in rows if r["group"] == group], cutoff)
        exam = [r for r in rows if r["group"] == group and "2024-01-01" <= r["u"] < "2025-01-01"]
        try:
            reports[group] = metrics(fit(before, group, cohort, cutoff), exam)
        except ValueError as error:
            reports[group] = {"status": "UNAVAILABLE", "reason": str(error)}
    old.write_new(directory / "models.json", models)
    old.write_new(directory / "metrics.json", reports)
    proof = {
        "model_equal": True,
        "model_hash": digest(models),
        "metrics_hash": digest(reports),
        "model_released": False,
        "failures": failures,
        "finished_at": repo.clock().isoformat(),
    }
    old.write_new(directory / "completion.json", proof)
    records = []
    old.MODEL_ROOT.mkdir(exist_ok=True)
    with get_engine().begin() as connection:
        connection.execute(
            text("""INSERT INTO direction_1d_training_run
          (run_id,cohort_id,train_as_of,spec,spec_hash,status,evidence)
          VALUES(:id,:cohort,:asof,CAST(:spec AS jsonb),:hash,'VERIFIED',CAST(:proof AS jsonb))"""),
            {
                "id": spec["run_id"],
                "cohort": cohort,
                "asof": now,
                "spec": canonical(spec),
                "hash": digest(spec),
                "proof": canonical(proof),
            },
        )
        for group, model in models.items():
            identity = str(uuid4())
            old.write_new(old.MODEL_ROOT / (identity + ".json"), model)
            connection.execute(
                text("""INSERT INTO direction_1d_model
              (model_id,run_id,cohort_id,group_id,horizon,file_name,content_hash,metadata,trained_at,expires_at)
              VALUES(:id,:run,:cohort,:group,1,:file,:hash,CAST(:meta AS jsonb),:trained,:expires)"""),
                {
                    "id": identity,
                    "run": spec["run_id"],
                    "cohort": cohort,
                    "group": group,
                    "file": identity + ".json",
                    "hash": digest(model),
                    "meta": canonical(model),
                    "trained": proof["finished_at"],
                    "expires": now + timedelta(days=history["retention_days"]),
                },
            )
            records.append({"model_id": identity, "group_id": group, "hash": digest(model)})
    old.write_new(directory / "model-manifest.json", records)
    return {
        "status": "SUCCEEDED" if len(records) == len(groups) else "PARTIAL" if records else "FAILED",
        "models": records,
        "failures": failures,
        "run": directory.name,
        "development_metrics": reports,
    }
