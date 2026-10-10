"""按用户明确选择导出/接入已训练的C模型；不重新拟合、不修改旧实验。

prepare仅导出并在全部冻结输入上对照原joblib；activate登记后原子启用002112路由。
rollback仅停用路由，保留已登记模型、原始预测和可恢复的采用记录。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import joblib
import numpy as np
from app.db.session import get_engine
from app.services import direction_1d_information as runtime
from app.services.direction_1d_protocol import canonical, digest
from app.services.direction_1d_training import MODEL_ROOT
from app.services.fund_exposure_common import read, save
from sqlalchemy import text

from scripts import fund_002112_event_comparison_data_v1 as data
from scripts import fund_002112_event_comparison_v1 as experiment

OUT = data.PY / ".local-runs/direction-1d-information/20261009-v1"
COHORT = "002112_INFORMATION_USER_20261009_V1"


def prepare() -> dict:
    if (OUT / "prepared.json").exists():
        return read(OUT / "prepared.json")
    protocol = experiment.prepare()
    rows = data.jsonl(data.ROOT / "dataset.jsonl")
    ledger = data.jsonl(data.ROOT / "fit-ledger.jsonl")
    models = {}
    run_id = str(uuid4())
    for phase in data.PHASES:
        fit_id = phase + "-FINAL-C_EVENTS"
        success = next(r for r in ledger if r["fit_id"] == fit_id and r["status"] == "SUCCEEDED")
        original = data.ROOT / "models" / (fit_id + ".joblib")
        if data.sha(original) != success["model_hash"]:
            raise ValueError("RESEARCH_MODEL_CHANGED")
        bundle = joblib.load(original)
        model_id = str(uuid4())
        model = {
            "protocol": "DIRECTION_1D_V2",
            "horizon": 1,
            "fund_code": "002112",
            "phase": phase,
            "feature_version": runtime.FEATURE_VERSION,
            "features": data.NAMES["C_EVENTS"],
            "group_id": runtime.GROUPS[phase],
            "cohort_id": COHORT,
            "target_definition": "UNIT_NAV_DIRECTION_THREE_STATE_V2",
            "classes": bundle["classifier"].classes_.tolist(),
            "unsupported_classes": ["FLAT"],
            "train_as_of": bundle["train_as_of"],
            "majority": bundle["majority"],
            "adoption_basis": "EXPLICIT_USER_SELECTION",
            "validation_status": "HISTORICAL_ONLY_NO_STABLE_GAIN",
            "medians": bundle["medians"].tolist(),
            "mean": bundle["scaler"].mean_.tolist(),
            "scale": bundle["scaler"].scale_.tolist(),
            "coef": bundle["classifier"].coef_[0].tolist(),
            "intercept": float(bundle["classifier"].intercept_[0]),
            "tfidf": {
                "vocabulary": {k: int(v) for k, v in bundle["vectorizer"].vocabulary_.items()},
                "idf": bundle["vectorizer"].idf_.tolist(),
            },
            "recipe_hashes": {
                str(path.relative_to(data.PY)).replace("\\", "/"): data.sha(path)
                for path in experiment.CODE
                if path.name != "fund_002112_event_comparison_v1.py"
            },
            "research_evidence": {
                "fit_id": fit_id,
                "model_sha256": success["model_hash"],
                "protocol_sha256": data.sha(data.ROOT / "frozen-protocol.json"),
                "training_rows": len(bundle["train_ids"]),
                "new_fits": 0,
            },
        }
        runtime.validate_model(model)
        matched = [row for row in rows if row["phase"] == phase]
        expected = experiment.probabilities(bundle, experiment.transform(bundle, matched))
        actual = np.asarray(
            [
                [
                    runtime.predict(model, {"numeric": row["groups"]["C_EVENTS"], "text": row["text"]})["class_scores"][
                        label
                    ]
                    for label in experiment.CLASSES
                ]
                for row in matched
            ]
        )
        error = float(np.max(np.abs(expected - actual)))
        if error > 1e-12:
            raise ValueError("EXPORT_PROBABILITY_MISMATCH")
        model["export_max_probability_diff"] = error
        (OUT / "models").mkdir(exist_ok=True)
        path = OUT / "models" / (model_id + ".json")
        path.write_text(canonical(model), encoding="utf-8")
        models[phase] = {
            "model_id": model_id,
            "model_hash": data.sha(path),
            "trained_at": success["at"],
            "group_id": runtime.GROUPS[phase],
            "checked_rows": len(matched),
            "max_diff": error,
        }
    prepared = {
        "fund_code": "002112",
        "run_id": run_id,
        "cohort_id": COHORT,
        "feature_version": runtime.FEATURE_VERSION,
        "models": models,
        "user_instruction": "我现在要把页面的模型接成又有净值，又有持仓行情，又有公告、政策和新闻的模型",
        "adoption_basis": "EXPLICIT_USER_SELECTION",
        "new_fits": 0,
        "enabled": False,
        "source_hashes": {
            str(p.relative_to(data.PY)).replace("\\", "/"): data.sha(p) for p in (data.SOURCE, data.MATERIAL)
        },
        "protocol": protocol,
    }
    save(OUT / "prepared.json", prepared)
    print(json.dumps({"models": models, "new_fits": 0}, ensure_ascii=False))
    return prepared


def activate() -> None:
    prepared = prepare()
    spec = {key: prepared[key] for key in ("fund_code", "user_instruction", "adoption_basis", "models", "new_fits")}
    with get_engine().begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(721129,2112)"))
        source = connection.execute(
            text("SELECT enabled,retention_days FROM source_registry WHERE source_code='TUSHARE_PRO_FUND'")
        )
        permission = source.mappings().one()
        if not permission["enabled"] or permission["retention_days"] <= 0:
            raise ValueError("SOURCE_UNAVAILABLE")
        now = connection.execute(text("SELECT clock_timestamp()")).scalar_one()
        connection.execute(
            text("""INSERT INTO direction_1d_training_run
            (run_id,cohort_id,train_as_of,spec,spec_hash,status,evidence)
            VALUES(:id,:cohort,:asof,CAST(:spec AS jsonb),:hash,'USER_SELECTED',CAST(:evidence AS jsonb))
            ON CONFLICT(run_id) DO NOTHING"""),
            {
                "id": UUID(prepared["run_id"]),
                "cohort": COHORT,
                "asof": experiment.FINAL_AS_OF,
                "spec": canonical(spec),
                "hash": digest(spec),
                "evidence": canonical(prepared["models"]),
            },
        )
        MODEL_ROOT.mkdir(exist_ok=True)
        for row in prepared["models"].values():
            filename = row["model_id"] + ".json"
            original = OUT / "models" / filename
            raw = original.read_bytes()
            if data.sha(original) != row["model_hash"]:
                raise ValueError("EXPORT_HASH_MISMATCH")
            runtime.validate_model(json.loads(raw))
            target = MODEL_ROOT / filename
            if target.exists():
                if target.read_bytes() != raw:
                    raise ValueError("MODEL_PATH_CONFLICT")
            else:
                with target.open("xb") as stream:
                    stream.write(raw)
            connection.execute(
                text("""INSERT INTO direction_1d_model
                (model_id,run_id,cohort_id,group_id,horizon,file_name,content_hash,metadata,trained_at,expires_at)
                VALUES(:id,:run,:cohort,:group,1,:filename,:hash,CAST(:metadata AS jsonb),:trained,:expires)
                ON CONFLICT(model_id) DO NOTHING"""),
                {
                    "id": UUID(row["model_id"]),
                    "run": UUID(prepared["run_id"]),
                    "cohort": COHORT,
                    "group": row["group_id"],
                    "filename": filename,
                    "hash": row["model_hash"],
                    "metadata": raw.decode(),
                    "trained": datetime.fromisoformat(row["trained_at"]),
                    "expires": now + timedelta(days=permission["retention_days"]),
                },
            )
    # 注册事务成功后才启用；切换前后路由均留不可变副本，失败可按同一登记包重试。
    if runtime.ACTIVE_FILE.exists():
        previous = read(runtime.ACTIVE_FILE)
        archive = OUT / ("previous-" + digest(previous) + ".json")
        if not archive.exists():
            save(archive, previous)
    active = {
        key: prepared[key] for key in ("fund_code", "feature_version", "models", "source_hashes", "adoption_basis")
    }
    active.update(enabled=True, activated_at=now.isoformat())
    save(runtime.ACTIVE_FILE, active, replace=runtime.ACTIVE_FILE.exists())
    archive = OUT / ("activation-" + digest(active) + ".json")
    if not archive.exists():
        save(archive, active)
    print("ACTIVATED_002112_INFORMATION", prepared["run_id"])


def rollback() -> None:
    active = read(runtime.ACTIVE_FILE)
    active["enabled"] = False
    save(runtime.ACTIVE_FILE, active, replace=True)
    print("INFORMATION_ROUTE_DISABLED_OLD_RECORDS_RETAINED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "activate", "rollback"))
    {"prepare": prepare, "activate": activate, "rollback": rollback}[parser.parse_args().command]()
