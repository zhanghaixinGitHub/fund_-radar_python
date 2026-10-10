"""冻结规则下的新事件特征对照与接入；最多16次拟合，不修改旧实验。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
from app.db.session import get_engine
from app.services import direction_1d_events as events
from app.services import direction_1d_information as runtime
from app.services.direction_1d_protocol import canonical, digest
from app.services.direction_1d_training import MODEL_ROOT
from app.services.fund_exposure_common import read, save
from scipy.optimize import minimize
from scipy.special import expit
from sqlalchemy import text

from scripts import fund_002112_event_comparison_data_v1 as data
from scripts import fund_002112_event_comparison_v1 as old

OUT = data.PY / ".local-runs/direction-1d-information/20261009-event-v2"
COHORT = "002112_EVENT_EVIDENCE_USER_20261009_V2"


def log(value):
    with (OUT / "fit-ledger.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(canonical({"at": datetime.now(data.ZONE).isoformat(), **value}) + "\n")


def fit(rows, name, phase, as_of):
    """固定L2正则，无截距、无中心化、非负约束；没有缺失标记或数量系数。"""
    path = OUT / "fits" / (name + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    ledger = data.jsonl(OUT / "fit-ledger.jsonl") if (OUT / "fit-ledger.jsonl").exists() else []
    if any(r["name"] == name for r in ledger) or sum(r["state"] == "STARTED" for r in ledger) >= 16:
        raise ValueError("FIT_BUDGET_OR_RETRY_BLOCKED")
    if len(rows) < 30 or len({r["label"] for r in rows}) < 2:
        return None
    assert all(data.moment(r["mature_at"]) <= data.moment(as_of) for r in rows)
    matrix = np.asarray([events.transform(r["information"]) for r in rows])
    y = np.asarray([r["label"] == "UP" for r in rows], dtype=float)
    log({"name": name, "state": "STARTED", "rows": len(rows)})

    def loss(coef):
        z = matrix @ coef
        return float(np.logaddexp(0, z).sum() - y @ z + 0.5 * coef @ coef), matrix.T @ (expit(z) - y) + coef

    fitted = minimize(
        loss,
        np.zeros(len(events.FEATURES)),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0, None)] * len(events.FEATURES),
        options={"maxiter": 1000},
    )
    if not fitted.success:
        log({"name": name, "state": "FAILED"})
        raise ValueError("EVENT_FIT_FAILED")
    model = {
        "protocol": "DIRECTION_1D_V2",
        "horizon": 1,
        "fund_code": "002112",
        "phase": phase,
        "feature_version": events.VERSION,
        "features": events.FEATURES,
        "group_id": events.GROUPS[phase],
        "cohort_id": COHORT,
        "target_definition": "UNIT_NAV_DIRECTION_THREE_STATE_V2",
        "classes": ["DOWN", "UP"],
        "unsupported_classes": ["FLAT"],
        "train_as_of": as_of,
        "majority": Counter(r["label"] for r in rows).most_common(1)[0][0],
        "intercept": 0,
        "coef": fitted.x.tolist(),
        "policy": events.POLICY,
        "training_rows": len(rows),
        "recipe_hash": data.sha(Path(events.__file__)),
        "validation_status": "HISTORICAL_ONLY_EXPERIMENTAL",
        "adoption_basis": "EXPLICIT_USER_SELECTION",
    }
    events.validate_model(model)
    path.parent.mkdir(exist_ok=True)
    path.write_text(canonical(model), encoding="utf-8")
    log({"name": name, "state": "SUCCEEDED", "hash": data.sha(path)})
    return model


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "prepared.json").exists():
        return read(OUT / "prepared.json")
    paths = [data.ROOT / x for x in ("dataset.jsonl", "input-evidence.jsonl", "all-predictions.jsonl")]
    protocol = {
        "version": events.VERSION,
        "policy": events.POLICY,
        "features": events.FEATURES,
        "hashes": {str(p): data.sha(p) for p in paths + [Path(events.__file__), Path(__file__)]},
        "max_fits": 16,
        "input": "原299条冻结输入与当时事件证据，不重新采集或重跑旧实验",
    }
    frozen = OUT / "protocol.json"
    if frozen.exists():
        if read(frozen) != protocol:
            raise ValueError("EVENT_PROTOCOL_CHANGED")
    else:
        save(frozen, protocol)
    proofs = {(r["phase"], r["target"]): r for r in data.jsonl(paths[1])}
    sessions = data.load_sources()["sessions"]
    rows = data.jsonl(paths[0])
    for row in rows:
        proof = proofs[row["phase"], row["target"]]
        assert row["input_hash"] == proof["input_hash"]
        row["information"] = {
            "numeric": row["groups"]["C_EVENTS"],
            "holding_coverage": row["groups"]["C_EVENTS"][11],
            "event_evidence": events.build_evidence(proof["events"], row["base"], row["cutoff"], sessions),
        }
    predictions, models = [], {}
    for phase in data.PHASES:
        selected = [r for r in rows if r["phase"] == phase]
        for block, start, end in old.BLOCKS:
            as_of = data.cutoff_for(start, phase)
            train = [r for r in selected if r["target"] < start and data.moment(r["mature_at"]) <= data.moment(as_of)]
            model = fit(train, phase + "-" + block, phase, as_of)
            if model:
                for row in selected:
                    if start <= row["target"] < end:
                        result = events.predict(model, row["information"])
                        predictions.append(
                            {
                                "phase": phase,
                                "target": row["target"],
                                "label": row["label"],
                                "direction": result["direction"],
                                "status": result["status"],
                                "decision": result["decision"],
                            }
                        )
        final = fit(
            [r for r in selected if data.moment(r["mature_at"]) <= data.moment(old.FINAL_AS_OF)],
            phase + "-FINAL",
            phase,
            old.FINAL_AS_OF,
        )
        if final is None:
            raise ValueError("FINAL_SAMPLE_INSUFFICIENT")
        model_id = str(uuid4())
        path = OUT / "models" / (model_id + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(canonical(final), encoding="utf-8")
        models[phase] = {
            "model_id": model_id,
            "model_hash": data.sha(path),
            "group_id": events.GROUPS[phase],
            "trained_at": datetime.now(data.ZONE).isoformat(),
        }
    old_predictions = data.jsonl(paths[2])
    summary = {}
    for phase in data.PHASES:
        sample = [r for r in predictions if r["phase"] == phase]
        emitted = [r for r in sample if r["status"] == "AVAILABLE"]
        summary[phase] = {
            "total": len(sample),
            "predicted": len(emitted),
            "abstained": len(sample) - len(emitted),
            "correct": sum(r["direction"] == r["label"] for r in emitted),
            "accuracy_when_predicted": (sum(r["direction"] == r["label"] for r in emitted) / len(emitted))
            if emitted
            else None,
            "coverage": len(emitted) / len(sample) if sample else 0,
            "reasons": dict(Counter(c for r in sample for c in r["decision"]["reason_codes"])),
        }
        same = {(r["phase"], r["target"]): r for r in sample}
        baseline = [
            r
            for r in old_predictions
            if r["group"] == "C_EVENTS" and r["phase"] == phase and (r["phase"], r["target"]) in same
        ]
        summary[phase]["old_same_dates"] = {
            "total": len(baseline),
            "correct": sum(r["actual"] == r["predicted"] for r in baseline),
            "down_predicted_up": sum(r["actual"] == "DOWN" and r["predicted"] == "UP" for r in baseline),
        }
        summary[phase]["new_down_predicted_up"] = sum(r["label"] == "DOWN" and r["direction"] == "UP" for r in sample)
    save(
        OUT / "summary.json",
        {
            "new": summary,
            "old_predictions_count": len(old_predictions),
            "claim": "拒判率单独报告，不将选择后的正确率称为整体改进",
        },
    )
    with (OUT / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(canonical(row) + "\n")
    prepared = {
        "fund_code": "002112",
        "run_id": str(uuid4()),
        "cohort_id": COHORT,
        "feature_version": events.VERSION,
        "models": models,
        "source_hashes": {
            str(p.relative_to(data.PY)).replace("\\", "/"): data.sha(p) for p in (data.SOURCE, data.MATERIAL)
        },
        "new_fits": sum(r["state"] == "STARTED" for r in data.jsonl(OUT / "fit-ledger.jsonl")),
        "adoption_basis": "EXPLICIT_USER_SELECTION",
        "summary": summary,
    }
    save(OUT / "prepared.json", prepared)
    print(json.dumps({"summary": summary, "fits": prepared["new_fits"]}, ensure_ascii=False))
    return prepared


def activate():
    prepared = prepare()
    with get_engine().begin() as c:
        c.execute(text("SELECT pg_advisory_xact_lock(721129,2112)"))
        source = (
            c.execute(text("SELECT enabled,retention_days FROM source_registry WHERE source_code='TUSHARE_PRO_FUND'"))
            .mappings()
            .one()
        )
        if not source["enabled"] or source["retention_days"] <= 0:
            raise ValueError("SOURCE_UNAVAILABLE")
        now = c.execute(text("SELECT clock_timestamp()")).scalar_one()
        c.execute(
            text("""INSERT INTO direction_1d_training_run(run_id,cohort_id,train_as_of,spec,spec_hash,status,evidence)
          VALUES(:id,:cohort,:asof,CAST(:spec AS jsonb),:hash,'USER_SELECTED',CAST(:evidence AS jsonb))
          ON CONFLICT DO NOTHING"""),
            {
                "id": UUID(prepared["run_id"]),
                "cohort": COHORT,
                "asof": old.FINAL_AS_OF,
                "spec": canonical(prepared),
                "hash": digest(prepared),
                "evidence": canonical(prepared["summary"]),
            },
        )
        for row in prepared["models"].values():
            filename = row["model_id"] + ".json"
            raw = (OUT / "models" / filename).read_bytes()
            if data.sha(OUT / "models" / filename) != row["model_hash"]:
                raise ValueError("EVENT_MODEL_HASH_MISMATCH")
            events.validate_model(json.loads(raw))
            target = MODEL_ROOT / filename
            if target.exists():
                if target.read_bytes() != raw:
                    raise ValueError("EVENT_MODEL_CONFLICT")
            else:
                target.write_bytes(raw)
            c.execute(
                text("""INSERT INTO direction_1d_model(model_id,run_id,cohort_id,group_id,horizon,file_name,
                content_hash,metadata,trained_at,expires_at) VALUES(:id,:run,:cohort,:group,1,:filename,:hash,
                CAST(:metadata AS jsonb),:trained,:expires) ON CONFLICT DO NOTHING"""),
                {
                    "id": UUID(row["model_id"]),
                    "run": UUID(prepared["run_id"]),
                    "cohort": COHORT,
                    "group": row["group_id"],
                    "filename": filename,
                    "hash": row["model_hash"],
                    "metadata": raw.decode(),
                    "trained": datetime.fromisoformat(row["trained_at"]),
                    "expires": now + timedelta(days=source["retention_days"]),
                },
            )
    if runtime.ACTIVE_FILE.exists():
        previous = read(runtime.ACTIVE_FILE)
        archive = OUT / ("previous-" + digest(previous) + ".json")
        if not archive.exists():
            save(archive, previous)
    active = {k: prepared[k] for k in ("fund_code", "feature_version", "models", "source_hashes", "adoption_basis")}
    active.update(enabled=True, activated_at=now.isoformat())
    save(runtime.ACTIVE_FILE, active, replace=runtime.ACTIVE_FILE.exists())
    print("ACTIVATED_002112_EVENT_EVIDENCE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "activate"))
    {"prepare": prepare, "activate": activate}[parser.parse_args().command]()
