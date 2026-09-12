"""固定预算的离线1日训练：冻结输入、四季度开发、首模及唯一同配方复现。"""

import hashlib
import importlib.metadata
import json
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from threadpoolctl import threadpool_limits

from app.db.session import get_engine
from app.services.direction_1d_protocol import (
    FEATURE_VERSION,
    FEATURES,
    PROTOCOL,
    RECIPE,
    TARGET,
    ZONE,
    calendar,
    canonical,
    digest,
    features,
    label,
    score,
)

RUN_ROOT = Path(__file__).resolve().parents[2] / ".local-runs"
MODEL_ROOT = RUN_ROOT / "direction-1d-models"


def write_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as f:
        f.write(canonical(value))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint() -> dict:
    root = Path(__file__).resolve().parent
    return {
        "code": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("direction_1d_protocol.py", "direction_1d_training.py")
        },
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "threadpoolctl")},
        "calendar_hash": calendar()[1],
    }


def build_samples(history: dict) -> list[dict]:
    days, _ = calendar()
    rows = []
    for fund in history["funds"]:
        values = {date.fromisoformat(r["date"]): r for r in fund["rows"]}
        for i in range(60, len(days) - 2):
            t, u, available = days[i : i + 3]
            if u.year > 2024:
                break
            wanted = days[i - 60 : i + 2]
            if not all(d in values for d in wanted):
                continue
            try:
                x = features([values[d]["nav"] for d in wanted[:-1]])
                answer = label(values[t]["nav"], values[u]["nav"])
            except ValueError:
                continue
            # ann_date可能更迟，取保守较晚时间；旧日期不伪装为本地过去已收到。
            mature = datetime.combine(available, time(8), ZONE)
            for d in (t, u):
                ann = values[d].get("ann_date")
                if ann:
                    mature = max(mature, datetime.combine(date.fromisoformat(ann), time(8), ZONE))
            rows.append(
                {
                    "fund_code": fund["fund_code"],
                    "family": fund["product_family_id"],
                    "group": fund["group_id"],
                    "t": str(t),
                    "u": str(u),
                    "x": x,
                    "y": answer["y"],
                    "actual_direction": answer["actual_direction"],
                    "mature_at": mature.isoformat(),
                    "momentum": int(values[t]["nav"] and float(values[t]["nav"]) > float(values[days[i - 1]]["nav"])),
                    "input_hash": digest([values[d] for d in wanted[:-1]]),
                    "label_hash": digest([values[t], values[u]]),
                    "kind": "HISTORICAL_RECONSTRUCTION",
                }
            )
    rows.extend(history.get("forward_samples", []))
    return sorted(rows, key=lambda r: (r["t"], r["family"], r["fund_code"]))


def weights(rows: list[dict]) -> np.ndarray:
    families = defaultdict(set)
    multiplicity = Counter((r["family"], r["t"]) for r in rows)
    for r in rows:
        families[r["family"]].add(r["t"])
    n_pairs = len(multiplicity)
    return np.asarray(
        [n_pairs / (len(families) * len(families[r["family"]]) * multiplicity[(r["family"], r["t"])]) for r in rows]
    )


def select_fit(rows, as_of: datetime):
    mature = [r for r in rows if datetime.fromisoformat(r["mature_at"]) <= as_of]
    if not mature:
        return []
    days, _ = calendar()
    last = max(date.fromisoformat(r["t"]) for r in mature)
    allowed = set(str(d) for d in days[max(0, days.index(last) - 503) : days.index(last) + 1])
    return [r for r in mature if r["t"] in allowed]


def fit(rows: list[dict], group: str, cohort: str, as_of: datetime) -> dict:
    """只在FIT拟合Scaler；按产品家族×日期控制权重；技术失败不启动新参数试验。"""
    counts = Counter({(r["family"], r["t"], r["y"]): 1 for r in rows})
    classes = Counter()
    for _, _, y in counts:
        classes[y] += 1
    if len({r["t"] for r in rows}) < 252 or min(classes.get(0, 0), classes.get(1, 0)) < 30:
        raise ValueError("INSUFFICIENT_GROUP_SAMPLES")
    x, y, w = np.asarray([r["x"] for r in rows]), np.asarray([r["y"] for r in rows]), weights(rows)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaler = StandardScaler().fit(x, sample_weight=w)
        classifier = LogisticRegression(**RECIPE).fit(scaler.transform(x), y, sample_weight=w)
    result = {
        "protocol": PROTOCOL,
        "horizon": 1,
        "target_definition": TARGET,
        "feature_version": FEATURE_VERSION,
        "features": list(FEATURES),
        "recipe": RECIPE,
        "group_id": group,
        "cohort_id": cohort,
        "coef": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "train_as_of": as_of.isoformat(),
        "fit_start": rows[0]["t"],
        "fit_end": rows[-1]["t"],
        "fit_count": len(rows),
        "distinct_dates": len({r["t"] for r in rows}),
        "family_dates": len({(r["family"], r["t"]) for r in rows}),
        "classes": dict(classes),
        "fit_hash": digest(rows),
        "weight_hash": digest(w.tolist()),
        "majority": int(float(np.dot(y, w)) > float(w.sum()) / 2),
        "weight_total": float(w.sum()),
    }
    restored = np.asarray([score(result, r["x"]) for r in rows])
    original = classifier.predict_proba(scaler.transform(x))[:, 1]
    result["restore_max_score_diff"] = float(np.max(np.abs(restored - original)))
    if result["restore_max_score_diff"] > 1e-12:
        raise ValueError("MODEL_RESTORE_MISMATCH")
    return result


def metrics(rows, scores, majority):
    if not rows:
        return {"count": 0, "accuracy": None, "reason": "NO_EXAM_SAMPLES"}
    y = [r["y"] for r in rows]
    predictions = [int(s > 0.5) for s in scores]

    def summary(p):
        correct = sum(a == b for a, b in zip(p, y, strict=True))
        recalls = [
            sum(a == b == cls for a, b in zip(p, y, strict=True)) / y.count(cls) if cls in y else None for cls in (1, 0)
        ]
        return {
            "correct": correct,
            "count": len(y),
            "accuracy": correct / len(y),
            "up_recall": recalls[0],
            "non_up_recall": recalls[1],
            "balanced_accuracy": sum(recalls) / 2 if None not in recalls else None,
        }

    return {
        "MODEL": summary(predictions),
        "ALWAYS_UP": summary([1] * len(y)),
        "ALWAYS_NON_UP": summary([0] * len(y)),
        "INITIAL_MAJORITY": summary([majority] * len(y)),
        "MOMENTUM": summary([r["momentum"] for r in rows]),
        "distinct_dates": len({r["u"] for r in rows}),
        "flat_count": sum(r["actual_direction"] == "FLAT" for r in rows),
    }


def freeze(history: dict, run_dir: Path, *, weekly_cohort: str | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    groups = sorted({f["group_id"] for f in history["funds"]})
    cohort = (
        "D1-"
        + digest(
            [
                {k: f[k] for k in ("fund_code", "product_family_id", "group_id", "group_evidence")}
                for f in history["funds"]
            ]
        )[:24]
    )
    spec = {
        "run_id": str(uuid4()),
        "protocol": PROTOCOL,
        "cohort_id": weekly_cohort or cohort,
        "groups": groups,
        "train_as_of": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "recipe": RECIPE,
        "data_hash": digest(history),
        "max_main_fits": (1 if weekly_cohort else 5) * len(groups),
        "max_replay_fits": (1 if weekly_cohort else 5) * len(groups),
        "mode": "WEEKLY" if weekly_cohort else "INITIAL",
        "artifact_directory": run_dir.name,
        "history_usage_manifest": history["history_usage_manifest"],
        "retention_days": history["retention_days"],
    }
    write_new(run_dir / "protocol.json", spec)
    write_new(run_dir / "history.json", history)


def verify_inputs(run_dir):
    spec, history = read(run_dir / "protocol.json"), read(run_dir / "history.json")
    if spec["fingerprint"] != fingerprint() or spec["data_hash"] != digest(history):
        raise ValueError("FROZEN_INPUT_OR_CODE_CHANGED")
    return spec, history


def build(run_dir):
    _, history = verify_inputs(run_dir)
    rows = build_samples(history)
    write_new(run_dir / "dataset.json", rows)
    write_new(
        run_dir / "dataset-manifest.json",
        {"hash": digest(rows), "count": len(rows), "kind": "HISTORICAL_RECONSTRUCTION"},
    )


def train(run_dir, replay: bool = False):
    spec, _ = verify_inputs(run_dir)
    rows = read(run_dir / "dataset.json")
    if digest(rows) != read(run_dir / "dataset-manifest.json")["hash"]:
        raise ValueError("DATASET_CHANGED")
    # 排他创建预算凭证，进程崩溃也不得无声多拟合；技术失败保留为SKIPPED。
    write_new(
        run_dir / "fit-budget-reserved.json",
        {"max_fits": spec["max_main_fits"], "reserved_at": datetime.now(ZONE).isoformat()},
    )
    models, reports, predictions = {}, [], []
    quarters = [
        ("2024Q1", "2024-01-01", "2024-04-01"),
        ("2024Q2", "2024-04-01", "2024-07-01"),
        ("2024Q3", "2024-07-01", "2024-10-01"),
        ("2024Q4", "2024-10-01", "2025-01-01"),
    ]
    for group in spec["groups"]:
        population = [r for r in rows if r["group"] == group]
        windows = [] if spec["mode"] == "WEEKLY" else quarters
        for name, start, end in [*windows, ("FINAL", spec["train_as_of"][:10], None)]:
            as_of = (
                datetime.fromisoformat(spec["train_as_of"])
                if name == "FINAL"
                else datetime.combine(date.fromisoformat(start), time(), ZONE)
            )
            selected = select_fit(population, as_of)
            key = group + "-" + name
            try:
                model = fit(selected, group, spec["cohort_id"], as_of)
            except (ValueError, ConvergenceWarning) as error:
                reports.append({"job": key, "status": "SKIPPED", "reason": str(error), "fit_count": len(selected)})
                continue
            models[key] = model
            exam = [r for r in population if end and start <= r["t"] < end]
            scores = [score(model, r["x"]) for r in exam]
            reports.append(
                {
                    "job": key,
                    "status": "SUCCEEDED",
                    "fit_count": len(selected),
                    "fit_hash": model["fit_hash"],
                    "metrics": metrics(exam, scores, model["majority"]),
                }
            )
            predictions.extend(
                {
                    "job": key,
                    "fund_code": r["fund_code"],
                    "t": r["t"],
                    "u": r["u"],
                    "score": s,
                    "y": r["y"],
                    "kind": "REPLAY_DIAGNOSTIC" if replay else "HISTORICAL_RECONSTRUCTION",
                }
                for r, s in zip(exam, scores, strict=True)
            )
    write_new(run_dir / "models.json", models)
    write_new(run_dir / "metrics.json", reports)
    write_new(run_dir / "predictions.json", predictions)
    write_new(
        run_dir / "completion.json",
        {
            "finished_at": datetime.now(ZONE).isoformat(),
            "model_hash": digest(models),
            "metrics_hash": digest(reports),
            "predictions_hash": digest(predictions),
            "successful_fits": len(models),
            "model_released": False,
            "status": "EXPERIMENTAL",
        },
    )


def verify(run_dir):
    spec, _ = verify_inputs(run_dir)
    completion = read(run_dir / "completion.json")
    for file, key in (("models", "model_hash"), ("metrics", "metrics_hash"), ("predictions", "predictions_hash")):
        if digest(read(run_dir / (file + ".json"))) != completion[key]:
            raise ValueError("ARTIFACT_CHANGED")
    if completion["successful_fits"] > spec["max_main_fits"]:
        raise ValueError("FIT_BUDGET_EXCEEDED")
    return completion


def replay(run_dir, out):
    verify(run_dir)
    write_new(run_dir / "replay-reserved.json", {"out": str(out), "at": datetime.now(ZONE).isoformat()})
    out.mkdir(parents=True, exist_ok=False)
    for file in ("protocol.json", "history.json", "dataset.json", "dataset-manifest.json"):
        write_new(out / file, read(run_dir / file))
    train(out, replay=True)
    a, b = read(run_dir / "models.json"), read(out / "models.json")
    if a != b:
        raise ValueError("REPLAY_MODEL_MISMATCH")
    pa, pb = read(run_dir / "predictions.json"), read(out / "predictions.json")
    difference = max((abs(a["score"] - b["score"]) for a, b in zip(pa, pb, strict=True)), default=0)
    evidence = {"model_equal": True, "prediction_count": len(pa), "max_score_diff": difference, "replay_dir": str(out)}
    write_new(run_dir / "replay-verification.json", evidence)
    return evidence


def register(run_dir):
    completion = verify(run_dir)
    proof = read(run_dir / "replay-verification.json")
    if proof["max_score_diff"] > 1e-12:
        raise ValueError("REPLAY_REQUIRED")
    spec = read(run_dir / "protocol.json")
    models = read(run_dir / "models.json")
    MODEL_ROOT.mkdir(exist_ok=True)
    records = []
    with get_engine().begin() as c:
        c.execute(
            text("""INSERT INTO direction_1d_training_run(run_id,cohort_id,train_as_of,spec,spec_hash,status,evidence)
          VALUES(:id,:cohort,:asof,CAST(:spec AS jsonb),:hash,'VERIFIED',CAST(:proof AS jsonb)) ON CONFLICT
          DO NOTHING"""),
            {
                "id": spec["run_id"],
                "cohort": spec["cohort_id"],
                "asof": spec["train_as_of"],
                "spec": canonical(spec),
                "hash": digest(spec),
                "proof": canonical({**completion, **proof}),
            },
        )
        for key, model in models.items():
            if not key.endswith("-FINAL"):
                continue
            mid = str(uuid4())
            payload = canonical(model)
            file_name = mid + ".json"
            write_new(MODEL_ROOT / file_name, model)
            h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            c.execute(
                text("""INSERT INTO
          direction_1d_model(model_id,run_id,cohort_id,group_id,horizon,file_name,content_hash,metadata,trained_at,expires_at)
              VALUES(:id,:run,:cohort,:group,1,:file,:hash,CAST(:meta AS jsonb),:trained,:expires) ON
          CONFLICT(cohort_id,group_id,run_id) DO NOTHING"""),
                {
                    "id": mid,
                    "run": spec["run_id"],
                    "cohort": spec["cohort_id"],
                    "group": model["group_id"],
                    "file": file_name,
                    "hash": h,
                    "meta": payload,
                    "trained": completion["finished_at"],
                    "expires": datetime.fromisoformat(spec["train_as_of"]) + timedelta(days=spec["retention_days"]),
                },
            )
            records.append({"model_id": mid, "group_id": model["group_id"], "hash": h})
    write_new(run_dir / "model-manifest.json", records)
    return records
