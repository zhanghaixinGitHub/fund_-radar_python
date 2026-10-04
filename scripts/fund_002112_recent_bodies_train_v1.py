"""近年正文的固定六组滚动比较；标签成熟后才训练，拟合逐笔记账，结果不发布。"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import warnings
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts import fund_002112_recent_bodies_v1 as source

io, ROOT = source.io, source.ROOT
CLASSES = ("DOWN", "FLAT", "UP")


def labels(rows, nav):
    """比较目标日与基准日单位净值，不改变旧实验下一交易日涨跌定义。"""
    out = []
    for row in rows:
        delta = Decimal(nav[row["target"]]["unit_nav"]) - Decimal(nav[row["base"]]["unit_nav"])
        out.append("UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT")
    return out


def freeze(root):
    """固定代码、输入和日历后才打开监督训练入口；变动会拒绝继续原轮次。"""
    if (root / "freeze.json").exists():
        verify_freeze(root)
        return
    assert (root / "prepared.json").exists()
    paths = list(root.glob("*.json")) + list(root.glob("*.jsonl")) + list(root.glob("*.npz"))
    paths += [Path(__file__).resolve(), Path(source.__file__).resolve(),
              io.PY / "scripts/test_fund_002112_recent_bodies_v1.py", Path(io.__file__).resolve(),
              Path(source.holdings.__file__).resolve()]
    # 已冻结的数值对照和净值仅复制；旧模型和旧预测不重跑、不覆盖。
    for name in ("nav-through-2025.json", "development-predictions/RF_D4__EVERY20.jsonl",
                 "development-predictions/LR_C1__EVERY20.jsonl"):
        old = source.PREVIOUS / name
        dest = root / "references" / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert io.sha(dest) == io.sha(old)
        else:
            with dest.open("xb") as handle:
                handle.write(old.read_bytes())
        paths.extend([old, dest])
    packages = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-learn", "joblib", "pypdfium2")}
    for path in paths.copy():
        if path.suffix == ".py":
            copied = root / "code-snapshot" / path.name
            copied.parent.mkdir(parents=True, exist_ok=True)
            with copied.open("xb") as handle:
                handle.write(path.read_bytes())
            paths.append(copied)
    io.save(root / "freeze.json", {"at": io.now(), "python": sys.version, "packages": packages,
                                   "files": {str(p.resolve()): io.sha(p) for p in paths}})


def verify_freeze(root):
    for path, digest in io.read(root / "freeze.json")["files"].items():
        if io.sha(path) != digest:
            raise ValueError("FROZEN_FILE_CHANGED:" + path)


def numeric_matrix(rows, candidate):
    result = []
    for row in rows:
        values = list(row["numeric"])
        if candidate not in ("RF_H", "LR_H"):
            values += row["counts"]
        if candidate == "RF_H_FACTS":
            values += row["facts"]
        result.append(values)
    return np.array(result, dtype=float)


def text_levels(candidate):
    return ["title", "body"] if candidate == "LR_H_BODY" else ["title"] if candidate == "LR_H_TITLE" else []


def transform(rows, matrices, indices, candidate, fitted=None):
    """数值中位数、缩放和文字IDF均只对训练行拟合；同一对象用于后续预测。"""
    values = numeric_matrix([rows[i] for i in indices], candidate)
    training = fitted is None
    if training:
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        values = imputer.fit_transform(values)
        scaler = StandardScaler().fit(values)
        fitted = {"imputer": imputer, "scaler": scaler, "idf": {}}
    else:
        values = fitted["imputer"].transform(values)
    blocks = [sparse.csr_matrix(fitted["scaler"].transform(values))]
    for level in text_levels(candidate):
        raw = matrices[level][indices]
        if training:
            fitted["idf"][level] = TfidfTransformer().fit(raw)
        blocks.append(fitted["idf"][level].transform(raw))
    return sparse.hstack(blocks, format="csr"), fitted


def fit_one(root, candidate, update_id, rows, matrices, nav, replay=False):
    """每次真实监督fit前记账；失败占用预算且保留，禁止隐式重试或换目录续搜。"""
    rid = candidate + "__" + update_id + ("__replay" if replay else "")
    folder = root / "runs" / rid
    if (folder / "complete.json").exists():
        return io.read(folder / "complete.json")
    ledger = io.lines(root / "fit-ledger.jsonl")
    if any(r["id"] == rid for r in ledger):
        raise ValueError("PREVIOUS_ATTEMPT_NOT_COMPLETED_NO_RETRY:" + rid)
    plan = io.read(root / "protocol.json")
    if len(ledger) >= plan["max_actual_supervised_fits"]:
        raise ValueError("FIXED_FIT_BUDGET_REACHED")
    cal = io.read(root / "calendar.json")
    update = cal["updates"][update_id]
    train = update["training"]
    evaluate = cal["development_indices"][update["start"]:update["start"] + 20]
    eligible = [i for i, r in enumerate(rows) if r["target"] < rows[evaluate[0]]["target"]
                and r["label_mature_at"] < update["cutoff_exclusive"]]
    assert eligible == train
    y = labels([rows[i] for i in train], nav)
    x, preprocessing = transform(rows, matrices, train, candidate)
    xe, _ = transform(rows, matrices, evaluate, candidate, preprocessing)
    model = (RandomForestClassifier(**plan["rf"], n_jobs=2) if candidate.startswith("RF")
             else LogisticRegression(**plan["lr"]))
    io.append(root / "fit-ledger.jsonl", {
        "at": io.now(), "id": rid, "candidate": candidate, "update": update_id, "replay": replay,
        "train_rows": train, "evaluation_rows": evaluate, "cutoff": update["cutoff_exclusive"],
        "max_label_mature_at": max(rows[i]["label_mature_at"] for i in train),
        "matrix_shape": list(x.shape), "actual_supervised_fits": 1,
    })
    try:
        with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise ValueError("MODEL_DID_NOT_CONVERGE")
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"model": model, "preprocessing": preprocessing, "candidate": candidate}
        joblib.dump(payload, folder / "model.joblib")
        with threadpool_limits(limits=2):
            probabilities = model.predict_proba(xe)
        aligned = np.zeros((len(evaluate), 3))
        for j, label in enumerate(model.classes_):
            aligned[:, CLASSES.index(label)] = probabilities[:, j]
        predictions = [{"target": rows[i]["target"], "as_of": rows[i]["as_of"], "model_id": rid,
                        "predicted": CLASSES[int(np.argmax(aligned[k]))], "probabilities": aligned[k].tolist()}
                       for k, i in enumerate(evaluate)]
        io.save_lines(folder / "predictions.jsonl", predictions)
        complete = {"at": io.now(), "id": rid, "candidate": candidate, "update": update_id,
                    "model_path": str(folder / "model.joblib"),
                    "model_sha256": io.sha(folder / "model.joblib"),
                    "predictions_sha256": io.sha(folder / "predictions.jsonl"),
                    "training_count": len(train), "evaluation_count": len(evaluate)}
        io.save(folder / "complete.json", complete)
        return complete
    except Exception as exc:
        io.save(folder / "failed.json", {"at": io.now(), "type": type(exc).__name__, "message": str(exc)})
        raise


def score(rows, nav, predictions):
    truth = labels(rows, nav)
    assert [r["target"] for r in rows] == [p["target"] for p in predictions]
    correct = sum(a == p["predicted"] for a, p in zip(truth, predictions, strict=True))
    return {"total": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "by_direction": {label: {"actual": truth.count(label),
                                    "correct": sum(a == label and p["predicted"] == label
                                                   for a, p in zip(truth, predictions, strict=True)),
                                    "predicted": sum(p["predicted"] == label for p in predictions)}
                             for label in CLASSES}}


def paired(rows, nav, first, second):
    truth = labels(rows, nav)
    improved = worsened = changed = 0
    for y, a, b in zip(truth, first, second, strict=True):
        improved += a["predicted"] != y and b["predicted"] == y
        worsened += a["predicted"] == y and b["predicted"] != y
        changed += a["predicted"] != b["predicted"]
    return {"changed": changed, "wrong_to_right": improved, "right_to_wrong": worsened,
            "net_additional_correct": improved - worsened}


def verify_models(root, rows, matrices):
    """独立读取保存模型重算全部预测；验证预处理只拟合训练行及每个标签的成熟边界。"""
    ledger = io.lines(root / "fit-ledger.jsonl")
    checks = 0
    for record in ledger:
        folder = root / "runs" / record["id"]
        done = io.read(folder / "complete.json")
        assert io.sha(folder / "model.joblib") == done["model_sha256"]
        assert io.sha(folder / "predictions.jsonl") == done["predictions_sha256"]
        saved = joblib.load(folder / "model.joblib")
        assert all(rows[i]["label_mature_at"] < record["cutoff"] for i in record["train_rows"])
        _, expected = transform(rows, matrices, record["train_rows"], record["candidate"])
        np.testing.assert_array_equal(expected["imputer"].statistics_, saved["preprocessing"]["imputer"].statistics_)
        np.testing.assert_array_equal(expected["scaler"].mean_, saved["preprocessing"]["scaler"].mean_)
        for level in text_levels(record["candidate"]):
            np.testing.assert_array_equal(expected["idf"][level].idf_, saved["preprocessing"]["idf"][level].idf_)
        xe, _ = transform(rows, matrices, record["evaluation_rows"], record["candidate"], saved["preprocessing"])
        with threadpool_limits(limits=2):
            probability = saved["model"].predict_proba(xe)
        aligned = np.zeros((len(record["evaluation_rows"]), 3))
        for j, label in enumerate(saved["model"].classes_):
            aligned[:, CLASSES.index(label)] = probability[:, j]
        predictions = io.lines(folder / "predictions.jsonl")
        np.testing.assert_allclose([p["probabilities"] for p in predictions], aligned, atol=1e-12, rtol=0)
        assert [p["predicted"] for p in predictions] == [CLASSES[int(np.argmax(p))] for p in aligned]
        checks += len(predictions)
    return {"models_read_back": len(ledger), "probability_rows_recomputed": checks,
            "training_only_preprocessing": True, "label_maturity_verified": True}


def run(root=ROOT):
    freeze(root)
    if (root / "completion.json").exists():
        return io.read(root / "completion.json")
    rows = io.lines(root / "inputs.jsonl")
    matrices = {k: sparse.load_npz(root / (k + "-matrix.npz")) for k in ("title", "body")}
    nav = io.read(root / "references/nav-through-2025.json")
    assert max(nav) <= "2025-12-31" and max(r["target"] for r in rows) <= "2025-12-31"
    cal = io.read(root / "calendar.json")
    plan = io.read(root / "protocol.json")
    for candidate in plan["candidates"]:
        verify_freeze(root)
        for update in cal["updates"]:
            fit_one(root, candidate, update, rows, matrices, nav)
        print(f"完成 {candidate}：13次拟合", flush=True)
    dev = [rows[i] for i in cal["development_indices"]]
    predictions, scores = {}, {}
    for candidate in plan["candidates"]:
        predictions[candidate] = [p for update in cal["updates"]
                                 for p in io.lines(root / "runs" / (candidate + "__" + update) / "predictions.jsonl")]
        assert len(predictions[candidate]) == 243
        scores[candidate] = score(dev, nav, predictions[candidate])
        io.save_lines(root / "development-predictions" / (candidate + ".jsonl"), predictions[candidate])
    for name in ("RF_D4", "LR_C1"):
        predictions["REFERENCE_" + name] = io.lines(root / "references" / (name + "__EVERY20.jsonl"))
        scores["REFERENCE_" + name] = score(dev, nav, predictions["REFERENCE_" + name])
    scores["ALWAYS_UP"] = score(dev, nav, [{"target": r["target"], "predicted": "UP"} for r in dev])
    selected = sorted(plan["candidates"], key=lambda c: (-scores[c]["correct"], c))[0]
    io.save(root / "selection.json", {"at": io.now(), "candidate": selected, "scores": scores,
                                       "development_only": True, "adoption": False})
    fit_one(root, selected, "U240", rows, matrices, nav, replay=True)
    original = io.lines(root / "runs" / (selected + "__U240") / "predictions.jsonl")
    repeated = io.lines(root / "runs" / (selected + "__U240__replay") / "predictions.jsonl")
    np.testing.assert_allclose([p["probabilities"] for p in original],
                               [p["probabilities"] for p in repeated], atol=1e-12, rtol=0)
    assert [p["predicted"] for p in original] == [p["predicted"] for p in repeated]
    verification = verify_models(root, rows, matrices)
    verify_freeze(root)
    protection = io.read(root / "protection-before.json")
    changed = [path for path, digest in protection["files"].items() if io.sha(path) != digest]
    verification["protected_existing_files"] = len(protection["files"])
    verification["changed_existing_files"] = changed
    assert not changed, changed
    comparisons = {}
    for a, b in (("REFERENCE_RF_D4", "RF_H"), ("RF_H", "RF_H_COUNTS"),
                 ("RF_H_COUNTS", "RF_H_FACTS"), ("LR_H", "LR_H_TITLE"), ("LR_H_TITLE", "LR_H_BODY")):
        comparisons[a + " -> " + b] = paired(dev, nav, predictions[a], predictions[b])
    result = {"at": io.now(), "scores": scores, "comparisons": comparisons, "selected": selected,
              "fit_count": len(io.lines(root / "fit-ledger.jsonl")), "verification": verification,
              "coverage": io.read(root / "prepared.json"), "new_2026_scores": None, "adoption": False}
    io.save(root / "completion.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    with io.writer_lock(args.root), io.offline_guard():
        print(io.canonical(run(args.root)), flush=True)
