"""固定 B0/B1/B2 的折内预处理、监督拟合和配对统计；无业务数据库或服务依赖。"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts import fund_002112_existing_events_features_v1 as f
from scripts import fund_002112_existing_events_sources_v1 as s

RECIPE = {"C": 1.0, "solver": "lbfgs", "tol": 1e-8, "max_iter": 1000, "random_state": 0}
CLASSES = ("DOWN", "FLAT", "UP")


def numeric(rows, model, timing):
    return np.asarray([r["n8"] + (r[timing] if model != "B0" else []) for r in rows], dtype=np.float64)


def fit_preprocessor(rows, members, events, model, timing):
    """只对真实训练行及其20日窗口中的文档拟合尺度、词表和IDF。"""
    prep = {
        "model": model,
        "timing": timing,
        "scaler": StandardScaler().fit(numeric(rows, model, timing)),
        "vectorizers": {},
        "training_document_indices": {},
        "numeric_order": f.N8 + (f.STATS if model != "B0" else []),
    }
    if model == "B2":
        indices = sorted({i for m in members for i, _ in m[timing]})
        for kind in f.TYPES:
            ids = [i for i in indices if events[i]["primary_type"] == kind]
            prep["training_document_indices"][kind] = ids
            vectorizer = None
            if ids:
                vectorizer = TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=1,
                    max_df=1.0,
                    max_features=1024,
                    norm="l2",
                    dtype=np.float64,
                )
                vectorizer.fit([events[i]["text"] for i in ids])
            prep["vectorizers"][kind] = vectorizer
    return prep


def transform(prep, rows, members, events):
    blocks = [sparse.csr_matrix(prep["scaler"].transform(numeric(rows, prep["model"], prep["timing"])))]
    if prep["model"] == "B2":
        for kind in f.TYPES:
            vectorizer = prep["vectorizers"][kind]
            if vectorizer is None:
                continue
            indices = sorted({i for m in members for i, _ in m[prep["timing"]] if events[i]["primary_type"] == kind})
            if not indices:
                blocks.append(sparse.csr_matrix((len(rows), len(vectorizer.vocabulary_))))
                continue
            lookup = {doc: j for j, doc in enumerate(indices)}
            doc_matrix = vectorizer.transform([events[i]["text"] for i in indices])
            rr, cc, vv = [], [], []
            for j, member in enumerate(members):
                window = [(lookup[i], 0.9**age) for i, age in member[prep["timing"]] if i in lookup]
                weight_sum = sum(w for _, w in window)
                for i, w in window:
                    rr.append(j)
                    cc.append(i)
                    vv.append(w / weight_sum)
            weights = sparse.csr_matrix((vv, (rr, cc)), shape=(len(rows), len(indices)))
            blocks.append(weights @ doc_matrix)
    return sparse.hstack(blocks, format="csr")


def train(prep, matrix, labels, reserve, max_iter=1000):
    """所有真实监督fit都必须先执行原子预算预占；预处理无监督fit不计监督次数。"""
    if len(labels) < 60 or len(set(labels)) < 2:
        raise ValueError("INSUFFICIENT_REAL_DATES_OR_CLASSES")
    params = {**RECIPE, "max_iter": max_iter}
    estimator = LogisticRegression(**params)
    attempt = reserve(params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(matrix, labels)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return {"preprocessor": prep, "estimator": estimator}, {
        "attempt": attempt,
        "converged": converged,
        "n_iter": estimator.n_iter_.tolist(),
        "classes": estimator.classes_.tolist(),
        "unlearned_classes": [c for c in CLASSES if c not in estimator.classes_],
        "training_classes": dict(Counter(labels)),
        "matrix_shape": list(matrix.shape),
        "parameters": params,
    }


def predict(model, matrix, rows):
    estimator = model["estimator"]
    probabilities = estimator.predict_proba(matrix)
    decisions = estimator.predict(matrix)
    return [
        {
            "target": r["target"],
            "predicted": str(decisions[i]),
            "probabilities": {
                c: float(probabilities[i, list(estimator.classes_).index(c)]) if c in estimator.classes_ else None
                for c in CLASSES
            },
        }
        for i, r in enumerate(rows)
    ]


def text_evidence(prep, rows, members, events):
    evidence = {}
    for k in f.TYPES:
        column = f.STATS.index(f"{k}_count_20")
        ids = sorted({i for m in members for i, _ in m[prep["timing"]] if events[i]["primary_type"] == k})
        vectorizer = prep["vectorizers"].get(k)
        evidence[k] = {
            "documents": len(ids),
            "training_dates": sum(row[prep["timing"]][column] > 0 for row in rows),
            "statistic_varies": len({row[prep["timing"]][column] for row in rows}) > 1,
            "text_characters": sum(len(events[i]["text"]) for i in ids),
            "vocabulary_size": len(vectorizer.vocabulary_) if vectorizer else 0,
            "vocabulary_hash": s.digest(vectorizer.vocabulary_) if vectorizer else None,
            "document_ids_hash": s.digest([events[i]["event_id"] for i in ids]),
            "publication_start": min((events[i]["published_date"] for i in ids), default=None),
            "publication_end": max((events[i]["published_date"] for i in ids), default=None),
            "training_dates_list": [r["target"] for r in rows if r[prep["timing"]][column] > 0],
            "document_indices": ids,
        }
    return evidence


def score(actual, predicted):
    return {
        "dates": len(actual),
        "correct": sum(a == b for a, b in zip(actual, predicted, strict=True)),
        "accuracy": sum(a == b for a, b in zip(actual, predicted, strict=True)) / len(actual) if actual else None,
        "classes": {
            k: {
                "actual": sum(a == k for a in actual),
                "correct": sum(a == b == k for a, b in zip(actual, predicted, strict=True)),
                "recall": (
                    sum(a == b == k for a, b in zip(actual, predicted, strict=True)) / sum(a == k for a in actual)
                )
                if k in actual
                else None,
            }
            for k in CLASSES
        },
    }


def block_interval(difference, session_indices):
    """固定5交易日块；真实日历有断档即切段，不把不相邻日期接成同一块。"""
    if not difference:
        return None
    blocks, block, previous = [], [], None
    for d, index in zip(difference, session_indices, strict=True):
        if block and (len(block) == 5 or index != previous + 1):
            blocks.append(block)
            block = []
        block.append(d)
        previous = index
    if block:
        blocks.append(block)
    sums = np.asarray([sum(b) for b in blocks])
    sizes = np.asarray([len(b) for b in blocks])
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(blocks), size=(2000, len(blocks)))
    delta = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {
        "low": float(np.quantile(delta, 0.025)),
        "high": float(np.quantile(delta, 0.975)),
        "block_size": 5,
        "blocks": len(blocks),
        "replicates": 2000,
        "seed": 0,
        "missing_sessions_preserved": True,
    }


def paired(rows, actual, base, event):
    a = [int(x == y) for x, y in zip(actual, base, strict=True)]
    b = [int(x == y) for x, y in zip(actual, event, strict=True)]
    delta = [j - i for i, j in zip(a, b, strict=True)]
    cells = Counter((i, j) for i, j in zip(a, b, strict=True))
    return {
        "baseline": score(actual, base),
        "candidate": score(actual, event),
        "net_correct_days": sum(delta),
        "cells": {
            "both_correct": cells[1, 1],
            "baseline_only": cells[1, 0],
            "candidate_only": cells[0, 1],
            "both_wrong": cells[0, 0],
        },
        "accuracy_difference_95pct_block_interval": block_interval(delta, [r["session_index"] for r in rows]),
    }


def compare(rows, actual, predictions, timing, majority):
    groups = {"all": list(range(len(rows)))}
    for i, row in enumerate(rows):
        year = row["target"][:4]
        quarter = year + "-Q" + str((int(row["target"][5:7]) - 1) // 3 + 1)
        groups.setdefault(year, []).append(i)
        groups.setdefault(quarter, []).append(i)
        present = [k for k in f.TYPES if row[timing][f.STATS.index(f"{k}_count_20")] > 0]
        groups.setdefault("with_collected_events" if present else "without_collected_events", []).append(i)
        for k in present:
            groups.setdefault("with_" + k, []).append(i)
    result = {}
    for name, indices in groups.items():
        subset = [rows[i] for i in indices]
        y = [actual[i] for i in indices]
        p = {k: [v[i] for i in indices] for k, v in predictions.items()}
        result[name] = {
            "models": {k: score(y, v) for k, v in p.items()},
            "train_majority_control": score(y, [majority] * len(y)),
            "pairs": {f"{b}-{a}": paired(subset, y, p[a], p[b]) for a, b in [("B0", "B1"), ("B0", "B2"), ("B1", "B2")]},
        }
    return result
