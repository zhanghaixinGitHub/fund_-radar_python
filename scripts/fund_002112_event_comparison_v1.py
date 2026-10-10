"""固定预算的002112离线A/B/C对照；仅写独立研究目录，不注册或采用模型。

执行：python -m scripts.fund_002112_event_comparison_v1 prepare|train|verify
prepare冻结输入与方案，train最多48次拟合，verify独立复算保存模型的概率和时间边界。
失败拟合也计入预算；重启只读回已成功的拟合，不自动重试失败或未完成拟合。
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
from app.services.direction_1d_protocol import RECIPE
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts import fund_002112_event_comparison_data_v1 as data

CLASSES = ("DOWN", "FLAT", "UP")
TIES = ("FLAT", "UP", "DOWN")
MAX_FITS = 48
FINAL_AS_OF = "2026-10-01T08:30:00+08:00"
CODE = [
    Path(__file__),
    Path(data.__file__),
    Path(data.legacy.__file__),
    Path(data.links.__file__),
    data.PY / "app/services/direction_1d_protocol.py",
]
BLOCKS = [
    (f"{y}Q{q}", f"{y}-{1 + (q - 1) * 3:02d}-01", f"{y + (q == 4)}-{1 if q == 4 else 1 + q * 3:02d}-01")
    for y, qs in ((2025, range(1, 5)), (2026, range(1, 4)))
    for q in qs
]


def identifier(row: dict) -> str:
    return row["phase"] + ":" + row["target"]


def journal(entry: dict) -> None:
    """拟合开始前落账；保留每次调用的身份、时间和失败原因，不覆盖历史行。"""
    with (data.ROOT / "fit-ledger.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": datetime.now(data.ZONE).isoformat(), **entry}, ensure_ascii=False) + "\n")


def prepare() -> dict:
    path = data.ROOT / "frozen-protocol.json"
    if path.exists():
        protocol = data.read(path)
        for name, value in protocol["hashes"].items():
            if data.sha(Path(name)) != value:
                raise ValueError("FROZEN_INPUT_OR_CODE_CHANGED: " + name)
        return protocol
    rows = data.jsonl(data.ROOT / "dataset.jsonl")
    protocol = {
        "fund": "002112",
        "frozen_at": datetime.now(data.ZONE).isoformat(),
        "max_fits": MAX_FITS,
        "target": "下一目标交易日单位净值相对前一交易日；原始十进制比较DOWN/FLAT/UP",
        "clock": "北京时间交易日15点前当日、15点起下一交易日；休市归下一交易日；午夜不跳日",
        "phases": data.PHASES,
        "groups": data.GROUPS,
        "blocks": BLOCKS,
        "recipe": RECIPE,
        "baseline_boundary": "A复用现用7特征与分类器参数，在同一002112历史样本重训；不是现用跨基金模型回放",
        "training": "2024年起扩展窗口，每季度初仅用当时标签已成熟的同截点样本；季度内不重训",
        "preprocessing": "训练集内中位数填补，缺失标志只加在扩展特征；训练集内标准化；不加样本权重",
        "text": {
            "group": "C_EVENTS",
            "analyzer": "char",
            "ngram_range": [2, 4],
            "min_df": 2,
            "max_features": 256,
            "fit_scope": "每折训练集；标题和已核验正文引文，每类最多16条，每条500字",
        },
        "minimum_train_rows": 30,
        "minimum_train_classes": 2,
        "absent_class": "无样本的类别不合成、不重标；概率置0。本轮缺FLAT，不能替换线上三态模型",
        "fit_budget": "7季度×2截点×3组=42次，加6个全样本研究模型；失败计数；无参数搜索、反转和重试",
        "final_as_of": FINAL_AS_OF,
        "metrics": [
            "accuracy",
            "DOWN recall",
            "UP recall",
            "DOWN predicted UP",
            "balanced recall",
            "Brier",
            "confidence bins",
            "by quarter",
            "coverage",
            "always UP",
            "train majority",
        ],
        "decision": "所有输出均研究用，不自动采用；方向改进同时检查跌日漏判、涨日表现、季度和覆盖率",
        "limitations": [
            "历史重建，不是未来验证",
            "首见存档不齐",
            "净值可用时间保守且造成非随机缺样",
            "并非完整新闻库",
            "只用已收盘价格，无当日盘中行情",
            "两个截点相关，不合并当独立样本",
            "单基金样本少，文本维度多，有过拟合风险；概率未经独立校准",
        ],
        "hashes": {
            str(p.resolve()): data.sha(p)
            for p in CODE
            + [
                data.SOURCE,
                data.MATERIAL,
                data.ROOT / "dataset.jsonl",
                data.ROOT / "input-evidence.jsonl",
                data.ROOT / "excluded.jsonl",
            ]
        },
        "input_counts": dict(Counter(row["phase"] for row in rows)),
        "adoption_allowed": False,
    }
    data.save(path, protocol)
    print("PROTOCOL_FROZEN", protocol["input_counts"], flush=True)
    return protocol


def raw_matrix(rows: list[dict], group: str) -> np.ndarray:
    return np.asarray(
        [[np.nan if value is None else value for value in row["groups"][group]] for row in rows], dtype=float
    )


def numerical(raw: np.ndarray, medians: np.ndarray, group: str) -> np.ndarray:
    """原件None仍保存在输入快照；模型计算时填值并保留扩展字段缺失标志。"""
    result = np.where(np.isnan(raw), medians, raw)
    return result if group == "A_NAV" else np.hstack([result, np.isnan(raw[:, 7:]).astype(float)])


def fit_bundle(rows: list[dict], group: str, as_of: str) -> dict:
    labels = [row["label"] for row in rows]
    if len(rows) < 30 or len(set(labels)) < 2:
        raise ValueError("INSUFFICIENT_TRAINING_DATA")
    assert all(data.moment(row["mature_at"]) <= data.moment(as_of) for row in rows)
    raw = raw_matrix(rows, group)
    medians = np.asarray([float(np.median(col[~np.isnan(col)])) if np.any(~np.isnan(col)) else 0 for col in raw.T])
    scaler = StandardScaler()
    matrix = scaler.fit_transform(numerical(raw, medians, group))
    vectorizer = None
    if group == "C_EVENTS":
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=256)
        # 若原件无足够正文，直接记为失败，不能静默把C当成B后宣称文本已参与。
        text_matrix = vectorizer.fit_transform([row["text"] for row in rows]).toarray()
        matrix = np.hstack([matrix, text_matrix])
    classifier = LogisticRegression(**RECIPE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(matrix, labels)
    counts = Counter(labels)
    majority = max(TIES, key=lambda label: counts[label])
    return {
        "group": group,
        "medians": medians,
        "scaler": scaler,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "train_ids": [identifier(row) for row in rows],
        "train_as_of": as_of,
        "majority": majority,
        "class_counts": dict(counts),
        "matrix_shape": matrix.shape,
        "feature_names": data.NAMES[group]
        + ([] if group == "A_NAV" else [name + "__missing" for name in data.NAMES[group][7:]])
        + ([] if vectorizer is None else ["text:" + t for t in vectorizer.get_feature_names_out()]),
        "adoption_allowed": False,
    }


def transform(bundle: dict, rows: list[dict]) -> np.ndarray:
    raw = raw_matrix(rows, bundle["group"])
    result = bundle["scaler"].transform(numerical(raw, bundle["medians"], bundle["group"]))
    if bundle["vectorizer"] is not None:
        result = np.hstack([result, bundle["vectorizer"].transform([row["text"] for row in rows]).toarray()])
    return result


def probabilities(bundle: dict, matrix: np.ndarray, independent=False) -> np.ndarray:
    classifier = bundle["classifier"]
    if independent:
        logits = matrix @ classifier.coef_.T + classifier.intercept_
        raw = (
            np.c_[1 - expit(logits[:, 0]), expit(logits[:, 0])]
            if len(classifier.classes_) == 2
            else softmax(logits, axis=1)
        )
    else:
        raw = classifier.predict_proba(matrix)
    output = np.zeros((len(matrix), 3))
    for i, label in enumerate(classifier.classes_):
        output[:, CLASSES.index(label)] = raw[:, i]
    return output


def direction(probability) -> str:
    return max(TIES, key=lambda label: probability[CLASSES.index(label)])


def planned(rows: list[dict]):
    """所有三组共用逐折训练/验证日期；多个截点分别报告，不随机拆散目标日。"""
    for phase in data.PHASES:
        available = [row for row in rows if row["phase"] == phase]
        for name, start, end in BLOCKS:
            as_of = data.cutoff_for(start, phase)
            train_rows = [
                row
                for row in available
                if row["target"] < start and data.moment(row["mature_at"]) <= data.moment(as_of)
            ]
            test_rows = [row for row in available if start <= row["target"] < end]
            for group in data.GROUPS:
                yield f"{phase}-{name}-{group}", group, train_rows, test_rows, as_of, name
        train_rows = [row for row in available if data.moment(row["mature_at"]) <= data.moment(FINAL_AS_OF)]
        for group in data.GROUPS:
            yield f"{phase}-FINAL-{group}", group, train_rows, [], FINAL_AS_OF, "FINAL"


def train() -> None:
    prepare()
    rows = data.jsonl(data.ROOT / "dataset.jsonl")
    ledger_path = data.ROOT / "fit-ledger.jsonl"
    ledger = data.jsonl(ledger_path) if ledger_path.exists() else []
    for fit_id, group, training, testing, as_of, block in planned(rows):
        entries = [entry for entry in ledger if entry["fit_id"] == fit_id]
        if entries:
            if entries[-1]["status"] != "SUCCEEDED":
                raise RuntimeError("NO_AUTOMATIC_RETRY: " + fit_id)
            continue
        if sum(entry["status"] == "RESERVED" for entry in ledger) >= MAX_FITS:
            raise RuntimeError("FIT_BUDGET_EXHAUSTED")
        reservation = {
            "fit_id": fit_id,
            "status": "RESERVED",
            "group": group,
            "block": block,
            "as_of": as_of,
            "train_ids": [identifier(row) for row in training],
            "test_ids": [identifier(row) for row in testing],
        }
        journal(reservation)
        ledger.append(reservation)
        try:
            bundle = fit_bundle(training, group, as_of)
            model_path = data.ROOT / "models" / (fit_id + ".joblib")
            model_path.parent.mkdir(exist_ok=True)
            if model_path.exists():
                raise FileExistsError(model_path)
            joblib.dump(bundle, model_path)
            predictions = []
            if testing:
                prob = probabilities(bundle, transform(bundle, testing))
                for row, p in zip(testing, prob, strict=True):
                    predictions.append(
                        {
                            "id": identifier(row),
                            "target": row["target"],
                            "phase": row["phase"],
                            "cutoff": row["cutoff"],
                            "input_hash": row["input_hash"],
                            "fit_id": fit_id,
                            "group": group,
                            "block": block,
                            "actual": row["label"],
                            "predicted": direction(p),
                            "probabilities": dict(zip(CLASSES, p.tolist(), strict=True)),
                            "majority": bundle["majority"],
                            "return": row["return"],
                        }
                    )
            data.save_lines(data.ROOT / "predictions" / (fit_id + ".jsonl"), predictions)
            entry = {
                "fit_id": fit_id,
                "status": "SUCCEEDED",
                "model_hash": data.sha(model_path),
                "train_count": len(training),
                "test_count": len(testing),
                "dimensions": bundle["matrix_shape"][1],
                "classes": bundle["class_counts"],
            }
            journal(entry)
            ledger.append(entry)
            print(
                "FIT",
                len([e for e in ledger if e["status"] == "RESERVED"]),
                fit_id,
                "train",
                len(training),
                "test",
                len(testing),
                flush=True,
            )
        except Exception as exc:
            journal({"fit_id": fit_id, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)[:400]})
            raise


def metrics(predictions: list[dict]) -> dict:
    n = len(predictions)
    counts = Counter(row["actual"] for row in predictions)
    recall = {
        label: (
            sum(row["actual"] == label == row["predicted"] for row in predictions) / counts[label]
            if counts[label]
            else None
        )
        for label in CLASSES
    }
    correct = sum(row["actual"] == row["predicted"] for row in predictions)
    bins = []
    for low, high in ((0.0, 0.6), (0.6, 0.8), (0.8, 1.0000001)):
        subset = [row for row in predictions if low <= max(row["probabilities"].values()) < high]
        bins.append(
            {
                "range": [low, min(high, 1.0)],
                "n": len(subset),
                "accuracy": sum(r["actual"] == r["predicted"] for r in subset) / len(subset) if subset else None,
                "mean_confidence": np.mean([max(r["probabilities"].values()) for r in subset]).item()
                if subset
                else None,
            }
        )
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "class_counts": dict(counts),
        "recall": recall,
        "balanced_recall": np.mean([r for r in recall.values() if r is not None]).item() if n else None,
        "down_predicted_up": sum(r["actual"] == "DOWN" and r["predicted"] == "UP" for r in predictions),
        "brier": sum(
            sum((r["probabilities"][label] - (r["actual"] == label)) ** 2 for label in CLASSES) for r in predictions
        )
        / n
        if n
        else None,
        "always_up_correct": counts["UP"],
        "train_majority_correct": sum(r["actual"] == r["majority"] for r in predictions),
        "confidence_bins": bins,
    }


def independent_nav(values: list) -> list:
    """以数组运算独立复算7项净值字段，不调用原features函数。"""
    nav = np.asarray(values, dtype=float)
    returns = nav[-20:] / nav[-21:-1] - 1
    decline = 0
    for falling in (np.diff(nav) < 0)[::-1]:
        if not falling:
            break
        decline += 1
    last = nav[1:]
    return [
        nav[-1] / nav[-6] - 1,
        nav[-1] / nav[-21] - 1,
        nav[-1] / nav[0] - 1,
        float(returns.std()),
        float(np.min(last / np.maximum.accumulate(last) - 1)),
        float((nav[-1] - last.min()) / np.ptp(last)),
        float(decline),
    ]


def verify() -> dict:
    prepare()
    rows = data.jsonl(data.ROOT / "dataset.jsonl")
    by_id = {identifier(row): row for row in rows}
    assert len(by_id) == len(rows)
    source = data.read(data.SOURCE)
    proofs = data.jsonl(data.ROOT / "input-evidence.jsonl")
    for proof in proofs:
        row = by_id[identifier(proof)]
        cutoff = data.moment(row["cutoff"])
        assert data.prediction_target(source["sessions"], row["cutoff"]) == (row["base"], row["target"])
        assert row["input_hash"] == data.digest([row["groups"], row["text"], row["cutoff"]])
        assert all(data.moment(source["nav"][day]["available_at"]) <= cutoff for day in proof["nav"]["dates"])
        assert np.max(np.abs(np.asarray(independent_nav(proof["nav"]["values"])) - row["groups"]["A_NAV"])) < 1e-12
        base, target = (Decimal(source["nav"][day]["unit_nav"]) for day in (row["base"], row["target"]))
        assert row["label"] == ("UP" if target > base else "DOWN" if target < base else "FLAT")
        if proof["market"]["report_available_at"]:
            assert data.moment(proof["market"]["report_available_at"]) <= cutoff
        assert max(proof["market"]["price_days"]) <= row["base"]
        for event in proof["events"]["events"]:
            assert data.moment(event["available_at"]) <= cutoff
            for link in event["links"]:
                assert data.moment(link["report_available_at"]) <= cutoff
                if link.get("profile"):
                    assert data.moment(link["profile"]["available_at"]) <= min(
                        cutoff, data.moment(event["available_at"])
                    )
        for event in proof["events"]["numeric_evidence"]:
            assert data.moment(event["available_at"]) <= cutoff
            assert data.moment(event["report_available_at"]) <= cutoff
    ledger = data.jsonl(data.ROOT / "fit-ledger.jsonl")
    assert sum(entry["status"] == "RESERVED" for entry in ledger) == MAX_FITS
    assert sum(entry["status"] == "SUCCEEDED" for entry in ledger) == MAX_FITS
    predictions, max_diff, dimensions = [], 0.0, {}
    for fit_id, _group, training, testing, as_of, _ in planned(rows):
        model_path = data.ROOT / "models" / (fit_id + ".joblib")
        success = next(e for e in ledger if e["fit_id"] == fit_id and e["status"] == "SUCCEEDED")
        assert data.sha(model_path) == success["model_hash"]
        bundle = joblib.load(model_path)
        assert bundle["train_ids"] == [identifier(r) for r in training]
        assert not set(bundle["train_ids"]) & {identifier(r) for r in testing}
        assert all(data.moment(r["mature_at"]) <= data.moment(as_of) for r in training)
        assert all(data.moment(as_of) <= data.moment(r["cutoff"]) for r in testing)
        dimensions[fit_id] = success["dimensions"]
        saved = data.jsonl(data.ROOT / "predictions" / (fit_id + ".jsonl"))
        assert [r["id"] for r in saved] == [identifier(r) for r in testing]
        if testing:
            restored = probabilities(bundle, transform(bundle, testing), independent=True)
            saved_probs = np.asarray([[r["probabilities"][label] for label in CLASSES] for r in saved])
            max_diff = max(max_diff, float(np.max(np.abs(restored - saved_probs))))
            assert max_diff < 1e-12
            assert all(direction(p) == r["predicted"] for p, r in zip(restored, saved, strict=True))
        predictions += saved
    summary = {
        "adoption_allowed": False,
        "historical_only": True,
        "fit_count": MAX_FITS,
        "input_rows": len(rows),
        "prediction_rows": len(predictions),
        "reload_max_probability_diff": max_diff,
        "model_dimensions": dimensions,
        "metrics": {},
        "coverage": {},
        "paired": {},
    }
    excluded = data.jsonl(data.ROOT / "excluded.jsonl")
    for phase in data.PHASES:
        phase_rows = [r for r in rows if r["phase"] == phase and r["target"] >= "2025-01-01"]
        phase_excluded = [r for r in excluded if r["phase"] == phase and r["target"] >= "2025-01-01"]
        summary["coverage"][phase] = {
            "predicted": len(phase_rows),
            "eligible_calendar_dates": len(phase_rows) + len(phase_excluded),
            "weekday_counts": dict(Counter(datetime.fromisoformat(r["target"]).strftime("%A") for r in phase_rows)),
        }
        summary["metrics"][phase] = {}
        keyed = {}
        for group in data.GROUPS:
            subset = [r for r in predictions if r["phase"] == phase and r["group"] == group]
            summary["metrics"][phase][group] = {
                "overall": metrics(subset),
                "quarters": {block: metrics([r for r in subset if r["block"] == block]) for block, _, _ in BLOCKS},
            }
            keyed[group] = {r["id"]: r for r in subset}
        assert set(keyed["A_NAV"]) == set(keyed["B_MARKET"]) == set(keyed["C_EVENTS"])
        summary["paired"][phase] = {}
        for group in ("B_MARKET", "C_EVENTS"):
            pairs = [(r, keyed["A_NAV"][key]) for key, r in keyed[group].items()]
            summary["paired"][phase][group] = {
                "candidate_right_A_wrong": sum(
                    r["predicted"] == r["actual"] and a["predicted"] != a["actual"] for r, a in pairs
                ),
                "A_right_candidate_wrong": sum(
                    r["predicted"] != r["actual"] and a["predicted"] == a["actual"] for r, a in pairs
                ),
            }
    data.save_lines(data.ROOT / "all-predictions.jsonl", predictions)
    data.save(data.ROOT / "summary.json", summary)
    print(
        json.dumps(
            {
                "fit_count": MAX_FITS,
                "max_diff": max_diff,
                "coverage": summary["coverage"],
                "results": {
                    phase: {g: m["overall"] for g, m in groups.items()} for phase, groups in summary["metrics"].items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "verify"))
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
