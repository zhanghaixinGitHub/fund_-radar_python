"""存量量价固定15候选，复用已核验训练实现；源文件不改，预算独立且不触及2026。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

from scripts import fund_002112_development_only_search_v1 as engine
from scripts import fund_002112_price_sources_v1 as source

io, ROOT = source.io, source.ROOT
GROUPS = {
    "NE": ["NE"],
    "NNE": ["N", "NE"],
    "MME": ["M", "ME"],
    "NMME": ["N", "M", "ME"],
    "ALL_PRICE": ["N", "NE", "M", "ME"],
}
RECIPES = ("LR_C1", "HGB_D2", "RF_D4")


def configure():
    """仅在当前进程显式绑定新根目录/输入组，不编辑任何旧脚本或旧批次。"""
    engine.ROOT = ROOT
    engine.GROUPS = GROUPS


def prepare():
    if (ROOT / "protocol.json").exists():
        return io.read(ROOT / "protocol.json")
    admission = io.read(ROOT / "source-audit.json")
    if admission["development_rows"] != 243 or not admission["all_new_features_finite"]:
        raise ValueError("SOURCE_COVERAGE_NOT_READY")
    candidates = [
        {"id": f"{g}__{r}", "group": g, "recipe": r, "window": None, "half_life": None} for g in GROUPS for r in RECIPES
    ]
    splits = io.read(source.timing.ROOT / "splits.json")
    io.save(ROOT / "splits.json", splits)
    plan = {
        "at": io.now(),
        "version": "EXISTING_PRICE_VOLUME_DEV_V1",
        "candidates": candidates,
        "candidate_count": 15,
        "max_actual_fits": 80,
        "input_groups": GROUPS,
        "recipes": RECIPES,
        "horizon": "D08:00_TO_NEXT_TRADING_DATE_U",
        "label_date_max": "2025-12-31",
        "outer_development_dates": 243,
        "no_2026_evaluation": True,
        "historical_2025_previously_seen": True,
        "hypotheses": {
            "NE": "Known NAV short returns and short/long/downside volatility",
            "MME": "Market20d trend/5d20d volatility and amount/volume relative prior20 mean",
            "NNE_NMME_ALL": "Independent groups and incremental information against fixedN8",
        },
        "methods": "FixedLR_C1,HGB_D2,RF_D4; seed0; no tuning on outer development answers",
        "preprocessing": "Training-only median+missing indicator+standardization; no threshold calibration",
        "selection": "DEV_CORRECT_DESC_MIN_FOLD_ACCURACY_DESC_FEATURES_ASC_ID_ASC",
        "baseline": "Reuse priorN8 development predictions,ALWAYS_UP,TRAINING_FOLD_MAJORITY",
        "replay": "WinnerV3 once, counts in80",
        "stop": "Complete15or80actualattempts; no grid expansion",
        "excluded": ["H", "F", "COUNTS", "GLOBAL", "FULL_MARKET_BREADTH"],
        "assumptions": ["CLOSE_NEXT_TRADING_DATE0800_RECONSTRUCTION", "HISTORICAL_FIRST_SEEN_UNPROVEN"],
        "deployment": False,
    }
    io.save(ROOT / "protocol.json", plan)
    oldfreeze = io.read(source.timing.ROOT / "freeze.json")
    sources = {Path(__file__), Path(source.__file__), Path(engine.__file__)}
    sources.update(Path(p) for p in oldfreeze["files"] if str(p).endswith(".py"))
    copied = {}
    for p in sorted(sources):
        dest = ROOT / "code-snapshot" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        copied[str(p)] = {"sha256": io.sha(p), "copy": str(dest)}
    env = io.read(source.timing.ROOT / "environment.json")
    assert env["python"] == sys.version
    assert all(importlib.metadata.version(p) == v for p, v in env["packages"].items())
    io.save(
        ROOT / "environment.json",
        {"at": io.now(), "python": env["python"], "packages": env["packages"], "source_copies": copied},
    )
    artifacts = [
        ROOT / n
        for n in (
            "inputs.jsonl",
            "nav-through-2025.json",
            "market-values-through-2025.json",
            "feature-lineage.jsonl",
            "source-manifest.json",
            "source-audit.json",
            "protocol.json",
            "splits.json",
            "environment.json",
            "breadth-admission.json",
            "corpus-selection-audit.json",
            "report-cross-evidence.json",
        )
    ]
    baseline_sources = {}
    for fold in engine.FOLDS:
        record = source.timing.ROOT / "runs" / f"N__WALL__DNONE__{fold}" / "complete.json"
        run = io.read(record)
        for p in (record, Path(run["model_path"]), Path(run["predictions_path"])):
            baseline_sources[str(p)] = io.sha(p)
            artifacts.append(p)
    io.save(ROOT / "baseline-source-freeze.json", baseline_sources)
    artifacts.append(ROOT / "baseline-source-freeze.json")
    io.save(ROOT / "freeze.json", {"at": io.now(), "files": {str(p): io.sha(p) for p in artifacts + list(sources)}})
    return plan


def bounded_fit(candidate, fold, replay=False):
    """训练入口外层强制80次上限；底层缓存返回不新增真实计次。"""
    rid = candidate["id"] + "__" + fold + ("__replay" if replay else "")
    if not (ROOT / "runs" / rid / "complete.json").exists() and len(io.lines(ROOT / "fit-ledger.jsonl")) >= 80:
        raise ValueError("PRICE_STAGE_80_ATTEMPTS_REACHED")
    return engine.fit(candidate, fold, replay=replay)


def run():
    configure()
    engine.access_guard()
    engine.verify_freeze()
    if (ROOT / "completion.json").exists():
        return io.read(ROOT / "completion.json")
    plan = io.read(ROOT / "protocol.json")
    rows = io.lines(ROOT / "inputs.jsonl")
    nav = io.read(ROOT / "nav-through-2025.json")
    splits = io.read(ROOT / "splits.json")
    dev = [r for r in rows if r["target"].startswith("2025")]
    results = []
    for c in plan["candidates"]:
        joined = []
        scores = []
        for fold in engine.FOLDS:
            fitted = bounded_fit(c, fold)
            p = engine.predicted(fitted)
            joined.extend(p)
            scores.append(engine.score([rows[i] for i in splits[fold]["evaluation"]], nav, p))
        result = {
            "candidate": c,
            "score": engine.score(dev, nav, joined),
            "fold_scores": scores,
            "min_fold_accuracy": min(v["accuracy"] for v in scores),
            "features": fitted["features"],
        }
        io.save_lines(ROOT / "development-predictions" / (c["id"] + ".jsonl"), joined)
        results.append(result)
        print(f"DEV {len(results)}/15 {c['id']}: {result['score']['correct']}/243", flush=True)
    ranking = sorted(
        results, key=lambda v: (-v["score"]["correct"], -v["min_fold_accuracy"], v["features"], v["candidate"]["id"])
    )
    io.save(ROOT / "candidate-results.json", results)
    io.save(
        ROOT / "selection.json", {"at": io.now(), "winner": ranking[0], "ranking": ranking, "development_only": True}
    )
    baselines = {}
    for fold, split in splits.items():
        er = [rows[i] for i in split["evaluation"]]
        tr = [rows[i] for i in split["training"]]
        old = io.read(source.timing.ROOT / "runs" / f"N__WALL__DNONE__{fold}" / "complete.json")
        saved = engine.predicted(old)
        tr_y = source.timing.labels(tr, nav)
        actual = source.timing.labels(er, nav)
        majority = max(engine.kernel.CLASSES, key=lambda c: (tr_y.count(c), c))
        baselines[fold] = {
            "N": engine.score(er, nav, saved),
            "always_up": engine.kernel.impact.stats.score(actual, ["UP"] * len(er)),
            "majority": engine.kernel.impact.stats.score(actual, [majority] * len(er)),
            "N_reference": {"path": old["predictions_path"], "sha256": old["predictions_sha256"]},
        }
    io.save(ROOT / "baselines.json", baselines)
    winner = ranking[0]["candidate"]
    repeat = bounded_fit(winner, "V3", replay=True)
    original = bounded_fit(winner, "V3")
    assert engine.predicted(repeat) == engine.predicted(original)
    # 树与线性方法通用：固定评估矩阵的概率逐项一致；模型文件各自留存，不要求序列化字节相同。
    done = {
        "at": io.now(),
        "status": "PRICE_VOLUME_DEVELOPMENT_COMPLETE",
        "actual_fits": len(io.lines(ROOT / "fit-ledger.jsonl")),
        "candidate_count": 15,
        "winner": ranking[0],
        "replay_probabilities_exact": True,
        "new_2026_scores": None,
        "adopted": False,
    }
    io.save(ROOT / "completion.json", done)
    return {k: v for k, v in done.items() if k != "winner"} | {
        "winner": winner["id"],
        "correct": ranking[0]["score"]["correct"],
    }


def verify():
    """不用训练评分器，加载全部模型独立读回；同时从原始序列再验新增特征。"""
    from decimal import Decimal

    from threadpoolctl import threadpool_limits

    configure()
    engine.access_guard()
    engine.verify_freeze()
    rows = io.lines(ROOT / "inputs.jsonl")
    nav = io.read(ROOT / "nav-through-2025.json")
    markets = io.read(ROOT / "market-values-through-2025.json")
    lineage = io.lines(ROOT / "feature-lineage.jsonl")
    assert max(nav) <= "2025-12-31"
    for row, ref in zip(rows, lineage, strict=True):
        v = np.asarray([float(nav[d]["unit_nav"]) for d in ref["nav_source_dates"]])
        ret = np.diff(v) / v[:-1]
        expected = [float(v[-1] / v[-1 - n] - 1) for n in (1, 2, 10, 40)] + [
            float(np.std(ret[-5:])),
            float(np.std(ret)),
            float(np.sqrt(np.mean(np.minimum(ret[-20:], 0) ** 2))),
        ]
        np.testing.assert_allclose(row["groups"]["NE"], expected, atol=1e-13, rtol=0)
        assert all(nav[d]["available_at"] <= ref["as_of"] for d in ref["nav_source_dates"])
        assert max(ref["market_source_dates"]) < ref["origin"] == row["base"] < row["target"]
        expected = []
        for code in source.timing.old.MARKETS:
            quotes = [markets[code][d] for d in ref["market_source_dates"]]
            rates = np.asarray([q["pct_chg"] / 100 for q in quotes])
            expected.extend(
                [float(np.prod(1 + rates[-20:]) - 1), float(np.std(rates[-5:])), float(np.std(rates[-20:]))]
            )
            expected.extend(
                float(quotes[-1][key] / np.mean([q[key] for q in quotes[:-1]]) - 1) for key in ("amount", "vol")
            )
        np.testing.assert_allclose(row["groups"]["ME"], expected, atol=1e-13, rtol=0)
    splits = io.read(ROOT / "splits.json")
    selection = io.read(ROOT / "selection.json")
    plan = io.read(ROOT / "protocol.json")
    definitions = [(c, f, False) for c in plan["candidates"] for f in engine.FOLDS] + [
        (selection["winner"]["candidate"], "V3", True)
    ]
    scores = {}
    checked = 0
    for c, fold, replay in definitions:
        rid = c["id"] + "__" + fold + ("__replay" if replay else "")
        run = io.read(ROOT / "runs" / rid / "complete.json")
        tr = [rows[i] for i in splits[fold]["training"]]
        er = [rows[i] for i in splits[fold]["evaluation"]]
        assert all(r["label_mature_at"] < splits[fold]["cutoff_exclusive"] for r in tr)
        assert max(r["target"] for r in tr) < min(r["target"] for r in er)
        x = np.asarray([[v for g in GROUPS[c["group"]] for v in r["groups"][g]] for r in tr])
        xe = np.asarray([[v for g in GROUPS[c["group"]] for v in r["groups"][g]] for r in er])
        saved = engine.predicted(run)
        m = joblib.load(run["model_path"])
        np.testing.assert_allclose(m["imputer"].statistics_, np.median(x, axis=0), atol=1e-13, rtol=0)
        np.testing.assert_allclose(m["scaler"].mean_, x.mean(axis=0), atol=1e-13, rtol=0)
        np.testing.assert_allclose(m["scaler"].var_, x.var(axis=0), atol=1e-13, rtol=0)
        with threadpool_limits(limits=2):
            probs = m["model"].predict_proba(m["scaler"].transform(m["imputer"].transform(xe)))
        aligned = np.zeros((len(er), 3))
        for j, label in enumerate(m["model"].classes_):
            aligned[:, engine.kernel.CLASSES.index(label)] = probs[:, j]
        correct = 0
        for r, p, value in zip(er, saved, aligned, strict=True):
            assert r["target"] == p["target"] and r["target"] <= "2025-12-31"
            np.testing.assert_allclose(p["probabilities"], value, atol=1e-13, rtol=0)
            assert p["predicted"] == engine.kernel.CLASSES[int(np.argmax(value))]
            a, b = Decimal(nav[r["base"]]["unit_nav"]), Decimal(nav[r["target"]]["unit_nav"])
            actual = "UP" if b > a else "DOWN" if b < a else "FLAT"
            correct += p["predicted"] == actual
        if not replay:
            scores.setdefault(c["id"], []).append(correct)
        checked += len(er)
    results = io.read(ROOT / "candidate-results.json")
    for result in results:
        assert scores[result["candidate"]["id"]] == [v["correct"] for v in result["fold_scores"]]
        assert sum(scores[result["candidate"]["id"]]) == result["score"]["correct"]
        assert result["score"]["dates"] == 243
        assert result["min_fold_accuracy"] == min(
            correct / len(splits[fold]["evaluation"])
            for fold, correct in zip(engine.FOLDS, scores[result["candidate"]["id"]], strict=True)
        )
    rank = sorted(
        results, key=lambda v: (-v["score"]["correct"], -v["min_fold_accuracy"], v["features"], v["candidate"]["id"])
    )
    assert rank == selection["ranking"]
    ledger = io.lines(ROOT / "fit-ledger.jsonl")
    assert len(ledger) == len(definitions) <= 80
    assert all(v["at"] > plan["at"] for v in ledger)
    baselines = io.read(ROOT / "baselines.json")
    for fold, split in splits.items():
        train_y = []
        actual = []
        for indices, output in ((split["training"], train_y), (split["evaluation"], actual)):
            for index in indices:
                row = rows[index]
                a, b = Decimal(nav[row["base"]]["unit_nav"]), Decimal(nav[row["target"]]["unit_nav"])
                output.append("UP" if b > a else "DOWN" if b < a else "FLAT")
        majority = max(engine.kernel.CLASSES, key=lambda c: (train_y.count(c), c))
        assert actual.count("UP") == baselines[fold]["always_up"]["correct"]
        assert actual.count(majority) == baselines[fold]["majority"]["correct"]
        reference = baselines[fold]["N_reference"]
        assert io.sha(Path(reference["path"])) == reference["sha256"]
        saved = io.lines(Path(reference["path"]))
        assert [p["target"] for p in saved] == split["evaluation_dates"]
        assert sum(p["predicted"] == a for p, a in zip(saved, actual, strict=True)) == baselines[fold]["N"]["correct"]
    outcome = {
        "at": io.now(),
        "models_verified": len(definitions),
        "predictions_recomputed": checked,
        "feature_rows_independently_recomputed": len(rows),
        "source_time_boundaries_checked": True,
        "all_candidate_scores_and_selection_match": True,
        "new_2026_scores": None,
        "strict_historical_first_seen_proven": False,
        "script_sha256": io.sha(Path(__file__)),
    }
    io.save(ROOT / "independent-verification.json", outcome)
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "run": run, "verify": verify}[args.command](), ensure_ascii=False, indent=2))
