"""独立回读晚间模型、重算概率、逐日时间链及训练样本，验证旧工作区保护。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np

from scripts import fund_002112_evening_fusion_v2 as m


def manual_probabilities(saved, records, branch):
    """独立组装保存的预处理结果；不调用训练器的 predict_expert。"""
    matrix = np.array([r["groups"][branch] for r in records], dtype=float)
    state = saved["preprocessing"]
    raw = matrix[:, state["keep"]]
    mask = np.isnan(raw)
    filled = raw.copy()
    for j in range(filled.shape[1]):
        filled[mask[:, j], j] = state["median"][j]
    encoded = np.hstack([filled, mask.astype(float)])
    probabilities = saved["model"].predict_proba(encoded)
    output = np.zeros((len(records), 3))
    for j, label in enumerate(saved["model"].classes_):
        output[:, m.CLASSES.index(label)] = probabilities[:, j]
    return output


def verify(root: Path):
    m.verify_freeze(root)
    rows = m.lines(root / "dataset.jsonl")
    by_target = {r["target"]: r for r in rows}
    source = m.read(root / "sources.json")
    lineage = m.lines(root / "daily-lineage.jsonl")
    events = 0
    for row, proof in zip(rows, lineage, strict=True):
        assert row["target"] == proof["target"]
        assert source["sessions"][source["sessions"].index(row["base"]) + 1] == row["target"]
        expected_as_of = (m.date.fromisoformat(row["target"]) - m.timedelta(days=1)).isoformat() + "T23:00:00+08:00"
        assert row["as_of"] == expected_as_of
        assert row["label_mature_at"] > row["as_of"]
        assert proof["history"]["maximum_available_at"] <= row["as_of"]
        assert max(proof["history"]["nav_days"]) <= row["base"]
        for e in proof.get("information", {}).get("events", []):
            assert m.moment(e["available_at"]) <= m.moment(row["as_of"])
            assert e["age"] >= 0
            events += 1
        if "market" in proof:
            assert max(proof["market"]["price_days"]) == row["base"]
            assert proof["market"]["report_available_at"] <= row["as_of"]
    maximum_error, model_count, probability_rows = 0.0, 0, 0
    probabilities = {}
    ledger = m.lines(root / "fit-ledger.jsonl")
    assert len({e["id"] for e in ledger}) == len(ledger) <= 100
    for entry in ledger:
        fit = m.read(root / "fits" / (entry["id"] + ".json"))
        path = root / "models" / (entry["id"] + ".joblib")
        assert m.sha(path) == fit["model_sha256"]
        branch, cutoff = entry["branch"], entry["cutoff"]
        expected = [
            r for r in rows if branch in r["groups"] and r["label_mature_at"] < cutoff and r["target"] < cutoff[:10]
        ]
        assert [r["target"] for r in expected] == entry["training_targets"]
        saved = joblib.load(path)
        matrix = np.array([r["groups"][branch] for r in expected], dtype=float)
        keep = ~np.isnan(matrix).all(axis=0)
        assert np.array_equal(keep, saved["preprocessing"]["keep"])
        assert np.allclose(np.nanmedian(matrix[:, keep], axis=0), saved["preprocessing"]["median"], atol=0, rtol=0)
        evaluate = [by_target[d] for d in fit["evaluation_targets"]]
        if evaluate:
            calculated = manual_probabilities(saved, evaluate, branch)
            delta = float(np.max(np.abs(calculated - np.array(fit["probabilities"]))))
            maximum_error = max(maximum_error, delta)
            assert delta <= 1e-12
            for row, probability in zip(evaluate, calculated, strict=True):
                probabilities[(entry["id"], row["target"])] = probability
            probability_rows += len(evaluate)
        model_count += 1
    predictions = m.lines(root / "historical-predictions.jsonl")
    for row in predictions:
        combined = np.zeros(3)
        for branch, weight in m.WEIGHTS.items():
            p = probabilities[(row["fit_prefix"] + branch, row["target"])]
            assert np.allclose(p, row["branches"][branch], atol=1e-12, rtol=0)
            combined += weight * p
        assert np.allclose(combined, row["weighted_55"], atol=1e-12, rtol=0)
        assert np.allclose(
            (np.array(row["branches"]["market"]) * 2 + row["branches"]["history"]) / 3,
            row["price_history"],
            atol=1e-12,
            rtol=0,
        )
    package = joblib.load(root / "002112-evening-1d-news55.joblib")
    final_counts = {}
    sample = [r for r in rows if "information" in r["groups"]][-5:]
    for branch in m.WEIGHTS:
        expert = package["experts"][branch]
        last = joblib.load(root / "models" / ("FULL_" + branch + ".joblib"))
        assert np.allclose(
            manual_probabilities(expert, sample, branch), manual_probabilities(last, sample, branch), atol=1e-12, rtol=0
        )
        final_counts[branch] = expert["row_count"]
    # 一次全量信息模型同种子重放，检查真实拟合可复现性；绝不增加候选或按成绩调参。
    replay_id = "REPLAY_FULL_information"
    original = package["experts"]["information"]
    if not any(e["id"] == replay_id for e in ledger):
        replay = m.fit_expert(root, replay_id, "information", rows, original["cutoff"], sample)
    else:
        replay = joblib.load(root / "models" / (replay_id + ".joblib"))
    replay_error = float(
        np.abs(
            manual_probabilities(replay, sample, "information") - manual_probabilities(original, sample, "information")
        ).max()
    )
    assert replay_error <= 1e-12
    before = m.read(root / "protection-before.json")
    # 本验证器是当前任务新增文件，验证过程中允许修正断言；既有项目文件仍须逐字不变。
    own_verifier = str(Path(__file__).resolve())
    changed = [
        path
        for repo in before.values()
        for path, digest in repo["files"].items()
        if path != own_verifier and (not Path(path).exists() or m.sha(Path(path)) != digest)
    ]
    assert not changed, changed
    after = m.git_snapshot()
    assert all(after[repo]["head"] == v["head"] for repo, v in before.items())
    rebuilt, _, coverage = m.build_dataset(source)
    assert rebuilt == rows
    assert coverage == m.read(root / "coverage.json")
    scores = m.read(root / "scores.json")
    for period in ["2025", "2026", "ALL"]:
        subset = [r for r in predictions if period == "ALL" or r["target"].startswith(period)]
        for method in scores[period]:
            correct = sum(m.CLASSES[int(np.argmax(r[method]))] == r["label"] for r in subset)
            assert scores[period][method]["correct"] == correct
            assert math.isclose(scores[period][method]["accuracy"], correct / len(subset))
    result = {
        "verified_at": m.now(),
        "passed": True,
        "models_read_back": model_count,
        "probability_rows_recomputed": probability_rows,
        "maximum_probability_error": maximum_error,
        "replay_maximum_error": replay_error,
        "actual_fits_including_replay": len(m.lines(root / "fit-ledger.jsonl")),
        "dataset_rows_rebuilt": len(rows),
        "event_time_links": events,
        "protected_workspace_files": sum(len(r["files"]) for r in before.values()),
        "source_hashes_unchanged": True,
        "old_workspace_file_changes": changed,
        "final_training_rows": final_counts,
        "formal_admission": False,
        "verification_code_sha256": m.sha(Path(__file__)),
        "roundtrip_tolerance": 1e-12,
        "verification_tolerance_note": "多线程随机森林求和存在末位浮点差异；逐行使用固定1e-12而不要求比特相同。",
    }
    m.write(root / "independent-verification.json", result)
    print(m.json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=m.ROOT)
    verify(parser.parse_args().root)
