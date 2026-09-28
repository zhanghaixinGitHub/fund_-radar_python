"""补资料实验的独立零拟合验收：不调用执行器、训练或比较函数。

直接核对磁盘字节、台账、同日正确集合，并用重新载入的模型逐行重算分数。
只读取本次独立目录内已冻结的 <=2024 输入；不访问数据库或外部来源。
"""

import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

RUN = Path(__file__).resolve().parents[1] / ".local-runs/fund-exposure-002112/peer-recovered-experiment/20260928-v1"
CLASSES = ("DOWN", "FLAT", "UP")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and set(value) == {"hash", "payload"}:
        assert digest(value["payload"]) == value["hash"]
        value = value["payload"]
    return value


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path, expected):
    assert sha(path) == expected, str(path)


def q(day):
    return f"{day[:4]}Q{(int(day[5:7]) - 1) // 3 + 1}"


def audit(path=RUN):
    checked(path / "protocol.json", read(path / "protocol-anchor.json")["sha256"])
    protocol = read(path / "protocol.json")
    assert protocol["fit_budget"] == 12 and protocol["old_package_budget_unchanged"] == 0
    for name, expected in protocol["frozen"].items():
        checked(path / name, expected)
    for name, expected in protocol["code"].items():
        checked(Path(name), expected)
    pools = {
        key: read(path / "frozen" / name)["train"]
        for key, name in (("recovered", "inputs.json"), ("original", "original-inputs.json"))
    }
    folds = read(path / "frozen/folds.json")
    old_folds = read(path / "frozen/original-folds.json")
    assert (len(pools["recovered"]), len(pools["original"])) == (5137, 4419)
    inputs = {key: {(r["fund_code"], r["target"]): r for r in rows} for key, rows in pools.items()}
    assert all(inputs["recovered"][key] == row for key, row in inputs["original"].items())
    events = list((path / "attempts").glob("*.json"))
    slots = protocol["slots"][: len(events)]
    assert len(events) <= 12 and {p.stem for p in events} == set(slots)
    previous = sha(path / "protocol.json")
    recomputed = {}
    model_checks = {}
    for number, slot in enumerate(slots, 1):
        event_path = path / "attempts" / (slot + ".json")
        event = read(event_path)
        assert (event["ordinal"], event["previous_sha256"], event["slot"]) == (number, previous, slot)
        assert event["protocol_sha256"] == sha(path / "protocol.json")
        previous = sha(event_path)
        cp = read(path / "checkpoints" / (slot + ".json"))
        assert cp["attempt_sha256"] == previous
        for name, expected in cp["files"].items():
            checked(path / name, expected)
        receipt = read(path / "fit-complete" / (slot + ".json"))
        manifest_path = path / "models" / (slot + ".manifest.json")
        checked(manifest_path, receipt["manifest_sha256"])
        manifest = read(manifest_path)
        raw = (path / "models" / (slot + ".joblib")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest["file_sha256"]
        # 仅在固定清单和检查点字节完整性校验通过后加载本项目自生成模型。
        fitted = joblib.load(io.BytesIO(raw))
        stage, variant, _ = slot.split("-")
        kind = "recovered" if variant == "L20_RECOVERED" else "original"
        fold = next(f for f in (folds if kind == "recovered" else old_folds) if f["name"] == stage)
        training = [inputs[kind][tuple(key)] for key in fold["train_ids"]]
        exam = next(f["exam"] for f in folds if f["name"] == stage)
        assert max(r["target"] for r in training) <= "2023-12-31"
        assert max(r["target"] for r in exam) <= "2024-12-31"
        assert fitted["training_hash"] == digest(training)
        assert fitted["weights_hash"] == digest(fold["weights"])
        assert fitted["classifier"].get_params() == protocol["recipe"]
        assert fitted["classifier"].classes_.tolist() == list(CLASSES)
        saved = read(path / "predictions" / (slot + ".json"))
        checks = {}
        for part, rows in (("train", training), ("exam", exam)):
            size = 7 if variant == "N7_ORIGINAL" else 20
            x = np.array([r["x"][:size] for r in rows], dtype=float)
            with threadpool_limits(limits=1):
                scores = fitted["classifier"].predict_proba(fitted["scaler"].transform(x))
            expected_scores = np.array([r["scores"] for r in saved[part]])
            assert scores.shape == expected_scores.shape == (len(rows), 3)
            assert np.isfinite(scores).all() and np.isfinite(expected_scores).all()
            delta = float(np.max(np.abs(scores - expected_scores)))
            assert delta <= 1e-12
            for row, stored, score in zip(rows, saved[part], scores, strict=True):
                assert (stored["fund_code"], stored["target"], stored["input_hash"], stored["actual_direction"]) == (
                    row["fund_code"],
                    row["target"],
                    digest(row),
                    row["actual_direction"],
                )
                answer = max(("FLAT", "UP", "DOWN"), key=lambda name: score[CLASSES.index(name)])
                assert answer == stored["direction"]
            checks[part] = {"rows": len(rows), "max_score_delta": delta}
        model_checks[slot] = checks
        recomputed[slot] = saved
        if slot.endswith("-replay"):
            main = slot.removesuffix("-replay") + "-main"
            assert saved == recomputed[main]
            assert receipt["state_sha256"] == read(path / "fit-complete" / (main + ".json"))["state_sha256"]
    results = {}
    for historical, filename in ((True, "historical-decision.json"), (False, "full-decision.json")):
        if not (path / filename).exists():
            continue
        result = read(path / filename)
        groups = {}
        for variant in ("L20_RECOVERED", "L20_ORIGINAL", "N7_ORIGINAL"):
            predictions = []
            for stage in ("2023Q2", "2023Q3", "2023Q4") if historical else ("FULL",):
                if historical and variant != "L20_RECOVERED":
                    predictions.extend(
                        read(path / f"frozen/controls/{stage}-{variant.split('_')[0]}-main.json")["exam"]
                    )
                else:
                    predictions.extend(recomputed[f"{stage}-{variant}-main"]["exam"])
            groups[variant] = {r["target"]: r for r in predictions}
            assert len(groups[variant]) == (161 if historical else 230)
        first = groups["L20_RECOVERED"]
        assert all(set(rows) == set(first) for rows in groups.values())
        correct = {
            name: {day for day, row in rows.items() if row["direction"] == row["actual_direction"]}
            for name, rows in groups.items()
        }
        for name, dates in correct.items():
            assert result["models"][name]["correct"] == len(dates)
            for category in CLASSES:
                assert result["models"][name]["class_correct"][category] == sum(
                    first[d]["actual_direction"] == category for d in dates
                )
            for quarter, metrics in result["quarters"].items():
                assert metrics[name]["correct"] == sum(q(d) == quarter for d in dates)
        candidate = correct["L20_RECOVERED"]
        checks = {
            "total_strictly_above_controls": all(
                len(candidate) > len(correct[n]) for n in ("L20_ORIGINAL", "N7_ORIGINAL")
            ),
            "total_strictly_above_constants": all(
                len(candidate) > sum(r["actual_direction"] == c for r in first.values()) for c in CLASSES
            ),
        }
        for c in CLASSES:
            days = {d for d, r in first.items() if r["actual_direction"] == c}
            checks["class_not_worse_" + c] = all(
                len(candidate & days) >= len(correct[n] & days) for n in ("L20_ORIGINAL", "N7_ORIGINAL")
            )
        if historical:
            checks["at_least_two_quarters_not_worse_L20"] = (
                sum(
                    sum(q(d) == quarter for d in candidate) >= sum(q(d) == quarter for d in correct["L20_ORIGINAL"])
                    for quarter in ("2023Q2", "2023Q3", "2023Q4")
                )
                >= 2
            )
        assert result["checks"] == checks and result["passed"] == all(checks.values())
        for control in ("L20_ORIGINAL", "N7_ORIGINAL"):
            gained, lost = candidate - correct[control], correct[control] - candidate
            pair = result["pairs"][control]
            assert (pair["gained"], pair["lost"], pair["net"]) == (len(gained), len(lost), len(gained) - len(lost))
            for c, item in pair["classes"].items():
                assert item["gained"] == sum(first[d]["actual_direction"] == c for d in gained)
                assert item["lost"] == sum(first[d]["actual_direction"] == c for d in lost)
            for quarter, item in pair["quarters"].items():
                assert item["gained"] == sum(q(d) == quarter for d in gained)
                assert item["lost"] == sum(q(d) == quarter for d in lost)
        results[filename] = {"checks": checks, "passed": all(checks.values())}
    checked(path / "decision.json", read(path / "decision-anchor.json")["sha256"])
    decision = read(path / "decision.json")
    assert decision["consumed_slots"] == decision["actual_new_fits"] == len(slots)
    assert decision["cumulative_actual_fits"] == 52 + len(slots)
    assert {p.stem for p in (path / "classifier-started").glob("*.json")} == set(slots)
    assert {p.stem for p in (path / "classifier-completed").glob("*.json")} == set(slots)
    if not results["historical-decision.json"]["passed"]:
        assert len(slots) == 6 and not decision["full_executed"]
        assert not (path / "full-decision.json").exists() and not decision["adopted"]
    before = read(path / "resume-before.json")
    for name, expected in before["files"].items():
        checked(path / name, expected)
    return {
        "passed": True,
        "audited_at": datetime.now().astimezone().isoformat(),
        "audit_pid": os.getpid(),
        "audit_script_sha256": sha(Path(__file__)),
        "models": model_checks,
        "stages": results,
        "new_fits_in_audit": 0,
        "actual_experiment_fits": len(slots),
        "resumed_q2_artifacts_unchanged": True,
        "original_4419_rows_preserved": True,
        "budget_and_stop_verified": True,
    }


if __name__ == "__main__":
    output = audit()
    with (RUN / "independent-result-audit.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, sort_keys=True, indent=2)
    print(json.dumps({k: v for k, v in output.items() if k not in ("models", "stages")}, ensure_ascii=False))
