"""固定旧第三轮的零拟合复核；只写独立修复证据目录，不能继续旧运行。

无模型路径、训练参数、日期参数或联网入口。只加载保护清单已固定的本项目
5 个模型；子进程禁用分类器和标准化器 fit。不得用此工具生成训练检查点。
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
from app.services import fund_002112_model_state_v2 as repaired
from app.services import fund_002112_round3_data as data
from app.services import fund_002112_round3_model as original
from app.services import fund_002112_round3_run as frozen

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / ".local-runs/fund-exposure-002112"
OLD = ROOT / "round3-runs/002112-r3-0ec4bfcbef4d93b3c49470f7"
OUTPUT = ROOT / "serialization-repair-v1"
SLOTS = (
    "2023Q2-N7-main",
    "2023Q2-N7-replay",
    "2023Q2-L20-main",
    "2023Q2-L20-replay",
    "2023Q2-T20-main",
)


def forbid_fit(*args, **kwargs):
    raise RuntimeError("ALL_FITTING_FORBIDDEN_IN_SERIALIZATION_AUDIT")


def disable_fitting():
    """在所有审计进程装上失败保护；只允许加载和推理。"""
    original.fit = forbid_fit
    repaired.StandardScaler.fit = forbid_fit
    repaired.LogisticRegression.fit = forbid_fit
    repaired.HistGradientBoostingClassifier.fit = forbid_fit


def protected_bytes(path):
    """以本次修复开始前的保护清单为锚点；读取后校验同一份内存字节。"""
    baseline = json.loads((OUTPUT / "protection-before.json").read_text(encoding="utf-8"))
    expected = baseline["files"][str(path.resolve())]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("PROTECTED_ARTIFACT_CHANGED:" + path.name)
    return raw


def protected_json(path):
    value = json.loads(protected_bytes(path))
    if set(value) != {"hash", "payload"} or data.digest(value["payload"]) != value["hash"]:
        raise ValueError("PROTECTED_ENVELOPE_INVALID")
    return value["payload"]


def inputs():
    """只读取原冻结截至 2024 年包中的 Q2 原训练/考试行，不重建来源或标签。"""
    protocol = protected_json(OLD / "protocol.json")
    if protocol["code"] != frozen.code_manifest() or protocol["environment"] != frozen.environment():
        raise ValueError("ORIGINAL_FROZEN_CODE_OR_ENVIRONMENT_CHANGED")
    values = {key: protected_json(OLD / (key + ".json")) for key in ("inputs", "folds")}
    for key, value in values.items():
        if data.digest(value) != protocol[key + "_hash"]:
            raise ValueError("ORIGINAL_INPUT_IDENTITY_CHANGED")
    training, exams, weights = frozen.fold_data(values, "2023Q2")
    return training, exams, weights


def worker(slot, *, restored_copy=False, manifest_hash=None):
    """两个独立进程分别核对旧文件、v2 副本；全量逐行比对原已保存预测。"""
    training, exams, weights = inputs()
    reference = protected_json(OLD / "predictions" / (slot + ".json"))
    old_manifest = protected_json(OLD / "model-manifests" / (slot + ".json"))
    if restored_copy:
        model = repaired.restore_model(OUTPUT / "verified-copies", slot, manifest_hash)
    else:
        raw = protected_bytes(OLD / "models" / (slot + ".joblib"))
        if hashlib.sha256(raw).hexdigest() != old_manifest["file_sha256"]:
            raise ValueError("ORIGINAL_MODEL_FILE_CHANGED")
        model = joblib.load(io.BytesIO(raw))
    if (
        model["training_hash"] != data.digest(training)
        or model["weights_hash"] != data.digest(weights)
        or model["variant"] != slot.split("-")[1]
    ):
        raise ValueError("ORIGINAL_MODEL_INPUT_MISMATCH")
    for name in ("variant", "features", "classes", "recipe", "training_hash", "weights_hash"):
        if model[name] != old_manifest[name]:
            raise ValueError("ORIGINAL_MODEL_MANIFEST_MISMATCH")
    if (
        model["scaler"].mean_.tolist() != old_manifest["mean"]
        or model["scaler"].scale_.tolist() != old_manifest["scale"]
    ):
        raise ValueError("ORIGINAL_SCALER_MANIFEST_MISMATCH")
    state = repaired.model_state_digest(model)
    checks = repaired.verify_all_predictions(model, training, exams, reference)
    if not restored_copy:
        manifest_hash = repaired.save_model(model, OUTPUT / "verified-copies", slot)
    return {
        "slot": slot,
        "pid": os.getpid(),
        "phase": "v2_fresh_restore" if restored_copy else "original_fresh_restore",
        "state_sha256": state,
        "manifest_sha256": manifest_hash,
        "all_rows": checks,
        "old_object_hash_matches": original.state_hash(model) == old_manifest["state_hash"],
        "new_real_fits": 0,
        "fit_methods_disabled": True,
    }


def child(slot, *, manifest_hash=None):
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-B",
        "-m",
        "scripts.fund_002112_serialization_verify",
        "--worker",
        "--slot",
        slot,
    ]
    if manifest_hash:
        command += ["--restored-copy", "--manifest-hash", manifest_hash]
    result = subprocess.run(
        command,
        cwd=PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    if result.returncode:
        raise RuntimeError("ZERO_FIT_CHILD_FAILED:" + result.stderr[-1600:])
    return json.loads(result.stdout)


def verify():
    destination = OUTPUT / "zero-fit-verification.json"
    if destination.exists() or (OUTPUT / "verified-copies").exists():
        raise FileExistsError("AUDIT_ALREADY_STARTED_NO_OVERWRITE")
    expected = {slot + ".json" for slot in SLOTS}
    if {path.name for path in (OLD / "attempts").glob("*.json")} != expected:
        raise ValueError("ORIGINAL_FIT_COUNT_CHANGED")
    results = []
    for slot in SLOTS:
        before = child(slot)
        after = child(slot, manifest_hash=before["manifest_sha256"])
        if before["state_sha256"] != after["state_sha256"] or before["all_rows"] != after["all_rows"]:
            raise ValueError("V2_ROUNDTRIP_CHANGED")
        results.append({"slot": slot, "before": before, "after": after})
        print(slot + ": 3420 训练行 + 50 检查日通过；新增拟合 0", flush=True)
    states = {value["slot"]: value["before"]["state_sha256"] for value in results}
    for variant in ("N7", "L20"):
        if states[f"2023Q2-{variant}-main"] != states[f"2023Q2-{variant}-replay"]:
            raise ValueError("EXISTING_REPLICA_STATE_MISMATCH")
    decision = protected_json(OLD / "historical-decision.json")
    evidence = {
        "at": data.now(),
        "schema": repaired.SCHEMA,
        "environment": repaired.runtime_environment(),
        "new_real_fits": 0,
        "previous_real_fits": 34,
        "old_round3_real_fits": 5,
        "lifetime_real_fits": 39,
        "old_status": decision["status"],
        "old_resume_allowed": False,
        "repair_code": {
            str(path.relative_to(PROJECT)): data.file_hash(path)
            for path in (PROJECT / "app/services/fund_002112_model_state_v2.py", Path(__file__).resolve())
        },
        "results": results,
        "existing_logistic_replicas_equal": True,
        "T20_independent_real_replica_available": False,
        "old_pre_save_full_numeric_state_recoverable": False,
        "historical_161_days_evaluation_completed": False,
        "new_protocol_bridge_decision": "NOT_GRANTED_BY_DIAGNOSTIC",
        "passed": True,
    }
    data.write_once(destination, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--slot", choices=SLOTS)
    parser.add_argument("--restored-copy", action="store_true")
    parser.add_argument("--manifest-hash")
    args = parser.parse_args()
    disable_fitting()
    if args.worker:
        if not args.slot or (args.restored_copy and not args.manifest_hash):
            parser.error("内部核验参数不完整")
        print(json.dumps(worker(args.slot, restored_copy=args.restored_copy, manifest_hash=args.manifest_hash)))
    else:
        if args.slot or args.restored_copy or args.manifest_hash:
            parser.error("只有内部核验使用槽位参数")
        verify()


if __name__ == "__main__":
    main()
