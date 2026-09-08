"""复盘已保存校准实验；不训练、不联网采集、不写库、不读取2025评分。

python -m scripts.historical_nav_diagnostics --calibration-run-key 旧校准UUID --run-key 新诊断UUID
"""

import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID

from app.core.config import get_settings
from app.core.logging import get_logger
from app.repositories.historical_nav_diagnostics import read_nav_diagnostic_snapshot
from app.schemas.historical_nav_calibration import HistoricalNavCalibrationRequest, HistoricalNavCalibrationResponse
from app.services.historical_nav_calibration import restore_calibrated_artifact
from app.services.historical_nav_diagnostics import audit_nav_snapshot, diagnose_prepared
from app.services.historical_nav_evaluation import _hash_parts, load_historical_nav_dataset
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from scripts.historical_nav_candidate import _write_json

logger = get_logger(__name__)
RUN_ROOT = Path(__file__).resolve().parents[1] / ".local-runs"


def _read_bounded(path: Path) -> bytes:
    with path.open("rb") as handle:
        content = handle.read(2097153)
    if len(content) > 2097152:
        raise ValueError("diagnostic input exceeds 2MiB")
    return content


def _run_directory(kind: str, key: UUID) -> Path:
    if not isinstance(key, UUID):
        raise ValueError("run key must be UUID")
    root = RUN_ROOT.resolve()
    directory = root / f"nav-{kind}-{key}"
    if directory.resolve().parent != root:
        raise ValueError("run directory escaped local root")
    return directory


def read_source_bundle(key: UUID) -> tuple[HistoricalNavCalibrationRequest, HistoricalNavCalibrationResponse, dict]:
    """固定文件名、体积上限、完成清单和JSON数值模型交叉校验；不执行模型代码。"""
    directory = _run_directory("calibration", key)
    manifest_bytes = _read_bounded(directory / "complete.json")
    manifest = json.loads(manifest_bytes)
    names = {"request.json", "report.json", "models.json"}
    if (
        manifest["run_key"] != str(key)
        or manifest["status"] != "CALIBRATION_EVALUATED"
        or set(manifest["files_sha256"]) != names
    ):
        raise ValueError("source experiment is not a completed fixed calibration")
    contents = {}
    for name in sorted(names):
        content = _read_bounded(directory / name)
        if hashlib.sha256(content).hexdigest() != manifest["files_sha256"][name]:
            raise ValueError("source experiment file hash mismatch")
        contents[name] = content
    request = HistoricalNavCalibrationRequest.model_validate_json(contents["request.json"])
    report = HistoricalNavCalibrationResponse.model_validate_json(contents["report.json"])
    models = json.loads(contents["models.json"])
    if (
        request.expected_dataset_hash != report.preparation.dataset_hash
        or manifest["dataset_hash"] != request.expected_dataset_hash
        or set(request.batch_ids) != set(report.preparation.batch_ids)
        or set(models) != {w.window.window_id for w in report.windows}
        or manifest["model_hashes"] != {w.window.window_id: w.model.model_hash for w in report.windows if w.model}
    ):
        raise ValueError("source request/report/model linkage mismatch")
    for window in report.windows:
        if restore_calibrated_artifact(json.dumps(models[window.window.window_id])) != window.model:
            raise ValueError("source model differs from source report")
    hashes = {**manifest["files_sha256"], "complete.json": hashlib.sha256(manifest_bytes).hexdigest()}
    return request, report, hashes


def run_diagnostics(calibration_run_key: UUID, run_key: UUID) -> tuple[Path, dict]:
    """读取旧产物+当前只读样本；只新建诊断目录，旧模型/批次完全保留。"""
    target = make_url(get_settings().ai_database_url)
    if target.host not in {"localhost", "127.0.0.1", "::1"} or target.database != "fund_ai":
        raise ValueError("diagnostic CLI is restricted to local fund_ai")
    directory = _run_directory("diagnostics", run_key)
    if directory.exists():
        raise FileExistsError("diagnostic run already exists")
    request, saved, source_hashes = read_source_bundle(calibration_run_key)
    data = load_historical_nav_dataset(request.evaluation_request())
    if request.expected_dataset_hash != data.report.dataset_hash:
        raise ValueError("saved batch dataset has changed")
    report = diagnose_prepared(data, saved)
    snapshot = read_nav_diagnostic_snapshot(tuple(f.fund_code for f in data.report.funds), data.report.source_code)
    report["nav_audit"] = audit_nav_snapshot(data, snapshot["nav_rows"])
    report["raw_snapshot_hash"] = _hash_parts(snapshot["nav_rows"])
    report["catalog_and_source"] = {k: v for k, v in snapshot.items() if k != "nav_rows"}
    report["source_calibration_run_key"] = str(calibration_run_key)
    report["source_files_sha256"] = source_hashes
    report["database_written"] = False
    report["admission"] = {
        "status": "NOT_APPROVED",
        "first_available_version": "NOT_PROVEN_CURRENT_SNAPSHOT_ONLY",
        "strict_trading_calendar": "NOT_VERIFIED",
        "dividend_adjustment": "SENSITIVITY_CHECK_ONLY_NOT_CERTIFIED",
        "source_authorization": "LOCAL_REGISTRY_ONLY_NOT_LEGAL_OR_PROVIDER_REVALIDATION",
        "representative_fund_universe": "THREE_PILOT_FUNDS_ONLY",
        "note": "净值重放一致不等于首次可得正确；元数据登记不等于重新取得授权。没有解除正式准入。",
    }
    # 再校验源实验，发现文件被并发改动就拒绝交付；不覆盖或回滚别人的文件。
    if read_source_bundle(calibration_run_key)[2] != source_hashes:
        raise ValueError("source experiment changed during diagnostics")
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    report = json.loads(json.dumps(report, default=str, ensure_ascii=False, allow_nan=False))
    digest = _write_json(directory / "report.json", report)
    completion = {
        "run_key": str(run_key),
        "source_calibration_run_key": str(calibration_run_key),
        "status": "DIAGNOSTICS_COMPLETED",
        "admission_status": "NOT_APPROVED",
        "purpose": "LEARNING_ONLY",
        "publication_status": "MODEL_NOT_RELEASED",
        "model_fitted": False,
        "database_written": False,
        "test_scored": False,
        "dataset_hash": data.report.dataset_hash,
        "raw_snapshot_hash": report["raw_snapshot_hash"],
        "files_sha256": {"report.json": digest},
        "source_files_sha256": source_hashes,
    }
    _write_json(directory / "complete.json", completion)
    return directory, completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run-key", type=UUID, required=True, help="已完成且要复盘的校准实验UUID")
    parser.add_argument("--run-key", type=UUID, required=True, help="新诊断目录UUID，不覆盖已有实验")
    args = parser.parse_args()
    logger.info("historical_nav_diagnostics.main >>> started, run_key=%s", args.run_key)
    try:
        directory, completion = run_diagnostics(args.calibration_run_key, args.run_key)
        print("DIAGNOSTICS_RESULT=" + json.dumps({"directory": str(directory), **completion}, ensure_ascii=False))
        logger.info("historical_nav_diagnostics.main >>> completed, run_key=%s", args.run_key)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError, RuntimeError, SQLAlchemyError):
        logger.exception("historical_nav_diagnostics.main >>> failed, run_key=%s", args.run_key)
        print("DIAGNOSTICS_RESULT=" + json.dumps({"run_key": str(args.run_key), "status": "FAILED_NO_PUBLICATION"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
