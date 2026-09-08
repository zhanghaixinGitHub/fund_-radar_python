"""本机固定时间校准实验：只读原批次，保存报告和组合模型，不覆盖旧实验或发布。

python -m scripts.historical_nav_calibration --request 候选训练请求.json --run-key 新UUID
"""

import argparse
import json
from pathlib import Path
from uuid import UUID

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.historical_nav_calibration import HistoricalNavCalibrationRequest
from app.services.historical_nav_calibration import evaluate_stored_calibration, restore_calibrated_artifact
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from scripts.historical_nav_candidate import _write_json, read_training_request

logger = get_logger(__name__)
RUN_ROOT = Path(__file__).resolve().parents[1] / ".local-runs"


def run_calibration(request: HistoricalNavCalibrationRequest, run_key: UUID) -> tuple[Path, dict]:
    """只新建当前本机实验目录；部分/不足报告可保存，但不标成完整模型验收成功。"""
    target = make_url(get_settings().ai_database_url)
    if target.host not in {"localhost", "127.0.0.1", "::1"} or target.database != "fund_ai":
        raise ValueError("local calibration CLI is restricted to local fund_ai")
    if not isinstance(run_key, UUID):
        raise ValueError("run key must be UUID")
    root = RUN_ROOT.resolve()
    directory = root / f"nav-calibration-{run_key}"
    if directory.resolve().parent != root:
        raise ValueError("run directory escaped local root")
    root.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    files = {"request.json": _write_json(directory / "request.json", request.model_dump(mode="json", by_alias=True))}
    result = evaluate_stored_calibration(request)
    files["report.json"] = _write_json(directory / "report.json", result.model_dump(mode="json"))
    models = {w.window.window_id: w.model.model_dump(mode="json") for w in result.windows if w.model}
    files["models.json"] = _write_json(directory / "models.json", models)
    readback = json.loads((directory / "models.json").read_text(encoding="utf-8"))
    for window in result.windows:
        if window.model:
            restored = restore_calibrated_artifact(json.dumps(readback[window.window.window_id]))
            if restored != window.model:
                raise ValueError("saved calibration model differs from response")
    completion = {
        "run_key": str(run_key),
        "status": result.status,
        "purpose": "LEARNING_ONLY",
        "publication_status": "MODEL_NOT_RELEASED",
        "database_written": False,
        "artifact_persisted": bool(models),
        "evaluated_window_count": result.evaluated_window_count,
        "dataset_hash": result.preparation.dataset_hash,
        "model_hashes": {w.window.window_id: w.model.model_hash for w in result.windows if w.model},
        "files_sha256": files,
    }
    # 最后写清单；部分窗口失败时status仍是PARTIAL/NO_VALID，不伪装全部完成。
    _write_json(directory / "complete.json", completion)
    return directory, completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="沿用含expectedDatasetHash的2022–2025候选请求")
    parser.add_argument("--run-key", type=UUID, required=True, help="新实验目录UUID，不覆盖旧文件")
    args = parser.parse_args()
    logger.info("historical_nav_calibration.main >>> started, run_key=%s", args.run_key)
    try:
        request = HistoricalNavCalibrationRequest.model_validate(read_training_request(args.request).model_dump())
        directory, completion = run_calibration(request, args.run_key)
        logger.info(
            "historical_nav_calibration.main >>> completed, run_key=%s, status=%s, windows=%s",
            args.run_key,
            completion["status"],
            completion["evaluated_window_count"],
        )
        print("CALIBRATION_RESULT=" + json.dumps({"directory": str(directory), **completion}, ensure_ascii=False))
        return 0 if completion["status"] == "CALIBRATION_EVALUATED" else 2
    except (SQLAlchemyError, OSError, ValueError, RuntimeError, ArithmeticError, ImportError, RuntimeWarning):
        logger.exception("historical_nav_calibration.main >>> failed, run_key=%s", args.run_key)
        print("CALIBRATION_RESULT=" + json.dumps({"run_key": str(args.run_key), "status": "FAILED_NO_PUBLICATION"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
