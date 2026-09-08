"""本机离线训练并保存JSON实验包；只读数据库，不覆盖已有目录，不发布模型。

python -m scripts.historical_nav_candidate --request 请求.json --run-key UUID
HTTP返回的artifact_persisted=false描述服务行为；本CLI的complete.json单独证明本地文件保存完成。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.historical_nav_training import HistoricalNavTrainingRequest
from app.services.historical_nav_training import restore_logistic_artifact, train_stored_historical_nav_candidate
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

logger = get_logger(__name__)
RUN_ROOT = Path(__file__).resolve().parents[1] / ".local-runs"


def read_training_request(path: Path) -> HistoricalNavTrainingRequest:
    """读取单个有界UTF-8 JSON；不接受pickle、外部矩阵或任意训练参数。"""
    with path.open("rb") as handle:
        content = handle.read(131073)
    if len(content) > 131072:
        raise ValueError("request exceeds 128KiB")
    return HistoricalNavTrainingRequest.model_validate_json(content)


def _write_json(path: Path, payload: dict) -> str:
    """仅新建文件，写完flush/fsync；返回校验指纹，任何错误不生成完成清单。"""
    content = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != content:
        raise OSError("artifact file readback mismatch")
    return hashlib.sha256(content).hexdigest()


def run_candidate(request: HistoricalNavTrainingRequest, run_key: UUID) -> tuple[Path, dict]:
    """先占用新目录再运行；失败只留诊断产物，不删除旧实验或无声覆盖。"""
    target = make_url(get_settings().ai_database_url)
    if target.host not in {"localhost", "127.0.0.1", "::1"} or target.database != "fund_ai":
        raise ValueError("local candidate CLI is restricted to local fund_ai")
    if not isinstance(run_key, UUID):
        raise ValueError("run key must be UUID")
    root = RUN_ROOT.resolve()
    directory = root / f"nav-candidate-{run_key}"
    # 既不接受用户输出路径，也不跟随预先创建的run-key链接到别处。
    if directory.resolve().parent != root:
        raise ValueError("run directory must stay within local run root")
    root.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    files = {"request.json": _write_json(directory / "request.json", request.model_dump(mode="json", by_alias=True))}
    result = train_stored_historical_nav_candidate(request)
    files["report.json"] = _write_json(directory / "report.json", result.model_dump(mode="json"))
    if result.model is not None:
        files["model.json"] = _write_json(directory / "model.json", result.model.model_dump(mode="json"))
        restored = restore_logistic_artifact((directory / "model.json").read_bytes())
        if restored != result.model:
            raise ValueError("saved model does not match in-memory artifact")
    completion = {
        "run_key": str(run_key),
        "status": result.status,
        "purpose": "LEARNING_ONLY",
        "publication_status": "MODEL_NOT_RELEASED",
        "database_written": False,
        "artifact_persisted": result.model is not None,
        "dataset_hash": result.preparation.dataset_hash,
        "model_hash": result.model.model_hash if result.model else None,
        "files_sha256": files,
    }
    # 最后写完成清单。只有清单可解析且各文件指纹一致，才算完整实验包。
    _write_json(directory / "complete.json", completion)
    return directory, completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="含expectedDatasetHash的训练请求JSON")
    parser.add_argument("--run-key", type=UUID, required=True, help="新实验UUID；已有目录不会覆盖")
    args = parser.parse_args()
    logger.info("historical_nav_candidate.main >>> started, run_key=%s", args.run_key)
    try:
        directory, completion = run_candidate(read_training_request(args.request), args.run_key)
        logger.info(
            "historical_nav_candidate.main >>> completed, run_key=%s, status=%s, model_hash=%s",
            args.run_key,
            completion["status"],
            completion["model_hash"],
        )
        print("CANDIDATE_RESULT=" + json.dumps({"directory": str(directory), **completion}, ensure_ascii=False))
        return 0 if completion["artifact_persisted"] else 2
    except (SQLAlchemyError, OSError, ValueError, RuntimeError, ArithmeticError, ImportError, RuntimeWarning):
        logger.exception("historical_nav_candidate.main >>> failed, run_key=%s", args.run_key)
        print("CANDIDATE_RESULT=" + json.dumps({"run_key": str(args.run_key), "status": "FAILED_NO_PUBLICATION"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
