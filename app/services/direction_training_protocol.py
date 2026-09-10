"""有限研究协议；冻结在任何新候选拟合之前，不生成正式发布证据。"""

import platform
import subprocess
from importlib.metadata import version

from app.services.direction_training_artifacts import ROOT, digest, file_hash, read_json, read_seal, seal, write_json
from app.services.direction_training_dataset import END, FUNDS, START, WINDOWS
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.historical_nav_training import PROTOCOL as LOGISTIC

BASELINES = ("ALWAYS_UP", "TRAIN_UP_FREQUENCY", "MOMENTUM_20D", "FIXED_MOMENTUM_SCORE")
CANDIDATES = ("A", "A_CAL", "B", "C")
TREE = dict(
    learning_rate=0.05,
    max_iter=100,
    max_leaf_nodes=7,
    max_depth=3,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)
VERSIONS = {
    "feature": "CASH_REINVESTMENT_FEATURE_V1",
    "label": "CASH_REINVESTMENT_FORWARD_20TD_V1",
    "sample": "DIRECTION_SAMPLE_2021_2024_V2",
    "research": "DIRECTION_TRAINING_V1",
}


def source_fingerprint():
    paths = sorted({*ROOT.glob("app/**/*.py"), *ROOT.glob("scripts/*.py"), ROOT / "pyproject.toml"})
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths})


def runtime():
    return {
        "python": platform.python_version(),
        **{n: version(n) for n in ("numpy", "scipy", "scikit-learn", "sqlalchemy", "pydantic", "threadpoolctl")},
    }


def specification():
    return {
        "protocol_version": "DIRECTION_TRAINING_V1",
        "evaluation_revision": "WINDOW_STRATIFIED_BLOCKS_AND_LOG_LOSS_CLIP_1E8",
        "purpose": "DEVELOPMENT_RESEARCH_ONLY",
        "funds": list(FUNDS),
        "start": str(START),
        "end": str(END),
        "protected_year": 2025,
        "versions": VERSIONS,
        "features": list(FEATURE_NAMES),
        "horizon_sessions": 20,
        "label_rule": "ROUND_HALF_UP_12_DECIMAL_RETURN_STRICTLY_POSITIVE",
        "windows": [dict(name=n, fit_end=f, cal_end=c, exam_end=e) for n, f, c, e in WINDOWS],
        "candidates": list(CANDIDATES),
        "baselines": list(BASELINES),
        "logistic": LOGISTIC.model_dump(mode="json"),
        "tree": TREE,
        "recent_months": 18,
        "calibration": "INDEPENDENT_CAL_SIGMOID_L2_C1_FINITE_SIGN_PRESERVED",
        "minimum": {"FIT": 252, "CAL": 60, "EXAM": 40},
        "threshold": 0.5,
        "answer_isolation": "PER_WINDOW_FIT_CAL_ONLY_ALL_PREDICTIONS_SEALED_BEFORE_EXAM_EXPORT",
        "rolling_reuse": "EARLIER_EXAM_DATES_MAY_BE_MATURE_HISTORY_FOR_LATER_WINDOWS",
        "weighting": "POOLED_NONOVERLAPPING_EXAM_THEN_EQUAL_FUND",
        "minimum_coverage": 0.9,
        "minimum_windows": 3,
        "bootstrap": {
            "sessions": 20,
            "repeats": 2000,
            "seed": 42,
            "confidence": 0.95,
            "minimum_complete_blocks": 5,
            "method": "PAIRED_SYNCHRONIZED_TIME_BLOCKS_WITHIN_WINDOW_NO_GAP_COMPRESSION",
        },
        "budget": {
            "threads": 1,
            "rows": 10000,
            "candidate_seconds": 120,
            "total_seconds": 1800,
            "process_memory_bytes": 4 * 1024**3,
        },
        "publication_status": "MODEL_NOT_RELEASED",
        "database_written": False,
        "test_scored": False,
    }


def freeze(folder):
    coverage = read_seal(folder, "coverage-manifest.json")
    report = read_json(folder / "coverage.json")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    protocol = {
        **specification(),
        "coverage_hash": coverage["manifest_hash"],
        "source_hash": report["source_hash"],
        "source_code_hash": source_fingerprint(),
        "runtime": runtime(),
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "calendar_hash": read_json(folder / "source.json")["calendar_hash"],
    }
    return seal(
        folder,
        "protocol-manifest.json",
        {"protocol.json": write_json(folder / "protocol.json", protocol)},
        status="FROZEN",
        coverage_hash=coverage["manifest_hash"],
    )


def load_protocol(folder, *, check_code=True):
    manifest = read_seal(folder, "protocol-manifest.json")
    coverage = read_seal(folder, "coverage-manifest.json")
    protocol = read_json(folder / "protocol.json")
    if any(protocol.get(k) != v for k, v in specification().items()):
        raise ValueError("PROTOCOL_UNSUPPORTED")
    if protocol["coverage_hash"] != coverage["manifest_hash"] or manifest["coverage_hash"] != coverage["manifest_hash"]:
        raise ValueError("PROTOCOL_COVERAGE_MISMATCH")
    if protocol["source_hash"] != digest(read_json(folder / "source.json")):
        raise ValueError("SOURCE_HASH_MISMATCH")
    if check_code and (protocol["source_code_hash"] != source_fingerprint() or protocol["runtime"] != runtime()):
        raise ValueError("FROZEN_CODE_OR_RUNTIME_CHANGED")
    return protocol, manifest
