"""比较前冻结研究约定；不读取源净值或任何考试成绩。"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.session import get_nav_sample_storage_engine
from app.models.cash_reinvestment import CashResearchRun
from app.services.cash_exam_plan import build_cash_exam_plan
from app.services.cash_reinvestment_research import FUNDS, VERSIONS
from app.services.historical_nav_calibration import WINDOWS
from app.services.model_comparison_artifacts import file_hash, fingerprint, read_json, write_json

REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
WEIGHT_HASH = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"


def settings() -> dict:
    plan = build_cash_exam_plan()
    return {
        "version": "CHRONOS2_CASH_COMPARISON_V1",
        "purpose": "HISTORICAL_EXPLORATION_ONLY",
        "publication_status": "MODEL_NOT_RELEASED",
        "funds": list(FUNDS),
        "data_versions": VERSIONS,
        "windows": [w.model_dump(mode="json") for w in WINDOWS],
        "planned_cutoffs": {
            w.window_id: [d.isoformat() for d in w.planned_cutoffs]
            for w in plan.windows
            if w.window_id != "INDEPENDENT_TEST_2025"
        },
        "calendar_hash": plan.calendar_hash,
        "model_id": "amazon/chronos-2",
        "revision": REVISION,
        "weight_sha256": WEIGHT_HASH,
        "pretraining_review": {
            "released": "2025-10-20",
            "independence_verified": False,
            "statement": "2023/2024 retrospective results; pretraining overlap not ruled out; 2024 previously observed",
            "sources": ["https://huggingface.co/amazon/chronos-2", "https://arxiv.org/html/2510.15821v1"],
        },
        "history_length": 61,
        "horizon": "20 + anchor_lag_sessions",
        "cross_learning": False,
        "covariates": False,
        "fine_tune": False,
        "dtype": "float32",
        "device": "cpu",
        "raw_score": (
            "(end_q50 - (last_known if lag=0 else step1_q50)) / "
            "max(end_q90-end_q10,1e-8*max(abs(last_known),1))"
        ),
        "quantile_levels": [0.1, 0.5, 0.9],
        "calibration": {
            "method": "sigmoid_L2",
            "C": 1,
            "solver": "lbfgs",
            "tol": 1e-8,
            "max_iter": 1000,
            "seed": 0,
            "weighting": "EQUAL_TOTAL_PER_FUND",
            "minimum_per_fund": 60,
            "reject_nonpositive_slope": True,
        },
        "minimum_fit_per_fund": 252,
        "main_metric": "equal_fund_macro_brier",
        "probability_threshold": 0.5,
        "log_loss_clip": 1e-8,
        "ece_bin_edges": [0, 0.2, 0.4, 0.6, 0.8, 1],
        "baselines": ["ALWAYS_UP", "TRAIN_UP_FREQUENCY", "MOMENTUM_20D", "FIXED_MOMENTUM_SCORE"],
        "bootstrap": {
            "block_sessions": 20,
            "draws": 2000,
            "seed": 42,
            "confidence": 0.95,
            "minimum_disjoint_blocks_per_fund": 5,
            "synchronized_across_funds": True,
            "method": "nonoverlapping fixed calendar blocks; retain gaps; paired Brier B-A",
        },
        "winner_rule": (
            "macro CI excludes 0; >=2 funds same direction; no significant opposite fund; "
            "equal valid coverage; >=5 complete blocks per fund"
        ),
        "budget": {
            "threads": 1,
            "batch_size": 8,
            "single_batch_seconds": 120,
            "total_seconds": 7200,
            "max_process_rss_bytes": 6 * 1024**3,
            "max_batches": 180,
            "max_samples": 5580,
        },
        "replay_tolerance": {"raw_score_absolute": 1e-5, "probability_absolute": 1e-6},
        "protected_period": "No 2025 NAV values, labels, scores or protection edits",
    }


def local_engine():
    engine = get_nav_sample_storage_engine()
    if engine.url.host not in ("127.0.0.1", "localhost") or engine.url.database != "fund_ai":
        raise ValueError("comparison requires the local fund_ai database")
    return engine


def freeze_protocol(folder: Path, source_run: UUID, expected_hash: str, requirements: Path) -> dict:
    # Only the original immutable batch selector is read, never the existing report scores/models.
    with Session(local_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        row = session.execute(
            select(CashResearchRun.dataset_hash, CashResearchRun.report["preparation"]["batch_ids"]).where(
                CashResearchRun.run_id == source_run
            )
        ).one()
        if row[0] != expected_hash:
            raise ValueError("original dataset hash mismatch")
        batch_ids = sorted(str(UUID(value)) for value in row[1])
    if not 1 <= len(batch_ids) <= 180 or len(set(batch_ids)) != len(batch_ids):
        raise ValueError("batch selector out of budget or duplicated")
    result = {
        "settings": settings(),
        "created_at": datetime.now(UTC).isoformat(),
        "source_run_id": str(source_run),
        "source_dataset_hash": expected_hash,
        "batch_ids": batch_ids,
        "requirements_sha256": file_hash(requirements),
        "base_requirements_sha256": file_hash(requirements.parent / "requirements.txt"),
    }
    result["protocol_hash"] = fingerprint(result)
    write_json(folder / "protocol.json", result)
    return result


def validate_protocol(folder: Path) -> dict:
    protocol = read_json(folder / "protocol.json")
    unsigned = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    if fingerprint(unsigned) != protocol["protocol_hash"] or protocol["settings"] != settings():
        raise ValueError("protocol modified or incompatible; create a new version/run")
    root = Path(__file__).resolve().parents[2]
    if (
        file_hash(root / "requirements-chronos2.txt") != protocol["requirements_sha256"]
        or file_hash(root / "requirements.txt") != protocol["base_requirements_sha256"]
    ):
        raise ValueError("dependency protocol changed")
    return protocol
