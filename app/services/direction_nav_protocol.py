"""公告可用性和有限缺失实验：六窗、三基金、一个固定模型，不检索超参数。"""

from app.services.direction_nav_data import GAP_POLICY, POLICY
from app.services.direction_training_artifacts import ROOT, file_hash
from app.services.direction_training_protocol import runtime, source_fingerprint, specification

SOURCE_RUN = "feff6919-beca-437a-9b44-5478f61aad44"
SOURCE_HASH = "8934dd2ba0fc610a5cfd45ad4a99d7d86c55abb0cce98a046289c83ed7df0c1c"
VERSION = "DIRECTION_NAV_AVAILABILITY_GAPS_V2"
SCENARIOS = (
    "LEGACY_CLEAN",
    "COMPLETE_CLEAN",
    "COMPLETE_DROP1",
    "COMPLETE_DROP2",
    "TOLERANT_CLEAN",
    "TOLERANT_DROP1",
    "TOLERANT_DROP2",
    "TOLERANT_NATURAL",
    "ALWAYS_UP",
    "ALWAYS_NON_UP",
    "TRAIN_UP_FREQUENCY",
    "MOMENTUM_20D",
    "FIXED_MOMENTUM_SCORE",
)
PAIRS = (
    ("COMPLETE_CLEAN", "LEGACY_CLEAN"),
    ("COMPLETE_DROP1", "COMPLETE_CLEAN"),
    ("COMPLETE_DROP2", "COMPLETE_CLEAN"),
    ("TOLERANT_CLEAN", "COMPLETE_CLEAN"),
    ("TOLERANT_DROP1", "COMPLETE_CLEAN"),
    ("TOLERANT_DROP2", "COMPLETE_CLEAN"),
    ("TOLERANT_DROP1", "COMPLETE_DROP1"),
    ("TOLERANT_DROP2", "COMPLETE_DROP2"),
)


def plan():
    base = specification()
    return {
        **{
            k: base[k]
            for k in (
                "funds",
                "start",
                "end",
                "protected_year",
                "windows",
                "features",
                "horizon_sessions",
                "label_rule",
                "minimum",
                "minimum_coverage",
                "minimum_windows",
                "bootstrap",
                "budget",
                "logistic",
            )
        },
        "version": VERSION,
        "source_run": SOURCE_RUN,
        "source_complete_hash": SOURCE_HASH,
        "availability_policy": POLICY,
        "gap_policy": GAP_POLICY,
        "cutoff": "END_OF_TRADING_DAY_PREVIOUS_SESSION_NAV_ONLY",
        "ann_date": "RAW_PRESERVED_NOT_USED_TO_VETO_NAV_INPUT_OR_MATURE_NAV_LABEL",
        "dividends": "UNCHANGED_ANNOUNCEMENT_AND_IMPLEMENTATION_GATES",
        "historical_first_publication_verified": False,
        "source_scope": "SAVED_AUTHORIZED_2021_2024_SNAPSHOT_ORIGINAL_THREE_FUNDS",
        "gap_treatment": "CAUSAL_PREVIOUS_VALUE_FILL_FEATURE_COPY_ONLY_KEEP_CALENDAR_AND_AUDIT_MASK",
        "protected_indices": [0, 40, 55, 56, 57, 58, 59, 60],
        "event_protection": "KNOWN_EFFECTIVE_DATE_AND_ONE_SESSION_EITHER_SIDE",
        "training_masks": "SHA256_FUND_CUTOFF_MOD3_CHOOSES_0_1_2_NO_LABEL_OR_VALUE_SELECTION",
        "mask_seed": "NAV_GAP_SEED_42",
        "training_comparison": "COMPLETE_AND_TOLERANT_USE_SAME_FIT_KEYS_AND_LABELS_FIXED_SEVEN_FEATURES",
        "mask_is_model_feature": False,
        "scenarios": list(SCENARIOS),
        "comparisons": [list(p) for p in PAIRS],
        "threshold": 0.5,
        "tolerance_gate": {
            "accuracy_ci_lower_min": -0.02,
            "brier_ci_upper_max": 0.01,
            "balanced_accuracy_delta_min": -0.02,
            "per_fund_accuracy_delta_min": -0.05,
            "require_clean_and_both_gap_scenarios": True,
            "interpretation": "EXPLORATORY_NONINFERIORITY_SCREEN_NOT_RELEASE_OR_GENERAL_MISSINGNESS_PROOF",
        },
        "answer_isolation": "ALL_WINDOW_PREDICTIONS_SEALED_BEFORE_EXAM_ANSWER_EXPORT",
        "rolling_reuse": base["rolling_reuse"],
        "publication_status": "MODEL_NOT_RELEASED",
        "test_scored": False,
        "database_written": False,
        "parameter_search": False,
        "source_code_hash": source_fingerprint(),
        "runtime": runtime(),
        "dependency_hash": file_hash(ROOT / "requirements.txt"),
    }
