"""错误驱动的有限线性模型对照；不增加隐藏层、不更改目标或考试窗口。"""

from app.services.direction_training_artifacts import ROOT, file_hash
from app.services.direction_training_protocol import runtime, source_fingerprint, specification

VERSION = "DIRECTION_LINEAR_REFINEMENT_V1"
SOURCE_RUN = "78607c5c-f35f-48c0-8008-e97a68e86beb"
SOURCE_HASH = "9e2ba7d908fe8387aeb5905003a3c68b755a3ea9630842b33c408f46e38a0589"
BRANCHES = ("REFERENCE", "RECENT_18M", "DROP_60D_GROUP", "PER_FUND")
BASELINES = ("ALWAYS_UP", "ALWAYS_NON_UP", "TRAIN_UP_FREQUENCY", "MOMENTUM_20D", "FIXED_MOMENTUM_SCORE")
FEATURE_INDICES = {b: (0, 1, 3, 6) if b == "DROP_60D_GROUP" else tuple(range(7)) for b in BRANCHES}
ABLATION_VERSION = "DIRECTION_LINEAR_SINGLE_FEATURE_ABLATION_V1"
ABLATION_DROPS = {"DROP_RETURN_60D": 2, "DROP_DRAWDOWN_60D": 4, "DROP_POSITION_60D": 5}
ABLATION_BRANCHES = ("REFERENCE", *ABLATION_DROPS)
ABLATION_INDICES = {b: tuple(i for i in range(7) if i != ABLATION_DROPS.get(b)) for b in ABLATION_BRANCHES}
PRIOR_RUN = "68a6fb3d-bcf1-45df-8f5f-bb56dbae200b"
PRIOR_HASH = "1d353641c767077cc2cdf85a5b4115c7a61a90d544bca881b228bd2a07ecc25f"
RECENCY_VERSION = "DIRECTION_LINEAR_MATURE_RECENCY_V1"
RECENCY_BRANCHES = ("REFERENCE", "MATURE_EXPANDING", "MATURE_SAME_COUNT")


def study_rules(version):
    if version == VERSION:
        return BRANCHES, FEATURE_INDICES
    if version == ABLATION_VERSION:
        return ABLATION_BRANCHES, ABLATION_INDICES
    if version == RECENCY_VERSION:
        return RECENCY_BRANCHES, dict.fromkeys(RECENCY_BRANCHES, tuple(range(7)))
    raise ValueError("LINEAR_STUDY_VERSION")


def fit_boundary(version, branch, window):
    branches, _ = study_rules(version)
    if branch not in branches:
        raise ValueError("LINEAR_BRANCH")
    return window["cal_end"] if version == RECENCY_VERSION and branch != "REFERENCE" else window["fit_end"]


def specification_v1():
    old = specification()
    return {
        **{
            k: old[k]
            for k in (
                "funds",
                "windows",
                "start",
                "end",
                "protected_year",
                "features",
                "minimum",
                "minimum_windows",
                "minimum_coverage",
                "bootstrap",
                "budget",
                "logistic",
            )
        },
        "version": VERSION,
        "evaluation_revision": "EXACT_COUNT_MACRO_ACCURACY_FLOAT_CI_TOLERANCE_1E12",
        "source_run": SOURCE_RUN,
        "source_scored_hash": SOURCE_HASH,
        "architecture": "LOGISTIC_REGRESSION_ZERO_HIDDEN_LAYERS",
        "hidden_layers": 0,
        "input_policy": "NAV_NEXT_SESSION_AVAILABLE_V2_REQUIRE_COMPLETE",
        "label_policy": "EXACT_FORWARD_20_TRADING_SESSIONS_CASH_REINVESTMENT_ROUND12_POSITIVE",
        "branches": list(BRANCHES),
        "baselines": list(BASELINES),
        "changes": {
            "REFERENCE": "UNCHANGED_SEVEN_FEATURE_POOLED_FULL_FIT",
            "RECENT_18M": "ONLY_FIT_CUTOFF_AFTER_FIT_END_MINUS_18_CALENDAR_MONTHS",
            "DROP_60D_GROUP": "ONLY_REMOVE_RETURN60_DRAWDOWN60_POSITION60_KEEP_ORIGINAL_SAMPLE_KEYS",
            "PER_FUND": "ONLY_FIT_EACH_FUND_SEPARATELY_SAME_HISTORY_FEATURES_PARAMETERS",
        },
        "feature_indices": {b: list(v) for b, v in FEATURE_INDICES.items()},
        "threshold": 0.5,
        "calibration": "NONE",
        "parameter_search": False,
        "exam_status": "PREVIOUSLY_OBSERVED_DEVELOPMENT_NOT_INDEPENDENT",
        "selection": {
            "require_no_failed_jobs": True,
            "require_accuracy_ci_lower_above_zero_against": ["REFERENCE", *BASELINES],
            "require_macro_accuracy_above_all_references": True,
            "require_balanced_accuracy_not_worse_than_each_reference": True,
            "minimum_positive_fund_fraction": 2 / 3,
            "minimum_positive_window_fraction": 2 / 3,
            "per_fund_accuracy_delta_min": -0.05,
            "brier_delta_vs_reference_max": 0.01,
            "ranking": "ACCURACY_DESC_BALANCED_ACCURACY_DESC_BRIER_ASC_BRANCH_NAME_ASC",
            "interpretation": "DEVELOPMENT_SCREEN_ONLY_REQUIRES_NEW_INDEPENDENT_EVIDENCE",
        },
        "independent": {
            "historical_period": "2025-01-01/2025-12-31",
            "require_candidate_frozen": True,
            "historical_independence_attested": False,
            "source_first_versions_verified": False,
            "fallback": "REAL_FUTURE_OBSERVATION_AFTER_QUALIFIED_CANDIDATE_FROZEN",
            "future_cutoff_count": 126,
            "forecast_time": "AFTER_20_00_ASIA_SHANGHAI",
            "no_backdated_predictions": True,
            "auto_retrain": False,
        },
        "scope": "OFFLINE_DEVELOPMENT_ONLY_NO_DATABASE_WRITES_OR_MODEL_RELEASE",
        "runtime": runtime(),
        "source_code_hash": source_fingerprint(),
        "dependency_hash": file_hash(ROOT / "requirements.txt"),
    }


def specification_for(version):
    study_rules(version)
    original = specification_v1()
    if version == VERSION:
        return original
    if version == RECENCY_VERSION:
        return {
            **original,
            "version": version,
            "branches": list(RECENCY_BRANCHES),
            "feature_indices": {b: list(range(7)) for b in RECENCY_BRANCHES},
            "changes": {
                "REFERENCE": original["changes"]["REFERENCE"],
                "MATURE_EXPANDING": "ONLY_EXTEND_FIT_TO_LABELS_AVAILABLE_BY_CAL_END_KEEP_START",
                "MATURE_SAME_COUNT": "SHIFT_TO_LATEST_MATURE_ROWS_KEEP_REFERENCE_COUNT_PER_FUND",
            },
            "hypothesis": "TEST_UNUSED_SIX_MONTH_GAP_WITH_EXPANDING_AND_COUNT_MATCHED_RECENCY_CONTROLS",
            "training_boundary": "ANSWER_AVAILABLE_AT_LE_CAL_END_STRICTLY_BEFORE_ALL_EXAM_CUTOFFS",
            "cal_period_role": "NO_CALIBRATOR_MATURE_HISTORY_ONLY_IN_RECENCY_VARIANTS",
            "evaluation_population": "UNCHANGED_SOURCE_READY_WINDOWS_AND_EXACT_EXAM_KEYS",
            "rolling_reuse": "EARLIER_EXAM_MAY_BE_MATURE_TRAINING_FOR_LATER_WINDOWS_DEVELOPMENT_ONLY",
            "research_fit_budget": {"maximum_jobs": 9, "maximum_models": 9, "replay_maximum_models": 9},
            "diagnostic_only": "FIT_UNUSED_GAP_EXAM_METRICS_AND_FEATURE_ASSOCIATIONS_NO_ADAPTIVE_TUNING",
            "audit": "INDEPENDENT_FEATURE_AND_LABEL_ARITHMETIC_SAME_SNAPSHOT_NOT_EXTERNAL_VALUE_ATTESTATION",
        }
    return {
        **original,
        "version": ABLATION_VERSION,
        "hypothesis_source_run": PRIOR_RUN,
        "hypothesis_source_complete_hash": PRIOR_HASH,
        "branches": list(ABLATION_BRANCHES),
        "feature_indices": {b: list(v) for b, v in ABLATION_INDICES.items()},
        "changes": {
            "REFERENCE": original["changes"]["REFERENCE"],
            **{
                b: f"ONLY_REMOVE_{original['features'][i]}_KEEP_ALL_ORIGINAL_SAMPLE_KEYS"
                for b, i in ABLATION_DROPS.items()
            },
        },
        "hypothesis": "IDENTIFY_EFFECT_OF_EACH_60D_COLUMN_AFTER_OBSERVED_GROUP_ABLATION_TRADEOFF",
        "previous_group_role": "OBSERVED_CONTEXT_ONLY_NOT_REFITTED_OR_ELIGIBLE_CANDIDATE",
        "input_history_sessions": 61,
        "feature_search": "THREE_PREDECLARED_LEAVE_ONE_OUT_VARIANTS_NO_ADAPTIVE_COMBINATIONS",
        "research_fit_budget": {"maximum_jobs": 12, "maximum_models": 12, "replay_maximum_models": 12},
    }
