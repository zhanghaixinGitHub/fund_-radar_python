"""错误驱动的有限线性模型对照；不增加隐藏层、不更改目标或考试窗口。"""

from datetime import date

from app.services.direction_training_artifacts import ROOT, file_hash
from app.services.direction_training_dataset import END, WINDOWS, exam_dates
from app.services.direction_training_protocol import runtime, source_fingerprint, specification
from app.services.trading_calendar import load_calendar

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
REGULARIZATION_VERSION = "DIRECTION_LINEAR_REGULARIZATION_V1"
REGULARIZATION_C = {"REFERENCE": 1.0, "L2_STRONG": 0.1, "L2_STRONGER": 0.01}
REGULARIZATION_BRANCHES = tuple(REGULARIZATION_C)
RECENCY_RUN = "54d65d24-acba-4e1e-8747-4eed4cf4492a"
RECENCY_HASH = "49254bbea5e48e113a7147ba80ff0284e11ed5e30ecbf4ff9891b62b143e0bee"
COMBINATION_VERSION = "DIRECTION_LINEAR_FEATURE_REGULARIZATION_V1"
COMBINATION_C = {"REFERENCE": 1.0, "DROP_60D_GROUP": 1.0, "L2_STRONGER": 0.01, "DROP_60D_GROUP_L2": 0.01}
COMBINATION_BRANCHES = tuple(COMBINATION_C)
COMBINATION_INDICES = {
    b: (0, 1, 3, 6) if b.startswith("DROP_60D_GROUP") else tuple(range(7)) for b in COMBINATION_BRANCHES
}
REGULARIZATION_RUN = "f5905e47-097f-4df7-bf85-3c3fdbc64c16"
REGULARIZATION_HASH = "1a6dc576e79dcb30a04df0a226222dd7a7469523e9401b188d4e87c796f2bacd"
COVERAGE_VERSION = "DIRECTION_LINEAR_FULL_QUARTERS_V1"
COVERAGE_BRANCHES = ("REFERENCE", "DROP_60D_GROUP_L2")
MARKET_VERSION = "DIRECTION_MATCHED_MARKET_V1"
MARKET_BRANCHES = ("REFERENCE", "SHARED_MARKET", "MATCHED_MARKET")
VOLUME_VERSION = "DIRECTION_VOLUME_INTERACTION_V1"
VOLUME_BRANCHES = ("REFERENCE", "AMOUNT_ACTIVITY", "AMOUNT_INTERACTION")
ALGORITHM_VERSION = "DIRECTION_ALGORITHM_VOLUME_V1"
ALGORITHM_BRANCHES = ("REFERENCE", "AMOUNT_ACTIVITY", "TREE_NAV", "TREE_AMOUNT")
BREADTH_VERSION = "DIRECTION_MARKET_BREADTH_V1"
BREADTH_BRANCHES = ("REFERENCE", "MARKET_BREADTH")
ROLLING_VERSION = "DIRECTION_ROLLING_UPDATE_V1"
ROLLING_BRANCHES = ("REFERENCE", "QUARTER_ALL", "QUARTER_252", "MONTH_ALL", "MONTH_252")
ETF_SHARE_VERSION = "DIRECTION_ETF_SHARE_CHANGE_V1"
ETF_SHARE_BRANCHES = ("REFERENCE", "ETF_SHARE_CHANGE")
FULL_QUARTER_VERSIONS = (
    COVERAGE_VERSION,
    MARKET_VERSION,
    VOLUME_VERSION,
    ALGORITHM_VERSION,
    BREADTH_VERSION,
    ROLLING_VERSION,
    ETF_SHARE_VERSION,
)
COMBINATION_RUN = "0bf0104f-9f25-4787-901c-f71dddc92e35"
COMBINATION_HASH = "887ae19234d1dfdb655ed6158afbde8602dd71ddfa2d4ea19b618108eaec1f24"


def planned_dates(version, window):
    lower, upper = (date.fromisoformat(window[k]) for k in ("cal_end", "exam_end"))
    if version not in FULL_QUARTER_VERSIONS:
        return exam_dates(lower, upper)[1]
    calendar = load_calendar()
    return [d for d in calendar.sessions if lower < d <= upper and calendar.future_sessions(d, 21)[-1] <= END]


def evaluation_asof(window):
    return window.get("label_asof", window["exam_end"])


def study_windows(version):
    if version not in FULL_QUARTER_VERSIONS:
        return [dict(name=n, fit_end=f, cal_end=c, exam_end=e) for n, f, c, e in WINDOWS]
    result = []
    for year in (2023, 2024):
        bounds = (date(year - 1, 12, 31), date(year, 3, 31), date(year, 6, 30), date(year, 9, 30), date(year, 12, 31))
        fits = (date(year - 1, 6, 30), date(year - 1, 9, 30), date(year - 1, 12, 31), date(year, 3, 31))
        for q in range(4):
            window = dict(
                name=f"DEV_{year}_Q{q + 1}_FULL_V1",
                fit_end=str(bounds[q] if version == ETF_SHARE_VERSION else fits[q]),
                cal_end=str(bounds[q]),
                exam_end=str(bounds[q + 1]),
            )
            dates = planned_dates(version, window)
            window["label_asof"] = str(load_calendar().future_sessions(dates[-1], 21)[-1])
            result.append(window)
    return result


def study_rules(version):
    if version == ETF_SHARE_VERSION:
        return ETF_SHARE_BRANCHES, {"REFERENCE": tuple(range(7)), "ETF_SHARE_CHANGE": tuple(range(8))}
    if version == VERSION:
        return BRANCHES, FEATURE_INDICES
    if version == ABLATION_VERSION:
        return ABLATION_BRANCHES, ABLATION_INDICES
    if version == RECENCY_VERSION:
        return RECENCY_BRANCHES, dict.fromkeys(RECENCY_BRANCHES, tuple(range(7)))
    if version == REGULARIZATION_VERSION:
        return REGULARIZATION_BRANCHES, dict.fromkeys(REGULARIZATION_BRANCHES, tuple(range(7)))
    if version == COMBINATION_VERSION:
        return COMBINATION_BRANCHES, COMBINATION_INDICES
    if version == COVERAGE_VERSION:
        return COVERAGE_BRANCHES, {b: COMBINATION_INDICES[b] for b in COVERAGE_BRANCHES}
    if version == MARKET_VERSION:
        return MARKET_BRANCHES, {b: tuple(range(7 if b == "REFERENCE" else 10)) for b in MARKET_BRANCHES}
    if version == VOLUME_VERSION:
        return VOLUME_BRANCHES, {b: tuple(range(7 + i)) for i, b in enumerate(VOLUME_BRANCHES)}
    if version == ALGORITHM_VERSION:
        return ALGORITHM_BRANCHES, {b: tuple(range(7 + i % 2)) for i, b in enumerate(ALGORITHM_BRANCHES)}
    if version == BREADTH_VERSION:
        return BREADTH_BRANCHES, {b: tuple(range(7 + i)) for i, b in enumerate(BREADTH_BRANCHES)}
    if version == ROLLING_VERSION:
        return ROLLING_BRANCHES, dict.fromkeys(ROLLING_BRANCHES, tuple(range(7)))
    raise ValueError("LINEAR_STUDY_VERSION")


def fit_boundary(version, branch, window):
    branches, _ = study_rules(version)
    if branch not in branches:
        raise ValueError("LINEAR_BRANCH")
    return window["cal_end"] if version == RECENCY_VERSION and branch != "REFERENCE" else window["fit_end"]


def regularization_c(version, branch):
    branches, _ = study_rules(version)
    if branch not in branches:
        raise ValueError("LINEAR_BRANCH")
    if version in (COMBINATION_VERSION, COVERAGE_VERSION):
        return COMBINATION_C[branch]
    return REGULARIZATION_C[branch] if version == REGULARIZATION_VERSION else 1.0


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
    if version in (
        MARKET_VERSION,
        VOLUME_VERSION,
        ALGORITHM_VERSION,
        BREADTH_VERSION,
        ROLLING_VERSION,
        ETF_SHARE_VERSION,
    ):
        raise ValueError("MARKET_STUDY_REQUIRES_MAPPING_AND_DEDICATED_FREEZE")
    original = specification_v1()
    if version == VERSION:
        return original
    if version == COVERAGE_VERSION:
        return {
            **original,
            "version": version,
            "windows": study_windows(version),
            "branches": list(COVERAGE_BRANCHES),
            "feature_indices": {b: list(COMBINATION_INDICES[b]) for b in COVERAGE_BRANCHES},
            "C_by_branch": {b: COMBINATION_C[b] for b in COVERAGE_BRANCHES},
            "penalty": "L2",
            "changes": {
                "REFERENCE": "LOCKED_SEVEN_FEATURE_C1_CONTROL",
                "DROP_60D_GROUP_L2": "LOCKED_FOUR_FEATURE_C001_RULE_FROM_PRIOR_COMBINATION",
            },
            "hypothesis_sources": {COMBINATION_RUN: COMBINATION_HASH},
            "parameter_search": False,
            "search_policy": "NO_NEW_COMBINATIONS_C_FEATURES_OR_THRESHOLDS",
            "fit_policy": "ORIGINAL_SIX_MONTH_GAP_QUARTERLY_FIT_END_RULE_UNCHANGED",
            "exam_policy": "ALL_QUARTER_CUTOFFS_WITH_LABEL_AVAILABLE_BY_2024_END",
            "label_evaluation_policy": "ALLOW_CROSS_QUARTER_TARGETS_BY_FIXED_LABEL_ASOF_NOT_TRAINING_INPUT",
            "old_overlap_role": "PARITY_AND_DESCRIPTIVE_ONLY_NEVER_INDEPENDENT_TEST",
            "research_fit_budget": {"maximum_jobs": 16, "maximum_models": 16, "replay_maximum_models": 16},
            "stop_policy": "ONE_FIXED_COVERAGE_BATCH_NO_ADAPTIVE_DATES_OR_PARAMETERS",
        }
    if version == COMBINATION_VERSION:
        return {
            **original,
            "version": version,
            "branches": list(COMBINATION_BRANCHES),
            "feature_indices": {b: list(v) for b, v in COMBINATION_INDICES.items()},
            "C_by_branch": dict(COMBINATION_C),
            "penalty": "L2",
            "changes": {
                "REFERENCE": original["changes"]["REFERENCE"],
                "DROP_60D_GROUP": original["changes"]["DROP_60D_GROUP"],
                "L2_STRONGER": "ONLY_REDUCE_C_FROM_1_TO_0_01",
                "DROP_60D_GROUP_L2": "REMOVE_60D_GROUP_AT_FIXED_C_0_01",
            },
            "hypothesis_sources": {PRIOR_RUN: PRIOR_HASH, REGULARIZATION_RUN: REGULARIZATION_HASH},
            "hypothesis": "TEST_ONE_COMBINATION_AFTER_GROUP_TRADEOFF_AND_L2_GAIN_NOT_ASSUMED_ADDITIVE",
            "new_hypotheses": ["DROP_60D_GROUP_L2"],
            "control_parity_required": ["REFERENCE", "DROP_60D_GROUP", "L2_STRONGER"],
            "parameter_search": True,
            "search_policy": "FIXED_TWO_BY_TWO_FOUR_CELLS_ONLY_ONE_NEW_COMBINATION_NO_ADAPTIVE_SEARCH",
            "fit_policy": "ORIGINAL_FULL_FIT_AND_FIT_END_IDENTICAL_SAMPLE_KEYS",
            "research_fit_budget": {"maximum_jobs": 12, "maximum_models": 12, "replay_maximum_models": 12},
            "stop_policy": "CLOSE_CURRENT_SNAPSHOT_COMBINATION_SEARCH_AFTER_THIS_BATCH_REGARDLESS_OF_RESULT",
            "next_research": "BROADER_DEVELOPMENT_TIME_COVERAGE_REQUIRES_SEPARATE_FIXED_PLAN",
            "diagnostic_only": "D_VS_C_FEATURE_EFFECT_D_VS_B_PENALTY_EFFECT_AND_INTERACTION_NO_NEW_GATES",
        }
    if version == REGULARIZATION_VERSION:
        return {
            **original,
            "version": version,
            "branches": list(REGULARIZATION_BRANCHES),
            "feature_indices": {b: list(range(7)) for b in REGULARIZATION_BRANCHES},
            "C_by_branch": dict(REGULARIZATION_C),
            "penalty": "L2",
            "changes": {
                "REFERENCE": original["changes"]["REFERENCE"],
                "L2_STRONG": "ONLY_REDUCE_C_FROM_1_TO_0_1",
                "L2_STRONGER": "ONLY_REDUCE_C_FROM_1_TO_0_01",
            },
            "hypothesis_source_run": RECENCY_RUN,
            "hypothesis_source_complete_hash": RECENCY_HASH,
            "hypothesis": "TEST_STRONGER_L2_AFTER_OBSERVED_TEMPORAL_INSTABILITY_NOT_PROVEN_OVERFITTING",
            "parameter_search": True,
            "search_policy": "EXACTLY_THREE_PREDECLARED_C_VALUES_NO_ADAPTIVE_VALUES_OR_COMBINATIONS",
            "fit_policy": "ORIGINAL_FULL_FIT_AND_FIT_END_NO_RECENCY_OR_FEATURE_CHANGES",
            "research_fit_budget": {"maximum_jobs": 9, "maximum_models": 9, "replay_maximum_models": 9},
            "diagnostic_only": "FIT_VS_EXAM_METRICS_AND_STANDARDIZED_COEFFICIENT_NORMS_NOT_CAUSAL_PROOF",
        }
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
