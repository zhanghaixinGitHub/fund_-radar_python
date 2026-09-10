"""剩余文档任务的有限方案，先冻结队列/特征/校准与不执行条件。"""

from app.services.direction_training_artifacts import ROOT, digest, file_hash
from app.services.direction_training_protocol import runtime, specification

P0_NAME = "direction-training-d3cd9504-4dee-4849-86d5-d33e40a27eb6"
FEATURE_MARKET = ("market_return_20d", "market_volatility_20d", "fund_minus_market_return_20d")


def plan():
    original = specification()
    return {
        "version": "DIRECTION_FOLLOWUP_V1_1",
        "evaluation_revision": "WINDOW_STRATIFIED_BLOCKS_AND_LOG_LOSS_CLIP_1E8",
        "purpose": "DEVELOPMENT_RESEARCH_ONLY",
        "start": original["start"],
        "end": original["end"],
        "protected_year": original["protected_year"],
        "versions": original["versions"],
        "features": original["features"],
        "horizon_sessions": original["horizon_sessions"],
        "label_rule": original["label_rule"],
        "logistic": original["logistic"],
        "baselines": original["baselines"],
        "weighting": original["weighting"],
        "answer_isolation": original["answer_isolation"],
        "rolling_reuse": original["rolling_reuse"],
        "dependency_file_hash": file_hash(ROOT / "requirements.txt"),
        "windows": original["windows"],
        "funds": original["funds"],
        "minimum": original["minimum"],
        "minimum_windows": 3,
        "bootstrap": original["bootstrap"],
        "minimum_coverage": 0.9,
        "budget": original["budget"],
        "cohort_rule": {
            "fund_type": "STOCK",
            "source_fund_type": "股票型",
            "market": "O",
            "established_on_or_before": "2021-01-01",
            "history_first_by": "2021-01-31",
            "history_end_at": "2024-12-31",
            "maximum_funds": 10,
            "order": "ORIGINAL_THEN_CODE_ASC",
            "duplicates": "MASTER_OR_NORMALIZED_PRODUCT_NAME",
            "exclude_foreign_terms": ["QDII", "越南", "恒生", "港股", "纳斯达克", "标普500"],
            "coverage": "ALL_ORIGINAL_READY_WINDOWS_FIT252_CAL60_EXAM40",
        },
        "market": {
            "source": "TUSHARE_PRO_FUND",
            "api": "index_daily",
            "index_code": "000300.SH",
            "expected_name": "沪深300",
            "basis": "PRICE_INDEX",
            "purpose": "SHARED_A_SHARE_MARKET_REFERENCE",
            "not_contractual_fund_benchmark": True,
            "features": list(FEATURE_MARKET),
            "availability_assumption": "PREVIOUS_SESSION_CLOSE_AVAILABLE_BY_NEXT_SESSION_END",
            "relative_return_alignment": "REQUIRE_FUND_ANCHOR_EQUALS_PREVIOUS_SESSION_NO_SHIFT",
            "historical_timestamp_verified": False,
            "years": [2021, 2022, 2023, 2024],
            "max_requests": 4,
            "max_rows_per_request": 366,
            "retry_count": 0,
        },
        "experiments": {
            "T05": ["ORIGINAL_7", "EXPANDED_7"],
            "T06": ["MARKET_MATCHED_7", "MARKET_10"],
            "T07": ["ORIGINAL_7", "CALIBRATED_6M"],
        },
        "direction_reference": "ORIGINAL_7",
        "probability_methods": ["NONE", "SHARED_SIGMOID_L2_C1"],
        "calibration_base_refit": False,
        "threshold": 0.5,
        "parameter_search": False,
        "test_policy": "DO_NOT_READ_2025_WITHOUT_QUALIFIED_FROZEN_CANDIDATE_AND_INDEPENDENCE_EVIDENCE",
        "forward_task": "DESIGN_ONLY_NO_SCHEDULER",
        "publication_status": "MODEL_NOT_RELEASED",
        "runtime": runtime(),
        "source_code_hash": code_hash(),
    }


def code_hash():
    files = sorted({*ROOT.glob("app/**/*.py"), *ROOT.glob("scripts/*.py"), ROOT / "pyproject.toml"})
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in files})
