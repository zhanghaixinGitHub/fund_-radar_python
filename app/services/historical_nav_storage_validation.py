"""写入前和读回后核对样本契约；只检查内存对象，不访问数据库。"""

import hashlib
import json
from collections import Counter
from decimal import Decimal

from app.models.historical_nav_sample import HistoricalNavSampleBatch
from app.schemas.historical_nav_storage import HistoricalNavStoredBatch
from app.services.historical_nav_samples import HistoricalNavSample

_METRICS = {
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "max_drawdown_60d",
    "relative_position_60d",
    "consecutive_decline_days",
}
_STATUSES = {"SCORABLE", "DATA_INSUFFICIENT", "LABEL_NOT_MATURED"}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def validate_batch_samples(
    batch: HistoricalNavSampleBatch | HistoricalNavStoredBatch, samples: tuple[HistoricalNavSample, ...]
) -> None:
    """补上数据库单行CHECK不能保证的范围、版本、来源、统计和题目/答案对应关系。"""
    _require(batch.fund_type == "STOCK" and batch.purpose == "LEARNING_ONLY", "invalid batch scope")
    _require(0 <= (batch.end_date - batch.start_date).days < 31, "invalid batch date window")
    dates = [sample.as_of_date for sample in samples]
    _require(len(samples) <= 31 and dates == sorted(set(dates)), "duplicate, unordered or excessive samples")
    counts = Counter(sample.eligibility_status for sample in samples)
    reasons = Counter(sample.unavailable_reason for sample in samples if sample.unavailable_reason is not None)
    _require(set(counts) <= _STATUSES, "unsupported sample status")
    _require(
        (batch.sample_count, batch.scorable_count, batch.data_insufficient_count, batch.label_not_matured_count)
        == (len(samples), counts["SCORABLE"], counts["DATA_INSUFFICIENT"], counts["LABEL_NOT_MATURED"]),
        "summary does not match sample rows",
    )
    _require(batch.unavailable_reasons == dict(reasons), "reason summary does not match sample rows")
    for sample in samples:
        _require(batch.start_date <= sample.as_of_date <= batch.end_date, "sample outside requested range")
        _require(sample.fund_code == batch.fund_code, "mixed funds")
        _require(sample.feature_version == batch.feature_version, "mixed feature versions")
        _require(sample.sample_rule_version == batch.sample_rule_version, "mixed sample rules")
        _require(sample.nav_value_basis in {"UNIT_NAV", "ACCUMULATED_NAV", "UNDETERMINED"}, "invalid NAV basis")
        complete = sample.eligibility_status == "SCORABLE"
        _require((sample.offline_label is not None) == complete, "answer presence does not match sample status")
        _require(sample.unavailable_reason is None if complete else bool(sample.unavailable_reason), "missing reason")
        _validate_payload(batch, sample)
        if sample.eligibility_status != "DATA_INSUFFICIENT":
            _require(sample.available_at is not None and sample.available_at >= sample.as_of_date, "invalid cutoff")
            _require(sample.nav_value_basis != "UNDETERMINED", "missing NAV basis")
            _require(sample.feature_payload["metrics"] is not None, "missing usable metrics")
        label = sample.offline_label
        if label is not None:
            _require(
                label.label_version == batch.label_version and label.horizon_trading_days == 20, "mixed label rules"
            )
            _require(label.label_end_date > sample.as_of_date, "answer ends before anchor")
            _require(label.label_available_at >= label.label_end_date, "invalid answer publication")
            _require(
                sample.available_at is not None and label.label_available_at > sample.available_at,
                "answer was known at feature cutoff",
            )
            _require(
                isinstance(label.future_return_20d, Decimal) and label.future_return_20d.is_finite(),
                "invalid answer numeric type",
            )
            _require(
                label.future_return_20d > -1 and label.label_up_20d == int(label.future_return_20d > 0),
                "answer direction does not match return",
            )


def _validate_payload(batch: HistoricalNavSampleBatch | HistoricalNavStoredBatch, sample: HistoricalNavSample) -> None:
    """只接受构建器的已知字段；未来答案不能以新增键或新增指标混入输入。"""
    payload = sample.feature_payload
    _require(isinstance(payload, dict), "feature payload must be an object")
    required = {"schema_version", "source", "input", "quality", "metrics"}
    _require(required <= set(payload) <= required | {"feature_definitions"}, "unexpected feature fields")
    _require(payload["schema_version"] == batch.feature_version, "payload schema mismatch")
    _require(
        payload["source"]
        == {
            "source_code": batch.source_code,
            "source_sync_run_id": str(batch.source_sync_run_id) if batch.source_sync_run_id else None,
            "nav_value_basis": sample.nav_value_basis,
        },
        "payload source or NAV basis mismatch",
    )
    inputs = payload["input"]
    _require(
        isinstance(inputs, dict)
        and set(inputs)
        <= {
            "as_of_date",
            "available_at",
            "usable_nav_observation_count",
            "minimum_required_nav_observation_count",
        },
        "unexpected feature input fields",
    )
    _require(inputs.get("as_of_date") == sample.as_of_date.isoformat(), "payload anchor mismatch")
    _require(
        inputs.get("available_at") == (sample.available_at.isoformat() if sample.available_at else None),
        "payload cutoff mismatch",
    )
    quality = payload["quality"]
    _require(isinstance(quality, dict) and set(quality) == {"status", "issues"}, "invalid quality fields")
    _require(
        isinstance(quality["issues"], list) and all(isinstance(item, str) for item in quality["issues"]),
        "invalid quality reasons",
    )
    metrics = payload["metrics"]
    if metrics is not None:
        _require(isinstance(metrics, dict) and set(metrics) == _METRICS, "unexpected model input metrics")
        _require(quality["status"] == "SCORABLE", "metrics quality mismatch")
        _require(
            inputs.get("usable_nav_observation_count") == inputs.get("minimum_required_nav_observation_count") == 61,
            "feature history length mismatch",
        )
        definitions = payload.get("feature_definitions")
        _require(isinstance(definitions, dict) and set(definitions) == _METRICS, "metric definitions mismatch")
        for key, value in metrics.items():
            if key == "consecutive_decline_days":
                _require(type(value) is int and 0 <= value <= 60, "invalid decline count")
            else:
                _require(isinstance(value, str) and Decimal(value).is_finite(), "invalid metric numeric type")
    else:
        _require(quality["status"] == "DATA_INSUFFICIENT", "missing metrics without rejection")
    # 与构建器保持相同的UTF-8、键排序和紧凑JSON规则；只计算特征，不包含标签。
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    _require(hashlib.sha256(canonical.encode("utf-8")).hexdigest() == sample.feature_hash, "feature hash mismatch")
