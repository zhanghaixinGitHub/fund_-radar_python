"""特殊类型公开证据评分的共同边界；类型公式须先独立验证，再逐一启用。

该层不把缺资料改成分数，也不计算各类型的观测指标。类型编译器负责从原件
重放；此处统一保护身份、来源归档、窗口、八维权重、冻结结果和事项门控。
"""

import importlib
import re
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, PASSIVE_BASIS, policy

FIELD = "typed_disclosure_calculation"
REVISION = "TYPED_PUBLIC_DISCLOSURE_ONE_YEAR_20261111_V1"
KEYS = {
    "return": "return",
    "risk": "risk",
    "execution": "efficiency",
    "stability": "stability",
    "management": "management",
    "holdings": "holdings",
    "liquidity": "liquidity",
    "fee": "cost",
}

# 未通过真实同源回放、独立产品及风险反例验证的类型不能正式接入。
# 完整和部分发布分别验收；允许已验证分项并不授予完整类型出分资格。
# 每个开关仅在具体证据与测试通过后启用，不由资料包自行声明。
PROFILES = {
    "GOLD": {
        "family": "INDEX",
        "basis": PASSIVE_BASIS,
        "module": "app.services.fund_rating_gold_disclosure",
        "enabled": True,
    },
    "QDII_ACTIVE": {
        "family": "QDII_EQUITY",
        "basis": ACTIVE_BASIS,
        "module": "app.services.fund_rating_qdii_disclosure",
        "enabled": True,
    },
    "BOND_INDEX": {
        "family": "INDEX",
        "basis": PASSIVE_BASIS,
        "module": "app.services.fund_rating_bond_index_nav_partial",
        "enabled": False,
        "partial_enabled": True,
    },
    "CONVERTIBLE": {
        "family": "CONVERTIBLE",
        "basis": ACTIVE_BASIS,
        "module": "app.services.fund_rating_bond_disclosure",
        "enabled": False,
        "partial_enabled": True,
    },
    "MIXED": {
        "family": "MIXED",
        "basis": ACTIVE_BASIS,
        "module": "app.services.fund_rating_bond_disclosure",
        "enabled": True,
    },
}


def number(value) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError("TYPED_DISCLOSURE_INVALID_NUMBER")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("TYPED_DISCLOSURE_INVALID_NUMBER") from error
    if not result.is_finite():
        raise ValueError("TYPED_DISCLOSURE_INVALID_NUMBER")
    return result


def check_binding(bundle: dict) -> tuple[dict, dict]:
    raw = bundle[FIELD]
    if raw.get("revision") != REVISION or raw.get("profile") not in PROFILES:
        raise ValueError("TYPED_DISCLOSURE_PROFILE")
    spec, inputs = PROFILES[raw["profile"]], raw["inputs"]
    if not spec["enabled"] and not spec.get("partial_enabled", False):
        raise ValueError("TYPED_DISCLOSURE_PROFILE_NOT_VALIDATED")
    if bundle.get("family") != spec["family"] or any(key.endswith("_calculation") and key != FIELD for key in bundle):
        raise ValueError("TYPED_DISCLOSURE_CONFLICTING_METHOD")
    if (
        not re.fullmatch(r"[0-9]{6}", str(bundle.get("fund_code", "")))
        or any(inputs.get(key) != bundle.get(key) for key in ("fund_code", "share_class", "as_of_date", "holding_date"))
        or any(inputs.get(key) != bundle["identity"].get(key) for key in ("legal_name", "manager_name"))
        or inputs.get("primary_ref") != bundle["original_report_archive"]["sha256"]
    ):
        raise ValueError("TYPED_DISCLOSURE_IDENTITY_BINDING")
    if date.fromisoformat(inputs["holding_date"]) > date.fromisoformat(inputs["as_of_date"]):
        raise ValueError("TYPED_DISCLOSURE_REPORT_AFTER_CUTOFF")
    sources = inputs.get("sources", {})
    if (
        not isinstance(sources, dict)
        or inputs["primary_ref"] not in sources
        or any(
            not isinstance(key, str)
            or not re.fullmatch(r"[a-f0-9]{64}", key)
            or not isinstance(ref, dict)
            or ref.get("sha256") != key
            for key, ref in sources.items()
        )
    ):
        raise ValueError("TYPED_DISCLOSURE_SOURCE_BINDING")
    if inputs.get("profile", raw["profile"]) != raw["profile"]:
        raise ValueError("TYPED_DISCLOSURE_PROFILE_BINDING")
    return spec, inputs


def validate_input_archive(bundle: dict, *, root: Path, archive_reader) -> None:
    check_binding(bundle)
    raw = bundle[FIELD]
    if archive_reader(raw["input_archive"], root) != raw["inputs"]:
        raise ValueError("TYPED_DISCLOSURE_INPUT_ARCHIVE_MISMATCH")


def _window(value: dict, as_of: str) -> None:
    ends = value.get("month_ends", [])
    if len(ends) != 13:
        raise ValueError("TYPED_DISCLOSURE_YEAR_WINDOW")
    parsed = [date.fromisoformat(day) for day in ends]
    months = [day.year * 12 + day.month for day in parsed]
    if (
        months != list(range(months[0], months[0] + 13))
        or ends[0] != value.get("from")
        or ends[-1] != value.get("through")
        or value.get("through") != as_of
        or not isinstance(value.get("valuation_points"), int)
        or isinstance(value["valuation_points"], bool)
        or value["valuation_points"] < 13
    ):
        raise ValueError("TYPED_DISCLOSURE_YEAR_WINDOW")


def calculate(bundle: dict, *, root: Path) -> dict:
    spec, inputs = check_binding(bundle)
    result = importlib.import_module(spec["module"]).calculate(inputs, root=root)
    points = result.get("dimensions", {})
    if set(points) != set(KEYS.values()) or not isinstance(result.get("blocked"), bool):
        raise ValueError("TYPED_DISCLOSURE_COMPILER_CONTRACT")
    for value in points.values():
        if value is not None and not 0 <= number(value) <= 100:
            raise ValueError("TYPED_DISCLOSURE_SCORE_RANGE")
    complete = spec["enabled"] and all(value is not None for value in points.values()) and not result["blocked"]
    if complete:
        _window(result.get("facts", {}).get("window", {}), inputs["as_of_date"])
    with localcontext() as context:
        context.prec = 60
        weights = policy(spec["basis"])["weights"]
        total = sum(number(points[key]) * weight / 100 for key, weight in weights.items()) if complete else None
    return {
        **result,
        "score": str(total.quantize(Decimal(".01"))) if total is not None else None,
        "weights": weights,
        "complete_type_validated": spec["enabled"],
        "partial_release_only": not spec["enabled"],
    }


def replay_snapshot(bundle: dict, dimensions: dict, weights: dict, *, root: Path) -> dict | None:
    result = calculate(bundle, root=root)
    for key, internal in KEYS.items():
        if number(weights[key]) != result["weights"][internal]:
            raise ValueError("TYPED_DISCLOSURE_OUTER_WEIGHTS")
        point = result["dimensions"][internal]
        actual = dimensions[key]
        if point is None:
            if actual["candidate"] is not None or number(actual["lower"]) != 0 or number(actual["upper"]) != 100:
                raise ValueError("TYPED_DISCLOSURE_UNKNOWN_BECAME_POINT")
        elif any(
            actual[bound] is None or number(actual[bound]) != number(point) for bound in ("lower", "upper", "candidate")
        ):
            raise ValueError("TYPED_DISCLOSURE_FROZEN_POINT_MISMATCH")
    return result if result["score"] is not None else None


def publication_check(entries: list[dict], *, root: Path) -> None:
    from app.services.fund_market_review import rating_review_required

    for entry in entries:
        if rating_review_required(entry["fund_code"]):
            raise ValueError("TYPED_DISCLOSURE_EVENT_REVIEW_REQUIRED")
        if calculate(entry, root=root)["blocked"]:
            raise ValueError("TYPED_DISCLOSURE_MATERIAL_RISK_UNRESOLVED")


def protect_current_snapshot(previous: dict, following: dict) -> None:
    """资料更新不得把已启用的真实类型换回旧股票模板；显式历史回退另行处理。"""
    old = {
        member["fund_code"]: member["bundle"][FIELD]["profile"]
        for member in previous.get("members", [])
        if member.get("bundle", {}).get(FIELD)
    }
    new = {
        member["fund_code"]: member["bundle"][FIELD]["profile"]
        for member in following.get("members", [])
        if member.get("bundle", {}).get(FIELD)
    }
    if any(new.get(code) != profile for code, profile in old.items()):
        raise ValueError("RATING_TYPED_DISCLOSURE_DOWNGRADE")
