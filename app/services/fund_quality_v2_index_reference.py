"""重读供应商指数身份与同期成分原件，保留股票切片诊断与完整V2指标的区别。

本模块不联网、不写库，也不从基金名称猜指数代码。调用方负责核验文件哈希、
实际取得时间和来源授权；本模块核对响应字段、指数身份、观察日、成分及总权重。
"""

import re
from decimal import Decimal

from app.services.fund_rating_rules import decimal


def response_rows(original: dict, *, required_fields: set[str], maximum: int) -> list[dict]:
    """响应成功码只是必要条件；每个原始字段、行宽、行数仍需重新核对。"""
    data = original.get("data") or {}
    fields, items = data.get("fields"), data.get("items")
    if original.get("code") != 0 or not isinstance(fields, list) or not isinstance(items, list):
        raise ValueError("QUALITY_INDEX_SOURCE_SCHEMA")
    if len(fields) != len(set(fields)) or not required_fields <= set(fields) or not 0 < len(items) <= maximum:
        raise ValueError("QUALITY_INDEX_SOURCE_FIELDS_OR_COUNT")
    if any(not isinstance(row, list) or len(row) != len(fields) for row in items):
        raise ValueError("QUALITY_INDEX_SOURCE_ROW_WIDTH")
    return [dict(zip(fields, row, strict=True)) for row in items]


def rebuild_index_reference(metadata: dict, weights: dict, *, index_code: str, observed_on: str) -> dict:
    """核对一个真实指数的单日完整股票成分，不将成分当成发行人或行业映射。

    权重单位由供应商定义为百分数，原值除100；允许总和的披露舍入误差，
    保留残差而不重新归一化、截取前十大或把缺失成分补成零。
    """
    identity = response_rows(metadata, required_fields={"ts_code", "name", "publisher"}, maximum=1)[0]
    if identity["ts_code"] != index_code or not identity["name"] or not identity["publisher"]:
        raise ValueError("QUALITY_INDEX_IDENTITY_UNVERIFIED")
    rows = response_rows(weights, required_fields={"index_code", "con_code", "trade_date", "weight"}, maximum=5000)
    target_date = observed_on.replace("-", "")
    by_security = {}
    for row in rows:
        if row["index_code"] != index_code or str(row["trade_date"]) != target_date:
            raise ValueError("QUALITY_INDEX_CODE_OR_OBSERVATION_MISMATCH")
        code = row["con_code"]
        if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}\.(?:SH|SZ|BJ)", code):
            raise ValueError("QUALITY_INDEX_SECURITY_UNSUPPORTED")
        value = decimal(row["weight"])
        if not 0 <= value <= 100 or code in by_security:
            raise ValueError("QUALITY_INDEX_WEIGHT_RANGE_OR_DUPLICATE")
        by_security[code] = value / 100
    total = sum(by_security.values())
    # 百分比合计最多容许0.01个百分点的公开舍入；这不是允许丢失低权重成分。
    if abs(total - 1) > Decimal("0.0001"):
        raise ValueError("QUALITY_INDEX_WEIGHTS_INCOMPLETE")
    return {
        "index_code": index_code,
        "index_name": identity["name"],
        "publisher": identity["publisher"],
        "observed_on": observed_on,
        "constituent_count": len(by_security),
        "weight_sum": str(total),
        "rounding_residual": str(1 - total),
        "weights": {code: str(value) for code, value in sorted(by_security.items())},
        "scope": "INDEX_STOCK_CONSTITUENTS_NOT_ISSUER_OR_INDUSTRY_MAPPING",
    }


def equity_slice_distance(positions: list[dict], reference: dict, *, expected_stock_amount: str) -> dict:
    """真实股票切片与指数成分的半L1距离，仅用于资料/类型诊断，不能直接冒充V2指标。

    分母是已完整对账的股票金额；本计算不包含现金、债券等资产，也没有完成
    同一发行人跨股票/债券/市场合并，所以禁止作为全资产发行人复制偏离计分。
    """
    expected = decimal(expected_stock_amount)
    if expected <= 0 or not positions:
        raise ValueError("QUALITY_INDEX_ACTUAL_STOCKS_REQUIRED")
    actual = {}
    for row in positions:
        code, amount = row["stock_code"], decimal(row["market_value_cny"])
        if code in actual or amount < 0:
            raise ValueError("QUALITY_INDEX_ACTUAL_DUPLICATE_OR_VALUE")
        actual[code] = amount
    total = sum(actual.values())
    if abs(total - expected) > Decimal("0.02"):
        raise ValueError("QUALITY_INDEX_ACTUAL_STOCKS_INCOMPLETE")
    actual = {code: amount / total for code, amount in actual.items()}
    target = {code: decimal(value) for code, value in reference["weights"].items()}
    distance = sum(abs(actual.get(code, 0) - target.get(code, 0)) for code in actual.keys() | target.keys()) / 2
    return {
        "status": "EQUITY_SECURITY_SLICE_DIAGNOSTIC_NOT_V2_METRIC",
        "half_l1_distance": str(distance),
        "actual_stock_amount": str(total),
        "actual_security_count": len(actual),
        "target_security_count": len(target),
        "actual_stock_hhi": str(sum(value * value for value in actual.values())),
        "target_stock_hhi": str(sum(value * value for value in target.values())),
        "limits": ["未含现金及其他资产", "证券代码不是完整发行人映射", "未完成行业分类映射或等级校准"],
    }
