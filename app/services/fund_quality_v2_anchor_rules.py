"""候选状态指标的规则锚点；不是经验样本、行业标准或已验收阈值。

完整职责覆盖由有效任职及职位原文证明后，才可能取1；不能为了让经验分位
出现两个取值而虚构职责空档。治理/持有人状态同样先完成原始事实核验。
规则端点只定义状态语义到候选分值的映射，不增加校准产品或数据覆盖。
"""

from copy import deepcopy
from decimal import Decimal

RULE_ANCHOR_VERSION = "QUALITY_RULE_DEFINED_ANCHORS_20261017_1"
_SPECS = {
    "key_person_coverage": {
        "domain": "CONTINUOUS_0_1",
        "points": [["0", "0"], ["1", "100"]],
        "meaning": "有效责任职位的日历覆盖比例；不表示履职质量、能力或治理清白",
    },
    "governance_state": {
        "domain": "ORDINAL_0_1_2",
        "points": [["0", "0"], ["1", "50"], ["2", "100"]],
        "meaning": "已证实重大未整改、非重大或已整改、完整核查无适用不利事项；未知不入包",
    },
    "holder_pressure_band": {
        "domain": "ORDINAL_0_1_2",
        "points": [["0", "100"], ["1", "50"], ["2", "0"]],
        "meaning": "明确全期无20%持有人、存在且无已核验净赎回、存在且有净赎回",
    },
}


def rule_anchor_specs() -> dict:
    """返回冻结候选语义的独立副本；调用方不能修改后续配置或隐藏规则选择。"""
    return {
        "version": RULE_ANCHOR_VERSION,
        "kind": "RULE_DEFINED_NOT_EMPIRICAL_SAMPLE_ENDPOINTS",
        "status": "CANDIDATE_NOT_INDUSTRY_STANDARD_OR_ACCEPTANCE",
        "metrics": deepcopy(_SPECS),
    }


def rule_points(name: str, observed: list[Decimal]) -> list[list[str]]:
    """只核验真实观测的物理/状态定义域，返回规则点；规则点不算观测或产品。"""
    spec = _SPECS[name]
    valid = (
        all(value.is_finite() and 0 <= value <= 1 for value in observed)
        if spec["domain"] == "CONTINUOUS_0_1"
        else all(value.is_finite() and value in {Decimal(0), Decimal(1), Decimal(2)} for value in observed)
    )
    if not valid:
        raise ValueError("QUALITY_V2_RULE_METRIC_DOMAIN:" + name)
    return deepcopy(spec["points"])
