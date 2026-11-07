"""未知执行原因与已证实重大未决事项应区分，避免错误阻挡独立业绩。"""

import pytest
from app.services.fund_rating_yy_manufacturing_performance import confirmed_material


@pytest.mark.parametrize(
    ("applicability", "status", "severity", "blocks"),
    [
        ("FUND_CONFIRMED", "UNRESOLVED", "MATERIAL", True),
        ("FUND_CONFIRMED", "UNRESOLVED", "SEVERE", True),
        ("OTHER_ENTITY", "UNRESOLVED", "MATERIAL", False),
        ("FUND_CONFIRMED", "RESOLVED", "SEVERE", False),
        ("FUND_CONFIRMED", "UNRESOLVED", None, False),
        (None, "UNRESOLVED", "MATERIAL", False),
    ],
)
def test_only_fund_confirmed_unresolved_material_events_block(applicability, status, severity, blocks):
    event = {"applicability": applicability, "status": status, "severity": severity}
    assert bool(confirmed_material([event])) is blocks
