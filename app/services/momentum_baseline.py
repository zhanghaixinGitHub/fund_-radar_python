"""既有20日固定公式的无副作用实现；分数不等于经过校准的预测概率。"""

from decimal import ROUND_HALF_UP, Decimal


def fixed_momentum_score(return_20d: Decimal) -> Decimal:
    """保持既有0.5+2r、0.05至0.95限幅、四位HALF_UP规则，不拟合任何参数。"""
    if not return_20d.is_finite():
        raise ValueError("return_20d must be finite")
    raw = Decimal("0.5") + return_20d * Decimal("2")
    return min(Decimal("0.9500"), max(Decimal("0.0500"), raw)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
