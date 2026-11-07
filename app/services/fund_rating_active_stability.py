"""已核发行人原件适配的稳定性入口；不接受外部分数或任意模块路径。"""

from pathlib import Path

from app.services import fund_rating_ef_stability, fund_rating_wj_stability

ADAPTERS = {"006730": fund_rating_wj_stability, "010737": fund_rating_ef_stability}


def calculate(inputs: dict, *, root: Path) -> dict:
    """仅调用已核本份额契约，从原件重建36月及25个滚动年度窗口。"""
    adapter = ADAPTERS.get(inputs.get("fund_code"))
    if adapter is None:
        raise ValueError("R5_ACTIVE_STABILITY_ADAPTER_NOT_VERIFIED")
    return adapter.calculate(inputs, root=root)
