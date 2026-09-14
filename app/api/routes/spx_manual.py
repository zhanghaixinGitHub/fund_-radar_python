"""标普500按需同步内部接口；浏览器仅经Java权限和CSRF校验后调用。"""

from fastapi import APIRouter, Depends

from app.api.dependencies import require_service_token
from app.services.direction_1d_spx_manual import status, synchronize

router = APIRouter(dependencies=[Depends(require_service_token)])


@router.get("/status")
def read_status():
    """只读本机最近手动回执，不因访问页面触发取数。"""
    return status()


@router.post("/sync")
def manual_sync():
    """固定SPX与服务端当前日期；不接受任意代码、历史日期、目录或时间覆盖。"""
    return synchronize()
