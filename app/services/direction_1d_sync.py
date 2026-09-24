"""同步中心的一日预测任务：Java 决定关注范围和留档，Python 只协调进度。"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_BASE = "/internal/v1/direction-1d/sync"
_REASONS = {
    "MISSED_DEADLINE": "目标估值日已收盘，等待最新净值后预测下一估值日",
    "NAV_CURRENT_NOT_READY": "当天净值尚未取得，等待净值后预测下一估值日",
    "NAV_LATEST_NOT_READY": "上一估值日净值尚未取得，补齐后再预测",
    "NAV_GAP": "所需历史净值存在缺日，补齐后再预测",
    "WINDOW_CHANGED": "预测期次已变化，请重新创建任务",
    "SPECIAL_POLICY_REQUIRED": "该基金类型暂不支持",
    "NOT_APPLICABLE": "该基金暂不适用现有模型",
    "MODEL_UNAVAILABLE": "模型暂不可用",
    "MODEL_PENDING": "模型尚未准备完成",
    "WAITING_DATA": "净值或必要输入尚未齐全",
    "DATA_PENDING": "净值或必要输入尚未齐全",
    "DATA_INSUFFICIENT": "历史数据不足",
    "NO_LONGER_FOLLOWED": "已不在有效关注范围",
    "QUEUED": "一日预测已排队，尚未确认留档，等待后续检查",
    "RUNNING": "一日预测仍在计算，尚未确认留档，等待后续检查",
}


@dataclass(frozen=True)
class Direction1dSyncResult:
    """单位均为去重后的基金只数；existing 是本期已有留档，不冒充新生成。"""

    target_date: date
    total: int
    created: int
    existing: int
    issues: tuple[str, ...]


class Direction1dSyncService:
    """逐基金执行，失败继续；不重试不确定的 POST，不获取账号或个人持仓信息。"""

    def __init__(self) -> None:
        settings = get_settings()
        token = settings.ai_service_token.get_secret_value()
        if not token:
            raise ValueError("预测同步服务令牌未配置")
        self._client = httpx.Client(
            base_url=settings.core_service_base_url.rstrip("/"),
            headers={"X-Service-Token": token},
            timeout=httpx.Timeout(120, connect=3),
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def sync(self, *, progress_reporter: Callable[[int, int, str | None, str], None]) -> Direction1dSyncResult:
        """冻结当前目标日，分页读取关注代码；跨期和数据不齐均保留原因，不伪造预测值。"""
        target = self.read_target_date()
        return self.sync_codes(self.read_fund_codes(), target=target, progress_reporter=progress_reporter)

    def read_target_date(self) -> date:
        """目标日取自核心服务核验过的一日窗口，不能用本机日期代替交易日。"""
        response = self._client.get(f"{_BASE}/window")
        response.raise_for_status()
        return date.fromisoformat(response.json()["targetNavDate"])

    def read_fund_codes(self) -> list[str]:
        """读取服务端分页去重后的有效关注范围，供一日与多周期使用同一批基金。"""
        codes, after = [], ""
        while True:
            response = self._client.get(f"{_BASE}/fund-codes", params={"after": after})
            response.raise_for_status()
            page = response.json()
            if (
                not isinstance(page, list)
                or len(page) > 100
                or any(not isinstance(code, str) or re.fullmatch(r"[0-9]{6}", code) is None for code in page)
                or page != sorted(set(page))
                or (page and page[0] <= after)
            ):
                raise ValueError("预测基金分页范围无效")
            if not page:
                break
            codes.extend(page)
            after = page[-1]
        return codes

    def sync_codes(
        self,
        codes: list[str],
        *,
        target: date,
        progress_reporter: Callable[[int, int, str | None, str], None],
    ) -> Direction1dSyncResult:
        """仅处理服务端已读取的基金范围；Java仍检查窗口、关注关系及原文留档回执。"""
        total, created, existing = len(codes), 0, 0
        issues = []
        progress_reporter(0, total, None, f"目标日 {target}，正在检查 {total} 只关注基金")
        for index, code in enumerate(codes, 1):
            try:
                response = self._client.post(f"{_BASE}/{code}", json={"targetNavDate": str(target)})
                response.raise_for_status()
                result = response.json()
                if result.get("status") == "PREDICTED":
                    if not isinstance(result.get("reused"), bool):
                        raise ValueError("预测保存回执不完整")
                    existing += int(result["reused"])
                    created += int(not result["reused"])
                else:
                    reason = result.get("reason") or result.get("status")
                    issues.append(f"{code}：{_REASONS.get(reason, '本期预测未生成，请查看预测覆盖详情')}")
            except Exception:
                issues.append(f"{code}：预测生成或留档未确认，请稍后检查或重试")
                logger.exception("direction_1d_sync.sync >>> prediction failed, fund_code=%s", code)
            progress_reporter(
                index,
                total,
                code,
                f"目标日 {target}：已检查 {index}/{total} 只，新生成 {created}，已有 {existing}，未生成 {len(issues)}",
            )
        return Direction1dSyncResult(target, total, created, existing, tuple(issues))
