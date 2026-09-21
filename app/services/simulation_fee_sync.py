"""同步中心费率任务：Python 抓取公共档案，Java 原子保存规则，不直连个人账本库。"""

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.simulation_fee import FundFeeProfile
from app.services.eastmoney_fee import fetch_fee_profile

logger = get_logger(__name__)
ProgressReporter = Callable[[int, int, str | None, str], None]


@dataclass(frozen=True)
class FeeSyncResult:
    """统计单位为基金只数；只有 Java 确认保存成功才计入 updated。"""

    total: int
    updated: int
    failures: tuple[str, ...]


class SimulationFeeSyncService:
    """最多四只并发，与旧全量初始化一致；单只失败不阻断其他基金。"""

    def __init__(self) -> None:
        settings = get_settings()
        token = settings.ai_service_token.get_secret_value()
        if not token:
            raise ValueError("费率同步服务令牌未配置")
        self._client = httpx.Client(
            base_url=settings.core_service_base_url.rstrip("/"),
            headers={"X-Service-Token": token},
            timeout=httpx.Timeout(15, connect=3),
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def sync(self, fund_code: str | None = None, *, progress_reporter: ProgressReporter) -> FeeSyncResult:
        """单只指定代码；全量只请求 Java 返回去重代码，不读取用户和持仓金额。"""
        if fund_code is None:
            response = self._client.get("/internal/v1/simulation/fee-sync/fund-codes")
            response.raise_for_status()
            codes = response.json()
        else:
            codes = [fund_code]
        if not isinstance(codes, list) or any(
            not isinstance(code, str) or re.fullmatch(r"[0-9]{6}", code) is None for code in codes
        ):
            raise ValueError("费率同步范围无效")
        codes = sorted(set(codes))
        total, updated, completed = len(codes), 0, 0
        failures = []
        progress_reporter(0, total, None, "正在抓取费率" if codes else "模拟范围内暂无基金，无需同步")
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="fee-fetch") as executor:
            futures = {executor.submit(self._sync_one, code): code for code in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    future.result()
                    updated += 1
                except Exception as error:
                    # 不把原始响应、连接配置或令牌传到任务详情；保留代码和可操作的失败类别。
                    reason = "费率抓取或保存未确认，请重试"
                    if isinstance(error, ValueError) and str(error).startswith("MANUAL_REQUIRED"):
                        reason = "费率结构暂不支持自动解析，请核对规则"
                    failures.append(f"{code}：{reason}")
                    logger.exception("simulation_fee_sync.sync >>> fund failed, fund_code=%s", code)
                completed += 1
                progress_reporter(completed, total, code, f"已处理 {completed}/{total} 只，成功 {updated} 只")
        return FeeSyncResult(total, updated, tuple(failures))

    def _sync_one(self, code: str) -> None:
        """先完整解析，再单次回写；不自动重试 POST，避免网络超时重复更新版本。"""
        profile = FundFeeProfile(**asdict(fetch_fee_profile(code)))
        response = self._client.post(
            "/internal/v1/simulation/fee-sync/profiles", json=profile.model_dump(mode="json", by_alias=True)
        )
        response.raise_for_status()
