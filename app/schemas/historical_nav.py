"""POST自备净值预览的请求/响应格式；普通GET读库调用不需要手填这些对象。

这里的类是接口数据结构，不是数据库实体。FastAPI先用Pydantic检查请求能否装进这些结构，
格式不合要求就返回422，检查通过才进入计算函数。GET直接返回HistoricalNavSample。
"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.historical_nav_samples import HistoricalNavSample

# 单次POST最多512条，控制同步计算与响应大小；正式批量任务不能绕过此入口硬塞全库。
MAX_PREVIEW_NAV_POINTS = 512
# 统一的净值类型：十进制有限正数，总位数最多24、小数位最多8；允许JSON中用字符串传入。
NavValue = Annotated[Decimal, Field(gt=0, max_digits=24, decimal_places=8, allow_inf_nan=False)]


class HistoricalNavPointRequest(BaseModel):
    """一条净值；公告日或累计净值允许缺失，由计算层给出对应状态。"""

    # extra=forbid拒绝拼错/多余字段；frozen=True防止校验后直接改写对象属性。
    model_config = ConfigDict(extra="forbid", frozen=True)

    # 净值属于哪一天，例如2025-08-07；Pydantic会将YYYY-MM-DD字符串解析成date。
    nav_date: date
    # 这条净值哪天公告，例如2025-08-08；未传或null表示未知，由计算层给出不足原因。
    ann_date: date | None = None
    # 单位净值，必须提供；例如传"1.12280000"，会被解析为Decimal。
    unit_nav: NavValue
    # 累计净值，可缺失。历史窗口和标签必须统一口径，不能逐日混用累计/单位净值。
    accumulated_nav: NavValue | None = None


class HistoricalNavPreviewRequest(BaseModel):
    """一次单基金小批量预览；as_of_date 只筛选返回结果，不是数据可得截止日。"""

    # 请求只接受本类列出的字段；用户资产、个人信息等额外字段不属于这个接口。
    model_config = ConfigDict(extra="forbid", frozen=True)

    # 基金代码必须是6位数字的字符串，如"008888"，用字符串保留前导零。
    fund_code: str = Field(pattern=r"^[0-9]{6}$", min_length=6, max_length=6)
    # 品类固定股票型；省略时默认STOCK，传BOND等其他值会被校验拒绝。
    fund_type: Literal["STOCK"] = "STOCK"
    # 自备数据的来源说明；默认MANUAL_PREVIEW表示手工预览，POST不会去数据库核验该声明。
    source_code: str = Field(default="MANUAL_PREVIEW", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    # 来源同步任务的唯一标识；教学/手工输入没有真实同步记录时应留空，不必编造一个ID。
    source_sync_run_id: UUID | None = None
    # 实际提供的净值列表：至少1条、最多512条。每条都会按上面的HistoricalNavPointRequest校验。
    nav_points: tuple[HistoricalNavPointRequest, ...] = Field(min_length=1, max_length=MAX_PREVIEW_NAV_POINTS)
    # 可选的结果筛选日期，只返回这天的样本；省略就返回全部。它不是“净值公告截止日”。
    as_of_date: date | None = None

    @model_validator(mode="after")
    def validate_nav_dates(self) -> Self:
        """拒绝重复/倒序日期和不在输入序列中的目标日，统一返回 HTTP 422。"""
        # mode=after表示各字段类型校验完后再做跨字段检查，避免拿未解析的日期字符串直接比较。
        # 逐对检查相邻记录；相等是重复日期，大于是倒序，都不能用于数第20条未来净值。
        if any(a.nav_date >= b.nav_date for a, b in zip(self.nav_points[:-1], self.nav_points[1:], strict=True)):
            raise ValueError("nav_points must be strictly ordered by nav_date")
        # 如果明确指定要看哪天，却没有提供那天数据，立即报错，不默默返回另一日期。
        if self.as_of_date is not None and not any(p.nav_date == self.as_of_date for p in self.nav_points):
            raise ValueError("as_of_date must exist in nav_points")
        return self


class HistoricalNavPreviewResponse(BaseModel):
    """纯计算预览结果；计数基于实际返回的 items，不能解释为模型已合格。"""

    # 固定PREVIEW_ONLY：本次只是预览，没有保存样本或发布预测。此响应包用于POST。
    mode: Literal["PREVIEW_ONLY"] = "PREVIEW_ONLY"
    # 请求一共提供多少条原始净值。例如给81条、只看一天，此值仍是81。
    input_nav_count: int
    # 实际返回的样本条数；只看一天时为1，省略日期时通常等于输入条数。
    sample_count: int
    # 返回样本中，特征与标签同时可用的条数；不是预测正确条数。
    scorable_count: int
    # 返回样本中，因历史长度、日期或净值质量等原因无法使用的条数。
    data_insufficient_count: int
    # 返回样本中，已能算特征但未来20条净值/公告还不齐的条数。
    label_not_matured_count: int
    # 样本列表，每个对象的详细字段见HistoricalNavSample；上面三种状态计数之和等于sample_count。
    items: tuple[HistoricalNavSample, ...]
