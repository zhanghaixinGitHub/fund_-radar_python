"""现金分红再投资研究样本：输入和答案独立、公式与边界可读，未取得正式准入。"""

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.trading_nav_window import CalendarProvenance


class CashSampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    fund_code: Literal["001632", "006730", "008888"] = Field(alias="fundCode", description="既有三只股票型试点")
    cutoff_date: date = Field(
        alias="cutoffDate",
        ge=date(2022, 1, 1),
        le=date(2024, 12, 31),
        description="信息截止日，按自然日结束解释；暂不开放2025样本数值",
    )


class CashSampleIssue(BaseModel):
    day: date | None = Field(description="缺失/冲突对应日，未知有效日时可为空")
    code: str = Field(description="稳定拒收原因，不是补数指令")


class CashReturnPoint(BaseModel):
    nav_date: date = Field(description="精确交易日")
    unit_nav: str = Field(description="该日单位净值，固定12位小数字符串")
    cash_per_share: str = Field(description="该日计入的每份现金分红；序列基准日不计入分红")
    daily_return: str | None = Field(description="(当日单位净值+现金)/前日单位净值-1；基准日为null")
    growth_index: str = Field(description="从100起步逐日连乘得到的研究净值，不是来源复权列")
    available_at: date = Field(description="截至该点整段序列所需净值/已使用分红的最晚公告日")
    dividend_event_keys: tuple[str, ...] = Field(description="当前日实际计入的来源事件标识，基准日/无事件日为空")


class CashFeaturePayload(BaseModel):
    feature_version: Literal["CASH_REINVESTMENT_FEATURE_V1"] = Field(
        default="CASH_REINVESTMENT_FEATURE_V1", description="新版独立输入指标版本"
    )
    nav_value_basis: Literal["CASH_REINVESTED_UNIT_NAV_V1"] = Field(
        default="CASH_REINVESTED_UNIT_NAV_V1", description="按单位净值及已知现金构建，不读取累计或供应商复权列"
    )
    fund_code: str = Field(description="基金标识，仅用于溯源，不是指标列")
    cutoff_date: date = Field(description="按当天结束筛选已知信息")
    anchor_nav_date: date = Field(description="截止时已公告的最新交易日净值日")
    available_at: date = Field(description="输入所需信息最晚公告日，必须<=cutoff")
    source_code: str = Field(description="共同启用来源")
    source_sync_run_id: UUID = Field(description="当前同步水位，不能证明历史首次版本")
    calendar_version: str = Field(description="固定交易日历版本")
    calendar_hash: str = Field(description="固定日历指纹，便于同版本复现")
    history_series: tuple[CashReturnPoint, ...] = Field(description="61个精确交易日的已知研究序列，含60段相邻变化")
    metrics: dict[str, str | int] = Field(
        description="沿用七项纯指标公式，改为输入本序列；连续下跌次数为整数，其余为8位小数字符串"
    )


class CashDirectionLabel(BaseModel):
    label_version: Literal["CASH_REINVESTMENT_FORWARD_20TD_V1"] = Field(
        default="CASH_REINVESTMENT_FORWARD_20TD_V1", description="独立答案版本，不能与旧累计净值标签混用"
    )
    label_base_date: date = Field(description="截止日及以前最近交易日；是答案的分母日，不强行使用更早的已知输入起点")
    label_end_date: date = Field(description="严格在cutoff之后第20个交易日")
    horizon_trading_days: Literal[20] = Field(default=20, description="恰好20段交易日收益，非自然日/源记录数")
    label_available_at: date = Field(description="21个所需净值点及计入分红的最晚公告日，后续训练还须按此隔离")
    future_return_20d: str = Field(description="现金日收益连乘减1，固定12位小数；0.01表示1%")
    label_up_20d: Literal[0, 1] = Field(description="按返回的12位收益判断：大于0为1，持平/下跌为0，避免显示0却判涨")
    label_series: tuple[CashReturnPoint, ...] = Field(description="独立的基准日+未来20日研究序列，仅作答案，不进入特征")


class CashSampleResponse(BaseModel):
    mode: Literal["READ_ONLY_CASH_REINVESTMENT_SAMPLE_PREVIEW"] = Field(
        default="READ_ONLY_CASH_REINVESTMENT_SAMPLE_PREVIEW", description="新版研究样本只读预览，不保存、不训练"
    )
    sample_rule_version: Literal["CASH_REINVESTMENT_SAMPLE_RULE_V1"] = Field(
        default="CASH_REINVESTMENT_SAMPLE_RULE_V1", description="独立样本选择/隔离规则"
    )
    status: Literal["RESEARCH_SAMPLE_READY", "INPUT_UNAVAILABLE", "LABEL_UNAVAILABLE"] = Field(
        description="当前样本计算状态，不等于正式训练或发布资格"
    )
    fund_code: str = Field(description="当前试点")
    cutoff_date: date = Field(description="用户传入的信息截止日")
    calendar: CalendarProvenance = Field(
        description="固定交易日与年度公告证据；跨年安排若当时未知，保留特征但不生成标签"
    )
    anchor_nav_date: date | None = Field(description="模型输入所用最新已知净值日")
    anchor_lag_sessions: int | None = Field(description="最新已知净值落后截至日最近交易日的天数，最多1天")
    label_base_date: date = Field(description="答案计算的基准交易日，可能尚未公告，所以不放进模型输入")
    label_end_date: date = Field(description="预先固定的未来第20个交易日，缺数不顺延")
    history_dates: tuple[date, ...] = Field(description="输入要求的精确61日，起点不可用时为空")
    future_dates: tuple[date, ...] = Field(description="严格晚于cutoff的20个交易日")
    input_issues: tuple[CashSampleIssue, ...] = Field(description="只由当时已知输入决定的拒收原因")
    label_issues: tuple[CashSampleIssue, ...] = Field(description="答案缺失或不适用原因，不回写输入")
    feature_payload: CashFeaturePayload | None = Field(
        description="模型输入及可复算路径；没有任何未来值、标签或未来质量标记"
    )
    feature_hash: str | None = Field(description="仅对feature_payload生成的哈希，未来答案变化不应改变它")
    offline_label: CashDirectionLabel | None = Field(description="后来答案，与feature_payload独立；不是预测概率")
    label_hash: str | None = Field(description="仅对答案生成的内容指纹，不写进特征")
    ignored_non_trading_nav_dates: tuple[date, ...] = Field(
        description="整个读取范围的非交易日源记录，仅作诊断，未删除"
    )
    usage: Literal["LEARNING_ONLY"] = Field(default="LEARNING_ONLY", description="仅允许研究检查，不自动导入旧训练器")
    training_eligible: Literal[False] = Field(default=False, description="事件完整性及历史首次版本等尚未通过正式准入")
    historical_versions_verified: Literal[False] = Field(default=False, description="当前源快照不是首次公开版本档案")
    dividend_history_complete_verified: Literal[False] = Field(default=False, description="无记录不能证明没有事件")
    database_written: Literal[False] = Field(default=False, description="没有保存或覆盖批次、标签、源数据")
    model_fitted: Literal[False] = Field(default=False, description="没有训练或校准模型")
    test_scored: Literal[False] = Field(default=False, description="没有使用2025测试成绩")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(
        default="MODEL_NOT_RELEASED", description="未发布用户可见预测"
    )
    limitations: tuple[str, ...] = Field(description="现金再投假设、事件完整性、时点版本、市场和非发布限制")
