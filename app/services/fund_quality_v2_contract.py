"""质量 V2 的原始指标契约；公式版本与候选权重一起冻结，不包含校准锚点。

这些是待真实样本验收的评价定义，并非行业标准。比例均以 0—1 表示，
不从第一版分项分数转换；尚无可靠适配的指标明确保留为待解析或待采集。
"""

from copy import deepcopy

from app.services.fund_quality_v2_asset_structure import STRUCTURE_POLICY
from app.services.fund_quality_v2_cash_reconciliation import CASH_RETURN_SOURCE_POLICY
from app.services.fund_rating_inputs import digest

CONTRACT_VERSION = "QUALITY_RAW_20261017_4"

# 同一字段只有一种单位和主要归属；口径变更必须改变契约摘要，旧包不能静默复用。
DEFINITIONS = {
    "relative_return_1y": ("return", "ratio", "12个月", "F末/F初 - B末/B初", "本份额再投资净值、合同基准"),
    "relative_return_3y": (
        "return",
        "ratio/year",
        "36个月",
        "(F末/F初)^(365.25/实际天数) - (B末/B初)^(365.25/实际天数)",
        "本份额再投资净值、合同基准",
    ),
    "extra_drawdown": ("risk", "ratio", "36个月", "max_t(1-(F_t/B_t)/max_s<=t(F_s/B_s))", "本份额再投资净值、合同基准"),
    "extra_downside": (
        "risk",
        "ratio/year^0.5",
        "36个月",
        "sqrt(A/N * sum(min(rF-rB,0)^2))",
        "同日基金与合同基准收益；A为已核验年估值次数",
    ),
    "tracking_error": (
        "efficiency",
        "ratio/year^0.5",
        "36个月",
        "sqrt(A * sum((rF-rB-mean(rF-rB))^2)/(N-1))",
        "同日基金与合同基准收益；不除以各自宽松程度不同的合同上限",
    ),
    "allocation_deviation": (
        "efficiency",
        "ratio",
        "36个月逐次完整披露",
        "各披露日实际资产权重到合同允许集合的最小半L1距离的均值",
        "完整资产配置、同期合同约束；缺一披露日不假定合规",
    ),
    "mandate_deviation": (
        "efficiency",
        "ratio",
        "36个月逐次完整披露",
        "已核验超出合同投资范围资产金额/同期净资产的披露日均值",
        "合同范围、完整穿透资产及合规核验",
    ),
    "rolling_target_success": (
        "stability",
        "ratio",
        "25个重叠12月窗口",
        "sum(相对回报>0记1，=0记0.5，否则0)/25",
        "本份额与合同基准37个月末；重叠窗口不作独立样本",
    ),
    "monthly_target_success": (
        "stability",
        "ratio",
        "36个月",
        "sum(月度相对回报>0记1，=0记0.5，否则0)/36",
        "本份额与合同基准37个月末；零超额目标为候选立场",
    ),
    "governance_state": (
        "management",
        "ordinal:0,1,2",
        "截至考察日及36个月事项",
        "0=已证实重大且未整改；1=非重大或已整改；2=完整核查且未见适用不利事项",
        "官方原文、适用对象、严重性、整改及独立人工复核；未查到不等于2",
    ),
    "key_person_coverage": (
        "management",
        "ratio",
        "36个月",
        "有明确且有效责任人覆盖的自然日数/窗口自然日数",
        "正式职责及任职生效记录；临时代理按正式职责核验，不重复惩罚换任",
    ),
    "team_months": (
        "management",
        "month",
        "当前团队截至考察日",
        "min(36,距最近正式团队变更的自然日数/30.4375)",
        "完整正式团队时间线及原文；任职不替代治理",
    ),
    "issuer_replication_deviation": (
        "holdings",
        "ratio",
        "同一最新完整披露日",
        "0.5*sum(|实际发行人权重-目标发行人权重|)",
        "含现金的全资产穿透权重、同期目标成分、发行人映射",
    ),
    "industry_replication_deviation": (
        "holdings",
        "ratio",
        "同一最新完整披露日",
        "0.5*sum(|完整资产行业及非权益类别权重-合同基准参照同类别权重|)",
        "资产总额作分母，负债单列；同日分类和指数全成分、完整资产账簿；详见industry_structure_policy",
    ),
    "issuer_excess_concentration": (
        "holdings",
        "ratio",
        "同一最新完整披露日",
        "max(0,发行人HHI-预先冻结同目标独立产品参照HHI中位数)",
        "完整发行人敞口及冻结参照；不从本次评分对象反推参照",
    ),
    "industry_excess_concentration": (
        "holdings",
        "ratio",
        "同一最新完整披露日",
        "max(0,行业HHI-预先冻结同目标同分类版本参照HHI中位数)",
        "完整行业敞口、报告日分类及冻结参照",
    ),
    "restricted_ratio": (
        "liquidity",
        "ratio",
        "最新完整报告日；距考察日不超过210天",
        "已核实受限资产金额合计/产品净资产",
        "全资产受限情况及资产负债表；股票之外未知受限情况不补0",
    ),
    "cash_target_deviation": (
        "liquidity",
        "ratio",
        "最新完整报告日",
        "max(合同现金下界-现金比例,0,现金比例-合同现金上界)",
        "报告货币资金、净资产及适用合同现金区间；无区间不猜目标",
    ),
    "holder_pressure_band": (
        "liquidity",
        "ordinal:0,1,2",
        "最近完整报告期",
        "0=明确全期无单一持有人>=20%；1=存在>=20%且无已核验净赎回；2=存在且有净赎回",
        "全期持有人阈值披露及申赎表；区间不当精确比例",
    ),
    "capacity_pressure": (
        "liquidity",
        "ratio",
        "最新报告日及同窗容量核验",
        "max(规模/冻结策略容量上界-1,0)+max(冻结持续运作下界/规模-1,0)",
        "产品规模、策略可交易容量及持续运作边界；不得默认2亿元",
    ),
    "cost_1y": (
        "cost",
        "ratio",
        "考察日适用费率，一万元365天",
        "外扣申购费/(1+申购费)+(1-申购费用比例)*赎回费+综合年运作费率",
        "本份额标准费用；综合运作费不叠加分项，ETF联接按合法减免核验",
    ),
}


def metric_contract() -> dict:
    """输出机器可读的缺失边界，便于逐指标盘点和后续资料适配。"""
    return {
        "version": CONTRACT_VERSION,
        "metrics": {
            key: dict(zip(("dimension", "unit", "window", "formula", "required_sources"), values, strict=True))
            for key, values in DEFINITIONS.items()
        },
        "missing_rule": "任一必要指标缺失时不补零、不重分权、不产生总评级",
        "classification": "产品主代码及主动/增强/被动/跨境主动来自原件；A/C不重复算产品",
        "benchmark": "合同原文、同币种完整同日历史；禁止通用指数替代",
        "industry_structure_policy": deepcopy(STRUCTURE_POLICY),
        "cash_reinvestment_source_policy": deepcopy(CASH_RETURN_SOURCE_POLICY),
        "time": "经济观察日<=考察日；实际取得时间<=冻结时间；历史重建不称事前评分",
        "status": "CANDIDATE_FORMULAS_NOT_EMPIRICALLY_VALIDATED",
    }


def contract_hash() -> str:
    return digest(metric_contract())
