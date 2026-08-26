"""导入用户确认的基金管理人公开页面核验样本。

该命令只写入六只已核验的目录记录和受限来源治理元数据；不访问网站、
不写入日净值、资讯或模型结果。正式全市场同步必须替换为获授权的数据源适配器。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.models.fund import FundMaster, FundShareClass, SourceRegistry

SOURCE_CODE = "MANUAL_PUBLISHER_VERIFIED_SAMPLE"


@dataclass(frozen=True)
class VerifiedFund:
    """一次性人工核验的基金目录字段；不包含净值、持仓或用户身份信息。"""

    fund_code: str
    fund_name: str
    master_name: str
    manager_name: str
    fund_type: str
    share_class: str
    established_date: date | None


VERIFIED_FUNDS: tuple[VerifiedFund, ...] = (
    VerifiedFund(
        "010710", "安信医药健康主题股票C", "安信医药健康主题股票型发起式证券投资基金", "安信基金管理有限责任公司",
        "STOCK", "C", date(2021, 1, 12),
    ),
    VerifiedFund(
        "160323", "华夏磐泰混合（LOF）A", "华夏磐泰混合型证券投资基金（LOF）", "华夏基金管理有限公司",
        "MIXED", "A", date(2016, 12, 26),
    ),
    VerifiedFund(
        "013275", "富国中证煤炭指数C", "富国中证煤炭指数型证券投资基金", "富国基金管理有限公司",
        "INDEX", "C", None,
    ),
    VerifiedFund(
        "007832", "博道伍佰智航股票C", "博道伍佰智航股票型证券投资基金", "博道基金管理有限公司",
        "STOCK", "C", date(2019, 9, 26),
    ),
    VerifiedFund(
        "002112", "德邦鑫星价值灵活配置混合C", "德邦鑫星价值灵活配置混合型证券投资基金", "德邦基金管理有限公司",
        "MIXED", "C", date(2015, 6, 19),
    ),
    VerifiedFund(
        "005312", "万家经济新动能混合C", "万家经济新动能混合型证券投资基金", "万家基金管理有限公司",
        "MIXED", "C", None,
    ),
)


def bootstrap_verified_fund_catalog() -> tuple[int, int]:
    """幂等写入来源登记、基金主实体和份额类别，返回新增主实体与份额数。"""
    created_masters = 0
    created_shares = 0
    with Session(get_engine()) as session, session.begin():
        source = session.scalar(select(SourceRegistry).where(SourceRegistry.source_code == SOURCE_CODE))
        if source is None:
            session.add(
                SourceRegistry(
                    source_code=SOURCE_CODE,
                    display_name="基金管理人公开页面（手工核验样本）",
                    source_kind="MANUAL_IMPORT",
                    license_scope="用户确认的一次性手工目录核验样本；未取得自动同步或批量抓取授权。",
                    rate_limit_per_minute=1,
                    retention_days=365,
                    enabled=False,
                )
            )
        for verified in VERIFIED_FUNDS:
            master = session.scalar(
                select(FundMaster).where(
                    FundMaster.manager_name == verified.manager_name,
                    FundMaster.fund_name == verified.master_name,
                )
            )
            if master is None:
                master = FundMaster(
                    fund_name=verified.master_name,
                    manager_name=verified.manager_name,
                    fund_type=verified.fund_type,
                    status="ACTIVE",
                    established_date=verified.established_date,
                )
                session.add(master)
                session.flush()
                created_masters += 1
            share = session.get(FundShareClass, verified.fund_code)
            if share is None:
                session.add(
                    FundShareClass(
                        fund_code=verified.fund_code,
                        fund_master_id=master.fund_master_id,
                        share_class=verified.share_class,
                fund_name=verified.fund_name,
                fund_type=verified.fund_type,
                status="ACTIVE",
                source_code=SOURCE_CODE,
            )
                )
                created_shares += 1
    return created_masters, created_shares


if __name__ == "__main__":
    masters, shares = bootstrap_verified_fund_catalog()
    print(f"manual_catalog_bootstrap masters_created={masters} shares_created={shares}")
