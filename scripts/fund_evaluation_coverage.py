"""D01 只读盘点：公开目录与资料覆盖，不加载研究标签、不产生评分或修改分类。"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.db.session import get_engine
from app.services.direction_1d_protocol import ZONE
from app.services.fund_exposure_common import save
from sqlalchemy import text


def audit() -> dict:
    """先限定当前启用且来源一致的基金；各资料分开聚合，避免多表连接放大计数。"""
    with get_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        connection.execute(text("SET LOCAL statement_timeout = '15s'"))
        count = connection.execute(text("SELECT count(*) FROM fund_share_class WHERE status='ACTIVE'")).scalar_one()
        if count > 5000:
            raise ValueError("COVERAGE_SCOPE_REQUIRES_PAGED_AUDIT")
        rows = [dict(r) for r in connection.execute(text("""
            WITH scope AS (
                SELECT s.*, r.source_id, p.invest_type, p.benchmark, p.found_date,
                       p.content_hash AS profile_hash, p.updated_at AS profile_received_at
                FROM fund_share_class s JOIN source_registry r ON r.source_code=s.source_code
                LEFT JOIN fund_profile p ON p.fund_code=s.fund_code AND p.source_id=r.source_id
                WHERE s.status='ACTIVE'
            ), nav AS (
                SELECT n.fund_code, min(nav_date) AS first_nav_date, max(nav_date) AS last_nav_date,
                       count(*) AS nav_rows, count(adjusted_nav) AS adjusted_rows,
                       count(*) FILTER (WHERE adjusted_nav<=0 OR unit_nav<=0) AS nonpositive_rows,
                       count(source_published_at) AS timestamped_rows,
                       count(net_asset) AS asset_rows, max(nav_date) FILTER (WHERE net_asset IS NOT NULL) AS asset_date
                FROM nav_daily n JOIN scope s ON s.fund_code=n.fund_code AND s.source_id=n.source_id
                GROUP BY n.fund_code
            ), managers AS (
                SELECT m.fund_code, count(*) AS manager_rows,
                       count(*) FILTER (WHERE end_date IS NULL) AS current_manager_rows,
                       min(begin_date) AS first_recorded_tenure
                FROM fund_manager_assignment m JOIN scope s ON s.fund_code=m.fund_code AND s.source_id=m.source_id
                GROUP BY m.fund_code
            ), dividends AS (
                SELECT d.fund_code, count(*) AS dividend_rows, max(ann_date) AS last_dividend_notice
                FROM fund_dividend d JOIN scope s ON s.fund_code=d.fund_code AND s.source_id=d.source_id
                GROUP BY d.fund_code
            )
            SELECT s.fund_code, s.fund_name, s.fund_type, s.fund_master_id, s.share_class,
                   s.invest_type, s.benchmark, s.found_date, s.profile_hash, s.profile_received_at,
                   n.first_nav_date, n.last_nav_date, n.nav_rows, n.adjusted_rows, n.nonpositive_rows,
                   n.timestamped_rows, n.asset_rows, n.asset_date,
                   m.manager_rows, m.current_manager_rows, m.first_recorded_tenure,
                   d.dividend_rows, d.last_dividend_notice
            FROM scope s LEFT JOIN nav n USING(fund_code) LEFT JOIN managers m USING(fund_code)
            LEFT JOIN dividends d USING(fund_code) ORDER BY s.fund_code
        """)).mappings()]
    types = Counter(r["fund_type"] for r in rows)
    categories = Counter((r["fund_type"], r["invest_type"]) for r in rows)
    # 目录主键存在不等于已经核验产品家族；当前分类无生效区间，不能作为历史同类排名。
    return {"at": datetime.now(ZONE).isoformat(), "read_only": True, "market_share_count": count,
            "market_types": dict(types), "source_categories": [
                {"type": key[0], "invest_type": key[1], "share_count": number}
                for key, number in sorted(categories.items(), key=lambda item: str(item[0]))],
            "eligible_score_categories": [], "total_score_published": False,
            "gaps": ["产品家族关系缺少经核验的来源和生效日期，不能仅凭数据库主键当作已去重。",
                     "当前细分类别没有历史生效区间，须核验同类产品数后才能比较。",
                     "调整后净值字段存在不等于已核验分红再投资收益序列、币种和修订版本。",
                     "经理任期不是总投资管理经验；经理、规模、历史积累的评分分段尚未定稿。",
                     "净资产原值缺少经过本项验收的单位及产品/份额范围，不能拿份额规模替代。"],
            "categories_below_minimum_before_deduplication": [
                {"type": key[0], "invest_type": key[1], "share_count": number}
                for key, number in sorted(categories.items(), key=lambda item: str(item[0])) if number < 30],
            "funds": rows}


if __name__ == "__main__":
    result = audit()
    # UUID/日期统一转字符串后留存哈希包；每次检查新建文件，保留之前的覆盖事实。
    result = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    target = Path("data/fund-insights/coverage") / (datetime.now(ZONE).strftime("%Y%m%d-%H%M%S") + ".json")
    save(target, result)
    print(json.dumps({key: value for key, value in result.items() if key != "funds"}, ensure_ascii=False))
    print(target)
