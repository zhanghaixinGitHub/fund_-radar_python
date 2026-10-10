"""批量检查已保存行情的内容版本；只读本地库，不获取外部行情或个人账目。"""

import hashlib
import json
from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import text

from app.db.session import get_engine
from app.repositories.fund_sync import TUSHARE_SOURCE_CODE
from app.services.trading_calendar import load_current_calendar


class RevisionQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(min_length=1, max_length=64)
    fund_code: str = Field(pattern=r"^[0-9]{6}$")
    start_date: date
    end_date: date
    # SIMULATION 检查结算区间；LABEL 只检查原预测的基准日、目标日与目标日权益事件。
    kind: Literal["SIMULATION", "LABEL"]

    @model_validator(mode="after")
    def valid_range(self):
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        if not self.start_date <= self.end_date <= today or (self.end_date - self.start_date).days > 3660:
            raise ValueError("行情版本查询范围不合法")
        return self


class RevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[RevisionQuery] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def unique_keys(self):
        if len({q.key for q in self.items}) != len(self.items):
            raise ValueError("查询标识不能重复")
        return self


# 由数据库在有界区间内聚合，每个查询只返回一行摘要。使用内容而非更新时间，
# 相同数据重新同步不会导致复盘；旧日期修正、删除、公告日变化均会改变版本。
# 参数绑定的 JSON 表限定 50 项；净值按基金、来源、日期定位，避免逐条 HTTP 查询。
REVISION_SQL = text("""
    WITH requested AS (
      SELECT * FROM jsonb_to_recordset(CAST(:queries AS jsonb))
      AS q(key text, fund_code text, start_date date, end_date date, kind text)
    )
    SELECT q.key, n.version AS nav_version, d.version AS dividend_version,
      (n.ready AND s.enabled AND s.authorized_api_names ? 'fund_nav') AS ready,
      jsonb_build_array(s.source_id,s.enabled,s.authorized_api_names,s.retention_days) AS source_version,
      CASE WHEN q.kind='SIMULATION' THEN jsonb_build_array(
        f.source_code,f.fund_name,r.status,r.dividends_verified_at,r.message) END AS settlement_version
    FROM requested q
    LEFT JOIN source_registry s ON s.source_code=:source
    LEFT JOIN fund_share_class f ON f.fund_code=q.fund_code
    LEFT JOIN simulation_market_refresh r ON r.fund_code=q.fund_code
    LEFT JOIN LATERAL (
      SELECT md5(COALESCE(string_agg(jsonb_build_array(nav_date,unit_nav,accumulated_nav,
                 ann_date,content_hash)::text, '|' ORDER BY nav_date),'')) AS version,
        COALESCE(bool_or(nav_date=q.end_date AND unit_nav>0
                 AND (ann_date IS NULL OR ann_date<=:today)),false) AS ready
      FROM nav_daily WHERE fund_code=q.fund_code AND source_id=s.source_id
        AND nav_date BETWEEN q.start_date AND q.end_date
        AND (q.kind='SIMULATION' OR nav_date IN (q.start_date,q.end_date))
    ) n ON true
    LEFT JOIN LATERAL (
      SELECT md5(COALESCE(string_agg(jsonb_build_array(source_event_key,record_date,ex_date,
                 nav_ex_date,pay_date,cash_dividend,process_status,content_hash)::text,
                 '|' ORDER BY source_event_key),'')) AS version
      FROM fund_dividend WHERE fund_code=q.fund_code AND source_id=s.source_id
        AND (q.kind='SIMULATION' OR ex_date=q.end_date OR nav_ex_date=q.end_date)
    ) d ON true
    ORDER BY q.key
""")


def revision_of(row: dict) -> str:
    """就绪状态也属于版本：未来公告到达可用日期时，未修改的净值仍能触发首次核对。"""
    value = {key: value for key, value in row.items() if key != "key"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def read_revisions(request: RevisionRequest) -> dict:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                REVISION_SQL,
                {
                    "queries": json.dumps([q.model_dump(mode="json") for q in request.items]),
                    "source": TUSHARE_SOURCE_CODE,
                    "today": today,
                },
            )
            .mappings()
            .all()
        )
    return {
        "calendar_revision": load_current_calendar().content_hash,
        "items": [{"key": row["key"], "revision": revision_of(dict(row)), "ready": bool(row["ready"])} for row in rows],
    }
