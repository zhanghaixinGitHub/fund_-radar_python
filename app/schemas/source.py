"""受限数据源治理诊断信息的 Pydantic 契约。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SourceDiagnostic(BaseModel):
    """单个已配置数据源的运行元数据，不包含凭据或原始内容。"""

    model_config = ConfigDict(frozen=True)

    source_code: str
    display_name: str
    source_kind: str
    license_scope: str
    rate_limit_per_minute: int
    retention_days: int
    enabled: bool
    last_success_at: datetime | None
    last_error_at: datetime | None
    last_error_summary: str | None
