"""仅内部服务可提交的同步范围；账号来自Java登录上下文，浏览器不能指定。"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MarketSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    owner_user_id: UUID | None = Field(default=None, alias="ownerUserId")
    materials_only: bool = Field(default=False, alias="materialsOnly")
