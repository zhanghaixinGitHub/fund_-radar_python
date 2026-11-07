"""同类比较专用的已核实分类快照；与评级、预测和个人账户资料隔离。"""

import logging
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import get_settings

logger = logging.getLogger(__name__)
CLASSIFICATION_FILE = Path(__file__).resolve().parents[1] / "data/fund_peer_classifications_v1.json"
REQUIRED_CRITERIA = frozenset({
    "fund_type", "management", "market", "currency", "allocation_policy", "sector_policy",
    "reference_id", "fx_policy", "vehicle",
})


class PeerSource(BaseModel):
    """原件指纹及核对摘录；文件相对评级原件目录，页面不返回这些内部字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    file: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    url: str
    title: str
    published_on: date
    excerpts: tuple[str, ...] = Field(min_length=1, max_length=12)


class PeerClassification(BaseModel):
    """一只实际份额的分类事实；各条件全部一致才进入同一比较组。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    fund_code: str = Field(pattern=r"^[0-9]{6}$")
    product_id: str = Field(min_length=1, max_length=128)
    share_class: str = Field(min_length=1, max_length=16)
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=600)
    criteria: dict[str, str]
    effective_from: date
    effective_to: date | None = None
    revoked: bool = False
    sources: tuple[PeerSource, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def complete_evidence(self):
        """不允许把缺失维度组成未知类别，也不接受没有原件身份的独立产品。"""
        if set(self.criteria) != REQUIRED_CRITERIA or any(
            not value.strip() or value.upper() in {"UNKNOWN", "UNVERIFIED"} for value in self.criteria.values()
        ):
            raise ValueError("peer criteria incomplete")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("peer classification interval reversed")
        if any(source.published_on > self.effective_from for source in self.sources):
            raise ValueError("peer classification predates source")
        return self

    def covers(self, start: date, end: date) -> bool:
        """有效期必须覆盖整个统计窗口，不能用窗口末尾才生效的类别解释历史。"""
        return (not self.revoked and self.effective_from <= start <= end
                and (self.effective_to is None or end <= self.effective_to))

    def same_group(self, other: "PeerClassification") -> bool:
        """比较全部固定条件；名称、近期收益和评级不参与匹配。"""
        return self.criteria == other.criteria


class PeerSnapshot(BaseModel):
    """发布的不可变分类版本；复核有效期是运维边界，不冒充合同到期日。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str
    reviewed_on: date
    review_expires_on: date | None = None
    records: tuple[PeerClassification, ...] = Field(max_length=5000)
    gaps: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_codes(self):
        codes = [record.fund_code for record in self.records]
        if (len(set(codes)) != len(codes)
                or (self.review_expires_on is not None and self.review_expires_on < self.reviewed_on)):
            raise ValueError("peer snapshot duplicate codes or invalid review interval")
        return self


@lru_cache(maxsize=4)
def _verified_snapshot(path: str, modified_ns: int, source_root: str) -> PeerSnapshot:
    """每个快照版本首次加载核对原件指纹；只读本地文件，不在请求中解析 PDF 或采集。"""
    source_path = Path(path)
    if source_path.stat().st_size > 2_000_000:
        raise ValueError("peer snapshot exceeds size limit")
    packet = PeerSnapshot.model_validate_json(source_path.read_bytes())
    root = Path(source_root).resolve()
    verified: set[str] = set()
    for record in packet.records:
        for ref in record.sources:
            if ref.sha256 in verified:
                continue
            original = (root / ref.file).resolve()
            from app.services.material_content import content_digest, source_bytes, source_size

            if not original.is_relative_to(root / "sources") or source_size(original) > 30_000_000:
                raise ValueError("peer original path or size invalid")
            if content_digest(source_bytes(original)) != ref.sha256:
                raise ValueError("peer original hash mismatch")
            verified.add(ref.sha256)
    return packet


def load_peer_classifications(today: date | None = None) -> dict[str, PeerClassification]:
    """损坏、缺文件、未来复核或复核过期时整体关闭严格比较，不退回大类凑数。"""
    today = today or datetime.now(ZoneInfo("Asia/Shanghai")).date()
    try:
        source_root = Path(get_settings().fund_insight_directory) / "ratings"
        packet = _verified_snapshot(str(CLASSIFICATION_FILE), CLASSIFICATION_FILE.stat().st_mtime_ns, str(source_root))
        if today < packet.reviewed_on or (packet.review_expires_on is not None and today > packet.review_expires_on):
            return {}
        return {record.fund_code: record for record in packet.records if not record.revoked}
    except (OSError, ValueError, TypeError):
        logger.warning("fund_peer_classifications.load_peer_classifications   >>>   分类资料不可用", exc_info=True)
        return {}
