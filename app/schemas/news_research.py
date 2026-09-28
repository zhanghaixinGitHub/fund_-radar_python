"""公告研究存储契约：完整保存既有事实和分析，未知值不生成方向或金额。"""

import hashlib
import json
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


def content_hash(value) -> str:
    """规范 JSON 的 SHA-256，拒绝 NaN/Infinity；用于不可变版本和完整读回。"""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class NewsResearchRecord(BaseModel):
    """一份公告在某个研究目标日的卡片及原分析；保留一文多事，不做金额汇总。"""

    model_config = ConfigDict(extra="forbid")
    card: dict[str, JsonValue]
    analysis: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_semantics(self):
        c, a = self.card, self.analysis
        try:
            for key in ("id", "target", "published_date"):
                if c[key] != a[key]:
                    raise ValueError("RESEARCH_CARD_ANALYSIS_IDENTITY")
            if c["company"]["stock_code"] != a["stock_code"]:
                raise ValueError("RESEARCH_COMPANY_IDENTITY")
            if c["prediction_eligible"] is not False or any(
                a[k] is not False for k in ("prediction_eligible", "training_eligible")
            ):
                raise ValueError("RESEARCH_ELIGIBILITY_NOT_APPROVED")
            if c["next_day_direction"] is not None or c["impact_magnitude"] is not None:
                raise ValueError("RESEARCH_UNKNOWN_DIRECTION_OR_IMPACT")
            target = date.fromisoformat(c["target"])
            if (
                date.fromisoformat(c["published_date"]) >= target
                or date.fromisoformat(c["holding"]["public_date"]) >= target
            ):
                raise ValueError("RESEARCH_PUBLICATION_NOT_PRIOR")
            if not c["unknowns"] or not c["uncovered"] or not c["analysis"]["horizon"]:
                raise ValueError("RESEARCH_UNKNOWNS_MISSING")
            events = c["events"]
            if (
                not events["ids"]
                or len(set(events["ids"])) != len(events["ids"])
                or not events["stage"]
                or events["aggregation"] != "EVIDENCE_ONLY_NO_ADDITION"
            ):
                raise ValueError("RESEARCH_EVENT_SEMANTICS")
            if events["ids"] != a["event_topics"] or events["stage"] != a["event_stage"]:
                raise ValueError("RESEARCH_EVENT_ANALYSIS_MISMATCH")
            if c["evidence"]["anchors"] != a["anchors"] or not c["evidence"]["anchors"]:
                raise ValueError("RESEARCH_ANCHOR_MISMATCH")
            for anchor in c["evidence"]["anchors"]:
                if type(anchor["page"]) is not int or anchor["page"] < 1 or not anchor["text"].strip():
                    raise ValueError("RESEARCH_INVALID_ANCHOR")
            for amount in c["facts"]["amounts"]:
                value, unit = amount["value"], amount["unit"]
                if value is None or unit is None:
                    if not amount.get("unknown_reason"):
                        raise ValueError("RESEARCH_AMOUNT_UNKNOWN_REASON")
                if value is not None and (
                    not isinstance(value, str) or len(value) > 80 or not Decimal(value).is_finite()
                ):
                    raise ValueError("RESEARCH_AMOUNT_VALUE")
                if unit is not None and (not isinstance(unit, str) or not unit.strip()):
                    raise ValueError("RESEARCH_AMOUNT_UNIT")
                if not amount["meaning"] or not amount["status"]:
                    raise ValueError("RESEARCH_AMOUNT_STAGE")
            for key in ("facts", "priced_in_status", "unknowns"):
                if key not in a:
                    raise ValueError("RESEARCH_ANALYSIS_FIELD_MISSING")
            content_hash(self.model_dump(mode="json"))
        except (KeyError, TypeError, AttributeError, ArithmeticError) as exc:
            raise ValueError("RESEARCH_INCOMPLETE_CARD") from exc
        return self


class NewsResearchBundle(BaseModel):
    """最多 100 份的不可变研究版本；更正必须换版本并指明前版，旧内容不覆盖。"""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = Field(default="NEWS_RESEARCH_STORAGE_V1", pattern=r"^NEWS_RESEARCH_STORAGE_V1$")
    dataset_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    revision: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    analysis_version: str = Field(min_length=1, max_length=128)
    fund_code: str = Field(pattern=r"^\d{6}$")
    source_manifest: dict[str, str] = Field(min_length=1)
    context: dict[str, JsonValue]
    supersedes_bundle_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    records: list[NewsResearchRecord] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def check_bundle(self):
        ids = [r.card["id"] for r in self.records]
        if ids != sorted(set(ids)):
            raise ValueError("RESEARCH_SAMPLE_IDS_NOT_UNIQUE_SORTED")
        for value in self.source_manifest.values():
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("RESEARCH_SOURCE_SHA256")
        content_hash(self.model_dump(mode="json"))
        return self
