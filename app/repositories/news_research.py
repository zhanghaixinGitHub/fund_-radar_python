"""公告研究版本的事务内追加和精确读回；不涉及业务事件或正式预测表。"""

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models.news_research import NewsResearchBundleRow, NewsResearchCardRow
from app.schemas.news_research import content_hash


def lock_dataset(session: Session, dataset_key: str) -> None:
    """同一研究集合串行提交，避免并发重试产生半批记录或交叉更正版本。"""
    session.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key,0))"), {"key": "news-research:" + dataset_key}
    )


def find_revision(session: Session, dataset_key: str, revision: str):
    return session.scalar(
        select(NewsResearchBundleRow).where(
            NewsResearchBundleRow.dataset_key == dataset_key, NewsResearchBundleRow.revision == revision
        )
    )


def read_bundle(session: Session, bundle_hash: str):
    """一个版本最多 100 条，按唯一样例身份排序恢复完整存储契约。"""
    head = session.get(NewsResearchBundleRow, bundle_hash)
    if head is None:
        return None
    rows = list(
        session.scalars(
            select(NewsResearchCardRow)
            .where(NewsResearchCardRow.bundle_hash == bundle_hash)
            .order_by(NewsResearchCardRow.sample_id)
        )
    )
    if len(rows) != head.card_count:
        raise ValueError("RESEARCH_INCOMPLETE_PERSISTED_BUNDLE")
    for row in rows:
        c = row.record_payload["card"]
        if (
            row.record_hash != content_hash(row.record_payload)
            or row.sample_id != c["id"]
            or row.fund_code != head.fund_code
            or row.stock_code != c["company"]["stock_code"]
            or row.target_date.isoformat() != c["target"]
            or row.published_date.isoformat() != c["published_date"]
            or row.pdf_sha256 != c["evidence"]["pdf_sha256"]
            or row.announcement_id != c["announcement_id"]
            or row.event_topics != c["events"]["ids"]
            or row.event_stage != c["events"]["stage"]
        ):
            raise ValueError("RESEARCH_PERSISTED_INDEX_OR_CONTENT_MISMATCH")
    return {**head.metadata_payload, "records": [r.record_payload for r in rows]}
