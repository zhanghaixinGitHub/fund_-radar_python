"""完整、幂等保存已核验的本地公告研究资料；不生成预测资格或业务审核结果。"""

import logging
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_nav_sample_storage_engine
from app.models.news_research import NewsResearchBundleRow, NewsResearchCardRow
from app.repositories import news_research as repository
from app.schemas.news_research import NewsResearchBundle, content_hash

logger = logging.getLogger(__name__)


def save_research_bundle(bundle: NewsResearchBundle) -> dict:
    """整批一个事务；重试完全相同内容返回原版本，内容更正必须显式链接旧版本。

    参数必须来自已做原文锚点/摘要核对的本地加载器。本方法验证结构与存储身份，
    不访问来源网站，不把落库等同于解析正确；任何卡片失败都会回滚整批。
    """
    # 调用者可能在 Pydantic 校验后修改嵌套字典，事务前重新构造并校验。
    payload = NewsResearchBundle.model_validate(bundle.model_dump(mode="json")).model_dump(mode="json")
    identity = content_hash(payload)
    created = False
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        repository.lock_dataset(session, bundle.dataset_key)
        previous = repository.find_revision(session, bundle.dataset_key, bundle.revision)
        if previous is not None:
            if previous.bundle_hash != identity or repository.read_bundle(session, identity) != payload:
                raise ValueError("RESEARCH_REVISION_CONTENT_CONFLICT")
        else:
            parent_id = payload["supersedes_bundle_hash"]
            existing = session.scalar(
                select(NewsResearchBundleRow.bundle_hash)
                .where(NewsResearchBundleRow.dataset_key == bundle.dataset_key)
                .limit(1)
            )
            if existing and parent_id is None:
                raise ValueError("RESEARCH_NEW_REVISION_NEEDS_PARENT")
            if parent_id is not None:
                parent = session.get(NewsResearchBundleRow, parent_id)
                if parent is None or parent.dataset_key != bundle.dataset_key or parent.fund_code != bundle.fund_code:
                    raise ValueError("RESEARCH_INVALID_PARENT")
                child = session.scalar(
                    select(NewsResearchBundleRow.bundle_hash)
                    .where(NewsResearchBundleRow.supersedes_bundle_hash == parent_id)
                    .limit(1)
                )
                if child:
                    raise ValueError("RESEARCH_PARENT_ALREADY_SUPERSEDED")
            session.add(
                NewsResearchBundleRow(
                    bundle_hash=identity,
                    dataset_key=bundle.dataset_key,
                    revision=bundle.revision,
                    fund_code=bundle.fund_code,
                    supersedes_bundle_hash=parent_id,
                    card_count=len(bundle.records),
                    metadata_payload={k: v for k, v in payload.items() if k != "records"},
                )
            )
            session.flush()
            for record in payload["records"]:
                c = record["card"]
                session.add(
                    NewsResearchCardRow(
                        bundle_hash=identity,
                        sample_id=c["id"],
                        record_hash=content_hash(record),
                        fund_code=bundle.fund_code,
                        stock_code=c["company"]["stock_code"],
                        target_date=date.fromisoformat(c["target"]),
                        published_date=date.fromisoformat(c["published_date"]),
                        announcement_id=c["announcement_id"],
                        pdf_sha256=c["evidence"]["pdf_sha256"],
                        event_topics=c["events"]["ids"],
                        event_stage=c["events"]["stage"],
                        record_payload=record,
                        purpose="RESEARCH",
                        validation_status="LOCAL_ANCHORS_VERIFIED",
                        confidence=None,
                        business_eligible=False,
                        prediction_eligible=False,
                        training_eligible=False,
                    )
                )
            session.flush()
            if repository.read_bundle(session, identity) != payload:
                raise ValueError("RESEARCH_ROUNDTRIP_MISMATCH")
            created = True
    logger.info(
        "news_research_storage.save_research_bundle   >>>   fund=%s bundle=%s cards=%s created=%s",
        bundle.fund_code,
        identity,
        len(bundle.records),
        created,
    )
    return {"bundle_hash": identity, "created": created, "cards": len(bundle.records)}


def read_research_bundle(bundle_hash: str) -> dict:
    """独立事务读回并核对全部内容摘要，失败不返回部分成功。"""
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        payload = repository.read_bundle(session, bundle_hash)
        if payload is None or content_hash(payload) != bundle_hash:
            raise ValueError("RESEARCH_BUNDLE_NOT_FOUND_OR_CHANGED")
        NewsResearchBundle.model_validate(payload)
        return payload
