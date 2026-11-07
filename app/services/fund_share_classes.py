"""全目录份额关联的采集与只读投影；不改变基金市场范围和历史产品主键。"""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_engine
from app.integrations.fund_share_class_source import ShareClassSource, parse_identity
from app.integrations.market_fund_reports import read_source
from app.models.fund import FundShareClass
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save

logger = logging.getLogger(__name__)
VERSION = "DISCLOSED_SHARE_CLASSES_V1"


def store_root() -> Path:
    return Path(get_settings().fund_insight_directory) / "share-classes"


def scope_codes() -> tuple[str, ...]:
    """仅取有效市场目录，最多 5000 个；不按关注账户过滤，也不扩散采集关联份额。"""
    with Session(get_engine()) as session:
        codes = tuple(session.scalars(select(FundShareClass.fund_code).where(
            FundShareClass.status == "ACTIVE",
        ).order_by(FundShareClass.fund_code).limit(5001)))
    if len(codes) > 5000:
        raise ValueError("SHARE_SCOPE_EXCEEDS_5000")
    return codes


def _load(root: Path) -> dict:
    path = root / "current.json"
    if not path.exists():
        return {"version": VERSION, "products": {}, "checks": {}}
    if path.stat().st_size > 12_000_000:
        raise ValueError("SHARE_SNAPSHOT_SIZE")
    packet = read(path)
    if packet.get("version") != VERSION:
        raise ValueError("SHARE_SNAPSHOT_VERSION")
    return packet


@contextmanager
def _sync_lock():
    """专用会话锁覆盖一次发布，避免 CLI 与既有同步并发改写索引；不锁业务数据。"""
    with get_engine().connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        acquired = connection.scalar(text("SELECT pg_try_advisory_lock(67329105483021)"))
        if not acquired:
            raise ValueError("SHARE_SYNC_ALREADY_RUNNING")
        try:
            yield
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(67329105483021)"))


def synchronize(codes=None, *, root=None, source=None, due_only=True, publish=True,
                progress=lambda *args: None) -> dict:
    """按全目录核对最新报告；逐基金失败隔离，成功快照原子发布且保留历史。

    publish=False 仍可读取公开原文并存证，但不修改页面使用的当前关系。
    失败不会回退旧报告或把空结果当作单份额；既有关系保留并标为待更新。
    """
    root = Path(root) if root is not None else store_root()
    requested = tuple(sorted(set(codes if codes is not None else scope_codes())))
    if len(requested) > 5000 or any(len(c) != 6 or not c.isascii() or not c.isdigit() for c in requested):
        raise ValueError("SHARE_SCOPE_INVALID")
    with _sync_lock():
        return _synchronize(requested, root, source, due_only, publish, progress)


def _synchronize(codes, root, source, due_only, publish, progress):
    packet = _load(root)
    client = source or ShareClassSource(root)
    stamp = now()
    profiles, items = {}, []
    try:
        for index, code in enumerate(codes, 1):
            old = packet["checks"].get(code, {})
            checked = datetime.fromisoformat(old["checkedAt"]) if old.get("checkedAt") else None
            if (due_only and old.get("status") == "VERIFIED" and checked
                    and timedelta(0) <= stamp - checked < timedelta(days=7)):
                try:
                    existing = packet["products"][old["productCode"]]
                    for ref in existing["sources"]:
                        read_source(root, ref)
                except (OSError, ValueError, KeyError):
                    pass  # 缺失或损坏的原件须重新取证，不能凭“七天内成功”继续跳过。
                else:
                    items.append({"fundCode": code, "status": "REUSED"})
                    continue
            try:
                entries = client.catalog(code)
                if not entries:
                    raise ValueError("SHARE_REPORT_NOT_FOUND")
                entry = max(entries, key=lambda e: (e["end"], e["published"], e["id"]))
                body, report_ref = client.report(code, entry)
                main, members = parse_identity(body, code)
                shares, refs = [], [report_ref]
                for member in members:
                    if member not in profiles:
                        profiles[member] = client.profile(member)
                    share, ref = profiles[member]
                    shares.append(share)
                    refs.append(ref)
                product = {
                    "productCode": main, "asOfDate": entry["end"], "publishedOn": entry["published"],
                    "checkedAt": stamp.isoformat(), "shares": sorted(shares, key=lambda s: s["fundCode"]),
                    "sources": refs, "anchorCode": code,
                }
                # 同一代码不能同时归属两个产品；不以覆盖旧映射的方式掩盖来源冲突。
                if any(p["productCode"] != main and set(members) & {s["fundCode"] for s in p["shares"]}
                       for p in packet["products"].values()):
                    raise ValueError("SHARE_PRODUCT_CONFLICT")
                previous = packet["products"].get(main)
                if previous and previous["asOfDate"] == product["asOfDate"] and (
                    {s["fundCode"] for s in previous["shares"]} != set(members)
                ):
                    raise ValueError("SHARE_MEMBERSHIP_CONFLICT")
                if not previous or (product["asOfDate"], product["publishedOn"]) >= (
                    previous["asOfDate"], previous["publishedOn"]
                ):
                    packet["products"][main] = product
                packet["checks"][code] = {"status": "VERIFIED", "checkedAt": stamp.isoformat(), "productCode": main}
                items.append({"fundCode": code, "status": "VERIFIED", "shareCount": len(members)})
            except Exception as error:
                logger.warning("fund_share_classes.synchronize   >>>   份额核对失败, fund_code=%s", code, exc_info=True)
                packet["checks"][code] = {"status": "UNVERIFIED", "checkedAt": stamp.isoformat(),
                                           "error": type(error).__name__ + ":" + str(error)[:180]}
                items.append({"fundCode": code, "status": "UNVERIFIED", "reason": str(error)[:180]})
            progress(index, len(codes), code, "基金份额关联已检查")
        if publish:
            packet["updatedAt"] = stamp.isoformat()
            versioned_save(root / "current.json", packet)
    finally:
        if source is None:
            client.close()
    failed = sum(i["status"] == "UNVERIFIED" for i in items)
    return {"status": "SUCCEEDED" if not failed else "PARTIAL_SUCCESS", "total": len(codes),
            "verified": len(codes) - failed, "unverified": failed, "published": publish, "items": items}


def read_relations(code: str, *, root=None) -> dict:
    """GET 仅读取有指纹依据的公共快照，不触发网络、解析报告或写库。"""
    root = Path(root) if root is not None else store_root()
    empty = {"fundCode": code, "status": "UNVERIFIED", "asOfDate": None, "checkedAt": None, "shares": []}
    try:
        packet = _load(root)
        matches = [p for p in packet["products"].values() if code in {s["fundCode"] for s in p["shares"]}]
        if len(matches) != 1:
            return empty
        product = matches[0]
        for ref in product["sources"]:
            read_source(root, ref)
        checked = datetime.fromisoformat(product["checkedAt"])
        current = now()
        if checked.tzinfo is None or checked > current or product["publishedOn"] > str(current.date()):
            return empty
        members = {s["fundCode"] for s in product["shares"]}
        if any("CONFLICT" in packet["checks"].get(c, {}).get("error", "") for c in members):
            return empty
        failed = any(packet["checks"].get(c, {}).get("status") == "UNVERIFIED" for c in members)
        stale = failed or current - checked > timedelta(days=35)
        return {"fundCode": code, "status": "STALE" if stale else "VERIFIED",
                "asOfDate": product["asOfDate"], "checkedAt": product["checkedAt"], "shares": product["shares"]}
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("fund_share_classes.read_relations   >>>   份额资料不可用, fund_code=%s", code, exc_info=True)
        return empty
