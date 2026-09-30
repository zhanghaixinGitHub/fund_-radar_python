"""近期基金公告的独立事实通路；不修改研究新闻资格，不调用模型或交易接口。"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse

from sqlalchemy import text

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_engine
from app.integrations.dbfund_reports import CATALOG, ORIGIN
from app.integrations.dbfund_supplement import SUPPLEMENT, article_text, client, fetch
from app.services.announcement_key_points import with_reading_summary
from app.services.company_news_facts import LIMITS as COMPANY_LIMITS
from app.services.company_news_facts import collect_company_news
from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_exposure_common import ROOT, now, read, save
from app.services.fund_materials_store import source_path

logger = get_logger(__name__)
FUND = "002112"
RULE = "FUND_PUBLICATION_FACT_V2"
# 只读取已在原资料任务登记的官方栏目，拒绝本轮自动发现新站点、登录或付费来源。
SOURCE_LIMITS = {"catalog_pages": 120, "bodies": 20, "lookback_days": 30, "minimum_interval_seconds": 600}


def directory() -> Path:
    return Path(get_settings().fund_insight_directory) / FUND / "news"


def published_day(item):
    """沿用已核实的展示日字段；publishDate 含站点迁移时间，不能作为原公告日。"""
    value = str(item.get("activationDate", ""))
    if not re.fullmatch(r"\d{13}", value):
        raise ValueError("NEWS_DATE_UNKNOWN")
    return datetime.fromtimestamp(int(value) / 1000, ZONE).date()


def inspect_document(item, pages, receipt, checked_at):
    """只核验“发布了这份文件”这一事实，不凭标题推断合同完成、收入实现或利好利空。"""
    title = str(item.get("title", "")).strip()
    body = re.sub(r"\s+", "", "\n".join(pages))
    compact = re.sub(r"\s+", "", title)
    day = published_day(item)
    related = "德邦鑫星价值" in compact or "002112" in compact
    # 文中的历史交易日、统计日不证明公告公开日。仅接受明确送出/公告日期，
    # 或末尾署名后的落款日期；即便命中也仅具备按日的公开事实资格。
    normalized = re.sub(r"(?<=\D)0(?=\d)", "", body)
    day_text = f"{day.year}年{day.month}月{day.day}日"
    dated = bool(re.search(r"(?:报告送出日期|公告日期|发布日期)[：:]?" + day_text, normalized))
    dated = dated or bool(re.search(r"德邦基金管理有限公司" + day_text + r"(?:第?\d+页?)?$", normalized[-160:]))
    valid = related and compact in body and dated and day <= checked_at.date() and len(body) >= 100
    evidence = {"catalog": item, "pages": pages, "receipt": receipt, "rule": RULE}
    return {
        "event_id": digest({"source": "DBFUND_OFFICIAL_PUBLIC", "id": item["contentId"]}),
        "content_hash": receipt["sha256"],
        "evidence_hash": digest(evidence),
        "title": title,
        "source_name": "德邦基金官网",
        "source_url": receipt["url"],
        "published_date": day.isoformat(),
        "publication_precision": "DAY",
        "first_received_at": receipt["received_at"],
        "effective_at": None,
        "stage": "已披露文件" if valid else "待核验",
        "summary": "基金已披露该文件；具体事项及影响需结合原文判断。"
        if valid
        else "已取得资料，正文身份或日期尚未核对完整。",
        "business_eligible": valid,
        "prediction_eligible": False,
        "training_eligible": False,
        "relation": "基金名称或份额代码与原文一致" if valid else "基金关系待核验",
        "evidence": evidence,
    }


def first_receipts(urls):
    """复用此前成功取得的相同原文时间，不把本次重读时间冒称系统首次取得。

    只读已登记官方来源的不可变收据；按网址与原文摘要同时匹配，修订稿不会沿用旧稿时间。
    """
    result = {}
    for path in (SUPPLEMENT / "public-receipt-versions").glob("*.json"):
        receipt = read(path)
        if receipt.get("url") not in urls or not receipt.get("received_at"):
            continue
        at = datetime.fromisoformat(receipt["received_at"])
        if at.tzinfo is None:
            continue
        key = (receipt["url"], receipt["sha256"])
        if key not in result or at < datetime.fromisoformat(result[key]["received_at"]):
            result[key] = receipt
    return result


def current(code):
    """公共页面只读；失败或范围未覆盖不得变成“没有风险”。"""
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("NEWS_FUND_INVALID")
    empty = {
        "fundCode": code,
        "checkedAt": None,
        "items": [],
        "complete": False,
        "limitations": ["尚未完成该基金的近期消息核验，不能据此判断没有重要事项。"],
    }
    pointer = directory() / "current.json"
    if code != FUND or not pointer.exists():
        return empty
    ref = read(pointer)
    if not re.fullmatch(r"[a-f0-9]{64}", ref["hash"]):
        raise ValueError("NEWS_REFERENCE_INVALID")
    snapshot = read(directory() / "snapshots" / (ref["hash"] + ".json"))
    if digest(snapshot) != ref["hash"] or snapshot["fund_code"] != code:
        raise ValueError("NEWS_SNAPSHOT_INVALID")
    fields = {
        "event_id": "eventId",
        # 仅交给 Java 识别同一原文的实质更正；Java 对浏览器响应屏蔽该技术字段。
        "content_hash": "evidenceHash",
        "title": "title",
        "source_name": "sourceName",
        "source_url": "sourceUrl",
        "published_date": "publishedDate",
        "first_received_at": "firstReceivedAt",
        "stage": "stage",
        "summary": "summary",
        "relation": "relation",
    }
    return {
        "fundCode": code,
        "checkedAt": snapshot["checked_at"],
        "complete": snapshot["complete"],
        "items": [
            {public: item[key] for key, public in fields.items()}
            for item in map(with_reading_summary, snapshot["items"])
            if item["business_eligible"]
        ],
        # 历次无网络补证可能重复追加相同范围说明，公共阅读只显示一次；原快照不改。
        "limitations": list(dict.fromkeys(snapshot["limitations"])),
    }


def synchronize(code, *, progress=lambda *args: None):
    """手动与一键共用有上限的来源核验；跨进程锁防止重复采集，十分钟内复用检查结果。"""
    if code != FUND:
        raise ValueError("仅支持明确指定 002112。")
    with get_engine().connect() as lock:
        if not lock.execute(text("SELECT pg_try_advisory_lock(721130,2112)")).scalar_one():
            raise ValueError("NEWS_SYNC_BUSY")
        try:
            state_file = directory() / "state.json"
            previous = read(state_file) if state_file.exists() else {}
            if (
                previous.get("finished_at")
                and (now() - datetime.fromisoformat(previous["finished_at"])).total_seconds()
                < SOURCE_LIMITS["minimum_interval_seconds"]
            ):
                return {**previous, "reused": True}
            return _collect(previous, progress)
        finally:
            lock.execute(text("SELECT pg_advisory_unlock(721130,2112)"))


def _collect(previous, progress):
    started = now()
    lower = started.date() - timedelta(days=SOURCE_LIMITS["lookback_days"])
    registry = read(source_path(ROOT, "supplement/public-catalog.json"))
    categories = sorted(registry["category_counts"])
    if not categories or len(categories) > 12 or any(not re.fullmatch(r"[a-f0-9]{28,32}", key) for key in categories):
        raise ValueError("NEWS_SOURCE_REGISTRY_INVALID")
    entries, receipts, errors, facts = {}, [], [], []
    total_pages = 0
    with client() as connection:
        for category in categories:
            try:
                page, pages, expected, seen = 1, 1, None, set()
                while page <= pages:
                    if total_pages >= SOURCE_LIMITS["catalog_pages"]:
                        raise ValueError("NEWS_CATALOG_LIMIT")
                    total_pages += 1
                    raw, receipt = fetch(
                        connection,
                        CATALOG,
                        params={"categoryId": category, "pageNumber": page, "pageSize": 15},
                        reuse=False,
                    )
                    data = json.loads(raw)
                    count = int(data["totalCount"])
                    if expected is None:
                        expected, pages = count, int(data["totalPage"])
                        if not 1 <= pages <= 100 or expected > 1500:
                            raise ValueError("NEWS_CATALOG_LIMIT")
                    if (
                        count != expected
                        or int(data["totalPage"]) != pages
                        or not isinstance(data.get("contents"), list)
                    ):
                        raise ValueError("NEWS_CATALOG_CHANGED")
                    receipts.append(receipt)
                    for item in data["contents"]:
                        identity = item["contentId"]
                        if identity in seen:
                            raise ValueError("NEWS_CATALOG_DUPLICATE")
                        seen.add(identity)
                        day = published_day(item)
                        if lower <= day <= started.date() and (
                            "德邦鑫星价值" in item["title"] or "002112" in item["title"]
                        ):
                            if identity in entries and entries[identity] != item:
                                raise ValueError("NEWS_CATALOG_CONFLICT")
                            entries[identity] = item
                    page += 1
                    progress(total_pages, SOURCE_LIMITS["catalog_pages"], FUND, "正在核对官方公告目录")
                if len(seen) != expected:
                    raise ValueError("NEWS_CATALOG_INCOMPLETE")
            except Exception as error:
                logger.exception("fund_news_sync._collect >>> 公告栏目核对未完成, category=%s", category)
                errors.append({"category": category, "type": type(error).__name__})
        earliest = first_receipts({urljoin(ORIGIN, item["url"]) for item in entries.values()})
        for index, item in enumerate(sorted(entries.values(), key=published_day, reverse=True)):
            if index >= SOURCE_LIMITS["bodies"]:
                errors.append({"type": "NEWS_BODY_LIMIT"})
                break
            try:
                url = urljoin(ORIGIN, item["url"])
                if urlparse(url).scheme != "https" or urlparse(url).netloc != "www.dbfund.com.cn":
                    raise ValueError("NEWS_SOURCE_NOT_REGISTERED")
                pdf = urlparse(url).path.lower().endswith(".pdf")
                raw, receipt = fetch(connection, url, "pdf" if pdf else "html", reuse=False)
                pages, status = article_text(raw, pdf)
                if status != "TEXT_EXTRACTED":
                    raise ValueError("NEWS_BODY_REVIEW_REQUIRED")
                fact = inspect_document(item, pages, receipt, now())
                original_receipt = earliest.get((receipt["url"], receipt["sha256"]))
                if original_receipt:
                    fact["first_received_at"] = original_receipt["received_at"]
                    fact["first_received_evidence"] = original_receipt
                # 同一原文反复检查保留首次取得时刻；更正文单独版本，不能覆盖旧事实。
                version = digest(
                    {
                        "body": fact["content_hash"],
                        "title": fact["title"],
                        "published_date": fact["published_date"],
                        "rule": RULE,
                    }
                )
                # 扁平内容寻址兼容 Windows 路径长度；身份与内容共同决定文件名。
                path = directory() / "events" / (digest({"event": fact["event_id"], "version": version}) + ".json")
                if path.exists():
                    fact = read(path)
                else:
                    save(path, fact)
                receipts.append(receipt)
                facts.append(fact)
            except Exception as error:
                logger.exception("fund_news_sync._collect >>> 公告正文核对未完成, content_id=%s", item["contentId"])
                errors.append({"content_id": item["contentId"], "type": type(error).__name__})
    company_coverage = None
    company_limited = False
    try:
        companies = collect_company_news(now(), progress)
        company_coverage = companies["coverage"]
        company_limited = companies["limited"]
        errors.extend({"part": "company", **error} for error in companies["errors"])
        for fact in companies["facts"]:
            # 披露关系也是解释的一部分；持仓报告发生变化时保存新事实版本。
            version = digest(
                {key: fact[key] for key in ("event_id", "content_hash", "title", "published_date", "relation")}
            )
            path = directory() / "events" / (version + ".json")
            if path.exists():
                fact = read(path)
            else:
                save(path, fact)
            facts.append(fact)
    except Exception as error:
        logger.exception("fund_news_sync._collect >>> 公司消息核验未完成")
        errors.append({"part": "company", "type": type(error).__name__})
    complete = not errors and not company_limited and all(item["business_eligible"] for item in facts)
    limits = [
        f"核对范围为基金近 30 天官方公告及最新披露持仓公司的最近 {COMPANY_LIMITS['documents']} 份公告；"
        "尚未覆盖全部公司、行业和政策消息。",
        "日期为来源披露日，未认定历史首次公开时刻；基金报告核对正文日期，公司公告核对原件身份与标题，取得时间单独保留。",
        "这些消息尚未进入走势预测，未验证预测效果；未列出事项不代表没有风险。",
    ]
    if company_limited:
        limits.insert(
            0, f"公司公告多于本次核验范围，仅核验最新 {COMPANY_LIMITS['documents']} 份；其余尚未在本次逐份核验。"
        )
    if not complete:
        limits.insert(0, "本次部分来源或正文尚未核验完整，已保留上次可读记录。")
    # 失败不发布一个空列表抹掉旧资料。已验证的新事实与原记录按事件身份合并，旧原文仍留档。
    old = directory() / "current.json"
    if old.exists() and not complete:
        ref = read(old)
        prior = read(directory() / "snapshots" / (ref["hash"] + ".json"))
        # 旧披露事实仍可读，但本轮未重新核验时不得冒称当前状态已经确认。
        # 只改变新快照的展示状态，原事件文件和旧快照不动；后续成功核验会恢复正常状态。
        # Java 提醒仅接受“已披露文件”，因此旧记录不会在来源失败时新建有效事项提醒。
        merged = {
            r["event_id"]: {
                **r,
                "stage": "此前已披露，本次待复核",
                "summary": "此前已核实该文件；本次来源或正文尚未核对完整，当前事项状态未确认。",
            }
            for r in prior["items"]
        }
        for fact in facts:
            if fact["business_eligible"] or fact["event_id"] not in merged:
                merged[fact["event_id"]] = fact
        facts = [r for r in merged.values() if str(lower) <= r["published_date"] <= str(started.date())]
    facts.sort(key=lambda row: (row["published_date"], row["event_id"]), reverse=True)
    if len(facts) > 50:
        limits.insert(0, "本页最多保留最近 50 条已核对消息，较早原文与历史核对记录仍然保留。")
        facts = facts[:50]
    # 阅读摘要仅使用已留存正文；披露资格、消息身份和训练输入仍由原通路决定。
    facts = [with_reading_summary(item) for item in facts]
    snapshot = {
        "fund_code": FUND,
        "checked_at": now().isoformat(),
        "complete": complete,
        "items": facts,
        "limitations": limits,
        "receipts": receipts,
        "errors": errors,
        "rules": SOURCE_LIMITS,
        "company_coverage": company_coverage,
    }
    snapshot_hash = digest(snapshot)
    archive = directory() / "snapshots" / (snapshot_hash + ".json")
    if archive.exists():
        if digest(read(archive)) != snapshot_hash:
            raise ValueError("NEWS_SNAPSHOT_INVALID")
    else:
        save(archive, snapshot)
    save(directory() / "current.json", {"hash": snapshot_hash}, replace=True)
    state = {
        "status": "SUCCEEDED" if complete else "PARTIAL_SUCCESS",
        "started_at": started.isoformat(),
        "finished_at": now().isoformat(),
        "last_success_at": now().isoformat() if complete else previous.get("last_success_at"),
        "items": len(facts),
        "verified": sum(row["business_eligible"] for row in facts),
        "errors": len(errors),
        "message": "近期公告核对完成。" if complete else "部分资料未核验完整，保留已取得的事实及具体限制。",
    }
    save(directory() / "state.json", state, replace=True)
    return state
