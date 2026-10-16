"""读取离线核实的基金资料要点；不触发取数、模型、预测或交易。"""

import re
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import get_settings
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read


def current(code: str) -> dict:
    """校验基金身份、内容哈希和证据日期后，投影已核实的公开事实。

    本入口仅消费离线发布的资料要点，不复用002112的来源或采集规则。
    新基金未发布快照时仍返回资料暂缺；complete不会因存在几条要点而自动变为真。
    """
    empty = {
        "fundCode": code,
        "checkedAt": None,
        "items": [],
        "complete": False,
        "limitations": ["尚未完成该基金的公告要点核验，不能据此判断没有重要事项。"],
    }
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("MATERIAL_NEWS_FUND_INVALID")
    root = Path(get_settings().fund_insight_directory) / code / "material-news"
    pointer = root / "current.json"
    if not pointer.exists():
        return empty
    ref = read(pointer)
    snapshot_hash = ref.get("hash", "")
    if not re.fullmatch(r"[a-f0-9]{64}", snapshot_hash):
        raise ValueError("MATERIAL_NEWS_REFERENCE_INVALID")
    snapshot = read(root / "snapshots" / (snapshot_hash + ".json"))
    if digest(snapshot) != snapshot_hash or snapshot.get("fund_code") != code:
        raise ValueError("MATERIAL_NEWS_SNAPSHOT_INVALID")
    cutoff = date.fromisoformat(snapshot["evidence_cutoff"])
    if cutoff > datetime.fromisoformat(snapshot["checked_at"]).date():
        raise ValueError("MATERIAL_NEWS_CUTOFF_INVALID")
    items = []
    for item in snapshot["items"]:
        # 业务事实与研究资格严格分开，离线整理资料也不得隐式激活预测或训练。
        if item.get("prediction_eligible") or item.get("training_eligible"):
            raise ValueError("MATERIAL_NEWS_QUALIFICATION_INVALID")
        if not item.get("business_eligible"):
            continue
        if item.get("verification_mode") != "SOURCE_AND_NUMERIC_CROSSCHECK":
            raise ValueError("MATERIAL_NEWS_VERIFICATION_MISSING")
        published = datetime.fromisoformat(item["published_date"]).date()
        if published > cutoff or not re.fullmatch(r"[a-f0-9]{64}", item["content_hash"]):
            raise ValueError("MATERIAL_NEWS_EVIDENCE_DATE_INVALID")
        url = urlparse(item["source_url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("MATERIAL_NEWS_SOURCE_INVALID")
        points = item.get("points", [])
        if (
            not isinstance(points, list)
            or not 1 <= len(points) <= 5
            or any(not isinstance(p, str) or not p.strip() or len(p) > 300 for p in points)
        ):
            raise ValueError("MATERIAL_NEWS_POINTS_INVALID")
        items.append(
            {
                "eventId": item["event_id"],
                "evidenceHash": item["content_hash"],
                "title": item["title"],
                "sourceName": item["source_name"],
                "sourceUrl": item["source_url"],
                "publishedDate": item["published_date"],
                "firstReceivedAt": item["first_received_at"],
                "stage": "已披露文件",
                "summary": "\n".join(points),
                "relation": item["relation"],
            }
        )
    return {
        "fundCode": code,
        "checkedAt": snapshot["checked_at"],
        "complete": bool(snapshot["complete"]),
        "items": items,
        "limitations": list(dict.fromkeys(snapshot["limitations"])),
    }
