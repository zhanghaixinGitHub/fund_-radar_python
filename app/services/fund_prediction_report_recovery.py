"""候选包登记已经存在且身份吻合的新季报，保留旧完整组合及部分披露边界。"""

import re
from copy import deepcopy
from datetime import date


def bind_latest_partial(material, latest):
    if latest.get("fullDisclosure") is not False or latest.get("endDate", "") <= material.get("report", {}).get(
        "endDate", ""
    ):
        return False
    old = material.get("source_manifest", {}).get("latest_partial_report", {})
    if old.get("report", {}).get("id") == latest.get("id"):
        return False
    article = latest.get("sourceArticleId", "")
    if not re.fullmatch(r"AN\d{18}", article):
        return False
    code = material["fund_code"]
    end = date.fromisoformat(latest["endDate"])
    anchors = []
    for document in material["documents"]:
        body = re.sub(r"\s+", "", document["body"])
        if (
            document.get("kind") == "REPORT"
            and code in document.get("codes", [])
            and article in document["id"] + document.get("url", "")
            and document.get("published_date") == latest["publishedDate"]
            and re.search(r"(?<!\d)" + code + r"(?!\d)", document["body"])
            and re.search(f"{end.year}年0?{end.month}月0?{end.day}日", body)
        ):
            anchors.append(document)
    if len(anchors) != 1:
        return False
    material.setdefault("source_manifest", {})["latest_partial_report"] = {
        "report": deepcopy(latest),
        "source_ref": anchors[0]["id"],
        "use_scope": "PARTIAL_DISCLOSURE_ORIGINAL_DATE_PRESERVED_NO_RENORMALIZATION",
    }
    return True
