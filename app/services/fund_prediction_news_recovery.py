"""把本轮核验过的公告原文接入候选综合包；不自动批准资料或判断方向。"""

from app.integrations.market_fund_reports import read_source
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read
from app.services.fund_workflow_materials import workflow_root


def refresh(material, *, root=None, checked_at=None):
    root = workflow_root(root)
    checked_at = checked_at or now()
    code = material["fund_code"]
    news = read(root / "news/funds" / (code + ".json"))
    if (
        news.get("status") != "SUCCEEDED"
        or not news.get("catalog_complete")
        or news.get("coverage", {}).get("end", "") < str(checked_at.date())
    ):
        raise ValueError("CURRENT_NOTICE_COVERAGE_REQUIRED")
    refs, deferred = [], []
    for article, item in news.get("items", {}).items():
        entry = item["entry"]
        if entry["published"] < news["coverage"]["start"]:
            continue
        source = item.get("source")
        if not source or not item.get("body"):
            raise ValueError("CURRENT_NOTICE_ORIGINAL_REQUIRED")
        read_source(root / "news", source)
        ref = "daily-notice:" + article + ":" + source["sha256"]
        document = {
            "id": ref,
            "kind": "ANNOUNCEMENT",
            "title": entry["title"],
            "url": "https://fund.eastmoney.com/gonggao/" + code + "," + article + ".html",
            "body": item["body"],
            "body_hash": digest(item["body"]),
            "body_scope": "FULL_TEXT",
            "published_date": entry["published"],
            "available_at": item["received_at"],
            "codes": [code],
            "raw_path": str((root / "news" / source["file"]).resolve()),
            "raw_sha256": source["sha256"],
        }
        if item.get("published_at"):
            document["published_at"] = item["published_at"]
        elif entry["published"] == str(checked_at.date()):
            from app.services.fund_notice_observed_proof import validate as observed_public_notice

            receipt_path = root / "news/body-receipts" / code / (article + ".json")
            if receipt_path.exists():
                receipt = read(receipt_path)
                document["publication_observed_proof"] = {
                    "kind": "PUBLIC_RESPONSE_FIRST_OBSERVED",
                    "article_id": article,
                    "receipt_hash": digest(receipt),
                    "observed_at": receipt["received_at"],
                }
            if not observed_public_notice(document, code, checked_at, root=root):
                deferred.append(article)
                continue
        if not any(d["id"] == ref for d in material["documents"]):
            material["documents"].append(document)
        refs.append(ref)
    if refs:
        material["comprehensive_coverage"]["DISCLOSURES"] = {"status": "AVAILABLE", "source_refs": refs}
    material["source_manifest"]["current_notice_check"] = {
        "coverage": news["coverage"],
        "status": news["status"],
        "finished_at": news.get("finished_at"),
        "source_refs": refs,
        "publication_time_required": deferred,
    }
    return {"added_sources": refs, "publication_time_required": deferred}
