"""独立从 PDF 重放已逐条判读的字段；公开日期与版本链不由标题猜测。"""

import json
import shutil
import subprocess
from datetime import date, datetime, timedelta
from decimal import Decimal

import pypdfium2 as pdfium
from app.services.fund_information_history_v1 import (
    OUT,
    ZONE,
    explicit_money,
    normalize,
    pdf_revision,
    read,
    save,
    sha,
)


def replay_field(field, pages):
    """指定页内精确短锚点核对；数值、原单位与业务口径分别保存。"""
    page = field["page"]
    text = normalize(pages[page - 1])
    quote = normalize(field["quote"])
    if text.count(quote) != 1:
        raise ValueError("REVIEW_ANCHOR_NOT_UNIQUE:" + field["name"])
    result = {k: v for k, v in field.items() if k not in {"quote", "page"}}
    result["anchor"] = {"page": page, "normalized_offset": text.index(quote), "text": quote}
    if field["unit"]:
        values = field["value"] if isinstance(field["value"], list) else [field["value"]]
        if any(not Decimal(str(v)).is_finite() for v in values):
            raise ValueError("NON_FINITE_REVIEWED_QUANTITY")
        if field["unit"] != "PERCENT":
            result["money"] = [explicit_money(v, field["unit"]) for v in values]
        if not field["basis"]:
            raise ValueError("REVIEW_QUANTITY_BASIS_MISSING")
    return result


def run():
    inventory = {r["document_id"]: r for r in read(OUT / "semantic-inventory.json")}
    specs = read(OUT / "review-specs-v1.json")
    reviewed = []
    renders = OUT / "review-page-renders"
    renders.mkdir(parents=True, exist_ok=True)
    for spec in specs["reviews"]:
        source = inventory[spec["id"]]
        raw = source["source"]
        if sha(raw["path"]) != raw["sha256"]:
            raise ValueError("REVIEW_SOURCE_CHANGED")
        original = read(source["text_path"])
        # 与旧页缓存分开重提取；核每个字段引用页的全文，避免只比较手抄的数字。
        with pdfium.PdfDocument(raw["path"]) as document:
            pages = [""] * len(document)
            needed = {f["page"] for f in spec["fields"]}
            for number in sorted(needed):
                page = document[number - 1]
                text = page.get_textpage()
                try:
                    pages[number - 1] = text.get_text_range()
                    if normalize(pages[number - 1]) != normalize(original["pages"][number - 1]):
                        raise ValueError("INDEPENDENT_PDF_TEXT_REPLAY_MISMATCH")
                    image_path = renders / f"{spec['id']}-p{number}.png"
                    if not image_path.exists():
                        renderer = shutil.which("pdftoppm")
                        if not renderer:
                            raise ValueError("PDF_RENDERER_UNAVAILABLE")
                        subprocess.run(
                            [
                                renderer,
                                "-f",
                                str(number),
                                "-l",
                                str(number),
                                "-scale-to",
                                "1400",
                                "-singlefile",
                                "-png",
                                raw["path"],
                                str(image_path.with_suffix("")),
                            ],
                            check=True,
                            capture_output=True,
                            timeout=45,
                        )
                finally:
                    text.close()
                    page.close()
            metadata = document.get_metadata_dict()
        issues = pdf_revision(metadata, source["published_date"])
        fields = [replay_field(f, pages) for f in spec["fields"]]
        # 更正稿同时有明确被更正文号和公开日期，才可登记显式版本关系。
        if spec.get("revises") and spec["revises"] not in inventory:
            raise ValueError("REVISION_TARGET_NOT_IN_SOURCE_SET")
        available = str(date.fromisoformat(source["published_date"]) + timedelta(days=1)) + "T08:00:00+08:00"
        reviewed.append(
            {
                "schema": "REVIEWED_FACT_FIELDS_V1",
                "document_id": spec["id"],
                "entity_id": source["entity_id"],
                "published_date": source["published_date"],
                "available_at": available if not issues else None,
                "source": raw,
                "text_source_sha256": sha(source["text_path"]),
                "category": spec["category"],
                "stage": spec["stage"],
                "period": spec.get("period"),
                "event_key": spec.get("event_key"),
                "revises": spec.get("revises"),
                "fields": fields,
                "unknown_fields": spec["unknown"],
                "pdf_metadata_issues": issues,
                "independent_pdf_page_replay_passed": True,
                "review_scope": "Only these source-anchored fields; no general parser precision claim",
                "training_ready": False,
                "production_used": False,
            }
        )
    save(OUT / "reviewed-facts-v1.json", reviewed)
    index = {r["document_id"]: r for r in reviewed}
    # 两期同标题公告按原回购报告文号分开，同一期累计数只保留时点、不相加。
    first = index["1209048230"]
    second = index["1209232990"]
    separate = index["1209048233"]
    assert first["event_key"] == second["event_key"] != separate["event_key"]
    chain = {
        "buyback_same_event": [first["document_id"], second["document_id"]],
        "separate_buyback_event": separate["document_id"],
        "basis": "Explicit original announcement 2020-016 versus 2020-056, not similar titles",
        "cumulative_amounts_must_not_be_summed": True,
        "contract_revision": {
            "original": "1210758701",
            "correction": "1210760613",
            "explicit_reference": "300953:2021-048",
            "correction_announcement": "300953:2021-049",
            "availability": index["1210760613"]["available_at"],
            "intraday_order": None,
            "version_chain_status": "EXPLICIT_REFERENCE_RESOLVED",
            "do_not_rewrite_original_fact": True,
        },
    }
    save(OUT / "reviewed-event-chains-v1.json", chain)
    summary = {
        "reviewed_documents": len(reviewed),
        "reviewed_fields": sum(len(r["fields"]) for r in reviewed),
        "independent_pdf_pages": sum(len({f["page"] for f in r["fields"]}) for r in specs["reviews"]),
        "metadata_conflicts": sum(bool(r["pdf_metadata_issues"]) for r in reviewed),
        "remaining_cumulative_buyback_candidates_unreviewed": sum(
            len(r["quantities"]) for r in inventory.values() if r["document_id"] not in index
        ),
        "reviewed_at": read(OUT / "fact-review-result-v1.json")["reviewed_at"]
        if (OUT / "fact-review-result-v1.json").exists()
        else datetime.now(ZONE).isoformat(),
        "new_fits": 0,
        "not_all_facts_semantically_verified": True,
    }
    save(OUT / "fact-review-result-v1.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
