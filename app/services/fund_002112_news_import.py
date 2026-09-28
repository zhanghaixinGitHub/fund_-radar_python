"""将已冻结的 12 份公告样例准备为研究存储版本，先核对本地文件及原文锚点。"""

import re
from pathlib import Path

from app.schemas.news_research import NewsResearchBundle
from app.services.fund_002112_input_audit import OUTPUT
from app.services.fund_002112_zero_fit_review import ROOT, file_hash, read_json


def prepare_existing_bundle() -> NewsResearchBundle:
    """不下载、不调用大模型、不补金额；沿用已核查卡与原分析，保留日期精度。"""
    protocol = read_json(OUTPUT / "protocol.json")
    before = read_json(OUTPUT / "protection-before.json")
    manifest = {}

    def checked(path, expected):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT.resolve()) or file_hash(path) != expected:
            raise ValueError("NEWS_RESEARCH_SOURCE_CHANGED")
        manifest[str(path)] = expected
        return path

    cards_spec, analysis_spec = (protocol["sources"][n] for n in ("news_cards", "news_analysis"))
    cards_path = checked(cards_spec["path"], cards_spec["sha256"])
    cards = read_json(cards_path)
    analysis_path = checked(analysis_spec["path"], analysis_spec["sha256"])
    analysis = read_json(analysis_path)
    validation_path = cards_path.parent / "news-validation.json"
    checked(validation_path, before["artifacts"][str(validation_path)])
    validation = read_json(validation_path)
    if validation["passed"] is not True or validation["cards"] != 12:
        raise ValueError("NEWS_RESEARCH_ORIGINAL_VALIDATION")
    for path, expected in validation["sources"].items():
        checked(path, expected)
    texts_path = analysis_path.parent / "extracted-pages.json"
    texts = {r["sample_id"]: r for r in read_json(texts_path)}
    records = {r["id"]: r for r in analysis["records"]}
    if len(cards["cards"]) != 12 or set(records) != {f"N{i:02}" for i in range(1, 13)}:
        raise ValueError("NEWS_RESEARCH_FIXED_SAMPLES")
    for card in cards["cards"]:
        text = texts[card["id"]]
        if not "2024-03-04" <= card["target"] <= "2024-03-08":
            raise ValueError("NEWS_RESEARCH_TARGET_SCOPE")
        if text["pdf_sha256"] != card["evidence"]["pdf_sha256"]:
            raise ValueError("NEWS_RESEARCH_PDF_IDENTITY")
        checked(ROOT / card["evidence"]["pdf_file"], card["evidence"]["pdf_sha256"])
        for anchor in card["evidence"]["anchors"]:
            if re.sub(r"\s+", "", anchor["text"]) not in re.sub(r"\s+", "", text["pages"][anchor["page"] - 1]):
                raise ValueError("NEWS_RESEARCH_ANCHOR_NOT_FOUND")
    return NewsResearchBundle(
        dataset_key="002112-existing-12-20240304-20240308",
        revision="v1",
        analysis_version="existing-human-analysis-and-round3-cards-v1",
        fund_code="002112",
        source_manifest=manifest,
        context={
            "analysis": {k: v for k, v in analysis.items() if k != "records"},
            "cards": {k: v for k, v in cards.items() if k != "cards"},
            "validation_scope": "LOCAL_SOURCE_HASHES_AND_ANCHORS_NOT_AUTOMATIC_SEMANTIC_ACCURACY",
            "amount_coverage": "EXPLICIT_AMOUNTS_ONLY_OTHERS_UNKNOWN",
        },
        records=[{"card": c, "analysis": records[c["id"]]} for c in cards["cards"]],
    )
